"""
Metadaten-Extraktion und CSV-Processing für KDR100 Kacheln.

Dieses Modul liest die CSV-Datei ein und extrahiert kritische Metadaten:
- Temporaler Stempel (Jahr aus longName)
- Räumliche Koordinaten (ul_x, ul_y, lr_x, lr_y -> center_x, center_y)
- Dateiinformationen
"""

import re
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import pandas as pd

from dataselector.data.spatial_schema import (
    coordinates_look_projected,
    normalize_spatial_schema,
)


class MetadataProcessor:
    """Verarbeitet Metadaten aus der KDR100 CSV-Datei."""

    def __init__(self, csv_path: str):
        """
        Initialisiert den Metadata Processor.

        Args:
            csv_path: Pfad zur CSV-Datei (KDR100_foliage_with_files_epsg3857.csv)
        """
        self.csv_path = Path(csv_path)
        self.df: pd.DataFrame = None
        self.crs_unit: Literal["degree", "meter", "unknown"] = "unknown"
        # Optional GeoPandas GeoDataFrame (set when geopandas is available)
        self.gdf = None
        self.source_crs: str | None = None
        self.metric_crs: str | None = None
        self.transform_applied: bool = False

    def load_csv(self) -> pd.DataFrame:
        """
        Lädt die Metadaten-Datei (CSV oder DBF) und validiert die Spalten.

        Returns:
            DataFrame mit den geladenen Daten
        """
        suffix = self.csv_path.suffix.lower()
        if suffix == ".csv":
            self.df = pd.read_csv(self.csv_path)
        elif suffix == ".dbf":
            try:
                from dbfread import DBF
            except Exception as e:
                raise ImportError(
                    "dbfread is required to read .dbf files. "
                    "Install with `pip install dbfread`."
                ) from e
            # dbfread may not support 'ignore_errors' in this version; try without first,
            # and fall back to explicit encoding if needed.
            # Try several decoding strategies if necessary
            last_exc = None
            for enc in (None, "utf-8", "latin1"):
                try:
                    if enc is None:
                        table = DBF(str(self.csv_path), load=True)
                    else:
                        table = DBF(str(self.csv_path), load=True, encoding=enc)
                    df = pd.DataFrame(list(table))
                    df.columns = [str(c) for c in df.columns]
                    self.df = df
                    last_exc = None
                    break
                except UnicodeDecodeError as e:
                    last_exc = e
                    # try next encoding
                    continue
                except TypeError:
                    # Some versions of dbfread may not accept encoding param as kwarg; try without
                    if enc is None:
                        try:
                            table = DBF(str(self.csv_path), load=True)
                            df = pd.DataFrame(list(table))
                            df.columns = [str(c) for c in df.columns]
                            self.df = df
                            last_exc = None
                            break
                        except Exception as ee:
                            last_exc = ee
                            continue
                    else:
                        last_exc = TypeError(
                            "dbfread does not accept encoding argument in this version"
                        )
                        continue
            if last_exc is not None:
                raise last_exc
        else:
            raise ValueError(
                f"Unbekanntes Dateiformat: {suffix}. Unterstützt: .csv, .dbf"
            )
        # Standardisiere/Mappe Spaltennamen auf erwartete Namen
        self.df = self._standardize_columns(self.df)
        required_columns = ["longName", "center_x", "center_y"]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Fehlende Spalten in Metadaten: {missing}")
        self._validate_coordinates()

        # Optional GeoPandas integration (lazy): if geopandas is available,
        # create a GeoDataFrame with Point geometries and set a heuristic CRS.
        self.gdf = None
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            if "center_x" in self.df.columns and "center_y" in self.df.columns:
                # Build Point geometries (x, y)
                geometries = [
                    Point(float(x), float(y))
                    for x, y in zip(
                        self.df["center_x"].values, self.df["center_y"].values
                    )
                ]
                # Construct GeoDataFrame (tolerant for minimal gpd-like APIs)
                try:
                    gdf = gpd.GeoDataFrame(self.df.copy(), geometry=geometries)
                except Exception:
                    gdf = gpd.GeoDataFrame(self.df.copy())
                    gdf["geometry"] = geometries
                # Heuristic CRS assignment
                if "epsg3857" in str(self.csv_path).lower() or self.crs_unit == "meter":
                    try:
                        gdf.set_crs(epsg=3857, inplace=True, allow_override=True)
                    except TypeError:
                        gdf.crs = "EPSG:3857"
                else:
                    try:
                        gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
                    except TypeError:
                        gdf.crs = "EPSG:4326"
                self.gdf = gdf
        except Exception:
            # geopandas not installed or failed to initialize — continue with pandas-only logic
            self.gdf = None

        return self.df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mappt unterschiedliche Feldnamen auf erwartete Standard-Spalten:
        'longName', optionale 'shortName' und kanonische Spatial-Spalten.
        """
        col_map = {}
        cols = list(df.columns)
        lc = {c.lower(): c for c in cols}

        def _find(candidates):
            # Find column by exact match or by containing the candidate.
            # Avoid matching single-letter candidates inside larger names.
            for cand in candidates:
                cand_lower = cand.lower()
                for col_lower, orig in lc.items():
                    if cand_lower == col_lower:
                        return orig
                    # only allow substring matches for multi-character candidates
                    if len(cand_lower) > 1 and cand_lower in col_lower:
                        return orig
            return None

        # longName candidates
        long_col = _find(
            ["longName", "longname", "long_name", "filename", "file", "name"]
        )
        # shortName candidates
        short_col = _find(["shortName", "shortname", "short_name", "short"])

        if long_col:
            col_map[long_col] = "longName"
        if short_col:
            col_map[short_col] = "shortName"

        df = df.rename(columns=col_map)

        # Canonicalize spatial columns to ul/lr + center.
        df = normalize_spatial_schema(df, require_bounds=True, copy=False)

        # Ensure shortName exists as trimmed string if present
        if "shortName" in df.columns:
            df["shortName"] = df["shortName"].astype(str).str.strip()
        # Ensure longName is cleaned; if missing try to infer from image_path or index
        if "longName" in df.columns:
            df["longName"] = df["longName"].astype(str).str.strip()
        else:
            # Derive longName from image_path basename if available, otherwise fallback to index-based names
            if "image_path" in df.columns:
                df["longName"] = (
                    df["image_path"].astype(str).apply(lambda x: Path(x).name)
                )
            else:
                df["longName"] = [f"img_{i}.png" for i in df.index.astype(str)]

        # Normalize numeric coordinate strings (comma -> dot) and convert to float.
        for col in ("ul_x", "ul_y", "lr_x", "lr_y", "center_x", "center_y"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _infer_source_epsg(self) -> int | None:
        """Infer source EPSG from metadata context when no explicit CRS object exists."""
        if "epsg3857" in str(self.csv_path).lower():
            return 3857
        if self.crs_unit == "meter":
            return 3857
        if self.crs_unit == "degree":
            return 4326
        return None

    def ensure_metric_crs(self, target_epsg: int = 25832, strict: bool = False):
        """
        Ensure that a metric GeoDataFrame exists. If GeoPandas is available,
        reproject `self.gdf` into `target_epsg` and cache it as `self.gdf_metric`.
        Returns the metric GeoDataFrame or None if not available.
        """
        if getattr(self, "gdf_metric", None) is not None:
            return self.gdf_metric

        # Preferred path: GeoPandas-backed reprojection.
        if getattr(self, "gdf", None) is not None:
            try:
                gdf = self.gdf
                # If CRS missing, set heuristically
                if gdf.crs is None:
                    if self.crs_unit == "meter":
                        try:
                            gdf.set_crs(epsg=3857, inplace=True, allow_override=True)
                        except Exception:
                            gdf.crs = "EPSG:3857"
                    else:
                        try:
                            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
                        except Exception:
                            gdf.crs = "EPSG:4326"

                gdf_metric = gdf.to_crs(epsg=target_epsg)
                gdf_metric["_proj_x"] = gdf_metric.geometry.x
                gdf_metric["_proj_y"] = gdf_metric.geometry.y
                self.gdf_metric = gdf_metric
                self.source_crs = str(gdf.crs) if gdf.crs is not None else None
                self.metric_crs = f"EPSG:{target_epsg}"
                self.transform_applied = (
                    self.source_crs is not None
                    and self.source_crs.upper() != self.metric_crs.upper()
                )
                self.metric_epsg = target_epsg
                return self.gdf_metric
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"Could not reproject metadata to EPSG:{target_epsg}: {exc}"
                    ) from exc

        # Fallback path without GeoPandas: transform center coordinates with pyproj.
        try:
            from pyproj import Transformer

            source_epsg = self._infer_source_epsg()
            if source_epsg is None:
                if strict:
                    raise RuntimeError(
                        "Unable to infer source CRS for strict metric projection."
                    )
                self.gdf_metric = None
                return None

            centers_x = self.df["center_x"].to_numpy(dtype=float)
            centers_y = self.df["center_y"].to_numpy(dtype=float)

            if source_epsg == target_epsg:
                proj_x = centers_x
                proj_y = centers_y
            else:
                transformer = Transformer.from_crs(
                    f"EPSG:{source_epsg}",
                    f"EPSG:{target_epsg}",
                    always_xy=True,
                )
                proj_x, proj_y = transformer.transform(centers_x, centers_y)

            gdf_metric = self.df.copy()
            gdf_metric["_proj_x"] = proj_x
            gdf_metric["_proj_y"] = proj_y
            self.gdf_metric = gdf_metric
            self.source_crs = f"EPSG:{source_epsg}"
            self.metric_crs = f"EPSG:{target_epsg}"
            self.transform_applied = source_epsg != target_epsg
            self.metric_epsg = target_epsg
            return self.gdf_metric
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"Strict metric CRS enforcement failed for EPSG:{target_epsg}: {exc}"
                ) from exc
            self.gdf_metric = None
            return None

    def _validate_coordinates(self):
        """
        Prüft heuristisch, ob Koordinaten in Grad (Lat/Lon) oder Meter (Projected) vorliegen.
        Setzt self.crs_unit entsprechend.
        """
        if self.df is None or "center_x" not in self.df.columns:
            return

        x_min, x_max = self.df["center_x"].min(), self.df["center_x"].max()
        y_min, y_max = self.df["center_y"].min(), self.df["center_y"].max()

        if coordinates_look_projected(self.df):
            self.crs_unit = "meter"
            print(
                "[GIS-INFO] Koordinaten scheinen projiziert (Meter) zu sein. "
                f"(Range x={x_min:.1f}..{x_max:.1f}, y={y_min:.1f}..{y_max:.1f})"
            )
            if "epsg3857" in str(self.csv_path).lower():
                print("[GIS-INFO] Dateiname deutet auf EPSG:3857 (Web Mercator) hin.")
        else:
            self.crs_unit = "degree"
            print("[GIS-INFO] Koordinaten scheinen geographisch (Grad) zu sein.")

        if self.crs_unit == "meter":
            print(
                "[WARNUNG] Achtung: Haversine-Distanz ist für Meter-Koordinaten nicht definiert! "
                "Nutze Euklidische Distanz als Fallback."
            )

    def convert_dbf_to_csv(self, output_path: str = None) -> str:
        """
        Konvertiert die geladene DBF-Datei in eine CSV (falls DBF).
        Gibt den Pfad zur erzeugten CSV zurück.
        """
        if self.csv_path.suffix.lower() != ".dbf":
            raise ValueError("convert_dbf_to_csv gilt nur für .dbf Dateien.")
        if self.df is None:
            self.load_csv()
        if output_path is None:
            output_path = str(self.csv_path.with_suffix(".csv"))
        self.df.to_csv(output_path, index=False)
        return output_path

    def resolve_image_paths(
        self,
        image_dir: str | Path,
        prefer_shortname: bool = True,
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Erzeugt eine neue Spalte `image_path` mit dem aufgelösten Pfad zur Bilddatei.
        Bevorzugt `shortName` falls vorhanden; fällt auf `longName` zurück.

        Args:
            image_dir: Verzeichnis mit Bildern (z.B. 'data/images')
            prefer_shortname: ob shortName bevorzugt wird
            strict: wenn True, wird ein fehlendes Bildverzeichnis als Fehler behandelt

        Returns:
            DataFrame mit neuer Spalte `image_path` (Pfad oder None)
        """
        if self.df is None:
            raise ValueError("CSV muss zuerst geladen werden (load_csv)")
        image_dir = Path(image_dir)
        image_dir_exists = image_dir.exists() and image_dir.is_dir()

        if strict and not image_dir_exists:
            raise FileNotFoundError(
                f"Image directory does not exist: {image_dir}. "
                "Feature extraction requires real images."
            )

        def _normalize_existing_path(raw_value):
            """Normalize CSV path cells so NaN/empty values behave like missing paths."""
            if raw_value is None or pd.isna(raw_value):
                return None
            text = str(raw_value).strip()
            if not text or text.lower() in {"nan", "none", "null"}:
                return None
            return text

        if not image_dir_exists:
            # Metadata-only mode: preserve existing image_path values when available.
            if "image_path" in self.df.columns:
                self.df["image_path"] = self.df["image_path"].apply(
                    _normalize_existing_path
                )
            else:
                self.df["image_path"] = [None] * len(self.df)
            self.df["image_filename"] = [
                Path(p).name if p is not None else None for p in self.df["image_path"]
            ]
            return self.df

        def _find_file(name: str):
            if not name or pd.isna(name):
                return None
            name = str(name).strip()
            stem = Path(name).stem
            candidates = [
                image_dir / name,
                image_dir / name.lower(),
                image_dir / (stem + ".png"),
                image_dir / (stem + ".PNG"),
                image_dir / (stem + ".jpg"),
                image_dir / (stem + ".jpeg"),
            ]
            for p in candidates:
                if p.exists():
                    return str(p)
            # fallback: case-insensitive stem search
            for f in image_dir.iterdir():
                if f.is_file() and f.stem.lower() == stem.lower():
                    return str(f)
            return None

        paths = []
        for _, row in self.df.iterrows():
            chosen = None
            has_shortname = (
                prefer_shortname
                and "shortName" in row
                and pd.notna(row["shortName"])
                and str(row["shortName"]).strip()
            )
            if has_shortname:
                chosen = _find_file(row["shortName"])
            if chosen is None and "longName" in row:
                chosen = _find_file(row["longName"])
            paths.append(chosen)

        self.df["image_path"] = paths
        # also add image filename column for backward compatibility
        self.df["image_filename"] = [
            Path(p).name if p is not None else None for p in paths
        ]
        return self.df

    def extract_year(self, filename: str | None) -> int | None:
        """
        Extrahiert das Jahr aus dem Dateinamen.

        Args:
            filename: Dateiname im Format "KDR_XXX_Name_YYYY.png"

        Returns:
            Jahr als Integer
        """
        if filename is None or pd.isna(filename):
            return None

        filename_str = str(filename).strip()
        if not filename_str:
            return None

        match = re.search(r"(\d{4})", filename_str)
        if match:
            return int(match.group(1))
        return None

    def add_temporal_metadata(self) -> pd.DataFrame:
        """
        Fügt zeitliche Metadaten (Jahr) zum DataFrame hinzu.

        Returns:
            DataFrame mit neuer 'year' Spalte
        """
        if self.df is None:
            raise ValueError("CSV muss zuerst geladen werden (load_csv)")

        extracted_years = self.df["longName"].apply(self.extract_year)
        if "year" in self.df.columns:
            existing = pd.to_numeric(self.df["year"], errors="coerce")
            self.df["year"] = existing.fillna(extracted_years)
        else:
            self.df["year"] = extracted_years
        return self.df

    @staticmethod
    def calculate_spatial_distance(y1: float, x1: float, y2: float, x2: float) -> float:
        """
        Berechnet die räumliche Distanz zwischen zwei Punkten.

        Für projizierte Koordinaten (Meter) wird euklidische Distanz in km
        berechnet. Für geographische Koordinaten (Grad) fällt die Methode auf
        Haversine zurück.

        Args:
            y1, x1: Koordinaten Punkt 1 (y/x oder lat/lon)
            y2, x2: Koordinaten Punkt 2 (y/x oder lat/lon)

        Returns:
            Distanz in Kilometern
        """
        if any(abs(v) >= 1000 for v in (y1, x1, y2, x2)):
            dx = x1 - x2
            dy = y1 - y2
            dist_meters = (dx * dx + dy * dy) ** 0.5
            return dist_meters / 1000.0

        # Standard Haversine für Grad-Koordinaten
        from math import atan2, cos, radians, sin, sqrt

        R = 6371  # Erdradius in km

        lat1, lon1, lat2, lon2 = map(radians, [y1, x1, y2, x2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def apply_spatial_filter(self, min_distance_km: float = 50.0) -> List[int]:
        """
        Wendet einen räumlichen Filter an, um zu nahe Kacheln zu vermeiden.

        Args:
            min_distance_km: Minimale Distanz zwischen Kacheln in km

        Returns:
            Liste von Indizes, die den Filter bestehen
        """
        if self.df is None:
            raise ValueError("CSV muss zuerst geladen werden")

        valid_indices = []
        coords = self.df[["center_y", "center_x"]].values

        for i, (y, x) in enumerate(coords):
            is_valid = True
            for j in valid_indices:
                y2, x2 = coords[j]
                if (
                    MetadataProcessor.calculate_spatial_distance(y, x, y2, x2)
                    < min_distance_km
                ):
                    is_valid = False
                    break
            if is_valid:
                valid_indices.append(i)

        return valid_indices

    def get_temporal_range(self) -> Tuple[int, int]:
        """
        Gibt die zeitliche Spanne der Daten zurück.

        Returns:
            Tupel (min_year, max_year)
        """
        if "year" not in self.df.columns:
            self.add_temporal_metadata()

        valid_years = self.df["year"].dropna()
        return (int(valid_years.min()), int(valid_years.max()))

    def get_summary_statistics(self) -> Dict:
        """
        Berechnet zusammenfassende Statistiken über den Datensatz.

        Returns:
            Dictionary mit Statistiken
        """
        if self.df is None:
            raise ValueError("CSV muss zuerst geladen werden")

        if "year" not in self.df.columns:
            self.add_temporal_metadata()

        stats = {
            "total_tiles": len(self.df),
            "temporal_range": self.get_temporal_range(),
            "spatial_extent": {
                "x_min": self.df["center_x"].min(),
                "x_max": self.df["center_x"].max(),
                "y_min": self.df["center_y"].min(),
                "y_max": self.df["center_y"].max(),
            },
            "years_distribution": self.df["year"].value_counts().to_dict(),
        }

        return stats
