"""
Metadaten-Extraktion und CSV-Processing für KDR100 Kacheln.

Dieses Modul liest die CSV-Datei ein und extrahiert kritische Metadaten:
- Temporaler Stempel (Jahr aus longName)
- Räumliche Koordinaten (N, left)
- Dateiinformationen
"""

import re
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import pandas as pd


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

    def load_csv(self) -> pd.DataFrame:
        """
        Lädt die Metadaten-Datei (CSV oder DBF) und validiert die Spalten.

        Returns:
            DataFrame mit den geladenen Daten
        """
        suffix = self.csv_path.suffix.lower()
        if suffix == ".csv":
            self.df = pd.read_csv(self.csv_path)
            # Normalize column names for compatibility
            if 'lat' in self.df.columns and 'N' not in self.df.columns:
                self.df = self.df.rename(columns={'lat': 'N'})
            if 'lon' in self.df.columns and 'left' not in self.df.columns:
                self.df = self.df.rename(columns={'lon': 'left'})
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
        required_columns = ["longName", "N", "left"]
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

            if "left" in self.df.columns and "N" in self.df.columns:
                # Build Point geometries (lon, lat)
                geometries = [
                    Point(float(x), float(y))
                    for x, y in zip(self.df["left"].values, self.df["N"].values)
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
        'longName', 'N', 'left'.
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
        lat_col = _find(["N", "n", "lat", "latitude", "y"])
        lon_col = _find(["left", "lon", "longitude", "x", "long"])

        if long_col:
            col_map[long_col] = "longName"
        if short_col:
            col_map[short_col] = "shortName"
        if lat_col:
            col_map[lat_col] = "N"
        if lon_col:
            col_map[lon_col] = "left"

        df = df.rename(columns=col_map)
        # Deriviere mittlere Koordinate, falls explizite Felder fehlen
        # Latitude (N) aus TOP/BOTTOM
        if "N" not in df.columns:
            top_cols = [c for c in df.columns if c.lower() == "top"]
            bot_cols = [c for c in df.columns if c.lower() == "bottom"]
            if top_cols and bot_cols:
                df["N"] = (df[top_cols[0]] + df[bot_cols[0]]) / 2
        # Longitude (left) aus LEFT/RIGHT
        if "left" not in df.columns:
            left_cols = [c for c in df.columns if c.lower() == "left"]
            right_cols = [c for c in df.columns if c.lower() == "right"]
            if left_cols and right_cols:
                df["left"] = (df[left_cols[0]] + df[right_cols[0]]) / 2

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
                df["longName"] = df.index.astype(str).apply(lambda i: f"img_{i}.png")

        # Normalize numeric coordinate strings (comma -> dot) and convert to float
        # Use robust conversions to avoid errors from mixed types or unexpected objects
        if "N" in df.columns:
            df["N"] = df["N"].astype(str).str.replace(",", ".", regex=False)
            df["N"] = pd.to_numeric(df["N"], errors="coerce")
        if "left" in df.columns:
            df["left"] = df["left"].astype(str).str.replace(",", ".", regex=False)
            df["left"] = pd.to_numeric(df["left"], errors="coerce")

        return df

    def ensure_metric_crs(self, target_epsg: int = 25832):
        """
        Ensure that a metric GeoDataFrame exists. If GeoPandas is available,
        reproject `self.gdf` into `target_epsg` and cache it as `self.gdf_metric`.
        Returns the metric GeoDataFrame or None if not available.
        """
        if getattr(self, "gdf", None) is None:
            return None

        if getattr(self, "gdf_metric", None) is not None:
            return self.gdf_metric

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
            self.metric_epsg = target_epsg
            return self.gdf_metric
        except Exception:
            self.gdf_metric = None
            return None

    def _validate_coordinates(self):
        """
        Prüft heuristisch, ob Koordinaten in Grad (Lat/Lon) oder Meter (Projected) vorliegen.
        Setzt self.crs_unit entsprechend.
        """
        if self.df is None or "N" not in self.df.columns:
            return

        lat_min, lat_max = self.df["N"].min(), self.df["N"].max()
        lon_min, lon_max = self.df["left"].min(), self.df["left"].max()

        # Heuristik: Wenn Werte außerhalb [-180, 180], sind es wahrscheinlich Meter
        if (lat_min < -90 or lat_max > 90) or (lon_min < -180 or lon_max > 180):
            self.crs_unit = "meter"
            print(
                f"[GIS-INFO] Koordinaten scheinen projiziert (Meter) zu sein. (Range: N={lat_min:.1f}..{lat_max:.1f})"
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
        self, image_dir: str | Path, prefer_shortname: bool = True
    ) -> pd.DataFrame:
        """
        Erzeugt eine neue Spalte `image_path` mit dem aufgelösten Pfad zur Bilddatei.
        Bevorzugt `shortName` falls vorhanden; fällt auf `longName` zurück.

        Args:
            image_dir: Verzeichnis mit Bildern (z.B. 'data/images')
            prefer_shortname: ob shortName bevorzugt wird

        Returns:
            DataFrame mit neuer Spalte `image_path` (Pfad oder None)
        """
        if self.df is None:
            raise ValueError("CSV muss zuerst geladen werden (load_csv)")
        image_dir = Path(image_dir)

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

    def extract_year(self, filename: str) -> int:
        """
        Extrahiert das Jahr aus dem Dateinamen.

        Args:
            filename: Dateiname im Format "KDR_XXX_Name_YYYY.png"

        Returns:
            Jahr als Integer
        """
        match = re.search(r"(\d{4})", filename)
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

        self.df["year"] = self.df["longName"].apply(self.extract_year)
        return self.df

    @staticmethod
    def calculate_spatial_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Berechnet die räumliche Distanz zwischen zwei Punkten.

        Wenn eine reprojektierte GeoDataFrame (`self.gdf_metric`) vorhanden ist,
        wird die euklidische Distanz in Metern berechnet und in Kilometer
        zurückgegeben. Ansonsten fällt die Methode auf die sphärische
        Haversine-Distanz zurück (Grad).

        Args:
            lat1, lon1: Koordinaten Punkt 1
            lat2, lon2: Koordinaten Punkt 2

        Returns:
            Distanz in Kilometern
        """
        # If we have projected coords available (meters), prefer them, but only if inputs
        # look like metric coordinates (large magnitude). Otherwise fall back to Haversine.
        if getattr(self, "gdf_metric", None) is not None:
            try:
                # Heuristic: if coordinates magnitude suggests meters (abs >= 1000)
                # Use any() so a projected coordinate pair like (0,0) and (0,1000) is treated
                # as metric rather than mistakenly falling back to Haversine.
                if any(abs(v) >= 1000 for v in (lat1, lon1, lat2, lon2)):
                    dx = lon1 - lon2
                    dy = lat1 - lat2
                    dist_meters = (dx * dx + dy * dy) ** 0.5
                    return dist_meters / 1000.0
                # else: inputs seem to be degrees → fall through to Haversine
            except Exception:
                # Fall back to Haversine if any error occurs
                pass

        # Standard Haversine für Grad-Koordinaten
        from math import atan2, cos, radians, sin, sqrt

        R = 6371  # Erdradius in km

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
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
        coords = self.df[["N", "left"]].values

        for i, (lat, lon) in enumerate(coords):
            is_valid = True
            for j in valid_indices:
                lat2, lon2 = coords[j]
                if (
                    MetadataProcessor.calculate_spatial_distance(lat, lon, lat2, lon2)
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
                "lat_min": self.df["N"].min(),
                "lat_max": self.df["N"].max(),
                "lon_min": self.df["left"].min(),
                "lon_max": self.df["left"].max(),
            },
            "years_distribution": self.df["year"].value_counts().to_dict(),
        }

        return stats
