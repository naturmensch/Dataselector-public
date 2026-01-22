import sys
import types
import pandas as pd
import pytest
from src.metadata_processor import MetadataProcessor


def _make_csv(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("longName,N,left\nKDR_1_1901.png,52.52,13.405\nKDR_2_1910.png,48.1351,11.5820\n")
    return str(p)


def test_load_csv_without_geopandas(tmp_path):
    csv_path = _make_csv(tmp_path)
    mp = MetadataProcessor(csv_path)
    df = mp.load_csv()
    assert isinstance(df, pd.DataFrame)
    # geopandas not installed: gdf should be None
    assert mp.gdf is None
    assert mp.crs_unit == "degree"


def test_load_csv_with_geopandas_monkeypatch(tmp_path, monkeypatch):
    csv_path = _make_csv(tmp_path)

    # Create a fake geopandas module with a minimal GeoDataFrame constructor
    mod_gpd = types.SimpleNamespace()

    def GeoDataFrame(df, geometry=None):
        gdf = pd.DataFrame(df).copy()
        if geometry is not None:
            gdf["geometry"] = geometry
        # provide a minimal set_crs to mimic geopandas API
        gdf.set_crs = lambda *args, **kwargs: None
        return gdf

    mod_gpd.GeoDataFrame = GeoDataFrame

    # Fake shapely.geometry.Point implementation
    geom_mod = types.ModuleType("shapely.geometry")

    def Point(x, y):
        return (x, y)

    geom_mod.Point = Point

    # Inject into sys.modules
    monkeypatch.setitem(sys.modules, "geopandas", mod_gpd)
    monkeypatch.setitem(sys.modules, "shapely.geometry", geom_mod)

    mp = MetadataProcessor(csv_path)
    df = mp.load_csv()

    # Now gdf should be set and contain geometry
    assert mp.gdf is not None
    assert "geometry" in mp.gdf.columns
    assert mp.gdf.iloc[0]["geometry"] == (13.405, 52.52)
