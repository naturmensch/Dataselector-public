import math

import geopandas as gpd
from shapely.geometry import Point

from src.spatial_facility_location import haversine_distance


def utm_distance_km(lat1, lon1, lat2, lon2, crs='EPSG:25832'):
    gdf = gpd.GeoDataFrame(
        geometry=[Point(lon1, lat1), Point(lon2, lat2)], crs='EPSG:4326'
    )
    gdf = gdf.to_crs(crs)
    x = gdf.geometry.x
    y = gdf.geometry.y
    dx = x.iloc[0] - x.iloc[1]
    dy = y.iloc[0] - y.iloc[1]
    return math.hypot(dx, dy) / 1000.0


def test_berlin_cologne_agree_within_reasonable_tolerance():
    # Berlin (52.5200, 13.4050), Cologne (50.9375, 6.9603)
    lat_b, lon_b = 52.52, 13.405
    lat_c, lon_c = 50.9375, 6.9603

    hav = haversine_distance(lat_b, lon_b, lat_c, lon_c)
    utm = utm_distance_km(lat_b, lon_b, lat_c, lon_c)

    # Difference should be small relative to distance; allow 0.5% or 1 km whichever is larger
    allowed = max(0.005 * ((hav + utm) / 2.0), 1.0)
    assert abs(hav - utm) < allowed, f"Haversine {hav} vs UTM {utm} differ more than {allowed} km"
