from rasterio import open as ropen
from pathlib import Path
p = Path('data/images/KDR_327.png')
print('file exists:', p.exists())
try:
    with ropen(p) as ds:
        print('bounds:', ds.bounds)
        print('crs:', ds.crs)
        print('w,h:', ds.width, ds.height)
except Exception as e:
    print('error opening with rasterio:', e)
