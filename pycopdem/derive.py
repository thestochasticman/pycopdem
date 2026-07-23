"""Topographic derivatives, computed on read — never stored.

Slope, aspect, and Heat Load Index are cheap array math; flow
accumulation and TWI run the pysheds conditioning chain (fill pits →
fill depressions → resolve flats → flow direction → accumulation).
Everything operates on an elevation window plus its pixel size, so the
functions stay pure of store concerns.
"""
import numpy as np

# pysheds 0.5 calls np.in1d which was removed in NumPy 2.0. Alias to np.isin
# (the documented replacement, same semantics) before pysheds is imported.
if not hasattr(np, 'in1d'):
    np.in1d = np.isin


def slope(dem: np.ndarray, xres: float, yres: float) -> np.ndarray:
    """Slope in degrees from an elevation array and pixel sizes (metres)."""
    gy, gx = np.gradient(dem, yres, xres)
    return np.arctan(np.sqrt(gx ** 2 + gy ** 2)) * (180 / np.pi)


def aspect(dem: np.ndarray, xres: float, yres: float) -> np.ndarray:
    """Aspect (direction of steepest descent) in degrees, 0/360=N 90=E 180=S 270=W."""
    gy, gx = np.gradient(dem, yres, xres)
    deg = np.degrees(np.arctan2(-gx, gy))
    return np.where(deg < 0, deg + 360, deg)


def accumulation(dem: np.ndarray, transform, nodata: float = -9999.0) -> np.ndarray:
    """Flow accumulation (cells) via pysheds on an in-memory window.

    The window is written to a temporary GeoTIFF because pysheds'
    conditioning chain reads from raster files.
    """
    import os
    import tempfile
    import rasterio
    from pysheds.grid import Grid

    filled = np.where(np.isnan(dem), nodata, dem).astype('float32')
    fd, path = tempfile.mkstemp(suffix='.tif')
    os.close(fd)
    try:
        with rasterio.open(
            path, 'w', driver='GTiff', height=dem.shape[0], width=dem.shape[1],
            count=1, dtype='float32', transform=transform, crs='EPSG:4326',
            nodata=nodata,
        ) as dst:
            dst.write(filled, 1)
        grid = Grid.from_raster(path, nodata=nodata)
        raster = grid.read_raster(path, nodata=nodata)
        inflated = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(raster)))
        fdir = grid.flowdir(inflated, nodata_out=0)
        return np.asarray(grid.accumulation(fdir, nodata_out=0))
    finally:
        os.remove(path)


def twi(acc: np.ndarray, slope_deg: np.ndarray) -> np.ndarray:
    """Topographic wetness index: ln(accumulation / tan(slope))."""
    ratio = acc / np.tan(np.radians(slope_deg))
    ratio[ratio <= 0] = 1
    return np.log(ratio)


def hli(slope_deg: np.ndarray, aspect_deg: np.ndarray, latitude: float) -> np.ndarray:
    """Heat Load Index (McCune & Keon 2002), ~0-1; higher = more insolation."""
    slope_rad = np.radians(slope_deg)
    lat_rad = np.radians(latitude)
    folded = np.radians(np.abs(180 - np.abs(aspect_deg - 225)))
    out = np.exp(
        -1.467
        + 1.582 * np.cos(lat_rad) * np.cos(slope_rad)
        - 1.5 * np.cos(folded) * np.sin(slope_rad) * np.sin(lat_rad)
        - 0.262 * np.sin(lat_rad) * np.sin(slope_rad)
        + 0.607 * np.sin(folded) * np.sin(slope_rad)
    )
    return np.clip(out, 0, 1)


def test_slope_of_inclined_plane():
    """A plane rising 1 m per metre east has slope 45° everywhere inside."""
    x = np.arange(50, dtype='float64')
    dem = np.tile(x, (50, 1))          # elevation == x coordinate (1 m pixels)
    s = slope(dem, 1.0, 1.0)
    return abs(float(s[25, 25]) - 45.0) < 1e-6


def test_aspect_points_downhill():
    """Rising to the east -> steepest descent faces west (270°)."""
    x = np.arange(50, dtype='float64')
    dem = np.tile(x, (50, 1))
    a = aspect(dem, 1.0, 1.0)
    return abs(float(a[25, 25]) - 270.0) < 1e-6


def test_flat_ground_hli_matches_formula():
    h = hli(np.zeros((5, 5)), np.zeros((5, 5)), -33.5)
    expected = np.exp(-1.467 + 1.582 * np.cos(np.radians(-33.5)))
    return abs(float(h[2, 2]) - min(expected, 1.0)) < 1e-9


def test_twi_monotonic_in_accumulation():
    s = np.full((3, 3), 10.0)
    low = twi(np.full((3, 3), 10.0), s)
    high = twi(np.full((3, 3), 1000.0), s)
    return float(high[1, 1]) > float(low[1, 1])


def test():
    return all([
        test_slope_of_inclined_plane(),
        test_aspect_points_downhill(),
        test_flat_ground_hli_matches_formula(),
        test_twi_monotonic_in_accumulation(),
    ])


if __name__ == '__main__':
    print(test())
