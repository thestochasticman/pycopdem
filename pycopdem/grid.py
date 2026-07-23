"""The fixed global grid every stored elevation pixel lives on.

The Copernicus GLO-30 DEM is served as 1° x 1° COG tiles on a
1-arc-second (1/3600°) EPSG:4326 lattice anchored at integer degrees.
This module defines the same lattice globally, chunked at 1200 x 1200
pixels (1/3°) — chosen so exactly 3 x 3 chunks nest inside every 1°
tile, which makes each chunk fetchable with a single integer-aligned
windowed read from one COG. Any bbox maps deterministically to a set
of chunk ids, which is what makes the store dedup-able: overlapping
AOIs resolve to overlapping chunk sets, and a chunk is only ever
downloaded once.

All functions here are pure — no I/O, no store access.
"""

PX_PER_DEG = 3600             # 1 arc-second pixels
CHUNK = 1200                  # pixels per chunk edge (1/3 degree)
LON0, LAT_TOP = -180.0, 90.0  # grid origin: top-left corner
WIDTH_PX = 360 * PX_PER_DEG
HEIGHT_PX = 180 * PX_PER_DEG
RES = 1.0 / PX_PER_DEG


def window_for_bbox(bbox: list[float]) -> tuple[int, int, int, int]:
    """Pixel window ``(row0, row1, col0, col1)`` covering ``bbox``, snapped
    outward to whole chunks. Rows increase southward (row 0 at +90°)."""
    west, south, east, north = bbox
    col0 = int((west - LON0) * PX_PER_DEG // CHUNK) * CHUNK
    col1 = -int(-((east - LON0) * PX_PER_DEG) // CHUNK) * CHUNK
    row0 = int((LAT_TOP - north) * PX_PER_DEG // CHUNK) * CHUNK
    row1 = -int(-((LAT_TOP - south) * PX_PER_DEG) // CHUNK) * CHUNK
    return (row0, row1, col0, col1)


def chunks_in_window(window: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """All chunk ids ``(cy, cx)`` inside a chunk-aligned pixel window."""
    row0, row1, col0, col1 = window
    return [
        (cy, cx)
        for cy in range(row0 // CHUNK, row1 // CHUNK)
        for cx in range(col0 // CHUNK, col1 // CHUNK)
    ]


def chunk_bounds(cy: int, cx: int) -> tuple[float, float, float, float]:
    """EPSG:4326 ``(west, south, east, north)`` bounds of a chunk."""
    west = LON0 + cx * CHUNK * RES
    north = LAT_TOP - cy * CHUNK * RES
    return (west, north - CHUNK * RES, west + CHUNK * RES, north)


def tile_of_chunk(cy: int, cx: int) -> tuple[int, int]:
    """The 1° DEM tile ``(lat, lon)`` (south-west corner, integer degrees)
    containing this chunk. Chunks nest exactly: one chunk, one tile."""
    west, south, _, _ = chunk_bounds(cy, cx)
    import math
    return (math.floor(south), math.floor(west))


def tile_name(lat: int, lon: int) -> str:
    """Copernicus GLO-30 tile stem for the 1° cell with SW corner (lat, lon)."""
    ns = f'S{abs(lat):02d}_00' if lat < 0 else f'N{lat:02d}_00'
    ew = f'W{abs(lon):03d}_00' if lon < 0 else f'E{lon:03d}_00'
    return f'Copernicus_DSM_COG_10_{ns}_{ew}_DEM'


def chunk_window_in_tile(cy: int, cx: int) -> tuple[int, int, int, int]:
    """Integer pixel window ``(row0, row1, col0, col1)`` of this chunk
    inside its containing tile's 3600 x 3600 raster."""
    r0 = (cy * CHUNK) % PX_PER_DEG
    c0 = (cx * CHUNK) % PX_PER_DEG
    return (r0, r0 + CHUNK, c0, c0 + CHUNK)


def coords_for_window(window: tuple[int, int, int, int]):
    """Pixel-centre coordinate arrays ``(lat, lon)`` for a window
    (lat descending)."""
    import numpy as np
    row0, row1, col0, col1 = window
    lon = LON0 + (np.arange(col0, col1) + 0.5) * RES
    lat = LAT_TOP - (np.arange(row0, row1) + 0.5) * RES
    return lat, lon


_BBOX = [148.36265, -33.52606, 148.38265, -33.50606]


def test_window_is_chunk_aligned():
    w = window_for_bbox(_BBOX)
    return all(v % CHUNK == 0 for v in w) and w[1] > w[0] and w[3] > w[2]


def test_chunks_nest_in_tiles():
    """Every chunk's pixel window inside its tile must be within [0, 3600]
    and aligned to CHUNK."""
    for cy, cx in chunks_in_window(window_for_bbox(_BBOX)):
        r0, r1, c0, c1 = chunk_window_in_tile(cy, cx)
        if not (0 <= r0 < r1 <= PX_PER_DEG and 0 <= c0 < c1 <= PX_PER_DEG):
            return False
        if r0 % CHUNK or c0 % CHUNK:
            return False
    return True


def test_tile_naming():
    return tile_name(-34, 148) == 'Copernicus_DSM_COG_10_S34_00_E148_00_DEM'


def test_chunk_bounds_inside_tile():
    for cy, cx in chunks_in_window(window_for_bbox(_BBOX)):
        tlat, tlon = tile_of_chunk(cy, cx)
        w, s, e, n = chunk_bounds(cy, cx)
        if not (tlon <= w and e <= tlon + 1 and tlat <= s and n <= tlat + 1):
            return False
    return True


def test_overlapping_bboxes_share_chunks():
    a = window_for_bbox(_BBOX)
    b = window_for_bbox([_BBOX[0] + 0.05, _BBOX[1], _BBOX[2] + 0.05, _BBOX[3]])
    return len(set(chunks_in_window(a)) & set(chunks_in_window(b))) > 0


def test():
    return all([
        test_window_is_chunk_aligned(),
        test_chunks_nest_in_tiles(),
        test_tile_naming(),
        test_chunk_bounds_inside_tile(),
        test_overlapping_bboxes_share_chunks(),
    ])


if __name__ == '__main__':
    print(test())
