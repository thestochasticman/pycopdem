"""One machine-wide elevation store that fills itself on demand.

Every elevation pixel this machine ever downloads lands in a single
sparse Zarr array on the fixed 1-arc-second EPSG:4326 grid
(:mod:`pycopdem.grid`):

    {config.tmp_dir}/copdem_store/
    ├── index.db      # SQLite ledger: which chunks are populated
    └── dem.zarr      # global sparse array; only written chunks exist on disk

Elevation is time-invariant, so there is no time axis and no date
bookkeeping — the ledger is just the set of populated ``(cy, cx)``
chunks. ``Store.get_ds(bbox)`` diffs the requested chunks against the
ledger, fetches only the missing ones (each chunk is a single
integer-aligned windowed read from one Copernicus COG tile), then
reads the window. Derivatives — slope, aspect, flow accumulation,
TWI, HLI — are computed on read (:mod:`pycopdem.derive`), never stored.
A 1° tile that does not exist on S3 is genuinely all ocean; its chunks
are stored as nodata and marked complete.
"""
import sqlite3
from attrs import frozen, field
from datetime import datetime, timezone
from os import makedirs

import numpy as np
import xarray as xr
import zarr

from borevitz_lab.config import Config, config as default_config
from pycopdem import derive, grid
from pycopdem.copdem import CopernicusDEM, defaultcopdem
from pycopdem.paths import Paths

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    cy INTEGER NOT NULL,
    cx INTEGER NOT NULL,
    written_at TEXT NOT NULL,
    PRIMARY KEY (cy, cx)
) WITHOUT ROWID;
"""

DERIVATIVES = ('slope', 'aspect', 'accumulation', 'twi', 'hli')

# Metres per degree of latitude (per degree of longitude scales by cos(lat)).
_M_PER_DEG = 111_320.0


@frozen
class Store:
    """The machine-wide elevation store: one grid, one ledger, zero re-downloads.

    Composed from :class:`borevitz_lab.config.Config` (where the store
    lives) and :class:`pycopdem.copdem.CopernicusDEM` (where tiles come
    from). No inheritance.

    Example:
        ```python
        from pycopdem.store import Store

        store = Store()
        ds = store.get_ds(bbox)                                 # elevation only
        ds = store.get_ds(bbox, derivatives=('slope', 'twi'))   # + derived layers
        ```
    """

    config: Config = default_config
    copdem: CopernicusDEM = defaultcopdem
    paths: Paths = field(init=False)

    paths.default(lambda s: Paths(s.config))

    def __attrs_post_init__(s):
        makedirs(s.paths.root, exist_ok=True)

    def _db(s) -> sqlite3.Connection:
        db = sqlite3.connect(s.paths.index_db)
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript(_SCHEMA)
        return db

    def _array(s, mode: str = 'a') -> zarr.Array:
        root = zarr.open_group(s.paths.store, mode=mode)
        try:
            return root['elevation']
        except KeyError:
            return root.create_array(
                'elevation',
                shape=(grid.HEIGHT_PX, grid.WIDTH_PX),
                chunks=(grid.CHUNK, grid.CHUNK),
                dtype='float32',
                fill_value=np.nan,
            )

    # -- fill -------------------------------------------------------------

    def fill(s, bbox: list[float]) -> int:
        """Ensure every chunk covering ``bbox`` is populated.

        Query-agnostic (and, elevation being time-invariant, date-free).
        Returns the number of chunks actually downloaded — 0 means the
        request was already fully covered and no network was touched.
        """
        wanted = grid.chunks_in_window(grid.window_for_bbox(bbox))
        db = s._db()
        try:
            done = set(db.execute('SELECT cy, cx FROM chunks').fetchall())
            missing = [c for c in wanted if c not in done]
            if not missing:
                return 0
            arr = s._array()
            for cy, cx in missing:
                s._fetch_chunk(arr, cy, cx)
                with db:
                    db.execute(
                        'INSERT OR REPLACE INTO chunks (cy, cx, written_at) VALUES (?, ?, ?)',
                        (cy, cx, datetime.now(timezone.utc).isoformat()),
                    )
            return len(missing)
        finally:
            db.close()

    def _fetch_chunk(s, arr: zarr.Array, cy: int, cx: int) -> None:
        """One integer-aligned windowed read from the chunk's COG tile."""
        import rasterio
        from rasterio.windows import Window

        tile = grid.tile_name(*grid.tile_of_chunk(cy, cx))
        r0, r1, c0, c1 = grid.chunk_window_in_tile(cy, cx)
        try:
            with rasterio.open(s.copdem.tile_url(tile)) as src:
                data = src.read(
                    1, window=Window(c0, r0, c1 - c0, r1 - r0)
                ).astype('float32')
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)
        except rasterio.errors.RasterioIOError:
            # No tile on S3 -> all ocean. Nodata is the truth for this chunk.
            data = np.full((grid.CHUNK, grid.CHUNK), np.nan, dtype='float32')
        arr[cy * grid.CHUNK:(cy + 1) * grid.CHUNK,
            cx * grid.CHUNK:(cx + 1) * grid.CHUNK] = data

    # -- read -------------------------------------------------------------

    def get_ds(s, bbox: list[float], derivatives: tuple[str, ...] = ()) -> xr.Dataset:
        """Return the elevation window for ``bbox``, downloading only what's
        missing first, with any requested derivatives computed on read.

        Args:
            bbox: ``[west, south, east, north]`` in EPSG:4326.
            derivatives: Any of ``'slope'``, ``'aspect'``,
                ``'accumulation'``, ``'twi'``, ``'hli'`` (dependencies are
                resolved automatically — e.g. ``'twi'`` implies computing
                slope and accumulation).

        Returns:
            xarray.Dataset with dims ``(lat, lon)`` on the fixed grid:
            ``elevation`` plus one variable per requested derivative.
        """
        unknown = set(derivatives) - set(DERIVATIVES)
        if unknown:
            raise ValueError(f'Unknown derivative(s): {sorted(unknown)} — pick from {DERIVATIVES}')
        s.fill(bbox)

        window = grid.window_for_bbox(bbox)
        row0, row1, col0, col1 = window
        dem = s._array(mode='r')[row0:row1, col0:col1]
        lat, lon = grid.coords_for_window(window)
        ds = xr.Dataset(
            {'elevation': (('lat', 'lon'), dem)},
            coords={'lat': lat, 'lon': lon},
            attrs={'crs': 'EPSG:4326', 'resolution_arcsec': 1},
        )

        if derivatives:
            centre_lat = float(lat.mean())
            xres = grid.RES * _M_PER_DEG * np.cos(np.radians(centre_lat))
            yres = grid.RES * _M_PER_DEG
            slope_deg = derive.slope(dem, xres, yres)
            need_acc = 'accumulation' in derivatives or 'twi' in derivatives
            acc = None
            if need_acc:
                from affine import Affine
                transform = Affine(grid.RES, 0, grid.LON0 + col0 * grid.RES,
                                   0, -grid.RES, grid.LAT_TOP - row0 * grid.RES)
                acc = derive.accumulation(dem, transform)
            if 'slope' in derivatives:
                ds['slope'] = (('lat', 'lon'), slope_deg)
            if 'aspect' in derivatives or 'hli' in derivatives:
                aspect_deg = derive.aspect(dem, xres, yres)
                if 'aspect' in derivatives:
                    ds['aspect'] = (('lat', 'lon'), aspect_deg)
                if 'hli' in derivatives:
                    ds['hli'] = (('lat', 'lon'), derive.hli(slope_deg, aspect_deg, centre_lat))
            if 'accumulation' in derivatives:
                ds['accumulation'] = (('lat', 'lon'), acc)
            if 'twi' in derivatives:
                ds['twi'] = (('lat', 'lon'), derive.twi(acc, slope_deg))
        return ds

    # -- Query adapter (the reproducibility layer speaks Query) -----------

    def fill_query(s, query) -> int:
        """:meth:`fill` for a :class:`borevitz_lab.query.Query` (dates ignored
        — elevation is time-invariant)."""
        return s.fill(query.bbox)

    def get_ds_query(s, query, derivatives: tuple[str, ...] = ()) -> xr.Dataset:
        """:meth:`get_ds` for a :class:`borevitz_lab.query.Query`."""
        return s.get_ds(query.bbox, derivatives=derivatives)


# -- offline tests (synthetic writes, no network) --------------------------

_TEST_BBOX = [148.36265, -33.52606, 148.38265, -33.50606]


def _tmp_store() -> Store:
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix='pycopdem_store_test_')
    return Store(config=Config(out_dir=tmpdir, tmp_dir=tmpdir))


def _prime_synthetic(store: Store, bbox, value: float = 500.0, tilt: float = 0.0):
    """Mark every chunk of bbox populated; elevation = value + tilt x column
    (a plane rising eastward when tilt > 0)."""
    window = grid.window_for_bbox(bbox)
    arr = store._array()
    db = store._db()
    with db:
        for cy, cx in grid.chunks_in_window(window):
            cols = np.arange(cx * grid.CHUNK, (cx + 1) * grid.CHUNK, dtype='float32')
            block = np.tile(value + tilt * cols, (grid.CHUNK, 1))
            arr[cy * grid.CHUNK:(cy + 1) * grid.CHUNK,
                cx * grid.CHUNK:(cx + 1) * grid.CHUNK] = block
            db.execute(
                'INSERT OR REPLACE INTO chunks (cy, cx, written_at) VALUES (?, ?, ?)',
                (cy, cx, 'synthetic'),
            )
    db.close()


def test_synthetic_write_read_roundtrip():
    store = _tmp_store()
    _prime_synthetic(store, _TEST_BBOX, value=512.0)
    ds = store.get_ds(_TEST_BBOX)
    return (
        float(ds['elevation'][0, 0]) == 512.0
        and ds.lat[0] > ds.lat[-1]  # lat descending
    )


def test_fill_skips_populated_chunks():
    store = _tmp_store()
    _prime_synthetic(store, _TEST_BBOX)
    return store.fill(_TEST_BBOX) == 0


def test_flat_ground_has_zero_slope():
    store = _tmp_store()
    _prime_synthetic(store, _TEST_BBOX, value=300.0)
    ds = store.get_ds(_TEST_BBOX, derivatives=('slope',))
    return float(ds['slope'].max()) == 0.0


def test_derivatives_on_tilted_plane():
    """A plane rising eastward -> positive slope, west-facing aspect,
    finite TWI, HLI in (0, 1]."""
    store = _tmp_store()
    _prime_synthetic(store, _TEST_BBOX, value=300.0, tilt=0.5)
    ds = store.get_ds(_TEST_BBOX, derivatives=('slope', 'aspect', 'twi', 'hli'))
    return (
        float(ds['slope'].min()) > 0.0
        and abs(float(ds['aspect'][5, 5]) - 270.0) < 1.0
        and np.isfinite(ds['twi'].values).all()
        and 0 < float(ds['hli'][5, 5]) <= 1
    )


def test_unknown_derivative_raises():
    store = _tmp_store()
    _prime_synthetic(store, _TEST_BBOX)
    try:
        store.get_ds(_TEST_BBOX, derivatives=('curvature',))
    except ValueError:
        return True
    return False


def test():
    return all([
        test_synthetic_write_read_roundtrip(),
        test_fill_skips_populated_chunks(),
        test_flat_ground_has_zero_slope(),
        test_derivatives_on_tilted_plane(),
        test_unknown_derivative_raises(),
    ])


if __name__ == '__main__':
    print(test())
