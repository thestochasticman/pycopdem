"""Fetch the elevation window (+ derivatives) for a query — via the
machine-wide store.

Thin compatibility wrapper: the heavy lifting (chunk-level dedup,
windowed COG reads, on-read derivatives) lives in
:class:`pycopdem.store.Store`. Kept as a module so the familiar
``download_terrain(query)`` entry point survives.
"""
import xarray as xr
from borevitz_lab.query import Query
from pycopdem.copdem import CopernicusDEM, defaultcopdem


def download_terrain(query: Query, derivatives: tuple[str, ...] = (),
                     copdem: CopernicusDEM = defaultcopdem) -> xr.Dataset:
    """Return the Copernicus 30 m elevation window for ``query.bbox``.

    Fetches only the grid chunks no previous request has populated —
    repeat and overlapping queries re-download nothing. Dates on the
    query are ignored: elevation is time-invariant.

    Args:
        query: The :class:`borevitz_lab.query.Query` (bbox is what matters).
        derivatives: Any of ``'slope'``, ``'aspect'``, ``'accumulation'``,
            ``'twi'``, ``'hli'`` — computed on read, never stored.
        copdem: Tile-source configuration; defaults to Copernicus on AWS.

    Returns:
        xarray.Dataset with dims ``(lat, lon)``: ``elevation`` plus one
        variable per requested derivative.
    """
    from pycopdem.store import Store
    store = Store(config=query.config, copdem=copdem)
    return store.get_ds_query(query, derivatives=derivatives)


def test_live_fetch_and_dedup():
    """Live: cold fetch covers the bbox; repeat and overlapping bboxes
    fetch nothing; derivatives come back finite."""
    import numpy as np
    import tempfile
    from borevitz_lab.config import Config
    from pycopdem.store import Store

    tmpdir = tempfile.mkdtemp(prefix='pycopdem_live_test_')
    cfg = Config(out_dir=tmpdir, tmp_dir=tmpdir)
    store = Store(config=cfg)
    bbox = [148.36265, -33.52606, 148.38265, -33.50606]

    fetched = store.fill(bbox)
    if fetched < 1:
        return False
    ds = store.get_ds(bbox)
    elev = ds['elevation'].values
    if not np.isfinite(elev).any() or float(np.nanmax(elev)) <= 0:
        return False
    # identical repeat -> nothing
    if store.fill(bbox) != 0:
        return False
    # overlapping bbox shifted ~2 km -> shares chunks, fetches at most the difference
    shifted = [bbox[0] + 0.02, bbox[1], bbox[2] + 0.02, bbox[3]]
    if store.fill(shifted) != 0:  # within the same 1/3-degree chunks here
        return False
    # derivatives computed on read
    dsd = store.get_ds(bbox, derivatives=('slope', 'aspect', 'twi', 'hli'))
    return (
        float(dsd['slope'].max()) >= 0
        and np.isfinite(dsd['twi'].values).any()
        and np.isfinite(dsd['hli'].values).any()
    )


def test():
    return test_live_fetch_and_dedup()


if __name__ == '__main__':
    print(test())
