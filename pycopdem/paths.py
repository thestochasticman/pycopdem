"""Derived on-disk locations of the machine-wide elevation store.

The store is keyed by :class:`borevitz_lab.config.Config` (one store per
data root, shared by every request on this machine). Rule of thumb
across the lab's packages: user-settable inputs → Config, derived
locations → Paths. No inheritance — composition only.
"""
from attrs import frozen, field
from borevitz_lab.config import Config, config as default_config


@frozen
class Paths:
    """Where the pycopdem store lives for a given Config.

    Attributes:
        config: The :class:`borevitz_lab.config.Config` supplying the data root.
        root: Store directory (``{config.tmp_dir}/copdem_store``).
        store: The sparse Zarr store holding every downloaded elevation chunk.
        index_db: SQLite ledger of populated chunks.

    Example:
        ```python
        from pycopdem.paths import Paths

        Paths().store  # '~/Downloads/BorevitzLab-Tmp/copdem_store/dem.zarr'
        ```
    """

    config: Config = default_config

    root: str = field(init=False)
    store: str = field(init=False)
    index_db: str = field(init=False)

    root.default(lambda s: f'{s.config.tmp_dir}/copdem_store')
    store.default(lambda s: f'{s.root}/dem.zarr')
    index_db.default(lambda s: f'{s.root}/index.db')


def test_paths_derive_from_config():
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix='pycopdem_paths_test_')
    cfg = Config(out_dir=tmpdir, tmp_dir=tmpdir)
    paths = Paths(cfg)
    return (
        paths.root == f'{tmpdir}/copdem_store'
        and paths.store == f'{tmpdir}/copdem_store/dem.zarr'
        and paths.index_db == f'{tmpdir}/copdem_store/index.db'
    )


def test():
    return test_paths_derive_from_config()


if __name__ == '__main__':
    print(test())
