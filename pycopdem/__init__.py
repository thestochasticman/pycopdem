# Light-weight exports only: pycopdem.store (and the download_terrain
# wrapper) pull in rasterio/zarr/pysheds, so those stay behind explicit
# submodule imports.
from pycopdem.copdem import CopernicusDEM, defaultcopdem
from pycopdem.paths import Paths

__all__ = [
    'CopernicusDEM',
    'defaultcopdem',
    'Paths',
]
