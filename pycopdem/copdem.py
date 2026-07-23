from attrs import frozen

@frozen
class CopernicusDEM:
    """Endpoint configuration for Copernicus GLO-30 DEM tiles on AWS.

    The DEM is global, 30 m (1 arc-second), served as public
    Cloud-Optimised GeoTIFFs — one per 1° x 1° cell. Tiles that would
    be entirely ocean do not exist; the store treats a missing tile as
    all-nodata.
    """

    base_url: str = 'https://copernicus-dem-30m.s3.amazonaws.com'

    def tile_url(self, tile: str) -> str:
        return f'{self.base_url}/{tile}/{tile}.tif'

defaultcopdem = CopernicusDEM()
