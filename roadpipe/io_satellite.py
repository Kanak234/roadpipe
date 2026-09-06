"""
roadpipe.io_satellite
Real satellite-mode input. Reads a multi-band GeoTIFF (e.g. a Sentinel-2 tile
of Hazaribagh downloaded from the Copernicus Browser, or ISRO Bhuvan / LISS-IV)
via rasterio, builds an RGB composite for the segmentation model, and keeps the
affine transform + CRS so every graph node can be mapped back to lon/lat.

Why not Google Earth / Google Maps: those tiles are copyright-restricted and
cannot be redistributed or used for model training. Sentinel-2 (Copernicus),
ISRO Bhuvan and OpenStreetMap are openly licensed and are what PS04 recommends.

Requires: rasterio  (optional; only needed for this mode).
"""

import numpy as np


def rasterio_available():
    try:
        import rasterio  # noqa: F401
        return True
    except Exception:
        return False


def _stretch(band, p_low=2, p_high=98):
    """Percentile contrast stretch to 0..255 uint8."""
    lo, hi = np.percentile(band, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((band - lo) / (hi - lo), 0, 1) * 255
    return out.astype(np.uint8)


def load_geotiff(path, rgb_bands=(4, 3, 2), out_size=None):
    """
    Load a GeoTIFF and return:
      rgb        : HxWx3 uint8 contrast-stretched composite
      ndvi       : HxW float32 NDVI (or None if NIR/Red unavailable)
      geo        : dict(transform, crs, bounds, width, height)

    rgb_bands : 1-indexed band numbers for (R, G, B). Sentinel-2 L2A as commonly
                stacked: B4=Red, B3=Green, B2=Blue, B8=NIR. Adjust to your file.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        count = ds.count
        # Choose bands safely
        rb, gb, bb = rgb_bands
        rb = min(rb, count); gb = min(gb, count); bb = min(bb, count)

        def read(b):
            if out_size:
                return ds.read(b, out_shape=(out_size[1], out_size[0]),
                               resampling=Resampling.bilinear).astype(np.float32)
            return ds.read(b).astype(np.float32)

        R = read(rb); G = read(gb); B = read(bb)
        rgb = np.dstack([_stretch(R), _stretch(G), _stretch(B)])

        ndvi = None
        # NIR is band 8 in Sentinel-2; use it if present for vegetation masking
        nir_band = 8 if count >= 8 else None
        if nir_band:
            NIR = read(nir_band)
            denom = (NIR + R)
            denom[denom == 0] = 1e-6
            ndvi = (NIR - R) / denom

        geo = {
            "transform": ds.transform,
            "crs": str(ds.crs),
            "bounds": tuple(ds.bounds),
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "orig_transform": ds.transform,
            "orig_size": (ds.width, ds.height),
        }
    return rgb, ndvi, geo


def pixel_to_lonlat(px, py, geo):
    """Map a pixel (col, row) in the working raster back to lon/lat.

    Accounts for any resize between the original GeoTIFF and the working image.
    """
    import rasterio
    from rasterio.transform import xy
    ox_w, oy_h = geo["orig_size"]
    sx = ox_w / geo["width"]
    sy = oy_h / geo["height"]
    col = px * sx
    row = py * sy
    x, y = xy(geo["orig_transform"], row, col)  # returns (x, y) in CRS units
    # If CRS is geographic, x=lon, y=lat. If projected, caller can reproject.
    return float(x), float(y)


def vegetation_mask_from_ndvi(ndvi, thr=0.3):
    """Boolean mask of likely tree/vegetation pixels (occlusion sources)."""
    if ndvi is None:
        return None
    return (ndvi > thr)
