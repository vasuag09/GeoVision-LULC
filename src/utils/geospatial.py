import rasterio
from rasterio.transform import from_origin
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def export_geotiff(
    data: np.ndarray,
    output_path: str,
    transform: rasterio.Affine,
    crs: str = "EPSG:32631", # Example standard
    cmap: np.ndarray = None
):
    """
    Exports a 2D numpy array (predictions) to a GeoTIFF.
    Args:
        data: 2D numpy array (H, W) of class indices.
        output_path: Path to write the .tif file.
        transform: Rasterio affine transform representing geospatial extent.
        crs: Coordinate reference system.
        cmap: Optional colormap for QGIS. Array of shape (num_classes, 3).
    """
    height, width = data.shape
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)
        
        if cmap is not None:
             # rasterio expects color dict {class_idx: (R, G, B, 255)}
             color_dict = {i: tuple(list(c) + [255]) for i, c in enumerate(cmap)}
             dst.write_colormap(1, color_dict)


def overlay_predictions(image: np.ndarray, prediction: np.ndarray, alpha: float = 0.5, colors: List[Tuple]=None):
    """
    Overlays categorical predictions onto an RGB image.
    Args:
        image: (H, W, 3) uint8 or float.
        prediction: (H, W) int.
        alpha: Blending factor.
        colors: List of RGB tuples for each class.
    """
    if colors is None:
        # Default categorical colors
        colors = plt.cm.get_cmap("tab10").colors[:prediction.max()+1]
        colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
    
    color_mask = np.zeros_like(image, dtype=np.float32)
    for c_idx, color in enumerate(colors):
        color_mask[prediction == c_idx] = color
        
    blended = (1 - alpha) * image.astype(np.float32) + alpha * color_mask
    return np.clip(blended, 0, 255).astype(np.uint8)

def reconstruct_tile_from_patches_geo(patches, positions, out_shape, transform):
    """
    Given an array of patches and their (x, y) start indices, reconstruct the full raster.
    Used for inferencing over huge geo-tiles.
    """
    reconstructed = np.zeros(out_shape, dtype=patches[0].dtype)
    counts = np.zeros(out_shape, dtype=np.float32)
    
    H_p, W_p = patches[0].shape[-2:]
    
    for patch, (y, x) in zip(patches, positions):
        reconstructed[y:y+H_p, x:x+W_p] += patch
        counts[y:y+H_p, x:x+W_p] += 1
        
    counts[counts == 0] = 1
    reconstructed = reconstructed / counts
    return np.round(reconstructed).astype(patches[0].dtype)
