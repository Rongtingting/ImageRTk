"""
Mesmer segmentation for MIBI data.
"""

import json
import time
import logging
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path

# mesmer imports
from deepcell.applications import Mesmer
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay


import skimage.measure
from skimage.exposure import rescale_intensity

import matplotlib.pyplot as plt

# from ..utils.pyqupath.geojson import mask_to_geojson_joblib


from pathlib import Path
from typing import Optional, Sequence, Dict
import tifffile



def segmentation_mibi_mesmer(input_dir: Path, 
                             output_dir: Path,
                             nuclear_markers: list = ["Histone H3", "dsDNA"],
                             membrane_markers: list = ["CD45", "Vimentin-works", "Pan-Keratin", "CD163"],
                             pixel_size_um=0.78,    
                             maxima_threshold=0.075,
                             interior_threshold=0.20,
                             scale: bool = True,
                             tag: str = None,
                             num_threads: int = 8
                             ):
    """
    Perform cell segmentation on MIBI data using Mesmer model.
    Args:   input_array (np.ndarray): 4D numpy array of shape (1, H, W, C) representing the multiplexed image data.
            pixel_size_um (float): Pixel size in micrometers. Default is 0.78 in EC Pembro.
            maxima_threshold (float): Maxima threshold for post-processing. Default is 0.075.
            interior_threshold (float): Interior threshold for post-processing. Default is 0.20.
    Returns:    np.ndarray: 3D numpy array of shape (H, W, 1) representing the segmentation mask.
    example:
        from pathlib import Path
        input_dir = Path("/path/to/mibi/image_directory")
        output_dir = Path("/path/to/save/segmentation_results")
        segmentation_mibi_mesmer(input_dir, output_dir)


    # marker_names_all = ["Histone H3", "dsDNA", "CD45", "Vimentin-works", "Pan-Keratin", "CD163"]   
    """
    # Path handling
    input_dir, output_dir, segmentation_dir = prepare_seg_paths(input_dir, output_dir, tag)
    # Set up logging
    # setup_logging(segmentation_dir / "segmentation.log")
    # prepare marker array
    marker_array = process_mesmer_segmentation_markers(input_dir=input_dir,
                                                       nuclear_markers=nuclear_markers,
                                                       membrane_markers=membrane_markers,
                                                       combine_method="sum",
                                                       scale=scale,
                                                       scale_type="none")
    # setup_gpu("1")
    # run mesmer segmentation
    mesmer = Mesmer()
    predictions = mesmer.predict(marker_array, image_mpp = pixel_size_um,
                                  postprocess_kwargs_whole_cell={"maxima_threshold" : maxima_threshold,
                                                                 "interior_threshold" : interior_threshold})
    

    # create outline overlay
    rgb_image, overlay = view_mesmer_segmentation_results(marker_array, predictions, plotshow=True)

    # predictions = predictions[0, :, :, 0]  # HW1
    # # save segmentation mask
    # segmentation_mask_f = segmentation_dir / "segmentation_mask.tiff"
    # tifffile.imwrite(str(segmentation_mask_f), predictions.astype(np.uint16))
    
    # rgb_image = create_rgb_image(marker_array, channel_colors = ["green", "blue"])
    # overlay = make_outline_overlay(rgb_data = rgb_image, predictions = predictions)
    segmentation_mask = predictions[0,...,0]
    segmentation_mask_file = segmentation_dir / "segmentation_mask.tiff"
    tifffile.imwrite(str(segmentation_mask_file), segmentation_mask)
    # logging.info("Segmentation completed.")

    # # save geojson for visualization in QuPath
    # mask_to_geojson_joblib(segmentation_mask, segmentation_mask_file, n_jobs=8)

    
    
    # Extract single-cell features
    # data, data_scale = extract_cell_features(marker_dict, segmentation_mask)
    marker_dict = get_marker_dict(
        unit_dir=input_dir,
        include=nuclear_markers + membrane_markers,
        exclude=None,
        recursive=False,
    )
    data, data_scale = SingleCellExraction(method="intensity",
                                           marker_dict=marker_dict,
                                           segmentation_mask=segmentation_mask)
    data.to_csv(segmentation_dir / "data.csv")
    data_scale.to_csv(segmentation_dir / "dataScaleSize.csv")
    # logging.info("Single-cell features extracted.")

    # Write parameters
    params = {
        "nuclear_markers": nuclear_markers,
        "membrane_markers": membrane_markers,
        "scale": scale,
        "pixel_size_um": pixel_size_um,
        "maxima_threshold": maxima_threshold,
        "interior_threshold": interior_threshold,
        "compartment": "whole-cell",
    }
    with open(
        f"{segmentation_dir}/parameter_segmentation.json", "w", encoding="utf-8"
    ) as file:
        json.dump(params, file, indent=4, ensure_ascii=False)
    
    return predictions



def get_marker_dict(
    unit_dir: Path,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    recursive: bool = False,
) -> Dict[str, "np.ndarray"]:
    """
    Load all marker TIFF files from a folder into a dictionary.

    Parameters
    ----------
    unit_dir : Path
        Directory that contains marker TIFF files.
    include : list[str], optional
        Marker names to include (only these will be loaded).
    exclude : list[str], optional
        Marker names to exclude.
    recursive : bool, default=False
        Whether to search subfolders recursively.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping marker name → image array.
    """

    unit_dir = Path(unit_dir)
    if not unit_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {unit_dir}")
    if not unit_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {unit_dir}")

    # logical constraints
    if include and exclude:
        raise ValueError("You can use either 'include' or 'exclude', not both.")
    
    include = set(include) if include else None
    exclude = set(exclude) if exclude else None


    # Glob pattern
    pattern = "**/*.tif*" if recursive else "*.tif*"
    marker_dict = {}

    for path in unit_dir.glob(pattern):
        if not path.is_file():
            continue
        marker_name = path.stem  # use original case

        # apply filters
        if include is not None:
            if marker_name not in include:
                continue
        elif exclude is not None:
            if marker_name in exclude:
                continue
        # (neither include nor exclude → load everything)

        try:
            img = tifffile.imread(path)
        except Exception as e:
            print(f"[WARNING] Failed to read {path.name}: {e}")
            continue

        marker_dict[marker_name] = img

    print(f"[INFO] Loaded {len(marker_dict)} markers from {unit_dir}")
    return marker_dict



def prepare_seg_paths(input_dir, output_dir, tag=None):
    """Validate and prepare input/output directory structure."""

    # Normalize input_dir
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if input_dir.is_file():
        print(f"[WARNING] '{input_dir}' is a file; using its parent folder instead.")
        input_dir = input_dir.parent
    elif not input_dir.is_dir():
        raise NotADirectoryError(f"Invalid directory path: {input_dir}")

    # Prepare output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tag handling
    if tag:
        # Clean up unsafe characters (optional)
        safe_tag = "".join(c for c in str(tag) if c.isalnum() or c in "-_")
        if safe_tag != tag:
            print(f"[INFO] Tag sanitized: '{tag}' → '{safe_tag}'")
        tag = safe_tag
    else:
        tag = time.strftime("%Y%m%d_%H%M%S")

    # Make subfolder for segmentation results
    segmentation_dir = output_dir / tag
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, output_dir, segmentation_dir


def process_mesmer_segmentation_markers(
    input_dir: Path,
    nuclear_markers: list = ("Histone H3", "dsDNA"),
    membrane_markers: list = ("CD45", "Vimentin-works", "Pan-Keratin", "CD163"),
    combine_method: str = "mean",   
    scale: bool = False,
    scale_type: str = "float",
):
    """
    Reads marker TIFFs, normalizes and aggregates nuclear/membrane markers,
    and returns a 4D NumPy array with shape (1, H, W, 2).

    Args:
        input_dir (Path or str): Folder containing marker TIFF files.
        nuclear_markers (sequence): Marker names for nuclear channel.
        membrane_markers (sequence): Marker names for membrane channel.
        combine_method (str): "mean", "sum", or "max".
        normalize (str):
            - "float": per-image normalization (0–1, float32)
            - "uint8": per-image normalization (0–255, uint8)
            - "none":  keep raw values

    Returns:
        np.ndarray: 4D array, shape (1, H, W, 2)
    """
    # Path handling
    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        print(f"[WARNING] '{input_dir}' is not a directory; using parent folder.")
        input_dir = input_dir.parent

    # # Normalization helper
    # def normalize_img(img):
    #     img = img.astype(np.float32)
    #     min_val, max_val = np.min(img), np.max(img)
    #     if max_val > min_val:
    #         img = (img - min_val) / (max_val - min_val)
    #     if normalize == "float":
    #         return img.astype(np.float32)
    #     elif normalize == "uint8":
    #         return (img * 255).astype(np.uint8)
    #     elif normalize == "none":
    #         return img
    #     else:
    #         raise ValueError(f"Unknown normalization mode: {normalize}")
    
    def scale_img(img, scale_type = "none"):
        if scale_type == "float":
            img = rescale_intensity(img, out_range=(0, 1)).astype(np.float32)
        elif scale_type == "uint8":
            img = rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
        elif scale_type == "none":
            return img
        else:
            raise ValueError(f"Unknown scale_type: {scale_type}")
        

    # Load one marker TIFF
    def load_marker(marker_name):
        matches = list(input_dir.glob(f"{marker_name}.tif")) + list(input_dir.glob(f"{marker_name}.tiff"))
        if not matches:
            print(f"[WARNING] Marker not found: {marker_name}")
            return None
        img = tifffile.imread(matches[0])
        if scale:
            img = scale_img(img, scale_type)
        return img

    # Combine multiple images for nuclear/membrane channels for using in mesmer
    def combine_images(image_list, combine_method="sum"):
        if len(image_list) == 0:
            raise ValueError("No images provided for combination")
        stack = np.stack(image_list, axis=0)
        if combine_method == "mean":
            return np.mean(stack, axis=0)
        elif combine_method == "sum":
            return np.sum(stack, axis=0)
        elif combine_method == "max":
            return np.max(stack, axis=0)
        else:
            raise ValueError(f"Invalid combine_method '{combine_method}'")

    # Load and combine nuclear markers
    nuclear_imgs = [img for m in nuclear_markers if (img := load_marker(m)) is not None]
    if not nuclear_imgs:
        raise FileNotFoundError("No nuclear marker images loaded.")
    nuclear_combined = combine_images(nuclear_imgs)

    # Load and combine membrane markers
    membrane_imgs = [img for m in membrane_markers if (img := load_marker(m)) is not None]
    if not membrane_imgs:
        raise FileNotFoundError("No membrane marker images loaded.")
    membrane_combined = combine_images(membrane_imgs)

    # Combine into final array (1, H, W, 2)
    combined_array = np.stack([nuclear_combined, membrane_combined], axis=-1)
    print(f"Combined nuclear and membrane channels with shapes: "
          f"{nuclear_combined.shape}, {membrane_combined.shape}")
    # combined_array = np.expand_dims(combined_array, 0)  # 1HWC
    combined_array = combined_array[np.newaxis, ...]
    print(f"Final combined array shape (1, H, W, 2): {combined_array.shape}")
    # check if nan values exist
    if np.isnan(combined_array).any():
        # raise ValueError("NaN values found in the combined array.")
        print("[WARNING] NaN values found in the combined array.")
        # process nan values
        # combined_array = np.nan_to_num(combined_array)

    print(
        f"Created array shape={combined_array.shape}, dtype={combined_array.dtype}, "
        f"method={combine_method}, scale={scale_type}"
    )
    return combined_array


def view_mesmer_segmentation_results(
    marker_array: np.ndarray,
    segmentation_mask: np.ndarray,
    plotshow: bool = True
):
    """
    Visualize Mesmer segmentation results.

    Parameters
    ----------
    input_array : np.ndarray
        4D numpy array of shape (1, H, W, C) representing the multiplexed image data.
    segmentation_mask : np.ndarray
        2D numpy array of shape (H, W) representing the segmentation mask.

    Returns
    -------
    np.ndarray
        RGB image with segmentation outlines overlaid on the input image.
    """
    rgb_image = create_rgb_image(marker_array, channel_colors=["green", "blue"])
    overlay = make_outline_overlay(rgb_data=rgb_image, predictions=segmentation_mask)
    if plotshow:
        idx = 0

        # plot the data
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(rgb_image[idx, ...])
        ax[1].imshow(overlay[idx, ...])

        ax[0].set_title('Raw data')
        ax[1].set_title('Predictions')

        for a in ax:
            a.axis('off')

        plt.show()
        # fig.savefig('mesmer-wc.png')
    return rgb_image, overlay


def extract_cell_features(
    marker_dict: dict[str, np.ndarray],
    segmentation_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract single cell features from segmentation mask.

    Parameters
    ----------
    marker_dict : dict
        Dictionary containing marker names as keys and corresponding images as
        values.
    segmentation_mask : np.ndarray
        A 2D segmentation mask with the same shape as the marker images, in
        which each cell is labeled with a unique integer.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Dataframe containing single-cell features.
        - Dataframe containing single-cell features with marker intensities
          scaled by cell size.
    """
    marker_name = [marker for marker in marker_dict.keys()]
    marker_array = np.stack([marker_dict[marker] for marker in marker_name], axis=2)

    # extract properties
    props = skimage.measure.regionprops_table(
        segmentation_mask,
        properties=["label", "area", "centroid"],
    )
    props_df = pd.DataFrame(props)
    props_df.columns = ["cellLabel", "cellSize", "Y_cent", "X_cent"]

    # extract marker intensity
    stats = skimage.measure.regionprops(segmentation_mask)
    n_cell = len(stats)
    n_marker = len(marker_name)
    sums = np.zeros((n_cell, n_marker))
    avgs = np.zeros((n_cell, n_marker))
    for i, region in enumerate(stats):
        # Extract the pixel values for the current region from the marker_array
        label_counts = [marker_array[coord[0], coord[1], :] for coord in region.coords]
        sums[i] = np.sum(label_counts, axis=0)  # Sum of marker intensities
        avgs[i] = sums[i] / region.area  # Average intensity per unit area

    sums_df = pd.DataFrame(sums, columns=marker_name)
    avgs_df = pd.DataFrame(avgs, columns=marker_name)
    data = pd.concat([props_df, sums_df], axis=1)
    data_scale_size = pd.concat([props_df, avgs_df], axis=1)
    return data, data_scale_size


def SingleCellExraction(method: str = "intensity",
                        marker_dict: dict[str, np.ndarray] = None,
                        segmentation_mask: np.ndarray = None):
    """
    Extract single cell features from segmentation mask."""
    if method == "intensity":
        data, data_scale = extract_cell_features(marker_dict, segmentation_mask)
    elif method == "nimbus":
        pass
        # nimbus = Nimbus(
        # fov_paths=fov_paths,
        # segmentation_naming_convention=segmentation_naming_convention,
        # output_dir=nimbus_output_dir,
        # exclude_channels=exclude_channels,
        # save_predictions=True,
        # batch_size=4,
        # test_time_aug=False,
        # input_shape=[1024,1024])

        # # check if all inputs are valid
        # nimbus.check_inputs()
    return data, data_scale