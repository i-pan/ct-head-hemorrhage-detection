import numpy as np

from scipy.ndimage import center_of_mass
from typing import Tuple, Optional


def crop_3d_with_center(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    crop_size: Tuple[int, int, int],
    crop_mode: str,
    object_center: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops a 3D/4D image and its 3D mask with specific constraints.

    If crop_mode is "foreground", this function ensures the provided or calculated
    object center is located within the central 75% of the cropped ROI. It will
    dynamically pad the array if the object is near an edge to satisfy this rule.

    Args:
        image_array (np.ndarray): The input 3D (D, H, W) or 4D (D, H, W, C)
                                  NumPy array.
        mask_array (np.ndarray): The 3D (D, H, W) NumPy array with a multi-class
                                 mask (0 is background).
        crop_size (Tuple[int, int, int]): The desired crop dimensions (d, h, w).
        crop_mode (str): The cropping strategy, "foreground", "background", or "random".
        object_center (Optional[Tuple[int, int, int]], optional):
            The (z, y, x) coordinates of the object's center.
            If None and crop_mode is "foreground", the center of mass of the
            entire foreground will be calculated and used. Defaults to None.

    Raises:
        ValueError: For invalid inputs such as incorrect dimensions, mismatched
                    shapes, or invalid crop_mode.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the cropped image and
                                       the cropped mask, both of size crop_size.
    """
    # --- 1. Input Validation ---
    if not isinstance(image_array, np.ndarray) or image_array.ndim not in [3, 4]:
        raise ValueError("image_array must be a 3D or 4D NumPy array.")
    if not isinstance(mask_array, np.ndarray) or mask_array.ndim != 3:
        raise ValueError("mask_array must be a 3D NumPy array.")
    if image_array.shape[:3] != mask_array.shape:
        raise ValueError(
            "Image and mask must have the same spatial dimensions (D, H, W)."
        )
    if crop_mode not in ["foreground", "background", "random"]:
        raise ValueError("crop_mode must be 'foreground', 'background', or 'random'.")

    # --- 2. Cropping Logic ---
    is_mask_empty = np.all(mask_array == 0)

    if crop_mode == "foreground" and not is_mask_empty:
        # If no center is provided, calculate the center of mass of the foreground
        if object_center is None:
            # Note: center_of_mass returns in (z, y, x) order, matching NumPy indexing
            object_center = center_of_mass(mask_array)

        object_center = tuple(int(round(c)) for c in object_center)

        return _get_foreground_crop(image_array, mask_array, crop_size, object_center)

    elif crop_mode == "background" and not is_mask_empty:
        # Attempt to find a crop with no foreground pixels
        return _get_background_crop(image_array, mask_array, crop_size)

    else:
        # Fallback for "background" with an empty mask, or "foreground" with an
        # empty mask, or any other unspecified case.
        return _get_random_crop(image_array, mask_array, crop_size)


def _get_foreground_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: Tuple[int, int, int],
    center: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates a crop centered around the object, adding padding if needed.
    """
    # Define the valid range for the object's position within the crop (central 90%)
    # This means the offset from the crop's start to the object center can be
    # between 5% and 95% of the crop dimension.
    lower_bound = 0.05
    upper_bound = 0.95

    start_coords = [0, 0, 0]
    for i in range(3):
        # Randomly choose where the center should be within the crop's central 75%
        offset_in_crop = np.random.randint(
            int(crop_size[i] * lower_bound), int(crop_size[i] * upper_bound) + 1
        )
        # Calculate the ideal start coordinate for the crop
        start_coords[i] = center[i] - offset_in_crop

    # --- Determine and apply necessary padding ---
    padding = [[0, 0] for _ in range(image.ndim)]
    adjusted_start_coords = [0, 0, 0]

    for i in range(3):
        # Padding needed at the beginning of the axis
        pad_before = max(0, -start_coords[i])
        # Padding needed at the end of the axis
        pad_after = max(0, (start_coords[i] + crop_size[i]) - image.shape[i])

        padding[i] = [pad_before, pad_after]
        # The new start coordinate is the old one plus the padding we added
        adjusted_start_coords[i] = start_coords[i] + pad_before

    padded_image = np.pad(image, padding, mode="constant", constant_values=0)
    padded_mask = np.pad(mask, padding[:3], mode="constant", constant_values=0)

    return _crop_at(padded_image, padded_mask, crop_size, tuple(adjusted_start_coords))


def _get_background_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: Tuple[int, int, int],
    max_attempts: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tries to find a crop containing only background."""
    # First, pad the image in case it's smaller than crop_size
    padded_image, padded_mask, _ = _pad_if_needed(image, mask, crop_size)

    for _ in range(max_attempts):
        cropped_image, cropped_mask = _get_random_crop(
            padded_image, padded_mask, crop_size
        )
        if np.all(cropped_mask == 0):
            return cropped_image, cropped_mask
    # If no pure background crop is found after max_attempts, return a random one.
    return _get_random_crop(padded_image, padded_mask, crop_size)


def _get_random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts a random crop, padding first if necessary."""
    padded_image, padded_mask, _ = _pad_if_needed(image, mask, crop_size)
    img_dims = padded_image.shape

    d_start = np.random.randint(0, img_dims[0] - crop_size[0] + 1)
    h_start = np.random.randint(0, img_dims[1] - crop_size[1] + 1)
    w_start = np.random.randint(0, img_dims[2] - crop_size[2] + 1)

    return _crop_at(padded_image, padded_mask, crop_size, (d_start, h_start, w_start))


def _pad_if_needed(
    image: np.ndarray, mask: np.ndarray, crop_size: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Pads arrays if they are smaller than the crop size."""
    padding = [[0, 0] for _ in range(image.ndim)]
    needs_padding = False
    for i in range(3):
        if image.shape[i] < crop_size[i]:
            padding[i][1] = crop_size[i] - image.shape[i]
            needs_padding = True

    if not needs_padding:
        return image, mask, padding

    padded_image = np.pad(image, padding, mode="constant", constant_values=0)
    padded_mask = np.pad(mask, padding[:3], mode="constant", constant_values=0)
    return padded_image, padded_mask, padding


def _crop_at(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: Tuple[int, int, int],
    start_coords: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to perform the crop at specific coordinates."""
    d, h, w = crop_size
    z, y, x = start_coords

    cropped_image = image[z : z + d, y : y + h, x : x + w]
    cropped_mask = mask[z : z + d, y : y + h, x : x + w]

    return cropped_image, cropped_mask


def create_overlapping_chunks(
    volume: np.ndarray,
    crop_size: Tuple[int, int, int],
) -> np.ndarray:
    """
    Splits a 3D/4D volume into a stack of overlapping chunks of a specific size.

    The function first pads the volume to ensure that it can be evenly divided
    by the sliding window operation. It then extracts chunks of `crop_size`
    with a 50% overlap (stride = crop_size / 2).

    Args:
        volume (np.ndarray): The input 3D (D, H, W) or 4D (D, H, W, C)
                             NumPy array.
        crop_size (Tuple[int, int, int]): The dimensions (d, h, w) of each chunk.

    Raises:
        ValueError: If the input volume is not 3D or 4D, or if crop_size
                    is not a tuple of 3 positive integers.

    Returns:
        np.ndarray: A single NumPy array containing all the extracted chunks,
                    stacked on a new first axis. The output shape will be
                    (N, d, h, w) or (N, d, h, w, C), where N is the total
                    number of chunks.
    """
    # --- 1. Input Validation ---
    if not isinstance(volume, np.ndarray) or volume.ndim not in [3, 4]:
        raise ValueError("Input volume must be a 3D or 4D NumPy array.")
    if (
        not isinstance(crop_size, tuple)
        or len(crop_size) != 3
        or not all(isinstance(d, int) and d > 0 for d in crop_size)
    ):
        raise ValueError("crop_size must be a tuple of 3 positive integers.")

    # --- 2. Calculate Required Padding ---
    strides = [dim // 2 for dim in crop_size]
    padding_widths = [[0, 0] for _ in range(volume.ndim)]

    for i in range(3):  # Iterate over D, H, W dimensions
        original_dim = volume.shape[i]
        crop_dim = crop_size[i]
        stride = strides[i]

        if original_dim <= crop_dim:
            # If the volume is smaller than the crop, pad it up to the crop size
            padding_needed = crop_dim - original_dim
        else:
            # Calculate how many steps are needed to cover the dimension
            # The uncovered part is (original_dim - crop_dim)
            n_steps = (original_dim - crop_dim + stride - 1) // stride

            # The total size required is the start of the last chunk plus the chunk size
            required_size = (n_steps * stride) + crop_dim
            padding_needed = required_size - original_dim

        # We add all padding to the end of the dimension
        padding_widths[i][1] = padding_needed

    padded_volume = np.pad(volume, padding_widths, mode="constant", constant_values=0)

    # --- 3. Generate and Collect Chunks ---
    all_chunks = []
    padded_dims = padded_volume.shape
    d_crop, h_crop, w_crop = crop_size
    d_stride, h_stride, w_stride = strides

    for z in range(0, padded_dims[0] - d_crop + 1, d_stride):
        for y in range(0, padded_dims[1] - h_crop + 1, h_stride):
            for x in range(0, padded_dims[2] - w_crop + 1, w_stride):
                chunk = padded_volume[z : z + d_crop, y : y + h_crop, x : x + w_crop]
                all_chunks.append(chunk)

    if not all_chunks:
        # Handle edge case where volume is empty
        empty_shape = (
            (0,) + crop_size
            if volume.ndim == 3
            else (0,) + crop_size + (volume.shape[3],)
        )
        return np.empty(empty_shape, dtype=volume.dtype)

    return np.stack(all_chunks, axis=0)
