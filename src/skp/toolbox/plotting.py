import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Sequence, Tuple, List


def plot_image_grid(
    image_array: np.ndarray,
    grid_size: int,
    axis: int,
    cmap: str = "gray",
    figsize: tuple = (10, 10),
) -> None:
    """
    Displays a grid of 2D image slices from a 3D or 4D NumPy array.

    The function selects evenly spaced slices along the specified axis, always
    including the first and the last slice.

    Args:
        image_array (np.ndarray): The input 3D (D, H, W) or 4D (D, H, W, C)
                                  NumPy array.
        grid_size (int): The dimension of the grid (e.g., a grid_size of 5
                         creates a 5x5 grid).
        axis (int): The axis from which to select the 2D slices (0, 1, or 2).
        cmap (str, optional): The colormap to use for displaying the images.
                              Defaults to "gray".
        figsize (tuple, optional): The size of the matplotlib figure.
                                   Defaults to (10, 10).

    Raises:
        ValueError: If the input array is not 3D or 4D, or if the axis is
                    invalid.

    Returns:
        None: The function displays a plot and does not return any value.
    """
    # --- Input Validation ---
    if not isinstance(image_array, np.ndarray) or image_array.ndim not in [3, 4]:
        raise ValueError("image_array must be a 3D or 4D NumPy array.")
    if axis not in [0, 1, 2]:
        raise ValueError("axis must be an integer: 0, 1, or 2.")
    if image_array.shape[axis] == 0:
        print("Input array is empty along the specified axis. Nothing to plot.")
        return

    # --- Slice Index Calculation ---
    num_images_to_plot = grid_size * grid_size
    total_slices = image_array.shape[axis]

    # Generate evenly spaced indices, ensuring the first and last are included.
    if num_images_to_plot > total_slices:
        # If the grid is larger than available slices, use all slices.
        indices = np.arange(total_slices)
    else:
        indices = np.linspace(0, total_slices - 1, num_images_to_plot).astype(int)

    # --- Plotting ---
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)

    # Flatten the axes array for easy iteration, handle grid_size=1 case
    axes_flat = np.array(axes).ravel()

    for i, slice_idx in enumerate(indices):
        # Create a slicer object to extract the 2D image
        slicer = [slice(None)] * image_array.ndim
        slicer[axis] = slice_idx
        img_slice = image_array[tuple(slicer)]

        # Plot the image slice
        axes_flat[i].imshow(img_slice, cmap=cmap)
        axes_flat[i].set_title(f"Slice: {slice_idx}")
        axes_flat[i].axis("off")  # Hide axes ticks and labels

    # Hide any unused subplots if the grid is larger than available slices
    for i in range(len(indices), len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    plt.show()


def plot_image_grid_side_by_side(
    image_array1: np.ndarray,
    image_array2: np.ndarray,
    grid_size: int,
    axis: int,
    cmap1: str = "gray",
    cmap2: str = "gray",
    figsize: Tuple[int, int] = (20, 10),
    titles: List[str] = ["Image 1", "Image 2"],
) -> None:
    """
    Displays a grid of side-by-side 2D slices from two 3D/4D NumPy arrays.

    The function selects evenly spaced slices along a specified axis from both
    arrays and plots them next to each other. It always includes the first and
    last slices. This is useful for comparing an image with its corresponding
    mask or another modality.

    Args:
        image_array1 (np.ndarray): The first input array. Can be 3D (D, H, W)
                                   or 4D (D, H, W, C).
        image_array2 (np.ndarray): The second input array. Must be 3D (D, H, W)
                                   and have the same spatial dimensions as
                                   image_array1.
        grid_size (int): The dimension of the grid (e.g., a grid_size of 4
                         creates a 4x4 grid of image pairs).
        axis (int): The axis from which to select the 2D slices (0, 1, or 2).
        cmap1 (str, optional): Colormap for the first image series.
                               Defaults to "gray".
        cmap2 (str, optional): Colormap for the second image series.
                               Defaults to "jet".
        figsize (tuple, optional): The total size of the matplotlib figure.
                                   Defaults to (12, 12).
        titles (List[str], optional): A list of two strings for the titles of
                                      the image columns. Defaults to
                                      ["Image 1", "Image 2"].

    Raises:
        ValueError: If inputs are invalid (not NumPy arrays, wrong dimensions,
                    mismatched shapes, or invalid axis).

    Returns:
        None: The function displays a plot and does not return any value.
    """
    # --- Input Validation ---
    if not isinstance(image_array1, np.ndarray) or image_array1.ndim not in [3, 4]:
        raise ValueError("image_array1 must be a 3D or 4D NumPy array.")
    if not isinstance(image_array2, np.ndarray) or image_array2.ndim not in [3, 4]:
        raise ValueError("image_array2 must be a 3D or 4D NumPy array.")
    if image_array1.shape[:3] != image_array2.shape[:3]:
        raise ValueError(
            "The first three dimensions (D, H, W) of both arrays must be identical."
        )
    if axis not in [0, 1, 2]:
        raise ValueError("axis must be an integer: 0, 1, or 2.")
    if image_array1.shape[axis] == 0:
        print("Input arrays are empty along the specified axis. Nothing to plot.")
        return

    # --- Slice Index Calculation ---
    num_pairs_to_plot = grid_size * grid_size
    total_slices = image_array1.shape[axis]

    # Generate evenly spaced indices, including the first and last.
    if num_pairs_to_plot >= total_slices:
        # If the grid can fit all slices, use all of them.
        indices = np.arange(total_slices)
    else:
        indices = np.linspace(0, total_slices - 1, num_pairs_to_plot).astype(int)

    # --- Plotting Setup ---
    # We need a grid that is grid_size x (grid_size * 2) to hold pairs.
    fig, axes = plt.subplots(grid_size, grid_size * 2, figsize=figsize)

    # Handle the case of grid_size=1, where axes is not a 2D array
    if grid_size == 1:
        axes = np.array([axes])

    axes_flat = axes.ravel()

    # --- Iterating and Displaying ---
    for i, slice_idx in enumerate(indices):
        # Determine the subplot indices for the current pair
        ax1_idx = i * 2
        ax2_idx = i * 2

        # --- Extract Slice from Array 1 ---
        slicer1 = [slice(None)] * image_array1.ndim
        slicer1[axis] = slice_idx
        img_slice1 = image_array1[tuple(slicer1)]

        # --- Extract Slice from Array 2 ---
        slicer2 = [slice(None)] * image_array2.ndim
        slicer2[axis] = slice_idx
        img_slice2 = image_array2[tuple(slicer2)]

        # --- Plot the pair ---
        ax1 = axes_flat[ax1_idx]
        ax2 = axes_flat[ax2_idx + 1]

        ax1.imshow(img_slice1, cmap=cmap1)
        ax1.set_title(f"{titles[0]} - Slice: {slice_idx}")
        ax1.axis("off")

        ax2.imshow(img_slice2, cmap=cmap2)
        ax2.set_title(f"{titles[1]} - Slice: {slice_idx}")
        ax2.axis("off")

    # --- Cleanup ---
    # Hide any unused subplots
    for i in range(len(indices) * 2, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout(pad=0.5)
    plt.show()


def draw_bounding_boxes(
    img: np.ndarray, bboxes: Sequence[int], mode: str = "xyxy"
) -> np.ndarray:
    assert mode in {"xyxy", "xywh"}, f"mode [{mode}] must be `xyxy` or `xywh`"
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for box in bboxes:
        if mode == "xyxy":
            x1, y1, x2, y2 = box
        elif mode == "xywh":
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
        img = cv2.rectangle(
            img, (x1, y1), (x2, y2), (255, 0, 0), int(0.005 * max(img.shape))
        )
    return img


def overlay_images(
    image: np.ndarray, overlay: np.ndarray, alpha: float = 0.7
) -> np.ndarray:
    overlaid = alpha * image + (1 - alpha) * overlay
    overlaid = overlaid.astype(np.uint8)
    return overlaid


def plot_3d_image(
    arr: np.ndarray, num_images: int, axis: int, cmap: str = "gray"
) -> None:
    for i in range(0, arr.shape[axis], arr.shape[axis] // num_images):
        if axis == 0:
            img = arr[i]
        elif axis == 1:
            img = arr[:, i]
        elif axis == 2:
            img = arr[:, :, i]
        plt.imshow(img, cmap=cmap)
        plt.show()


def plot_3d_image_side_by_side(
    arr1: np.ndarray, arr2: np.ndarray, num_images: int, axis: int, cmap: str = "gray"
) -> None:
    assert arr1.shape[:3] == arr2.shape[:3], f"{arr1.shape} does not match {arr2.shape}"
    for i in range(0, arr1.shape[axis], arr1.shape[axis] // num_images):
        if axis == 0:
            img1, img2 = arr1[i], arr2[i]
        elif axis == 1:
            img1, img2 = arr1[:, i], arr2[:, i]
        elif axis == 2:
            img1, img2 = arr1[:, :, i], arr2[:, :, i]
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap=cmap)
        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap=cmap)
        plt.show()
