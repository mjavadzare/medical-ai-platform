import numpy as np
import pandas as pd
import cv2

def get_mean_std_rgb(df: pd.DataFrame):
    """
    Calculate per-channel mean and standard deviation of RGB images.

    ```
    The images are read using OpenCV, converted from BGR to RGB,
    and normalized from the range [0, 255] to [0, 1] before
    calculating the statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image file paths.
        The DataFrame must contain a ``file_path`` column.

    Returns
    -------
    mean : numpy.ndarray
        Mean pixel value for the R, G, and B channels.
        Values are in the range [0, 1].

    std : numpy.ndarray
        Standard deviation of pixel values for the R, G, and B
        channels. Values are in the range [0, 1].

    Raises
    ------
    ValueError
        If an image cannot be read from the specified file path.

    Notes
    -----
    The statistics are calculated across all pixels from all
    images in the provided DataFrame.

    For model training, these statistics should generally be
    calculated using only the training dataset to avoid data leakage.
    """

    sum_rgb = np.zeros(3)
    sum_sq_rgb = np.zeros(3)
    pixel_count = 0

    for file in df["file_path"]:

        img = cv2.imread(str(file))

        if img is None:
            raise ValueError(f"Cannot read image: {file}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        sum_rgb += img.sum(axis=(0, 1))
        sum_sq_rgb += (img ** 2).sum(axis=(0, 1))

        height, width, _ = img.shape
        pixel_count += height * width

    mean = sum_rgb / pixel_count

    std = np.sqrt(
        (sum_sq_rgb / pixel_count) - (mean ** 2)
    )

    return mean, std
