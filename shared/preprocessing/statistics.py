import numpy as np
import pandas as pd
import cv2


def get_mean_std_rgb(df: pd.DataFrame):

    sum_rgb = np.zeros(3)
    sum_sq_rgb = np.zeros(3)
    pixel_count = 0

    for file in df["file_path"]:

        img = cv2.imread(str(file))

        if img is None:
            raise ValueError(f"Cannot read image: {file}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0,1]
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