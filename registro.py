import sys
from typing import Tuple

import nibabel as nib
import numpy as np
from skimage.registration import optical_flow_ilk, optical_flow_tvl1
from skimage.transform import resize, warp
from skimage.filters import threshold_otsu


def read_nii(nii_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    image = nib.load(nii_filename)
    affine = image.affine
    image = image.get_fdata()
    return (image, affine)

def register(image_1: np.ndarray, image_2: np.ndarray) -> np.ndarray:
    resized_image_2 = resize(image_2, image_1.shape)
    otsu_1 = threshold_otsu(image_1)
    otsu_2 = threshold_otsu(resized_image_2)
    binary_image_1 = image_1 >= otsu_1
    binary_image_2 = resized_image_2 >= otsu_2
    d, v, u = optical_flow_tvl1(binary_image_1, binary_image_2)
    nd, nr, nc = image_1.shape
    deep_coords, row_coords, col_coords = np.meshgrid(
        np.arange(nd), np.arange(nr), np.arange(nc), indexing="ij"
    )
    warped_image_2 = warp(
        resized_image_2,
        np.array([deep_coords + d, row_coords + v, col_coords + u]),
        cval=resized_image_2.min()
    )
    return warped_image_2


def save_nii(image: np.ndarray, affine: np.ndarray, nii_filename: str):
    img = nib.Nifti1Image(image, affine)
    nib.save(img, nii_filename)


def main():
    image_1, affine_1 = read_nii(sys.argv[1])
    image_2, affine_2 = read_nii(sys.argv[2])
    registered_image_2 = register(image_1, image_2)
    save_nii(registered_image_2, affine_1, sys.argv[3])


if __name__ == "__main__":
    main()
