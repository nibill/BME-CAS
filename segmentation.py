import numpy as np
from scipy import ndimage
import queue


def region_grow(image, seed_point):
    """
    Performs a region growing on the image from seed_point
    :param image: An 3D grayscale input image
    :param seed_point: The seed point for the algorithm
    :return: A 3D binary segmentation mask with the same dimensions as image
    """
    segmentation_mask = np.zeros(image.shape, np.bool)
    checked = np.zeros_like(segmentation_mask);

    segmentation_mask[seed_point] = True
    checked[seed_point] = True
    needs_check = get_nbd(seed_point, checked, image.shape)

    t = 5

    while len(needs_check) > 0:
        pt = needs_check.pop()

        if checked[pt]: continue

        checked[pt] = True

        imin = max(pt[0] - t, 0)
        imax = min(pt[0] + t, image.shape[0] - 1)
        jmin = max(pt[1] - t, 0)
        jmax = min(pt[1] + t, image.shape[1] - 1)
        kmin = max(pt[2] - t, 0)
        kmax = min(pt[2] + t, image.shape[2] - 1)

        if image[pt] >= image[imin:imax + 1, jmin:jmax + 1, kmin:kmax + 1].mean():
            segmentation_mask[pt] = True
            needs_check += get_nbd(pt, checked, image.shape)

    return segmentation_mask


def get_nbd(pt, checked, dims):
    nbhd = []

    if (pt[0] > 0) and not checked[pt[0] - 1, pt[1], pt[2]]:
        nbhd.append((pt[0] - 1, pt[1], pt[2]))
    if (pt[1] > 0) and not checked[pt[0], pt[1] - 1, pt[2]]:
        nbhd.append((pt[0], pt[1] - 1, pt[2]))
    if (pt[2] > 0) and not checked[pt[0], pt[1], pt[2] - 1]:
        nbhd.append((pt[0], pt[1], pt[2] - 1))

    if (pt[0] < dims[0] - 1) and not checked[pt[0] + 1, pt[1], pt[2]]:
        nbhd.append((pt[0] + 1, pt[1], pt[2]))
    if (pt[1] < dims[1] - 1) and not checked[pt[0], pt[1] + 1, pt[2]]:
        nbhd.append((pt[0], pt[1] + 1, pt[2]))
    if (pt[2] < dims[2] - 1) and not checked[pt[0], pt[1], pt[2] + 1]:
        nbhd.append((pt[0], pt[1], pt[2] + 1))

    return nbhd
