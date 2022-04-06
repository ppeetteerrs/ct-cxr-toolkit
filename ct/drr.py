from itertools import product

import numpy as np
import SimpleITK as sitk
from deepdrr import MobileCArm, Volume
from deepdrr.geo import FrameTransform
from deepdrr.projector import Projector


def get_drr(img: sitk.Image, move_by: int = 750) -> np.ndarray:
    # sitk.WriteImage(img, "hi.nii.gz")
    # Extract pixel and material data from sitk
    hu_values = np.transpose(sitk.GetArrayFromImage(img), [2, 1, 0]).astype(float)
    data = Volume._convert_hounsfield_to_density(hu_values)
    materials = Volume.segment_materials(
        hu_values, use_thresholding=True, use_cached=False
    )

    # Set up nifti affine
    affine = np.zeros((4, 4))
    origin = img.GetOrigin()
    direction = np.array(img.GetDirection()).reshape((3, 3))
    for (i, j) in product(range(3), repeat=2):
        if direction[i][j] != 0:
            affine[i][j] = direction[i][j]
    affine[:3, 3] = origin
    affine[:2, :] *= np.sign(origin[2]) + int(origin[2] == 0)
    affine[3, 3] = 1.0
    anatomical_from_ijk = FrameTransform(affine)
    volume = Volume(data, materials, anatomical_from_ijk, None, "RAS")
    volume.facedown()
    carm = MobileCArm()
    carm.reposition(volume.center_in_world)
    carm.move_to(alpha=0, beta=-180, degrees=True)
    carm.move_by((0, 0, move_by))
    with Projector(volume, carm=carm) as projector:
        drr = projector()
    return (drr * 255).astype(np.uint8)
