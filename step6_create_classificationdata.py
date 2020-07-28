# -*- coding: utf-8 -*-
"""Extract EV BOLD from movie."""

from pathlib import Path
import numpy as np
import pandas as pd
import nilearn.image as image
import nibabel as nib
from scipy import stats
from scipy.stats import pearsonr

# Directories
base_dir = Path('/mnt/c/LAQ/NMA/groupwork')
func_dir = base_dir.joinpath('func', 'movie')

bold_fid = func_dir.joinpath(f'cut_bold', 'cut_combine.nii.gz').as_posix()
bold_img = image.load_img(bold_fid)
img_affine  = bold_img.affine
bold_dat = bold_img.get_data()

l1 = bold_dat[:,:,:,[0, 3, 9, 10, 12, 17]]
l2 = bold_dat[:,:,:,[2, 4, 6, 11, 13, 15]]
l3 = bold_dat[:,:,:,[1, 5, 7, 8, 14, 16]]

s1 = bold_dat[:,:,:,[1, 2, 3, 4, 11, 15]]
s2 = bold_dat[:,:,:,[6, 7, 8, 12, 13, 17]]
s3 = bold_dat[:,:,:,[0, 5, 9, 10, 14, 16]]

t1 = bold_dat[:,:,:,[4, 9, 10, 11, 12, 14]]
t2 = bold_dat[:,:,:,[0, 3, 5, 6, 13, 15]]
t3 = bold_dat[:,:,:,[1, 2, 7, 8, 16, 17]]

a1 = bold_dat[:,:,:,[6, 8, 10, 11, 16, 17]]
a2 = bold_dat[:,:,:,[0, 3, 7, 12, 14, 15]]
a3 = bold_dat[:,:,:,[1, 2, 4, 5, 9, 13]]

nib.Nifti1Image(l1,img_affine).to_filename(base_dir.joinpath('classification_data', 'logic', 'logic_1.nii.gz').as_posix())
nib.Nifti1Image(l2,img_affine).to_filename(base_dir.joinpath('classification_data', 'logic', 'logic_2.nii.gz').as_posix())
nib.Nifti1Image(l3,img_affine).to_filename(base_dir.joinpath('classification_data', 'logic', 'logic_3.nii.gz').as_posix())

nib.Nifti1Image(s1,img_affine).to_filename(base_dir.joinpath('classification_data', 'space', 'space_1.nii.gz').as_posix())
nib.Nifti1Image(s2,img_affine).to_filename(base_dir.joinpath('classification_data', 'space', 'space_2.nii.gz').as_posix())
nib.Nifti1Image(s3,img_affine).to_filename(base_dir.joinpath('classification_data', 'space', 'space_3.nii.gz').as_posix())

nib.Nifti1Image(t1,img_affine).to_filename(base_dir.joinpath('classification_data', 'tool', 'tool_1.nii.gz').as_posix())
nib.Nifti1Image(t2,img_affine).to_filename(base_dir.joinpath('classification_data', 'tool', 'tool_2.nii.gz').as_posix())
nib.Nifti1Image(t3,img_affine).to_filename(base_dir.joinpath('classification_data', 'tool', 'tool_3.nii.gz').as_posix())


nib.Nifti1Image(a1,img_affine).to_filename(base_dir.joinpath('classification_data', 'activity', 'activity_1.nii.gz').as_posix())
nib.Nifti1Image(a2,img_affine).to_filename(base_dir.joinpath('classification_data', 'activity', 'activity_2.nii.gz').as_posix())
nib.Nifti1Image(a3,img_affine).to_filename(base_dir.joinpath('classification_data', 'activity', 'activity_3.nii.gz').as_posix())