# -*- coding: utf-8 -*-
"""corralation."""

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
rating = pd.read_csv(base_dir.joinpath('rating.txt'), sep='\t')
[a,b,c,d] = bold_dat.shape
logic_dat = np.zeros([a, b, c])
space_dat = np.zeros([a, b, c])
tool_dat = np.zeros([a, b, c])
activity_dat = np.zeros([a, b, c])

for h in range(0,a):
    for i in range(0,b):
        for j in range(0,c):
            tempdat = bold_dat[h,i,j,:]
            temp_l = rating.logic
            temp_s = rating.space
            temp_t = rating.tool
            temp_a = rating.activity
            pccs_l = pearsonr(tempdat, temp_l)
            pccs_s = pearsonr(tempdat, temp_s)
            pccs_t = pearsonr(tempdat, temp_t)
            pccs_a = pearsonr(tempdat, temp_a)
            logic_dat[h,i,j] = pccs_l[1]
            space_dat[h,i,j] = pccs_s[1]
            tool_dat[h,i,j] = pccs_t[1]
            activity_dat[h,i,j] = pccs_a[1]

nib.Nifti1Image(logic_dat,img_affine).to_filename(base_dir.joinpath('logic_dat.nii.gz').as_posix())
nib.Nifti1Image(space_dat,img_affine).to_filename(base_dir.joinpath('space_dat.nii.gz').as_posix())
nib.Nifti1Image(tool_dat,img_affine).to_filename(base_dir.joinpath('tool_dat.nii.gz').as_posix())
nib.Nifti1Image(activity_dat,img_affine).to_filename(base_dir.joinpath('activity_dat.nii.gz').as_posix())


