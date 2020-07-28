# -*- coding: utf-8 -*-
"""Extract EV BOLD from movie."""

from pathlib import Path
import numpy as np
import pandas as pd
import nilearn.image as image
import nibabel as nib

# Directories
base_dir = Path('/mnt/c/LAQ/NMA/groupwork')
func_dir = base_dir.joinpath('func', 'movie')


sub_list = ['1', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
ev = pd.read_csv(base_dir.joinpath('ev.txt'), sep='\t')
[n,m] = ev.shape
sub_dat = np.zeros([61, 73, 61, 16])
cut_combine = np.zeros([61, 73, 61, 18])
for cut_id in range(0, 18): 
    i = -1
    for sub_id in sub_list:
        i = i+1
        input_dir = func_dir.joinpath(f'sub{sub_id}')
        bold_fid = input_dir.joinpath(f'sub{sub_id}_cut{cut_id+1}.nii.gz').as_posix()
        bold_img = image.load_img(bold_fid)
        img_affine  = bold_img.affine
        bold_dat = bold_img.get_data()
        new_dat = np.mean(bold_dat, axis=3)
        sub_dat[:,:,:,i] = new_dat

    cut_dat = np.mean(sub_dat, axis=3)
    cut_combine[:,:,:,cut_id] = cut_dat
    output_dir = func_dir.joinpath(f'cut_bold')
    output_dir.mkdir(exist_ok=True, parents=True)
    nib.Nifti1Image(cut_dat,img_affine).to_filename(output_dir.joinpath(f'cut{cut_id+1}.nii.gz').as_posix())
    nib.Nifti1Image(cut_combine,img_affine).to_filename(output_dir.joinpath('cut_combine.nii.gz').as_posix())
    
    
    
