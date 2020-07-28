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

for sub_id in sub_list:
    bold_fid = func_dir.joinpath(f'sherlock_movie_s{sub_id}.nii').as_posix()
    bold_img = image.load_img(bold_fid)
    img_affine  = bold_img.affine
    bold_dat = bold_img.get_data()
    output_dir = func_dir.joinpath(f'sub{sub_id}')
    output_dir.mkdir(exist_ok=True, parents=True)
    for cut_id in range(0, n):
        start_tr = ev.cut_start[cut_id] - 1
        end_tr = ev.cut_end[cut_id]
        cut_dat = bold_dat[:,:,:,start_tr:end_tr]
        nib.Nifti1Image(cut_dat,img_affine).to_filename(output_dir.joinpath(f'sub{sub_id}_cut{cut_id+1}.nii.gz').as_posix())


    