# -*- coding: utf-8 -*-
"""calculate the correlation within ev."""

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

sub_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
index_sub = -1
p = np.zeros([16,18])
mini = np.zeros(190)
correlation_ev = np.zeros(18)
for sub_id in sub_list:
    index_sub = index_sub + 1
    for cut_id in range(0, 18):
        cut_id = cut_id + 1
        bold_fid = func_dir.joinpath(f'sub{sub_id}', f'sub{sub_id}_cut{cut_id}.nii.gz').as_posix()
        bold_img = image.load_img(bold_fid)
        img_affine  = bold_img.affine
        bold_dat = bold_img.get_data()
        i = -1
        for tr_id1 in range(0, 19):
            for tr_id2 in range(tr_id1+1, 20):
                i = i+1
                x1 = bold_dat[:,:,:,tr_id1]
                x2 = bold_dat[:,:,:,tr_id2]
                a = x1.flatten()
                b = x2.flatten()
                pccs = pearsonr(a, b)
                mini[i] = pccs[0]
                m = np.mean(mini)
        index_cut = cut_id-1
        p[index_sub, index_cut] = m
        correlation_ev = np.mean(p, axis=0)
np.savetxt('p.csv', p, delimiter = ',')  
np.savetxt('correlation_ev.csv', correlation_ev, delimiter = ',')  






                


