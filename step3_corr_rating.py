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

corr = np.zeros([4,2])
rating = pd.read_csv(base_dir.joinpath('ev_rating.txt'), sep='\t')

corr[0,0] = pearsonr(rating.logic, rating.ev_pear)[0]
corr[1,0] = pearsonr(rating.space, rating.ev_pear)[0]
corr[2,0] = pearsonr(rating.tool, rating.ev_pear)[0]
corr[3,0] = pearsonr(rating.activity, rating.ev_pear)[0]
corr[0,1] = pearsonr(rating.logic, rating.ev_pear)[1]
corr[1,1] = pearsonr(rating.space, rating.ev_pear)[1]
corr[2,1] = pearsonr(rating.tool, rating.ev_pear)[1]
corr[3,1] = pearsonr(rating.activity, rating.ev_pear)[1]

np.savetxt('corr_4.csv', corr, delimiter = ',')  