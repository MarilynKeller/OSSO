"""
Copyright©2022 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import pickle as pkl
import osso.config as cg

skel_data_folder = cg.data_folder + 'skeleton/'

class Regressors:
    
    def __init__(self, gender):
        
        betas_regressor_file = skel_data_folder + f'betas_regressor_{gender}.pkl'
        ldm_regressor_file = skel_data_folder + f'ldm_regressor_{gender}.pkl'
        sp_ldm_indices_file = skel_data_folder + f'ldm_indices.pkl'
        skeleton_pca_file = skel_data_folder + f'skeleton_pca_{gender}.pkl'
        
        self.betas_regressor  = pkl.load(open(betas_regressor_file, 'rb'))
        self.ldm_regressor    = pkl.load(open(ldm_regressor_file, 'rb'))
        self.sp_ldm_indices   = pkl.load(open(sp_ldm_indices_file, 'rb'))
        self.skeleton_pca     = pkl.load(open(skeleton_pca_file, 'rb'))