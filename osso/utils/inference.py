"""
Copyright©2022 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import logging
import pickle as pkl
import numpy as np
import chumpy as ch
from psbody.mesh import Mesh, MeshViewer

from osso.star_model.star import STAR
from osso.utils.display import show_meshes_with_ldm
from osso.utils.regressors import Regressors
from osso.utils.init_skel_model import init_stitched_puppet
import osso.utils.loss as loss
import osso.config as cg


def pose_lying(skin_model, gender):
    lying_pose = pkl.load(open(cg.lying_pose_path.format(gender=gender), 'rb'))
    skin_model.pose[3:] = lying_pose


def infer_skeleton_shape_verts(skin_betas, betas_regressor, skeleton_pca):   
    # Predict skeleton betas from the skin
    skel_betas = np.dot(np.hstack([skin_betas, [1.]]), betas_regressor)[:10]    

    # From the skeleton betas, compute the skeleton vertices
    M = skeleton_pca['M']
    B = skeleton_pca['B']
    v = M + np.squeeze(np.dot(B[:, :, :len(skel_betas)], skel_betas[:,np.newaxis]))
    return v


def optimize_bone_loc(sp, skin_mesh, reg, verbose, display):
    # Predict ldm from skin
    pred_landmarks = reg.ldm_regressor.dot(skin_mesh.v)

    params = {}
    params['pair_contact'] = 1
    params['k_stitching'] = 1
    params['k_pose_trans'] = 1e-3
    params['k_pose_rot'] = 1e-3

    pair_contact_cost = loss.pair_contact_cost(sp, skin_mesh)
    ldm_loss = loss.ldm_loss(sp, reg.sp_ldm_indices, pred_landmarks)

    # Set up the objective
    objs = {}
    objs['ldm'] = ldm_loss 
    objs['contacts'] = params['pair_contact'] * pair_contact_cost


    free_variables = []
    free_variables.append(sp.r_abs)
    free_variables.append(sp.t)

    if display:
        mv = MeshViewer(keepalive=False)
    def on_step(_):
        if display:
            show_meshes_with_ldm(mv, sp, reg.sp_ldm_indices, pred_landmarks, skin_mesh)

    opt = {'maxiter':20, 'disp':verbose}
    logging.info('Optimizing for the bones location ...')
    ch.minimize(objs, x0=free_variables, method='dogleg', callback=on_step, options=opt)


def infer_skeleton(skin_pkl_path, skel_mesh_path, skel_pkl_path, skin_mesh_lying_path, gender, display=False, verbose=False):
    """Infer a skeleton mesh given STAR parameters.

    Args:
        skin_pkl_path (str): Path to a pickle file of a dictionary containing STAR parameters ("verts", "betas")
        skel_mesh_path (str): Path to save the infered lying down skeleton mesh
        skel_pkl_path (str): Path to save the infered lying down skeleton object
        skin_mesh_lying_path (_type_): Path to save the corresponding lying down body shape
        gender (str): Gender of the input body mesh, must be in ['male', 'female']
        display (bool, optional): If True, show the meshes during the alignment process. Defaults to False.
        verbose (bool, optional): If True, print the losses during optimization. Defaults to False.
    """

    # Load skin and pose it in lying down pose
    skin_data = pkl.load(open(skin_pkl_path, 'rb'))
    assert set(['verts', 'betas', 'pose']).issubset(skin_data.keys())
    nb_beta = skin_data['betas'].shape[0]
    skin_model = STAR(gender, num_betas=nb_beta)
    skin_model.betas[:] = skin_data['betas']
    pose_lying(skin_model, gender)
        
    # Visualize
    if display:
        Mesh(skin_model.r, skin_model.f).show() 
    skin_mesh = Mesh(skin_model.r, skin_model.f)
    
    #Load learned regressors
    reg = Regressors(gender)
    
    # Infer Skeleton shape in tpose
    skin_betas = skin_data['betas'][:10]
    skeleton_verts = infer_skeleton_shape_verts(skin_betas, reg.betas_regressor, reg.skeleton_pca)
    
    # Vizualize the infered skeleton shape
    m = Mesh(np.vstack([skin_mesh.v, skeleton_verts]))      
    
    # load stitched puppet with the regressed skeleton template
    sp = init_stitched_puppet(skeleton_verts)
    
    # Optimize for the bones location
    optimize_bone_loc(sp, skin_mesh, reg, verbose, display)
    
    pkl.dump(sp, open(skel_pkl_path, 'wb'))
    Mesh(sp.r, sp.f).write_ply(skel_mesh_path)
    Mesh(skin_model.r, skin_model.f).write_ply(skin_mesh_lying_path)
    
