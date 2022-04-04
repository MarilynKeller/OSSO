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
from psbody.mesh.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewers
import chumpy as ch
from osso.utils import loss
from osso.utils.display import show_ball_joints, show_skin_bone_springs

def center_skeleton(gv, skin_mesh):
    # Center the skeleton with respect to the skin
    offset = np.mean(gv.r, axis=0) - np.mean(skin_mesh.v, axis=0) 

    offset_stacked = np.array(list(offset)*(gv.t.shape[0]//3))

    gv.t[:] = gv.t.r - offset_stacked

    return gv


def pose_skeleton(skel_pkl_lying_path, star_mesh_lying_path, skin_mesh_path, skel_mesh_posed, use_fast=True, display=False, verbose=True, joint_only=False):
    """Give a lying down skeleton model, the corresponding lying down body mesh, repose the skeleton to a target pose

    Args:
        skel_pkl_lying_path (str): Path to the input skeleton model
        star_mesh_lying_path (str): Path to the corresponding lying down body mesh
        skin_mesh_path (_type_): Path to a STAR mesh that has the target pose
        skel_mesh_posed (_type_): Path to save the output skeleton mesh
        use_fast (bool, optional): Set to False to optimize for more iterations. Defaults to True.
        display (bool, optional): If True, show the meshes during the alignment process. Defaults to False.
        joint_only (bool, optional): If True, load a previous result before joints correction and only apply joint correction.

    """
    if use_fast:
        max_iter_1 = 10
        max_iter_2 = 15
    else:
        max_iter_1 = 20
        max_iter_2 = 30        
    
    sp = pkl.load(open(skel_pkl_lying_path, 'rb'))
    
    source_skin = Mesh(filename=star_mesh_lying_path)
    target_skin = Mesh(filename=skin_mesh_path)
    
    free_variables = []
    free_variables.append(sp.r_abs)
    free_variables.append(sp.t)

    # Set up the costs
    stitching = loss.stitching(sp, free_clavicle=False) # stitching reference is the unposed model
    ball_joint_cost = loss.ball_joints_cost(sp, sp.r)

    knee_cost = loss.knee_ligament_cost(sp, sp.r)
    clavicle_cost = loss.clavicle_stitch_cost(sp, sp.r)

    skin_bone_spring_cost, skin_indices, skel_indices = loss.skin_bone_spring(sp, target_skin_mesh=target_skin, skin_mesh_init=source_skin)
    
    stitch_joints = [0,3,4,5,9,17,19] # We use the stitching cost only for the spine and ankles
    stitch_indices = []
    for sj in stitch_joints:
        stitch_indices += list(np.where(sp.pairIdx == sj)[0])

    # Translate the skeleton to align it roughly to the skin
    if not joint_only:
        sp = center_skeleton(sp, target_skin)

    if joint_only == True:
        r_abs, t = pkl.load(open(skel_mesh_posed.replace('.ply','_no_joints.pkl'), 'rb'))
        sp.r_abs[:] = r_abs
        sp.t[:] = t


    if display:
        mvs = MeshViewers((1,3))
        mv = mvs[0][0]
        mv.set_background_color(np.array([1,1,1.0]))
    
    def on_step(_):
        if display:
            gv_mesh = Mesh(sp.r, sp.f, vc='yellow')
            show_skin_bone_springs(gv_mesh, target_skin, skel_indices, skin_indices, skin_bone_spring_cost, mv)
            mvs[0][1].set_dynamic_meshes([gv_mesh, target_skin])
            show_ball_joints(sp, mvs[0][2], [Mesh(sp.r)])

    on_step(None)
    # mv.get_mouseclick()

    # Pre pose the skeleton using the stitched puppet model
    print('Pre pose the skeleton using the stitched puppet model')
    objs = {}
    objs['skin_spring'] = skin_bone_spring_cost
    objs['stitching'] =  10 * stitching
    objs['clavicle_cost'] =  clavicle_cost
    opt = {'maxiter':max_iter_1, 'disp':verbose}

    if not joint_only:
        ch.minimize(objs, x0=free_variables, method='dogleg', callback=on_step, options=opt)
    
        objs['stitching'] =  stitching
        ch.minimize(objs, x0=free_variables, method='dogleg', callback=on_step, options=opt)

        # uncomment to save the skeleton before joints correction
        # Mesh(sp.r, sp.f).write_ply(skel_mesh_posed.replace('.ply','_no_joints.ply'))
        # pkl.dump([sp.r_abs.r, sp.t.r ], open(skel_mesh_posed.replace('.ply','_no_joints.pkl'), 'wb'))


    # Enforce bone heads to be in their sockets
    objs = {}
    objs['stitching'] = 1e-1 * stitching[stitch_indices] 
    objs['tendon_cost'] = 1 * knee_cost
    objs['ball_joint'] = 1.1 * ball_joint_cost 
    objs['skin_spring'] = 5*1e-2 * skin_bone_spring_cost

    logging.info('Refine articulations')
    on_step(None)
    opt = {'maxiter':max_iter_2}
    ch.minimize(objs, x0=free_variables, method='dogleg', callback=on_step, options=opt)
    
    Mesh(sp.r, sp.f).write_ply(skel_mesh_posed)

    return sp
