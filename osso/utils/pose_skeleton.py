"""
Copyright©2022 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import copy
import importlib
import logging
import os
import pickle as pkl
import timeit
import numpy as np
from psbody.mesh.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewers
import chumpy as ch
import tqdm
from osso.utils import loss
from osso.utils.display import show_ball_joints, show_skin_bone_springs
from osso.star_model.star import STAR
from osso.utils.exports import export_per_part


def center_skeleton(gv, skin_mesh):
    # Center the skeleton with respect to the skin
    offset = np.mean(gv.r, axis=0) - np.mean(skin_mesh.v, axis=0) 

    offset_stacked = np.array(list(offset)*(gv.t.shape[0]//3))

    gv.t[:] = gv.t.r - offset_stacked



def pose_skeleton(skel_pkl_lying_path, star_mesh_lying_path, skin_mesh_path, skel_mesh_posed, use_fast=True, display=False, verbose=True, joint_only=False, per_parts=False):
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
    target_skin = Mesh(filename=skin_mesh_path)
    
    pose_skeleton_from_mesh(skel_pkl_lying_path, star_mesh_lying_path, target_skin, skel_mesh_posed, use_fast=True, display=False, verbose=True, joint_only=False, per_parts=per_parts)
    

def pose_skeleton_from_mesh(skel_pkl_lying_path, star_mesh_lying_path, target_skin, skel_mesh_posed, use_fast=True, display=False, verbose=True, joint_only=False, per_parts=False):
    
    if use_fast:
        max_iter_1 = 10
        max_iter_2 = 15
    else:
        max_iter_1 = 20
        max_iter_2 = 30        
    
    assert os.path.exists(skel_pkl_lying_path), f'Could not find lying down skeleton model at {skel_pkl_lying_path}'
    sp = pkl.load(open(skel_pkl_lying_path, 'rb'))
    assert sp is not None, 'Skeleton model could not be loaded'
    
    source_skin = Mesh(filename=star_mesh_lying_path)
    
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

    if joint_only == True:
        r_abs, t = pkl.load(open(skel_mesh_posed.replace('.ply','_no_joints.pkl'), 'rb'))
        sp.r_abs[:] = r_abs
        sp.t[:] = t
    else:
        # Translate the skeleton to align it roughly to the skin
        center_skeleton(sp, target_skin)


    if display:
        mvs = MeshViewers((1,3), keepalive=False)
        mv = mvs[0][0]
    
    def on_step(_):
        # show_meshes_with_ldm(mv, gv, gv_ldm_indices, pred_landmarks, skin_mesh)
        if display:
            gv_mesh = Mesh(sp.r, sp.f, vc='yellow')
            show_skin_bone_springs(gv_mesh, target_skin, skel_indices, skin_indices, skin_bone_spring_cost, mv)
            mvs[0][1].set_dynamic_meshes([gv_mesh, target_skin])
            # mvs[0][2].set_dynamic_meshes([gv_mesh, Mesh(target_skin.v)])
            show_ball_joints(sp, mvs[0][2], [Mesh(sp.r)])
            # print(ball_joint_cost.r)

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

        # Uncomment to save the skeleton before joints correction
        Mesh(sp.r, sp.f).write_ply(skel_mesh_posed.replace('.ply','_no_joints.ply'))
        pkl.dump([sp.r_abs.r, sp.t.r ], open(skel_mesh_posed.replace('.ply','_no_joints.pkl'), 'wb'))


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
    
    if per_parts:
        out_folder = skel_mesh_posed.replace('.ply', '_per_parts')
        export_per_part(sp, out_folder)

    return sp



def pose_skeleton_seq(skel_pkl_lying_path, 
                        star_mesh_lying_path, 
                        skin_pkl_path,
                        gender,
                        skin_verts_list,
                        frame_step,
                        skel_params_dict_path,
                        skel_mesh_posed_path, use_fast=True, 
                        display=False, verbose=True, joint_only=False):


    if use_fast:
        max_iter_1 = 10
        max_iter_2 = 1
    else:
        max_iter_1 = 20
        max_iter_2 = 30        
    
    
    source_skin = Mesh(filename=star_mesh_lying_path)

    sp0 = pkl.load(open(skel_pkl_lying_path, 'rb'))

    t0 = timeit.default_timer()
    
    sp = copy.deepcopy(sp0)
    # if sp is not None:
    #     sp0 = copy.deepcopy(sp)
    # sp = pkl.load(open(skel_pkl_lying_path, 'rb'))

    free_variables = []
    free_variables.append(sp.r_abs)
    free_variables.append(sp.t)

    # Set up the costs
    stitching = loss.stitching(sp, free_clavicle=False) # stitching reference is the unposed model
    ball_joint_cost = loss.ball_joints_cost(sp, sp.r)

    knee_cost = loss.knee_ligament_cost(sp, sp.r)
    clavicle_cost = loss.clavicle_stitch_cost(sp, sp.r)

    stitch_joints = [0,3,4,5,9,17,19] # We use the stitching cost only for the spine and ankles
    stitch_indices = []
    for sj in stitch_joints:
        stitch_indices += list(np.where(sp.pairIdx == sj)[0])

    print(timeit.default_timer() - t0, ' sec')

    if display:
        mvs = MeshViewers((1,3), keepalive=False)
        mv = mvs[0][0]



    last_frame_id = 0

    skel_params_dict = dict(np.load(skel_params_dict_path)) #the np file object is non mutable so we convert to a dictionary
    
    # import ipdb; ipdb.set_trace()
    n_frame = skin_verts_list.shape[0]
    for frame_id in tqdm.tqdm(range(0, n_frame, frame_step)):

        if np.any(skel_params_dict['rot'][frame_id, :]):
            # do not recompute already computed frames
            continue

        t0 = timeit.default_timer()

        skel_mesh_posed_path_fid = skel_mesh_posed_path.format(frame_id=frame_id)
        
        #todo define skin
        # skin_data = pkl.load(open(skin_pkl_path, 'rb'))
        # assert set(['verts', 'betas', 'pose']).issubset(skin_data.keys())
        # nb_beta = skin_data['betas'].shape[0]
        # skin_model = STAR(gender, num_betas=nb_beta)
        # skin_model.betas[:] = skin_data['betas']
        # skin_model.pose[:] = skel_params_dict[frame_id]
        # target_skin = Mesh(skin_model.r, skin_model.f)
        target_skin = Mesh(skin_verts_list[frame_id], source_skin.f)


        if frame_id == 0:
        # Translate the skeleton to align it roughly to the skin
        # if not joint_only:
            center_skeleton(sp, target_skin)
        else:
            sp.r_abs[:] =  skel_params_dict['rot'][last_frame_id, :]
            sp.t[:] = skel_params_dict['trans'][last_frame_id, :]
            

        skin_bone_spring_cost, skin_indices, skel_indices = loss.skin_bone_spring(sp, sp0, target_skin_mesh=target_skin, skin_mesh_init=source_skin)
        # import ipdb; ipdb.set_trace()

        if joint_only == True:
            r_abs, t = pkl.load(open(skel_mesh_posed_path_fid.replace('.ply','_no_joints.pkl'), 'rb'))
            sp.r_abs[:] = r_abs
            sp.t[:] = t



        
        def on_step(_):
            # show_meshes_with_ldm(mv, gv, gv_ldm_indices, pred_landmarks, skin_mesh)
            if display:
                gv_mesh = Mesh(sp.r, sp.f, vc='yellow')
                show_skin_bone_springs(gv_mesh, target_skin, skel_indices, skin_indices, skin_bone_spring_cost, mv)
                mvs[0][1].set_dynamic_meshes([gv_mesh, target_skin])
                # mvs[0][2].set_dynamic_meshes([gv_mesh, Mesh(target_skin.v)])
                show_ball_joints(sp, mvs[0][2], [Mesh(sp.r)])
                # print(ball_joint_cost.r)
                print(skin_bone_spring_cost)

        on_step(None)

        # import ipdb; ipdb.set_trace()
        # continue
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

            # Uncomment to save the skeleton before joints correction
            # Mesh(sp.r, sp.f).write_ply(skel_mesh_posed_path_fid.replace('.ply','_no_joints.ply'))
            # pkl.dump([sp.r_abs.r, sp.t.r ], open(skel_mesh_posed_path_fid.replace('.ply','_no_joints.pkl'), 'wb'))

        if False:
            # import ipdb; ipdb.set_trace()
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
            
        Mesh(sp.r, sp.f).write_ply(skel_mesh_posed_path_fid)

        # import ipdb; ipdb.set_trace()
        skel_params_dict['rot'][frame_id, :] = sp.r_abs.r
        skel_params_dict['trans'][frame_id, :] = sp.t.r

        
        np.savez(skel_params_dict_path, **skel_params_dict)

        last_frame_id = frame_id
        print('Frame aligned in ', timeit.default_timer() - t0, ' sec')

    return skel_params_dict
