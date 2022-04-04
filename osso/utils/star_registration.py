"""
Copyright©2022 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import pickle as pkl
from psbody.mesh import Mesh, MeshViewers
import chumpy as ch
import numpy as np
from osso.star_model.star import STAR

def register_star(skin_mesh_path, star_mesh_path, star_pkl_path, gender, display=False, verbose=False):
    """Given a body mesh, align the body model STAR to it.

    Args:
        skin_mesh_path (str): Path to the input body mesh. Must have SMPL or STAR topology, i.e. 6890 verts, 13776 faces
        star_mesh_path (str): Path to save the aligned STAR mesh
        star_pkl_path (_type_): Path to save the aligned STAR parameters as .pkl
        gender (str): Gender of the input body mesh, must be in ['male', 'female']
        display (bool, optional): If True, show the meshes during the alignment process. Defaults to False.
        verbose (bool, optional): If True, print the losses during optimization. Defaults to False.
    """

    assert ('.ply' in skin_mesh_path) or ('.obj' in skin_mesh_path), 'skin path must be in .ply or .obj format'

    skin_mesh = Mesh(filename=skin_mesh_path)

    sv = STAR(gender, num_betas=10)
    assert skin_mesh.v.shape[0] == sv.r.shape[0], (f'Input mesh should have the same topology as STAR or SMPL. Mesh has {skin_mesh.v.shape[0]} vertices, {sv.v.shape[0]} expected')

    #weights
    params={}
    params['p2p'] = 1
    params['betas'] = 1
    params['pose'] = 1

    if display:
        mv = MeshViewers((1, 2), keepalive=False)

    def on_step(_):
        if display:
            Skel_mesh = Mesh(sv.r, sv.f, vc="green")
            mv[0][0].set_dynamic_meshes([Skel_mesh])

            # Visual inspection
            skin_mesh.set_vertex_colors(vc="white")
            mv[0][1].set_dynamic_meshes([skin_mesh, Skel_mesh])

    sv.trans[:] = np.mean(skin_mesh.v, axis=0)

    #set objectives
    p2p =  sv - skin_mesh.v
    betas =  sv.betas
    pose =  sv.pose

    objs = {
        'p2p' : params['p2p'] * p2p,
        'betas' : params['betas'] * betas  ,
        'pose' : params['pose'] * pose ,
    }

    # Optimize
    options={'maxiter': 5, 'disp': verbose}
    ch.minimize(objs, [sv.pose[0:3], sv.trans], method='dogleg', callback=on_step, options=options)
    ch.minimize(objs, [sv.betas[:3], sv.pose, sv.trans], method='dogleg', callback=on_step, options=options)

    for i in range(2):
        ch.minimize(objs, [sv.betas, sv.pose, sv.trans], method='dogleg', callback=on_step, options=options)
        params['p2p'] *= 100
        objs['p2p'] = params['p2p'] * p2p

    sv_data = {}
    sv_data['pose'] =   sv.pose.r
    sv_data['betas'] =  sv.betas.r
    sv_data['trans'] =  sv.trans.r
    sv_data['verts'] =  sv.r
    sv_data['faces'] = sv.f

    if star_pkl_path:
        Mesh(sv, sv.f).write_ply(star_mesh_path)
        pkl.dump(sv_data, open(star_pkl_path, 'wb'))

    return
