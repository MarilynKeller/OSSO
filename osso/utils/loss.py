"""
Copyright©2022 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import numpy as np
import osso.config as cg
import pickle as pkl
import chumpy as ch


def pair_contact_cost(sp, skin_mesh):
    skin_indices, skel_indices, pair_weights = pkl.load(open(cg.skin_skel_contacts_file, 'rb'))
    skel_contact_pair_indices = sp.from_template_topology(skel_indices)

    skin_thickness = 0.005
    skin_points = skin_mesh.v[skin_indices, :]
    skin_points = skin_points - skin_thickness * skin_mesh.estimate_vertex_normals()[skin_indices, :]
    pair_contact_cost = pair_weights * ch.sum((skin_points - sp[skel_contact_pair_indices, :]) ** 2, axis=1)
    return pair_contact_cost


def ldm_loss(sp, skel_ldm_indices, pred_ldm):
    """ Loss to minimize the distance between landmarks predicted from a skin shape 
    and the corresponding skeleton model vertices"""
    return sp[skel_ldm_indices] - pred_ldm


def compute_stitching_template(sp):
    return  sp[sp.interfIdx[0],:].r - sp[sp.interfIdx[1],:].r


def stitching(sp, stitching_template_tpose=None, free_clavicle=True):
    
    if stitching_template_tpose is None:
        # Compute the template stitching cost given the skeleton is not connected
        stitching_template_tpose = compute_stitching_template(sp)

    stitch_weight =  np.ones((len(sp.pairIdx),3))

    if free_clavicle:
        pair_idx = [6,7]
        for pid in pair_idx:
            I = np.where(sp.pairIdx == pid)[0]
            stitch_weight[I] = 0
    
    stitching = stitch_weight * (sp[sp.interfIdx[0],:] - sp[sp.interfIdx[1],:] - stitching_template_tpose)
    return stitching


def get_joint_pairs():

    joint_pairs = [
    ['femur_PL', 'hips_L'],
    ['femur_PR', 'hips_R'],
    ['humerus_DL', 'ulna_PL'],
    ['humerus_DR', 'ulna_PR'],
    
    ['humerus_PL', 'shoulder_L'],
    ['humerus_PR', 'shoulder_R'],

    ['femur_DL', 'tibia_PL'],
    ['femur_DR', 'tibia_PR'],
    
    # ['clavicle_L', 'sternum_L'],
    # ['clavicle_R', 'sternum_R'],
    ]

    return joint_pairs


def spring(v1, v2, r0):
    return ch.linalg.norm(v2-v1) - r0


def sphere_fit_ch(verts):
    # Code adapted from rom https://jekel.me/2015/Least-Squares-Sphere-Fit/ 
    # Close form solution of the sphere fitting

    # Assemble the A matrix
    spX = verts[:, 0]
    spY = verts[:, 1]
    spZ = verts[:, 2]
    
    A = ch.concatenate([spX[:, np.newaxis]*2,  
                        spY[:, np.newaxis]*2,
                        spZ[:, np.newaxis]*2, 
                        np.ones((len(spX),1))], axis=1)

    # Assemble the f matrix
    f = ((spX*spX) + (spY*spY) + (spZ*spZ))[:, np.newaxis]

    C, residules, rank, singval = ch.linalg.lstsq(A,f)
    center = C[:3,0]

    # Solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = ch.sqrt(t)

    assert center.dr_wrt(verts) is not None
    return center, radius


def get_ball_sphere(sp, b_name, bjoints_dict, return_chumpy = False):
    # Fit spheres to specific points of the mesh
    # The returned parameters are differenciable wrt gv verts
    b_index = bjoints_dict['names'].index(b_name)
    b_verts = bjoints_dict['verts_ids'][b_index]

    verts = sp[b_verts]
    if type(verts) == np.ndarray:
        verts = ch.array(verts)

    c, r = sphere_fit_ch(verts)
    
    if return_chumpy:
        assert c.dr_wrt(verts) is not None
        assert c.dr_wrt(sp) is not None
        return c, r
    else:
        return c.r, r.r


def ball_joints_cost(sp, sp0):

    bjoints_dict = pkl.load(open(cg.ball_joints_path, 'rb'))
    joints_cost_list = []
    
    for b1_name, b2_name in get_joint_pairs():

        return_chumpy = True
    
        v1, _ = get_ball_sphere(sp, b1_name, bjoints_dict, return_chumpy)
        v2, _ = get_ball_sphere(sp, b2_name, bjoints_dict, return_chumpy)

        v10, _ = get_ball_sphere(sp0, b1_name, bjoints_dict)
        v20, _ = get_ball_sphere(sp0, b2_name, bjoints_dict)
        r0 = np.linalg.norm(v10-v20)

        if ('femur' in b1_name) and ('hips' in b2_name):
            r0 = 0

        c = spring(v1, v2, r0)
        joints_cost_list.append(c)

    cost = ch.concatenate(joints_cost_list)
    assert cost.dr_wrt(sp) is not None
    return cost


def ligament_cost(sp, sp_init, ligament_pairs_file):
    ligament_array = pkl.load(open(ligament_pairs_file, 'rb'))

    cost = []
    for (i, j) in ligament_array:
        d0 = np.linalg.norm(sp_init[i] - sp_init[j])
        d = ch.linalg.norm(sp[i] - sp[j])
        cost.append(d - d0)

    return ch.concatenate(cost)


def knee_ligament_cost(sp, sp_init):
    return ligament_cost(sp, sp_init, cg.knee_ligament_pairs_file)


def clavicle_stitch_cost(sp, sp_init):
    return ligament_cost(sp, sp_init, cg.clavicle_ligament_pairs_file)


def skin_bone_spring(gv, target_skin_mesh, skin_mesh_init):

    skin_indices, skel_indices =  pkl.load(open(cg.skin_skeleton_pairs_file, 'rb'))

    vect0 = gv.r[skel_indices,:] - skin_mesh_init.v[skin_indices,:]
    vect = (gv[skel_indices,:] - target_skin_mesh.v[skin_indices,:])

    d0 = np.linalg.norm(vect0, axis=1)
    e = 0.01 # minimum skin tickness (mm)
    d0[(d0 < e)] = e
    
    skin_normal = target_skin_mesh.estimate_vertex_normals()[skin_indices,:]
    # Per line dot product
    coeff = -ch.sign(ch.sum(vect * skin_normal, axis=1) / (ch.sqrt(vect[:,0]**2 + vect[:,1]**2 + (vect[:,2]**2))))
    cost =  (ch.sqrt(vect[:,0]**2 + vect[:,1]**2 + (vect[:,2]**2))) * coeff.T - d0
    return cost, skin_indices, skel_indices
