"""
Copyright©2022 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import numpy as np
from psbody.mesh import Mesh
from psbody.mesh.sphere import Sphere
from psbody.mesh.lines import Lines
import pickle as pkl
import osso.config as cg
import random

from osso.utils.loss import sphere_fit_ch


def location_to_spheres(loc, color=(1,0,0), radius=0.02):
    """Given an array of 3D points, return a list of spheres located at those positions.

    Args:
        loc (numpy.array): Nx3 array giving 3D positions
        color (tuple, optional): One RGB float color vector to color the spheres. Defaults to (1,0,0).
        radius (float, optional): Radius of the spheres in meters. Defaults to 0.02.

    Returns:
        list: List of spheres Mesh
    """

    cL = [Sphere(np.asarray([loc[i, 0], loc[i, 1], loc[i, 2]]), radius).to_mesh() for i in range(loc.shape[0])]
    for spL in cL:
        spL.set_vertex_colors(np.array(color)) 
    return cL


def show_meshes_with_ldm(mv, gv, gv_ldm_indices, pred_landmarks, skin_mesh):
    
    c_ldm_model = location_to_spheres(gv[gv_ldm_indices].r, color=(0.5,0,0))
    c_ldm_pred = location_to_spheres(pred_landmarks, color=(1,1,1))
    
    M = Mesh(gv.r, gv.f)
    mesh_list = [M] + c_ldm_pred + c_ldm_model + [Mesh(skin_mesh.v)]
    mv.set_dynamic_meshes(mesh_list)


def show_skin_bone_springs(skel_mesh, skin_mesh, skel_indices, skin_indices, cost, mv):

    mv.set_dynamic_meshes([skel_mesh, Mesh(skin_mesh.v)])

    lines_list = []

    from matplotlib import cm
    # import ipdb; ipdb.set_trace()
    cost = cost/cost.max()
    colors = cm.jet(cost)[:, :3]

    for i, j, c in zip(skel_indices, skin_indices, colors):
        vi = skel_mesh.v[i]
        vj = skin_mesh.v[j]

        col = np.linalg.norm(c)
        li = Lines([vi,vj], [0,1], ec=c)
        lines_list.append(li)

    mv.set_dynamic_lines(lines_list)
    # mv.set_dynamic_meshes([skel_mesh, Mesh(skin_mesh.v)])


def color_gradient(N, scale=1, shuffle=False, darken=False, pastel=False):
    """Return a list of N color values forming a gradient"""
    import colorsys
    if darken:
        V = 0.75
    else:
        V = 1

    if pastel:
        S = 0.5
    else:
        S = 1

    HSV_tuples = [((N-x) * 1.0 / (scale*N), S, V) for x in range(N)] # blue - grean - yellow - red gradient
    RGB_list = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), HSV_tuples))

    if shuffle:
        random.Random(0).shuffle(RGB_list) #Seeding the random this way always produces the same shuffle
    return RGB_list


def show_ball_joints(skel_v, mv, mesh_list):

    bjoints_dict = pkl.load(open(cg.ball_joints_path, 'rb'))
    colors = color_gradient(len(bjoints_dict['verts_ids']))

    for i, verts_ids in enumerate(bjoints_dict['verts_ids']):
        verts = skel_v[verts_ids, :]
        center, radius = sphere_fit_ch(verts)

        s = Sphere(center.r, 0.02).to_mesh(color=colors[i])
        mesh_list.append(s)
        mv.set_dynamic_meshes(mesh_list)
