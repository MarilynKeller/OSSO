#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2016 Max Planck Society. All rights reserved.
# Modified by Ahmed A. A. Osman Feb 2020

import chumpy
from chumpy.ch import MatVecMult
import chumpy as ch
from .utils import verts_core , axis2quat , verts_core

def ischumpy(x):
    return hasattr(x, 'dterms')






def verts_decorated_quat(trans,
                    pose,
                    v_template,
                    J_regressor,
                    weights,
                    kintree_table,
                    f,
                    posedirs=None,
                    betas=None,
                    add_shape=True,
                    shapedirs=None,
                    want_Jtr=False,
                    use_poseblends = True,
                    joint_offset=None,
                    pose_dir_offset=None):

    for i, which in enumerate([trans, pose, v_template, weights, posedirs, betas, shapedirs]):
        if which is not None:
            assert ischumpy(which)

    v = v_template

    if betas is None:
        betas = ch.zeros(shapedirs.shape[-1])

    if shapedirs is not None:
        v_shaped = v + shapedirs.dot(betas)
    else:
        v_shaped = v

    #Add Shape of the model.
    quaternion_angles = axis2quat(pose.reshape((-1, 3))).reshape(-1)[4:]
    if betas is not None:
        shape_feat = betas[1]
        feat = ch.concatenate([quaternion_angles,shape_feat],axis=0)
        if use_poseblends:
            poseblends = posedirs.dot(feat)
            if pose_dir_offset is not None:
                poseblends = poseblends + pose_dir_offset
            v_posed = v_shaped + poseblends
        else:
            v_posed = v_shaped
    else:
        v_posed = v_shaped


    if joint_offset is None:
        import numpy as np
        joint_offset = np.zeros((J_regressor.shape[0], 3))

    J_tmpx = MatVecMult(J_regressor, v_shaped[:, 0]) + joint_offset[:,0]
    J_tmpy = MatVecMult(J_regressor, v_shaped[:, 1]) + joint_offset[:,1]
    J_tmpz = MatVecMult(J_regressor, v_shaped[:, 2]) + joint_offset[:,2]
    J = chumpy.vstack((J_tmpx, J_tmpy, J_tmpz)).T

    result, meta = verts_core(pose,v_posed, J, weights, kintree_table, want_Jtr=True)
    Jtr = meta.Jtr if meta is not None else None
    tr = trans.reshape((1, 3))
    result = result + tr
    Jtr = Jtr + tr

    result.trans = trans
    result.f = f
    result.pose = pose
    result.v_template = v_template
    result.J = J
    result.weights = weights

    result.posedirs = posedirs
    result.v_posed = v_posed
    result.shapedirs = shapedirs
    result.betas = betas
    result.v_shaped = v_shaped
    result.J_transformed = Jtr

    result.J_regressor = J_regressor
    result.kintree_table = kintree_table

    return result