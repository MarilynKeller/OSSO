"""
Copyright©2022 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import numpy as np
import chumpy as ch
from chumpy import Ch
import numpy as np
import osso.config as cg
from gloss_skeleton.gloss.gsp import SpVerts
from gloss_skeleton.gloss.stitched_skel import StitchedSkeleton


def load_skeleton_model(gv_data=None):

    model_file = cg.skeleton_model_file

    skel = StitchedSkeleton()
    skel.load_model(model_file)

    # Initialization of the model
    ch_r_abs = Ch(np.zeros((skel.nParts * 3)))
    ch_t = Ch(skel.template_t.flatten())

    gv = SpVerts(skel_model=skel, r_abs=ch_r_abs, t=ch_t)

    if not gv_data is None:
        gv.r_abs[:] =  gv_data["r_abs"]
        gv.t[:] =  gv_data["t"]

    return gv


def init_stitched_puppet(skeleton_verts):
    
    gv = load_skeleton_model()

    # Replace the skeleton model template by the personalized template skeleton_verts
    i_end = 0
    for part_i in gv.model.partSet:
        nV = gv.model.shape[part_i]['T'].shape[0]/3
        assert nV == int(nV)
        nV = int(nV)
        i_start = i_end
        i_end = i_start + nV

        part_verts_offset =  np.mean(skeleton_verts[i_start: i_end, :] - gv.model.shape[part_i]['T'].reshape(-1,3), axis=0)
        part_verts = skeleton_verts[i_start: i_end, :] 
        part_verts = part_verts - part_verts_offset
        part_verts = part_verts.reshape(nV*3)

        gv.model.shape[part_i]['T'][:] = part_verts

    return gv