"""
Copyright©2022 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import logging
import os
from pathlib import Path
import sys
import argparse
from osso.utils.inference import infer_skeleton
from osso.utils.pose_skeleton import pose_skeleton
from osso.utils.star_registration import register_star

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Given a body mesh, infer the underlying skeleton')
    
    parser.add_argument('--mesh_input', help='Path to a mesh with STAR or SMPL topology', default=None, required=True)
    parser.add_argument('--pkl_input', help='Path to a STAR registration. The file must be a pickle containing the keys "betas", "verts", "pose"', default=None)
    parser.add_argument('-g', '--gender', help='Gender of the subject to infer', required=True, choices=['female', 'male'])
    parser.add_argument('-p', '--posing_only', help='Run only the posing part', action='store_true')
    parser.add_argument('-v', '--verbose', help='Print optimization steps', action='store_true')
    parser.add_argument('-D', '--display', help='Display optimization steps', action='store_true')
    parser.add_argument('-F', '--force_recompute', help='Force recomputing meshes', action='store_true')
    parser.add_argument('--more_it', help='Use more  optimization iterations', action='store_true')
    
    args = parser.parse_args()
    
    skin_mesh_path = args.mesh_input

    gender = args.gender
    posing_only = args.posing_only
    verbose = args.verbose
    display = args.display
    force_recompute = args.force_recompute


    os.makedirs('out/tmp/', exist_ok=True)

    file_name = Path(args.mesh_input).stem

    star_mesh_path = f'out/tmp/{file_name}_star_posed.ply'
    star_pkl_path = f'out/tmp/{file_name}_star_posed.pkl'

    skel_mesh_lying_path = f'out/tmp/{file_name}_skel_lying.ply'
    skel_pkl_lying_path = f'out/tmp/{file_name}_skel_lying.pkl'
    star_mesh_lying_path = f'out/tmp/{file_name}_star_lying.ply'

    skel_mesh_posed = f'out/{file_name}_skel_posed.ply'

        
    # Register STAR to the input body mesh
    if not os.path.exists(star_pkl_path) or not os.path.exists(star_mesh_path):   
        logging.info(f'Registering STAR to mesh {skin_mesh_path} ...')
        register_star(skin_mesh_path, star_mesh_path, star_pkl_path, gender, display=display, verbose=verbose)
        logging.info(f'STAR registration saved as {star_pkl_path}.')  
    else:
        logging.info(f'STAR registration already exists in {star_pkl_path}. Force (-F) too recompute')  
        
    # Infer the skeleton in lying down pose
    if not os.path.exists(skel_pkl_lying_path):
        logging.info(f'Inferring the skeleton for skin {star_pkl_path} ...')  
        infer_skeleton(star_pkl_path, skel_mesh_lying_path, skel_pkl_lying_path, star_mesh_lying_path, gender, display=display, verbose=verbose)
        logging.info(f'Inferred skeleton mesh saved as {skel_mesh_lying_path}.')
    else:
        logging.info(f'Inferred lying down skeleton already exists in {skel_mesh_lying_path}. Force (-F) too recompute')  
    
    #Pose the skeleton to the target pose
    if not os.path.exists(skel_mesh_posed) or force_recompute:
        logging.info(f'Posing the infered skeleton to match the target pose {skin_mesh_path} ...') 
        for path in [skel_pkl_lying_path, star_mesh_lying_path, skin_mesh_path] :
            assert os.path.exists(path), f'Missing file {path}'
            
        pose_skeleton(skel_pkl_lying_path, star_mesh_lying_path, skin_mesh_path, skel_mesh_posed, use_fast=not args.more_it, display=display, verbose=verbose)
        logging.info(f'Posed skeleton saved as {skel_mesh_posed}.')
    else:
        logging.info(f'Posed skeleton already exists in {skel_mesh_posed}. Force (-F) too recompute')  
        
    
 
    
    