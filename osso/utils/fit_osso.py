import argparse
from email import parser
import socket
import numpy as np
from osso.utils.star_registration import register_star
from osso.utils.inference import infer_skeleton
from osso.utils.pose_skeleton import pose_skeleton_from_mesh

import trimesh
import os
from psbody.mesh import Mesh


class AmassSkelAlign():

    def __init__(self,
                smpl_data,
                out_folder,
                unpose_hands=True,
                display = True,
                frame=None):

        self.smpl_data = smpl_data
        self.unpose_hands = unpose_hands
        self.out_folder = out_folder
        self.display = display
        
        self.gender = smpl_data['gender']

        self.skel_mesh_posed_folder = os.path.join(self.out_folder, 'skel')
        self.skel_mesh_posed_path = os.path.join(self.skel_mesh_posed_folder, 'skel_{frame_id:04d}.ply' )
        
        os.makedirs(self.out_folder, exist_ok=True)
        os.makedirs(self.skel_mesh_posed_folder, exist_ok=True)
        


    def generate_star_body(self):

        # Generate the subject's SMPL mesh
        mesh = trimesh.Trimesh(vertices=self.smpl_data['verts'][0], faces=self.smpl_data['faces'], process=False)

        self.skin_mesh_path = os.path.join(self.out_folder, 'smpl.obj')
        mesh.export(self.skin_mesh_path)

        star_mesh_path = os.path.join(self.out_folder, 'star.ply')
        self.star_pkl_path = os.path.join(self.out_folder, 'star.pkl')

        if not os.path.exists(self.star_pkl_path) or not os.path.exists(star_mesh_path):  
            register_star(self.skin_mesh_path, star_mesh_path, self.star_pkl_path, self.gender, display=self.display, verbose=True)


    def infer_template_skeleton(self):
        self.skel_mesh_lying_path = os.path.join(self.out_folder, 'skel_lying.ply')
        self.skel_pkl_lying_path = os.path.join(self.out_folder, 'skel_lying.pkl')
        self.star_mesh_lying_path = os.path.join(self.out_folder, 'star_lying.ply')

        if not os.path.exists(self.skel_pkl_lying_path) or True:
            infer_skeleton(self.star_pkl_path, self.skel_mesh_lying_path, self.skel_pkl_lying_path, self.star_mesh_lying_path, 
                        self.gender, display=self.display, verbose=True)



    def pose_skeleton(self, frame_id):    

        vertices = self.seq_amass.vertices[frame_id]
        faces = self.seq_amass.faces
        target_skin = Mesh(vertices, faces)

        if not os.path.exists(self.skel_mesh_posed_path):
            pose_skeleton_from_mesh(self.skel_pkl_lying_path, 
                                    self.star_mesh_lying_path, 
                                    target_skin, 
                                    self.skel_mesh_posed_path.format(frame_id=frame_id), use_fast=True, 
                                    display=self.display, verbose=True, joint_only=False)
    

 
def fit_osso(smpl_data, output_folder, display):
    """ Fit OSSO to a SMPL mesh.
    @param smpl_data: dict containing the SMPL parameters, vertices and faces
    @param output_folder: path to output the generated data
    @param display: whether to display the meshes during the fitting process
    """
    
    skel_align = AmassSkelAlign(
        smpl_data = smpl_data,
        out_folder = output_folder,
        unpose_hands = True,
        display=display
    )

    skel_align.generate_star_body()
    skel_align.infer_template_skeleton()
