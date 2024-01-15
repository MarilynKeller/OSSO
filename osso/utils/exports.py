import os 

from gloss_skeleton.gloss.stitched_skel import StitchedSkeleton
import osso.config as cg
from psbody.mesh import Mesh
from osso.utils.init_skel_model import load_skeleton_model

def export_per_part(sp, out_folder):
    
    # Save the sp skeleton mesh per part
    sk = StitchedSkeleton()
    skel_path = cg.skeleton_model_file
    
    os.makedirs(out_folder, exist_ok=True)
    sk.load_model(skel_path)
    
    gv = load_skeleton_model()

    i_end = 0
    for part_i in gv.model.partSet:
        part_name = sk.names[part_i]
        nV = gv.model.shape[part_i]['T'].shape[0]/3
        assert nV == int(nV)
        nV = int(nV)
        i_start = i_end
        i_end = i_start + nV

        part_mesh = Mesh(v = sp.r[i_start: i_end], f=gv.model.partFaces[part_i])
        # part_mesh.show()
        part_mesh.write_ply(os.path.join(out_folder, f'{part_i}_{part_name}.ply'))
        print(f'Exported {part_name} as {os.path.join(out_folder, f"{part_i}_{part_name}.ply")}')
        