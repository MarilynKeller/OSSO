
import os

package_directory = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(package_directory, '../data')

skeleton_model_file = os.path.join(data_folder,'skeleton/skeleton_model.pkl')

path_male_star = os.path.join(data_folder, "star_1_1/male/model.npz" )
path_female_star = os.path.join(data_folder, "star_1_1/female/model.npz" )

lying_pose_path = os.path.join(data_folder, 'skeleton/lying_pose_{gender}.pkl')
skin_skel_contacts_file = os.path.join(data_folder, 'loss/skin_skel_contacts.pkl')
ball_joints_path = os.path.join(data_folder, 'loss/gloss_ball_joints_ldm.pkl')
clavicle_ligament_pairs_file = os.path.join(data_folder, 'loss/clavicle_ligament_pairs.pkl')
knee_ligament_pairs_file = os.path.join(data_folder, 'loss/knee_ligament_pairs.pkl')
skin_skeleton_pairs_file = os.path.join(data_folder, 'loss/skin_skeleton_pairs.pkl')
