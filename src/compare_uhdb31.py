import numpy as np
import scipy.spatial.distance as ssd
import glob
import os

uhdb31_ft_dir = '/home/xiang/Code/FaceNet/uhdb31/ft'
output_dir = '/home/xiang/Code/FaceNet/uhdb31'

gallery_id = 11
gallery_ft = []
gallery_ft_list = glob.glob(uhdb31_ft_dir + "/*11.txt")
gallery_id_list = []
for g_ft_path in gallery_ft_list:
    bname = os.path.splitext(os.path.basename(g_ft_path))[0]
    gallery_id_list.append(bname[0:5])
    gallery_ft.append(np.loadtxt(g_ft_path))

for pose_id in range(1, 22):
    if pose_id != 11:
        if pose_id < 10:
            pose_str = '0'+str(pose_id)
        else:
            pose_str = str(pose_id)

        cur_out_dir = os.path.join(output_dir, 'P'+pose_str)
        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)

        probe_ft_list = glob.glob(uhdb31_ft_dir + '/*'+pose_str+'.txt')
        probe_id_list = []
        probe_ft = []
        for p_ft_path in probe_ft_list:
            bname = os.path.splitext(os.path.basename(p_ft_path))[0]
            probe_id_list.append(bname[0:5])
            probe_ft.append(np.loadtxt(p_ft_path))

        with open(os.path.join(cur_out_dir, 'gallery_id.txt'), 'w') as f:
            for gallery_id in gallery_id_list:
                f.write(gallery_id + '\n')

        with open(os.path.join(cur_out_dir, 'probe_id.txt'), 'w') as f:
            for probe_id in probe_id_list:
                f.write(probe_id + '\n')

        scores = np.ones((len(probe_id_list), len(gallery_id_list)))*99999
        for i, p_ft in enumerate(probe_ft):
            for j, g_ft in enumerate(gallery_ft):
                s = ssd.cosine(p_ft, g_ft)
                scores[i, j] = s

        np.savetxt(os.path.join(cur_out_dir, 'score.txt'), scores)