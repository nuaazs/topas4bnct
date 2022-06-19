# ██╗██╗███╗   ██╗████████╗
# ██║██║████╗  ██║╚══██╔══╝
# ██║██║██╔██╗ ██║   ██║
# ██║██║██║╚██╗██║   ██║
# ██║██║██║ ╚████║   ██║
# ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝
# @Time    : 2022-05-27  09:56:33
# @Author  : 𝕫𝕙𝕒𝕠𝕤𝕙𝕖𝕟𝕘
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : http://iint.icu/
# @File    : /home/zhaosheng/4dct_work/get_input.py
# @Describe: get topas input from nii file.

import os
import argparse
import ants
import subprocess
from IPython import embed
import sys
from utils import make_father,generate_input_file,nii_to_array,get_source_loc,get_tumor_loc_from_seg,plot_loc
import numpy as np
import input_template
if __name__ == "__main__":

    backbones  = [
             "saru_r1_003",
             "resnet_r1",
             "unet_128_r1_003",
             "pix2pix_256_r1",
             "pix2pix_128_r1",
             ]
    pnames = ["003","011","016","022","033","077","086","099","102","136"]
    
    for backbone in backbones:
        for pname in pnames:
            scale = 1
            nii = f"/home/zhaosheng/paper2/online_code/response/result_niis/{backbone}_{pname}.nii"
            tumor_seg_nii = f"/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/seg/{pname}_gtv.npy"
            binary = "/home/zhaosheng/topas/cpp/binary"
            root = "./topas"
            
            ct_img = ants.image_read(nii)
            if scale !=1:
                ct_img = ants.resample_image(ct_img,np.array(ct_img.spacing)*scale,False,1)
            print("CT Image: ")
            print(ct_img)
            
            voxel_loc,tumor_img = get_tumor_loc_from_seg(seg_file=tumor_seg_nii,ct_spacing=ct_img.spacing)
            
            assert nii[-4:] == ".nii", "error: --nii file path should endwith '.nii' !"

            filename = nii.split("/")[-1].split(".")[0]

            # makedirs
            dat_path = os.path.join(root,"dats",filename+".dat")
            txt_path = os.path.join(root,"txts",filename+".txt")
            input_path = os.path.join(root,"inputs",filename+".txt")
            os.makedirs(os.path.join(root,"outputs"),exist_ok=True)
            make_father(dat_path)
            make_father(input_path)
            nii_save_path = os.path.join(root,"niis")
            os.makedirs(nii_save_path,exist_ok=True)

            # generate files
            shape,spacing,ct_array = nii_to_array(ct_img=ct_img,txt_path=txt_path,filename=filename+".nii",nii_save_path=nii_save_path)
            
            # calc source loc    
            source_loc,source_rot,source_voxel_loc = get_source_loc(shape=shape,spacing=spacing,voxel_loc=voxel_loc,ssd=50)


            print(f"# Tumor Info:\n\t{voxel_loc}")

            plot_loc(ct_array=ct_array,tumor_img=tumor_img,scale=scale,source_voxel_loc=source_voxel_loc,voxel_loc=voxel_loc,png_path=f"./source_loc_png/{backbone}_{pname}.png")

            # txt -> dat
            cmd = f"{binary} {txt_path} {dat_path}"
            subprocess.call(cmd, shell=True)

            input = input_template.INPUT
            output_prefix = f"{filename}_"
            input_txt = generate_input_file(shape=shape,spacing=spacing,source_loc=source_loc,rot=source_rot,dat_path=dat_path,my_template=input,output_prefix=output_prefix)

            # write topas input file
            with open(input_path, 'w') as f:
                f.write(input_txt)
            print(f"\nDone")

            