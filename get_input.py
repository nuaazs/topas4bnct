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
    """
    python get_input.py --nii /home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/real_ct_niis/003.nii \
                        --root ./topas \
                        --scale 1 \
                        --binary /home/zhaosheng/topas/cpp/binary \
                        --tumor_seg_nii /home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/seg/003_gtv.npy
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nii', type=str, default="0",help='ct nii file path.(endwith .nii)')
    parser.add_argument('--root', type=str, default="/home/limingzhu/4dct/topas",help='output file path')
    parser.add_argument('--scale', type=int, default=2,help='scale, 2 means resample 512,512,128 -> 256,256,64')
    parser.add_argument('--tumor_loc',type=str, default=None,help='tumor location (sampe shape of ct nii file), split by csv, such as (177,86,37)')
    parser.add_argument('--tumor_seg_nii',type=str, default=None,help='tumor seg file (sampe shape of ct nii file),nii or npy.')
    parser.add_argument('--binary',type=str, default="/home/zhaosheng/topas/cpp/binary",help='cpp bin')
    parser.add_argument('--istry',action='store_true',default=False,help='')
    

    args = parser.parse_args()
    if args.istry:
        print("just test")

    ct_img = ants.image_read(args.nii)
    if args.scale !=1:
        ct_img = ants.resample_image(ct_img,np.array(ct_img.spacing)*args.scale,False,1)
    print("CT Image: ")
    print(ct_img)
    if args.tumor_loc:
        voxel_loc = [int(item) for item in args.tumor_loc.split(',')]
    elif args.tumor_seg_nii:
        voxel_loc,tumor_img = get_tumor_loc_from_seg(seg_file=args.tumor_seg_nii,ct_spacing=ct_img.spacing)
    else:
        print("error: no source location.")
        sys.exit()
    #voxel_loc = (int(voxel_loc[0]/args.scale),int(voxel_loc[1]/args.scale),int(voxel_loc[2]/args.scale))
    assert args.nii[-4:] == ".nii", "error: --nii file path should endwith '.nii' !"

    filename = args.nii.split("/")[-1].split(".")[0]

    # makedirs
    dat_path = os.path.join(args.root,"dats",filename+".dat")
    txt_path = os.path.join(args.root,"txts",filename+".txt")
    input_path = os.path.join(args.root,"inputs",filename+".txt")
    os.makedirs(os.path.join(args.root,"outputs"),exist_ok=True)
    make_father(dat_path)
    make_father(input_path)
    nii_save_path = os.path.join(args.root,"niis")
    os.makedirs(nii_save_path,exist_ok=True)

    # generate files
    shape,spacing,ct_array = nii_to_array(ct_img=ct_img,txt_path=txt_path,filename=filename+".nii",nii_save_path=nii_save_path)
    
    # calc source loc    
    source_loc,source_rot,source_voxel_loc = get_source_loc(shape=shape,spacing=spacing,voxel_loc=voxel_loc,ssd=50)


    print(f"# Tumor Info:\n\t{voxel_loc}")

    # if args.tumor_seg_nii:
    #     plot_loc(ct_array=ct_array,tumor_img=tumor_img,scale=args.scale,source_voxel_loc=source_voxel_loc,voxel_loc=voxel_loc,png_path="temp.png")
    # else:
    plot_loc(ct_array=ct_array,tumor_img=tumor_img,scale=args.scale,source_voxel_loc=source_voxel_loc,voxel_loc=voxel_loc,png_path=f"./source_loc_png/{filename}.png")

    # txt -> dat
    cmd = f"{args.binary} {txt_path} {dat_path}"
    subprocess.call(cmd, shell=True)
    if args.istry:
        input = input_template.INPUT_TEST
        output_prefix = f"{filename}_"
        input_txt = generate_input_file(shape=shape,spacing=spacing,source_loc=source_loc,rot=source_rot,dat_path=dat_path.split("/")[-1],my_template=input,output_prefix=output_prefix)
    else:
        input = input_template.INPUT
        output_prefix = f"{filename}_"
        input_txt = generate_input_file(shape=shape,spacing=spacing,source_loc=source_loc,rot=source_rot,dat_path=dat_path,my_template=input,output_prefix=output_prefix)
    
    # write topas input file
    with open(input_path, 'w') as f:
        f.write(input_txt)
    print(f"\nDone")

    