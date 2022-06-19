import numpy as np
import os
import nibabel as nib
from pathlib import Path
import ants
import subprocess
import input_template
from IPython import embed
from scipy import ndimage
import numpy
import matplotlib.pyplot as plt
import sys

def make_father(save_path):
    """make father path

    Args:
        save_path (_type_): _description_
    """
    path = Path(save_path)
    father = path.parent
    os.makedirs(father,exist_ok=True)

def generate_input_file(shape,spacing,source_loc,rot,dat_path,my_template,output_prefix):
    """generate topas input filename

    Args:
        shape (set): ct shape
        spacing (set): ct spacing
        source_loc (set): source loc
        rot (set): source rot
        dat_path (string): dat file path
        my_template (string): template
        output_prefix (string): prefix

    Returns:
        string: input content
    """
    my_template = my_template.replace("<!X_SHAPE>",str(int(shape[0])))
    my_template = my_template.replace("<!Y_SHAPE>",str(int(shape[1])))
    my_template = my_template.replace("<!Z_SHAPE>",str(int(shape[2])))

    my_template = my_template.replace("<!X_SIZE>",str(spacing[0]))
    my_template = my_template.replace("<!Y_SIZE>",str(spacing[1]))
    my_template = my_template.replace("<!Z_SIZE>",str(spacing[2]))

    my_template = my_template.replace("<!SOURCE_ROT_X>",str(int(rot[0])))
    my_template = my_template.replace("<!SOURCE_ROT_Y>",str(int(rot[1])))
    my_template = my_template.replace("<!SOURCE_ROT_Z>",str(int(rot[2])))

    my_template = my_template.replace("<!SOURCE_LOC_X>",str(int(source_loc[0])))
    my_template = my_template.replace("<!SOURCE_LOC_Y>",str(int(source_loc[1])))
    my_template = my_template.replace("<!SOURCE_LOC_Z>",str(int(source_loc[2])))
    my_template = my_template.replace("<!OUTPUT_PATH_PERFIX>",output_prefix)

    my_template = my_template.replace("<!DAT_PATH>",dat_path)
    return my_template

def nii_to_array(ct_img,txt_path,filename,nii_save_path=None):
    """nii image to np array(ct -> material number)

    Args:
        ct_img (ants image): ct
        txt_path (string): _description_
        filename (sting): _description_
        nii_save_path (string, optional): _description_. Defaults to None.

    Returns:
        shape,spacing,ct_array: _description_
    """
    img = ct_img
    if nii_save_path:
        img.to_file(os.path.join(nii_save_path,filename))
    print(img)
    ct_array = img.numpy()
    for i in range(ct_array.shape[0]):
        for j in range(ct_array.shape[1]):
            for k in range(ct_array.shape[2]):
                if ct_array[i,j,k]<-950:
                    ct_array[i,j,k]=1
                elif ct_array[i,j,k]<-120:
                    ct_array[i,j,k]=2
                elif ct_array[i,j,k]<-88:
                    ct_array[i,j,k]=3
                elif ct_array[i,j,k]<-53:
                    ct_array[i,j,k]=4
                elif ct_array[i,j,k]<-23:
                    ct_array[i,j,k]=5
                elif ct_array[i,j,k]<7:
                    ct_array[i,j,k]=6
                elif ct_array[i,j,k]<18:
                    ct_array[i,j,k]=7
                elif ct_array[i,j,k]<80:
                    ct_array[i,j,k]=8
                elif ct_array[i,j,k]<120:
                    ct_array[i,j,k]=9
                elif ct_array[i,j,k]<200:
                    ct_array[i,j,k]=10
                elif ct_array[i,j,k]<300:
                    ct_array[i,j,k]=11
                elif ct_array[i,j,k]<400:
                    ct_array[i,j,k]=12
                elif ct_array[i,j,k]<500:
                    ct_array[i,j,k]=13
                elif ct_array[i,j,k]<600:
                    ct_array[i,j,k]=14
                elif ct_array[i,j,k]<700:
                    ct_array[i,j,k]=15
                elif ct_array[i,j,k]<800:
                    ct_array[i,j,k]=16
                elif ct_array[i,j,k]<900:
                    ct_array[i,j,k]=17
                elif ct_array[i,j,k]<1000:
                    ct_array[i,j,k]=18
                elif ct_array[i,j,k]<1100:
                    ct_array[i,j,k]=19
                elif ct_array[i,j,k]<1200:
                    ct_array[i,j,k]=20
                elif ct_array[i,j,k]<1300:
                    ct_array[i,j,k]=21
                elif ct_array[i,j,k]<1400:
                    ct_array[i,j,k]=22
                elif ct_array[i,j,k]<1500:
                    ct_array[i,j,k]=23
                elif ct_array[i,j,k]<1600:
                    ct_array[i,j,k]=24
                else:
                    ct_array[i,j,k]=25

    array1 = ct_array.transpose(2,0,1)
    
    spacing = img.spacing
    shape = ct_array.shape
    # print(f"Shape:{shape} Spacing : {spacing}")
    array1 = array1.copy().flatten()
    new_array = array1.reshape(len(array1),1)
    make_father(txt_path)
    with open(txt_path, 'w') as f4:
        np.savetxt(f4, new_array, delimiter=' ', newline='\n',fmt="%i")
    
    return shape,spacing,ct_array

# Enter the voxel coordinates of the tumor along with the voxel size and array shape to get the true bits of the source (mm)
def get_source_loc(shape,spacing,voxel_loc,ssd):
    """get voxel coordinates of source

    Args:
        shape (_type_): _description_
        spacing (_type_): _description_
        voxel_loc (_type_): _description_
        ssd (_type_): _description_

    Returns:
        _type_: _description_
    """
    tumor_x_loc = (voxel_loc[1]-shape[0]/2)*spacing[0]
    tumor_y_loc = -1*(voxel_loc[0]-shape[1]/2)*spacing[1]
    tumor_z_loc = (voxel_loc[2]-shape[2]/2)*spacing[2]
    # embed()
    left_delta = 256 - voxel_loc[0]
    right_delta = voxel_loc[0]
    behind_delta = 256 - voxel_loc[1]
    front_delta = voxel_loc[1]
    _list = np.array([left_delta,right_delta,front_delta,behind_delta])
    _orients = ['left','right','front','behind']
    orient = _orients[np.argmin(_list)]

    if orient == "left":
        source_y_loc = -shape[1]/2*spacing[1]-ssd
        source_x_loc = tumor_x_loc
        source_z_loc = tumor_z_loc
        # rot = "\tx: 90. deg\n\ty: 0. deg\n\tz: 0. deg"
        rot = [90,0,0]
        source_voxel_loc = [shape[0],voxel_loc[1],voxel_loc[2]]
    elif orient == "right":
        source_y_loc = shape[1]/2*spacing[1]+ssd
        source_x_loc = tumor_x_loc
        source_z_loc = tumor_z_loc
        rot = "\tx: -90. deg\n\ty: 0. deg\n\tz: 0. deg"
        rot = [-90,0,0]
        source_voxel_loc = [0,voxel_loc[1],voxel_loc[2]]
    elif orient == "front":
        source_x_loc = -shape[0]/2*spacing[0]-ssd
        source_y_loc = tumor_y_loc
        source_z_loc = tumor_z_loc
        # rot = "\tx: 0. deg\n\ty: -90. deg\n\tz: 0. deg"
        rot = [0,-90,0]
        source_voxel_loc = [voxel_loc[0],0,voxel_loc[2]]
    elif orient == "behind":
        source_x_loc = shape[0]/2*spacing[0]+ssd
        source_y_loc = tumor_y_loc
        source_z_loc = tumor_z_loc
        # rot = "x: 0. deg\ny: 90. deg\nz: 0. deg"
        rot = [0,90,0]
        source_voxel_loc = [voxel_loc[0],shape[1],voxel_loc[2]]
    loc = [source_x_loc,source_y_loc,source_z_loc]
    print(f"# Source Info:\n\tSource Location:{loc}\n\tRot:{rot}\n\tSource Location voxel:{source_voxel_loc}\n\tOirent:{orient}")
    
    return loc,rot,source_voxel_loc

# 
def get_body_centered(array):
    """get body centered

    Args:
        array (_type_): _description_

    Returns:
        _type_: _description_
    """
    return ndimage.measurements.center_of_mass(array)

# Enter the voxel coordinates of the tumor along with the voxel size and array shape to get the true bits of the source (mm)
def get_tumor_loc_from_seg(seg_file,ct_spacing):
    """get tumor loc from seg file(nii,)

    Args:
        seg_file (_type_): _description_
        ct_spacing (_type_): _description_

    Returns:
        _type_: _description_
    """
    if seg_file.endswith('.npy'):
        array =np.load(seg_file)
        get_body_centered(array)
        tumor_img = ants.from_numpy(array)
        tumor_img.set_spacing(ct_spacing)
        assert tumor_img.spacing == ct_spacing
        voxel_loc = get_body_centered(tumor_img.numpy())

    else:
        tumor_img = ants.image_read(seg_file)
        print(f"# Load tumor image from {seg_file}:\n\tShape:{tumor_img.shape},Spacing:{tumor_img.spacing},")
        if tumor_img.spacing != ct_spacing:
            tumor_img = ants.resample_image(tumor_img,ct_spacing,False,1)
        print(tumor_img)
        
        voxel_loc = get_body_centered(tumor_img.numpy())
        # embed()
        print(voxel_loc)
    return (int(voxel_loc[0]),int(voxel_loc[1]),int(voxel_loc[2])),tumor_img

def plot_loc(ct_array,tumor_img,scale,source_voxel_loc,voxel_loc,png_path):
    """plot source location

    Args:
        ct_array (_type_): _description_
        tumor_img (_type_): _description_
        scale (_type_): _description_
        source_voxel_loc (_type_): 源的像素坐标位置
        voxel_loc (_type_): 肿瘤像素坐标位置
        png_path (_type_): _description_
    """
    # plot source and tumor location
    # if scale!=1:
    #     tumor_img = ants.resample_image(tumor_img,np.array(tumor_img.spacing)*scale,False,1)
    tumor_npy = tumor_img.numpy()#.transpose(1,0,2)
    # ct_array = ct_array.transpose(1,0,2)
    print(tumor_npy.shape)
    plt.figure(dpi=200)
    plt.subplot(1,3,1)
    plt.imshow(ct_array[:,:,voxel_loc[2]], cmap='gray')
    plt.imshow(tumor_npy[:,:,voxel_loc[2]], cmap='jet', alpha=0.5)
    y = [source_voxel_loc[0],voxel_loc[0]]
    x = [source_voxel_loc[1],voxel_loc[1]]
    plt.plot(x, y, color="red", linewidth=3)
    # plt.suptitle("cross section")

    plt.subplot(1,3,2)
    plt.imshow(ct_array[voxel_loc[0],:,:], cmap='gray')
    plt.imshow(tumor_npy[voxel_loc[0],:,:], cmap='jet', alpha=0.5)
    y = [source_voxel_loc[1],voxel_loc[1]]
    x = [source_voxel_loc[2],voxel_loc[2]]
    
    if x[0] == x[1] and y[0] == y[1]:
        plt.scatter(x[0],y[0], color="red",linewidths=3)
        print(f"scatter:{x[0]},{y[0]}")
    else:
        plt.plot(x, y, color="red", linewidth=3)
        print(f"plot:{x},{y}")
    # plt.suptitle("coronal section")

    plt.subplot(1,3,3)
    plt.imshow(ct_array[:,voxel_loc[1],:], cmap='gray')
    plt.imshow(tumor_npy[:,voxel_loc[1],:], cmap='jet', alpha=0.5)
    y = [source_voxel_loc[0],voxel_loc[0]] # [128, 256, 32]  128, 142, 32
    x = [source_voxel_loc[2],voxel_loc[2]]
    if x[0] == x[1] and y[0] == y[1]:
        plt.scatter(x[0],y[0], color="red",linewidths=3)
        print(f"scatter:{x[0]},{y[0]}")
    else:
        plt.plot(x, y, color="red", linewidth=3)
        print(f"plot:{x},{y}")
    # plt.suptitle("sagittal section")
    plt.savefig(png_path)
    # plt.show()
    # embed()
