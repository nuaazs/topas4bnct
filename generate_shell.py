import os
root = "./raw_data"
command = "echo 'start!' "
pnames = [pname for pname in os.listdir(root) if "nohup" not in pname]

for pname in pnames:
    p_path = os.path.join(root,pname)
    niis = sorted([os.path.join(p_path,_file) for _file in os.listdir(p_path) if "gtv" not in _file and ".nii" in _file])
    # print(niis)
    seg = [os.path.join(p_path,_file) for _file in os.listdir(p_path) if "gtv" in _file and ".nrrd" in _file][0]
    for nii in niis:
        command+=f"&& python3 get_input.py --nii {nii} --root ./topas --scale 2 --tumor_seg_nii {seg} "
print(command)
