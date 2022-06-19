import matplotlib.pyplot as plt
import scipy.io as scio
import nibabel as nb
import numpy as np
import os
import ants

class DVH(object):
    """Get DVH from the dose results of MC."""
    def __init__(self,
        pname = "hanjianying",
        mc_out_file_path="/home/zhaosheng/topas_4dct/csv",
        nii_path="/home/zhaosheng/topas_4dct/topas/niis",
        cache_path="/home/zhaosheng/topas_4dct/postprocess/cache/",
        redundancy=False,
        reload=False
        ):

        self.mc_out_file_path = mc_out_file_path
        self.cache_path = cache_path
        self.redundancy = redundancy
        self.nii_path = nii_path
        self.reload = reload
        os.makedirs(self.cache_path,exist_ok=True)
        t0_img_path = os.path.join(self.nii_path,f"{pname}_t0.nii")
        self.raw_img_t0  = ants.image_read(t0_img_path)
        self.ct = self.raw_img_t0.numpy()
        self.pname = pname

    def get_dose(self):
        dose_result = {
            "boron":[],
            "fast":[],
            "nitrogen":[],
            "gamma":[]
        }
        
        for t_index in [0]:#range(0,10):
            for dose_type in ["boron","fast","nitrogen","gamma"]:
                if self.reload or (not (os.path.exists(os.path.join(self.cache_path,f"{self.pname}_t{t_index}.nii")))):
                    dose_array = self._get_dose_array_from_csv(t_index)[dose_type]
                    
                    dose_result[dose_type].append(dose_array)
                    dose_img = ants.from_numpy(dose_array)
                    dose_img.set_spacing(self.raw_img_t0.spacing)
                    ants.image_write(dose_img,os.path.join(self.cache_path,f"{self.pname}_t{t_index}_{dose_type}.nii"))

                else:
                    dose_img = ants.image_read(os.path.join(self.cache_path,f"{self.pname}_t{t_index}_{dose_type}.nii"))
                    dose_result[dose_type].append(dose_img.numpy())
            
    def _array_from_MC(self,MC_OUT_FILE,mode="txt"):
        """Get a two-dimensional numpy array from the output file
           of Monte Carlo software.

        Args:
            MC_OUT_FILE ([str]): [The path to the output file]
        Returns:
            [array]: [The result of this output file in a two-dimensional array format]
        """
        if mode == "txt":
            if (not self.reload) and os.path.exists(MC_OUT_FILE[:-4]+".npy"):
                filename_ = MC_OUT_FILE[:-4]+".npy"
                # if self.redundancy:
                print(f"\t->Loading npy file : {filename_}")
                array_ = np.load(filename_)
                # if self.redundancy:
                print(f"\t->Shape:{array_.shape}")
                return array_
            # if self.redundancy:
            print(f"\t->Reading MC output file : {MC_OUT_FILE}")
            with open(MC_OUT_FILE,encoding = 'utf-8') as f:
                data = np.loadtxt(f,delimiter = ",", skiprows = 8)
            output = data[:,3].reshape(256,256,-1)
            np.save(MC_OUT_FILE[:-4],output)
            if self.redundancy:
                print(f"\t->Saving MC output file : {MC_OUT_FILE}")
                print(f"\t->Shape:{output.shape}")
            return output

        if mode == "mat":
            data = scio.loadmat(MC_OUT_FILE)
            return data[[ky for ky in data.keys() if "_" not in ky][0]].transpose(2,0,1)

    def _get_dose_array_from_csv(self,t_index):
        dose = {}
        dose["boron"] = self._array_from_MC(os.path.join(self.mc_out_file_path,f"{self.pname}_t{t_index}_boron10.csv"))
        dose["fast"] = self._array_from_MC(os.path.join(self.mc_out_file_path,f"{self.pname}_t{t_index}_fast10.csv"))
        dose["gamma"] = self._array_from_MC(os.path.join(self.mc_out_file_path,f"{self.pname}_t{t_index}_gamma10.csv"))
        dose["nitrogen"] = self._array_from_MC(os.path.join(self.mc_out_file_path,f"{self.pname}_t{t_index}_nitrogen10.csv"))
        # TODO
        
        return dose


if __name__ == "__main__":
    dvh = DVH(pname="caowujun",nii_path="/home/zhaosheng/calc/raw_data/caowujun",mc_out_file_path="/home/zhaosheng/calc/topas",cache_path="/home/zhaosheng/calc/postprocess/cache")
    print(dvh.get_dose())
