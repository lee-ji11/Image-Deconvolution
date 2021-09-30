import cv2
import numpy as np
import math
import pandas as pd
import os
import natsort

namelist, Decon_low_mselist, Decon_low_psnrlist, best_low_mselist, best_low_psnrlist = [],[],[],[],[]

Decon_dirname = 'full Location'
noise_dirname = 'Location/'
low_noise_dirname = 'Location/'

tlist = os.listdir(noise_dirname)
condition = '.jpg'
image_files = [file for file in tlist if file.endswith(condition)]
image_files = natsort.natsorted(image_files)

print(len(image_files))

for ii in image_files:
    print(ii)
    namelist.append(str(ii))
    Decon = cv2.imdecode(np.fromfile(Decon_dirname+"/{}".format(ii)),cv2.IMREAD_UNCHANGED)
    best_contrast = cv2.imdecode(np.fromfile(noise_dirname+"/{}".format(ii)),cv2.IMREAD_UNCHANGED) # 노이즈 이미지
    low_contrast = cv2.imdecode(np.fromfile(low_noise_dirname+"/{}".format(ii)),cv2.IMREAD_UNCHANGED) # 노이즈 이미지

    Decon = Decon.astype(np.float64) / 255.
    best_contrast = best_contrast.astype(np.float64) / 255.
    low_contrast = low_contrast.astype(np.float64) / 255.

    Decon_low_mse = np.mean((Decon-low_contrast)**2)
    Decon_low_mselist.append(Decon_low_mse)

    # Decon_low_psnr = cv2.PSNR(Decon, low_contrast)
    Decon_low_psnr = 10 * math.log10(1. / Decon_low_mse)
    Decon_low_psnrlist.append(Decon_low_psnr)

    best_low_mse = np.mean((best_contrast-low_contrast)**2)
    best_low_mselist.append(best_low_mse)

    # best_low_psnr = cv2.PSNR(best_contrast,low_contrast)
    best_low_psnr = 10 * math.log10(1. / best_low_mse)
    best_low_psnrlist.append(best_low_psnr)

    print(str(ii)+"Decon-low-mse: ", Decon_low_mse)
    print(str(ii)+'Decon-low-PSNR: ',Decon_low_psnr)
    print(str(ii)+"Decon-best-mse: ", best_low_mse)
    print("Decon-best-psnr: ",best_low_psnr)

raw_data = {'Image name' : namelist,
            'Decon-LOW MSE' : Decon_low_mselist,
            'Decon-LOW PSNR' : Decon_low_psnrlist,
            'BEST-LOW MSE' : best_low_mselist,
            'BEST-LOW PSNR' : best_low_psnrlist}
raw_data = pd.DataFrame(raw_data)
raw_data.to_excel(excel_writer='NEW_Cell_21_PSNR.xlsx', index=False)
