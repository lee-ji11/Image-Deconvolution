from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2
import numpy as np
import argparse
import natsort
import os
import pandas as pd


Decon_dirname = 'G:/210626_databackup_deconvolution/범011/20210107_범011_7day/'
best_noise_dirname = 'bestimage/'
low_noise_dirname = 'lowimage/'

tlist = os.listdir(best_noise_dirname)
condition = '.jpg'
image_files = [file for file in tlist if file.endswith(condition)]
image_files = natsort.natsorted(image_files)

namelist, decon_low_ssimlist, best_low_ssimlist = [],[],[]

for ii in image_files:
    print(ii)
    namelist.append(str(ii))
    Decon = cv2.imdecode(np.fromfile(Decon_dirname+"/{}".format(ii), np.uint8),cv2.IMREAD_UNCHANGED)
    best_contrast = cv2.imdecode(np.fromfile(best_noise_dirname+"/{}".format(ii), np.uint8),cv2.IMREAD_UNCHANGED) # 노이즈 이미지
    low_contrast = cv2.imdecode(np.fromfile(low_noise_dirname+"/{}".format(ii), np.uint8),cv2.IMREAD_UNCHANGED) # 노이즈 이미지

    Decon1 = cv2.cvtColor(Decon, cv2.COLOR_BGR2GRAY)
    best = cv2.cvtColor(best_contrast, cv2.COLOR_BGR2GRAY)
    low = cv2.cvtColor(low_contrast, cv2.COLOR_BGR2GRAY)

    (decon_low_score, diff) = compare_ssim(Decon1, low, full=True)
    (best_low_score, diff) = compare_ssim(best, low, full=True)
    diff = (diff * 255).astype("uint8")

    print(f"decon_low_SSIM: {decon_low_score}")
    print(f"best_low_SSIM: {best_low_score}")
    decon_low_ssimlist.append(decon_low_score)
    best_low_ssimlist.append(best_low_score)

    # thresh = cv2.threshold(diff, 0, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for c in cnts:
    #     area = cv2.contourArea(c)
    #     if area > 40:
    #         x, y, w, h = cv2.boundingRect(c)
    #         cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #         cv2.drawContours(imageB, [c], -1, (0, 0, 255), 2)
    # imageA = cv2.resize(imageA,(0,0),fx=0.5,fy=0.5)
    # imageB = cv2.resize(imageB,(0,0),fx=0.5,fy=0.5)
    # cv2.imshow("Original", imageA)
    # cv2.imshow("Modified", imageB)
    # cv2.waitKey(0)

raw_data = {'Image name' : namelist,
        'Decon-LOW SSIM' : decon_low_ssimlist,
        'BEST-LOW SSIM' : best_low_ssimlist}
raw_data = pd.DataFrame(raw_data)
raw_data.to_excel(excel_writer='NEW_Cell_11_SSIM.xlsx', index=False)
