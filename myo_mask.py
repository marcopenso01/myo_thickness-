"""
Created on Thu Oct  7 09:56:34 2021

@author: Marco Penso
"""

import scipy
import scipy.io
import os
import numpy as np
import logging
import h5py
from skimage import transform
import pydicom
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,",",y)
        X.append(y)
        Y.append(x)
        cv2.destroyAllWindows()

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def setDicomWinWidthWinCenter(vol_data, winwidth, wincenter):
    vol_temp = np.copy(vol_data)
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)
    
    vol_temp = ((vol_temp[:]-min)*dFactor).astype('int16')

    min_index = vol_temp < 0
    vol_temp[min_index] = 0
    max_index = vol_temp > 255
    vol_temp[max_index] = 255

    return vol_temp


input_folder = r'F:\CT-tesi\Segmentation\1'
output_file = os.path.join(input_folder, 'pre_proc_all.hdf5')
hdf5_file = h5py.File(output_file, "w")

pat_addrs = []
               
mat = scipy.io.loadmat(os.path.join(input_folder, 'ART1.mat'))
vol_art = mat['ART1']

vol_art = vol_art -1024

vol_art_transp = vol_art.transpose([2,0,1])

vol_art_flip = flip_axis(vol_art_transp,1)

vol_bas_win = setDicomWinWidthWinCenter(vol_art_flip, 300, 150)

n =  vol_art_flip.shape[0]
first_sl = 400
last_sl = 560
for i in range(first_sl, last_sl, 10):
  fig = plt.figure()
  plt.title(i)
  plt.imshow(vol_bas_win[i,...])
  plt.show()


vol_art_flip = vol_art_flip[first_sl:last_sl,...]
vol_bas_win = vol_bas_win[first_sl:last_sl,...]

X = []
Y = []
a = vol_bas_win[10,...]
a = a[...,np.newaxis]
b = np.concatenate((a,a,a,), axis=-1)
cv2.imshow("image", b.astype('uint8'))
cv2.namedWindow('image')
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)

a = vol_bas_win[100,...]
a = a[...,np.newaxis]
b = np.concatenate((a,a,a,), axis=-1)
cv2.imshow("image", b.astype('uint8'))
cv2.namedWindow('image')
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)

x = abs(int((X[0]+X[1])/2))
y = abs(int((Y[0]+Y[1])/2))

FWHM = 110
art = vol_art_flip[:,X[0]-FWHM:X[0]+FWHM, Y[0]-FWHM:Y[0]+FWHM]
win = vol_bas_win[:,X[0]-FWHM:X[0]+FWHM, Y[0]-FWHM:Y[0]+FWHM]

print(vol_art_flip.shape, vol_art_flip.dtype, vol_art_flip.min(), vol_art_flip.max())
print(vol_bas_win.shape, vol_bas_win.dtype, vol_bas_win.min(), vol_bas_win.max())

for i in range(0, vol_bas_win.shape[0], 10):
  fig = plt.figure()
  plt.title(i)
  plt.imshow(vol_bas_win[i,...])
  plt.show()

num_slices = vol_art_flip.shape[0]
size = vol_art_flip.shape[1:3]

dt = h5py.special_dtype(vlen=str)
hdf5_file.create_dataset('ART', [num_slices] + list(size), dtype=np.int16)
hdf5_file.create_dataset('ARTwin', [num_slices] + list(size), dtype=np.uint8)
hdf5_file.create_dataset('paz', (num_slices,), dtype=dt)

hdf5_file['ART'][:] = vol_art_flip[None]
hdf5_file['ARTwin'][:] = vol_bas_win[None]
for i in range(num_slices):
    hdf5_file['paz'][i, ...] = input_folder.split('\\')[-1]

hdf5_file.close()





# -------------------------------------------------------------------------
# ----------------------- segmenting myo ----------------------------------

drawing=False # true if mouse is pressed
mode=True

def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y
  
def imfill(img, dim):
    img = img[:,:,0]
    img = cv2.resize(img, (dim, dim))
    img[img>0]=255
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)


path = r'F:\CT-tesi\Segmentation\2'


output_file = os.path.join(path, 'seg_myo.hdf5')
hdf5_file = h5py.File(output_file, "w")

file = os.path.join(path, 'pre_proc.hdf5')
data = h5py.File(file, 'r')
art = data['ART'][()]

art_img = []
mask_myo = []
mask_epi = []
mask_endo = []
tit=['epicardium', 'endocardium']
print('len', len(art))

for i in range(0, len(art), 4):
    img = art[i]
    art_img.append(img)
    print("{}/{}".format(i, len(art)))
    for ii in range(2):

        dim = img.shape[0]

        img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)    
        clahe = cv2.createCLAHE(clipLimit = 1.5)
        img = clahe.apply(img)
        img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_CUBIC)

        image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

        cv2.namedWindow(tit[ii])
        cv2.setMouseCallback(tit[ii],paint_draw)
        while(1):
            cv2.imshow(tit[ii],img)
            k=cv2.waitKey(1)& 0xFF
            if k==27: #Escape KEY
                if ii==0:
                    
                    im_out1 = imfill(image_binary, dim)
                    im_out1[im_out1>0]=255
                    #fig = plt.figure()
                    #plt.imshow(im_out1)
                    #plt.show()
                    
                elif ii==1:
                                            
                    im_out2 = imfill(image_binary, dim)
                    im_out2[im_out2>0]=255
                    #fig = plt.figure()
                    #plt.imshow(im_out2)
                    #plt.show()
                break
        cv2.destroyAllWindows()
        
    im_out1[im_out1>0]=1
    im_out2[im_out2>0]=1
    mask = im_out1 - im_out2
    mask_epi.append(im_out1)
    mask_endo.append(im_out2)
    mask_myo.append(mask)
    plt.figure()
    plt.imshow(mask)

num_slices = len(mask_myo)
size = mask_myo[0].shape
dt = h5py.special_dtype(vlen=str)
hdf5_file.create_dataset('art_img', [num_slices] + list(size), dtype=np.int16)
hdf5_file.create_dataset('mask_myo', [num_slices] + list(size), dtype=np.uint8)
hdf5_file.create_dataset('mask_end', [num_slices] + list(size), dtype=np.uint8)
hdf5_file.create_dataset('mask_epi', [num_slices] + list(size), dtype=np.uint8)
hdf5_file.create_dataset('paz', (len(mask_myo),), dtype=dt)

for ii in range(len(mask_myo)):
    hdf5_file['art_img'][ii] = art_img[ii]
    hdf5_file['mask_myo'][ii] = mask_myo[ii]
    hdf5_file['mask_end'][ii] = mask_end[ii]
    hdf5_file['mask_epi'][ii] = mask_epi[ii]
    hdf5_file['paz'][ii] = path.split("\\")[-1]
    
data.close()
hdf5_file.close()
