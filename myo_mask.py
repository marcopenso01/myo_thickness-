import os
import numpy as np
import h5py
import cv2
import pydicom # for reading dicom files
import matplotlib.pyplot as plt


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def imfill(img):
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)


def crop_or_pad_slice_to_size(slice, nx, ny):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
    
    for i in range(len(stack)):
        
        img = stack[i]
            
        x, y = img.shape
        
        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2
    
        if x > nx and y > ny:
            slice_cropped = img[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(stack)>1:
            RGB.append(slice_cropped)
    
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped
    
    
def crop_or_pad_slice_to_size_specific_point(slice, nx, ny, cx, cy):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
        
    for i in range(len(stack)):
        img = stack[i]
        x, y = img.shape
        y1 = (cy - (ny // 2))
        y2 = (cy + (ny // 2))
        x1 = (cx - (nx // 2))
        x2 = (cx + (nx // 2))
    
        if y1 < 0:
            img = np.append(np.zeros((x, abs(y1))), img, axis=1)
            x, y = img.shape
            y1 = 0
        if x1 < 0:
            img = np.append(np.zeros((abs(x1), y)), img, axis=0)
            x, y = img.shape
            x1 = 0
        if y2 > 512:
            img = np.append(img, np.zeros((x, y2 - 512)), axis=1)
            x, y = img.shape
        if x2 > 512:
            img = np.append(img, np.zeros((x2 - 512, y)), axis=0)
    
        slice_cropped = img[x1:x1 + nx, y1:y1 + ny]
        if len(stack)>1:
            RGB.append(slice_cropped)
        
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped

    
    
def generator_mask(img, green_pixels):
    
    mask_epi = imfill(green_pixels)
    
    if len(np.argwhere(mask_epi)) == len(np.argwhere(green_pixels)):
        val = 11
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
        result = cv2.morphologyEx(green_pixels, cv2.MORPH_CLOSE, kernel)
        while len(np.argwhere(result)) == len(np.argwhere(imfill(result))):
            val += 2
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
            result = cv2.morphologyEx(green_pixels, cv2.MORPH_CLOSE, kernel)
        mask_epi = imfill(result)
        
    red_pixels = cv2.inRange(img, (110, 0, 0), (255, 100, 100))
    
    if len(np.argwhere(red_pixels)) > 5:
        
        if len(np.argwhere(red_pixels)) != len(np.argwhere(imfill(red_pixels))):
            
            mask_endo = imfill(red_pixels)  #mask LV
        else:
            val = 11
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
            result = cv2.morphologyEx(red_pixels, cv2.MORPH_CLOSE, kernel)
            while len(np.argwhere(result)) == len(np.argwhere(imfill(result))):
                val += 2
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
                result = cv2.morphologyEx(red_pixels, cv2.MORPH_CLOSE, kernel)
            mask_endo = imfill(result)
        
        if len(np.argwhere(green_pixels)) == len(np.argwhere(imfill(green_pixels))):
            mask_epi = imfill(mask_endo+mask_epi)

        mask_myo = mask_epi - mask_endo  #mask Myo
        
        mask_myo[mask_myo>0]=1
    
    return mask_myo


def prepare_data(input_folder, output_file, nx, ny):
    
    hdf5_file = h5py.File(output_file, "w")
    
    # 1: 'RV', 2: 'Myo', 3: 'LV'
    addrs = []
    MASK = []
    IMG_SEG = []  #img in uint8 con segmentazione
    IMG_RAW = []  #img in float senza segmentazione

    path_seg = os.path.join(input_folder, 'seg')
    path_seg = os.path.join(path_seg, os.listdir(path_seg)[0])

    path_raw = os.path.join(input_folder, 'raw')
    path_raw = os.path.join(path_raw, os.listdir(path_raw)[0])


    for i in range(len(os.listdir(path_seg))):
        dcmPath = os.path.join(path_seg, os.listdir(path_seg)[i])
        data_row_img = pydicom.dcmread(dcmPath)
        img = data_row_img.pixel_array
        img = crop_or_pad_slice_to_size(img, 390, 390)
        temp_img = img.copy()
        for r in range(0, img.shape[0]):
            for c in range(0, img.shape[1]):
                if img[r,c,0] == img[r,c,1] == img[r,c,2]:
                    temp_img[r,c,:]=0

        green_pixels = cv2.inRange(temp_img, (0, 110, 0), (125, 255, 125))

        if len(np.argwhere(green_pixels)) > 5:
            
            green_pixels2 = green_pixels.copy()
            for xx in range(len(np.argwhere(green_pixels))):
                coord = np.argwhere(green_pixels)[xx]
                if temp_img[coord[0],coord[1],0] == temp_img[coord[0],coord[1],1]:
                    green_pixels2[coord[0],coord[1]] = 0
            
            final_mask = generator_mask(temp_img, green_pixels2)
            
            print(i+1, final_mask.max(), final_mask.min())

            if final_mask.max() > 1:
                print('ERROR: max value of the mask %d is %d' % (i+1, final_mask.max()))
                plt.figure()
                plt.imshow(final_mask)
                plt.title('error mask %d' % (i+1));
                final_mask[final_mask>1]=1
                plt.figure()
                plt.imshow(final_mask)
                plt.title('corrected mask %d' % (i+1));
            MASK.append(final_mask)
            addrs.append(dcmPath)
            IMG_SEG.append(img)

            # save data raw
            dcmPath = os.path.join(path_raw, os.listdir(path_raw)[i])
            data_row_img = pydicom.dcmread(dcmPath)
            img = data_row_img.pixel_array
            img = crop_or_pad_slice_to_size(img, 390, 390)
            IMG_RAW.append(img)
                
    CX = []
    CY = []
    LEN_X = []
    LEN_Y = []
    for ii in range(0,7):
        
        img = MASK[ii]
        index = img > 0
        a = img.copy()
        a[index] = 1
        #plt.imshow(a)
        contours, hier = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_left_x = 1000
        top_left_y = 1000
        bottom_right_x = 0
        bottom_right_y = 0
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            if x < top_left_x:
                top_left_x = x
            if y < top_left_y:
                top_left_y= y
            if x+w-1 > bottom_right_x:
                bottom_right_x = x+w-1
            if y+h-1 > bottom_right_y:
                bottom_right_y = y+h-1        
        top_left = (top_left_x, top_left_y)
        bottom_right = (bottom_right_x, bottom_right_y)
        #print('top left=',top_left)
        #print('bottom right=',bottom_right)
        cx = int((top_left[1]+bottom_right[1])/2)   #row
        cy = int((top_left[0]+bottom_right[0])/2)   #column
        len_x = int(bottom_right[1]-top_left[1]) +5
        len_y = int(bottom_right[0]-top_left[0]) +5
        #print(len_x, len_y)
        CX.append(cx)
        CY.append(cy)
        LEN_X.append(len_x)
        LEN_Y.append(len_y)
        
        '''
        # plot crop region
        for i in range(top_left[0],bottom_right[0]+1):
            a[top_left[1]-1,i]=1
        for i in range(top_left[0],bottom_right[0]+1):
            a[bottom_right[1]+1,i]=1
        for i in range(top_left[1],bottom_right[1]+1):
            a[i, top_left[0]-1]=1
        for i in range(top_left[1],bottom_right[1]+1):
            a[i, bottom_right[0]+1]=1
        plt.figure()
        plt.imshow(a)
        '''
    cx = int(np.asarray(CX[:]).mean())
    cy = int(np.asarray(CY[:]).mean())
    len_x = int(np.asarray(LEN_X).max())
    len_y = int(np.asarray(LEN_Y).max())
    
    len_max = max(len_x, len_y)
    
    print(len_max)
    
    for i in range(len(IMG_SEG)):
        
        if len_max+34 < nx and len_max+34 < ny:
            IMG_SEG[i] = crop_or_pad_slice_to_size_specific_point(IMG_SEG[i], nx, ny, cx, cy)
            IMG_RAW[i] = crop_or_pad_slice_to_size_specific_point(IMG_RAW[i], nx, ny, cx, cy)
            MASK[i] = crop_or_pad_slice_to_size_specific_point(MASK[i], nx, ny, cx, cy)
        else:
            tt = 34
            IMG_SEG[i] = crop_or_pad_slice_to_size_specific_point(IMG_SEG[i], len_max+tt, len_max+tt, cx, cy)
            IMG_RAW[i] = crop_or_pad_slice_to_size_specific_point(IMG_RAW[i], len_max+tt, len_max+tt, cx, cy)
            MASK[i] = crop_or_pad_slice_to_size_specific_point(MASK[i], len_max+tt, len_max+tt, cx, cy)
            
            IMG_SEG[i] = cv2.resize(IMG_SEG[i], (nx, ny), interpolation=cv2.INTER_AREA)
            IMG_RAW[i] = cv2.resize(IMG_RAW[i], (nx, ny), interpolation=cv2.INTER_AREA)
            MASK[i] = cv2.resize(MASK[i], (nx, ny), interpolation=cv2.INTER_NEAREST)
        '''
        plt.figure()
        plt.imshow(IMG_SEG[i])
        plt.title(i)
        plt.figure()
        plt.imshow(MASK[i])
        plt.title(i)
        '''
    dt = h5py.special_dtype(vlen=str)
    hdf5_file.create_dataset('paz', (len(addrs),), dtype=dt)
    hdf5_file.create_dataset('num_img', (len(addrs),), dtype=dt)
    hdf5_file.create_dataset('mask', [len(addrs)] + [nx, ny], dtype=np.uint8)
    hdf5_file.create_dataset('img_seg', [len(addrs)] + [nx, ny, 3], dtype=np.uint8)
    hdf5_file.create_dataset('img_raw', [len(addrs)] + [nx, ny], dtype=np.float32)

    
    for i in range(len(addrs)):
         hdf5_file['paz'][i, ...] = addrs[i].split("\\")[0].split("/")[-1]
         hdf5_file['num_img'][i, ...] = addrs[i].split("\\")[-1]
         hdf5_file['mask'][i, ...] = MASK[i]
         hdf5_file['img_seg'][i, ...] = IMG_SEG[i]
         hdf5_file['img_raw'][i, ...] = IMG_RAW[i]
    
    # After loop:
    hdf5_file.close()
    

def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                nx,
                                ny):
        
    '''
    This function is used to load and if necessary preprocesses the dataset
    
    :param input_folder: Folder where the raw data is located 
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :parm nx, ny: crop size
    
    :return: Returns an h5py.File handle to the dataset
    '''  
    data_file_name = 'maskMyo.hdf5'

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path):

        print('This configuration of mode, size and target resolution has not yet been preprocessed')
        print('Preprocessing now!')
        prepare_data(input_folder, data_file_path, nx, ny)

    else:

        print('Already preprocessed this configuration. Loading now!')

    #return h5py.File(data_file_path, 'r')


if __name__ == '__main__':

    # Paths settings
    input_folder = r'F:/Myo_thickness/MR/train'
    nx = 160
    ny = 160
    for file in os.listdir(input_folder):
        path = os.path.join(input_folder, file)
        if not os.path.exists(os.path.join(path, 'pre_proc')):
            print(path)
            preprocessing_folder = os.path.join(path, 'pre_proc')
    
            d=load_and_maybe_process_data(path, preprocessing_folder, nx, ny)
