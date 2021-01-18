import os

import numpy as np

import skimage
from skimage import io, exposure

import warnings
from multiprocessing import Pool
from sklearn.cluster import KMeans  ### row col index fix

#####################################################################################
########################################Row and Column Index Update #################
def load_nanowells(a_center_fname):
    ''''''
    f = open(a_center_fname)
    lines = f.readlines()
    f.close()

    nanowells = []

    for line in lines:
        line = line.rstrip().split('\t')
        line = [float(i) for i in line]
        nanowells.append(line)

    return nanowells


def write_nanowells_new(a_center_fname_new, nanowells):

    f = open(a_center_fname_new, 'w')

    for nanowell in nanowells:
        temp = nanowell[0:5]
        temp = [int(temp[0]), int(temp[1]), int(temp[2]), float(temp[3]), int(temp[4])]
        temp = [str(i) for i in temp]
        temp = '\t'.join(temp) + '\n'
        f.writelines(temp)

    f.close()

    return 1


def sort_nanowell(a_center_fname, a_center_clean_fname, a_center_clean_sorted_fname, RC):
    '''
        This function will load the full nanowell file a_center.txt, create approximate grid using k-means;
        Sort nanowells in a_center_clean based on real Row and Column coordinates
    '''

    ### Step 1: load a_center.txt
    org_nanowells = load_nanowells(a_center_fname)

    clean_nanowells = load_nanowells(a_center_clean_fname)

    ### Step 2: get row_coords, col_coords
    row_margin = []
    col_margin = []

    for nanowell in org_nanowells:
        row_margin.append([nanowell[0], 0])
        col_margin.append([0,nanowell[1]])

    row_margin = np.array(row_margin)
    col_margin = np.array(col_margin)

    kmeans_row = KMeans(n_clusters=RC, random_state=0).fit(row_margin)
    kmeans_col = KMeans(n_clusters=RC, random_state=0).fit(col_margin)

    row_coords = kmeans_row.cluster_centers_
    col_coords = kmeans_col.cluster_centers_

    row_coords_flat = sorted([int(i[0]) for i in row_coords])
    col_coords_flat = sorted([int(j[1]) for j in col_coords])

    ### Step 3: update clean_nanowells --> clean_nanowells_RC
    index_nanowells = []
    for nanowell in clean_nanowells:
        R = 1
        d0 = 2048
        for i in range(RC):
            d = abs(nanowell[0]-row_coords_flat[i])
            if d<d0:
                d0 = d
                R = i + 1

        C = 1
        d0 = 2048
        for j in range(RC):
            d = abs(nanowell[1]-col_coords_flat[j])
            if d<d0:
                d0 = d
                C = j + 1

        index = RC*(R-1) + C
        nanowell.append(index)
        index_nanowells.append(nanowell)

    sorted_nanowells = sorted(index_nanowells, key=lambda l:l[4])

    ### Step 4: write clean_nanowells_RC to file a_center_sorted_fname

    write_nanowells_new(a_center_clean_sorted_fname, sorted_nanowells)

    return sorted_nanowells

#####################################################################################


def write_nanowell_info(fname, info_array):
    f = open(fname, 'w')
    for info in info_array:
        line = str(info[0]) + '\t' + str(info[1]) + '\t' + str(info[2]) + '\t' + format(info[3], '.4f') + '\n'
        f.writelines(line)
    f.close()


def scale_image_faster(img, clip_min, clip_max, gamma):
    img[img < clip_min] = 0
    img[img > clip_max] = clip_max

    min_value = 0
    # max_value = np.amax(img)
    max_value = clip_max

    if max_value > 0:
        img = img - min_value
        img = 256.0 / (max_value - min_value + 1) * img
        img[img > 255] = 255

    img = img.astype('uint8')

    if gamma == True:
        img = exposure.adjust_gamma(img, 0.8)

    return img


def CROP_IMAGES(Args):
    #called upon by CROP_IMAGES_BLOCK 
    fname_in = Args[0]
    OUTPUT_PATH = Args[1]
    DATASET = Args[2]
    BLOCK = Args[3]
    t = Args[4]
    CH = Args[5]
    meta_all_clean = Args[6]
    crop_8bit = Args[7]
    crop_16bit = Args[8]
    clip_min = Args[9]
    clip_max = Args[10]
    gamma = Args[11]

    img = io.imread(fname_in)
    # rescale the intensity of image to the full range of histogram for visualization
    img_corrected = exposure.rescale_intensity(img, in_range='image', out_range='dtype')

    folder_dir_8bit = OUTPUT_PATH + DATASET + '/' + BLOCK + '/images/crops_8bit_s/'
    folder_dir_16bit = OUTPUT_PATH + DATASET + '/' + BLOCK + '/images/crops_16bit_s/'
    folder_dir_vis = OUTPUT_PATH + DATASET + '/' + BLOCK + '/images/crops_vis/'

    #choose the correct name 
    total_nanowell = len(meta_all_clean)
    for idx in range(1, total_nanowell + 1):
        img_crop_fname_8bit = folder_dir_8bit + 'imgNo' + str(idx) + CH + '/imgNo' + str(idx) + CH + '_t' + str(
            t) + '.tif'
        img_crop_fname_16bit = folder_dir_16bit + 'imgNo' + str(idx) + CH + '/imgNo' + str(idx) + CH + '_t' + str(
            t) + '.tif'
        img_crop_fname_vis = folder_dir_vis + 'imgNo' + str(idx) + CH + '/imgNo' + str(idx) + CH + '_t' + str(
            t) + '.tif'

        #select the centroid of the sorted nanowell 
        x_center = int(meta_all_clean[idx - 1][0])
        y_center = int(meta_all_clean[idx - 1][1])

        #do the cropping and saving 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if crop_16bit == True:
                io.imsave(img_crop_fname_16bit, img[y_center - 140:y_center + 141, x_center - 140:x_center + 141])
            if crop_8bit == True:
                # io.imsave(img_crop_fname_8bit, scale_image_faster(img[y_center-140:y_center+141, x_center-140:x_center+141], clip_min, clip_max, gamma))
                io.imsave(img_crop_fname_8bit,
                          img_corrected[y_center - 140:y_center + 141, x_center - 140:x_center + 141])
            # save corrected (histogram stretched) images for visualization only
            io.imsave(img_crop_fname_vis, img_corrected[y_center - 140:y_center + 141, x_center - 140:x_center + 141])


def CROP_IMAGES_BLOCK(RAW_INPUT_PATH, OUTPUT_PATH, DATASET, BLOCK, FRAMES, Output_Type, channel_index_dict, CORES,
                      Nanowell_Size, Block_Size, CLIP_ARGS, GAMMA):
    # Clean Nanowell Information
    meta_fname = OUTPUT_PATH + DATASET + '/' + BLOCK + '/meta/a_centers.txt'
    f = open(meta_fname)
    meta_all = f.readlines()
    meta_all_clean = []
    for line in meta_all:
        temp = line.rstrip().split('\t')
        x_center = int(temp[0])
        y_center = int(temp[1])
        nanowell_class = int(float(temp[2]))
        nanowell_score = float(temp[3])
        if nanowell_class == 1:
            if (x_center - Nanowell_Size / 2) > 0 and (x_center + Nanowell_Size / 2) < Block_Size - 1:
                if (y_center - Nanowell_Size / 2) > 0 and (y_center + Nanowell_Size / 2) < Block_Size - 1:
                    meta_all_clean.append([x_center, y_center, nanowell_class, nanowell_score])

                    # write nanowell info
    meta_all_clean_fname = OUTPUT_PATH + DATASET + '/' + BLOCK + '/meta/a_centers_clean.txt'
    write_nanowell_info(meta_all_clean_fname, meta_all_clean)

    # Write the sorted nanowell info for the RC update: -----------------------------------------------------------------------------------------------
    meta_all_clean_sorted_fname = os.path.join(OUTPUT_PATH,DATASET,BLOCK) + '/meta/a_centers_clean_sorted.txt'
    RC = 6
    meta_all_clean_sorted = sort_nanowell(meta_fname, meta_all_clean_fname, meta_all_clean_sorted_fname, RC)
    meta_all_clean = meta_all_clean_sorted #replace with the sorted ones, crop and then move on. 

    #--------------------------------------------------------------------------------------------------------------------------------------------------

    # make directory for images
    total_nanowell = len(meta_all_clean)
    folder_dir = OUTPUT_PATH + DATASET + '/' + BLOCK + '/images/crops_8bit_s/'
    folder_dir_16bit = OUTPUT_PATH + DATASET + '/' + BLOCK + '/images/crops_16bit_s/'
    folder_dir_vis = OUTPUT_PATH + DATASET + '/' + BLOCK + '/images/crops_vis/'

    for idx in range(1, total_nanowell + 1):
        temp_folder_name_CH0 = folder_dir + '/imgNo' + str(idx) + 'CH0'
        temp_folder_name_CH1 = folder_dir + '/imgNo' + str(idx) + 'CH1'
        temp_folder_name_CH2 = folder_dir + '/imgNo' + str(idx) + 'CH2'
        temp_folder_name_CH3 = folder_dir + '/imgNo' + str(idx) + 'CH3'
        os.system('mkdir ' + temp_folder_name_CH0)
        os.system('mkdir ' + temp_folder_name_CH1)
        os.system('mkdir ' + temp_folder_name_CH2)
        os.system('mkdir ' + temp_folder_name_CH3)
        temp_folder_name_CH3 = folder_dir_16bit + '/imgNo' + str(idx) + 'CH3'
        os.system('mkdir ' + temp_folder_name_CH3)
        temp_folder_name_CH1 = folder_dir_16bit + '/imgNo' + str(idx) + 'CH1'
        os.system('mkdir ' + temp_folder_name_CH1)

        # create folders for visualization
        os.system('mkdir ' + folder_dir_vis + '/imgNo' + str(idx) + 'CH0')
        os.system('mkdir ' + folder_dir_vis + '/imgNo' + str(idx) + 'CH1')
        os.system('mkdir ' + folder_dir_vis + '/imgNo' + str(idx) + 'CH2')
        os.system('mkdir ' + folder_dir_vis + '/imgNo' + str(idx) + 'CH3')

    Parameter_List = []  # contains [fname_in, CH, meta_all_clean ,8bit?, 16bit?]

    for CH in Output_Type["uint8"]:
        for t in range(1, 1 + FRAMES):
            temp = []
            fname_in = get_fname_in(RAW_INPUT_PATH, OUTPUT_PATH, DATASET, BLOCK, CH, t)
            CH_Index = channel_index_dict[CH]
            crop_8bit = True
            if CH in Output_Type["uint16"]:
                crop_16bit = True
            else:
                crop_16bit = False
            temp.append(fname_in)
            temp.append(OUTPUT_PATH)
            temp.append(DATASET)
            temp.append(BLOCK)
            temp.append(t)
            temp.append(CH_Index)
            temp.append(meta_all_clean)
            temp.append(crop_8bit)
            temp.append(crop_16bit)

            temp.append(CLIP_ARGS[CH][0])
            temp.append(CLIP_ARGS[CH][1])

            if CH in GAMMA:
                gamma = True
                temp.append(gamma)
            else:
                gamma = False
                temp.append(gamma)

            Parameter_List.append(temp)

    pool = Pool(processes=CORES)
    pool.map(CROP_IMAGES, Parameter_List)
    pool.close()


def get_fname_in(RAW_INPUT_PATH, OUTPUT_PATH, DATASET, BLOCK, CH, t):
    BID = 's' + BLOCK[1:]  ### 1:4 generally, 2:4 occassionaly

    fname = OUTPUT_PATH + DATASET + '/' + BLOCK + '/temp/preprocess/' + DATASET + '_' + BID + 't' + str(t).zfill(
        2) + CH + '.tif'

    if os.path.isfile(fname) == False:
        fname = RAW_INPUT_PATH + DATASET + '/' + DATASET + '_' + BID + 't' + str(t).zfill(2) + CH + '.tif'

    return fname


def DT_CROP_IMAGES(RAW_INPUT_PATH, OUTPUT_PATH, DATASET, BLOCKS, FRAMES, Output_Type, channel_index_dict, CORES,
                   Nanowell_Size, Block_Size, CLIP_ARGS, GAMMA):
    for BLOCK in BLOCKS:
        print("CROPPING BLOCK " + BLOCK + " ......")
        CROP_IMAGES_BLOCK(RAW_INPUT_PATH, OUTPUT_PATH, DATASET, BLOCK, FRAMES, Output_Type, channel_index_dict, CORES,
                          Nanowell_Size, Block_Size, CLIP_ARGS, GAMMA)
