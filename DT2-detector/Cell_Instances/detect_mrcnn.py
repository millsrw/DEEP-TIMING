'''
Created by RWM: 3 aPRIL 2020
Last edit: 24 August 2020 
Version: 4

This script contains functions to test TIMING data with mrcnn 
The input is DT formatting 
The output is the mask detections in an output for mrcnn 

NOTES: 
    Implement mrcnn output similar to original DT 

To Do: 
    


'''
# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os, time
import sys
import json
import datetime
import numpy as np
import skimage.io as io
import skimage
# from imgaug import augmenters as iaa
# import cv2
import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR =os.path.abspath("../../")
ROOT_DIR = os.path.join(ROOT_DIR, 'DT2-detector', 'Cell_Instances')


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import os, time
import sys
import json
import datetime
import numpy as np
import skimage.io as io
import skimage
import numpy as np
import matplotlib.pyplot as plt

############################################################
#  Configurations
############################################################

class NucleusConfig(Config):

    def __init__(self, dataset):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

        self.dataset = dataset


    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 1  # Background + nucleus
    NUM_CLASSES = 1 + 3  # Background + 2 nucleus
    VAL_IMAGE_IDS = []


    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################
def initialise(dataset, DT_DATASET, blocks_list, LOGS_DIR, weights_path): 
    ''' Initialises DT inputs for applying MRCNN

    Args: 
        dataset (str): full path of dataset directory 
        DT_DATASET (str): particular dataset ID name 
        blocks_list (list): list of block ids e.g.: [B001, B002, B003]
        LOGS_DIR (str): path of logs directory that trained mrcnn 
        weights_path (str): location to weights for inference 
    '''
    # Root directory of the project
    import os
    import sys
    import random
    import math
    import re
    import time
    import numpy as np
    import tensorflow as tf
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import json
    import datetime
    import skimage.io as io
    import skimage
    # from imgaug import augmenters as iaa
    # import cv2
    from skimage.exposure import rescale_intensity
    from skimage import measure 


    # Import Mask RCNN
    #ROOT_DIR = os.path.abspath("../../")
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    from mrcnn.config import Config
    from mrcnn import visualize
    from mrcnn.visualize import display_images
    import mrcnn.model as modellib
    from mrcnn.model import log
    
    DATASET_DIR =dataset

    DEVICE = "/gpu:0"

    # Create model in inference mode
    config = NucleusInferenceConfig(dataset)
    config.display()
    model = modellib.MaskRCNN(mode="inference",
                                  config=config,
                                  model_dir=LOGS_DIR)
    #Load weights: 
    model.load_weights(weights_path, by_name=True)

    print('Running on dataset: ', DT_DATASET)

    detect(model, DATASET_DIR, DT_DATASET, blocks_list)

def make_DT_folder(path): 
    ''' Create folder architecture for mask images to mimic the original Deep Timing img folder 

    Args: 
        path (str): path to mask location
    '''
    channels = ['CH0', 'CH1', 'CH2', 'CH3'] #space for three classes CH3 will be for beads 
    for channel in channels: 
        if os.path.exists(os.path.join(path + channel)) == False: 
            os.mkdir(path + channel) #this will be like: B71/images/masks/imgNo9 then adding CH0 - CH3 to the end 
        else: 
            continue 


def make_folder(path, num): #generate mrcnn folder based on desired path and name =num
    ''' Generates a Mask R-CNN type folder based on the given path and number 

    Args: 
        path (str): string to path to create 
        num (str): name of channel ID 
    '''
    if os.path.exists(os.path.join(path,num))== False:
        os.mkdir(os.path.join(path,num))
        new_dir = os.path.join(path,num)
        if os.path.exists(os.path.join(new_dir,'images'))== False:
            os.mkdir(os.path.join(new_dir,'images'))
        if os.path.exists(os.path.join(new_dir,'masks'))== False:
            os.mkdir(os.path.join(new_dir,'masks'))

def blank_mask(mask,path):
    ''' Store a blank image mask (zero array) 

    Args: 
        mask (array): imge of size to generate mask 
        path (str): location to store mask  
    '''
    emp  = np.zeros(mask.shape)##np.full((mask.shape), False, dtype=bool)
    io.imsave(path,emp)  #save as a tif image 

def load_image(path):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image



def write_nanowell_info(fname, info_array): 
    ''' Write nanowell information (original TIMING code)

    Args: 
        fname (str): path to write location 
        info_array (list): list containing nanowell information from object detection 
    '''
    f = open(fname,'w')

    # Cell_ID	x	y	w	h	class	score
    for info in info_array:
        line = str(info[0]) + '\t' +str(info[1]) + '\t' + str(info[2]) + '\t' + str(info[3]) + '\t' +str(info[4]) + '\t' + format(info[5],'.1f') + '\t' + format(info[6],'.4f') + '\n'
        f.writelines(line)
    f.close()

def write_mask_info(fname, info_array): 
    ''' Write nanowell information (new MRCNN addition)

    Args: 
        fname (str): path to write location 
        info_array (list): list containing nanowell information from instance segmentation 
    '''
    f = open(fname,'w')

    # Cell_ID	centroidx, centroidy,	area	class	score
    for info in info_array:
        if True in np.isnan(info): 
            print(info)
        else: 
            line = str(int(info[0]))  + '\t' + str(int(info[1])) + '\t' +  str(int(info[2]))+'\t' +  str(int(info[3]))  + '\t' + format(info[4],'.1f') + '\t' + format(info[5],'.4f') + '\n'
            print(line)
            f.writelines(line)
    f.close()
    

def detect(model, dataset_dir, DT_DATASET, Blocks_list):
    ''' Perform instance segmentation using MRCNN 

    Args: 
        model (tf model): tensorflow model (trained)
        dataset_dir (str): path to dataset location 
        DT_DATASET (str): unique dataset ID 
        Blocks_list (list): list of block IDs e.g. [B001, B002...]
    '''
    from skimage.exposure import rescale_intensity
    from skimage.external import tifffile
    from skimage import measure 
    import shutil 

    dataset_name = str(DT_DATASET)

    # Load over images
    for block in Blocks_list:
        print("Processing Block =",block) #print to see if ch0 is working
        mask_path = os.path.join(dataset_dir, block, 'images', 'masks')
        if os.path.exists(mask_path) == False: 
            os.mkdir(mask_path)

        for img_dir in os.listdir(os.path.join(dataset_dir, block, 'images', 'crops_8bit_s')):
            if "CH0" in img_dir: 

                for lls_imgName in os.listdir(os.path.join(dataset_dir, block, 'images', 'crops_8bit_s', img_dir)): 
                    #go through the images

                    print('   Processing Image =' , lls_imgName) #this will be like: B71/images/masks/imgNo9CH0/imgNo9CH0_t73.tif

                    save_path = os.path.join(mask_path, img_dir.split('CH0')[0]) 
                    #Make folder for DT: 
                    make_DT_folder(save_path) #save files here for each channel 

                    name = lls_imgName.split('CH0')[1].split('.tif')[0] #this saves the time info 

                    source_id =os.path.join(os.path.join(dataset_dir, block, 'images', 'crops_8bit_s', img_dir, lls_imgName))

                    fixed_name = lls_imgName.split('.tif')[0] + '.jpeg' #save the image as jpeg so we can test correctly 
                    fixed_img =  np.array(io.imread(source_id))
                    #adjust the brightness of the image
                    p2, p98 = np.percentile(fixed_img, (0, 98))  
                    fixed_img= rescale_intensity(fixed_img,in_range=(p2, p98))
                    io.imsave( os.path.join(save_path + 'CH0',fixed_name), fixed_img, quality = 100) #save as jpeg

                    image = load_image(os.path.join(save_path + 'CH0',fixed_name)) #load image for running 

                    #--------------
                    # Detect objects
                    r = model.detect([image], verbose=0)[0]
                    # Encode image to RLE. Returns a string of multiple lines
                    rle = mask_to_rle(source_id, r["masks"], r["scores"])

                    #Let's get the binary mask of the prediction:
                    class_id=r['class_ids']
                    class_id=class_id.tolist()
                    mask_img=r["masks"]
                    class_scores = r['scores']
                    class_scores= class_scores.tolist() 
                    boxes = r['rois'] #these are the bounding boxes 
                    #--------------------do the detection ----------------------------------------------------------------------------------
                    NANOWELL_SIZE =  1 #281 #as set in DT parametrers 
    
                    meta_fname_prefix = dataset_dir + '/' + block + '/labels/DET/FRCNN-Fast/raw/'
                    os.system('mkdir ' + meta_fname_prefix)
                    meta_fname_prefix2 = dataset_dir + '/' + block + '/labels/DET/FRCNN-Fast/raw/' + img_dir.split('CH0')[0]
                    os.system('mkdir ' + meta_fname_prefix2)

                    meta_fname_prefix_mask = dataset_dir + '/' + block + '/labels/DET/MRCNN/'
                    os.system('mkdir ' + meta_fname_prefix_mask)
                    meta_fname_prefix_mask2 = dataset_dir + '/' + block + '/labels/DET/MRCNN/' + img_dir.split('CH0')[0]
                    # if os.path.exists(meta_fname_prefix2):
                    #     shutil.rmtree(meta_fname_prefix2) #remove prior detections. 
                    os.system('mkdir ' + meta_fname_prefix_mask2) 
                        
                    meta_fname = meta_fname_prefix +  img_dir.split('CH0')[0] + '/' +  img_dir.split('CH0')[0] + name + '.txt'
                    meta_array = []

                    meta_fname_mask = meta_fname_prefix_mask +  img_dir.split('CH0')[0] + '/' +  img_dir.split('CH0')[0] + name + '.txt'
                    meta_array_mask = []
                    MAXIMUM_CELL_DETECTED = 13 # 5 effectors, 5 targets, and 3 beads 
                    
                    for cell_count in range(MAXIMUM_CELL_DETECTED): #go through all the cells detected 
                        temp_meta = []
                        try:  
                            ymin = boxes[cell_count][0]
                            ymax = boxes[cell_count][2]
                            xmin = boxes[cell_count][1]
                            xmax = boxes[cell_count][3]
                            x = int((xmin)*NANOWELL_SIZE)
                            y = int((ymin)*NANOWELL_SIZE)
                            w = int((xmax-xmin)*NANOWELL_SIZE)
                            h = int((ymax-ymin)*NANOWELL_SIZE)
                            cell_class = class_id[cell_count]
                            cell_score = class_scores[cell_count]
                            temp_meta.append(cell_count)
                            temp_meta.append(x)
                            temp_meta.append(y)
                            temp_meta.append(w)
                            temp_meta.append(h)
                            temp_meta.append(cell_class)
                            temp_meta.append(cell_score)
                            meta_array.append(temp_meta)
                        except: 
                            temp_meta.append(cell_count)
                            temp_meta.append(0)
                            temp_meta.append(0)
                            temp_meta.append(0)
                            temp_meta.append(0)
                            temp_meta.append(1) #default set to class 1 
                            temp_meta.append(0)
                            meta_array.append(temp_meta)
                    # countt=0 #uncomment / modify the following lines to output other cell-specific feature calculations 
                    # counte=0
                    # for cell_count_mask in range (10): # 5 target and 5 effectors  
                    #     temp_meta_mask = [] 
                    #     try:
                    #         temp_mask = skimage.img_as_ubyte(mask_img)*1 #in case the mask is blank
                    #         temp_mask = skimage.transform.resize(temp_mask,  (NANOWELL_SIZE, NANOWELL_SIZE), anti_aliasing =True)
                    #         cell_class = class_id[cell_count_mask]
                    #         cell_score = class_scores[cell_count_mask]
                    #          #Calculate centroid and area from mask: 
                    #         M = measure.moments(temp_mask[:,:,cell_count_mask])
                    #         centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
                    #         area = M[0, 0]
                    #         # print('this is the centroid and area', centroid, area/2)
                    #         # area2=cv2.countNonZero((temp_mask[:,:,cell_count_mask]))
                    #         # print('The area could be: ', area2)
                    #         # print('shape of mask is: ',temp_mask[:,:,cell_count_mask].shape )
                    #         # print('\n')
                    #         #make one for the mask data and appended table 
                    #         temp_meta_mask.append(cell_count_mask)
                    #         temp_meta_mask.append(centroid[0]) #x
                    #         temp_meta_mask.append(centroid[1])#y
                    #         temp_meta_mask.append(area)
                    #         temp_meta_mask.append(cell_class)
                    #         temp_meta_mask.append(cell_score)
                    #         meta_array_mask.append(temp_meta_mask)
                    #         if int(cell_class) == 1: 
                    #             counte = counte+1
                    #         else: 
                    #             countt = countt+1 
                    #     except: 
                    #         temp_meta_mask.append(cell_count_mask) #only do this one time, otherwise it's excessive 
                    #         temp_meta_mask.append(0)
                    #         temp_meta_mask.append(0)
                    #         temp_meta_mask.append(0)
                    #         if counte < 5: 
                    #             temp_meta_mask.append(1)
                    #             counte = counte+1
                    #         else: 
                    #             temp_meta_mask.append(2)
                    #             countt = countt+1                                
                    #         temp_meta_mask.append(0)
                    #         meta_array_mask.append(temp_meta_mask)

                            
                    write_nanowell_info(meta_fname,meta_array)
                    # write_mask_info(meta_fname_mask, meta_array_mask) 

                    #--------------------generate the mask outputs: ------------------------------------------------------------------------
                    #print('There are ', len(class_id), ' masks to save. ')
                    if not class_id: # if there's not a detection 
                                blank_mask(image, os.path.join(save_path + 'CH1', img_dir.split('CH0')[0] + 'CH1' + name + '.png'))
                                blank_mask(image, os.path.join(save_path + 'CH2', img_dir.split('CH0')[0] + 'CH2' + name + '.png'))
                                blank_mask(image, os.path.join(save_path + 'CH3', img_dir.split('CH0')[0] + 'CH3' + name + '.png')) 
                    else: #if there is a detection 
                        mask_img = skimage.img_as_ubyte(mask_img)
                        if 1 not in class_id : 
                            blank_mask( mask_img[:,:,0]*1, os.path.join(save_path + 'CH1', 
                                img_dir.split('CH0')[0] + 'CH1' + name + '.png')) 
                        if 2 not in class_id :
                            blank_mask( mask_img[:,:,0]*1, os.path.join(save_path + 'CH2', 
                                img_dir.split('CH0')[0] + 'CH2' + name + '.png'))  
                        if 3 not in class_id: 
                            blank_mask( mask_img[:,:,0]*1, os.path.join(save_path + 'CH3', 
                                img_dir.split('CH0')[0] + 'CH3' + name + '.png'))  

                        mask_to_save = np.zeros((mask_img[:,:,0]*1).shape) #initialize mask to zeros 

                        for count, mask_per_class in enumerate (class_id): #go through the list of classes per mask
                            temp_mask = mask_img[:,:,count]*(int(count +1 )*0.25) #*int(colors[count]) #get the image 
                            #class_name =  lls_imgName.split('.tif')[0] + '.jpeg'
                            if count > 0: 
                                if mask_per_class == class_id[count-1] and class_scores[count] > 0.9: #make sure it's a reliable mask 
                                    indicies = temp_mask != 0
                                    mask_to_save[indicies] = temp_mask[indicies] 
                                elif mask_per_class != class_id[count-1] and class_scores[count] >0.9: 
                                    mask_to_save = temp_mask 
                            else: 
                                if class_scores[count] > 0.9: #again make sure the mask is reliable 
                                    mask_to_save=temp_mask 

                            save_path_mask = os.path.join(save_path + 'CH' + str(mask_per_class), 
                                img_dir.split('CH0')[0] + 'CH' + str(mask_per_class) + name + '.png')
                            io.imsave(save_path_mask, mask_to_save) #save as a tif file 
