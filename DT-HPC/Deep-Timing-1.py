# coding: utf-8

# In[1]:


# Import Packages
import sys, time, os 
import numpy as np
import shutil
from PIL import Image
from skimage.io import imsave, imread
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import disk
from numpy import zeros
from skimage import img_as_ubyte


# In[2]:

# Include PATHS
DEEP_TIMING_HOME = '/project/roysam/rwmills/NavinLab/DEEP-TIMING/'

MRCNN_HOME = DEEP_TIMING_HOME + 'DT2-detector/Cell_Instances/'

sys.path.append(DEEP_TIMING_HOME + 'DT1-preprocessor/')

sys.path.append(DEEP_TIMING_HOME + 'DT2-detector/Well/')
sys.path.append(DEEP_TIMING_HOME + 'DT2-detector/Well/faster-rcnn/')

sys.path.append(DEEP_TIMING_HOME + 'DT2-detector/Cell')

sys.path.append(DEEP_TIMING_HOME + 'DT2-detector/Cell_Instances/')

sys.path.append(DEEP_TIMING_HOME + 'DT3-tracker/')

sys.path.append(DEEP_TIMING_HOME + 'DT4-feature/')

# In[3]:

seg= "segmentation" # set segmentation to y or n to call on mrccn or not 

# STEP 0: Specify Parameters for experiments
CORES = 10

# DATASET = "20161215_MM_02_MotileTar"

# DATASET = "20161216_MM_04_ControlTar"

# DATASET = "20170330_MM_02_Nalm6"

DATASET = "dataid"

# RAW_INPUT_PATH = "/brazos/varadarajan//project/roysam/rwmills/TIMING/Lysosomes/datasets/Killing_events/20180720_RR_CART_NALM6_TIFF_Blocks35/"
# OUTPUT_PATH = "/brazos/varadarajan/rwmills/"

# RAW_INPUT_PATH = "/brazos/varadarajan//project/roysam/rwmills/TIMING/Lysosomes/datasets/Killing_events/" #"/project/roysam//project/roysam/rwmills/TIMING/Lysosomes/datasets/Killing_events/"
# OUTPUT_PATH = "/project/roysam/rwmills/TIMING/DEEP-TIMING_rachel/result/"
RAW_INPUT_PATH = "projdir"
OUTPUT_PATH = "resultdir"

microscope = 'zeiss'
#CH0 : phase_contrast
#CH1 : effectors
#CH2 : targets
#CH3 : death
channel_index_dict = {"c1_ORG":"CH0", "c2_ORG":"CH3", "c3_ORG":"CH2", "c4_ORG":"CH1"}
channel_name_dict = {"c1_ORG":"phase_contrast", "c2_ORG":"death", "c3_ORG":"targets", "c4_ORG":"effectors"}



# Input and Output Data Type:
Input_Type = "uint16" # or "uint8"
Output_Type = {"uint8": ['c1_ORG', 'c2_ORG', 'c3_ORG', 'c4_ORG'], "uint16":['c2_ORG',]}
#Output_Type = {"uint8": ['c3_ORG',], "uint16":['c3_ORG',]}

# GAMMA = ['c2_ORG']
GAMMA = []

NUM_DIGITS = 3
BLOCKS = ['B'+str(melisa).zfill(NUM_DIGITS)]
MAX_NANOWELL_PER_BLOCK = 36
FRAMES = numframes
Block_Size = 2048
Nanowell_Size = 281




# A. Preprocessing Pipeline
UMX_Channel = [['c3_ORG','c2_ORG', 0.95], ['c4_ORG', 'c3_ORG', 0.2]]   # Args0 - Args1 * Args2
BKG_Channel = []
# ENHANCE_Channel = ['c2_ORG', 'c3_ORG', 'c4_ORG']

# B. Cell Detector Config
Channel_Mix = ['CH0','CH1','CH2'] # Mix Channels for cell detection, e.g. CH0+CH1+CH2
Cell_Detector_Type = 'FRCNN-Fast' # Other Options include [1]'FRCNN-Slow' or [2]'SSD'

# C. Cell Tracker Config
Cell_Tracker_Type = 'EZ' # Other options include [1]'EZ'

# D. Feature Calculation Config
Effector_Max_Number = 4
Target_Max_Number = 4
Effector_Feature_List = ['x', 'y', 'AR', 'SPEED', 'DM']
Target_Feature_List = ['x', 'y', 'AR', 'SPEED', 'CR', 'DM']
CNN_LSTM_DM = 'True'


# In[ ]:


# Initialize the output folders
from DT_Init import *

t1 = time.time()

DATASET_PATH = OUTPUT_PATH + DATASET + '/'

DT_Initializer(DATASET_PATH, BLOCKS, CORES) #RWM CHANGE 

print("Initialization Time Cost: " + str(time.time() - t1 ))


# In[ ]:


# from DT_Init import *

# DATASET_PATH = OUTPUT_PATH + DATASET + '/'

# DT_Reset(DATASET_PATH)


# In[ ]:


# STEP 1: Detect Nanowells
# Get the sample CH0 Image from each Block, Convert it to 8 bit
from generate_CH0_Sample import *

t1 = time.time()

generate_CH0_samples_master(RAW_INPUT_PATH, OUTPUT_PATH, DATASET, BLOCKS, microscope, CORES) #RWM CHANGE 

print("CH0 Sample Preparation Time Cost: " + str(time.time() - t1 ))


# In[ ]:


# Run Nanowell Detection with each Image, save the results
from nanowell_detector import detect_nanowells

t1 = time.time()

PATH_TO_CKPT=DEEP_TIMING_HOME + 'DT2-detector/Well/faster-rcnn/experiment2/models/frozen_inference_graph.pb'
PATH_TO_LABELS=DEEP_TIMING_HOME + 'DT2-detector/Well/faster-rcnn/experiment2/data/TIMING_nanowell_detection.pbtxt'
PATH_TO_OUTPUT_DIR = OUTPUT_PATH + DATASET + '/'

detect_nanowells(PATH_TO_CKPT, PATH_TO_LABELS, PATH_TO_OUTPUT_DIR, BLOCKS, Nanowell_Number=36, BLOCK_SIZE=2048) #RWM CHANGE 

print("CH0 Sample Preparation Time Cost: " + str(time.time() - t1 ))


# In[ ]:


# STEP 2.1: Preprocessing Steps if necessary(umx, bg, enhance)
from DT_Preprocessor import *

t1 = time.time()

DT_UNMIX(DEEP_TIMING_HOME, RAW_INPUT_PATH, OUTPUT_PATH, DATASET, BLOCKS, FRAMES, UMX_Channel, CORES)

print("STEP-2 UNMIXING TIME COST: " + str(time.time() - t1))


# In[ ]:


# STEP 2.2: ENHANCE Steps
# min_clip_value = 315
# max_clip_value = 1000
# min_pixel_value = 0
# max_pixel_value = 2000


# ENHANCE_Channel = ["c2_ORG"]
# ENHANCE_Parameter = [min_pixel_value, max_pixel_value, min_clip_value, max_clip_value]

# from DT_Preprocessor import *

# t1 = time.time()

# DT_CLIP_ENHANCE(DEEP_TIMING_HOME, RAW_INPUT_PATH, OUTPUT_PATH, DATASET, BLOCKS, FRAMES, ENHANCE_Channel, ENHANCE_Parameter,CORES)

# print("STEP-2.2 ENHANCE TIME COST: " + str(time.time() - t1))


# In[ ]:


# STEP 3: Crop the images, put them in Bxxx/images/crop_8bit_s or Bxxx/images/crop_16bit_s
# from nanowell_cropper import *
from nanowell_cropper import * #this is the new file

t1 = time.time()

CLIP_ARGS = {"c1_ORG":[2000, 20000],"c2_ORG":[160, 180],"c3_ORG":[100, 800],"c4_ORG":[250, 1000]}

DT_CROP_IMAGES(RAW_INPUT_PATH, OUTPUT_PATH, DATASET, BLOCKS, FRAMES, Output_Type, channel_index_dict, CORES, Nanowell_Size, Block_Size, CLIP_ARGS, GAMMA) #RWM CHANGE for RC update 
#row and column update: 
# crop_nanowells(DEEP_TIMING_HOME,DATASET,RAW_INPUT_PATH,OUTPUT_PATH,BLOCKS,FRAMES,Nanowell_Size,Block_Size,channel_index_dict) #rwm

print("STEP-3 IMAGE CROP TIME COST: " + str(time.time() - t1))


# In[ ]:


# STEP 4.1: Cell Detection, including Channel mixing, cell detection, bboxes cleaning
if seg == 'n' or seg=='no': # if we choose not to do segmentation 
    from DT_Cell_Detector import *

    CH_MIX = ['CH0', 'CH1', 'CH2']
    Detector_Type = 'FRCNN-Fast'
    MAXIMUM_CELL_DETECTED = 10

    t1 = time.time()

    DT_Cell_Detector(DEEP_TIMING_HOME, OUTPUT_PATH, DATASET, BLOCKS, FRAMES, CH_MIX, Detector_Type, MAXIMUM_CELL_DETECTED, Nanowell_Size, CORES)

elif seg =='y' or seg =='yes': #we choose to do segmentation 
    logs = MRCNN_HOME + 'weights/' #denote log directory 
    #load the weights: if you change experiments, this weight file must change
    weights =os.path.join(MRCNN_HOME, 'weights', 'mask_rcnn_nucleus_0081.h5')  

    from detect_mrcnn import * # import all functions from this script 

    t1 = time.time()

    initialise( os.path.join(OUTPUT_PATH, DATASET), DATASET, BLOCKS, logs, weights)

    os.chdir(DEEP_TIMING_HOME)
    
print("STEP-4 CELL DETECTION TIME COST: " + str(time.time() - t1))

# In[ ]:


# STEP 4.2: Cell Detection Confinement Constraint
from DT_Cell_Detection_Cleaner import *

t1 = time.time()

CC_THRESHOLD = 0.70
SCORE_THRESHOLD = 0.70
MAX_E_COUNT = 5
MAX_T_COUNT = 5
Detector_Type = 'FRCNN-Fast'

DT_Cell_Cleaner(OUTPUT_PATH, DATASET, BLOCKS, FRAMES, CC_THRESHOLD, SCORE_THRESHOLD, MAX_E_COUNT, MAX_T_COUNT, Detector_Type, CORES)

print("STEP-4.2 CELL DETECTION CLEANING TIME COST: " + str(time.time() - t1))


# In[ ]:


# # STEP 5.2: Cell Tracking, options for EZ_track and SIAMESE_track
from DT_EZ_Tracker import *

Cell_Tracker_Type = 'EZ'

DETECTOR_TYPE = Cell_Detector_Type
TRACKER_TYPE = Cell_Tracker_Type

t1 = time.time()

if Cell_Tracker_Type == 'EZ':
    for BLOCK in BLOCKS:
        print('CELL TRACKING for ' + BLOCK + '......')
        DT_EZ_TRACKER(OUTPUT_PATH, DATASET, BLOCK, FRAMES, DETECTOR_TYPE, TRACKER_TYPE, CORES)
    
print("STEP-5 CELL TRACKING TIME COST: " + str(time.time() - t1))    


# In[ ]:


# STEP 6: Feature Calculation, including CNN-LSTM Model/Apoptosis Intensity calculation using CH3

from DT_FEATURE_WIZARD import *

DETECTOR_TYPE = Cell_Detector_Type
TRACKER_TYPE = Cell_Tracker_Type

Effector_Max_Number = 3
Target_Max_Number = 3
Effector_Feature_List = ['x', 'y', 'AR', 'SPEED', 'DM']
Target_Feature_List = ['x', 'y', 'AR', 'SPEED', 'CR', 'DM']

PARAMETER = [Effector_Max_Number, Effector_Feature_List, Target_Max_Number, Target_Feature_List]

t1 = time.time()

for BLOCK in BLOCKS:
    print('FEATURE CALCULATION for ' + BLOCK + '......')
    DT_FEATURE_EXTRACTOR(OUTPUT_PATH, DATASET, BLOCK, FRAMES, DETECTOR_TYPE, TRACKER_TYPE, PARAMETER, CORES)
    
#generate_combined_feat_table(OUTPUT_PATH, DATASET, BLOCKS, FRAMES, DETECTOR_TYPE)

#Now for the new table: 

from DT_FEATURE_WIZARD_EXPANDED import *

for BLOCK in BLOCKS:
    print('FEATURE CALCULATION for ' + BLOCK + '......')
    DT_FEATURE_EXTRACTOR_(OUTPUT_PATH, DATASET, BLOCK, FRAMES, DETECTOR_TYPE, TRACKER_TYPE, PARAMETER, CORES, seg)
    
print("STEP-6 CELL FEATURE CALCULATION TIME COST: " + str(time.time() - t1))   