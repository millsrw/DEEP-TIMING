
# coding: utf-8

# In[1]:


# Import Packages
import sys, time, os


# In[2]:


DEEP_TIMING_HOME = '/project/roysam/rwmills/NavinLab/DEEP-TIMING/'


sys.path.append(DEEP_TIMING_HOME + 'DT4-feature/')


# In[3]:

# STEP 0: Specify Parameters for experiments
CORES = 20

DATASET = "dataid"
RAW_INPUT_PATH = "projdir"
OUTPUT_PATH = "resultdir"

NUM_DIGITS = 3
BLOCKS = ['B'+str(i).zfill(NUM_DIGITS) for i in range(sblk,eblk)]
FRAMES = numframes

Cell_Detector_Type = 'FRCNN-Fast' # Other Options include [1]'FRCNN-Slow' or [2]'SSD'
# 
# # C. Cell Tracker Config
Cell_Tracker_Type = 'EZ' # Other options include [1]'EZ'

#  In[4]: 

# STEP 6: Feature Calculation, including CNN-LSTM Model/Apoptosis Intensity calculation using CH3

from DT_FEATURE_WIZARD import *

DETECTOR_TYPE = Cell_Detector_Type
TRACKER_TYPE = Cell_Tracker_Type
    
generate_combined_feat_table_(OUTPUT_PATH, DATASET, BLOCKS, FRAMES, DETECTOR_TYPE)


from DT_FEATURE_WIZARD_EXPANDED import *

#RWM table: 
save_path= OUTPUT_PATH + DATASET + 'features/'
generate_combined_feat_table_5by5(OUTPUT_PATH, DATASET, BLOCKS, FRAMES, DETECTOR_TYPE, save_path) #new edit


print("STEP-6b NEW TABLE: " + str(time.time() - t1))   
 
