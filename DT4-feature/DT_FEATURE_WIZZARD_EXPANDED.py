import os
from multiprocessing import Pool

import numpy as np


import skimage
from skimage import io
from skimage import color
from skimage import measure
from itertools import combinations
from numpy.linalg import norm
import math 

from collections import Counter
Size = 6

#for row and column update: 
def load_nanowells(a_center_fname):
    '''
    Update Row and Column Index to match physical location
    '''
    f = open(a_center_fname)
    lines = f.readlines()
    f.close()

    nanowells = []

    for line in lines:
        line = line.rstrip().split('\t')
        line = [float(i) for i in line]
        nanowells.append(line)

    return nanowells

def DT_FEATURE_EXTRACTOR(OUTPUT_PATH, DATASET, BLOCK, FRAMES, DETECTOR_TYPE, TRACKER_TYPE, PARAMETER, CORES):
    '''
        For Nanowells in each block, create the cell Mask from bbox information.
        
    '''
    
    ### STEP 1: Load Nanowell Available in the BLOCK, make directory
    os.system('mkdir ' + os.path.join(OUTPUT_PATH, DATASET,'features','1_Well_Pool')) #make directories so we can save the information RWM
    os.system('mkdir ' + os.path.join(OUTPUT_PATH, DATASET,'features','2_Cell_Pool')) # this is the directory that gets filled RWM 
    os.system('mkdir '+  os.path.join(OUTPUT_PATH, DATASET,'features','3_Cell_Pool')) #save expanded features here 
    
    
    ### STEP 2: Provide BLOCK and NANOWELL index, run NANO_FEATURE_EXTRACTOR()
    BLOCK_ET_LIST = get_BLOCK_ET_count(OUTPUT_PATH, DATASET, BLOCK, DETECTOR_TYPE) #return the nanowell infromation: well id, num e, num t, for each block RWM
    
    ARG_LIST = []
    for NANO_INFO in BLOCK_ET_LIST: #load that information here RWM
        NANOWELL = NANO_INFO[0]
        E_NUM = NANO_INFO[1]
        T_NUM = NANO_INFO[2]
        
        temp = [OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, DETECTOR_TYPE, TRACKER_TYPE, E_NUM, T_NUM] #send this information on to paralell processing RWM
        ARG_LIST.append(temp)
        generate_cell_pool(temp) #THIS DOES THE FEATURE CALCULATION!!!! RWM 
        
    #p = Pool(processes = CORES)
    #p.map(generate_cell_pool, ARG_LIST)
    
    ### STEP 3: Generate the combined feature Table
    # generate_combined_feature_table()


def mean_displacement(cell_count, T, cell_feat, msd_location, beta_location): 
    '''
    calculate the mean displacement given: https://www.pnas.org/content/105/2/459#disp-formula-1
    input: 
        cell_count (int): number of cells avaliable (T_count or E_count)
        T (int): number of frames 
        cell_feat (np.array): feature list for target cells or effector cells. We'll be grabbing the xy location from these. 
    output: 
        cell_feat (list): feature list with two additions: 
            - mean_displacement (np.array): a list of mean displacement values to append to a t_ or e_feat list 
            - beta (np.array): mean displacement increase wrt time, another list to append to a feature list 
    '''
    # The general formula for MSD is: msd= ((x1-x0)^2 + (y1-y0)^2) / num_samples 
    for index in range(0, cell_count): #then for all the target cells 
        for t in range(0, T-1): #and for each time point 
            if (cell_feat[index][0][t] == -1000 and cell_feat[index][1][t] == -1000) or (cell_feat[index][0][t+1] == -1000 and cell_feat[index][1][t+1] == -1000): #if there aren't any entries
                pass #ignore 
            else: #otherwise 
                temp1 = np.power(cell_feat[index][0][t+1]-cell_feat[index][0][t],2) #calculate the squared difference in the x 
                temp2 = np.power(cell_feat[index][1][t+1]-cell_feat[index][1][t],2) #and calculate the squared difference in the y
                #add the msd to the feature list  
                cell_feat[index][msd_location][t+1] = (temp1 + temp2) / 2 # add the two and divide by the number of frames we're averaging, this is the squared mean displacement 
                d=np.sqrt(temp1 + temp2) #now take to log form to see the increase 
                #store the log form into matrix beta, then add the beta value to the feature list 
                # try: 
                # if t > 0: 
                entry_value = (d*np.log( (temp1 + temp2) / 2 )) / (d*np.log(t+2))
                if math.isnan(entry_value) is False: 
                    cell_feat[index][beta_location][t+1]= (d*np.log( (temp1 + temp2) / 2 )) / (d*np.log(t+2)) #according to site this is the increase wrt time t, keep in mind time isn't zero indexed 
                # except: 
                    # pass #can't divide for time =0 
    return cell_feat #return the expanded feature list 


def acceleration_velocity(cell_count, T, cell_feat, acv_location):
    '''
    calculate the acceleration velocity given: https://www.pnas.org/content/105/2/459#disp-formula-1
    input: 
        cell_count (int): number of cells avaliable (T_count or E_count)
        T (int): number of frames 
        cell_feat (np.array): feature list for target cells or effector cells. We'll be grabbing the xy location from these. 
    output: 
        cell_feat (np.array) : acceleration velocity for given cells added to the feature list 
    '''
    for index in range(0, cell_count): #then for all the target cells 
        for t in range(0, T-2): #and for each time point 
            if (cell_feat[index][0][t] == -1000 and cell_feat[index][1][t] == -1000) or (cell_feat[index][0][t+1] == -1000 and cell_feat[index][1][t+1] == -1000): #if there aren't any entries
                pass #ignore 
            else: #otherwise 
                # velocity from point 0 
                vx0=cell_feat[index][0][t+1]-cell_feat[index][0][t]  #difference in x t+1 and t divided by the number of time steps 
                vy0=cell_feat[index][1][t+1]-cell_feat[index][1][t] #same thing as above but the difference in y 
                #velocity from the next point
                vx=cell_feat[index][0][t+2]-cell_feat[index][0][t+1]  #difference in x+2 and x+1 divided by the number of time steps 
                vy= cell_feat[index][1][t+2]-cell_feat[index][1][t+1]  #same thing as above but for y 
                #Now calculate the velocioty in each direction
                vel_x = (vx*vx0) / 2 # multiply together and divide by the average time point 
                vel_y = (vy*vy0) /2 #multiply together and divide by the average time point 
                #now calculate the acceleration velocity and add it to the table 
                entry_value = vel_x + vel_y
                cell_feat[index][acv_location][t+2] =entry_value

    return cell_feat


def identify_contacts(img_frames_CH1, img_frames_CH2, img_frames_CH3, T, binaryplaceholder_value, type_detector=0): 
    '''
    calculate the contacts between cells: t-cells and t-cells, target-cells and target-cells, and t and target cells 
    input: 
        img_frames_CH1 (np.array): stack of effector images from tracking (binary color coded boxes)
        img_frames_CH2 (np.array): stack of target images from tracking (binary color coded boxes)
        img_frames_CH3 (np.array): 16-bit images from death channel to do feature calculation. In the future also need to add binary masks from mrcnn. 
        T (int): the number of frames 
        type_detector (int): 0 for bbx, 1 for mrcnn (this is a placeholder - we don't need it now, but we may discover something later)
    output: 
        contact_features (np.array) : contact information based on the original timing calculation 
        pairwise_distances (np.array) : pairwise distances of cells through their cenntroids 
    '''
    #get all the possible contact combinations 
    combinations_targets = ['E1', 'E2', 'E3', 'E4', 'E5', 'T1','T2', 'T3','T4','T5'] #possible combinations names
    combos = combinations(combinations_targets,2) #get the combinations for detection and contact 
    combos_list =list(combos) #change to a list so we can reference the combinatiosn 
    num_combos=len(combos_list) #this is the number of combinations of contact we can have, *2 because of the binarization marker for Navin's lab 
    contact_features=np.ones((num_combos*2, T))*-1000 #store the combination contact information here (make sure to account for space for the marker)
    pairwise_distances=np.ones((num_combos, T))*-1000 #store the distance information here 
    
    #set up arrays to store the areas in 
    areas_E = np.zeros((5, T))
    areas_T=np.zeros((5,T))
    #set up arrays to store centroid information for pairwise distances 
    centroids_E=np.zeros((5,2, T)) *-1000
    centroids_T = np.zeros((5,2, T)) *-1000


    #Lets get all the cell information  
    for t in range(0, T): #for all the time frames 
        #Get the cell regions - for mrcnn we'll use the masks, but here we'll use the fluorescent channels 
        regions_e = skimage.measure.regionprops(img_frames_CH1[t], intensity_image=img_frames_CH3[t]) #get the region information in the same way as the effectors RWM 
        for region in regions_e: #go through all the regions 
            index=region.label-1 #this is the cell id 
            area=region.area #this is the area
            areas_E[index][t] = area #fill this array for the idea and area 
            # #get the centroids 
            centroids_E[index][0][t] = region.centroid[0]
            centroids_E[index][1][t] = region.centroid[1]
        regions_t = skimage.measure.regionprops(img_frames_CH2[t], intensity_image=img_frames_CH3[t]) #get the region information in the same way as the effectors RWM
        for region in regions_t: #go through all the regions 
            index=region.label-1 #this is the cell id 
            area=region.area #this is the area

            areas_T[index][t] = area #fill this array for the idea and area  
            centroids_T[index][0][t]=region.centroid[0]
            centroids_T[index][1][t]=region.centroid[1]
        #Now at this point we've collected all the cell areas and stored them in an array 
        # Lets see what combinations we need to do the contacts for each cell 
        combo_count=0 #go through each combination 
        dist_count=0 
        for comparison_combo in combos_list: #go through each combination 
            element1=comparison_combo[0] #this is the first element 
            element2=comparison_combo[1] #this is the second element that we can compare, each will have a different criteria 
            #make base mask: 
            mask = np.full(img_frames_CH1[t].shape, False, dtype=bool )
            if 'E' in element1: # if we're working with an effector cell then set the mask to be an effector 
                mask_name = int(element1.split('E')[1]) -1 #get which mask we're looking for 
                #only works for MRCNN: 
                # if type_detector == 1: 
                mask = img_frames_CH1[t] ==int(mask_name) #try to extract this cell. < it's a logical mask 
                mask=~mask #flip the bool 
                #extract the centroid for E: 
                cent_ele1 = np.asarray([centroids_E[mask_name][0][t], centroids_E[mask_name][1][t]]) 

            elif 'T' in element1: 
                mask_name = int(element1.split('T')[1]) -1 #get which mask we're looking for 
                # if type_detector == 1: 
                mask = img_frames_CH2[t] ==int(mask_name) #try to extract this cell. < it's a logical mask 
                mask=~mask #flip the bool 
                #extract the centroid for T
                cent_ele1 = np.asarray([centroids_T[mask_name][0][t], centroids_T[mask_name][1][t]])

            #So at this point we've created the mask to use, now let's decide which crop of the image to evaluate, which is element2 
            if 'E' in element2: # if we are working with an effector cell 
                cell_name = int(element2.split('E')[1]) -1 #tells us which area of E to grab 
                cell_crop = areas_E[int(cell_name)][t] #since we made a filler matrix we can grab this value even if there's no cell here
                #get the centroid information 
                # cent_ele2 = np.asarray([centroids_Ex[cell_name][t], centroids_Ey[cell_name][t]])
                cent_ele2 = np.asarray([centroids_E[cell_name][0][t], centroids_E[cell_name][1][t]])
                #Do the calculation for distances
                if 0 in cent_ele1: #if it's out of bounds or doesn't exist  
                    pass
                elif 0 in cent_ele2: 
                    pass 
                else: 
                    pairwise_distances[dist_count][t]= norm(np.array(cent_ele2) - np.array(cent_ele1)) #calculate the euclidean distance 
                    # print('norm E', norm(np.array(cent_ele2) - np.array(cent_ele1)))
                    contact_features[combo_count][t] = float(np.sum(img_frames_CH1[t][mask] == (cell_name+1))) / float(cell_crop)
                contact_features[combo_count + 1][t] = binaryplaceholder_value # marker 


            elif 'T' in element2: #if we're working with a target cell 
                cell_name = int(element2.split('T')[1]) -1 #get which target we have 
                cell_crop = areas_T[int(cell_name)][t] #get the right area 
                #get the centroid info: 
                # cent_ele2 = np.asarray([centroids_Tx[cell_name][t], centroids_Ty[cell_name][t]])
                cent_ele2 = np.asarray([centroids_T[cell_name][0][t], centroids_T[cell_name][1][t]])
                #Do the calculation for distances
                if 0 in cent_ele1: #if it's out of bounds or doesn't exist  
                    pass
                elif 0 in cent_ele2: 
                    pass 
                else: 
                    pairwise_distances[dist_count][t]= norm(np.array(cent_ele2) - np.array(cent_ele1)) #calculate the euclidean distance 
                    # print('norm T', norm(np.array(cent_ele2) - np.array(cent_ele1)))
                    contact_features[combo_count][t] = float(np.sum(img_frames_CH2[t][mask] == (cell_name+1))) / float(cell_crop) 
                contact_features[combo_count + 1][t] = binaryplaceholder_value #marker 
    
            combo_count=combo_count+2 #go to the next combination 
            dist_count = dist_count + 1 #go to the next combination distance calculation 

    return  contact_features, pairwise_distances #return array with contact information on. Append this to the end of the table 



def generate_features(cell_count, T, cell_feat, scores_matrix, img_frames_CHx, img_frames_CH3, binaryplaceholder_value):  
    ''' This function generates features for each cell within a nanowell 
    Inputs: 
        cell_count (int): the number of cells per cell type 1 or 2 for 1 or 2 effectors for this nanowell 
        T(int): the number of frames 
        cell_feat(np.array): an array to fill with features 
        scores_matrix(np.array): an array of cnn scores from faster-rcnn or mask rcnn 
        img_frames_CHx (np.array): video of tracking-labled masks (bbx or mrcnn) for each cell in a nanowell, T frames long 
        img_frames_CH3 (np.array): video of the death channel for nanowell 
        binaryplaceholder_value (int): tells us if we're running for effectors or targets (effectors are -1, targets are -2) and adds a placeholder value for Navin's binarization 
    Outputs: 
        cell_feat (np.array): filled in cell feature table of same size as the input with each row a different feature
    '''  
    for t in range(0, T): #for all the time frames RWM
        # centroid_x, centroid_y, AR, Speed, Death are the original features**
        regions = skimage.measure.regionprops(img_frames_CHx[t], intensity_image=img_frames_CH3[t]) #get the regions RWM
        #regions are the mask (bbx or maskrcnn output) of the cell, intensity values are calculated
        # from the intensity of the death channel in the region of the mask from the cell-specic channel. RWM
        marker = np.ones((1,T),dtype=np.int)*(binaryplaceholder_value) #add a marker for binarization (for Navin's post processing)  RWM
        for region in regions: #for each region detected RWM
            index = region.label-1 #get the index number, so for region 1 detected it's 0, for region 2 it's 1..., these are the cell ids 
            (centroid_y, centroid_x) = region.centroid #get the centroid information, it should be the centroid relative to the entire image not bbx RWM
            aspect_ratio = region.minor_axis_length/(region.major_axis_length+1) #get the aspect ratio RWM- hengyang original 
            # death_marker = region.mean_intensity - hengyang original commented out 
            # death_marker = region.max_intensity - hengyang original commented out 
            
            cell_feat[index][0][t] = centroid_x #add the x location to the feature table RWM
            cell_feat[index][1][t] = centroid_y #add the y location to the feature table RWM
            cell_feat[index][2][t] = aspect_ratio #add the aspect ratio to the feature table RWM
            #Let's add the area on 
            cell_feat[index][6][t] = region.area #calculate and add on the area of the region RWM 
            cell_feat[index][7][t] = float(scores_matrix[t][index]) #add the cnn scores to the table RWM 
            #add the marker: 
            cell_feat[index][5] = marker #add a marker after death for Navin post processing
            #Death is calculated here ---------------------------------------------------------------------------------------------------------------------------------------------------------
            death_marker = np.mean(img_frames_CH3[t][int(centroid_y-3):int(centroid_y+3), int(centroid_x-3):int(centroid_x+3)]) # get the mean value of the centroid over a span of 6 pixels RWM        
            cell_feat[index][4][t] = death_marker #add the death inforamtion to the feature table RWM
            #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Hengyang original speed calculation 
    for index in range(0, cell_count): #go through all the cells (effectors or targets) RWM 
        for t in range(0, T-1): #for each frame RWM
            if (cell_feat[index][0][t] == -1000 and cell_feat[index][1][t] == -1000) or (cell_feat[index][0][t+1] == -1000 and cell_feat[index][1][t+1] == -1000): #if there's been no entry RWM
                pass 
            else: #otherwise if there has been an entry RWM 
                temp1 = np.power(cell_feat[index][0][t]-cell_feat[index][0][t+1],2) #calculate the location in x over one frame RWM 
                temp2 = np.power(cell_feat[index][1][t]-cell_feat[index][1][t+1],2) #calulate the locationin y over one frame RWM
                cell_feat[index][3][t] = np.sqrt(temp1 + temp2) #take the square root of the sum of the two values and add it to row 3
    #add motility information: mean squared displacement and acceleration velocity 
    cell_feat = mean_displacement(cell_count, T, cell_feat, 8, 9) #append msd values to E_feat table 
    cell_feat = acceleration_velocity(cell_count, T, cell_feat, 10) #append msd values to E_feat table 

    return cell_feat #return feature table for one nanowell/ cell type
   

def get_BLOCK_ET_count(OUTPUT_PATH, DATASET, BLOCK, DETECTOR_TYPE):
    '''
        returns the selected nanowell in one block
        [[ID1, E#, T#], [ID2, E#, T#], ...]
    '''
    
    BLOCK_ET_COUNT = [] 
    BLOCK_ET_FNAME = OUTPUT_PATH + DATASET + '/' + BLOCK + '/labels/DET/' + DETECTOR_TYPE + '/raw/selected_nanowells.txt' #read in this file from faster-rcnn (or mrcnn) RWM
    
    f = open(BLOCK_ET_FNAME)
    lines = f.readlines()
    f.close()
    
    for line in lines: #loop through each line RWM
        temp = line.rstrip().split('\t') #the file is tab delimited RWM
        ID = int(temp[0]) #get the nanowell ID RWM
        E_count = int(temp[1]) #get the number of effector cells RWM
        T_count = int(temp[2]) #get the number of t cells RWM
        BLOCK_ET_COUNT.append([ID, E_count, T_count]) #add it to the list of nanowell info RWM
        
    return BLOCK_ET_COUNT #return a list with nanowell information for each block RWM


def generate_cell_pool(ARGS): #THIS DOES THE FEATURE CALCULATION!!! CHANGES TO FEATURES SHOULD BE ADDED HERE RWM 
    OUTPUT_PATH = ARGS[0]
    DATASET = ARGS[1]
    BLOCK = ARGS[2]
    NANOWELL = ARGS[3]
    FRAMES = ARGS[4]
    DETECTOR_TYPE = ARGS[5]
    TRACKER_TYPE = ARGS[6]
    E_NUM = ARGS[7]
    T_NUM = ARGS[8]
    
    # step 1: load image sequences:
    #so return two nd matricies (or videos) of the bounding box for the cells and a collection of the death information RWM (a bunch of grey boxes color coded to the number and location)
    # img_frames_CH1, img_frames_CH2, img_frames_CH3 , E_scores, T_scores= create_img_frames_mrcnn(OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, DETECTOR_TYPE, TRACKER_TYPE, E_NUM, T_NUM) #load mrcnn info in and not the bbx 
    #for regular DT with no masks 
    img_frames_CH1, img_frames_CH2, img_frames_CH3 , E_scores, T_scores= create_img_frames(OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, DETECTOR_TYPE, TRACKER_TYPE, E_NUM, T_NUM) #load mrcnn info in and not the bbx 

    
    # step 2: get the # of E and T %%%This part adopt a lot of the original codes, so there are alias of variables declared here
    E_count = 0
    T_count = 0
    T = FRAMES
    #verify the number of cells through the collection of images RWM
    if len(img_frames_CH1) == T: #if we've collected the proper number of images for the effector cells RWM
        for i in range(0, T): #for each frame RWM
            temp1 = np.amax(img_frames_CH1[i]) #get the maximum value from the collection of images (remember this denotes the number of cells present) RWM
            if temp1 > E_count: #if it's greater than the number of effectors that are supposed to be here RWM
                E_count = temp1 #then listen to the fluorescent channel and change the number of effecttors to this value RWM

    if len(img_frames_CH2) == T: #do the same thing for target cells RWM
        for i in range(0, T):
            temp2 = np.amax(img_frames_CH2[i])
            if temp2 > T_count:
                T_count = temp2    
    
    # step 3: calculate features for E and T candidates
    if E_count>0: #if there are some effectors present RWM
        E_count = int(E_count)
        E_feat = np.ones((E_count, 11, T), dtype=np.float)*(-1000) #make this empty marker, again, if we change the features-thus number of rows to add, this needs to change from 5 to whatever RWM
        E_feat = generate_features(E_count, T, E_feat, E_scores, img_frames_CH1, img_frames_CH3, -1 )



    if T_count>0: #do the same for targets, if there's a target here 
        T_count = int(T_count) #get the number of target cells RWM
        T_feat = np.ones((T_count, 11, T), dtype=np.float)*(-1000) #make a T_feat list to fill in, if we want more values we'll need to make it bigger than 5 entries RWM 
        T_feat = generate_features(T_count, T, T_feat, T_scores, img_frames_CH2, img_frames_CH3 , -2)


    
    #Step 3b: generate contact information RWM 
    if E_count > 0 and T_count > 0 or E_count >1 or T_count > 1: 
        distance_assessment = []
        # print('Generating contacts for nanowell: ', NANOWELL, E_count, T_count)
        contacts, distances =identify_contacts(img_frames_CH1, img_frames_CH2, img_frames_CH3, T, -4 ) #output an array with all the cell-cell contact information for 5 cell types each 
        distance_assessment = np.vstack((contacts, distances)) #stack together 
        # print(distance_assessment.shape)
        # distance_assessment = list(distance_assessment) #convert to list? 

    # step 4: write the features to disk
    if E_count > 0:
        for i in range(0,E_count):
            fname_prefix = os.path.join(OUTPUT_PATH, DATASET,'features/3_Cell_Pool/')
            fname = BLOCK + 'No' + str(NANOWELL) + 'E' + str(i+1) + '.txt'
            fname = fname_prefix + fname
            write_cell_feature(fname, E_feat[i])

    if T_count > 0:
        for i in range(0, T_count):
            fname_prefix = os.path.join(OUTPUT_PATH, DATASET,'features/3_Cell_Pool/')
            fname = BLOCK + 'No' + str(NANOWELL) + 'T' + str(i+1) + '.txt'
            fname = fname_prefix + fname
            write_cell_feature(fname, T_feat[i])     

    if E_count > 0 and T_count > 0 or E_count >1 or T_count > 1:  #if either cell is present write it out 
        fname_prefix = os.path.join(OUTPUT_PATH, DATASET,'features/3_Cell_Pool/')   
        fname = BLOCK + 'No' + str(NANOWELL) + 'ET_contact' + '.txt' #writes out the cell contact info regardless of the number of cells 
        fname = fname_prefix + fname
        write_cell_feature(fname,distance_assessment) #we'll only have one of these files per nanowell 
    

def create_img_frames(OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, DETECTOR_TYPE, TRACKER_TYPE, E_NUM, T_NUM):
    img_frames_CH1 = np.zeros((FRAMES, 281, 281), dtype=np.uint8) #save the effector information RWM
    img_frames_CH2 = np.zeros((FRAMES, 281, 281), dtype=np.uint8) #save the target information RWM
    img_frames_CH3 = np.zeros((FRAMES, 281, 281), dtype=np.uint16) #save the death information RWM
    #if we choose to gather the phase contrast videos: 
    # img_frames_CH0= np.zeros((FRAMES, 281, 281), dtype=np.uint8) 

    #gather score information 
    E_scores =np.zeros((FRAMES, 5), dtype=np.float32) #save the effector information RWM
    T_scores = np.zeros((FRAMES, 5), dtype=np.float32) #save the effector information RWM

    ### CH1
    if E_NUM > 0: #go through all the effectors that are there RWM
        #try:
        #return a list that contains a list for each tracking information of the effector cell RWM 
        label_E_sequence = load_bbox_sequence(OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, 'E', E_NUM, DETECTOR_TYPE, TRACKER_TYPE) 
        for t in range(FRAMES): #go through all the frames 
            for N in range(E_NUM): #for every effector here 
                x = int(label_E_sequence[t][N][1]) #for list for the Nth effector cell at time point t the first entry is x RWM 
                y = int(label_E_sequence[t][N][2]) #and is y RWM 
                w = int(label_E_sequence[t][N][3]) #and is width RWM 
                h = int(label_E_sequence[t][N][4]) #and is height RWM 
                score = float(label_E_sequence[t][N][6]) 
                E_scores[t][N]=score
                
                dw = int(0.08*w) #add some pixel values on to expend the width RWM 
                dh = int(0.08*h) #add some pixel values on to expand the height RWM 

                if w>4 and h>4: #if they're greater than 4 (not sure why, maybe to show there's a legit cell here worth looking at) RWM 
                    #create a Nd binary mask of effector cell bounding boxes 
                    img_frames_CH1[t][y+dh:y+h-dh, x+dw:x+w-dw] = N + 1 # build this matrix where the layer is the time frame, and the x and y is the bounding box for the cell, and make binary with value N to show which effector RWM 
        #except:
            #print("CH1 ERROR")
            #pass


    ### CH2
    if T_NUM > 0: #do the same for target cells RWM 
        #try:
        label_T_sequence = load_bbox_sequence(OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, 'T', T_NUM, DETECTOR_TYPE, TRACKER_TYPE)
        for t in range(FRAMES):
            for N in range(T_NUM):
                x = int(label_T_sequence[t][N][1])
                y = int(label_T_sequence[t][N][2])
                w = int(label_T_sequence[t][N][3])
                h = int(label_T_sequence[t][N][4])
                score = float(label_T_sequence[t][N][6]) 
                T_scores[t][N]=score

                dw = int(0.1*w)
                dh = int(0.1*h)
                
                if w>4 and h>4:
                    img_frames_CH2[t][y+dh:y+h-dh, x+dw:x+w-dw] = N + 1
                    #make an N-d matrix where each layer is a frame of the bounding box information for each target 
        #except:
            #print("CH2 ERROR")
            #pass

    ### CH3
    if E_NUM > 0 or T_NUM > 0: #if there's anything in this nanowell RWM 
        try:
            for t in range(FRAMES): #go through all the frames RWM 
                fname = OUTPUT_PATH + DATASET + '/' + BLOCK + '/images/crops_16bit_s/' + 'imgNo' + str(NANOWELL) + 'CH3/imgNo' + str(NANOWELL) + 'CH3_t' + str(t+1) + '.tif' #get the death information from the 16 bit images RWM 
                img_frames_CH3[t] = io.imread(fname) #read it in and add it to the collection RWM 
                #If we want to read in the phase contrast images for other purposes do that here too 
                # fname_ch0 = OUTPUT_PATH + DATASET + '/' + BLOCK + '/images/crops_8bit_s/' + 'imgNo' + str(NANOWELL) + 'CH0/imgNo' + str(NANOWELL) + 'CH0_t' + str(t+1) + '.tif'
                # set into the phase contrast 
                # img_frames_CH0[t] = io.imread(fname_ch0)

        except:
            print("CH3 ERROR")
            pass
        
    return img_frames_CH1, img_frames_CH2, img_frames_CH3 , E_scores, T_scores#so return two nd matricies (or videos) of the bounding box for the cells and a collection of the death information RWM 
        

def create_img_frames_mrcnn(OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, DETECTOR_TYPE, TRACKER_TYPE, E_NUM, T_NUM):
    img_frames_CH1 = np.zeros((FRAMES, 281, 281), dtype=np.uint8) #save the effector information RWM
    img_frames_CH2 = np.zeros((FRAMES, 281, 281), dtype=np.uint8) #save the target information RWM
    img_frames_CH3 = np.zeros((FRAMES, 281, 281), dtype=np.uint16) #save the death information RWM
    E_scores =np.zeros((FRAMES, 5), dtype=np.float32) #save the effector information RWM
    T_scores = np.zeros((FRAMES, 5), dtype=np.float32) #save the effector information RWM
    #if we choose to gather the phase contrast videos: 
    # img_frames_CH0 = np.zeros((FRAMES, 281, 281), dtype=np.uint8) 

    ### CH1
    if E_NUM > 0: #go through all the effectors that are there RWM
        #try:
        #return a list that contains a list for each tracking information of the effector cell RWM 
        label_E_sequence = load_bbox_sequence(OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, 'E', E_NUM, DETECTOR_TYPE, TRACKER_TYPE) 
        mask_path = os.path.join( OUTPUT_PATH + DATASET , BLOCK , 'images', 'masks', 'imgNo' + str(NANOWELL) + 'CH1')
        for t in range(FRAMES): #go through all the frames 
            effector_mask = io.imread(os.path.join(mask_path, 'imgNo' + str(NANOWELL) + 'CH1_t' + str(t+1) + '.png')) #read in the corresponding mrcnn mask 
            for N in range(E_NUM): #for every effector here 
                x = int(label_E_sequence[t][N][1]) #for list for the Nth effector cell at time point t the first entry is x RWM 
                y = int(label_E_sequence[t][N][2]) #and is y RWM 
                w = int(label_E_sequence[t][N][3]) #and is width RWM 
                h = int(label_E_sequence[t][N][4]) #and is height RWM 
                score = float(label_E_sequence[t][N][6]) 
                E_scores[t][N]=score
                

                dw = int(0.08*w) #add some pixel values on to expend the width RWM 
                dh = int(0.08*h) #add some pixel values on to expand the height RWM 

                if w>4 and h>4: #if they're greater than 4 (not sure why, maybe to show there's a legit cell here worth looking at) RWM 
                    #create a Nd binary mask of effector cell bounding boxes 
                    #but instead of bbx we want it to be the binary mask with the proper color so we need to load in the mask too 
                    #so the bbx tracking info get's the correct cell 
                    #and the mask tells us which pixles to use 
                    #and the N+1 is to make sure we've got the right color 
                    color=N+1 #this is the color corresponding to tracking 
                    masks_here = len(np.unique(effector_mask[y-dh:y+h+dh, x-dw:x+w+dw]))-1 # this tells us how many cells are in overlap in this particular crop 
                    
                    if masks_here > 1: 
                        mask_values = np.unique(effector_mask[y-dh:y+h+dh, x-dw:x+w+dw])#eliminate the background 
                        mask_values=np.delete(mask_values, 0)
                        mask_canidates=[]
                        for mask in mask_values: #loop through all the potential values 
                            #extract the most likely one and save it 
                            mask_canidates.append(np.sum( effector_mask[y-dh:y+h+dh, x-dw:x+w+dw] == mask )) #sum all the values in the crop with this particular mask and append to the canidates list 
                        best_canidate_index = mask_canidates.index(np.amax(mask_canidates)) #find the place with the most likely mask 
                        mask_value = mask_values[best_canidate_index]
                        actual_mask = (effector_mask[y-dh:y+h+dh, x-dw:x+w+dw] == mask_value) * color

                    else: #just use any color > 0 as the mask selection 
                        actual_mask= (effector_mask[y-dh:y+h+dh, x-dw:x+w+dw] > 0) * color #get the correct bbx location to the cell, and get all the pixels that are in the cell, and change the bool mask to the color value
                        img_frames_CH1[t][y-dh:y+h+dh, x-dw:x+w+dw] =actual_mask # build this matrix where the layer is the time frame, and the x and y is the pixels for the cell from tracking
        #except:
            #print("CH1 ERROR")
            #pass


    ### CH2
    if T_NUM > 0: #do the same for target cells RWM 
        #try:
        label_T_sequence = load_bbox_sequence(OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, 'T', T_NUM, DETECTOR_TYPE, TRACKER_TYPE)
        mask_path = os.path.join( OUTPUT_PATH + DATASET , BLOCK , 'images', 'masks', 'imgNo' + str(NANOWELL) + 'CH2')
        for t in range(FRAMES):
            target_mask = io.imread(os.path.join(mask_path, 'imgNo' + str(NANOWELL) + 'CH2_t' + str(t+1) + '.png')) #read in the corresponding mrcnn mask
            for N in range(T_NUM):
                x = int(label_T_sequence[t][N][1])
                y = int(label_T_sequence[t][N][2])
                w = int(label_T_sequence[t][N][3])
                h = int(label_T_sequence[t][N][4])
                score = float(label_T_sequence[t][N][6]) 
                T_scores[t][N] =score 

                dw = int(0.1*w)
                dh = int(0.1*h)
                
                if w>4 and h>4:
                    color=N+1 #this is the color corresponding to tracking 
                    masks_here = len(np.unique(target_mask[y-dh:y+h+dh, x-dw:x+w+dw]))-1 # this tells us how many cells are in overlap in this particular crop 
                    
                    if masks_here > 1: 
                        mask_values =np.unique(target_mask[y-dh:y+h+dh, x-dw:x+w+dw])
                        mask_values=np.delete(mask_values, 0)
                        mask_canidates=[]
                        for mask in mask_values: #loop through all the potential values 
                            #extract the most likely one and save it 
                            mask_canidates.append(np.sum( target_mask[y-dh:y+h+dh, x-dw:x+w+dw] == mask )) #sum all the values in the crop with this particular mask and append to the canidates list 
                        best_canidate_index = mask_canidates.index(np.amax(mask_canidates)) #find the place with the most likely mask 
                        mask_value = mask_values[best_canidate_index]
                        # print(mask_path, t+1)
                        # print('Mask info', mask_values, mask_canidates, mask_value)
                        actual_mask = (target_mask[y-dh:y+h+dh, x-dw:x+w+dw] == mask_value) * color

                    else: #just use any color > 0 as the mask selection 
                        actual_mask= (target_mask[y-dh:y+h+dh, x-dw:x+w+dw] > 0) * color #get the correct bbx location to the cell, and get all the pixels that are in the cell, and change the bool mask to the color value
                        img_frames_CH2[t][y-dh:y+h+dh, x-dw:x+w+dw] = actual_mask #set it to the proper cell deleniation mask values 
                        #make an N-d matrix where each layer is a frame of the bounding box information for each target 
        #except:
            #print("CH2 ERROR")
            #pass

    ### CH3
    if E_NUM > 0 or T_NUM > 0: #if there's anything in this nanowell RWM 
        try:
            for t in range(FRAMES): #go through all the frames RWM 
                fname = OUTPUT_PATH + DATASET + '/' + BLOCK + '/images/crops_16bit_s/' + 'imgNo' + str(NANOWELL) + 'CH3/imgNo' + str(NANOWELL) + 'CH3_t' + str(t+1) + '.tif' #get the death information from the 16 bit images RWM 
                img_frames_CH3[t] = io.imread(fname) #read it in and add it to the collection RWM 
                #If we want to read in the phase contrast images for other purposes do that here too 
                # fname_ch0 = OUTPUT_PATH + DATASET + '/' + BLOCK + '/images/crops_8bit_s/' + 'imgNo' + str(NANOWELL) + 'CH0/imgNo' + str(NANOWELL) + 'CH0_t' + str(t+1) + '.tif'
                # set into the phase contrast 
                # img_frames_CH0[t] = io.imread(fname_ch0)
        except:
            print("CH3 ERROR")
            pass
        
    return img_frames_CH1, img_frames_CH2, img_frames_CH3 , E_scores, T_scores#so return two nd matricies (or videos) of the bounding box for the cells and a collection of the death information RWM 


def load_bbox_sequence(OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, cell_type, cell_count, DETECTOR_TYPE, TRACKER_TYPE):

    # load label_E_sequence
    if cell_type == 'E': #if we're working with effectors RWM
        label_E_sequence = [] #fill in this list RWM
        E_num = cell_count #the number of effector cells in this nanowell RWM
        for t in range(1, FRAMES + 1): #go through all the frames RWM
            if E_num > 0: #and if there's an effector cell here RWM
                #load the tracking information RWM
                label_E_fname = OUTPUT_PATH + DATASET + '/' + BLOCK + '/labels/TRACK/' + TRACKER_TYPE + '/' + DETECTOR_TYPE + '/imgNo' + str(NANOWELL) + '/label_E_t' + str(t).zfill(3) + '.txt'
                f = open(label_E_fname) #read the tracking information file for the effector RWM
                lines = f.readlines()
                f.close()
                temp_E = []
                for line in lines: #go through the file RWM
                    line = line.rstrip().split('\t') #it's tab delimited RWM
                    line = [float(kk) for kk in line] #change each to a float RWM
                    temp_E.append(line) #and append the information to the list RWM
                    label_E_sequence.append(temp_E) #and add this list to the larger list of the effector sequence RWM 

        return label_E_sequence #return a list that contains a list for each tracking information of the effector cell RWM 

    if cell_type == 'T':
        label_T_sequence = []
        T_num = cell_count
        for t in range(1, FRAMES + 1):
            if T_num > 0:
                label_T_fname = OUTPUT_PATH + DATASET + '/' + BLOCK + '/labels/TRACK/' + TRACKER_TYPE + '/' + DETECTOR_TYPE + '/imgNo' + str(NANOWELL) + '/label_T_t' + str(t).zfill(3) + '.txt'
                f = open(label_T_fname)
                lines = f.readlines()
                f.close()
                temp_T = []
                for line in lines:
                    line = line.rstrip().split('\t')
                    line = [float(kk) for kk in line]
                    temp_T.append(line)
                    label_T_sequence.append(temp_T)

        return label_T_sequence        

    
def No2RC(x, size):
    R = int((x-1)/size) + 1
    C = int(x - (R-1)*size)

    return [R,C]    


def generate_combined_feat_table_5by5(OUTPUT_PATH, DATASET, BLOCKS, FRAMES, DETECTOR_TYPE, save_path):
    # read in the cell count of all nanowells in the Block and generate the estimated Effector& Target cell count
    print("ASSEMBLE ALL THE FEATURES......")
    # step 1 create the file Table_Exp.txt
    Dataset_Output_Path = OUTPUT_PATH + DATASET + '/'
    
    T = FRAMES
    Table_Exp = [] #Table to Fill in RWM

    for BID in BLOCKS: #go through all the blocks RWM
        # step 2 get the nanowell number and cell count
        BLOCK_ET_LIST = get_BLOCK_ET_count(OUTPUT_PATH, DATASET, BID, DETECTOR_TYPE) #returns nanowell information from each block such as id, and number of effectors and targets from the object detection network RWM
        for NANO_INFO in BLOCK_ET_LIST:
            well_ID = NANO_INFO[0]
            E_count = NANO_INFO[1]
            T_count = NANO_INFO[2]

            # [R,C] = No2RC(well_ID, 6)
            
            l0 = len(Table_Exp)

        # step 3 cell Pool update
            flag_E = 0
            flag_T =0

            sorted_nanowells = load_nanowells(Dataset_Output_Path+ BID + '/meta/a_centers_clean_sorted.txt') ### row column index

            block = int(BID[1:4]) # get the block Id RWM
            # [R,C] = No2RC(well_ID, 6) #convert to nanowell rows and columns RWM
            [C,R] = No2RC(int(sorted_nanowells[well_ID-1][4]), 6) ### row column index update - and make sure it's the flip because this is the case 
            # print("BLOCK : " + BID + " NANOWELL: " + str(well_ID)  + ' Row and Column: ' + str(R) + ' '+  str(C))
            print("Block, Row, Column: " + BID  + ' ' + str(R) + ' '+  str(C))
            print(" NANOWELL: " + str(well_ID) , 'Nanowell should be: ' + str(sorted_nanowells[well_ID-1][4]))
               
            x_temp=[] #fill in this list RWM
            for E_num in range(1,6): #go through five times since there's five effectors RWM
                flag_E=1 
                E_fname = Dataset_Output_Path + 'features/3_Cell_Pool/' +BID+'No'+str(well_ID)+'E' + str(E_num) + '.txt' #make sure we load these particular features! RWM
                #Note: if you change the features to write, you need to change the function that writes out these .txt files!! RWM
                if os.path.isfile(E_fname) ==True: 
                    f_E=open(E_fname,'r') #read in the file RWM
                    x = f_E.readlines()
                    f_E.close()
                    for line in x: #for each line in the file RWM
                        x_temp.append(str(block) + '\t' + str(R) + '\t' + str(C) + '\t' + line) #add the contects to this mini list x_temp RWM
                else: #if it doesn't exist RWM
                    x=np.ones((11,T),dtype=np.int)*(-1000) #create a marker for an empty line RWM
                    for line in x: #go through all the lines RWM
                        line1 = [str(i) for i in line] #make them strings RWM
                        x_temp.append(str(block) + '\t' + str(R) + '\t' + str(C) + '\t' + '\t'.join(line1) + '\n') #add the block, row, and column info to the line and add it to the list RWM
            
            #Now write out for Targets if there's at least one effector present 
            if flag_E == 1: #now it's time to go through all the targets, flag_E will always be 1 because that means it processed an effector cell first, but if you want a case not to write out then change this RWM
                Table_Exp = Table_Exp + x_temp #add the table onto the mini list of effector information we made above RWM
                #write in the features made in the generation step - there are 5 rows of target information 
                y_temp=[]
                for T_num in range(1,6): #now let's go through all the targets RWM, we don't loop 6 times because we can just append empty information to the end after we're done to save time RWM
                    T_fname = Dataset_Output_Path + 'features/3_Cell_Pool/' +BID+'No'+str(well_ID)+'T' + str(T_num) +'.txt' #go get these features! Again, if we want to change the featuers, you need to rewrite these txt files RWM
                    flag_T = 1 #if we successfully write the target information 

                    #Note: if you change the features to write, you need to change the function that writes out these .txt files!! RWM
                    if os.path.isfile(T_fname) ==True: 
                        f_T=open(T_fname,'r') #read in the file RWM
                        y = f_T.readlines()
                        f_T.close()
                        for line in y: #for each line in the file RWM
                            y_temp.append(str(block) + '\t' + str(R) + '\t' + str(C) + '\t' + line) #add the contects to this mini list x_temp RWM
                    else: #if it doesn't exist RWM
                        x=np.ones((11,T),dtype=np.int)*(-1000) #create a marker for an empty line RWM
                        for line in x: #go through all the lines RWM
                            line1 = [str(i) for i in line] #make them strings RWM
                            y_temp.append(str(block) + '\t' + str(R) + '\t' + str(C) + '\t' + '\t'.join(line1) + '\n') #add the block, row, and column info to the line and add it to the list RWM
                Table_Exp = Table_Exp + y_temp #add the first five rows on to the original table RWM
                    

            #Now add the contacts (if there are any)
            if flag_E ==1 and flag_T ==1 : #if we successfully pass the first two loops#int(E_count) > 0 and int(T_count) > 0 or int(E_count) >1 or int(T_count) > 1:  #if either cell is present write in the contact info 
                
                #Now for bead placeholder: 
                d_temp=[]
                x=np.ones((7*3,T),dtype=np.int)*(-1000) #create a marker for an empty line RWM
                for line in x: 
                    line1=[str(i) for i in line] #make them strings RWM
                    d_temp.append(str(block) + '\t' + str(R) + '\t' + str(C) + '\t' + '\t'.join(line1) + '\n') 
                Table_Exp = Table_Exp + d_temp #add on the beads 
                
                #pull the file info 
                contact_fname = Dataset_Output_Path + 'features/3_Cell_Pool/' +BID+'No'+str(well_ID)+'ET_contact.txt' #go get these features! Again, if we want to change the featuers, you need to rewrite these txt files RWM
                c_temp = []
                if os.path.isfile(contact_fname) ==True:
                    f_C=open(contact_fname,'r') #try to read in the above file RWM 
                    c = f_C.readlines()
                    f_C.close()
                    for line in c: #for each line in the file RWM
                        c_temp.append(str(block) + '\t' + str(R) + '\t' + str(C) + '\t' + line) #add the block, row, column, and file information to the y_temp list RWM
                else: 
                    x=np.ones((45*2+45,T),dtype=np.int)*(-1000) #create a marker for an empty line RWM
                    for line in x: 
                        line1=[str(i) for i in line] #make them strings RWM
                        c_temp.append(str(block) + '\t' + str(R) + '\t' + str(C) + '\t' + '\t'.join(line1) + '\n') 
                #add to the big table: 
                Table_Exp = Table_Exp + c_temp
                

            

            marker4 = [] #then fill a new marker RWM
            for i in range(1,T+1):#go through all the time frames RWM
                marker4.append(str(i)) #add a list of 0 1 2 3 4... T to show we're changing nanowells RWM
            Table_Exp.append(str(block) + '\t' + str(R) + '\t' + str(C) + '\t' + '\t'.join(marker4) + '\n') #and add the nanowell  information on so we can add it to the table RWM

            l1 = len(Table_Exp) #l1 is the length of the table so we can show where we're at in the process RWM
            
            # print("ADDING lines: " + str(l1-l0)) #and print that here RWM
            print('     E and T: ' + str(E_count) + ' ' + str(T_count))
            print('')
            
            
        # step 4 write the useful features to Table_Exp.txt
    fname = os.path.join(save_path, DATASET+ '_Table_Extended.txt')
    f = open(fname, 'w')
    f.writelines(Table_Exp)
    f.close()



def write_cell_feature(fname, feature_array):

    float_formatter = lambda x: "%.2f" % x

    if 'E' in fname.split('/')[-1]:
        f = open(fname,'w')
        for i in range(0,11): #was 5 and is now 8 to match the extended featuers 
            line = [float_formatter(x) for x in feature_array[i]] # include the BID and No in the first two elements of feature
            line = '\t'.join(line) + '\n'
            f.writelines(line)
        f.close()

    if 'T' in fname.split('/')[-1]:
        f = open(fname,'w')
        for i in range(0,11): #ws 6 and is now 8 to match the number of featuers 
            line = [float_formatter(x) for x in feature_array[i]]
            line = '\t'.join(line) + '\n'
            f.writelines(line)
        f.close()
    
    if 'ET_contact' in fname.split('/')[-1]: #gets the nanowell id 
        f = open(fname,'w')
        for i in range(0,45*2 + 45): # this is num_combos 
            line = [float_formatter(x) for x in feature_array[i]] # include the BID and No in the first two elements of feature
            line = '\t'.join(line) + '\n'
            f.writelines(line)
        f.close()
