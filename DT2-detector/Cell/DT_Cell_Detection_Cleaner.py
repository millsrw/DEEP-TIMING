import os
import numpy as np

from multiprocessing import Pool


def DT_Cell_Cleaner(OUTPUT_PATH, DATASET, BLOCKS, FRAMES, CC_THRESHOLD, SCORE_THRESHOLD, MAX_E_COUNT, MAX_T_COUNT, Detector_Type, CORES):
    for BLOCK in BLOCKS:
        print("Applying Confinement Constraint " + BLOCK + " ...... ")
        
        DT_Cell_Clean_Nanowell(OUTPUT_PATH, DATASET, BLOCK, FRAMES, CC_THRESHOLD, SCORE_THRESHOLD, Detector_Type, CORES)
        
        
def DT_Cell_Clean_Nanowell(OUTPUT_PATH, DATASET, BLOCK, FRAMES, CC_THRESHOLD, SCORE_THRESHOLD, Detector_Type, CORES):
    
    # Get Number of Nanowells
    Nanowells = get_nanowell_numbers(OUTPUT_PATH, DATASET, BLOCK)        
    
    Parameter_List = []
    for Nanowell in range(1, Nanowells+1):
        temp = []
        
        label_FRCNN = OUTPUT_PATH + DATASET + '/' + BLOCK + '/labels/DET/' + Detector_Type +'/raw/'
        os.system('rm -rf ' + label_FRCNN + 'selected_nanowells.txt')
        label_FRCNN_OUT = OUTPUT_PATH + DATASET + '/' + BLOCK + '/labels/DET/' + Detector_Type +'/clean/'
        os.system('mkdir ' + label_FRCNN_OUT)
        
        
        temp.append(label_FRCNN)
        temp.append(label_FRCNN_OUT)
        temp.append(Nanowell)
        temp.append(FRAMES)
        temp.append(CC_THRESHOLD)
        temp.append(SCORE_THRESHOLD)
        
        #clean_label(temp)
        
        Parameter_List.append(temp)
        
    pool = Pool(processes = CORES)
    
    pool.map(clean_label, Parameter_List)
    
    pool.close()
        
    
    
def get_nanowell_numbers(OUTPUT_PATH, DATASET, BLOCK):
    meta_all_clean_fname = OUTPUT_PATH + DATASET + '/' + BLOCK + '/meta/a_centers_clean.txt'
    f = open(meta_all_clean_fname)
    temp = f.readlines()
    f.close()
    
    return(len(temp))    
    
def write_record(fname, labels):
    w = len(labels)
    f = open(fname,'w')
    for i in range(w):
        temp = labels[i]
        temp = [str(i) for i in temp]
        line = '\t'.join(temp) + '\n'
        f.writelines(line)
    f.close()


def clean_label(Args):
    
    # print(Args)
    
    label_FRCNN = Args[0]
    label_FRCNN_OUT = Args[1]
    nanowell = Args[2]
    frames = Args[3]
    
    detection_confidence_threshold = Args[5]     #0.85
    confinement_constraint_threshold = Args[4]   #0.85

    labels_E = []
    labels_T = []
    E_count = np.zeros(frames)
    T_count = np.zeros(frames)

    # step 1: load the detection results
    for t in range(1, frames + 1):
        fname = label_FRCNN + 'imgNo' + str(nanowell) + '/' + 'imgNo' + str(nanowell) + '_t' + str(t) + '.txt'
        E = []
        T = []


        Head = ['ID', 'X', 'Y', 'W', 'H', 'Type', 'Score']
        E.append(Head)
        T.append(Head)

        f = open(fname)
        lines = f.readlines()
        f.close()

        for line in lines:
            x = line.rstrip().split('\t')
            x = [float(i) for i in x]

            if x[5] == 1:
                E.append(x)
                if x[6] > detection_confidence_threshold:
                    E_count[t-1] += 1

            if x[5] == 2:
                T.append(x)
                if x[6] > detection_confidence_threshold:
                    T_count[t-1] += 1

        labels_E.append(E)
        labels_T.append(T)

    # step 2: vote for E and T numbers
    E_hist = {}
    T_hist = {}

    E_count = E_count.tolist()
    T_count = T_count.tolist()

    E_unique = np.unique(E_count)
    T_unique = np.unique(T_count)

    E_max = 0
    E_max_number = '0'
    for EE in E_unique:
        E_hist[str(int(EE))] = E_count.count(EE)
        if E_hist[str(int(EE))] > E_max:
            E_max = E_hist[str(int(EE))]
            E_max_number = str(int(EE))

    T_max = 0
    T_max_number = '0'
    for TT in T_unique:
        T_hist[str(int(TT))] = T_count.count(TT)
        if T_hist[str(int(TT))] > T_max:
            T_max = T_hist[str(int(TT))]
            T_max_number = str(int(TT))

    E_confidence = float(E_max)/float(frames)
    T_confidence = float(T_max)/float(frames)

    # step 3: write the confinement constraint result in new folder with E/T separated
    if E_confidence > confinement_constraint_threshold and T_confidence > confinement_constraint_threshold:
        # attach the nanowell number to the selected list
        fname = label_FRCNN + 'selected_nanowells.txt'
        f = open(fname, 'a')
        line = str(nanowell) + '\t' + str(int(E_max_number))  +'\t' + str(int(T_max_number)) + '\n'
        f.writelines(line)
        f.close()

        # impose confinement constraint to cleaned labels, update the IDs
        labels_E_clean = []
        for t in range(1, frames + 1):
            #temp_E1 = np.asarray(labels_E[t-1][1:])
            #temp_E2 = temp_E1[(-temp_E1[:,6]).argsort()]
            temp_E2 = labels_E[t-1][1:]
            temp_E = []
            for i in range(int(E_max_number)):
                try:
                    temp_E.append(temp_E2[i])
                except:
                    temp_E.append([1.0, 1.0, 1.0, 3.0, 3.0, 1.0, 0.0000])
                temp_E[-1][0] = i + 1
            labels_E_clean.append(temp_E)

        labels_T_clean = []
        for t in range(1, frames + 1):
            #temp_T1 = np.asarray(labels_T[t-1][1:])
            #temp_T2 = temp_T1[(-temp_T1[:,6]).argsort()]
            temp_T2 = labels_T[t-1][1:]
            temp_T = []
            
            for i in range(int(T_max_number)):
                try:
                    temp_T.append(temp_T2[i])
                except:
                    temp_T.append([1.0, 1.0, 1.0, 3.0, 3.0, 2.0, 0.0000])
                temp_T[-1][0] = i + 1
            labels_T_clean.append(temp_T)

        # write the labels to imgNoxxNew folder
        for t in range(1, frames + 1):
            os.system('mkdir ' + label_FRCNN_OUT + 'imgNo' + str(nanowell))
            fname_E = label_FRCNN_OUT + 'imgNo' + str(nanowell) + '/imgNo' + str(nanowell) + 'E_t' + str(t) + '.txt'
            fname_T = label_FRCNN_OUT + 'imgNo' + str(nanowell) + '/imgNo' + str(nanowell) + 'T_t' + str(t) + '.txt'
            write_record(fname_E, labels_E_clean[t-1])
            write_record(fname_T, labels_T_clean[t-1])