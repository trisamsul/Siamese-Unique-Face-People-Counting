import cv2
import os
import numpy as np
import glob


def load_data(root_path, image_size, mode, normalized):
    images = []
    labels = []
    labels_bin = []
    img_names = []
    cls = []

    classes = glob.glob(os.path.join(root_path,'*'))
    classes =[os.path.basename(w) for w in classes]

    print('Load dataset...')
    for fields in classes:   
        index = classes.index(fields)
        print('read directory: {} (index: {})'.format(fields, index))
        path = os.path.join(root_path,fields, '*jpg')
        files = glob.glob(path)
        for fl in files:
            print('read file: '+fl)
            if mode==0:
                image = cv2.imread(fl,0)
                image = cv2.equalizeHist(image)
            else:
                image = cv2.imread(fl)
            
            image = cv2.resize(image, image_size,0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            if normalized:
                image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels_bin.append(label)
            labels.append(index)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    labels_bin = np.array(labels_bin)

    return images, labels, labels_bin, img_names, cls

def get_crossval_idx(label, k, shuffle):
    idx_list = np.zeros(len(label))-1
    cls_list = set(label)
    rshuffle = np.arange(0,len(label))
    if shuffle:
        np.random.shuffle(rshuffle)       
    
    for c in cls_list:
        counter_k = 1
        for idx in rshuffle:
            l = label[idx]
            if c==l:
                idx_list[idx] = counter_k
                counter_k = counter_k+1
            if counter_k>k:
                counter_k=1
    idx_list = np.array(idx_list)

    return idx_list

def get_holdout_idx(label, composition, shuffle):
    idx_list = np.zeros(len(label))-1
    unique, counts = np.unique(label, return_counts=True)
    rshuffle = np.arange(0,len(label))
    if shuffle:        
        np.random.shuffle(rshuffle)

    for i in range(0,len(unique)):
        c = unique[i]
        count = counts[i]
        limit = composition * count
        counter_k = 1
        for idx in rshuffle:
            l = label[idx]
            if c==l:
                if counter_k <= limit:
                    idx_list[idx] = 1
                else:
                    idx_list[idx] = 2
                counter_k = counter_k+1

    idx_list = np.array(idx_list)

    return idx_list
            
            

