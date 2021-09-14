# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:02:33 2021

@author: m.beuque
"""
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from xml.dom import minidom
import xml.etree.ElementTree as ET



def extract_transform_matrix(path_mis):
    elmt_matrix = []
    # parse an xml file by name
    mydoc = minidom.parse(path_mis)
    
    items = mydoc.getElementsByTagName('TeachPoint')
     
    for elem in items:  
        temp = str(elem.firstChild.data)
        temp = temp.split(";")
        val = []
        val.append(list(temp[0].split(",")))
        val.append(list(temp[1].split(",")))
        elmt_matrix.append(val)
    
    elmt_matrix = np.array(elmt_matrix)
    fixed_points_optical = elmt_matrix[:3,0]
    moving_points_motor =  elmt_matrix[:3,1]
    moving_points_optical = elmt_matrix[3:,0]
    fixed_points_histo = elmt_matrix[3:,1]  
    
    return  fixed_points_optical, moving_points_motor,moving_points_optical,fixed_points_histo

def process_spotlist(path_spotlist) :
    f = open(path_spotlist) 
    new_spotlist = []
    for line in f:
        if '#' not in line:
            elmts = line.split()
            temp_x = elmts[2][4:7]
            temp_y = elmts[2][-3:]
            elmts.remove(elmts[2])
            elmts.insert(2,temp_y)
            elmts.insert(2, temp_x)
            new_spotlist.append(elmts)  
    spotlist = np.array(new_spotlist, dtype = int)
    f.close()
    return spotlist

def polygon_extraction(path_mis) :
    ##get rois data
    tree = ET.parse(path_mis)
    root = tree.getroot()
    
    dict_roi = {}
    for roi in root.iter('ROI'):
        name_temp = roi.get('Name')
        roi_list = []
        for point in roi.getchildren() :
            temp_xy = point.text.split(',')
            roi_list.append(tuple([ int(x) for x in temp_xy ]))
        #array_roi = np.array(roi_list)
        #array_roi = array_roi.astype(int)
        dict_roi[name_temp] = roi_list
    return dict_roi

def extract_center_pos(path_mis):
    mydoc = minidom.parse(path_mis)
    
    items = mydoc.getElementsByTagName('View')
    string_centerpos = items[0].attributes['CenterPos'].value
    center_pos = list(string_centerpos.split(","))
    return np.array(center_pos, dtype = int)

def assemble_dataset_supervised_learning(full_labels,list_dataset,path_data, data_type):
    frames = []
    labels= []
    for j,f in enumerate(list_dataset):
        temp_df = pd.read_csv(os.path.join(path_data,f))
        selected_labels=full_labels[full_labels["slide"] ==f[14:-4]]
        unified_labels = list(selected_labels["unified_label"])
        image_names = list(selected_labels["image_name"])
        list_index = []
        selected_names = []
        if data_type == "stroma":
            for i, elmt in enumerate(unified_labels):
                if elmt[-1] == "g":
                    labels.append("gland")
                    list_index.append(i)
                    selected_names.append(image_names[i])
                elif elmt[-1] == "t":
                    labels.append("epithelial tissue")
                    list_index.append(i)
                    selected_names.append(image_names[i])
        if data_type == "grade":
            for i, elmt in enumerate(unified_labels):
                if elmt == "lowgrade_g":
                    labels.append("low grade")
                    list_index.append(i)
                    selected_names.append(image_names[i])
                elif elmt == "highgrade_g":
                    labels.append("high grade")
                    list_index.append(i)
                    selected_names.append(image_names[i])
                elif elmt == "healthy_g":
                    labels.append("non-dysplasia")
                    list_index.append(i)
                    selected_names.append(image_names[i])

        temp_df = temp_df.iloc[list_index]
        if not temp_df.empty:
            #temp_df.insert(loc=0, column='labels', value=np.array(labels))
            #col_dataset = [j for i in range(len(temp_df.index))]
            print(f[14:-4] + " has " + str(range(len(temp_df.index))) + " values")
            col_dataset = [f[14:-4] for i in range(len(temp_df.index))]
            #temp_df.insert(loc=1, column='dataset_name', value=col_dataset) 
            temp_df.insert(loc=1, column='dataset_name', value=col_dataset) 
            temp_df.insert(loc=2, column='image_name', value=selected_names) 
            frames.append(temp_df)
    print("size frames: ", len(frames))
    full_dataset = pd.concat(frames)
    del full_dataset['Unnamed: 0']
    return full_dataset, labels

def print_confusion_matrix(y_true, y_pred, class_names, figsize = (6,5), fontsize=14,normalize=False):
    cm = confusion_matrix(y_true, y_pred,class_names)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    sns.set(font_scale=1.4)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    df_cm = pd.DataFrame(
        cm, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        if normalize:
            fmt = '.2f' 
            heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt,cmap="Blues",vmin=0, vmax=1)
        else:
            fmt = 'd' 
            heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt,cmap="Blues",vmax=max(np.sum((y_pred==class_names[0] )*(y_true==class_names[0])),np.sum((y_pred==class_names[1] )*(y_true==class_names[1]))),vmin=0)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.tight_layout()
    return fig



