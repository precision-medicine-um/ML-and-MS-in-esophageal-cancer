# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 10:27:47 2021

@author: m.beuque
"""
from matplotlib.image import imread
import cv2
import tqdm
import pandas as pd
import os
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from sklearn.utils import shuffle
import torch


def generate_dataset_tissue_type(main_path,path_data,SEED):
    data = pd.read_csv(path_data,sep = ',' )

    X = []
    y = []
    paths = []
    for slide in tqdm.tqdm(os.listdir(os.path.join(main_path, 'Slides'))):
        tile_path = os.path.join(main_path, 'Slides',slide,'tiles')
        gland = data[(data['labels']=='stroma') & (data['dataset_name']==slide)]['image_name']
        tissue = data[(data['labels']=='epithelial tissue') & (data['dataset_name']==slide)]['image_name']
        gland = list(gland)
        tissue = list(tissue)
        for image_path in gland:
            if os.path.isfile(os.path.join(tile_path, image_path)):
                X.append(imread(os.path.join(tile_path, image_path)))
                y.append(0)
                paths.append(os.path.join(tile_path, image_path))
        for image_path in tissue:
            if os.path.isfile(os.path.join(tile_path, image_path)):
                X.append(imread(os.path.join(tile_path, image_path)))
                y.append(1)
                paths.append(os.path.join(tile_path, image_path))

    X = np.array(X)
    for j, elmt in enumerate(X):
        if elmt.shape !=(96,96,3):
            X[j] = cv2.resize(elmt,(96,96),interpolation = cv2.INTER_CUBIC)
    X,y = shuffle(X,y,random_state=SEED)
    
    return X, y, paths

def generate_dataset_grade(main_path,path_data): 
    #path_data is the path to the csv containing the information of the ing or testing or validation dataset
    #main_path contains the folder "Slides" were the H&E tiles where stored
    data = pd.read_csv(path_data,sep = ',' )
    X = []
    y = []
    paths = []
    
    for slide in tqdm.tqdm(os.listdir(os.path.join(main_path, 'Slides'))):
        tile_path = os.path.join(main_path, 'Slides',slide,'tiles')
        healthy = data[(data['labels']=='non-dysplasia') & (data['dataset_name']==slide)]['image_name'] 
        lowgrade = data[(data['labels']=='low grade') & (data['dataset_name']==slide)]['image_name']
        highgrade = data[(data['labels']=='high grade') & (data['dataset_name']==slide)]['image_name']
        healthy = list(healthy)
        lowgrade = list(lowgrade)
        highgrade = list(highgrade)
        for image_path in healthy:
            if os.path.isfile(os.path.join(tile_path, image_path)):
                X.append(imread(os.path.join(tile_path, image_path)))
                y.append("non-dysplasia")
                paths.append(os.path.join(tile_path, image_path))
            else:
                print("error for non-dysplasia")
        for image_path in lowgrade:
            if os.path.isfile(os.path.join(tile_path, image_path)):
                X.append(imread(os.path.join(tile_path, image_path)))
                y.append("low grade")
                paths.append(os.path.join(tile_path, image_path))
            else :
                print("error for low grade")
        for image_path in highgrade:
            if os.path.isfile(os.path.join(tile_path, image_path)):
                X.append(imread(os.path.join(tile_path, image_path)))
                y.append("high grade")
                paths.append(os.path.join(tile_path, image_path))
            else :
                print("error for high grade")

    #rescale the images to the same size
    for j, elmt in enumerate(X):
        if elmt.shape !=(96,96,3):
            X[j] = cv2.resize(elmt,(96,96),interpolation = cv2.INTER_CUBIC)
    X = np.array(X)
    y = np.array(y)
    return X,y,paths

#regular dataset generation
class CancerDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,X_all, y_all, transform = transforms.Compose([transforms.CenterCrop(64),transforms.ToTensor()])):
        'Initialization'
        self.labels = y_all
        self.list_IDs = X_all
        self.transform = transform
        self.image_files_list = [str(s) for s in range(len(self.list_IDs))]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.list_IDs[index]
        X = self.transform(image=X)
        X = X['image']
        # Load data and get label
        y = self.labels[index]
        return X, y
    
def df_dl_features(X,paths,data_transforms,classifier):
    features = {}
    for i,temp_X in tqdm.tqdm(enumerate(X)):
        tensor_X=data_transforms(image=temp_X)
        tensor_X=tensor_X["image"]
        tensor_X.unsqueeze_(0)
        output=torch.flatten(classifier(tensor_X)).detach().numpy()
        features[paths[i]]=output.flatten()
    features=pd.DataFrame.from_dict(features)
    features=features.T
    return features