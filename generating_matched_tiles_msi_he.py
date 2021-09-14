# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:34:02 2019

@author: m.beuque
"""
import numpy as np
import cv2
from pyimzML.ImzmlParser import ImzMLParser
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from utils import extract_transform_matrix,process_spotlist,polygon_extraction
from tqdm import tqdm
import os

def generator_msi_data(slide, plot, generate,main_path):
    # main_path : initialize the paths with the MSI slide folder
    
    #find all necessary files
    slide_files = os.listdir(os.path.join(main_path,slide))
    sample_name = slide
    path_slide = os.path.join(main_path, slide)
    for file in slide_files:
        if file.endswith('imzML'):
            imzMLfile = os.path.join(path_slide,file)
        elif file.endswith('mis'):
            path_mis = os.path.join(path_slide,file)
        elif file.endswith('txt'):
            path_spotlist = os.path.join(path_slide,file)
        elif file.endswith('jpg'):
            path_jpg = os.path.join(path_slide,file)
    histo_img = cv2.imread(path_jpg)
    path_tiles_dir = os.path.join(path_slide, 'tiles')
    if not os.path.exists(path_tiles_dir):
        os.makedirs(path_tiles_dir)
    
    ##get informations from IMZML file
    imzML = ImzMLParser(imzMLfile)
    
    nr_of_datapoints = len(imzML.mzLengths)
    nr_of_spectra = imzML.mzLengths[0] 
    array_coord = np.array(imzML.coordinates)
    cols  = max(array_coord[:,0])
    rows  = max(array_coord[:,1])
    
    ##reading and storing MSI data and read pixel coordinates
    msidata = np.zeros((nr_of_datapoints,nr_of_spectra))
    xy_positions = np.zeros((nr_of_datapoints, 2))
    for idx, (x,y,z) in enumerate(imzML.coordinates):
        mzs, intensities = imzML.getspectrum(idx)
        msidata[idx, :] = intensities
        xy_positions[idx,:]= [x, y]
    msidata = np.transpose(msidata)
    
    ##calculate total ion count values for normalization
    tics = np.sum(msidata,0)
    canvas = np.zeros((rows,cols))
    for i in range(nr_of_datapoints):
        canvas[int(xy_positions[i,1]-1), int(xy_positions[i,0]-1)] = tics[i]
    
    if plot : 
        plt.figure(figsize=(12,12))
        plt.imshow(np.squeeze(canvas))
        plt.title('Canvas')
        plt.legend()
        plt.show()
        
    if plot : 
        plt.figure(figsize=(12,12))
        plt.imshow(np.squeeze(histo_img))
        plt.title('Canvas')
        plt.legend()
        plt.show()
        
    #normalize msi_data
    repeated_tics =np.repeat(tics, list( msidata.shape)[0], axis = 0)
    repeated_tics = repeated_tics.reshape(msidata.shape)
    msidata = np.divide(msidata, repeated_tics)
    
    del repeated_tics
   
    ## create geometric transformations
    fixed_points_optical, moving_points_motor,moving_points_optical,fixed_points_histo = extract_transform_matrix(path_mis)
    motor2optical = cv2.getAffineTransform(np.float32(moving_points_motor), np.float32(fixed_points_optical))
    optical2histo =  cv2.getAffineTransform(np.float32(moving_points_optical), np.float32(fixed_points_histo))
    
    spotlist = process_spotlist(path_spotlist)
    
    moving_points_msi = np.float32(spotlist[:,[2,3]][[0, int(nr_of_datapoints/2), nr_of_datapoints -1]])
    fixed_points_motor = np.float32(spotlist[:,[0,1]][[0, int(nr_of_datapoints/2), nr_of_datapoints -1]])
    msi2motor = cv2.getAffineTransform(moving_points_msi, fixed_points_motor)
    
    new_msi2motor = np.concatenate((np.transpose(msi2motor), np.array([[0],[0],[1]])), axis = 1)
    new_motor2optical = np.concatenate((np.transpose(motor2optical), np.array([[0],[0],[1]])), axis = 1)
    new_optical2histo = np.concatenate((np.transpose(optical2histo), np.array([[0],[0],[1]])), axis = 1)
    
    
    new_msi3Dhisto = np.dot(np.dot(new_msi2motor,new_motor2optical), new_optical2histo)
    new_msi2Dhisto = new_msi3Dhisto[:,0:2]
    
    #overlay MSI image and histopathology image
    img_rows, img_cols, ch = histo_img.shape
    transform_img = cv2.warpAffine(canvas,np.transpose(new_msi2Dhisto),(img_cols,img_rows))
    overlay = cv2.addWeighted(np.float64(histo_img[:,:,1]), 1, transform_img, 0.01, 0) 
    
    if plot :
        plt.figure()
        plt.imshow(transform_img)
        plt.title('resized canvas')
        plt.legend()
        plt.show()
        plt.figure()
        plt.imshow(overlay)
        plt.title('resized canvas and overlay')
        plt.legend()
        plt.show()
    
    #create dictionnary with raster number and label
    dict_roi = polygon_extraction(path_mis)
    # consider region of interest 
    #find maximum coordinates for polygon
    max_x = 0
    max_y = 0
    for i, poly in enumerate(list(dict_roi.keys())) : 
        polygon = np.array(dict_roi[poly])
        if max(polygon[:,0]) > max_x :
            max_x = max(polygon[:,0])
        if max(polygon[:,1]) > max_y :
            max_y = max(polygon[:,1])
            
    img_test = Image.new('L', (max_x,max_y), 0)
    for i, poly in enumerate(list(dict_roi.keys())) : 
        polygon = dict_roi[poly]
        ImageDraw.Draw(img_test).polygon(polygon, outline=i+1, fill=i+1)
    mask = np.array(img_test)
    transform_poly = cv2.warpAffine(mask,np.transpose(new_optical2histo[:,0:2]),(img_cols,img_rows))
    
    
    if plot :
        plt.figure()
        plt.imshow(mask)
        plt.title('drawing of all the regions of interest')
        plt.legend()
        plt.show()
        plt.figure()
        plt.imshow(transform_poly)
        plt.title('drawing of all the regions of interest resized')
        plt.legend()
        plt.show()
    
    
    if generate:
        #create the information table
        collect_points = []
        for i in tqdm(range(nr_of_datapoints)):
            y_ms, x_ms = int(xy_positions[i,1]-1), int(xy_positions[i,0]-1)
            x_histo = x_ms*np.transpose(new_msi2Dhisto)[1,0] + y_ms*np.transpose(new_msi2Dhisto)[1,1] + np.transpose(new_msi2Dhisto)[1,2]
            y_histo = x_ms*np.transpose(new_msi2Dhisto)[0,0] + y_ms*np.transpose(new_msi2Dhisto)[0,1] + np.transpose(new_msi2Dhisto)[0,2]
            collect_points.append([x_histo, y_histo])
            
        collect_points = np.array(collect_points)
        labels = []
        size_image = 96
        tile = np.float64(histo_img)
        name_rois =list( dict_roi.keys())
        histo_img = np.float64(histo_img)
        with open(os.path.join(r'.\msi_tables','table_for_sample_' + sample_name + '.txt'), "a", encoding="utf-8") as file:
            header = list(np.array(mzs, dtype = 'U25')) + ['label', 'image_name', 'x', 'y']
            header = ';'.join(header)
            file.write(header)
            file.write("\n")
            file.close()
        for i in tqdm(range(nr_of_datapoints)):
            with open(os.path.join(r'.\msi_tables','table_for_sample_' + sample_name + '.txt'), "a", encoding="utf-8") as file:
                if int(collect_points[i,1] - size_image//2) > 0 and int(collect_points[i,0] - size_image//2) > 0 and collect_points[i,1] < img_cols and collect_points[i,0] < img_rows:
                    labeled_tile = transform_poly[int(collect_points[i,0] - size_image//2): int(collect_points[i,0] + size_image//2) ,int(collect_points[i,1] - size_image//2): int(collect_points[i,1] + size_image//2) ]
                    length, width = labeled_tile.shape
                    temp_label = transform_poly[int(collect_points[i,0]), int(collect_points[i,1])] 
                    labels.append(temp_label)
                    if temp_label > 0:
                        temp_title = "tile_"+ str(int(xy_positions[i,1]-1)) + "_" + str(int(xy_positions[i,0]-1)) + ".jpg"
                        if not os.path.isfile(os.path.join(path_tiles_dir,temp_title)):
                            cv2.imwrite(os.path.join(path_tiles_dir,temp_title), tile[int(collect_points[i,0] - size_image//2): int(collect_points[i,0] + size_image//2) ,int(collect_points[i,1] - size_image//2): int(collect_points[i,1] + size_image//2) ])
                        row = np.concatenate((msidata[:,i], np.array([name_rois[temp_label-1], temp_title, int(xy_positions[i,1]-1), int(xy_positions[i,0]-1)])), axis = None)
                        row = ';'.join(list(row))
                        file.write(row)
                        file.write("\n")
                        file.close()
        del histo_img
    return('sample ' + sample_name + ' done')
    

main_path = '.'    
for slide in os.listdir(main_path):
    done = generator_msi_data(slide, plot = False, generate = True,main_path)
    print(done)