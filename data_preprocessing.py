# -*- coding: utf-8 -*-

import os
import sys
import cv2
import glob
import numpy as np
import json
import itertools
from collections import defaultdict
import unittest
from matplotlib import pyplot as plt

from path_constants import CITYSCAPESPATH


def get_cities(img_dir='disparity'):
    path_left_imgs = os.path.normpath(os.path.join(CITYSCAPESPATH, img_dir))
    cities = {mode: os.listdir(path_left_imgs + '/' + mode)
                for mode in ['train', 'test', 'val']}
    return cities


def ReadFilePaths(mode, city, subdir, file_type='.png'):
    """returns list of paths to files
    mode --- str, ['train', 'test', 'val']
    city --- str, city name
    subdir --- str, ['disparity', 'leftImg8bit', 'gtFine', etc.]
    file_type --- str, ['.png', '.json']
    """

    # absolute path to the directory with disparities
    path_disp_dir = os.path.normpath(os.path.join(CITYSCAPESPATH, subdir, mode, city))
    # list of absolute paths to images
    path_imgs = glob.glob(path_disp_dir + '/*' + file_type)
    path_imgs.sort()
    print '({}, {}): number of pngs is {}'.format(mode,city, len(path_imgs))
    return path_imgs

    
# extract the polygons correspond to the label
def ExtractPolygons(json_file, label='road'):
    with open(json_file) as anno:
        data = json.load(anno)
    return [d['polygon'] for d in data["objects"] if d['label'] == label]


# make disparitis = 0 for the road points
def ExcludeRoadDisp(img, anno_img, pixels):
    """pixels is a list of lists [[0, 0, 0], [255, 255, 255]]"""
    masked_img = img.copy()
    for pixel in pixels:
        excl_road = (anno_img != np.array([[pixel]]))[:, :, 0].astype(np.uint8)
        masked_img = cv2.bitwise_and(masked_img, masked_img, mask=excl_road)

    return masked_img

def ExcludeNoise(img):
    pass

def CropRoI(img):
    y_shape, x_shape, _ = img.shape

    X = np.zeros(4)
    Y = np.zeros(4)

    #top left
    X[0], Y[0] = x_shape // 4, y_shape // 3
    
    #bottom left
    X[1], Y[1] = 0, y_shape
    
    #bottom right
    X[2], Y[2] = x_shape, Y[1]

    #top right
    X[3], Y[3] = 4 * x_shape // 5, Y[0]
    
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    roi_corners = np.array([zip(X, Y)], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    # apply the mask
    masked_image = cv2.bitwise_and(img, img, mask=mask)

    # save the result
    #cv2.imwrite('image_masked.png', masked_image)

    return masked_image


def Plot(img):
    plt.figure(figsize=(15,7))
    plt.imshow(img)

    
def VisualPredDisp (pred_disp, disp_map, img):
    """
    Input:
	pred_disp -- value of predicted disparity
	disp_map -- disparity disp_map
	img
    Output:
	vis_img -- img for visualization with highlighted areas
    """
    
    error = pred_disp * 0.1
    color = [128,64,128]
    vis_img = img.copy()
    x = np.arange(disp_map.shape[1])
    y = np.arange(disp_map.shape[0])
    
    for i,j in itertools.product(y,x):
        if pred_disp - error <= disp_map[i,j,0] <= pred_disp + error:
            vis_img[i,j,:] = color
           
    return vis_img


class TestDataLoading(unittest.TestCase):
    """Здесь могли быть ваши юнит-тесты ;)"""
    pass

if __name__ == '__main__':
    unittest.main()
