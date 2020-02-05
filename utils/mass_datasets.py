import sys
sys.path.append('./utils')
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
#import vision
import argparse
import torchvision


Utils_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# Utils_DIR = os.path.dirname(os.path.abspath(__file__))

class MASSBuilding(data.Dataset):
    def __init__(self,partition='Train',split='train'):
        self.dataset = os.path.join(Utils_DIR,'mass_buildings',partition)
        self.labelpath = os.path.join(self.dataset,'map','%s')
        self.imagepath = os.path.join(self.dataset,'sat','%s')

        infos = pd.read_csv(os.path.join(self.dataset,'{}.csv'.format(split)))
        #get img and label ids
        self.ids = infos['id'].tolist()

        #get label class
        self.nb_class =1
        img = imread((self.labelpath.replace("\\", "/")+'.tif' ) % self.ids[0])
        self.img_rows,self.img_cold = img.shape[:2]
    def __len__(self):
        return len(self.ids)

class msBD(MASSBuilding):
    def __getitem__(self,idx):
        img_id = self.ids[idx]
        img_sat = imread((self.imagepath.replace("\\", "/") +'.tiff')% img_id)
        img_sat = (img_sat/255).astype('float32')

        img_map = imread((self.labelpath.replace("\\","/") +'.tif' )% img_id)
        img_map = (np.expand_dims(img_map,-1)/255).astype('float32')
        img_map =(img_map/255).astype('float32')
        # print(img_map.shape)
        # print(img_sat.shape)

        img_sat = torch.from_numpy(np.transpose(img_sat,(2, 0, 1)))
        img_map = torch.from_numpy(np.transpose(img_map,(2, 0, 1)))
        return img_sat ,img_map

    def show(self, idx):
        img_sat = imread(self.imagepath % self.ids[idx])
        img_map = imread(self.labelpath % self.ids[idx])

        frame, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(11,5))
        frame.suptitle('Sample {} in mass_building Dataset'.format(idx))
        ax1.imshow(img_sat)
        ax2.imshow(img_map,'gray')
        plt.show()
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-partition',type=str,default='Train',help='partition within of the dataset')
    parser.add_argument('-split',type=str,default='train',help='split of the data within ["train","validation","test"]')
    args= parser.parse_args()

    satmapdata = msBD(args.partition,args.split)
    sat, map = satmapdata[0]
    satmapdata.show(0)