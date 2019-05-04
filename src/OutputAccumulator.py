#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm_notebook as tqdm
import shutil
from random import randint

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.utils.model_zoo as model_zoo
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle


# In[3]:


rgb_preds='backup/SpatialVideoPreds'
opf_preds = 'backup/MotionVideoPreds'

with open(rgb_preds,'rb') as f:
    rgb =pickle.load(f)
f.close()
with open(opf_preds,'rb') as f:
    opf =pickle.load(f)
f.close()


# In[7]:


class DataSplitter():
    def __init__(self, path = 'Data/'):
        self.path = path
        self.actionLabel={}
        with open(self.path+'classInd.txt') as f:
            lines = f.readlines()
            lines = [line.strip('\r\n') for line in lines]
        f.close()
        for line in lines:
            label,action = line.split(' ')
            if action not in self.actionLabel.keys():
                self.actionLabel[action]=label

    def splitTestTrain(self):
        self.trainVideo = self.file2_dic(self.path+'trainlist.txt')
        self.testVideo = self.file2_dic(self.path+'testlist.txt')
        print('number of train and test videos', len(self.trainVideo),len(self.testVideo))
        return self.trainVideo, self.testVideo

    def file2_dic(self,fname):
        with open(fname) as f:
            lines = f.readlines()
            content = [line.strip('\r\n') for line in lines]
        f.close()
        dic={}
        for line in lines:
            video = line.split('/',1)[1].split(' ',1)[0]   # v_ApplyEyeMakeup_g01_c03.avi
            key = video.split('_',1)[1].split('.',1)[0]    # ['v', 'ApplyEyeMakeup_g01_c03.avi'], ['ApplyEyeMakeup_g01_c03', 'avi']
            label = self.actionLabel[line.split('/')[0]]   # 1
            dic[key] = int(label)
        return dic
    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# if __name__ == '__main__':
#     splitter = DataSplitter()
#     train_video,test_video = splitter.splitTestTrain()
#     print(len(train_video),len(test_video))  # ApplyEyeMakeup_g08_c01': 1  


# In[8]:


class spatialDataset(Dataset):  
    def __init__(self, dic, rootDir= "Data/", mode = 'train', transform=None):
        self.keys = list(dic.keys())   # name of videos
        self.values = list(dic.values())   # number of frames - 10
        self.rootDir = rootDir+"jpegs_256/"
        self.mode =mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def loadImage(self,videoName, index):
        path = self.rootDir + 'v_'+videoName+'/'
        try:
            img = Image.open(path + 'frame' +str(index).zfill(6)+'.jpg')
        except:
            n = 1
            tempPath = path + 'frame' +str(index).zfill(6)+'.jpg'
            while not os.path.exists(tempPath):
                tempPath = path + 'frame' +str(index+n).zfill(6)+'.jpg'
                if n>0:
                    n= -1*n
                else:
                    n = -1*n +1
            img = Image.open(tempPath)
        transformed_img = self.transform(img)
        img.close()
        return transformed_img

    def __getitem__(self, idx):
        if self.mode == 'train':
            videoName, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            clips = []
            clips.append(random.randint(1, nb_clips//3))
            clips.append(random.randint(nb_clips//3, nb_clips*2//3))
            clips.append(random.randint(nb_clips*2//3, nb_clips+1))    # take three random frame from 3-halves of video
        elif self.mode == 'val':
            videoName, index = self.keys[idx].split(' ')
            frameIndex =abs(int(index))
        label = self.values[idx]
        label = int(label)-1
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.loadImage(videoName, index)    # img1 : IMAGE_DATA
            sample = (data, label)  # (3 frames , label)
        elif self.mode=='val':
            data = self.loadImage(videoName,frameIndex)
            sample = (videoName, data, label)   #  ApplyEyeMakeup_g08_c01 , IMAGE_DATA , label
        return sample


# In[9]:


class spatialDataloader():
    def __init__(self, dataSplitter, batchSize = 8, numWorkers=4, path = 'Data/'):
        self.batchSize=batchSize
        self.numWorkers=numWorkers
        self.dataPath=path
        self.frameCount ={}
        self.trainVideo, self.testVideo = dataSplitter.splitTestTrain()
        with open(self.dataPath+'frame_count.pickle','rb') as file:
            dicFrame = pickle.load(file)
        file.close()
        for line in dicFrame :
            videoname = line.split('_',1)[1].split('.',1)[0]
            self.frameCount[videoname]=dicFrame[line]

    def getData(self):
        trainLoader = self.getTrainLoader()
        valLoader = self.getValLoader()
        return trainLoader, valLoader, self.testVideo


    def getTrainLoader(self):
        self.dic_training={}
        for video in self.trainVideo:  # ApplyEyeMakeup_g08_c01': 1 
            nFrame = self.frameCount[video]-10+1
            key = video+' '+ str(nFrame)
            self.dic_training[key] = self.trainVideo[video]    # ApplyEyeMakeup_g08_c01 NUNBER_OF_FRAMES : 1
            
        transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
        training_set = spatialDataset(dic=self.dic_training, mode='train', transform= transform)
        print('==> Training data :',len(training_set),'frames')
        print(training_set[1][0]['img1'].size())

        trainLoader = DataLoader(
            dataset=training_set, 
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=self.numWorkers)
        return trainLoader

    def getValLoader(self):
        # for each video extract 19 frames and send it with label
        self.dic_testing={}
        for video in self.testVideo:  # ApplyEyeMakeup_g08_c01': 1
            nFrame = self.frameCount[video]-10+1
            interval = nFrame//19
            for i in range(19):
                key = video+' '+str(i*interval+1)
                self.dic_testing[key] = self.testVideo[video]  # ApplyEyeMakeup_g08_c01 FRAME_NUMBER : 1
                
        transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
        validation_set = spatialDataset(dic=self.dic_testing, mode='val', transform= transform)
        print('==> Validation data :',len(validation_set),'frames')
        valLoader = DataLoader(
            dataset=validation_set, 
            batch_size=self.batchSize, 
            shuffle=False,
            num_workers=self.numWorkers)
        return valLoader





# if __name__ == '__main__':
#     dataloader = spatialDataloader(splitter)
#     trainLoader,valLoader,test_video = dataloader.getData()
#     print(len(trainLoader))


# In[18]:


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batchSize = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batchSize))
    return res


# In[19]:


splitter = DataSplitter()
dataloader = spatialDataloader(splitter)
_,_,testVideo = dataloader.getData()

overallPreds = np.zeros((len(rgb.keys()),10))
labels = np.zeros(len(rgb.keys()))
correct=0
index=0
for name in sorted(rgb.keys()):   
    r = rgb[name]
    o = opf[name]
    label = int(testVideo[name])-1

    overallPreds[index,:] = (r+o)
    labels[index] = label
    index+=1         
    if np.argmax(r+o) == (label):
        correct+=1

labels = torch.from_numpy(labels).long()
overallPreds = torch.from_numpy(overallPreds).float()

top1,top5 = accuracy(overallPreds, labels, topk=(1,5))     

print(top1,top5)


# In[ ]:





# In[ ]:




