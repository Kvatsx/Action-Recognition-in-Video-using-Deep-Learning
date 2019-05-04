#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
# os.chdir('drive/My Drive')


# In[2]:


# !ls "CV/ActionInVideo/ForFinalSubmission"


# In[3]:


# os.chdir('CV/ActionInVideo/ForFinalSubmission')


# In[4]:


# !ls


# In[5]:


# LOCATION = "/content/drive/My Drive/CV/ActionInVideo/ForFinalSubmission/"


# In[6]:


# !pwd


# In[7]:


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


# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[9]:


def recordInfo(logFile,data,header=None):
    if os.path.exists(logFile):
        with open(logFile, 'a') as log:
            log.write(data)
    else:
        with open(logFile, 'w') as log:
            log.write(header)
            log.write(data)
            
def savePickle(name, toSave):
    file = open(name, 'wb')
    pickle.dump(toSave, file)
    file.close()

def loadPickle(name):
    file = open(name, 'rb')
    data = pickle.load(file)
    file.close()
    return data


# In[10]:


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


# In[11]:


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


# In[12]:


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


# In[13]:


def weight_transform(modelDict, pretrainedDict, channel):
    weightDict  = {k:v for k, v in pretrainedDict.items() if k in modelDict}
    w3 = pretrainedDict['conv1.weight']
    if channel == 3:
        wt = w3
    else:
        S=0
        for i in range(3):
            S += w3[:,i,:,:]
        avg = S/3.
        newW3 = torch.FloatTensor(64,channel,7,7)
        for i in range(channel):
            newW3[:,i,:,:] = avg.data
        wt = newW3
    weightDict['conv1_custom.weight'] = wt
    modelDict.update(weightDict)
    return modelDict


def conv3x3(inChannels, outChannels, stride=1):
    return nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride,padding=1, bias=False)

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, nb_classes=10, channel=20):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.makeLayer(block, 64, layers[0])
        self.layer2 = self.makeLayer(block, 128, layers[1], stride=2)
        self.layer3 = self.makeLayer(block, 256, layers[2], stride=2)
        self.layer4 = self.makeLayer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_custom = nn.Linear(512 * block.expansion, nb_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def makeLayer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc_custom(x)
        return out
    
def resnet34(pretrained=False, channel= 3, **kwargs):
    model = ResNet(ResidualBlock, [3, 4, 6, 3], nb_classes=10, channel=channel, **kwargs)
    if pretrained:
       pretrainedDict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')                 
       modelDict = model.state_dict()
       modelDict=weight_transform(modelDict, pretrainedDict, channel)
       model.load_state_dict(modelDict)
    return model


# In[14]:


class SpatialCNN():
    def __init__(self, lr, trainLoader, valLoader, test_video, numEpochs = 200, batchSize = 8, resume='spatialModel'):
        self.numEpochs=numEpochs
        self.lr=lr
        self.batchSize=batchSize
        self.resume=resume
        self.startEpoch=0
        self.trainLoader=trainLoader
        self.valLoader=valLoader
        self.best_prec1=0
        self.test_video=test_video
        self.build_model()

    def build_model(self):
        self.model = resnet34(pretrained= True, channel=3).to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    
    def train(self):
        if self.resume:
            if os.path.isfile(self.resume):
                checkpoint = torch.load(self.resume)
                self.startEpoch = checkpoint['epoch'] + 1
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})".format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        cudnn.benchmark = True
        
        with tqdm(total = self.numEpochs * len(self.trainLoader)) as pbar:
            for self.epoch in range(self.startEpoch, self.numEpochs):
                print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.numEpochs))
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                top1 = AverageMeter()
                top5 = AverageMeter()
                self.model.train()    
                end = time.time()
                for i, (data_dict,label) in enumerate(self.trainLoader):
                    pbar.update(1)
                    data_time.update(time.time() - end)
                    label = label.to(device)
                    target_var = Variable(label).to(device)
                    output = Variable(torch.zeros(len(data_dict['img1']),10).float()).to(device)
                    for i in range(len(data_dict)):
                        key = 'img'+str(i)
                        data = data_dict[key]
                        input_var = Variable(data).to(device)
                        output += self.model(input_var)
                    loss = self.criterion(output, target_var)
                    prec1, prec5 = self.accuracy(output.data, label, topk=(1, 5))
                    losses.update(loss.item(), data.size(0))
                    top1.update(prec1.item(), data.size(0))
                    top5.update(prec5.item(), data.size(0))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_time.update(time.time() - end)
                    end = time.time()
                header = 'Epoch,Loss,Prec@1,Prec@5\n'
                info = '{0},{1},{2},{3}\n'.format(self.epoch,round(losses.avg,5),round(top1.avg,4),round(top5.avg,4))
                recordInfo('spatialTrainingLog.txt', info, header)
                print(header,info)
                
                if self.epoch % 1 == 0:
                    prec1, val_loss = self.validate()
                    is_best = prec1 > self.best_prec1
                    self.scheduler.step(val_loss)
                    if is_best:
                        self.best_prec1 = prec1
                        with open('spatialVideoPreds','wb') as f:
                            pickle.dump(self.dictVideoLevelPreds,f)
                        f.close()
                torch.save({ 'epoch': self.epoch, 'state_dict': self.model.state_dict(),'best_prec1': self.best_prec1, 'optimizer' : self.optimizer.state_dict() }, self.resume)
            
    def evaluate(self):
        self.epoch = 0
        prec1, val_loss = self.validate()
        return prec1, val_loss
    
    def accuracy(self,output, target, topk=(1,)):
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


    def validate(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.numEpochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.eval()
        self.dictVideoLevelPreds={}
        end = time.time()
        with tqdm(total =  len(self.valLoader)) as pbar:
            for i, (keys,data,label) in enumerate(self.valLoader):
                pbar.update(1)
                label = label.to(device)
                with torch.no_grad():
                    data_var = Variable(data).to(device)
                    label_var = Variable(label).to(device)

                output = self.model(data_var)
                batch_time.update(time.time() - end)
                end = time.time()
                preds = output.data.cpu().numpy()
                numData = preds.shape[0]
                for j in range(numData):
                    videoName = keys[j].split('/',1)[0]
                    if videoName not in self.dictVideoLevelPreds.keys():
                        self.dictVideoLevelPreds[videoName] = preds[j,:]
                    else:
                        self.dictVideoLevelPreds[videoName] += preds[j,:]
        videoTop1, videoTop5, videoLoss = self.frameToVideoLevelAccuracy()
        header = 'Epoch,Loss,Prec@1,Prec@5\n'
        info = '{0},{1},{2},{3}\n'.format(self.epoch,np.round(videoLoss,5),np.round(videoTop1,3),np.round(videoTop5,3))
        recordInfo('spatialValLog.txt', info, header)
        print(header,info)
        return videoTop1, videoLoss

    def frameToVideoLevelAccuracy(self):
        correct = 0
        videoLevelPreds = np.zeros((len(self.dictVideoLevelPreds),10))                                                 # needs some fixing
        videoLevelLabels = np.zeros(len(self.dictVideoLevelPreds))
        ii=0
        for name in sorted(self.dictVideoLevelPreds.keys()):
            preds = self.dictVideoLevelPreds[name]
            label = int(self.test_video[name])-1
            videoLevelPreds[ii,:] = preds
            videoLevelLabels[ii] = label
            ii+=1         
            if np.argmax(preds) == (label):
                correct+=1
        #top1 top5
        videoLevelLabels = torch.from_numpy(videoLevelLabels).long()
        videoLevelPreds = torch.from_numpy(videoLevelPreds).float()
        top1,top5 = self.accuracy(videoLevelPreds, videoLevelLabels, topk=(1,5))
        loss = self.criterion(Variable(videoLevelPreds).to(device), Variable(videoLevelLabels).to(device))     
        top1 = float(top1.numpy())
        top5 = float(top5.numpy())
        return top1,top5,loss.data.cpu().numpy()


# In[ ]:





# In[15]:


splitter = DataSplitter()
data_loader = spatialDataloader(splitter)
trainLoader, valLoader, test_video = data_loader.getData()

model = SpatialCNN(lr=1e-4,trainLoader=trainLoader, valLoader=valLoader,test_video=test_video)
model.train()


# In[ ]:


# import cv2
# x = cv2.imread('Data/jpegs_256/v_BasketballDunk_g25_c02/frame000035.jpg')
# os.path.exists('Data/jpegs_256/v_BasketballDunk_g25_c02')


# In[ ]:





# In[ ]:





# In[ ]:




