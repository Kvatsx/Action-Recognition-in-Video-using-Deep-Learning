#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[3]:


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


# In[4]:


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


splitter = DataSplitter()
trainVideo,testVideo = splitter.splitTestTrain()
print(len(trainVideo),len(testVideo))  # ApplyEyeMakeup_g08_c01': 1  


# In[5]:


class motionDataset(Dataset):  
    def __init__(self, dic, inChannel, rootDir = './Data', mode = 'train', transform=None):
        #Generate a 16 Frame clip
        self.keys=list(dic.keys())
        self.values=list(dic.values())
        self.rootDir = rootDir
        self.transform = transform
        self.mode=mode
        self.inChannel = inChannel
        self.imgRows=224
        self.imgCols=224

    def stackImages(self):
        name = 'v_'+self.video
        u = self.rootDir+ 'u/' + name
        v = self.rootDir+ 'v/'+ name
        
        flow = torch.FloatTensor(2*self.inChannel,self.imgRows,self.imgCols)
        i = int(self.clips_idx)


        for j in range(self.inChannel):
            idx = i + j
            idx = str(idx)
            frame_idx = 'frame'+ idx.zfill(6)
            horiImage = u +'/' + frame_idx +'.jpg'
            vertiImage = v +'/' + frame_idx +'.jpg'
            
            imgHori=(Image.open(horiImage))
            imgVerti=(Image.open(vertiImage))

            H = self.transform(imgHori)
            V = self.transform(imgVerti)

            
            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V      
            imgHori.close()
            imgVerti.close()  
        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        
        if self.mode == 'train':
            self.video, nb_clips = self.keys[idx].split('-')
            self.clips_idx = random.randint(1,int(nb_clips))
        elif self.mode == 'val':
            self.video,self.clips_idx = self.keys[idx].split('-')
        
        label = self.values[idx]
        label = int(label)-1 
        data = self.stackImages()

        if self.mode == 'train':
            sample = (data,label)
        elif self.mode == 'val':
            sample = (self.video,data,label)
        return sample


# In[6]:


class motionDataLoader():
    def __init__(self,splitter, batchSize = 8, numWorkers = 0, inChannel = 10,  rootPath = './Data/'):
        self.batchSize=batchSize
        self.numWorkers = numWorkers
        self.frame_count={}
        self.inChannel = inChannel
        self.dataPath=rootPath + 'tvl1_flow/'
        # split the training and testing videos
        splitter = splitter
        self.trainVideo, self.testVideo = splitter.splitTestTrain()
        with open(rootPath+'/frame_count.pickle','rb') as file:
            dic_frame = pickle.load(file)
        file.close()
        for line in dic_frame :
            videoname = line.split('_',1)[1].split('.',1)[0]
            self.frame_count[videoname]=dic_frame[line] 
        
    def getData(self):
        trainLoader = self.getTrainLoader()
        valLoader = self.getValLoader()
        return trainLoader, valLoader, self.testVideo

                            
    def getTrainLoader(self):
        self.dic_video_train={}
        for video in self.trainVideo:
            nb_clips = self.frame_count[video]-10+1
            key = video +'-' + str(nb_clips)
            self.dic_video_train[key] = self.trainVideo[video] 
        training_set = motionDataset(dic=self.dic_video_train, inChannel=self.inChannel, rootDir=self.dataPath,
            mode='train',
            transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            ]))
        print('==> Training data :',len(training_set),' videos',training_set[1][0].size())
        trainLoader = DataLoader(
            dataset=training_set, 
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=self.numWorkers,
            pin_memory=True
            )
        return trainLoader

    def getValLoader(self):
        self.dic_test_idx = {}
        #print len(self.testVideo)
        for video in self.testVideo:
            n,g = video.split('_',1)

            sampling_interval = int((self.frame_count[video]-10+1)/19)
            for index in range(19):
                clip_idx = index*sampling_interval
                key = video + '-' + str(clip_idx+1)
                self.dic_test_idx[key] = self.testVideo[video]
        validation_set = motionDataset(dic= self.dic_test_idx, inChannel=self.inChannel, rootDir=self.dataPath ,
            mode ='val',
            transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            ]))
        print('==> Validation data :',len(validation_set),' frames',validation_set[1][1].size())
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.batchSize, 
            shuffle=False,
            num_workers=self.numWorkers)
        return val_loader
    
dataloader = motionDataLoader(splitter)
trainLoader,valLoader,testVideo = dataloader.getData()
print(len(trainLoader))


# In[7]:


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
       modelDict =weight_transform(modelDict, pretrainedDict, channel)
       model.load_state_dict(modelDict)
    return model


# In[14]:


class MotionCNN():
    def __init__(self, lr, trainLoader, testLoader,test_video,resume = 'motionModel', endEpochs = 200, batchSize =8,channel = 10*2):
        self.endEpochs=endEpochs
        self.lr=lr
        self.batchSize=batchSize
        self.resume=resume
        self.startEpoch=0
        self.trainLoader=trainLoader
        self.testLoader=testLoader
        self.best_prec1=0
        self.channel=channel
        self.test_video=test_video
        self.build_model()


    def build_model(self):
        #build model
        self.model = resnet34(pretrained= True, channel=self.channel).to(device)
        #print self.model
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
        
        
    def evaluate(self):
        self.epoch=0
        prec1, val_loss = self.validate_1epoch()
        return prec1, val_loss
    
    def train(self):
        if self.resume:
            if os.path.isfile(self.resume):
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        cudnn.benchmark = True
        
        with tqdm(total = self.endEpochs * len(self.trainLoader)) as pbar:
            for self.epoch in range(self.startEpoch, self.endEpochs):
                print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.endEpochs))
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                top1 = AverageMeter()
                top5 = AverageMeter()
                #switch to train mode
                self.model.train()    
                end = time.time()
                # mini-batch training
                for i, (data,label) in enumerate(self.trainLoader):
                    pbar.update(1)
                    # measure data loading time
                    data_time.update(time.time() - end)

                    label = label.to(device)
                    input_var = Variable(data).to(device)
                    target_var = Variable(label).to(device)

                    # compute output
                    output = self.model(input_var)
                    loss = self.criterion(output, target_var)

                    # measure accuracy and record loss
                    prec1, prec5 = self.accuracy(output.data, label, topk=(1, 5))
                    losses.update(loss.item(), data.size(0))
                    top1.update(prec1.item(), data.size(0))
                    top5.update(prec5.item(), data.size(0))

                    # compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                header = 'Epoch,Loss,Prec@1,Prec@5\n'
                info = '{0},{1},{2},{3}\n'.format(self.epoch,round(losses.avg,5),round(top1.avg,4),round(top5.avg,4))
                recordInfo('motionTrainingLog.txt', info, header)
                print(header,info)
                

                if self.epoch % 1 == 0:
                    prec1, val_loss = self.validate()
                    is_best = prec1 > self.best_prec1
                    #lr_scheduler
                    self.scheduler.step(val_loss)
                    # save model
                    if is_best:
                        self.best_prec1 = prec1
                        with open('motionVideoPreds','wb') as f:
                            pickle.dump(self.dictToVideoLevelPreds,f)
                        f.close() 
                torch.save({ 'epoch': self.epoch, 'state_dict': self.model.state_dict(),'best_prec1': self.best_prec1, 'optimizer' : self.optimizer.state_dict() }, self.resume)

       
    def validate(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.endEpochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.eval()
        self.dictToVideoLevelPreds={}
        end = time.time()
        with tqdm(total =  len(self.testLoader)) as pbar:
            for i, (keys,data,label) in enumerate(self.testLoader):
                pbar.update(1)
                label = label.to(device)
                with torch.no_grad():
                    data_var = Variable(data).to(device)
                    label_var = Variable(label ).to(device)
                output = self.model(data_var)
                batch_time.update(time.time() - end)
                end = time.time()
                preds = output.data.cpu().numpy()
                nb_data = preds.shape[0]
                for j in range(nb_data):
                    videoName = keys[j].split('-',1)[0] # ApplyMakeup_g01_c01
                    if videoName not in self.dictToVideoLevelPreds.keys():
                        self.dictToVideoLevelPreds[videoName] = preds[j,:]
                    else:
                        self.dictToVideoLevelPreds[videoName] += preds[j,:]

        #Frame to video level accuracy
        video_top1, video_top5, video_loss = self.frameToVideoLevelAccuracy()
        header = 'Epoch,Loss,Prec@1,Prec@5\n'
        info = '{0},{1},{2},{3}\n'.format(self.epoch,np.round(video_loss,5),np.round(video_top1,3),np.round(video_top5,3))
        recordInfo('motionValLog.txt', info, header)
        print(header,info)
        return video_top1, video_loss
    
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
    

    def frameToVideoLevelAccuracy(self):
        correct = 0
        videoLevelPreds = np.zeros((len(self.dictToVideoLevelPreds),10))
        videoLevelLabels = np.zeros(len(self.dictToVideoLevelPreds))
        ii=0
        for key in sorted(self.dictToVideoLevelPreds.keys()):
            name = key.split('-',1)[0]

            preds = self.dictToVideoLevelPreds[name]
            label = int(self.test_video[name])-1
                
            videoLevelPreds[ii,:] = preds
            videoLevelLabels[ii] = label
            ii+=1         
            if np.argmax(preds) == (label):
                correct+=1

        #top1 top5
        videoLevelLabels = torch.from_numpy(videoLevelLabels).long()
        videoLevelPreds = torch.from_numpy(videoLevelPreds).float()

        loss = self.criterion(Variable(videoLevelPreds).to(device), Variable(videoLevelLabels).to(device))    
        top1,top5 = self.accuracy(videoLevelPreds, videoLevelLabels, topk=(1,5))     
                            
        top1 = float(top1.numpy())
        top5 = float(top5.numpy())
            
        return top1,top5,loss.data.cpu().numpy()


# In[15]:


trainLoader, valLoader, test_video = dataloader.getData()
model = MotionCNN(lr=1e-2,trainLoader=trainLoader, testLoader=valLoader,test_video=test_video)
model.train()


# In[ ]:





# In[ ]:





# In[ ]:




