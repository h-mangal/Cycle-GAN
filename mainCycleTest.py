import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
import models
import dataLoader
from imageBuffer import imageBuffer
from PIL import Image
from glob import glob


def save_image(data, name):
    data = data.cpu().squeeze(0).numpy()
    data = np.ascontiguousarray(data.transpose((1, 2, 0)) )
    data = np.clip((data + 1) * 127.5, 0, 255).astype(np.uint8)
    img = Image.fromarray(data)
    img.save(name)

def select_model(template):
    modelName = glob(template)
    modelNoArr = [int(x.split('.')[0].split('_')[-1]) for x in modelName]
    modelNo = max(modelNoArr)
    modelName = template.split('*')[0] + str(modelNo) + '.pth'
    return modelName

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot1', required=True, help='path to dataset1')
parser.add_argument('--dataroot2', required=True, help='path to dataset2')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--experiment', default=None, help='where to store the test result')
parser.add_argument('--advWeight', type=float, default=1, help='the weight for adversarial loss')
parser.add_argument('--cycleWeight', type=float, default=10, help='the weight for cycle consistency loss')
parser.add_argument('--lsGAN', action='store_true', help='whether use least square gan or not')
parser.add_argument('--gpuId', type=int, default=0, help='the gpu id')
parser.add_argument('--nepoch', type=int, default=200, help='the number of epoch for training')
parser.add_argument('--bufferSize', type=int, default=50, help='the number of images in the buffer')
parser.add_argument('--learnRate', type=float, default=0.0002, help='the learning rate of the network')
parser.add_argument('--isInstanceNorm', action='store_true', help='use instance normalization instead of batch normalization')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'checkpoints'
    if opt.lsGAN:
        opt.experiment += '_lsgan'
    else:
        opt.experiment += '_dcgan'

opt.experiment += ('_' + opt.dataroot1.split('/')[0] )
opt.experimentAtoB = os.path.join(opt.experiment, 'testAtoB')
opt.experimentBtoA = os.path.join(opt.experiment, 'testBtoA')
os.system('mkdir -p  {0}'.format(opt.experimentAtoB))
os.system('mkdir -p  {0}'.format(opt.experimentBtoA))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dset = dataLoader.BatchLoader(
        dataroot1=opt.dataroot1,
        dataroot2=opt.dataroot2,
        batch_size = opt.batchSize,
        im_size = [opt.imageSize, opt.imageSize],
        isRandom = False,
        phase='TEST'
        )

imgsDomA = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
imgsDomB = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
netG_AtoB = models.Generator(opt.isInstanceNorm)
netG_BtoA = models.Generator(opt.isInstanceNorm)
modelName_AtoB = select_model('{0}/netG_AtoB_iter_*.pth'.format(opt.experiment) )
modelName_BtoA = select_model('{0}/netG_BtoA_iter_*.pth'.format(opt.experiment) )
netG_AtoB.load_state_dict(torch.load(modelName_AtoB) )
netG_BtoA.load_state_dict(torch.load(modelName_BtoA) )
print(modelName_AtoB)
print(modelName_BtoA)


if opt.cuda:
    netG_AtoB = netG_AtoB.cuda(opt.gpuId)
    netG_BtoA = netG_BtoA.cuda(opt.gpuId)
    imgsDomA = imgsDomA.cuda(opt.gpuId)
    imgsDomB = imgsDomB.cuda(opt.gpuId)

imgsDomA = Variable(imgsDomA)
imgsDomB = Variable(imgsDomB)

j = 0
while True:
    # load a batch of data
    imgsDomA_cpu, imgsDomB_cpu, isEpoch = dset.LoadBatch()

    # handle incomplete batch at end of epoc
    batchSize = np.shape(imgsDomA_cpu)[0]
    imgsDomA.data.resize_(imgsDomA_cpu.shape)
    imgsDomB.data.resize_(imgsDomB_cpu.shape)

    # load the new batch onto the gpu
    imgsDomA.data.copy_(torch.from_numpy(imgsDomA_cpu))
    imgsDomB.data.copy_(torch.from_numpy(imgsDomB_cpu))

    fakeB = netG_AtoB(imgsDomA)
    fakeA = netG_BtoA(imgsDomB)

    save_image(imgsDomA.data, '{0}/{1}_A.png'.format(opt.experimentAtoB, j) )
    save_image(fakeB.data, '{0}/{1}_AtoB.png'.format(opt.experimentAtoB, j) )
    save_image(imgsDomB.data, '{0}/{1}_B.png'.format(opt.experimentBtoA, j) )
    save_image(fakeA.data, '{0}/{1}_BtoA.png'.format(opt.experimentBtoA, j) )

    j += 1

    if j % 50 == 0:
        print('Finish %d' % j)

    if isEpoch == True:
        break
