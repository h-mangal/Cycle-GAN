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


def save_image(data, name):
    data = data.cpu().squeeze(0).numpy()
    data = np.ascontiguousarray(data.transpose((1, 2, 0)) )
    data = np.clip((data + 1) * 127.5, 0, 255).astype(np.uint8)
    img = Image.fromarray(data)
    img.save(name)


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot1', required=True, help='path to dataset1')
parser.add_argument('--dataroot2', required=True, help='path to dataset2')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--advWeight', type=float, default=1, help='the weight for adversarial loss')
parser.add_argument('--cycleWeight', type=float, default=10, help='the weight for cycle consistency loss')
parser.add_argument('--identityWeight', type=float, default=0, help='the weight of the identity loss')
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
os.system('mkdir -p  {0}'.format(opt.experiment))

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
        )

imgsDomA = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
imgsDomB = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
reallabel = torch.FloatTensor(opt.batchSize)
fakelabel = torch.FloatTensor(opt.batchSize)
labelBtoA = torch.FloatTensor(opt.batchSize)
labelAtoB = torch.FloatTensor(opt.batchSize)


netG_AtoB = models.Generator(opt.isInstanceNorm)
netD_B = models.Discriminator70()
netG_BtoA = models.Generator(opt.isInstanceNorm)
netD_A = models.Discriminator70()

netD_B = netD_B.apply(models.weights_init)
netD_A = netD_A.apply(models.weights_init)

optim_G_AtoB = optim.Adam(netG_AtoB.parameters(), lr=opt.learnRate)
optim_D_B = optim.Adam(netD_B.parameters(), lr=opt.learnRate)
optim_G_BtoA = optim.Adam(netG_BtoA.parameters(), lr=opt.learnRate)
optim_D_A = optim.Adam(netD_A.parameters(), lr=opt.learnRate)

if opt.cuda:
    netG_AtoB = netG_AtoB.cuda(opt.gpuId)
    netD_B = netD_B.cuda(opt.gpuId)
    netG_BtoA = netG_BtoA.cuda(opt.gpuId)
    netD_A = netD_A.cuda(opt.gpuId)
    imgsDomA = imgsDomA.cuda(opt.gpuId)
    imgsDomB = imgsDomB.cuda(opt.gpuId)
    reallabel = reallabel.cuda(opt.gpuId)
    fakelabel = fakelabel.cuda(opt.gpuId)
    labelAtoB = labelAtoB.cuda(opt.gpuId)
    labelBtoA = labelBtoA.cuda(opt.gpuId)

imgsDomA = Variable(imgsDomA)
imgsDomB = Variable(imgsDomB)
reallabel = Variable(reallabel)
fakelabel = Variable(fakelabel)
labelAtoB = Variable(labelAtoB)
labelBtoA = Variable(labelBtoA)

real, fake = 1, 0
if opt.lsGAN:
    criterionGAN = nn.MSELoss()
else:
    criterionGAN = nn.BCELoss()
criterionCycle = nn.L1Loss()
criterionIdentity = nn.L1Loss()

lossCycle_A_log = []
lossD_A_log = []
lossG_BtoA_log = []
lossCycle_B_log = []
lossD_B_log = []
lossG_AtoB_log = []

realA_buffer = imageBuffer(opt.bufferSize)
fakeA_buffer = imageBuffer(opt.bufferSize)
realB_buffer = imageBuffer(opt.bufferSize)
fakeB_buffer = imageBuffer(opt.bufferSize)

j = 0
for epoch in range(0, opt.nepoch):
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

        ######### Train the Discriminator A  #########
        ##############################################
        realA_batch = realA_buffer.query(imgsDomA)
        realA_pred  = netD_A(realA_batch)
        reallabel.data.resize_(realA_pred.size())
        reallabel.data.fill_(real)
        lossA_real = criterionGAN(realA_pred, reallabel)

        fakeA = netG_BtoA(imgsDomB ).detach()
        fakeA_batch = fakeA_buffer.query(fakeA)
        fakeA_pred = netD_A(fakeA_batch)
        fakelabel.data.resize_(fakeA_pred.size() )
        fakelabel.data.fill_(fake)
        lossA_fake = criterionGAN(fakeA_pred, fakelabel)

        lossA_D = 0.5 * (lossA_real + lossA_fake)
        lossA_D.backward(retain_variables=True)
        lossD_A_log.append(lossA_D.data[0])

        optim_D_A.step()
        optim_D_A.zero_grad()
        optim_G_BtoA.zero_grad()

        ######### Train the Discriminator B #########
        #############################################
        realB_batch = realB_buffer.query(imgsDomB)
        realB_pred = netD_B(realB_batch)
        reallabel.data.resize_(realB_pred.size() )
        reallabel.data.fill_(real)
        lossB_real =criterionGAN(realB_pred, reallabel)

        fakeB = netG_AtoB(imgsDomA).detach()
        fakeB_batch = fakeB_buffer.query(fakeB)
        fakeB_pred = netD_B(fakeB_batch)
        fakelabel.data.resize_(fakeB_pred.size() )
        fakelabel.data.fill_(fake)
        lossB_fake = criterionGAN(fakeB_pred, fakelabel)

        lossB_D = 0.5 * (lossB_real + lossB_fake)
        lossB_D.backward()
        lossD_B_log.append(lossB_D.data[0])

        optim_D_B.step()
        optim_D_B.zero_grad()
        optim_G_AtoB.zero_grad()

        ######### Train the Generator BtoA #########
        ############################################
        fakeA = netG_BtoA(imgsDomB)
        cycleB = netG_AtoB(fakeA)
        lossCycle_B = opt.cycleWeight *criterionCycle(cycleB, imgsDomB)

        fakeA_pred = netD_A(fakeA)
        labelBtoA.data.resize_(fakeA_pred.size() )
        labelBtoA.data.fill_(real)
        lossGAN_BtoA = opt.advWeight *criterionGAN(fakeA_pred, labelBtoA)
        loss_BtoA = lossCycle_B + lossGAN_BtoA

        lossCycle_B_log.append(lossCycle_B.data[0])
        lossG_BtoA_log.append(lossGAN_BtoA.data[0])

        ######### Train the Generator AtoB #########
        ############################################
        fakeB = netG_AtoB(imgsDomA)
        cycleA = netG_BtoA(fakeB)
        lossCycle_A = opt.cycleWeight * criterionCycle(cycleA, imgsDomA)

        fakeB_pred = netD_B(fakeB)
        labelAtoB.data.resize_(fakeB_pred.size() )
        labelAtoB.data.fill_(real)
        lossGAN_AtoB = opt.advWeight * criterionGAN(fakeB_pred, labelAtoB)
        loss_AtoB = lossCycle_A + lossGAN_AtoB

        lossCycle_A_log.append(lossCycle_A.data[0])
        lossG_AtoB_log.append(lossGAN_AtoB.data[0])

        lossG = loss_BtoA + loss_AtoB

        if opt.identityWeight > 0:
            lossIdentity_AtoB = opt.identityWeight * criterionIdentity(fakeB, imgsDomA)
            lossIdentity_BtoA = opt.identityWeight * criterionIdentity(fakeA, imgsDomB)
            lossG += lossIdentity_AtoB + lossIdentity_BtoA

        lossG.backward()

        optim_G_AtoB.step()
        optim_G_BtoA.step()
        optim_G_AtoB.zero_grad()
        optim_G_BtoA.zero_grad()
        optim_D_A.zero_grad()
        optim_D_B.zero_grad()
        # visualize progress

        if j % 20 == 0:
            if opt.identityWeight > 0 :
                print('[%d][%d] D_A: %.4f D_A_real: %.4f D_A_fake: %.4f Cycle_A: %.4f GAN_BtoA: %.4f Identity_BtoA %.4f' %
                    (epoch, j, lossA_D.data[0], lossA_real.data[0], lossA_fake.data[0], lossCycle_A.data[0], lossGAN_BtoA.data[0], lossIdentity_BtoA.data[0]))
                print('[%d][%d] D_B: %.4f D_B_real: %.4f D_B_fake: %.4f Cycle_B: %.4f GAN_AtoB: %.4f Identity_AtoB %.4f' %
                    (epoch, j, lossB_D.data[0], lossB_real.data[0], lossB_fake.data[0], lossCycle_B.data[0], lossGAN_AtoB.data[0], lossIdentity_AtoB.data[0]) )
            else:
                print('[%d][%d] D_A: %.4f D_A_real: %.4f D_A_fake: %.4f Cycle_A: %.4f GAN_BtoA: %.4f ' %
                    (epoch, j, lossA_D.data[0], lossA_real.data[0], lossA_fake.data[0], lossCycle_A.data[0], lossGAN_BtoA.data[0]))
                print('[%d][%d] D_B: %.4f D_B_real: %.4f D_B_fake: %.4f Cycle_B: %.4f GAN_AtoB: %.4f ' %
                    (epoch, j, lossB_D.data[0], lossB_real.data[0], lossB_fake.data[0], lossCycle_B.data[0], lossGAN_AtoB.data[0]) )
        if j%50 == 0:
            np.save('{0}/lossD_A.npy'.format(opt.experiment), np.array(lossD_A_log))
            np.save('{0}/lossCycle_A.npy'.format(opt.experiment), np.array(lossCycle_A_log))
            np.save('{0}/lossGAN_BtoA.npy'.format(opt.experiment), np.array(lossG_BtoA_log))
            np.save('{0}/lossD_B.npy'.format(opt.experiment), np.array(lossD_B_log))
            np.save('{0}/lossCycle_B.npy'.format(opt.experiment), np.array(lossCycle_B_log))
            np.save('{0}/lossGAN_AtoB.npy'.format(opt.experiment), np.array(lossG_AtoB_log))
            save_image(imgsDomA.data, '{0}/{1}_real_A.png'.format(opt.experiment, epoch) )
            save_image(fakeA.data,    '{0}/{1}_fake_A.png'.format(opt.experiment, epoch) )
            save_image(cycleA.data,   '{0}/{1}_cycle_A.png'.format(opt.experiment, epoch) )
            save_image(imgsDomB.data, '{0}/{1}_real_B.png'.format(opt.experiment, epoch) )
            save_image(fakeB.data,    '{0}/{1}_fake_B.png'.format(opt.experiment, epoch) )
            save_image(cycleB.data,   '{0}/{1}_cycle_B.png'.format(opt.experiment, epoch) )

        if epoch >= 100:
            rate = (opt.nepoch - epoch) / (opt.nepoch - 100)
            for param_group in optim_G_AtoB.param_groups:
                param_group['lr'] = opt.learnRate * rate
            for param_group in optim_G_BtoA.param_groups:
                param_group['lr'] = opt.learnRate * rate
            for param_group in optim_D_A.param_groups:
                param_group['lr'] = opt.learnRate * rate
            for param_group in optim_D_B.param_groups:
                param_group['lr'] = opt.learnRate * rate
        j = j+1
        if isEpoch == True:
            epoch += 1
            if epoch % 1 == 0:
                torch.save(netG_AtoB.state_dict(), '{0}/netG_AtoB_iter_{1}.pth'.format(opt.experiment, epoch))
                torch.save(netG_BtoA.state_dict(), '{0}/netG_BtoA_iter_{1}.pth'.format(opt.experiment, epoch))
                torch.save(netD_B.state_dict(), '{0}/netD_B_iter_{1}.pth'.format(opt.experiment, epoch))
                torch.save(netD_A.state_dict(), '{0}/netD_A_iter_{1}.pth'.format(opt.experiment, epoch))
            break
