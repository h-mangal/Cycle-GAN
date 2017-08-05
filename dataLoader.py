import numpy as np
from random import shuffle
from PIL import Image
import os


class BatchLoader(object):
    def __init__(self, dataroot1, dataroot2, batch_size, im_size,
            resize_crop=True, isRandom=True, phase='TRAIN' ):
        self.dataroot1   = dataroot1
        self.dataroot2   = dataroot2
        self.batch_size  = batch_size
        self.im_size     = im_size
        self.resize_crop = resize_crop
        self.phase = phase.upper()
        self.isRandom = isRandom

        self.filelist1 = []
        for f in os.listdir(dataroot1):
            if f.startswith('.'):
                continue
            self.filelist1.append(f)

        self.filelist2 = []
        for f in os.listdir(dataroot2):
            if f.startswith('.'):
                continue
            self.filelist2.append(f)

        self.cur1, self.cur2 = 0, 0
        if isRandom and phase != 'TEST':
            shuffle(self.filelist1)
            shuffle(self.filelist2)

    def LoadBatch(self):
        imgs1 = np.zeros([self.batch_size,3]+self.im_size)
        imgs2 = np.zeros([self.batch_size,3]+self.im_size)
        isNewEpoch = False
        for i in range(self.batch_size):
            if self.cur1 == len(self.filelist1):
                self.cur1 = 0
                if self.isRandom:
                    shuffle(self.filelist1)
                isNewEpoch = True
                if self.phase == 'TEST':
                    break
            if self.cur2 == len(self.filelist2):
                self.cur2 = 0
                if self.isRandom:
                    shuffle(self.filelist2)
                if self.phase == 'TEST':
                    isNewEpoch = True
                    break

            imgs1[i,...] = self.LoadImg(self.dataroot1+self.filelist1[self.cur1])
            imgs2[i,...] = self.LoadImg(self.dataroot2+self.filelist2[self.cur2])
            self.cur1 += 1
            self.cur2 += 1
        return imgs1, imgs2, isNewEpoch


    def LoadImg(self,fname):
        im = Image.open(fname)
        if self.resize_crop:
            im = self.Resize_Crop(im)
        im = np.asarray(im,dtype='float32')/127.5-1
        if im.ndim == 2:
            im = np.concatenate([im[:,:,None]]*3,2)
        return im.transpose(2,0,1)

    def Resize_Crop(self,im):
        w0,h0 = im.size
        h1,w1 = self.im_size
        sh = float(h1)/float(h0)
        sw = float(w1)/float(w0)
        if sh<sw:
            im = im.resize((int(w1),int(np.ceil(h0*sw))))
        else:
            im = im.resize((int(np.ceil(w0*sh)),int(h1)))
        im = im.crop(( int(im.size[0]/2-w1/2), int(im.size[1]/2-h1/2),
            int(im.size[0]/2+w1/2+w1%2), int(im.size[1]/2+h1/2+h1%2) ))
        return im



