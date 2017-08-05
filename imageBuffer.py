import random
import torch
from torch.autograd import Variable


class imageBuffer(object):
    def __init__(self, bufferSize=50):
        self.Buffer = []
        self.bufferSize = bufferSize

    def query(self, inImages):
        batchSize, _, _, _ = inImages.size()
        returnImages = []
        for n in range(batchSize):
            image = inImages.data[n, :].unsqueeze(0)
            if len(self.Buffer) < self.bufferSize:
                self.Buffer.append(image)
                returnImages.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    returnImages.append(image)
                else:
                    rId = random.randint(0, self.bufferSize-1)
                    tmp = self.Buffer[rId].clone()
                    self.Buffer[rId] = image
                    returnImages.append(tmp)
        outImages = Variable(torch.cat(returnImages, 0) )
        return outImages
