import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import argparse
from copy import deepcopy
#from progress.bar import IncrementalBar

#from dataset import Cityscapes, Facades, Maps
#from dataset import transforms as T
from gan.generator2 import UnetGenerator
from gan.discriminator2 import ConditionalDiscriminator
from gan.criterion import GeneratorLoss, DiscriminatorLoss
from gan.utils import Logger, initialize_weights

#parser = argparse.ArgumentParser(prog = 'top', description='Train Pix2Pix')
#parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
#parser.add_argument("--dataset", type=str, default="facades", help="Name of the dataset: ['facades', 'maps', 'cityscapes']")
#parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
#parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
#args = parser.parse_args()



import torch
from torch.utils.data import Dataset 
import cv2
import numpy as np 
import pandas as pd 
import pickle
from random import randrange

class ImagePathToTensor(object):

    def __init__(self):
        pass

    def __call__(self, path):

        imA = cv2.imread(path)

        imB = deepcopy(imA)

        roi = np.reshape(imB,(-1,roi.shape[-1]))

        np.random.seed(choice)

        np.take(roi,np.random.rand(roi.shape[0]).argsort(),axis=0,out=roi) # np.random.rand(X.shape[0]).argsort())

        imB = np.reshape(roi,imB.shape)
        
        # Reshape the vector back into an image of the original size
        imB = cv2.GaussianBlur(imB,(3,3),cv2.BORDER_DEFAULT)

        imA = imA/255
        imB = imB/255

        imA = (imA-0.5)/0.5
        imB = (imB-0.5)/0.5

        imA = torch.from_numpy(imA)
        imB = torch.from_numpy(imB)

        imA = imA.permute(-1,0,1)
        imB = imB.permute(-1,0,1)

        imA = imA.float() 
        imB = imB.float()

        return imA, imB
        
class ImageDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = ImagePathToTensor() 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        imA = self.dataframe.iloc[index].path

        if self.transform: imA, imB = self.transform(imA)

        return imA, imB

lr = 0.0002
epochs = 200
batch_size = 1 

device = ('cuda')

# transforms = T.Compose([T.Resize((256,256)),
                        # T.ToTensor(),
                        # T.Normalize(mean=[0.5, 0.5, 0.5],
                        #              std=[0.5, 0.5, 0.5])])
# models
print('Defining models!')
generator = UnetGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)

generator.apply(initialize_weights)
discriminator.apply(initialize_weights)

# optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
# loss functions
g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()
# dataset
# print(f'Downloading "{args.dataset.upper()}" dataset!')
# if args.dataset=='cityscapes':
#     dataset = Cityscapes(root='.', transform=transforms, download=True, mode='train')
# elif args.dataset=='maps':
#     dataset = Maps(root='.', transform=transforms, download=True, mode='train')
# else:
#     dataset = Facades(root='.', transform=transforms, download=True, mode='train')

dataset = dataload.ImageDataset("train_gan.csv") #,transform=torchvision.transforms.Compose([dataload.ImagePathToTensor()]))
# train_loader = DataLoader(train_dataset, batch_size = batch, shuffle = True, drop_last=True)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print('Start of training process!')

save_file1 = 'Run_gan/disc.pth'
save_file2 = 'Run_gan/gen.pth'

loss_file1 = 'Run_gan/disc.txt'
loss_file2 = 'Run_gan/gan.txt'


#logger = Logger(filename=args.dataset)
for epoch in range(epochs):
    ge_loss=0.
    de_loss=0.
    start = time.time()
    #bar = IncrementalBar(f'[Epoch {epoch+1}/{args.epochs}]', max=len(dataloader))
    for real, x in dataloader: #real is imA, x is imB. Goal is to go from imB to imA
        x = x.to(device)
        real = real.to(device)

        # Generator`s loss
        fake = generator(x)
        fake_pred = discriminator(fake, x)
        g_loss = g_criterion(fake, real, fake_pred)

        # Discriminator`s loss
        fake = generator(x).detach()
        fake_pred = discriminator(fake, x)
        real_pred = discriminator(real, x)
        d_loss = d_criterion(fake_pred, real_pred)

        # Generator`s params update
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Discriminator`s params update
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        # add batch losses
        ge_loss += g_loss.item()
        de_loss += d_loss.item()
        bar.next()
    #bar.finish()  
    # obttain per epoch losses
    g_loss = ge_loss/len(dataloader)
    d_loss = de_loss/len(dataloader)
    # count timeframe
    end = time.time()
    tm = (end - start)

    if epoch%10==0:
        sf2 = save_file2.split('.')[0]
        sf2 = sf2+'_'+str(epoch)+'.pth'
        torch.save(generator.state_dict(), sf2)

        sf2 = save_file1.split('.')[0]
        sf2 = sf2+'_'+str(epoch)+'.pth'
        torch.save(discriminator.state_dict(), sf2)
        #print(counter,'/',len(train_loader),' ',epoch,datetime.datetime.now())



    #logger.add_scalar('generator_loss', g_loss, epoch+1)
    #logger.add_scalar('discriminator_loss', d_loss, epoch+1)
    #logger.save_weights(generator.state_dict(), 'generator')
    #logger.save_weights(discriminator.state_dict(), 'discriminator')
    print("[Epoch %d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (epoch+1, g_loss, d_loss, tm))
#logger.close()
print('End of training process!')