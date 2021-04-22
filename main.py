import sys
import numpy as np
from model import *
import pandas as pd
from loss import My_loss
from torch import nn
import torch.optim as optim
import torch.utils.data
from loss import FeatureExtractor
from collections import Counter

from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torch.autograd
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os


epochs =100
print("load LR train set..........")
LR_train_data = pd.read_csv("LR.csv",header=None)
LR_train_data = np.array(LR_train_data[0]).reshape((-1,1,50,50))
LR_train_data = LR_train_data/255.
LR_train_data = torch.tensor(LR_train_data).type(torch.FloatTensor)
print("load HR train set..........")
HR_train_data = pd.read_csv("HR.csv",header=None)
HR_train_data = np.array(HR_train_data[0]).reshape((-1,1,200,200))
HR_train_data = HR_train_data/255.
HR_train_data = torch.tensor(HR_train_data).type(torch.FloatTensor)
print("load segmentation train set..........")
S_train_data = pd.read_csv("S.csv",header=None)
S_train_data = np.array(S_train_data[0]).reshape((-1,1,200,200)).astype(np.int)
S_train_data = torch.tensor(S_train_data).type(torch.FloatTensor)


HR_train_loader = torch.utils.data.DataLoader(HR_train_data,batch_size=20)
LR_train_loader = torch.utils.data.DataLoader(LR_train_data,batch_size=20)
S_train_loader = torch.utils.data.DataLoader(S_train_data,batch_size=20)
device = torch.device('cuda:0')
netG = Generator().to(device)
netD = Discriminator().to(device)
mseloss = nn.MSELoss().to(device)
myloss = My_loss().to(device)
biloss = nn.BCELoss().to(device)
featur_extractor = FeatureExtractor().to(device)
featur_extractor.eval()
criterion_conten = nn.L1Loss()
optimizerG = optim.Adam(netG.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD = optim.Adam(netD.parameters(),lr=0.0001,betas=(0.5,0.999))
Loss_list_D = []
Loss_list_G = []
Accuracy_list_D = []
Accuracy_list_DS = []
Accuracy_list_G = []
pore = []
for epoch in range(epochs):
    i = 0
    p_mask = 0
    train_loss_D = 0
    train_acc_D_batch = 0
    train_acc_DS_batch = 0
    train_acc_D = 0
    num_correct_D = 0
    train_loss_G = 0
    train_acc_G = 0
    num_correct_G = 0
    num_correct_DS = 0
    netG.train()
    netD.train()
    for data_HR, data_LR, data_S in zip(HR_train_loader, LR_train_loader, S_train_loader):
        i+=1

        data_G, target_G = data_LR.to(device), data_HR.cuda()
        data_D, target_D= data_HR.to(device),data_S.cuda()


        #training generator
        optimizerD.zero_grad()
        predict_S = netD(data_D)

        d_loss = biloss(predict_S, target_D)
        fake_img = netG(data_G).detach()
        d_loss_fake = biloss(netD(fake_img), target_D)

        loss_D = 0.5 * d_loss + 0.5 * d_loss_fake
        loss_D.backward()
        optimizerD.step()

        optimizerG.zero_grad()
        gen_hr = netG(data_G)

        gen_feature = featur_extractor(gen_hr)
        real_feature = featur_extractor(target_G)
        loss_content = criterion_conten(gen_feature,real_feature.detach())
        loss_GH= mseloss(gen_hr,target_G)

        d_loss_lr,p= myloss(netD(gen_hr), target_D)
        d_loss_lr = d_loss_lr.data.cuda()
        loss_G = loss_GH+loss_content+d_loss_lr

        loss_G.backward()
        optimizerG.step()
        #training discriminator
        p_mask = p+p_mask

        train_loss_D += float(loss_D.item())
        train_loss_G += float(loss_G.item())
        correct_D = torch.eq(torch.where(netD(gen_hr)>0.5,torch.ones_like(netD(gen_hr)),torch.zeros_like(netD(gen_hr))),target_D).sum().float().item()
        correct_DS = torch.eq(torch.where(predict_S>0.5,torch.ones_like(predict_S),torch.zeros_like(predict_S)),target_D).sum().float().item()
        correct_G = torch.eq(gen_hr,target_G).sum().float().item()
        num_correct_D += correct_D
        num_correct_DS += correct_DS
        num_correct_G += correct_G
        sys.stdout.write("[EPoch %d/%d] [Batch:%d/%d] [D loss: %f] [G loss:%f] [D ACC:%f] [DS ACC:%f] [G ACC:%f] [pore: %f]\n" %(epoch,epochs,i,len(HR_train_loader),loss_D.item(),loss_G.item(),correct_D/len(target_G.flatten()),correct_DS/len(target_G.flatten()),correct_G/len(target_G.flatten()),p_mask/i))
        train_acc_D_batch += correct_D/len(target_G.flatten())
        train_acc_DS_batch += correct_DS/len(target_G.flatten())
    if epoch % 5== 0 and epoch != 0:
        save_image(data_G.cpu().data, 'photo/real_images_{}.png'.format(epoch))
        save_image(fake_img.cpu().data, 'photo/fake_images_{}.png'.format(epoch))
        save_image(fake_img[0].cpu().data, 'photo/fake_images_single_{}.png'.format(epoch))
        save_image(data_G[0].cpu().data, 'photo/train_images_single_{}.png'.format(epoch))
        save_image(data_D[0].cpu().data, 'photo/real_images_single_{}.png'.format(epoch))
        save_image(data_S.cpu().data, 'photo/fake_images_toimg_{}.png'.format(epoch))
        save_image(data_D.cpu().data, 'photo/label_images_{}.png'.format(epoch))

    Loss_list_D.append(train_loss_D/len(HR_train_loader))
    Loss_list_G.append(train_loss_G/len(HR_train_loader))
    Accuracy_list_D.append(train_acc_D_batch/len(HR_train_loader))
    Accuracy_list_DS.append(train_acc_DS_batch/len(HR_train_loader))
    Accuracy_list_G.append(num_correct_G/len(HR_train_loader))

    pore.append(p_mask/len(HR_train_loader))




    if epoch % 50==0 and epoch != 0:
        np.savetxt("doc/pore_{}.csv".format(epoch), np.array(pore))
        torch.save(netD.state_dict(),"model/netD_%d.pth"%epoch)
        torch.save(netG.state_dict(),"model/netE_%d.pth"%epoch)
        np.savetxt("doc/Loss_D_{}.csv".format(epoch),np.array(Loss_list_D))
        np.savetxt("doc/Loss_G_{}.csv".format(epoch),np.array(Loss_list_G))
        np.savetxt("doc/Accuracy_D_{}.csv".format(epoch),np.array(Accuracy_list_D))
        np.savetxt("doc/Accuracy_DS_{}.csv".format(epoch),np.array(Accuracy_list_DS))
        np.savetxt("doc/Accuracy_G_{}.csv".format(epoch),np.array(Accuracy_list_G))

torch.save(netD.state_dict(),"model/netD.pth")
torch.save(netG.state_dict(),"model/netE.pth")
np.savetxt("doc/pore.csv", np.array(pore))
np.savetxt("doc/Loss_D.csv", np.array(Loss_list_D))
np.savetxt("doc/Loss_G_.csv", np.array(Loss_list_G))
np.savetxt("doc/Accuracy_D.csv", np.array(Accuracy_list_D))
np.savetxt("doc/Accuracy_DS.csv", np.array(Accuracy_list_DS))
np.savetxt("doc/Accuracy_G.csv", np.array(Accuracy_list_G))
