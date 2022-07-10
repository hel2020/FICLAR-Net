import sys
import os
import warnings
from coder import *
from gan_models import *
from utils import *
import torch
import torch.nn as nn
torch.set_printoptions(profile='full')
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
np.set_printoptions(threshold=np.inf)
import argparse
import json
import cv2
import dataset
import time
from sklearn import *


parser = argparse.ArgumentParser(description='PyTorch FICLAR-Net')

parser.add_argument('-train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('-test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default='FICLAR-Net_checkpoint.pth.tar', type=str,
                    help='path to the pretrained model')

parser.add_argument('--gpu', metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('--task', metavar='TASK', type=str,
                    help='task id to use.')


def main():
    global args, best_prec1

    args = parser.parse_args()
    args.original_lr = 1e-5
    args.lr = 1e-5
    args.batch_size = 1
    args.momentum = 0.95
    args.original_decay=1e-7
    args.decay =  1*1e-7
    args.start_epoch = 0
    args.epochs = 1000
    args.scales = [1, 1, 1, 1]
    args.workers = 0
    args.seed = time.time()
    args.print_freq = 50
    #args.train_json = './Munich_train.json'
    #args.test_json = './Munich_test.json'
    args.train_json = './AOD_train.json'
    args.test_json = './AOD_test_cai.json'
    best_F1 = 0
    best_P = 0
    best_R = 0
    best_ACC = 0

    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)

    args.gpu = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    en=Encoder()
    de=Decoder()
    single_conv=Single_Conv()
    gen= Generator()
    dis= Discriminator()
    

    en= en.cuda()
    de= de.cuda()
    single_conv=single_conv.cuda()
    gen= gen.cuda()
    dis= dis.cuda()

    en_optimizer = torch.optim.Adam(en.parameters(), lr=1e-4)
    de_optimizer = torch.optim.Adam(de.parameters(), lr=1e-4)
    single_conv_optimizer = torch.optim.Adam(single_conv.parameters(), lr=1e-4)
    dis_optimizer = torch.optim.Adam(dis.parameters(), lr=1e-6)
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4)


    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_F1= checkpoint['best_F1']
            best_P= checkpoint['best_P']
            best_R= checkpoint['best_R']
            best_ACC= checkpoint['best_ACC']
            en.load_state_dict(checkpoint['en_state_dict'], False)
            de.load_state_dict(checkpoint['de_state_dict'], False)
            single_conv.load_state_dict(checkpoint['single_conv_state_dict'], False)
            gen.load_state_dict(checkpoint['gen_state_dict'], False)
            dis.load_state_dict(checkpoint['dis_state_dict'],False)
            en_optimizer.load_state_dict(checkpoint['en_optimizer'])
            de_optimizer.load_state_dict(checkpoint['de_optimizer'])
            single_conv_optimizer.load_state_dict(checkpoint['single_conv_optimizer'])
            gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.epochs):

        train(train_list, en,de,single_conv,gen,dis,en_optimizer,de_optimizer,single_conv_optimizer,gen_optimizer,dis_optimizer, epoch)
        F1,P,R,ACC = validate(val_list, en,de,single_conv,gen,dis)
        args.task = "FICLAR-Net_"
        is_best = F1 < best_F1
        best_F1 = max(F1, best_F1)
        best_P = max(P, best_P)
        best_R = max(R, best_R)
        best_ACC = max(ACC, best_ACC)

        print(' best_F1 {best_F1:.4f} '.format(best_F1=best_F1), ' best_P {best_p:.4f} '.format(best_p=best_P),
              ' best_R {best_r:.4f} '.format(best_r=best_R), ' best_ACC {best_acc:.4f} '.format(best_acc=best_ACC)
              )
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'en_state_dict': en.state_dict(),
            'de_state_dict': de.state_dict(),
            'single_conv_state_dict': single_conv.state_dict(),
            'gen_state_dict': gen.state_dict(),
            'dis_state_dict': dis.state_dict(),
            'best_F1': best_F1,
            'best_P': best_P,
            'best_R': best_R,
            'best_ACC': best_ACC,
            'en_optimizer': en_optimizer.state_dict(),
            'de_optimizer': de_optimizer.state_dict(),
            'single_conv_optimizer': single_conv_optimizer.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
        },is_best, args.task)

def dice_loss(target, predictive, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss



def train(train_list, en,de,single_conv,gen,dis, en_optimizer,de_optimizer,single_conv_optimizer,gen_optimizer,dis_optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args.batch_size,
                            num_workers=args.workers),
                            
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f, wd %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr,args.decay))

    en.train()
    de.train()
    gen.train()
    dis.train()

    end = time.time()
    L1_loss = nn.SmoothL1Loss()# nn.L1Loss()
    loss =nn.MSELoss()# nn.BCELoss()
    smooth_loss = nn.SmoothL1Loss()

    for i, (img, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        labels = labels.type(torch.FloatTensor).unsqueeze(0).cuda()
        labels = Variable(labels/255.)
######################################## single_conv is the Connection Module
        mid, c1, c2, c3, c4 = en(img)
        G_input = c1
        fake_mid = gen(G_input, 'Without_Noise')
        # mix_feature = (1 - alpha) * mid + alpha * fake_mid.detach()
        mix_feature = single_conv(mid,fake_mid)
        fake_img_detect = de(mid, mix_feature, c1, c2, c3, c4)
        #print(fake_img_detect, labels)
        model_loss = loss(fake_img_detect, labels)+ 0.5 * dice_loss(fake_img_detect, labels)
        en_optimizer.zero_grad()
        de_optimizer.zero_grad()
        single_conv_optimizer.zero_grad()
        model_loss.backward()
        en_optimizer.step()
        de_optimizer.step()
        single_conv_optimizer.step()
##########################################
        if epoch>500:
           mid, c1, c2, c3, c4 = en(img)
           z = torch.randn(c1.shape[0],c1.shape[1], c1.shape[2], c1.shape[3]).cuda()
           G_input = c1+z#torch.cat((c1, z), dim=1)
           fake_mid = gen(G_input)
           # mix_feature = (1 - alpha) * mid + alpha * fake_mid.detach()
           mix_feature = single_conv(mid,fake_mid)
           fake_img_detect = de(mid, mix_feature, c1, c2, c3, c4)
           # print('mix_train_model')
           mix_model_loss = loss(fake_img_detect, labels) #+ 0.01 * dice_loss(fake_img_detect, labels)
           en_optimizer.zero_grad()
           de_optimizer.zero_grad()
           single_conv_optimizer.zero_grad()
           mix_model_loss.backward()
           en_optimizer.step()
           de_optimizer.step()
           single_conv_optimizer.step()
#######################################
           mid, c1, c2, c3, c4 = en(img)
           z = torch.randn(c1.shape[0],c1.shape[1], c1.shape[2], c1.shape[3]).cuda()
           G_input = c1+z#torch.cat((c1, z), dim=1)
           fake_mid = gen(G_input)
           d_real = dis(mid.detach())
           # print('train_Discriminator_model')
           d_real_loss = L1_loss(d_real, torch.ones_like(d_real)*0.9)
           # mix_feature = (1 - alpha) * mid + alpha * fake_mid
           mix_feature = single_conv(mid,fake_mid)
           d_fake = dis(mix_feature.detach())
           d_fake_loss = L1_loss(d_fake, torch.zeros_like(d_fake))
           d_loss = (d_real_loss + d_fake_loss) * 0.5
           dis_optimizer.zero_grad()
           d_loss.backward()
           dis_optimizer.step()
############################################
           mid, c1, c2, c3, c4 = en(img)
           z = torch.randn(c1.shape[0],c1.shape[1], c1.shape[2], c1.shape[3]).cuda()
           G_input = c1+z#torch.cat((c1, z), dim=1)
           fake_mid = gen(G_input)
           # mix_feature = (1 - alpha) * mid+ alpha * fake_mid
           mix_feature = single_conv(mid,fake_mid)
           d_img_detect = de(mid, mix_feature, c1, c2, c3, c4)
           d_fake = dis(mix_feature)
           # print('train_Generator_model')
           g_loss = L1_loss(d_fake, torch.ones_like(d_fake))
           # g_loss -= criterion(d_img_detect, labels)
           gen_optimizer.zero_grad()
           g_loss.backward()
           gen_optimizer.step()

        if i % args.print_freq == 0 and epoch>500:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'model_Loss {model_loss:.4f} ({model_loss:.4f})\t'
                  'mix_Loss {mix_loss:.4f} ({mix_loss:.4f})\t'
                  'gen_Loss {gen_loss:.4f} ({gen_loss:.4f})\t'
                  'dis_Loss {dis_loss:.4f} ({dis_loss:.4f})\t'
                .format(
                epoch, i, len(train_loader), model_loss=model_loss, mix_loss=mix_model_loss,
                gen_loss=g_loss,dis_loss=d_loss))
        if i % args.print_freq == 0 and epoch<=500:
           print('Epoch: [{0}][{1}/{2}]\t'
                  'model_Loss {model_loss:.4f} ({model_loss:.4f})\t'
                  .format(
                epoch, i, len(train_loader), model_loss=model_loss))


def validate(val_list, en,de,single_conv,gen,dis):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        batch_size=args.batch_size)

    en.eval()
    de.eval()
    gen.eval()
    dis.eval()

    bestF1 = 0
    bestACC = 0
    bestp = 0
    bestr = 0
    F1_list = []
    ACC_list = []
    P_list = []
    R_list = []

    with torch.no_grad():

      for i, (img, labels) in enumerate(test_loader):
      
          img = img.cuda()
          img = Variable(img)
          labels = labels.type(torch.FloatTensor).unsqueeze(0).cuda()
          labels = Variable(labels/255.)
          mid, c1, c2, c3, c4 = en(img)
          G_input = c1
          fake_mid = gen(G_input, 'Without_Noise')
          mix_feature = single_conv(mid,fake_mid)
          y = de(mid, mix_feature, c1, c2, c3, c4)
          y_true = torch.squeeze(labels).cpu().numpy()
          y_pred = torch.squeeze(y).cpu().numpy()
          y_pred = np.where(y_pred > 0.5, 1, 0)
          
          #f1, p, r, acc = score(y_pred, y_true)
          #y_pred = np.reshape(y_pred, newshape=(-1))
          #y_true = np.reshape(y_true, newshape=(-1))
          #F1 = metrics.f1_score(y_true, y_pred, pos_label=1, average="macro", sample_weight=None)
          img_array = y_pred.reshape(y_pred.shape[0],y_pred.shape[1])
          img_array = img_array.astype(np.uint8)
          y_true = y_true.astype(np.uint8)
          retval_y_pred, labels_y_pred, stats_y_pred, centroids_y_pred = cv2.connectedComponentsWithStats(img_array, connectivity=8)
          retval_y_true, labels_y_true, stats_y_true, centroids_y_true = cv2.connectedComponentsWithStats(y_true, connectivity=8)
          TP=0
          FN=0
          FP=0
          for y_t in centroids_y_true:    #计算TP,FN
              a=0
              for i,y_p in enumerate(centroids_y_pred):
                 if stats_y_pred[i][4]>=100:
                   d= ( (y_t[0]-y_p[0])**2 + (y_t[1]-y_p[1])**2 )**0.5
                   if d<=6:
                      TP+=1
                   else:
                      a+=1
              if a==len(centroids_y_pred):
                 FN+=1
          for j,y_p in enumerate(centroids_y_pred):   #计算FP
              b=0
              for y_t in centroids_y_true:
                 if stats_y_pred[j][4]>=100:
                   d= ( (y_t[0]-y_p[0])**2 + (y_t[1]-y_p[1])**2 )**0.5
                   if d>6:
                      b+=1
              if b==len(centroids_y_true):
                 FP+=1
          F1=(2*TP)/(2*TP++FP+FN)
          F1_list.append(F1)
          #best_F1=sum(F1_list) / len(F1_list)
          '''
          ACC_list.append(acc)
          #best_ACC = sum(ACC_list) / len(ACC_list)
          P_list.append(p)
          #best_p = sum(P_list) / len(P_list)
          R_list.append(r)
          #best_r = sum(R_list) / len(R_list)
          '''
      best_F1=sum(F1_list) / len(F1_list)
      best_ACC = 0#sum(ACC_list) / len(ACC_list)
      best_p = 0#sum(P_list) / len(P_list)
      best_r = 0#sum(R_list) / len(R_list)
      
      print(' F1 {F1:.4f} '.format(F1=best_F1),' P {p:.4f} '.format(p=best_p),
            ' R {r:.4f} '.format(r=best_r),' ACC {acc:.4f} '.format(acc=best_ACC)
            )

    return bestF1, bestp, bestr, bestACC


def adjust_learning_rate(optimizer, epoch):

    args.lr = args.original_lr
    args.decay =  args.original_decay
    n=epoch//500
    if n>=0 and n<4:
      for i in range(n):
         args.lr=args.lr*0.1
         args.decay =  args.decay*10
    else:
       n=3
       for i in range(n):
         args.lr=args.lr*0.1
         args.decay =  args.decay*10

    '''
    for i in range(len(args.steps)):   # args.steps = [-1, 1, 100, 150]

        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break   weight_decay
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        param_group['weight_decay'] = args.decay


def score(output, label):
    TP,TN,FP,FN = 0, 0, 0,0
    for j in range(output.shape[0]):
        for k in range(output.shape[1]):
            #if label[j][k] == 0:
                if output[j][k] == 0 and label[j][k] != 0:
                    FN += 1
                    # print("count1",count1)
                if output[j][k] == 0 and label[j][k] == 0:
                    TN += 1
                # iou = (count1 + count2)/(det.shape[0] * det.shape[1])
            #if output[j][k] != 0:
                # TP
                if output[j][k] != 0 and label[j][k] != 0:
                    TP += 1
                    # print("count0=",count0)
                if output[j][k] != 0 and label[j][k] == 0:
                    FP += 1
                    # print("count2=",count2)


    p=((1e-8+TP)/(1e-8+TP+FP))
    r=((1e-8+TP)/(1e-8+TP+FN))
    f1 =2*p*r/(p+r)
    acc=((1e-8+TP+TN)/(1e-8+TP+FP+TN+FN))
    tp=TP
    fp=FP
    return f1,p,r,acc

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m,nn.Conv2d):
           #torch.nn.init.uniform_(m.weight.data, a=0, b=1)
           #torch.nn.init.xavier_normal_(m.weight.data)
           torch.nn.init.kaiming_uniform(m.weight.data, a=0, mode='fan_in')
           if m.bias is not None:
              m.bias.data.fill_(0.)
        elif isinstance(m,nn.BatchNorm2d):
              m.weight.data.fill_(1.0)
              m.bias.data.fill_(0.)
        elif isinstance(m,nn.Linear):
             torch.nn.init.normal_(m.weight.data,0,0.01)
             # m.weight.data.normal_(0,0.01)
             m.bias.data.fill_(0.)

class AverageMeter(object):
    """Computes and stores the average and current value"""

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


if __name__ == '__main__':
    main()        