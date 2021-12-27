import torch
from torch.nn import init
import numpy as np
import random
import math
import os
from matplotlib import pyplot as plt
from PIL import Image
import scipy.signal
from tqdm import tqdm
from torch.autograd import Variable


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class RandomErasing(object):
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, criterion, optimizer, epoch, epochs, step, train_loader, cuda):
    loss = 0
    print('Start Train')
    model = model.train()

    with tqdm(total=step, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= step:
                break
            images, targets = batch[0], batch[1]
            
            with torch.no_grad():
                if cuda:
                    images = Variable(images.cuda().detach())
                    targets = Variable(targets.cuda().detach())
                else:
                    images = Variable(images)
                    targets = Variable(targets)

            optimizer.zero_grad()
            outputs = model(images)

            loss_value = criterion(outputs, targets)
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            pbar.set_postfix(**{'loss' : loss / (iteration + 1), 'lr' : get_lr(optimizer)})
            pbar.update(1)
    
    print('Finish Train')

    return loss


def val_one_epoch(model, criterion, optimizer, epoch, epochs, step, val_loader, cuda):
    loss = 0
    model.eval()
    print('Start Validation')

    with tqdm(total=step, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= step:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(images.cuda().detach())
                    targets = Variable(targets.cuda().detach())
                else:
                    images = Variable(images)
                    targets = Variable(targets)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss_value = criterion(outputs, targets)
            
            loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': loss / (iteration + 1)})
            pbar.update(1)
    
    print('Finish Validation')
    
    return loss


def fliplr(image):
    inv_idx = torch.arange(image.size(3) - 1, -1, -1).long()
    img_flip = image.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloader):
    features = torch.FloatTensor()

    for data in dataloader:
        image, label = data
        image_f = fliplr(image)

        input_image = Variable(image).cuda()
        input_image_f = Variable(image_f).cuda()

        outputs = model(input_image) + model(input_image_f)
        # 计算每个特征的二范数
        feature_norm = torch.norm(outputs, p=2, dim=1, keepdim=True)
        feature = outputs.div(feature_norm.expand_as(outputs))

        features = torch.cat((features, feature.data.cpu()), 0)
    
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, _ in img_path:
        filename = os.path.basename(path)
        # 获取标签（分类id）
        label = filename[0:4]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        # 获取camera的id
        camera = filename.split('c')[1]
        camera_id.append(int(camera[0]))
    
    return camera_id, labels


def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)  # 把query特征放到一列上
    score = torch.mm(gf, query)  # 计算余弦距离，余弦距离等于L2归一化之后的内积
    score = score.squeeze(1).cpu()
    score = score.numpy()

    index = np.argsort(score)  # 按余弦距离进行排序，对应名次
    index = index[::-1]  # 逆序

    query_index = np.argwhere(gl == ql)  # 找出gallery label和query label相同的位置
    camera_index = np.argwhere(gc == qc)  # 找出gallery camera和query camera相同的位置

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)  # 找出label相同但camera不同的位置
    junk_index1 = np.argwhere(gl == -1)  # 错误检测的图像
    junk_index2 = np.intersect1d(query_index, camera_index)  # 相同的人在同一摄像头下的图像
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
