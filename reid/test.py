import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from torchvision import datasets
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2，3"

import scipy.io

from models import Resnet50_ft
from utils.utils import extract_feature, get_id, evaluate
from utils.dataloader import preprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')

    parser.add_argument('--dataset_path', type=str, default='/mnt/disk2/data_sht/Market-1501/pytorch')
    parser.add_argument('--num_classes', type=int, default=751)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--h', type=int, default=256)
    parser.add_argument('--w', type=int, default=128)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    num_classes = args.num_classes
    dataset_path = args.dataset_path
    h, w = args.h, args.w
    batch_size = args.batch_size
    checkpoint = args.checkpoint

    # 加载数据
    _, transform = preprocess(args.h, args.w, 0)

    gallery_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'gallery'), transform)
    query_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'query'), transform)

    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    class_names = query_dataset.classes

    # 加载模型
    model = Resnet50_ft(num_classes)
    model.load_state_dict(torch.load(checkpoint))
    # 去除分类头
    model.classifier.classifier = nn.Sequential()

    model = model.eval()
    if args.cuda:
        model = model.cuda()
    
    with torch.no_grad():
        gallery_feature = extract_feature(model, gallery_loader)
        query_feature = extract_feature(model, query_loader)

    # 其他检测评估的指标
    gallery_cam, gallery_label = get_id(gallery_dataset.imgs)
    query_cam, query_label = get_id(query_dataset.imgs)

    # 开始评估
    result = {
        'gallery_f': gallery_feature.numpy(),
        'gallery_label': gallery_label,
        'gallery_cam': gallery_cam,
        'query_f': query_feature.numpy(),
        'query_label': query_label,
        'query_cam': query_cam
    }
    scipy.io.savemat('result.mat', result)

    os.system('python evaluate.py')
