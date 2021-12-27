import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2，3"

from models import Resnet50_ft
from utils.dataloader import preprocess
from utils.utils import train_one_epoch, val_one_epoch
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/mnt/disk2/data_sht/Market-1501/pytorch')
    parser.add_argument('--epochs', type=int, default=60)
    # 选择backbone
    # TODO 可选：resnet50
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--h', type=int, default=256)
    parser.add_argument('--w', type=int, default=128)

    parser.add_argument('--erase_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    model_name = args.backbone
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    epochs = args.epochs

    tb_writer = SummaryWriter()

    # 加载数据集
    train_transformer, val_transformer = preprocess(args.h, args.w, args.erase_rate)
    train_path = os.path.join(args.dataset_path, 'train_all')
    val_path = os.path.join(args.dataset_path, 'val')

    train_dataset = datasets.ImageFolder(train_path, train_transformer)
    val_dataset = datasets.ImageFolder(val_path, val_transformer)

    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)

    class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # 生成模型
    if args.backbone == 'resnet50':
        model = Resnet50_ft(len(class_names))
    
    model = model.cuda()

    # 准备优化器
    optim_name = optim.SGD
    # 为backbone和分类头分配不同的学习率
    if args.backbone == 'PCB':
        pass
    else:
        ignored_params = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        classifier_params = model.classifier.parameters()
        optimizer = optim_name([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': classifier_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.93)

    criterion = nn.CrossEntropyLoss()

    train_step = train_dataset_size // batch_size
    val_step = val_dataset_size // batch_size

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, epoch, epochs, train_step, train_loader, args.cuda)
        val_loss = val_one_epoch(model, criterion, optimizer, epoch, epochs, val_step, val_loader, args.cuda)
        scheduler.step()

        # loss_history.append_loss(train_loss / train_step, val_loss / val_step)
        tb_writer.add_scalar('train loss', train_loss / train_step, epoch)
        tb_writer.add_scalar('val loss', val_loss / val_step, epoch)

        print('Epoch:'+ str(epoch + 1) + '/' + str(epochs))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (train_loss / train_step, val_loss / val_step))
        # 保存模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'weights/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, train_loss / train_step, val_loss / val_step))

    print('ALL DONE!')