
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os, argparse
from datetime import datetime
from torch.utils.data import random_split, DataLoader
from model.MBAF-PVNet_VGG_models import MBAF_PVNet_VGG
from data import get_loader
from utils import clip_gradient, adjust_lr
import iou

torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
opt = parser.parse_args()

print(f'Learning Rate: {opt.lr}, ResNet: {opt.is_ResNet}')

# Build model
model = MBAF_PVNet_VGG()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), opt.lr)

# Load dataset
image_root = '/tmp/pycharm_project_729/PV_dataset/JPEGImages/'
gt_root = '/tmp/pycharm_project_729/PV_dataset/SegmentationClass/'
full_dataset = get_loader(image_root, gt_root, batchsize=1, trainsize=opt.trainsize).dataset

# Split dataset 8:1:1
dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchsize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=False)

total_step = len(train_loader)
CE = nn.BCEWithLogitsLoss()
IOU = iou.IOU(size_average=True)


def train_one_epoch(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        s1, s2, s3, s4, s5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = model(images)
        loss = sum(CE(s, gts) + IOU(sig, gts) for s, sig in
                   zip([s1, s2, s3, s4, s5], [s1_sig, s2_sig, s3_sig, s4_sig, s5_sig]))
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.6f}, Loss: {:.4f}'.format(
                datetime.now(), epoch, opt.epoch, i, total_step,
                opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.item()))


def validate(val_loader, model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, gts in val_loader:
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            s1, s2, s3, s4, s5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = model(images)
            loss = sum(CE(s, gts) + IOU(sig, gts) for s, sig in
                       zip([s1, s2, s3, s4, s5], [s1_sig, s2_sig, s3_sig, s4_sig, s5_sig]))
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")


def save_model(model, epoch):
    save_path = 'models/ACCoNet_VGG/'
    os.makedirs(save_path, exist_ok=True)
    filename = f"{save_path}ACCoNet_ResNet.pth.{epoch}" if opt.is_ResNet else f"{save_path}ACCoNet_VGG.pth.{epoch}"
    torch.save(model.state_dict(), filename)
    print(f"Model saved at {filename}")


def test(test_loader, model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, gts in test_loader:
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            s1, s2, s3, s4, s5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = model(images)
            loss = sum(CE(s, gts) + IOU(sig, gts) for s, sig in
                       zip([s1, s2, s3, s4, s5], [s1_sig, s2_sig, s3_sig, s4_sig, s5_sig]))
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")


print("Let's go!")
for epoch in range(1, opt.epoch + 1):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train_one_epoch(train_loader, model, optimizer, epoch)

    if epoch % 5 == 0:
        validate(val_loader, model)
        save_model(model, epoch)

        print("Testing final model on test set...")
        test(test_loader, model)
