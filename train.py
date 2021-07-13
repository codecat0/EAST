"""
@File : train.py
@Author : CodeCat
@Time : 2021/7/13 上午10:01
"""


import os
import time
import argparse

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data

from my_dataset import Custom_dataset
from model.east_resnet50 import EAST
from utils.loss import Loss


def main(opt):
    file_num = len(os.listdir(opt.train_img_path))
    trainset = Custom_dataset(opt.train_img_path, opt.train_gt_path)
    train_loader = data.DataLoader(
        dataset=trainset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True
    )
    criterion = Loss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = EAST()
    model.to(device)
    if opt.weights != '':
        weights_dict = torch.load(opt.weights, map_location=device)
        model.load_state_dict(weights_dict, strict=False)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[opt.epochs//2], gamma=0.1)

    for epoch in range(opt.epochs):
        model.train()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                epoch + 1, opt.epochs, i + 1, int(file_num / opt.batch_size), time.time() - start_time, loss.item()))

        scheduler.step()
        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/opt.batch_size), time.time() - epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('='*50)
        if (epoch+1) % opt.interval == 0:
            torch.save(model.state_dict(), os.path.join(opt.pths_path, 'model_epoch_{}.pth'.format(epoch+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img_path', type=str, default='./ICDAR_2015/train_img')
    parser.add_argument('--train_gt_path', type=str, default='./ICDAR_2015/train_gt')
    parser.add_argument('--pths_path', type=str, default='./weights')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--weights', type=str, default='')

    opt = parser.parse_args()
    print(opt)
    main(opt)