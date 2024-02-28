import torch
import torch.nn.functional as F

import numpy as np
import argparse

import imageio
import time

from model.MyNet import MyNet
from data import test_dataset

import os


def pred_generate(epoch):
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    opt = parser.parse_args()

    dataset_path = ''

    model = MyNet()

    model.load_state_dict(torch.load('result/weight/MyNet.pth.{}'.format(epoch)))

    model.cuda()
    model.eval()

    #test_datasets = ['ORSSD_aug']
    test_datasets = ['EORSSD_aug']
    # test_datasets = ['ORS-4199_aug']
    for dataset in test_datasets:
        save_path = 'eval_utils/pred/MyNet/EORSSD/'
        #save_path = 'eval_utils/pred/MyNet/ORS-4199/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + '/' + dataset + '/' + 'test/image/'
        gt_root = dataset_path + '/' + dataset + '/' + 'test/GT/'
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res, _ = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res * 255
            res = res.astype(np.uint8)
            imageio.imsave(save_path + name, res)
