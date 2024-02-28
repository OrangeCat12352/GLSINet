# copy from https://github.com/Jun-Pu/Evaluation-on-salient-object-detection
import torch
import torch.nn as nn
import argparse
import os.path as osp
import os

from eval_utils.dataloader import EvalDataset
from eval_utils.evaluator import Eval_thread


def main(cfg):
    root_dir = cfg.root_dir
    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
    else:
        output_dir = root_dir
    gt_dir = osp.join(root_dir, 'gt')
    pred_dir = osp.join(root_dir, 'pred')
    if cfg.methods is None:
        method_names = os.listdir(pred_dir)
    else:
        # method_names = cfg.methods.split(' ')
        method_names = cfg.methods
    if cfg.datasets is None:
        dataset_names = os.listdir(gt_dir)
    else:
        # dataset_names = cfg.datasets.split(' ')
        dataset_names = cfg.datasets
    threads = []
    for dataset in dataset_names:
        for method in method_names:
            loader = EvalDataset(osp.join(pred_dir, method, dataset), osp.join(gt_dir, dataset))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            threads.append(thread)
    for thread in threads:
        result = thread.run()
        print_state = result[0]
        Sm_info = result[1]
        mae_info = result[2]
        maxEm_info = result[1]
        fbw_info = result[1]
        print(print_state)
        return Sm_info, mae_info, maxEm_info, fbw_info

    # if __name__ == '__main__':


def train_val():
    # MODEL_NAME = os.listdir(os.getcwd() + '/pred/')
    # MODEL_NAME.sort(key=lambda x: x[:])
    MODEL_NAME = ['MyNet']
    # DATA_NAME = os.listdir(os.getcwd() + '/gt/')
    # DATA_NAME.sort(key=lambda x: x[:])
    #DATA_NAME = ['ORS-4199']
    DATA_NAME = ['EORSSD']
    path = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default=MODEL_NAME)
    parser.add_argument('--datasets', type=str, default=DATA_NAME)
    # parser.add_argument('--root_dir', type=str, default=os.getcwd())
    parser.add_argument('--root_dir', type=str, default=path)
    # parser.add_argument('--save_dir', type=str, default=os.getcwd() + '/result/')
    parser.add_argument('--save_dir', type=str, default=path + '/result/')
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    Sm_info, mae_info, maxEm_info, fbw_info = main(config)
    return Sm_info, mae_info, maxEm_info, fbw_info


if __name__ == '__main__':
    a, b = train_val()
    print(a)
    print(b)
