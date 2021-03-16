import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--masks', default='/home/youssef/Documents/phdYoop/datasets/synthgiraffe/ID/PNG')
parser.add_argument('--dataset', default='synthgiraffe')
args = parser.parse_args()

orig_im = args.masks
mpaths = sorted(glob.glob(os.path.join(orig_im, '*.png')))
np.random.seed(42)
np.random.shuffle(mpaths)
lenm = len(mpaths)

trainpaths = mpaths[:int(0.9*len(mpaths))]
valpaths = mpaths[int(0.9*len(mpaths)):]

dataset = args.dataset
trainpath = os.path.join('./data', dataset + '1_bg0', 'train')
valpath = os.path.join('./data', dataset + '1_bg0', 'val')

os.makedirs(trainpath, exist_ok=True)
os.makedirs(valpath, exist_ok=True)

for tf in trainpaths:
    os.symlink(tf, os.path.join(trainpath, os.path.basename(tf)))
    # os.symlink(tf, os.path.join(trainpath, 'japanesemaple_%s.png' % str(int(os.path.basename(tf).split('_')[1][:4]))))

for vf in valpaths:
    os.symlink(vf, os.path.join(valpath, os.path.basename(vf)))
    # os.symlink(vf, os.path.join(valpath, 'japanesemaple_%s.png' % str(int(os.path.basename(vf).split('_')[1][:4]))))