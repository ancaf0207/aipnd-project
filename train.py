

import numpy as np

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models

from torch import nn
from torch import optim
import torch.nn.functional as F


from PIL import Image



from fx_training import loading_data

from fx_training import build_and_train_model





import argparse
parser = argparse.ArgumentParser('Arguments for the train.py file')
parser.add_argument('data_dir', type=str, help='Provide the path to the training/validation data')
parser.add_argument('--save_dir',default='/home/workspace/aipnd-project', type=str, help='Where should the trained model be saved to?')
parser.add_argument('--arch', type=str, default='vgg16', help="What VGG you want to use: 'vgg11', 'vgg13' or 'vgg16'? ")
parser.add_argument('--learning_rate', type=int, default=0.5, help='What learning rate do you want to use (default =0.05)?')
parser.add_argument('--hidden_units', type=dict, default=4096, help='How many hidden units do you want to use (default =4096)?')
parser.add_argument('--epochs', type=int, default='1', help='For how many epochs do you want to train the model (default = 1)')
parser.add_argument('--device', type=str, default='cpu', help='do you want to use cpu or gpu?')



args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units=args.hidden_units
epochs=args.epochs
device=args.device


print('I am starting to train the model')


build_and_train_model(data_dir,save_dir, arch, learning_rate, hidden_units, epochs, device)

