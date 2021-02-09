# Functions for prediction



import numpy as np

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models

from torch import nn
from torch import optim
import torch.nn.functional as F


from PIL import Image



from fx_predicting import checkpoint_loading

from fx_predicting import process_image

from fx_predicting import predict

import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


import argparse
parser = argparse.ArgumentParser('Arguments for the predict.py file')
parser.add_argument('image_path', type=str, help='Provide the path of the flower image you want the model to identify')
parser.add_argument('checkpoint_path', type=str, help='What model checkpoint do you want to use? ')
parser.add_argument('--top_k', type=int, default=5, help='How many predictions do you want to show?')
parser.add_argument('--category_names', type=dict, default=cat_to_name, help='Do you have a specific mapping json file?')
parser.add_argument('--data_dir', type=str, default='cpu', help='so you want to use cpu or gpu?')

args = parser.parse_args()
image_path = args.image_path
checkpoint_path = args.checkpoint_path
topk = args.top_k
category_names=args.category_names
data_dir=args.data_dir




predict(image_path, checkpoint_path, topk, category_names,data_dir)
