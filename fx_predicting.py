# Functions for prediction


import numpy as np

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models

from torch import nn
from torch import optim
import torch.nn.functional as F


from PIL import Image




import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



def checkpoint_loading(path):
  
    if torch.cuda.is_available():
        location=lambda storage, loc: storage.cuda()
    else:
        location='cpu'
    
    checkpoint = torch.load(path, map_location=location)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    return model






def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image=Image.open(image_path)
    pil_transforms =  transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 
    
    processed_image=pil_transforms(image)
    
    return processed_image
    
       
    





def predict(image_path, checkpoint_path, topk=5, name_map=cat_to_name, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
   
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #load the model
    model=checkpoint_loading(checkpoint_path)
    
    
    model.to(device)
    model.eval()
    
    image=process_image(image_path)
    image.unsqueeze_(0)
    
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        log_probabilities = model.forward(image.cpu())
        probabilities = torch.exp(log_probabilities)
        top_prob, top_cls = probabilities.topk(topk, dim = 1)   
        class_to_idx_inverted = {model.class_to_idx[c]: c for c in model.class_to_idx}
    
    class_probability=top_prob.cpu().numpy()[0]
     
    # create the list of predicted classes
    classes = list()
    for label in top_cls.cpu().numpy()[0]:
        classes.append(class_to_idx_inverted[label])
    
    # create the list of predicted classes
    flower_label=[cat_to_name[flower] for flower in classes]
    
    
    
    print('You wanted to know what flower is '+image_path+'. The model calculated these options: ', classes, 'with these probabilities: ', class_probability)
    print('The name of the flowers are:', flower_label)
    print('I used for processing the location ', device)



def show_results(image_path, model, topk=5):
    
    # transform the image (for ease of visualization)
    pil_transforms =  transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224)])
    
    # run the prediction and extract the details needed for the chart.
    
    flower_label=[cat_to_name[flower] for flower in predict(image_path, model, topk)[1]]
    
    predict(image_path, model, topk)[1]
    pred_prob=predict(image_path, model, topk)[0]

    
    fig=plt.figure(figsize=(10,5))
    
    # plot the image 
    ax21=plt.subplot(121)
    ax21=plt.imshow(pil_transforms(Image.open(image_path)))

    # plot the predictions' probability 
    ax22=plt.subplot(122)
    ax22=plt.barh(flower_label,pred_prob)
    ax22=plt.xlabel('Predicted Probability')
    ax22=plt.ylabel('Flower name')
    fig.suptitle('Top '+str(topk)+' flower name predictions', fontsize=14)
    fig.tight_layout(pad=5.0)





