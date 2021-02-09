
# Imports here


import numpy as np

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models

from torch import nn
from torch import optim
import torch.nn.functional as F


from PIL import Image


from workspace_utils import keep_awake

from workspace_utils import active_session



def loading_data(data_path):

    
    train_dir = data_path + '/train'
    valid_dir = data_path + '/valid'
    test_dir = data_path + '/test'



    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms =  transforms.Compose([transforms.RandomRotation(20),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]) 


    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])



    # TODO: Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)




    # TODO: Using the image datasets and the trainforms, define the dataloaders

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)



    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    return   train_loader, valid_loader, test_loader, train_data
        

        
        
def build_and_train_model(data_path,checkpoint_dir, base_model='vgg16', learning_rate=0.05, hidden_units=4096, epochs=1, device='cuda'):

    train_loader=loading_data(data_path)[0]
    valid_loader=loading_data(data_path)[1]

    models_options={'vgg11':models.vgg11(pretrained=True),'vgg13':models.vgg13(pretrained=True),'vgg16':models.vgg16(pretrained=True)}
    model = models_options[base_model]
    

    
    l_rate=learning_rate

    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # do not backpropagate through the parameters of the loaded model
    for param in model.parameters():
        param.requires_grad = False

    # define my feed forward
    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.Dropout(0.3),
                                     nn.ReLU(),
                                     nn.Linear(hidden_units,102),
                                     nn.LogSoftmax(dim=1))
    # define the criterion
    criterion = nn.NLLLoss()

    # define optimizer of parameters - only for the classifier, not the imported model
    optimizer = optim.Adam(model.classifier.parameters(), lr=l_rate)

    model.to(device);

    steps = 0
    running_loss = 0
    print_every = 10


    # Train the NW

    for i in keep_awake(range(5)):  #anything that happens inside this loop will keep the workspace active
        # do iteration with lots of work here
        with active_session():
            # do long-running work here


            for epoch in range(epochs):
                for images, labels in train_loader:
                    steps += 1
                    # Move input and label tensors to the default device
                    images, labels = images.to(device), labels.to(device)

                    # reset gradient to 0 
                    optimizer.zero_grad()

                    # forward step
                    result = model.forward(images)
                    # calculate loss
                    loss = criterion(result, labels)
                    # backpropagate
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    if steps % print_every == 0:
                        eval_loss = 0
                        accuracy = 0

                        # Set the model in validation mode 

                        model.eval()

                        # turn off gradient during validation

                        with torch.no_grad():
                            for images, labels in valid_loader:
                                images, labels = images.to(device), labels.to(device)
                                result = model.forward(images)
                                batch_loss = criterion(result, labels)

                                eval_loss += batch_loss.item()

                                # Calculate accuracy
                                ps = torch.exp(result)
                                top_p, top_class = ps.topk(1, dim=1)
                                equals = top_class == labels.view(*top_class.shape)
                                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            else:
                        # print testing vs validation loss and accuracy

                        print(f"Epoch {epoch+1}/{epochs}.. "
                              f"Train loss: {running_loss/print_every:.3f}.. "
                              f"Eval loss: {eval_loss/len(valid_loader):.3f}.. "
                              f"Eval accuracy: {accuracy/len(valid_loader):.3f}")
                        running_loss = 0
                        model.train()

                        # EXPORT THE CHECKPOINT!

    model.class_to_idx = loading_data(data_path)[3].class_to_idx

    checkpoint = {'network': base_model,
                  'input_size': 25088,
                  'output_size': 102,
                  'learning_rate': learning_rate,       
                  'batch_size': 64,
                  'classifier' : model.classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, checkpoint_dir+'/checkpoint_terminal.pth')
     
    print('Your model has been trained and saved as checkpoint in the folder you indicated.', checkpoint_dir)


