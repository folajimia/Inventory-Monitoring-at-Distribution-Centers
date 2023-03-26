#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
from torch.optim import lr_scheduler

import logging
import sys
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    print("Testing Model on Whole Testing Dataset")
    
    # putting the model into evaluation mode
    model.eval()
    #initalizing test loss and correctly classified samples
    test_loss = 0 
    correct_pred = 0
    
    #turning off gradient calcultation and iterating over test loader batches
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # move data and target to GPU
            data = data.to(device)
            target = target.to(device)
            
            # Forward passing the data thru the model
            output = model(data)
            
            #calc test loss with averaging over the batch
            test_loss +=criterion(output, target).item()* data.size(0)
            
            #the predicted class for each sample(index of the maximum log probability)
            pred = output.argmax(dim=1, keepdim=True)
            
            #counting the number of correctly classified samples 
            correct_pred = pred.eq(target.view_as(pred)).sum().item()
    
    # Average the test loss over test samples
    test_loss /= len(test_loader.dataset)
    
    # printing out the test loss and accuracy
    
    print(
        "\nTest : Average Test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct_pred, len(test_loader.dataset), 100.0 * correct_pred / len(test_loader.dataset)
        )
    )
    

def train(model, train_loader, validation_loader, criterion, optimizer, device, epochs):
    
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    
    logger.info("Start Model Training")
    
     # putting the model into evaluation mode
    model.train()
    #initalizing test loss and correctly classified samples
    running_loss = 0 
    correct_pred = 0
    
    # Tterating over train loader batches
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # move data and target to GPU
        data = data.to(device)
        target = target.to(device)
        
        #Reset gradients of the models parameter to zero
        optimizer.zero_grad()
        
        # Forward passing the data thru the model to make predictions
        output = model(data)
        
        #making predictions for each mini batch
        pred = output.argmax(dim=1, keepdim=True)
        
        #counting the number of correctly classified samples(correct predictions) 
        correct_pred += pred.eq(target.view_as(pred)).sum().item()
        
        #calc running loss with averaging over the batch
        loss = criterion(output, target)
        
        #adding loss multiplied by batch size
        running_loss +=loss.item()* data.size(0)
        
        # gradient of the loss
        loss.backward()

        # update the models parameter based on computed gradients
        optimizer.step()
        
        # print training progress every 10 mini-batches
        if batch_idx % 10 == 0:
            print(
                "Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}". format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                
                )
            )
    
    # printing out the running loss and accuracy
    
    print(
        "n\Train : Average Test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            running_loss/len(train_loader.dataset), correct_pred, len(train_loader.dataset), 100.0 * correct_pred / len(train_loader.dataset)
        )
    )
            
    return model



    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    #initializes a pretrained model
    
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features

    model.fc = nn.Sequential(
                   nn.Linear(num_features, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 5))
    return model



def create_data_loaders(train_data, test_data, validation_data, batch_size):
    
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    #define image transformers for training data
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])

    # define image transformers for testing data
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
  
    
    # transform images
    train_dataset = torchvision.datasets.ImageFolder(root=train_data, transform=training_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_data, transform=testing_transform)
    validation_set = torchvision.datasets.ImageFolder(root = validation_data, transform = testing_transform)
    
    
    # load data
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    validation_data_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size) 
    
    
    return train_data_loader, test_data_loader, validation_data_loader




def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Running on Device {device}")
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)


    train_loader, test_loader, validation_loader = create_data_loaders(
                        args.train_data, args.test_data, args.val_data, args.batch_size)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    model = train(model, train_loader, validation_loader,
                  loss_criterion, optimizer, device, args.epochs)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Start Model Testing")
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Model Saving")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size", type=int, default=64
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=32
    )
    parser.add_argument(
        "--epochs", type=int, default=10
    )
    parser.add_argument(
        "--lr", type=float, default=1.0
    )
    parser.add_argument(
        "--model-dir", type=str, default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--train_data", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument(
        "--test_data", type=str, default=os.environ["SM_CHANNEL_TEST"]
    )
    parser.add_argument(
        "--val_data", type=str, default=os.environ["SM_CHANNEL_VAL"]
    )
    
    args = parser.parse_args()
    print("args to train_model.py : ", args)
    
    args=parser.parse_args()
    
    main(args)
