import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def net():
    #using a pretrained resnet50
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    # get number of features for the last connected layer of the model
    num_features=model.fc.in_features
    # replace the model's last fully connected layer with a layer that has input features same as num_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 5))
    return model

def model_fn(model_dir):
    print("Model directory is -")
    print(model_dir)
    # Check if GPU is enabled
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #instantiate neural net
    model = net().to(device)
    
    #open saved model
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the model")
        #load model to device
        model.load_state_dict(torch.load(f, map_location = device))
        print('MODEL-LOADED')
        logger.info('model loaded successfully')
    model.eval()
    return model

# process input data for inference
def input_fn(request_body, content_type='image/jpeg'):
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    logger.info('In the predict fn')
    #define transformation
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    logger.info("transforming input")
    #apply transformation
    input_object=test_transform(input_object)
    #move to GPU if available
    if torch.cuda.is_available():
        input_object = input_object.cuda()
    # make prediction
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction