import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import streamlit as st


@st.cache
def readLabels():
    classes_labels = []
    f = open("labels.txt")
    content = f.read()
    class_list = content.split("\n")
    f.close()
    for classes in class_list:
        x = classes.index('\'')
        y = classes.rindex('\'')
        classes_labels.append(classes[x+1:y])
    return classes_labels

def detect(filename):
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device('cpu')
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
    model.eval()
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)


    with torch.no_grad():
        output = model(input_batch)
    _, pre = torch.max(output.data, 1)
    return pre.numpy()[0]

st.title("Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = detect(uploaded_file)
    st.write(readLabels()[label])