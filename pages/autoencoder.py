import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms as T
from torchvision.utils import save_image
from torchsummary import summary

import os
from torchvision import transforms as T
from torchvision.io import read_image

import streamlit as st

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            # nn.Dropout(),
            nn.SELU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            # nn.Dropout(),
            nn.SELU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SELU()
            )
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True) #<<<<<< Bottleneck
        
        #decoder
        # Как работает Conv2dTranspose https://github.com/vdumoulin/conv_arithmetic

        self.unpool = nn.MaxUnpool2d(2, 2)
        
        self.conv0_t = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.SELU()
            )
        self.conv1_t = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=2),
            nn.BatchNorm2d(256),
            nn.SELU()
            )
        self.conv2_t = nn.Sequential(
            nn.ConvTranspose2d(256, 1, kernel_size=4, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )        

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x, indicies = self.pool(x) # ⟸ bottleneck
        return x, indicies

    def decode(self, x, indicies):
        x = self.unpool(x, indicies)
        x = self.conv0_t(x)
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        return x

    def forward(self, x):
        latent, indicies = self.encode(x)
        out = self.decode(latent, indicies)      

        return out


model = ConvAutoencoder()

model.load_state_dict(torch.load('/home/alex/projects/Computer-Vision/autoencoder_model_weights.pt'))


st.subheader('Это приложение производит очистку текста от шума методом автоэнкодинга :ru:')
# st.markdown("# Функция ❄️")
st.sidebar.markdown("Очистка текста от шума")


uploaded_file = st.file_uploader('Загрузите изображение')
if uploaded_file is not None:
    st.image(uploaded_file, use_column_width=True)
    image = Image.open(uploaded_file)
    image = image.convert('L')
    image = T.ToTensor()(image).unsqueeze(0)
    image = T.Resize((250, 600))(image)
    model.eval()  # в модель подается torch.Size([1, 1, 258, 540])
    img2 = model(image)
    st.image(img2.detach().numpy().squeeze(0).squeeze(0), caption='Изображение, очищенное от шумов', use_column_width=True)
