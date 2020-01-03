import torch
from torch.utils import data
import numpy as np
from random import randrange
from Downsampling import OneHotEncodingImage


class DatasetFirst(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, data2, attributes):
        'Initialization'
        self.data = data['b_']
        self.data_processed = data2['downsampledImg']["downsampledData"]
        self.attributes = attributes['df_final']["block0_values"] ### HER JOHAN


  def __len__(self):
        'Denotes the total number of samples'
        return self.data_processed.shape[0]
#
  def __getitem__(self, index):
        'Generates one sample of data'
        data_processed = self.data_processed[index]
        data = self.data[index][0]
        attributes = self.attributes[index]
        wrong_index = (index + randrange(self.data_processed.shape[0])) % self.data_processed.shape[0]
        wrong_data = self.data[wrong_index][0]
        wrong_processed = self.data_processed[wrong_index]
        segmentatedImage_onehot = OneHotEncodingImage(data, 7)
        data_wrong_onehot = OneHotEncodingImage(wrong_data, 7)

        return attributes, data_processed, data, segmentatedImage_onehot, wrong_data, wrong_processed, data_wrong_onehot



class DatasetSecond(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, data2, attributes):
        'Initialization'
        self.images = data['ih']
        self.data_processed = data['b_']
        self.attributes = attributes['df_final']["block0_values"]

  def __len__(self):
        'Denotes the total number of samples'
        return self.images.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        data_processed = self.data_processed[index][0]
        data = self.images[index]
        attributes = self.attributes[index]
        wrong_index = (index + randrange(self.data_processed.shape[0])) % self.data_processed.shape[0]
        wrong_data = self.images[wrong_index]
        wrong_processed = self.data_processed[wrong_index][0]
        segmentatedImage_onehot = OneHotEncodingImage(data_processed, 7)
        segmentatedImage_onehot_wrong = OneHotEncodingImage(wrong_processed, 7)
        

        return attributes, data_processed, data, wrong_data, wrong_processed, segmentatedImage_onehot, segmentatedImage_onehot_wrong