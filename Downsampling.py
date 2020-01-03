import torch
import numpy as np
from PIL import Image

def OneHotEncodingImage(a, ncols):
    #ncols = a.max()+1
    out = np.zeros( (a.size,ncols), dtype=np.uint8)
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out

def DownsampleImage(image, downsampleSize):
  imageFromNumpy = Image.fromarray(image)
  downsampledImage = np.array(imageFromNumpy.resize(downsampleSize))
  #downsampledImage = misc.imresize(image, downsampleSize,  interp='bicubic')
  return downsampledImage

def DownsizeAttributes(image):
  image[image == 0] = 3
  image[image == 1] = 3
  image[image == 4] = 3
  image[image == 7] = 3
  image[image == 8] = 3

  image[image == 2] = 0
  image[image == 5] = 1
  image[image == 6] = 2
  return image

def GetSegmentationImageReady(image, downsampleSize):
  downsizedImage = DownsizeAttributes(image)
  oneHotEncodedImage = OneHotEncodingImage(downsizedImage[0], 4)
  
  downsampledImage = np.empty((4,8,8))
  downsampledImage[0] = DownsampleImage(oneHotEncodedImage[0], downsampleSize)
  downsampledImage[1] = DownsampleImage(oneHotEncodedImage[1], downsampleSize)
  downsampledImage[2] = DownsampleImage(oneHotEncodedImage[2], downsampleSize)
  downsampledImage[3] = DownsampleImage(oneHotEncodedImage[3], downsampleSize)
  return downsampledImage

def normalize_image(real_image):
    real_image = (real_image - real_image.min()) / (real_image.max()-real_image.min())
    return real_image

