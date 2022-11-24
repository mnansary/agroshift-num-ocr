#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import cv2 
import numpy as np
import random
from PIL import Image
import math
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path

def random_exec(poplutation=[0,1],weights=[0.7,0.3],match=0):
    return random.choices(population=poplutation,weights=weights,k=1)[0]==match

#---------------------------------------------------------------
# image utils
#---------------------------------------------------------------
def stripPads(arr,
              val):
    '''
        strip specific value
        args:
            arr :   the numpy array (2d)
            val :   the value to strip
        returns:
            the clean array
    '''
    # x-axis
    arr=arr[~np.all(arr == val, axis=1)]
    # y-axis
    arr=arr[:, ~np.all(arr == val, axis=0)]
    return arr

def padToHeight(img,height,pad_val=255):
    # shape
    h,w=img.shape    
    pad_loc=random.choice(["up","down"])
    _pad =np.ones((height-h,w))*pad_val
        
    if pad_loc=="up":img =np.concatenate([_pad,img],axis=0)
    else:img =np.concatenate([img,_pad],axis=0)
    return img

#---------------------------------------------------------------
# parsing utils
#---------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

#----------------------------------------
# noise utils
#----------------------------------------
def gaussian_noise(height, width):
    """
        Create a background with Gaussian noise (to mimic paper)
    """

    # We create an all white image
    image = np.ones((height, width)) * 255

    # We add gaussian noise
    cv2.randn(image, 235, 10)

    return np.array(Image.fromarray(image).convert("RGB"))

def quasicrystal(height, width):
    """
        Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
    """

    image = Image.new("L", (width, height))
    pixels = image.load()

    frequency = random.random() * 30 + 20  # frequency
    phase = random.random() * 2 * math.pi  # phase
    rotation_count = random.randint(10, 20)  # of rotations

    for kw in range(width):
        y = float(kw) / (width - 1) * 4 * math.pi - 2 * math.pi
        for kh in range(height):
            x = float(kh) / (height - 1) * 4 * math.pi - 2 * math.pi
            z = 0.0
            for i in range(rotation_count):
                r = math.hypot(x, y)
                a = math.atan2(y, x) + i * math.pi * 2.0 / rotation_count
                z += math.cos(r * math.sin(a) * frequency + phase)
            c = int(255 - round(255 * z / rotation_count))
            pixels[kw, kh] = c  # grayscale
    return np.array(image.convert("RGB"))

    #---------------------wrapper
def paper_noise(img):
    if random_exec(weights=[0.75,0.25]):
        h,w=img.shape
        back_fn=random.choice([quasicrystal,gaussian_noise])
        back=back_fn(h,w)
        r=random.randint(0,25)
        g=random.randint(0,25)
        b=random.randint(0,25)
        # background
        back[img==0]=(r,g,b)
        back=255-back
        # fore ground
        r=random.randint(0,25)
        g=random.randint(0,25)
        b=random.randint(0,25)
        back[img!=0]=(r,g,b)
        img=np.copy(back)
    else:
        img=cv2.merge((img,img,img))
        img=255-img
    if random_exec():
        img=cv2.GaussianBlur(img,(3,3),0)
    return img