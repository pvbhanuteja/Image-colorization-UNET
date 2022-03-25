import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from datetime import datetime
import shutil, argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, rgb2gray, lab2rgb
plt.switch_backend('agg')


def get_time_str():
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    # convert to datetime
    date_time = datetime.fromtimestamp(ts)
    # convert timestamp to string in dd-mm-yyyy HH:MM:SS
    str_date_time = date_time.strftime("%d-%m-%Y-%H:%M")
    return(str_date_time)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 128
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def visualize_image(grayscale_input, ab_input=None, show_image=False, save_path=None, save_name=None):
    '''Show or save image given grayscale (and ab color) inputs. Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf() # clear matplotlib plot
    ab_input = ab_input.cpu()
    grayscale_input = grayscale_input.cpu()    
    if ab_input is None:
        grayscale_input = grayscale_input.squeeze().numpy() 
        if save_path is not None and save_name is not None: 
            plt.imsave(grayscale_input, '{}.{}'.format(save_path['grayscale'], save_name) , cmap='gray')
        if show_image: 
            plt.imshow(grayscale_input, cmap='gray')
            plt.show()
    else: 
        color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
        color_image = color_image.transpose((1, 2, 0))  
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
        color_image = lab2rgb(color_image.astype(np.float64))
        grayscale_input = grayscale_input.squeeze().numpy()
        if save_path is not None and save_name is not None:
            plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
            plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
        if show_image: 
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(grayscale_input, cmap='gray')
            axarr[1].imshow(color_image)
            plt.show()


def combine_channels(gray_input, ab_input, lab_version):
    '''
    Function for combining the grayscale and ab layers into a single Lab image 
    and converting it back to RGB.
    Two Lab versions are allowed:
    * 1 - the output of the a/b channels is in the range of [-1,1]
    * 2 - the output of the a/b channels is in the range of [0,1]
    Parameters
    ----------
    gray_input : torch.tensor
        A tensor containing the grayscale image
    ab_input : torch.tensor
        A tensor containing the corresponding a/b channels of a Lab image
    lab_version : int 
        Version of the Lab formatting used 
    Returns
    -------
    gray_output : np.ndarray
        The grayscale image
    color_output : np.ndarray
        The RGB image obtained from the Lab color space
    '''
    
    if gray_input.is_cuda: gray_input = gray_input.cpu()
    if ab_input.is_cuda: ab_input = ab_input.cpu()
    
    # combine channels
    color_image = torch.cat((gray_input, ab_input), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    
    # reverse the transformation from DataLoaders
    if lab_version == 1:
        color_image = color_image * [100, 128, 128]
    elif lab_version == 2:
        color_image = color_image * [100, 255, 255] - [0, 128, 128]
    else:
        raise ValueError('Incorrect Lab version!!!')
    
    # prepare the grayscale/RGB imagers
    gray_output = gray_input.squeeze().numpy()
    color_output = lab2rgb(color_image.astype(np.float64))
    
    return gray_output, color_output

def save_temp_results(gray_input, ab_input, lab_version, save_path=None, save_name=None):
    '''
    Show/save rgb image from grayscale and ab channels
    Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}
    '''
    
    plt.clf() # clear matplotlib 
    
    gray_output, color_output = combine_channels(gray_input, ab_input, lab_version)
    
    if save_path is not None and save_name is not None: 
        plt.imsave(arr=gray_output, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_output, fname='{}{}'.format(save_path['colorized'], save_name))

def save_pred_results(gray_input, ab_input, lab_version, save_path=None, save_name=None):
    '''
    Show/save rgb image from grayscale and ab channels
    Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}
    '''
    
    plt.clf() # clear matplotlib 
    
    gray_output, color_output = combine_channels(gray_input, ab_input, lab_version)
    
    if save_path is not None and save_name is not None: 
        plt.imsave(arr=color_output, fname='{}{}'.format(save_path['colorized'], save_name))