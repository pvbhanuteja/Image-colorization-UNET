from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models import *
from utils import progress_bar, get_time_str
from colorize_data import *
from models import *
from utils import progress_bar, get_time_str, visualize_image, save_temp_results, combine_channels, save_pred_results
import argparse


# parser = argparse.ArgumentParser(description='PyTorch Image Color Inference')
# parser.add_argument('--batch_size', '-b', default=32, type=int, help='batch size')
# parser.add_argument('--ckp_last', '-cp', default='/home/grads/b/bhanu/img_color/runs/adam_vgg_unet/06-02-2022-14:32/models/ckpt_best.pth', type=str, help='checkpoint path')
# parser.add_argument('--data_path', '-dp', default='./preds/inputs', type=str, help='prediction images folder')
# parser.add_argument('--lab_version', '-lv',type=int, default=1, metavar='N',
#                         help='version of lab scaling (default: 1)')

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = 'cpu'

net = UNet(n_channels=1, n_classes=2, bilinear=True)

net = net.to(device)

#Load model path here from any of models

checkpoint = torch.load(f'./runs/adam_vgg_unet/06-02-2022-14:32/models/ckpt_best.pth',map_location=torch.device(device))
net.load_state_dict(checkpoint['net'])
transform_pred = transforms.Compose([
    transforms.Resize(size=(256,256)),
])


#LOAD Prediction gray images folder here (Make sure you add some subfolder all like preds/input folder so that torchvision dataloder knows the format)
# Else directly paste your images here /preds/inputs/all/

prednset = ColorizeData(root = './preds/inputs',lab_version = 1, transform= transform_pred)
predloader = torch.utils.data.DataLoader(prednset, batch_size=32, shuffle=True, num_workers=32)

if __name__=='__main__':
    net.eval()
    with torch.no_grad():
        for batch_idx, (input_gray, input_ab, target) in enumerate(predloader):
            input_gray, input_ab = input_gray.to(device), input_ab.to(device)
            outputs = net(input_gray)
            progress_bar(batch_idx, len(predloader))
            # for j in range(len(outputs)):
            #     if j % 10 == 0 :
            #         gray_output, color_output = combine_channels(input_gray[j], outputs[j].data.detach(), args.lab_version)
                    # writer.add_images('Outputs', np.stack((gray_output,color_output),axis=0), epoch)
                    # writer.add_images('color-output',np.expand_dims(color_output,0),epoch)
                    # writer.add_images('gray-input',np.expand_dims(gray_output,0),epoch)

            if not os.path.isdir(f'./preds/outputs/'):
                os.makedirs(f'./preds/outputs/')
            for j in range(len(outputs)):
                save_path = {'grayscale': f'', 'colorized': f'./preds/outputs/'}
                save_name = 'img-{}.jpg'.format(batch_idx * predloader.batch_size + j)
                save_pred_results(input_gray[j], ab_input=outputs[j].data.detach(),lab_version=1, save_path=save_path, save_name=save_name)
