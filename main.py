import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import uuid
import time
import os
import argparse
from colorize_data import  ColorizeData
from utils import progress_bar, get_time_str, visualize_image, save_temp_results, combine_channels
from torch.utils.tensorboard import SummaryWriter
from models import *
import torchgeometry as tgm
import numpy as np
import vgg_loss

parser = argparse.ArgumentParser(description='PyTorch Image Color Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--run_name','-rn',type=str, help='your experiment name',default=f'{get_time_str()}_default')
parser.add_argument('--batch_size', '-b', default=32, type=int, help='batch size')
parser.add_argument('--ckp_last', '-cl', default=True, type=bool, help='resume with last checkpoint if false resume with best checkpoint')
parser.add_argument('--num_epochs', '-ne', default=200, type=int, help='number of epochs')
parser.add_argument('--lab_version', '-lv',type=int, default=1, metavar='N',
                        help='version of lab scaling (default: 2)')
timestamp = get_time_str()
print(timestamp)
args = parser.parse_args()
if args.resume:
    folder_name = args.run_name
else:
    folder_name = f'runs/{args.run_name}/{timestamp}'
writer = SummaryWriter(folder_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 10000000000  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')



transform_test = transforms.Compose([
    transforms.Resize(size=(256,256)),
])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainset = ColorizeData(root = '/home/grads/b/bhanu/img_color/data/train/',lab_version = args.lab_version, transform= transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testset = ColorizeData(root ='/home/grads/b/bhanu/img_color/data/val/', lab_version = args.lab_version, transform= transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=32)


# Model
print('==> Building model..')
# net = Net()
net = UNet(n_channels=1, n_classes=2, bilinear=True)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)

net = net.to(device)



if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.run_name), 'Error: no checkpoint directory found!'
    if args.ckp_last:
        checkpoint = torch.load(f'./{args.run_name}/models/ckpt_last.pth')
    else:
        checkpoint = torch.load(f'./{args.run_name}/models/ckpt_best.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']



# criterion = nn.MSELoss()
# criterion = tgm.losses.SSIM(11)
criterion = vgg_loss.WeightedLoss([vgg_loss.VGGLoss(shift=2),
                                  nn.MSELoss(),
                                  vgg_loss.TVLoss(p=1)],
                                 [1, 40, 10]).to(device)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters())
# optimizer =  optim.RMSprop(net.parameters(), lr=args.lr,weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    criterion.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (input_gray, input_ab, org_imgs) in enumerate(trainloader):
        input_gray, input_ab = input_gray.to(device), input_ab.to(device)
        # print(inputs.shape,targets.shape)
        optimizer.zero_grad()
        outputs = net(input_gray)
        # loss = criterion(outputs, targets)
        # c_outs = []
        # print(outputs[0].data.shape)
        # for j in range(len(outputs)):
        #     gray_output, color_output = combine_channels(input_gray[j], outputs[j].data.detach(), args.lab_version)
        #     c_outs.append(color_output)
        # c_outs = np.asarray(c_outs)
        # c_outs = torch.from_numpy(c_outs*255).float().to(device)
        # c_outs.requres_grad = True
        # org_imgs = org_imgs.to(device)
        # org_imgs.requres_grad = True

        # print(c_outs[0])
        # print(org_imgs[0])
        loss = criterion(outputs, input_ab)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # predicted = outputs.data.max(1, keepdim=True)[1]
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        # correct += predicted.eq(targets.data.max(1, keepdim=True)[1]).sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.10f'
                     % (train_loss/(batch_idx+1)))
    writer.add_scalar("Loss/train", train_loss/(batch_idx+1), epoch)
    for name, weight in net.named_parameters():
        writer.add_histogram(name,weight, epoch)
        writer.add_histogram(f'{name}.grad',weight.grad, epoch)


    if (epoch+1)%10 == 0:
        print('Saving model..')
        state = {
            'net': net.state_dict(),
            'loss': best_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'./{folder_name}/models'):
            os.makedirs(f'./{folder_name}/models')
        torch.save(state, f'./{folder_name}/models/ckpt_last.pth')
    return train_loss


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (input_gray, input_ab, target) in enumerate(testloader):
            input_gray, input_ab = input_gray.to(device), input_ab.to(device)
            outputs = net(input_gray)
            loss = criterion(outputs, input_ab)

            test_loss += loss.item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.10f'
                         % (test_loss/(batch_idx+1)))
            # for j in range(len(outputs)):
            #     if j % 10 == 0 :
            #         gray_output, color_output = combine_channels(input_gray[j], outputs[j].data.detach(), args.lab_version)
                    # writer.add_images('Outputs', np.stack((gray_output,color_output),axis=0), epoch)
                    # writer.add_images('color-output',np.expand_dims(color_output,0),epoch)
                    # writer.add_images('gray-input',np.expand_dims(gray_output,0),epoch)

            if True:
                if not os.path.isdir(f'./{folder_name}/outputs/gray/'):
                    os.makedirs(f'./{folder_name}/outputs/gray/')
                if not os.path.isdir(f'./{folder_name}/outputs/color/'):
                    os.makedirs(f'./{folder_name}/outputs/color/')
                for j in range(len(outputs)):
                    if j % 10 == 0 :
                        save_path = {'grayscale': f'./{folder_name}/outputs/gray/', 'colorized': f'./{folder_name}/outputs/color/'}
                        save_name = 'img-{}-epoch-{}.jpg'.format(batch_idx * testloader.batch_size + j, epoch)
                        save_temp_results(input_gray[j], ab_input=outputs[j].data.detach(),lab_version=args.lab_version, save_path=save_path, save_name=save_name)
    # Save checkpoint.
    
    writer.add_scalar("Loss/test", test_loss/(batch_idx+1), epoch)
    if test_loss < best_acc:
        print('Saving best model..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(f'./{folder_name}/models'):
            os.makedirs(f'./{folder_name}/models')
        torch.save(state, f'./{folder_name}/models/ckpt_best.pth')
        best_acc = test_loss

if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        train_loss = train(epoch)
        test(epoch)
        scheduler.step(train_loss)
    writer.add_hparams(
        {"lr": args.lr, "bsize": args.batch_size},
        {
            "loss": best_acc
        },
    )