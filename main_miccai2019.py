import datetime
import os

import scipy.io as sio
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from config_path import training_root, testing_root
from dataset_miccai2019 import ImageFolder_miccai2019
from evaluator_miccai2019 import *
from misc import AvgMeter, check_mkdir, progress_bar, crf_refine
from models import *
from loss import *
import argparse
import yaml
from PIL import Image
import numpy as np
import math
import SimpleITK as sitk
from torchsummary import summary

torch.cuda.set_device(0)

# Load parameters
ckpt_path = './ckpt_miccai2019'
# ckpt_path = './ckpt_tumor'
exp_name = 'Resnet18_UNET' # specific model name
# exp_name = 'FPN' # specific model name
args_config = os.path.join('./models', exp_name, 'config.yaml')
args = yaml.load(open(args_config))

parser = argparse.ArgumentParser()
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--save_pred', '-s', action='store_true', help='save prediction sample')
parser.add_argument('--test', '-t', action='store_true', help='test model')
flag = parser.parse_args()
best_eval = 0 # best test evaluation
start_epoch = 0 # start from epoch 0 or last checkpoint epoch
# Data
# if args['resize']:
#     transform = transforms.Compose([
#         transforms.Resize((args['resizeTo'],args['resizeTo'])),
#         transforms.ToTensor(),
#         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         transforms.Normalize([0.141, 0.141, 0.141], [0.140, 0.140, 0.141])
#     ])
#     target_transform = transforms.Compose([
#         transforms.Resize((args['resizeTo'],args['resizeTo'])),
#         transforms.ToTensor(),
#     ])
# else:
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         transforms.Normalize([0.141, 0.141, 0.141], [0.140, 0.140, 0.141])
#     ])
#     target_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

to_pil = transforms.ToPILImage()

train_set = ImageFolder_miccai2019(training_root, None, None, None) # no transform
train_loader = DataLoader(train_set, batch_size=1, num_workers=12, shuffle=True)


# Model
print('==> Building model..')
net = ResNetUNet(n_class=1).cuda()
# net = UNETplusplus(n_class=1).cuda()
# net = DAF().cuda()
# net = FPN().cuda()
# net = FPN_DS().cuda()
# net = FPN_DS2().cuda()
# net = FPN_DS_V3().cuda()
# net = FPN_DS_V4().cuda()
# net = FPN_multi_task().cuda()
# net = FPN_multi_task_V2().cuda()
# summary(net, (3, 448, 448)) #summary for test model except for fpn

if flag.resume or flag.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint_path = os.path.join(ckpt_path, exp_name)
    print(checkpoint_path)
    # assert os.path.isdir('checkpoint_path'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(checkpoint_path, 'model.pth'))
    net.load_state_dict(checkpoint['net'])
    best_eval = checkpoint['eval']
    eval_type = checkpoint['eval_type']

optimizer = optim.SGD([
    {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
     'lr': 2 * args['lr']},
    {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
     'lr': args['lr'], 'weight_decay': args['weight_decay']}
], momentum=args['momentum'])

check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss_record, bce_loss_record, dice_loss_record = AvgMeter(), AvgMeter(), AvgMeter()
    for batch_idx, data in enumerate(train_loader):
        if epoch == args['lr_step']:
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] / args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] / args['lr_decay']
        inputs_volume, labels_volume = data
        inputs_volume = norm_filter(inputs_volume) #normlization via means and std
        sub_batch_len = math.ceil(inputs_volume.shape[1]/args['train_batch_size'])
        print_flag=0
        for sub_batch_idx in range(sub_batch_len):
            #split volume to batchs
            start = sub_batch_idx*args['train_batch_size']
            end = (sub_batch_idx+1)*args['train_batch_size'] if (sub_batch_idx+1)*args['train_batch_size']<inputs_volume.shape[1] else inputs_volume.shape[1]
            inputs = inputs_volume[:, start:end, :, :].permute(1, 0, 2, 3)
            inputs = inputs.expand(torch.Size((inputs.shape[0], 3, inputs.shape[2], inputs.shape[3])))
            labels = labels_volume[:, start:end, :, :].permute(1, 0, 2, 3)
            if torch.max(labels).item() == 0:
                print_flag += 1
                print(print_flag)
                continue
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            # outputs = net(inputs)

            # BCE loss and dice loss can be used
            criterion_bce = nn.BCELoss()
            criterion_dice = Dice_loss()
            # if not isinstance(fnd_out, list):
            loss_bce = criterion_bce(outputs, labels)
            loss_dice = criterion_dice(outputs, labels)
            # loss_bce = criterion_bce(outputs, labels)
            # loss_dice = criterion_dice(outputs, labels)
            # else:
            #     loss_bce = criterion_bce(outputs, labels)
            #     loss_dice = criterion_dice(outputs, labels)
            #     for fnd_mask, fpd_mask in zip(fnd_out, fpd_out):
            #         loss_bce += criterion_bce(fnd_mask, fnd) + criterion_bce(fpd_mask, fpd)


            # else:
            #     loss_bce_each = [None] * len(outputs)
            #     loss_dice_each = [None] * len(outputs)
            #     for idx in range(len(outputs)):
            #         loss_bce_each[idx] = criterion_bce(outputs[idx], labels)
            #         loss_dice_each[idx] = criterion_dice(outputs[idx], labels)
            #     loss_bce = sum(loss_bce_each)
            #     loss_dice = sum(loss_dice_each)
            # coeff = loss_dice.item()/(loss_bce.item()+1e-5) if loss_dice.item()/(loss_bce.item()+1e-5) < 1 else 1
            coeff = 1
            loss = coeff*loss_bce + loss_dice
            # loss = loss_bce + loss_dice
            loss.backward()
            optimizer.step()
            train_loss_record.update(loss.item(), batch_size)
            bce_loss_record.update(loss_bce.item(), batch_size)
            dice_loss_record.update(loss_dice.item(), batch_size)
            log = 'iter: %d | [bce loss: %.5f], [dice loss: %.5f],[Total loss: %.5f], [lr: %.8f]' % \
                  (epoch, bce_loss_record.avg, dice_loss_record.avg, train_loss_record.avg, optimizer.param_groups[1]['lr'])
            progress_bar(batch_idx, len(train_loader), log)


# Testing
def test(epoch):
    global best_eval
    net.eval()
    # evaluator = Evaluator_Miou(2)
    evaluator_dice = Evaluator_dice()
    evaluator_F1 = Evaluator_F1()
    with torch.no_grad():
        for idx, sample_name in enumerate(os.listdir(os.path.join(testing_root))):
            volume_path = os.path.join(testing_root, sample_name, 'data.nii.gz')
            gt_path = os.path.join(testing_root, sample_name, 'label.nii.gz')
            itk_CT = sitk.ReadImage(volume_path)
            itk_gt = sitk.ReadImage(gt_path)
            torch_CT = itk_transfor(itk_CT)
            torch_gt = itk_transfor(itk_gt)
            torch_CT = norm_filter(torch_CT)
            sub_batch_len = torch_CT.shape[0]
            gt_volume = np.array(torch_gt, dtype=np.uint8)
            for sub_batch_idx in range(sub_batch_len):
                img = torch_CT[sub_batch_idx, :, :]
                img = img.expand(torch.Size((1, 3, *img.shape)))
                img_var = Variable(img).cuda()
                outputs = net(img_var)
                prediction = np.array(outputs.data.squeeze().cpu())
                # prediction = crf_refine(np.array(img.permute(0,2,3,1)).squeeze(0).astype(np.uint8), prediction.astype(np.uint8))

                # if flag.save_pred:
                #     check_mkdir(os.path.join(ckpt_path, exp_name, 'prediction'))
                #     Image.fromarray(prediction).save(
                #         os.path.join(ckpt_path, exp_name, 'prediction', img_name))
                pred = Image.fromarray(prediction).convert('1')
                pred = np.array(pred, dtype=np.uint8).reshape((1, *pred.size))
                if sub_batch_idx == 0:
                    pred_volume = pred
                else:
                    pred_volume = np.concatenate((pred_volume, pred), axis=0)

            evaluator_dice.add_batch(pred_volume, gt_volume)
            evaluator_F1.add_batch(pred_volume, gt_volume)

            current_dice = evaluator_dice.get_dice()
            progress_bar(idx, len(os.listdir(os.path.join(testing_root))), 'Dice: %.4f' % (current_dice))
        # Miou = evaluator.Frequency_Weighted_Intersection_over_Union()
        dice = evaluator_dice.get_dice()
        F1 = evaluator_F1.get_F1()
        print('Mean dice is %.4f | Mean F1 is %.4f'%(dice, F1))

        # Save checkpoint.
        if dice > best_eval and not flag.test:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'eval': dice,
                'epoch': epoch,
                'eval_type': 'dice'
            }
            checkpoint_path = os.path.join(ckpt_path, exp_name)
            if not os.path.isdir(checkpoint_path):
                os.mkdir(checkpoint_path)
            torch.save(state, os.path.join(checkpoint_path, 'model.pth'))
            best_eval = dice


def itk_transfor(itk):
    img_arr = sitk.GetArrayFromImage(itk).astype(np.float32)
    torch_itk = torch.from_numpy(img_arr)
    return torch_itk

def norm_filter(inputs):
    #normal to 0-1
    inputs = (inputs-torch.min(inputs))/(torch.max(inputs)-torch.min(inputs))
    #normal via mean and std
    outputs = (inputs-0.0381)/0.0878
    return outputs

for epoch in range(start_epoch, start_epoch+args['iter_num']):
    if not flag.test:
        train(epoch)
        test(epoch)
    else:
        test(epoch)
        break


