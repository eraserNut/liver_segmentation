from torch import nn
import torch


class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()

    def forward(self, pred, target, smooth=1.):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + smooth) / (
                    pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        return loss.mean()


class DS_loss(nn.Module):
    def __init__(self):
        super(DS_loss, self).__init__()

    def forward(self, pred, target, target_fn, target_fp):
        pred = pred.contiguous()
        target = target.contiguous()
        target_fn = target_fn.contiguous()
        target_fp = target_fp.contiguous()
        weight_fn = target_fn.sum(dim=2).sum(dim=2) / (target_fn.sum(dim=2).sum(dim=2) + target_fp.sum(dim=2).sum(dim=2))
        batch_size = weight_fn.size()[0]
        maps_fn = []
        for i in range(batch_size):
            map_fn = torch.ones((target_fn.size()[2], target_fn.size()[3])).cuda() * weight_fn.data[i]
            maps_fn.append(map_fn.unsqueeze(0))
        maps_fn = torch.cat(maps_fn, dim=0)
        maps_fn = maps_fn.unsqueeze(1)
        l1_loss = -maps_fn*target*torch.log(pred+1e-8) - (1-maps_fn)*(1-target)*torch.log(1-pred+1e-8)
        l2_loss = -maps_fn*target_fn*torch.log(pred+1e-8) - (1-maps_fn)*target_fp*torch.log(1-pred+1e-8)
        return l1_loss.mean()+l2_loss.mean()
