import numpy as np


class Evaluator_Miou(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        tmp = gt_image[mask]
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

class Evaluator_dice(object):
    def __init__(self):
        self.diceAll = 0.
        self.num = 0

    def add_batch(self, pred, target):
        intersection = (pred * target).sum(axis=0).sum(axis=0)
        current_dice = (2. * intersection) / (
                    pred.sum(axis=0).sum(axis=0) + target.sum(axis=0).sum(axis=0))
        self.diceAll += current_dice
        self.num += 1
        return current_dice
    def get_dice(self):
        assert(self.num != 0)
        return self.diceAll/self.num

class Evaluator_Precision(object):
    def __init__(self):
        self.precisionAll = 0.
        self.num = 0
    def add_batch(self, pred, target):
        tp = (pred * target).sum(axis=0).sum(axis=0)
        current_precision = tp / pred.sum(axis=0).sum(axis=0)
        self.precisionAll += current_precision
        self.num += 1
    def get_Precision(self):
        assert(self.num != 0)
        return self.precisionAll/self.num

class Evaluator_Recall(object):
    def __init__(self):
        self.recallAll = 0.
        self.num =0
    def add_batch(self, pred, target):
        tp = (pred * target).sum(axis=0).sum(axis=0)
        current_recall = tp / target.sum(axis=0).sum(axis=0)
        self.recallAll += current_recall
        self.num += 1
    def get_recall(self):
        assert(self.num != 0)
        return self.recallAll/self.num

class Evaluator_F1(object):
    def __init__(self):
        self.f1All = 0
        self.num = 0
        self.eval_pre = Evaluator_Precision()
        self.eval_recall = Evaluator_Recall()
    def add_batch(self,pred,target):
        self.eval_pre.add_batch(pred, target)
        self.eval_recall.add_batch(pred, target)
        precision = self.eval_pre.get_Precision()
        recall = self.eval_recall.get_recall()
        self.f1All += 2 * (precision * recall) / (precision + recall)
        self.num += 1
    def get_F1(self):
        assert (self.num != 0)
        return self.f1All/self.num
