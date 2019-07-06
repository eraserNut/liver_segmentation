import numpy as np


class Evaluator_dice(object):
    def __init__(self):
        self.diceAll = 0.
        self.num = 0

    def add_batch(self, pred, target):
        intersection = (pred * target).sum()
        current_dice = (2. * intersection) / (pred.sum() + target.sum())
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
        tp = (pred * target).sum()
        current_precision = tp / pred.sum() if pred.sum() != 0 else 0
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
        tp = (pred * target).sum()
        current_recall = tp / target.sum()
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
        self.f1All += 2 * (precision * recall) / (precision + recall) if (precision * recall) / (precision + recall) != 0 else 0
        self.num += 1
    def get_F1(self):
        assert (self.num != 0)
        return self.f1All/self.num
