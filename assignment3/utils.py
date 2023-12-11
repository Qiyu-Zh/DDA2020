import argparse
import numpy as np
import hiddenlayer as hl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch import ClassType, Tensor
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from types import FunctionType
from typing import Dict, List, Tuple


def eval_metric(y_true: Tensor, y_pred_prob: Tensor, num_class: int = 3, methods: List[str] = ['acc', 'auc', 'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'f1'], multi_class: bool = True, epsilon: float = 1e-7) -> Dict[str, Tensor]:
    r"""
    Return a dict of accuracy evaluation metircs. Input data should be on cpu. Only valid evaluation method results will be returned.
    """

    if y_true.ndim != 1:
        raise ValueError('Input y_true should be in dimension 1')
    if y_pred_prob.ndim != 2:
        raise ValueError('Input y_pred_prob should be in batch and probability form')
    
    possible_eval_metirc = set(['acc', 'auc', 'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'f1'])
    metrics = possible_eval_metirc.intersection(set(methods))

    results = {}
    y_pred_label = torch.argmax(y_pred_prob, 1)

    if 'acc' in metrics:
        results['acc'] = torch.sum(y_pred_label == y_true) / y_true.size(0)

    if 'auc' in metrics:
        results['auc'] = roc_auc_score(y_true, y_pred_prob, multi_class='ovo' if multi_class else 'raise')

    if 'f1' in metrics:
        one_hot_y = F.one_hot(y_true, num_class if multi_class else 2)

        tp = (one_hot_y * y_pred_prob).sum(dim=0)
        fp = ((1 - one_hot_y) * y_pred_prob).sum(dim=0)
        fn = (one_hot_y * (1 - y_pred_prob)).sum(dim=0)

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * precision * recall / (precision + recall + epsilon)
        f1 = f1.clamp(min=epsilon, max=1-epsilon)

        results['f1'] = 1 - f1.mean()

        if 'tn' in metrics:
            results['tn'] = ((1 - one_hot_y) * (1 - y_pred_prob)).sum(dim=0).mean()
        for e_method in ['tp', 'fp', 'fn', 'precision', 'recall']:
            if e_method in metrics:
                results[e_method] = eval(e_method).mean()
        
    else:
        if set(['tp', 'fp', 'fn', 'precision', 'recall']).intersection(methods) != set():
            one_hot_y = F.one_hot(y_true, num_class if multi_class else 2)

            tp = (one_hot_y * y_pred_prob).sum(dim=0)
            fp = ((1 - one_hot_y) * y_pred_prob).sum(dim=0)
            fn = (one_hot_y * (1 - y_pred_prob)).sum(dim=0)

            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)

            for e_method in ['tp', 'fp', 'fn', 'precision', 'recall']:
                if e_method in metrics:
                    results[e_method] = eval(e_method).mean()
        
    return results


def train_model(
    model: models.ResNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_func: nn.CrossEntropyLoss,
    eval_metric: FunctionType,
    arg: argparse.ArgumentParser
    ) -> Dict[str, Tensor]:

    model = model.to(arg.device)
    criterion = loss_func()
    optimizer = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arg.epochs // 10, eta_min=0, last_epoch=-1)

    train_states = []
    test_states = []

    for epoch in range(arg.epochs):
        print(epoch + 1, arg.epochs)
        model.train()
        ys, outputs = [], []
        for (b_x, b_y) in tqdm(train_loader):
            ys.append(b_y)
            b_x, b_y = b_x.to(arg.device), b_y.to(arg.device)

            output = model(b_x)
            loss = criterion(output, b_y)
            outputs.append(nn.Softmax(dim=1)(output).detach().cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_states.append(eval_metric(torch.stack(ys[:-1]).view(-1, ), torch.stack(outputs[:-1]).view(-1, arg.num_class), arg.num_class, ['acc', 'auc', 'f1']))

        print(train_states[-1])

    return
