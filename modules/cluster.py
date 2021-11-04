from utils.logger import CompleteLogger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist


def k_means(loader: DataLoader, netG: nn.Module, netF: nn.Module, DEVICE: torch.device, type_dist: str = 'cosine',  threshold:int = 0, epsilon: float = 1e-5):
    with torch.no_grad():
        for idx, (X, label) in enumerate(loader):
            X = X.to(DEVICE)
            feat = netG(X)
            output = netF(feat)
            try:
                all_feat = torch.cat((all_feat, feat.float().cpu()), 0)
                all_output = torch.cat((all_output, output.float().cpu()), 0)
                all_label = torch.cat((all_label, label.float().cpu()), 0)
            except:
                all_feat = feat.float().cpu()
                all_output = output.float().cpu()
                all_label = label.float().cpu()

    all_output = F.softmax(all_output, dim=1)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() ==
                         all_label).item() / float(all_label.size()[0])
    if type_dist == 'cosine':
        all_feat = torch.cat((all_feat, torch.ones(all_feat.size(0), 1)), 1)
        all_feat = (all_feat.t() / torch.norm(all_feat, p=2, dim=1)).t()

    all_feat = all_feat.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_feat)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_feat, initc[labelset], type_dist)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_feat)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_feat, initc[labelset], type_dist)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_feat)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str)
    return pred_label.astype('int')
