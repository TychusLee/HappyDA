import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
import matplotlib
from utils import Entropy

matplotlib.use('Agg')


def vispdf(source_output: torch.Tensor, target_output: torch.Tensor,
              filename: str, source_color='r', target_color='b', metric='entropy', bin=100):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_output (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_output (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    prob = torch.softmax(torch.cat([source_output, target_output]), dim=1)
    if metric == 'entropy':
        criterion = Entropy(prob)
    if metric == 'probability':
        criterion, _ = torch.max(prob, dim=1)
    if metric == 'probdiff':
        top, _ = torch.topk(prob, k=2, dim=1)
        criterion = top[:, 0]-top[:, 1]
    criterion = criterion.view([-1]).numpy()
    src_criterion = criterion[:source_output.size(0)]
    tgt_criterion = criterion[source_output.size(0):]
    # map features to 2-d using TSNE
    src_p, src_x = np.histogram(src_criterion, bins=bin)
    src_x = src_x[:-1] + (src_x[1] - src_x[0])/2
    tgt_p, tgt_x = np.histogram(tgt_criterion, bins=bin)
    tgt_x = tgt_x[:-1] + (tgt_x[1] - tgt_x[0])/2

    # domain labels, 1 represents source while 0 represents target
    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.plot(src_x, src_p, source_color, label='source')
    plt.plot(tgt_x, tgt_p, target_color, label='target')
    plt.xlabel(metric)
    plt.ylabel('number')
    plt.title(metric+' pdf')
    plt.savefig(filename)
