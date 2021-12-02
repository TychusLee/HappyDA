import os
import os.path as osp
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from .basedataset import BaseDataset
from .indexdataset import IndexDataset
from . import datasetconfig
from .translist import *
import torch

DatasetDic = {"base": BaseDataset, "indexed": IndexDataset}


# def office31(root: str, domain: str, type: str = "base", **kwargs) -> VisionDataset:
#     if domain not in OFFICE31["file_list"]:
#         raise KeyError
#     filepath = os.path.join(root, OFFICE31["file_list"][domain])
#     return DatasetDic[type](root, filepath, OFFICE31["classes"], **kwargs)


# def officehome(root: str, domain: str, type: str = "base", **kwargs) -> VisionDataset:
#     if domain not in OFFICEHOME["file_list"]:
#         raise KeyError
#     filepath = os.path.join(root, OFFICEHOME["file_list"][domain])
#     return DatasetDic[type](root, filepath, OFFICEHOME["classes"], **kwargs)


def getDataset(name: str, root: str, domain: str, type: str = "base", **kwargs) -> VisionDataset:
    config = datasetconfig.__dict__[name]
    if domain not in config["file_list"]:
        raise KeyError
    filepath = osp.join(root, config["file_list"][domain])
    return DatasetDic[type](root, filepath, config["classes"], **kwargs)


def splitDataset(name: str, root: str, domain: str, type: str = "base", ratio: float = 0.9, ordered: bool=False, **kwargs):
    config = datasetconfig.__dict__[name]
    domain = domain.upper()
    if domain not in config["file_list"]:
        raise KeyError
    savepath = osp.join("splitdata", name, domain)
    if not osp.exists(savepath):
        os.system('mkdir -p '+savepath)
    filepath = osp.join(root, config["file_list"][domain])
    train_file = osp.join(savepath, "train.txt")
    val_file = osp.join(savepath, "val.txt")
    with open(filepath, "r") as f:
        all_txt = f.readlines()
        train_size = int(len(all_txt)*ratio)
        train_txt, val_txt = torch.utils.data.random_split(all_txt, [train_size, len(all_txt)-train_size])
        if ordered:
            train_txt.indices.sort()
            val_txt.indices.sort()
        with open(train_file, "w") as fo:
            fo.writelines(train_txt)
        with open(val_file, "w") as fo:
            fo.writelines(val_txt)
    return DatasetDic[type](root, train_file, config["classes"], transform=train_transform_rand, **kwargs), DatasetDic[type](root, val_file, config["classes"], transform=val_transform, **kwargs)
        
