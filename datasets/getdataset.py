import os
from torchvision.datasets.vision import VisionDataset
from .basedataset import BaseDataset
from .indexdataset import IndexDataset
from . import datasetconfig

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
    filepath = os.path.join(root, config["file_list"][domain])
    return DatasetDic[type](root, filepath, config["classes"], **kwargs)
