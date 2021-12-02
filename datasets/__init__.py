from .basedataset import BaseDataset
from .indexdataset import IndexDataset
from .getdataset import getDataset, splitDataset
from .iterator import ForeverDataIterator
from . import translist
from . import datasetconfig

DataInfo = {}
for name in datasetconfig.__all__:
    DataInfo[name] = datasetconfig.__dict__[name]
    
__all__ = ["BaseDataset", "IndexDataset", "getDataset", "splitDataset", "ForeverDataIterator", "translist", "DataInfo"]
