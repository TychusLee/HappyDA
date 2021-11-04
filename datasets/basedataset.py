from torch.utils import data
from torchvision.datasets import VisionDataset
from typing import Optional, Callable, Tuple, Any, List
from torchvision.datasets.folder import default_loader
import os


class BaseDataset(VisionDataset):
    def __init__(
        self,
        root_dir: str,
        data_list_path: str,
        classes: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ):
        super().__init__(
            root_dir, transform=transform, target_transform=target_transform
        )
        self.classes = classes
        self.class_to_index = {cat: idx for idx,
                               cat in enumerate(self.classes)}
        self.samples = self.parse_file(data_list_path)
        self.data_list_path = data_list_path
        self.loader = loader
        self.root_dir = root_dir

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        pos, target = self.samples[index]
        img = self.loader(pos)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def parse_file(self, file_path) -> List[Tuple[str, int]]:
        data_list = []
        with open(file_path, "r") as f:
            for line in f.readlines():
                pos, target = line.split()
                if not os.path.isabs(pos):
                    pos = os.path.join(self.root, pos)
                target = int(target)
                data_list.append((pos, target))
        return data_list

    @property
    def num_classes(self) -> int:
        return len(self.classes)
