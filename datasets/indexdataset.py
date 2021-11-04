from . import BaseDataset
from typing import Optional, Callable, Tuple, Any, List
from torchvision.datasets.folder import default_loader


class IndexDataset(BaseDataset):
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
            root_dir, data_list_path, classes, transform, target_transform, loader
        )

    def __getitem__(self, index: int) -> Tuple[Tuple[Any, int], int]:
        # pos, target = self.samples[index]
        # img = self.loader(pos)
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        img, target = super().__getitem__(index)
        return (img, target), index
