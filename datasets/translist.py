import torchvision.transforms as T

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform_rand = T.Compose(
    [
        T.Resize((256, 256)),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ]
)
train_transform_center = T.Compose(
    [
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ]
)
val_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ]
)