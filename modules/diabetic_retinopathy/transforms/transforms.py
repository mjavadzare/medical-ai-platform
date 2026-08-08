from torchvision import transforms


def get_train_transforms(
    mean: list[float],
    std: list[float]
):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_val_transforms(
    mean: list[float],
    std: list[float]
):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])