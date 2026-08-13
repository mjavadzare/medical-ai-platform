from torchvision import transforms


def get_train_transforms(
    mean: list[float],
    std: list[float]
):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(15),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.1
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])


def get_val_transforms(
    mean: list[float],
    std: list[float]
):
    return transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])