import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from configs.paths import TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR, RAW_DATA_DIR
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_dataset(raw_dir: Path = RAW_DATA_DIR, test_size=0.1, val_size=0.1):
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    for _class in classes:
        CLASS_DIR = raw_dir / _class
        images = list(CLASS_DIR.glob('*.jpg')) + list(CLASS_DIR.glob('*.png'))

        train_val, test = train_test_split(images, test_size=test_size, random_state=42)
        train, validation = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=42)

        for split, imgs, split_dir in zip(['train', 'validation', 'test'], [train, validation, test], [TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR]):
            split_class_dir = split_dir / _class
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for img in imgs:
                shutil.copy(img, split_class_dir / img.name)

def get_dataloaders(batch_size=32, image_size=224):
    transfrom_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR, transform=transfrom_train)
    validation_dataset = datasets.ImageFolder(VAL_DATA_DIR, transform=transform_val)
    test_dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, prefetch_factor=4, pin_memory=True, persistent_workers=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, validation_loader, test_loader 