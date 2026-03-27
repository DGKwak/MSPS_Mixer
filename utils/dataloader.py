import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def make_datasets(train_dir,
                  val_dir,
                  test_dir,
                  img_size):
    # 데이터셋 로드
    data_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transform)

    return train_dataset, val_dataset, test_dataset

def make_dataloaders(train_dataset,
                     val_dataset,
                     test_dataset,
                     num_workers,
                     batch_size,
                     random_state):

    generator = torch.Generator()
    generator.manual_seed(random_state)

    # DataLoader
    train_loader = DataLoader(train_dataset,
                              num_workers=num_workers,
                              batch_size=batch_size,
                              shuffle=True,
                              generator=generator)
    
    val_loader = DataLoader(val_dataset,
                            num_workers=num_workers,
                            batch_size=batch_size,
                            shuffle=False)
    
    test_loader = DataLoader(test_dataset,
                             num_workers=num_workers,
                             batch_size=batch_size,
                             shuffle=False)
    
    return train_loader, val_loader, test_loader