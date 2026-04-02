import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.MSPS_Mixer_rev02 import MultiscaleMixer

from utils.dataloader import make_datasets, make_dataloaders
from utils.earlystopping import EarlyStopping
from utils.helper import create_logger, set_seed

from trainer import Trainer

from loss.loss_func import FocalLoss

@hydra.main(config_path='./config', config_name='main', version_base=None)
def main(cfg):
    metadata = {
        'Experiment Name': cfg.experiment_name,
        'Dataset': cfg.data.dataset_name,
        'Input_size': cfg.data.input_size,
        'Random_state': cfg.random_state,
        'Epochs': cfg.epochs,
        'Batch_size': cfg.batch_size,
        'Learning_rate': cfg.learning_rate,
        'Patch Size': cfg.model.patches,
        'Patch Dim': cfg.model.patch_dim,
        'Dropout': cfg.model.dropout,
        'Layers': cfg.model.num_layers,
        'Activation': cfg.model.activation,
    }

    set_seed(cfg.random_state)
    log_out = HydraConfig.get().runtime.output_dir
    logger = create_logger(log_out, dist_rank=0, name=cfg.experiment_name)

    logger.info(metadata)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset = make_datasets(cfg.data.train_dir,
                                                             cfg.data.val_dir,
                                                             cfg.data.test_dir,
                                                             cfg.data.input_size)
    
    # 데이터셋 정보 출력
    logger.info(f"총 데이터셋 크기: {len(train_dataset)}")
    logger.info(f"검증 데이터셋 크기: {len(val_dataset)}")
    logger.info(f"클래스 수: {len(train_dataset.classes)}")
    logger.info(f"클래스 목록: {train_dataset.classes}")
    
    train_loader, val_loader, test_loader = make_dataloaders(train_dataset=train_dataset,
                                                             val_dataset=val_dataset,
                                                             test_dataset=test_dataset,
                                                             num_workers=cfg.num_workers,
                                                             batch_size=cfg.batch_size,
                                                             random_state=cfg.data.random_state)
    
    # Model
    model = MultiscaleMixer(
        in_channels=cfg.model.in_channels,
        patch_dim=cfg.model.patch_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        patches=cfg.model.patches,
        stride=cfg.model.stride,
        shift_size=cfg.model.shift_size,
        shift=cfg.model.shift,
        num_patches=cfg.model.num_patches,
        act=cfg.model.activation,
    ).to(device)

    # loss Function
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
    focal = FocalLoss(alpha=torch.tensor([1.75, 1.0, 1.75, 1.0, 1.0, 1.0]), 
                      task_type='multi-class', 
                      num_classes=6).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    
    early_stopping = EarlyStopping(patience=30, mode='min', verbose=True, logger=logger)

    config = {'epochs': cfg.epochs, 'experiment_name': cfg.experiment_name,
              'classes': train_dataset.classes, 'metadata': metadata}
    components = {'optimizer': optimizer, 'scheduler': scheduler, 'device': device}
    loss_config = {'cross_entropy': cross_entropy, 'focal': focal, 'lambda_aux': cfg.lambda_aux}
    paths = {'csv': cfg.csv_path, 'best_model': cfg.best_model_path, 'cofusion': cfg.confusion_path}

    # Create trainer and start training
    trainer = Trainer(config=config, model=model, components=components, loss_config=loss_config, paths=paths, logger=logger)
    trainer.train(train_loader, val_loader, early_stopping)

    # Test the best model
    trainer.test(test_loader)

if __name__ == '__main__':
    main()
