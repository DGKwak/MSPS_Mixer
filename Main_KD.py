import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR

from sklearn.metrics import confusion_matrix, classification_report

import model.MSPS_Mixer_KD_teacher01 as MSPS_tea
import model.MSPS_Mixer_KD_student01 as MSPS_stu

from loss.loss_func import FocalLoss

from utils.earlystopping import EarlyStopping
from utils.dataloader import make_datasets, make_dataloaders
from utils.helper import create_logger, set_seed

from trainer import Trainer_for_KD

@hydra.main(config_path='./config', config_name='main_KD', version_base=None)
def main(cfg):
    metadata = {
        'Experiment Name': cfg.experiment_name,
        'Dataset': cfg.data.dataset_name,
        'Random_state': cfg.random_state,
        'Epochs': cfg.epochs,
        'Batch_size': cfg.batch_size,
        'Learning_rate': cfg.learning_rate,
        'Student_Input_size': cfg.data.student_input_size,
        'Student_Patch_size': cfg.model.student_patches,
        'Student_Patch_Dim': cfg.model.student_patch_dim,
        'Student_Dropout': cfg.model.student_dropout,
        'Student_Layers': cfg.model.student_num_layers,
        'Student_Activation': cfg.model.student_activation,
        'Teacher_Input_size': cfg.data.teacher_input_size,
        'Teacher_Patch_size': cfg.model.teacher_patches,
        'Teacher_Patch_Dim': cfg.model.teacher_patch_dim,
        'Teacher_Dropout': cfg.model.teacher_dropout,
        'Teacher_Layers': cfg.model.teacher_num_layers,
        'Teacher_Activation': cfg.model.teacher_activation,
    }

    log_out = HydraConfig.get().runtime.output_dir
    logger = create_logger(log_out, dist_rank=0, name=cfg.experiment_name)

    logger.info(metadata)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    stu_train_dataset, stu_val_dataset, stu_test_dataset = make_datasets(cfg.data.train_dir,
                                                                         cfg.data.val_dir,
                                                                         cfg.data.test_dir,
                                                                         cfg.data.student_input_size)
    
    tea_train_dataset, _, _ = make_datasets(cfg.data.train_dir,
                                            cfg.data.val_dir,
                                            cfg.data.test_dir,
                                            cfg.data.teacher_input_size)

    # 데이터셋 정보 출력
    logger.info(f"총 데이터셋 크기: {len(stu_train_dataset)}")
    logger.info(f"검증 데이터셋 크기: {len(stu_val_dataset)}")
    logger.info(f"클래스 수: {len(stu_train_dataset.classes)}")
    logger.info(f"클래스 목록: {stu_train_dataset.classes}")

    train_loader, val_loader, test_loader = make_dataloaders(train_dataset=stu_train_dataset,
                                                             val_dataset=stu_val_dataset,
                                                             test_dataset=stu_test_dataset,
                                                             num_workers=cfg.num_workers,
                                                             batch_size=cfg.batch_size,
                                                             random_state=cfg.data.random_state,)
    
    teacher_loader = make_dataloaders(train_dataset=tea_train_dataset,
                                      num_workers=cfg.num_workers,
                                      batch_size=cfg.batch_size,
                                      random_state=cfg.data.random_state,
                                      mode='KD')
    
    set_seed(cfg.random_state)

    # Model
    student_model = MSPS_stu.MultiscaleMixer(
        in_channels=cfg.model.in_channels,
        patch_dim=cfg.model.student_patch_dim,
        dropout=cfg.model.student_dropout,
        num_layers=cfg.model.student_num_layers,
        patches=cfg.model.student_patches,
        stride=cfg.model.student_stride,
        shift_size=cfg.model.student_shift_size,
        shift=cfg.model.student_shift,
        num_patches=cfg.model.student_num_patches,
        act=cfg.model.student_activation,
    ).to(device)
    
    teacher_model = MSPS_tea.MultiscaleMixer(
        in_channels=cfg.model.in_channels,
        patch_dim=cfg.model.teacher_patch_dim,
        dropout=cfg.model.teacher_dropout,
        num_layers=cfg.model.teacher_num_layers,
        patches=cfg.model.teacher_patches,
        stride=cfg.model.teacher_stride,
        shift_size=cfg.model.teacher_shift_size,
        shift=cfg.model.teacher_shift,
        num_patches=cfg.model.teacher_num_patches,
        act=cfg.model.teacher_activation,
    )
    teacher_model.load_state_dict(torch.load(cfg.teacher_checkpoint, map_location=device))
    teacher_model.to(device)
    teacher_model.eval()

    # loss Function
    cross_entropy = nn.CrossEntropyLoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    MSELoss = nn.MSELoss(reduction='mean')
    focal = FocalLoss(alpha=torch.tensor([1.75, 1.0, 1.75, 1.0, 1.0, 1.0]), 
                      task_type='multi-class', 
                      num_classes=6).to(device)

    projection = nn.ModuleList([
        nn.Conv1d(64, 128, kernel_size=1).to(device),
        nn.Conv1d(64, 128, kernel_size=1).to(device),
        nn.Conv1d(64, 128, kernel_size=1).to(device),
        nn.Conv1d(64, 128, kernel_size=1).to(device)
        ])

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(list(student_model.parameters())+list(projection.parameters()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    
    early_stopping = EarlyStopping(patience=30, mode='min', verbose=True)

    config = {'epochs': cfg.epochs, 'experiment_name': cfg.experiment_name,
              'classes': stu_train_dataset.classes, 'metadata': metadata}
    components = {'optimizer': optimizer, 'scheduler': scheduler, 'device': device}
    loss_config = {'cross_entropy': cross_entropy, 'focal': None, 'lambda_aux': cfg.lambda_aux}
    paths = {'csv': cfg.csv_path, 'best_model': cfg.best_model_path, 'cofusion': cfg.confusion_path}

    # KLDivLoss, MSELoss, lambda_kl, temperature
    KD_loss = {'KLDivLoss': KLDivLoss, 'MSELoss': MSELoss, 'lambda_kl': cfg.lambda_kl, 'temperature': cfg.temperature}
    
    # Create trainer and start training
    trainer = Trainer_for_KD(config=config, model=student_model, components=components, loss_config=loss_config, paths=paths, logger=logger,
                             teacher_model=teacher_model, projection=projection, KD_loss=KD_loss)
    trainer.train(teacher_loader=teacher_loader, student_train_loader=train_loader, student_val_loader=val_loader, early_stopping=early_stopping)

    # Test the best model
    trainer.test(test_loader)

if __name__ == '__main__':
    main()
