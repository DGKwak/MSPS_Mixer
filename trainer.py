import os
import datetime
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils.visualization import plot_confusion_matrix
from torch import nn

class Trainer:
    def __init__(self,
                 config,        # epochs, experiment_name, metadata, classes
                 model, 
                 components,    # optimizer, scheduler, device
                 loss_config,   # cross_entropy, focal, lambda_aux
                 paths,         # train_csv_path, test_csv_path, best_model_path, coffusion_path
                 logger):
        # Unpack config
        self.epoch = config['epochs']
        self.experiment_name = config['experiment_name']
        self.metadata = config['metadata']
        self.classes = config['classes']

        self.model = model

        # Unpack components
        self.optimizer = components['optimizer']
        self.scheduler = components['scheduler']
        self.device = components['device']

        # Unpack loss config
        self.cross_entropy = loss_config['cross_entropy']
        self.focal = loss_config['focal']
        self.lambda_aux = loss_config['lambda_aux']

        self.logger = logger

        # Setting Paths
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        os.makedirs(paths['csv'], exist_ok=True)
        os.makedirs(paths['best_model'], exist_ok=True)
        os.makedirs(paths['cofusion'], exist_ok=True)
        
        self.train_csv_path = f"{paths['csv']}/{self.experiment_name}_train_results_{timestamp}.csv"
        self.test_csv_path = f"{paths['csv']}/{self.experiment_name}_test_results_{timestamp}.csv"
        self.best_model_path = f"{paths['best_model']}/{self.experiment_name}_best_model_{timestamp}.pth"
        self.cofusion_path = f"{paths['cofusion']}/{self.experiment_name}_confusion_matrix_{timestamp}.png"

    def train(self, train_loader, val_loader, early_stopping):
        results = []
        train_accuracies, val_accuracies = [], []
        train_losses, val_losses = [], []

        best_val_loss = float('inf')

        for epoch in tqdm(range(self.epoch)):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._val_epoch(val_loader, mode="Validation")

            self.scheduler.step()
            
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }
            results.append(epoch_result)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info(f"Epoch {epoch + 1}: 검증 손실이 감소했습니다. 최적의 모델을 {self.best_model_path}에 저장했습니다.")

            self.logger.info(f"lr : {self.scheduler.get_last_lr()[0]} \n Epoch {epoch + 1}/{self.epoch}, \
                        Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                self.logger.info("Early stopping triggered. Training halted.")
                break
        
        results_df = pd.DataFrame(results)
        
        with open(self.train_csv_path, 'w') as f:
            for key, value in self.metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("\n")
            results_df.to_csv(f, index=False)
        
        self.logger.info(f"Training results saved to {self.train_csv_path}")

        self.logger.info(f"\n=== 학습 완료 요약 ===")
        self.logger.info(f"최고 훈련 정확도: {max(train_accuracies):.4f}")
        self.logger.info(f"최고 검증 정확도: {max(val_accuracies):.4f}")
        self.logger.info(f"최종 훈련 손실: {train_losses[-1]:.4f}")
        self.logger.info(f"최종 검증 손실: {val_losses[-1]:.4f}")
        self.logger.info(f"최종 훈련 정확도: {train_accuracies[-1]:.4f}")
        self.logger.info(f"최종 검증 정확도: {val_accuracies[-1]:.4f}")
    
    def test(self, loader):
        self.logger.info(f"Loading best model from {self.best_model_path} for testing...")

        test_accuracy, predictions, targets = self._val_epoch(loader, mode="Test")

        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"\n=== Classification Report ===")
        self.logger.info(classification_report(targets, predictions))

        plot_confusion_matrix(targets, predictions, self.classes, self.experiment_name, self.cofusion_path)
        self.logger.info(f"Confusion matrix saved to {self.cofusion_path}")

        test_metadata = self.metadata.copy()
        test_metadata.update({
            'Test Accuracy': test_accuracy,
            'Best Model Path': self.best_model_path,
            'Confusion Matrix Path': self.cofusion_path
        })

        with open(self.test_csv_path, 'w') as f:
            for key, value in test_metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("\n")
            # 클래스별 정확도도 저장
            f.write("# Classification Report\n")
            f.write(classification_report(targets, predictions, target_names=self.classes))
        
        self.logger.info(f"Test results saved to {self.test_csv_path}")

    def _train_epoch(self, loader):
        self.model.train()

        total_loss = 0.0
        correct = 0

        for x, y in loader:
            if y.ndim == 2:
                y = torch.argmax(y, dim=1)
            
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logit = self.model(x)

            ce_loss = self.focal(logit, y)
            aux_loss = 0

            z = self.model.get_Mixer_outputs()

            if len(z) == 1:
                aux_loss = 0
            else:
                for out in z:
                    aux_loss += self.cross_entropy(out, y)

            loss = ce_loss + self.lambda_aux * aux_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logit.argmax(1) == y).sum().item()
    
        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    def _val_epoch(self, loader, mode="Validation"):
        self.model.eval()

        total_loss = 0.0
        correct = 0

        all_preds = []
        all_targets = []

        with torch.inference_mode():
            for x, y in loader:
                if y.ndim == 2:
                    y = torch.argmax(y, dim=1)
                
                x, y = x.to(self.device), y.to(self.device)

                logit = self.model(x)

                if mode == "Validation":
                    if self.focal is not None:
                        main_loss = self.focal(logit, y)
                    else:
                        main_loss = self.cross_entropy(logit, y)
                    aux_loss = 0

                    z = self.model.get_Mixer_outputs()

                    if len(z) != 1:
                        for out in z:
                            aux_loss += self.cross_entropy(out, y)
                    
                    loss = main_loss + self.lambda_aux * aux_loss
                    total_loss += loss.item() * x.size(0)
                
                correct += (logit.argmax(1) == y).sum().item()

                all_preds.extend(logit.argmax(1).cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        accuracy = correct / len(loader.dataset)

        if mode == "Validation":
            return total_loss / len(loader.dataset), accuracy
        else:
            return accuracy, all_preds, all_targets

class Trainer_for_KD(Trainer):
    def __init__(self, 
                 teacher_model, 
                 projection,
                 KD_loss,           # KLDivLoss, MSELoss, lambda_kl, temperature
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.projection = projection

        # Unpack KD loss
        self.KLDivLoss = KD_loss['KLDivLoss']
        self.MSELoss = KD_loss['MSELoss']
        self.lambda_kl = KD_loss['lambda_kl']
        self.temperature = KD_loss['temperature']
    
    def train(self, teacher_loader, student_train_loader, student_val_loader, early_stopping):
        results = []
        train_accuracies, val_accuracies = [], []
        train_losses, val_losses = [], []

        best_val_loss = float('inf')

        for epoch in tqdm(range(self.epoch)):
            train_loss, train_acc = self._train_epoch(teacher_loader, student_train_loader)
            val_loss, val_acc = self._val_epoch(student_val_loader, mode="Validation")

            self.scheduler.step()

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }
            results.append(epoch_result)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info(f"Epoch {epoch + 1}: 검증 손실이 감소했습니다. 최적의 모델을 {self.best_model_path}에 저장했습니다.")

            self.logger.info(f"lr : {self.scheduler.get_last_lr()[0]} \n Epoch {epoch + 1}/{self.epoch}, \
                        Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                self.logger.info("Early stopping triggered. Training halted.")
                break
        
        results_df = pd.DataFrame(results)
        
        with open(self.train_csv_path, 'w') as f:
            for key, value in self.metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("\n")
            results_df.to_csv(f, index=False)
        
        self.logger.info(f"Training results saved to {self.train_csv_path}")

        self.logger.info(f"\n=== 학습 완료 요약 ===")
        self.logger.info(f"최고 훈련 정확도: {max(train_accuracies):.4f}")
        self.logger.info(f"최고 검증 정확도: {max(val_accuracies):.4f}")
        self.logger.info(f"최종 훈련 손실: {train_losses[-1]:.4f}")
        self.logger.info(f"최종 검증 손실: {val_losses[-1]:.4f}")
        self.logger.info(f"최종 훈련 정확도: {train_accuracies[-1]:.4f}")
        self.logger.info(f"최종 검증 정확도: {val_accuracies[-1]:.4f}")

    def _compute_gram_matrix(self, feat):
        B, C, N = feat.size()

        gram = torch.bmm(feat, feat.transpose(1, 2))

        return gram / N

    def _train_epoch(self, teacher_loader, student_loader):
        self.model.train()

        total_loss = 0.0
        correct = 0

        for (x_s, y_s), (x_t, _) in zip(student_loader, teacher_loader):
            x_s, y_s = x_s.to(self.device), y_s.to(self.device)
            x_t = x_t.to(self.device)

            self.optimizer.zero_grad()
            student_logit = self.model(x_s)
            teacher_logit = self.teacher_model(x_t)

            student_ds = self.model.get_ds_outputs()
            teacher_ds = self.teacher_model.get_ds_outputs()

            ce_loss = self.cross_entropy(student_logit, y_s)
            # ce_loss = self.focal(student_logit, y_s)

            student_soft_label = nn.functional.log_softmax(student_logit / self.temperature, dim=1)
            teacher_soft_label = nn.functional.softmax(teacher_logit / self.temperature, dim=1)
            kl_loss = self.KLDivLoss(student_soft_label, teacher_soft_label) * (self.temperature ** 2)

            mse_loss = 0
            for idx, (s_feat, t_feat) in enumerate(zip(student_ds, teacher_ds)):
                s_proj = self.projection[idx](s_feat)

                s_gram = self._compute_gram_matrix(s_proj)
                t_gram = self._compute_gram_matrix(t_feat)

                mse_loss += self.MSELoss(s_gram, t_gram)
            
            loss = ce_loss + self.lambda_kl * kl_loss + self.lambda_aux * mse_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x_s.size(0)
            correct += (student_logit.argmax(1) == y_s).sum().item()

        return total_loss / len(student_loader.dataset), correct / len(student_loader.dataset)