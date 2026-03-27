import torch
import torch.nn as nn
import torch.nn.functional as F

class Info_NCELoss(nn.Module):
    def __init__(self, temperature=0.5, device=None):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, inter, intra):
        batch_size = inter.shape[0]
        
        z1_norm = F.normalize(inter, dim=1)
        z2_norm = F.normalize(intra, dim=1)
        
        features = torch.cat([z1_norm, z2_norm], dim=0)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = torch.arange(2 * batch_size, device=self.device)
        labels = (labels + batch_size) % (2 * batch_size)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class Info_NCELoss_test(nn.Module):
    def __init__(self, temperature=0.5, device=None):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, output_list):
        batch_size = output_list[0].shape[0]
        
        z1 = torch.mean(output_list[0], dim=2, keepdim=False)
        z2 = torch.mean(output_list[1], dim=2, keepdim=False)

        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        features = torch.cat([z1_norm, z2_norm], dim=0)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = torch.arange(2 * batch_size, device=self.device)
        labels = (labels + batch_size) % (2 * batch_size)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.cosine_similarity = nn.CosineSimilarity(dim=1)
    
    def forward(self, s_x, t_x):
        cos_sim = self.cosine_similarity(s_x, t_x)
        loss = 1 - torch.mean(cos_sim)

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss