import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class MorphologyAwareLoss(nn.Module):
    def __init__(self, classes: int, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.classes = classes
        self.alpha = alpha  # Вес Dice (региональная согласованность)
        self.beta = beta  # Вес Boundary (контурная точность)

        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Вычисление морфологического градиента для выделения границ включений."""
        kernel = torch.ones(3, 3, device=mask.device).unsqueeze(0).unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        dilated = torch.nn.functional.conv2d(mask.float(), kernel, padding=1)
        eroded = torch.nn.functional.conv2d(
            mask.float(), kernel, padding=1, stride=1
        )
        eroded = torch.nn.functional.max_pool2d(
            -mask.float().unsqueeze(1), kernel_size=3, stride=1, padding=1
        ) * -1
        return (dilated - eroded).clamp(0, 1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=self.classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dice_loss = self.dice(logits, targets_one_hot)

        # Boundary loss: применяется к logits и one-hot targets
        probas = torch.sigmoid(logits)
        boundary_pred = self._get_boundary(probas)
        boundary_target = self._get_boundary(targets_one_hot)
        boundary_loss = self.bce(boundary_pred, boundary_target)

        return self.alpha * dice_loss + self.beta * boundary_loss