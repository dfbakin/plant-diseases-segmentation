"""LightningModule for image classification with optional CAM evaluation."""

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from src.metrics.cam_evaluation import (
    CAMEvaluator,
    CAMMethod,
    CAMMetricsAccumulator,
    get_target_layer,
)


class ClassificationModule(L.LightningModule):
    """Lightning module for image classification.

    Handles train/val/test loops with accuracy, F1, and per-class metrics.
    Optionally evaluates CAM quality on samples with GT masks.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        multi_label: bool = False,
        warmup_epochs: int = 0,
        min_lr: float = 1e-5,
        # CAM evaluation settings
        model_name: str | None = None,
        cam_method: CAMMethod = "gradcam",
        cam_threshold: float = 0.5,
        enable_cam_eval: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.multi_label = multi_label
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.model_name = model_name
        self.enable_cam_eval = enable_cam_eval

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # Classification metrics
        self._binary_metric = False
        if multi_label:
            self.criterion = nn.MultiLabelSoftMarginLoss()
            if num_classes == 1:
                self._binary_metric = True
                _make_ap = lambda: torchmetrics.classification.BinaryAveragePrecision()
            else:
                _make_ap = lambda: torchmetrics.classification.MultilabelAveragePrecision(
                    num_labels=num_classes
                )
            self.train_mAP = _make_ap()
            self.val_mAP = _make_ap()
            self.test_mAP = _make_ap()
        else:
            metric_args = {
                "task": "multiclass",
                "num_classes": num_classes,
            }
            self.train_acc = torchmetrics.Accuracy(**metric_args)
            self.val_acc = torchmetrics.Accuracy(**metric_args)
            self.val_f1 = torchmetrics.F1Score(**metric_args, average="macro")
            self.test_acc = torchmetrics.Accuracy(**metric_args)
            self.test_f1 = torchmetrics.F1Score(**metric_args, average="macro")

        # CAM evaluation (lazy initialization)
        self._cam_evaluator: CAMEvaluator | None = None
        self._cam_method = cam_method
        self._cam_threshold = cam_threshold
        self.val_cam_metrics = CAMMetricsAccumulator()
        self.test_cam_metrics = CAMMetricsAccumulator()

        self.save_hyperparameters(ignore=["model", "class_weights"])

    def _get_cam_evaluator(self) -> CAMEvaluator:
        """Lazy initialization of CAM evaluator."""
        if self._cam_evaluator is None:
            if self.model_name is None:
                raise ValueError("model_name required for CAM evaluation")
            target_layer = get_target_layer(self.model, self.model_name)
            self._cam_evaluator = CAMEvaluator(
                model=self.model,
                target_layer=target_layer,
                cam_method=self._cam_method,
                threshold=self._cam_threshold,
            )
        return self._cam_evaluator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_loss(
        self,
        output: torch.Tensor | list,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.multi_label:
            if isinstance(output, list):
                cls_logits, all_cls_emb, patch_logits = output
                loss = self.criterion(cls_logits, target)
                for embed in all_cls_emb:
                    loss = loss + 0.1 * self.criterion(embed.mean(-1), target)
                loss = loss + self.criterion(patch_logits, target)
                return loss
            return self.criterion(output, target)
        return F.cross_entropy(
            output,
            target.long(),
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        output = self(images)
        loss = self.compute_loss(output, labels)

        if self.multi_label:
            logits = output[0] if isinstance(output, list) else output
            preds = torch.sigmoid(logits.detach())
            tgt = labels.int()
            if self._binary_metric:
                preds, tgt = preds.squeeze(-1), tgt.squeeze(-1)
            self.train_mAP.update(preds, tgt)
        else:
            preds = output.detach().argmax(dim=1)
            self.train_acc.update(preds, labels)

        self.log(
            "train/loss",
            loss.detach(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=images.size(0),
        )
        return loss

    def on_train_epoch_start(self) -> None:
        # Release CAM hooks before training (may be active from val sanity check)
        if self._cam_evaluator is not None:
            self._cam_evaluator.cam.activations_and_grads.release()
            self._cam_evaluator = None

    def on_train_epoch_end(self) -> None:
        if self.multi_label:
            self.log(
                "train/mAP",
                self.train_mAP.compute(),
                prog_bar=True,
            )
            self.train_mAP.reset()
        else:
            self.log(
                "train/acc",
                self.train_acc.compute(),
                prog_bar=True,
            )
            self.train_acc.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images, labels = batch["image"], batch["label"]
        output = self(images)

        if self.multi_label:
            logits = output[0] if isinstance(output, list) else output
            loss = self.compute_loss(logits, labels)
            preds = torch.sigmoid(logits.detach())
            tgt = labels.int()
            if self._binary_metric:
                preds, tgt = preds.squeeze(-1), tgt.squeeze(-1)
            self.val_mAP.update(preds, tgt)
        else:
            logits = output
            loss = self.compute_loss(logits, labels)
            preds = logits.detach().argmax(dim=1)
            self.val_acc.update(preds, labels)
            self.val_f1.update(preds, labels)

        self.log(
            "val/loss",
            loss.detach(),
            prog_bar=True,
            on_epoch=True,
            batch_size=images.size(0),
        )

        # CAM evaluation for samples with GT masks
        if self.enable_cam_eval and "mask" in batch:
            self._evaluate_cam_batch(batch, self.val_cam_metrics)

    def _evaluate_cam_batch(self, batch: dict, accumulator: CAMMetricsAccumulator) -> None:
        """Evaluate CAM quality for a batch with GT masks."""
        images = batch["image"]
        labels = batch["label"]
        masks = batch["mask"]

        cam_eval = self._get_cam_evaluator()

        # Enable gradients and inference_mode(False) for CAM computation
        with torch.inference_mode(False):
            with torch.enable_grad():
                images_with_grad = images.detach().clone().requires_grad_(True)
                cams = cam_eval.generate_cam(images_with_grad, labels)

        # Explicitly clear computation graph and CAM internal buffers to prevent memory leak
        del images_with_grad
        cam_eval.cam.activations_and_grads.activations.clear()
        cam_eval.cam.activations_and_grads.gradients.clear()

        # Compute metrics for each sample
        for i in range(len(images)):
            cam = cams[i]
            gt_mask = masks[i].cpu().numpy()
            metrics = cam_eval.compute_metrics(cam, gt_mask)
            accumulator.update(metrics)

        del cams

    def on_validation_epoch_end(self) -> None:
        if self.multi_label:
            self.log(
                "val/mAP",
                self.val_mAP.compute(),
                prog_bar=True,
            )
            self.val_mAP.reset()
        else:
            self.log("val/acc", self.val_acc.compute(), prog_bar=True)
            self.log("val/f1", self.val_f1.compute())
            self.val_acc.reset()
            self.val_f1.reset()

        # Log CAM metrics
        if self.enable_cam_eval and self.val_cam_metrics.count > 0:
            cam_metrics = self.val_cam_metrics.compute()
            # Standard localization metrics
            self.log("val/cam_pointing_acc", cam_metrics["pointing_acc"])
            self.log("val/cam_energy_inside", cam_metrics["energy_inside"])
            self.log("val/cam_iou", cam_metrics["iou"], prog_bar=True)
            self.log("val/cam_f1", cam_metrics["f1"])
            # Multi-region metrics
            self.log("val/cam_region_coverage", cam_metrics["region_coverage"], prog_bar=True)
            self.log("val/cam_peak_coverage", cam_metrics["peak_coverage"])
            self.log("val/cam_peak_precision", cam_metrics["peak_precision"])
            self.log("val/cam_num_gt_regions", cam_metrics["num_gt_regions"])
            self.log("val/cam_num_peaks", cam_metrics["num_peaks"])
            self.val_cam_metrics.reset()

        # Release CAM hooks to prevent memory accumulation during training
        if self._cam_evaluator is not None:
            self._cam_evaluator.cam.activations_and_grads.release()
            self._cam_evaluator = None

    def test_step(self, batch: dict, batch_idx: int) -> None:
        images, labels = batch["image"], batch["label"]
        output = self(images)

        if self.multi_label:
            logits = output[0] if isinstance(output, list) else output
            loss = self.compute_loss(logits, labels)
            preds = torch.sigmoid(logits.detach())
            tgt = labels.int()
            if self._binary_metric:
                preds, tgt = preds.squeeze(-1), tgt.squeeze(-1)
            self.test_mAP.update(preds, tgt)
        else:
            logits = output
            loss = self.compute_loss(logits, labels)
            preds = logits.detach().argmax(dim=1)
            self.test_acc.update(preds, labels)
            self.test_f1.update(preds, labels)

        self.log(
            "test/loss",
            loss.detach(),
            on_epoch=True,
            batch_size=images.size(0),
        )

        # CAM evaluation
        if self.enable_cam_eval and "mask" in batch:
            self._evaluate_cam_batch(batch, self.test_cam_metrics)

    def on_test_epoch_end(self) -> None:
        if self.multi_label:
            self.log("test/mAP", self.test_mAP.compute())
            self.test_mAP.reset()
        else:
            self.log("test/acc", self.test_acc.compute())
            self.log("test/f1", self.test_f1.compute())
            self.test_acc.reset()
            self.test_f1.reset()

        # Log CAM metrics
        if self.enable_cam_eval and self.test_cam_metrics.count > 0:
            cam_metrics = self.test_cam_metrics.compute()
            # Standard localization metrics
            self.log("test/cam_pointing_acc", cam_metrics["pointing_acc"])
            self.log("test/cam_energy_inside", cam_metrics["energy_inside"])
            self.log("test/cam_iou", cam_metrics["iou"])
            self.log("test/cam_precision", cam_metrics["precision"])
            self.log("test/cam_recall", cam_metrics["recall"])
            self.log("test/cam_f1", cam_metrics["f1"])
            # Multi-region metrics
            self.log("test/cam_region_coverage", cam_metrics["region_coverage"])
            self.log("test/cam_peak_coverage", cam_metrics["peak_coverage"])
            self.log("test/cam_peak_precision", cam_metrics["peak_precision"])
            self.log("test/cam_num_gt_regions", cam_metrics["num_gt_regions"])
            self.log("test/cam_num_peaks", cam_metrics["num_peaks"])
            self.test_cam_metrics.reset()

        # Release CAM hooks to free memory
        if self._cam_evaluator is not None:
            self._cam_evaluator.cam.activations_and_grads.release()
            self._cam_evaluator = None

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        max_epochs = self.trainer.max_epochs if self.trainer else 100
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(max_epochs - self.warmup_epochs, 1), eta_min=self.min_lr
        )
        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, total_iters=self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        else:
            scheduler = cosine
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        images = batch["image"]
        output = self(images)
        if self.multi_label:
            logits = output[0] if isinstance(output, list) else output
            probs = torch.sigmoid(logits)
            return {
                "predictions": (probs > 0.5).int(),
                "probabilities": probs,
                "names": batch.get("name", []),
            }
        return {
            "predictions": output.argmax(dim=1),
            "probabilities": F.softmax(output, dim=1),
            "names": batch.get("name", []),
        }
