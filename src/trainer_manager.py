import os
import time
import logging
from typing import Tuple, List, Dict, Any, Optional
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from models.MIL_models import FCMIL
from trainer import _run_epoch
from .utils import rm_n_mkdir, build_dataset, collate_fn, save_checkpoint
from dataset import MILDataset  # 如果数据集类名未改，可沿用

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingManager:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')

        self._set_random_seed()
        self._setup_directories()

        self.writer = SummaryWriter(self.log_dir)
        self.best_loss = float('inf')
        self.best_auc = 0.0  # FC-MIL 论文主要使用 AUC 评估 [cite: 404, 417]
        self.current_epoch = 0

        self._build_datasets()
        self._build_model()
        self._build_optimizer()

        if getattr(self.cfg, 'pretrained_path', None):
            self._load_pretrained_model()

    def _set_random_seed(self) -> None:
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_directories(self) -> None:
        self.checkpoint_dir = "./checkpoints/"
        self.log_dir = f'./logs/{self.cfg.model_name}_{self.cfg.data_inst}/'
        self.checkpoint_path = f"{self.checkpoint_dir}/{self.cfg.model_name}.pth"

        rm_n_mkdir(self.checkpoint_dir)
        rm_n_mkdir(self.log_dir)

    def _build_datasets(self) -> None:

        try:
            self.dataset_train = MILDataset(self.cfg.train_splits, self.cfg)
            self.dataset_val = MILDataset(self.cfg.val_splits, self.cfg)

            self.dataloaders = {
                'train': DataLoader(
                    self.dataset_train,
                    batch_size=self.cfg.train_batch_size,
                    shuffle=True,  # Bag 间可以打乱，Bag 内 Patch 顺序在 Dataset 中保持
                    num_workers=self.cfg.num_workers,
                    pin_memory=True,
                    collate_fn=collate_fn
                ),
                'val': DataLoader(
                    self.dataset_val,
                    batch_size=self.cfg.test_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=True,
                    collate_fn=collate_fn
                )
            }
            logger.info(f"Datasets built: Train={len(self.dataset_train)}, Val={len(self.dataset_val)}")
        except Exception as e:
            logger.error(f"Dataset build error: {e}");
            raise

    def _build_model(self) -> None:

        try:
            # n_class 对应分类任务类别数 [cite: 391]
            self.model = FCMIL(n_classes=self.cfg.n_class).to(self.device)

            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            logger.info("FC-MIL Model initialized")
        except Exception as e:
            logger.error(f"Model build error: {e}");
            raise

    def _build_optimizer(self) -> None:

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.lr,
            weight_decay=1e-5  # 论文推荐值 [cite: 398]
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )

    def train(self) -> None:
        try:
            for epoch in range(self.cfg.epochs):
                self.current_epoch = epoch

                # 训练阶段：包含 CE + CR Loss [cite: 376]
                train_metrics = _run_epoch(
                    model=self.model, dataloader=self.dataloaders['train'],
                    epoch=epoch, cfg=self.cfg, writer=self.writer,
                    optimizer=self.optimizer, is_training=True
                )

                # 验证阶段
                val_metrics = _run_epoch(
                    model=self.model, dataloader=self.dataloaders['val'],
                    epoch=epoch, cfg=self.cfg, writer=self.writer,
                    is_training=False
                )

                self._log_metrics(train_metrics, val_metrics)
                self._save_best_model(val_metrics)
                self.scheduler.step(val_metrics['loss'])

        except Exception as e:
            logger.error(f"Training error: {e}");
            raise

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict) -> None:
        logger.info(f"Epoch {self.current_epoch + 1}:")
        # FC-MIL 关注整体 Loss 和 AUC [cite: 404]
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        logger.info(
            f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics.get('auc', 0):.4f}")

    def _save_best_model(self, val_metrics: Dict) -> None:
        # 以验证集 AUC 或 Loss 作为保存标准 [cite: 417]
        current_auc = val_metrics.get('auc', 0)
        if current_auc > self.best_auc:
            self.best_auc = current_auc
            torch.save(self.model.state_dict(), self.checkpoint_path)
            logger.info(f"New best AUC: {self.best_auc:.4f} saved.")

    def close(self) -> None:
        self.writer.close()