import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Dict, Optional, Any

from src.causal_mil_loss import causal_mil_loss


def _run_epoch(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               epoch: int,
               cfg: Any,
               optimizer: Optional[torch.optim.Optimizer] = None,
               is_training: bool = True) -> Dict[str, float]:

    if is_training:
        model.train()
    else:
        model.eval()

    # 初始化指标统计
    total_loss = 0.0
    total_ce_loss = 0.0
    total_cr_loss = 0.0
    correct = 0
    total = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger(__name__)
    context = torch.enable_grad() if is_training else torch.no_grad()

    try:
        with context:
            for batch_idx, (bag_feats, labels) in enumerate(
                    tqdm(dataloader, desc=f"{'Train' if is_training else 'Val'} Epoch {epoch}")):

                bag_feats = bag_feats.to(device)
                labels = labels.to(device)


                logits, attn_weights = model(bag_feats)


                loss_ce = nn.CrossEntropyLoss()(logits, labels)


                current_cr_loss = torch.tensor(0.0).to(device)
                if is_training:

                    current_cr_loss = causal_mil_loss(
                        model=model,
                        y_orig=logits,
                        bag_feats=bag_feats,
                        attention_weights=attn_weights,
                        topk_ratio=getattr(cfg, 'topk_ratio', 0.03),
                        lam=getattr(cfg, 'lam', 0.5)
                    )


                    alpha = getattr(cfg, 'alpha_cr', 0.1)
                    loss = loss_ce + alpha * current_cr_loss
                else:
                    loss = loss_ce

                # 5. 反向传播与优化
                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 6. 统计更新
                total_loss += loss.item()
                total_ce_loss += loss_ce.item()
                if is_training:
                    total_cr_loss += current_cr_loss.item()

                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        # 计算平均指标
        num_batches = len(dataloader)
        avg_metrics = {
            'loss': total_loss / num_batches,
            'loss_ce': total_ce_loss / num_batches,
            'loss_cr': total_cr_loss / num_batches if is_training else 0.0,
            'accuracy': correct / total
        }

        return avg_metrics

    except Exception as e:
        logger.error(f"Error in {'training' if is_training else 'validation'} phase: {e}")
        raise