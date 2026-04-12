import torch
import torch.nn.functional as F


def causal_mil_loss(model, y_orig, bag_feats, attention_weights, topk_ratio=0.03, lam=0.5):

    B, N, C = bag_feats.shape
    topk = max(1, int(N * topk_ratio))

    with torch.no_grad():
        attn = attention_weights.view(-1)
        topk_idx = torch.topk(attn, topk)[1]
        low_idx = torch.topk(attn, topk, largest=False)[1]


    soft_orig = F.softmax(y_orig, dim=1).detach()


    drop_feats = bag_feats.clone()
    drop_feats[:, topk_idx, :] = 0
    y_drop, _ = model(drop_feats)
    log_soft_drop = F.log_softmax(y_drop, dim=1)


    replace_feats = bag_feats.clone()
    replace_feats[:, low_idx, :] = bag_feats[:, topk_idx, :]
    y_replace, _ = model(replace_feats)
    log_soft_replace = F.log_softmax(y_replace, dim=1)


    kl_drop = F.kl_div(log_soft_drop, soft_orig, reduction='batchmean')
    kl_replace = F.kl_div(log_soft_replace, soft_orig, reduction='batchmean')


    causal_loss = -lam * kl_drop + (1.0 - lam) * kl_replace

    return causal_loss

