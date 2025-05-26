import torch


def get_mlm_acc(pred_token_ids, gt_token_ids):
    argmax = torch.argmax(pred_token_ids, dim=2)
    acc = (gt_token_ids == argmax).sum() / gt_token_ids.numel()
    return acc.item()
