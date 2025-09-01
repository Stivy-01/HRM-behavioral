import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.losses import ACTLossHead


def load_metadata(data_path: str, split: str):
    with open(os.path.join(data_path, split, "dataset.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(arch_path: str, data_meta: dict, batch_size: int, device: torch.device) -> ACTLossHead:
    try:
        import yaml
        arch = yaml.safe_load(open(arch_path, "r", encoding="utf-8"))
    except Exception:
        arch = {
            "H_cycles": 2, "L_cycles": 2, "H_layers": 4, "L_layers": 4,
            "hidden_size": 512, "num_heads": 8, "expansion": 4,
            "puzzle_emb_ndim": 0, "pos_encodings": "rope",
            "halt_max_steps": 16, "halt_exploration_prob": 0.0,
            "forward_dtype": "bfloat16",
        }
    model_cfg = {
        **arch,
        "batch_size": batch_size,
        "seq_len": int(data_meta["seq_len"]),
        "vocab_size": int(data_meta["vocab_size"]),
        "num_puzzle_identifiers": int(data_meta["num_puzzle_identifiers"]),
    }
    model = HierarchicalReasoningModel_ACTV1(model_cfg)
    loss_head = ACTLossHead(model, loss_type=arch.get("loss", {}).get("loss_type", "stablemax_cross_entropy"))
    return loss_head.to(device)


@torch.no_grad()
def eval_split(model: ACTLossHead, dataset: PuzzleDataset, device: torch.device):
    model.eval()
    total_tokens = 0
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    for set_name, batch, _global_bs in dataset:
        batch = {k: v.to(device) for k, v in batch.items()}
        carry = model.model.initial_carry(batch)
        carry, outputs = model.model(carry, batch)
        logits = outputs["logits"].to(torch.float32)
        labels = batch["labels"]
        mask = labels != -100
        n = mask.sum().item()
        if n == 0:
            continue
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100, reduction="sum")
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=-1)
        top1 = probs.argmax(dim=-1)
        total_top1 += ((top1 == labels) & mask).sum().item()
        topk_vals, topk_idx = probs.topk(k=min(5, probs.shape[-1]), dim=-1)
        total_top5 += ((topk_idx == labels.unsqueeze(-1)).any(dim=-1) & mask).sum().item()
        total_tokens += n
    ppl = float(np.exp(total_loss / max(1, total_tokens))) if total_tokens > 0 else float("nan")
    acc1 = total_top1 / max(1, total_tokens)
    acc5 = total_top5 / max(1, total_tokens)
    return {"tokens": total_tokens, "perplexity": ppl, "top1": acc1, "top5": acc5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--arch", default="HRM/config/arch/hrm_behavior.yaml")
    ap.add_argument("--checkpoint", required=False)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    meta = load_metadata(args.data_path, args.split)
    model = build_model(args.arch, meta, batch_size=args.batch_size, device=device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
    ds = PuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_path=args.data_path, global_batch_size=args.batch_size, test_set_mode=True, epochs_per_iter=1, rank=0, num_replicas=1), split=args.split)
    metrics = eval_split(model, ds, device)
    print(json.dumps({args.split: metrics}, indent=2))


if __name__ == "__main__":
    main()


