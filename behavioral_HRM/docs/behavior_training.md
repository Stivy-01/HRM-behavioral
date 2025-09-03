## Training HRM on behavior dataset

### Prerequisites
- Build dataset at `HRM/data/behavior-v1/{train,val,test}` (see dataset docs).
- Confirm `vocab.json` exists at dataset root.

### Configs
- `config/arch/hrm_behavior.yaml` — `puzzle_emb_ndim: 0`, other defaults as in hrm_v1.
- `config/cfg_behavior_pretrain.yaml` — set `data_path` to your dataset root.

### Launch
```
python pretrain.py hydra.run.dir=. --config-name cfg_behavior_pretrain
```

### Notes
- Batch size depends on GPU memory; start at 256 and adjust.
- For quick smoke tests, reduce `epochs` and `eval_interval`.

