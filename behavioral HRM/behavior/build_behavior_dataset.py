from __future__ import annotations

import os
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .tokenizer import Vocab, DurationBinner, CircadianConfig, TokenizationConfig, build_event_tokens, window_context_tokens


@dataclass
class BuilderConfig:
    sqlite_paths: List[str]
    output_root: str
    seq_len: int = 1024
    stride: int = 512
    lights_on_hour: int = 7
    lights_off_hour: int = 19
    include_time_tokens: bool = True
    train_keys: List[str] = None  # type: ignore
    val_keys: List[str] = None    # type: ignore
    test_keys: List[str] = None   # type: ignore


def read_event_filtered_rows(con: sqlite3.Connection):
    cur = con.cursor()
    cols = (
        "id", "idanimalA", "idanimalB", "idanimalC", "idanimalD",
        "name", "startframe", "endframe", "duration", "duration_seconds",
        "event_start_datetime", "GENOTYPE_A", "SETUP_A", "GENOTYPE_B", "SETUP_B",
        "GENOTYPE_C", "SETUP_C", "GENOTYPE_D", "SETUP_D",
    )
    q = f"SELECT {', '.join(cols)} FROM EVENT_FILTERED ORDER BY event_start_datetime"
    for row in cur.execute(q):
        yield dict(zip(cols, row))


def assign_split(basename: str, cfg: BuilderConfig) -> str:
    if cfg.train_keys and basename in cfg.train_keys:
        return "train"
    if cfg.val_keys and basename in cfg.val_keys:
        return "val"
    if cfg.test_keys and basename in cfg.test_keys:
        return "test"
    return "train"


def build_dataset(cfg: BuilderConfig):
    os.makedirs(cfg.output_root, exist_ok=True)
    vocab = Vocab()
    binner = DurationBinner()
    circ = CircadianConfig(lights_on_hour=cfg.lights_on_hour, lights_off_hour=cfg.lights_off_hour)
    tok_cfg = TokenizationConfig()

    splits = {s: {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]} for s in ("train", "val", "test")}
    per_split_group_counts = defaultdict(int)
    per_split_puzzle_counts = defaultdict(int)

    for db_path in cfg.sqlite_paths:
        base = os.path.basename(db_path)
        split = assign_split(base, cfg)
        per_split_group_counts[split] += 1
        con = sqlite3.connect(db_path)
        try:
            per_animal: Dict[int, List[Dict]] = defaultdict(list)
            for ev in read_event_filtered_rows(con):
                if ev["idanimalA"] is not None and ev["idanimalB"] is None and ev["idanimalC"] is None and ev["idanimalD"] is None:
                    per_animal[ev["idanimalA"]].append(ev)
                elif ev["idanimalA"] is not None and ev["idanimalB"] is not None and ev["idanimalC"] is None and ev["idanimalD"] is None:
                    per_animal[ev["idanimalA"]].append(ev)
                    per_animal[ev["idanimalB"]].append(ev)
                else:
                    continue
            for animal_id, events in per_animal.items():
                events.sort(key=lambda e: (e["event_start_datetime"], e["startframe"]))
                stream: List[int] = []
                geno = None
                setup = None
                for ev in events:
                    is_dyad = ev["idanimalB"] is not None
                    is_active = None
                    partner_genotype = None
                    if is_dyad:
                        if ev["idanimalA"] == animal_id:
                            is_active = True
                            partner_genotype = ev.get("GENOTYPE_B")
                            geno = geno or ev.get("GENOTYPE_A")
                            setup = setup or ev.get("SETUP_A")
                        elif ev["idanimalB"] == animal_id:
                            is_active = False
                            partner_genotype = ev.get("GENOTYPE_A")
                            geno = geno or ev.get("GENOTYPE_B")
                            setup = setup or ev.get("SETUP_B")
                    else:
                        geno = geno or ev.get("GENOTYPE_A")
                        setup = setup or ev.get("SETUP_A")
                    toks = build_event_tokens(
                        vocab=vocab,
                        event_name=ev["name"],
                        is_dyad=is_dyad,
                        is_active=is_active,
                        duration_seconds=float(ev["duration_seconds"] or 0.0),
                        partner_genotype=partner_genotype,
                        duration_binner=binner,
                        config=tok_cfg,
                    )
                    stream.extend(toks)
                if not stream:
                    continue
                first = events[0]
                hour = int(str(first["event_start_datetime"])[11:13]) if first["event_start_datetime"] else 0
                if cfg.include_time_tokens:
                    ctx = window_context_tokens(vocab=vocab, setup=str(setup or "neutral"), self_genotype=str(geno or "wt"), hour=hour, circadian=circ, config=tok_cfg)
                else:
                    ctx = [
                        vocab.get_id(f"SETUP_{str(setup or 'neutral').upper()}"),
                        vocab.get_id(f"SELF_{str(geno or 'wt').upper()}"),
                    ]
                seq = ctx + stream
                L, S = cfg.seq_len, cfg.stride
                for start in range(0, max(1, len(seq) - 1), S):
                    window = seq[start:start + L]
                    if len(window) < 64:
                        break
                    labels = window[1:] + [vocab.get_id("EOS")]
                    pad = L - len(window)
                    if pad > 0:
                        window = window + [vocab.get_id("PAD")] * pad
                        labels = labels + [vocab.get_id("PAD")] * pad
                    ds = splits[split]
                    ds["inputs"].append(window)
                    ds["labels"].append(labels)
                ds = splits[split]
                if not ds["puzzle_indices"]:
                    ds["puzzle_indices"].append(0)
                    ds["group_indices"].append(0)
                num_new = len(ds["inputs"]) - ds["puzzle_indices"][-1]
                ds["puzzle_indices"].append(ds["puzzle_indices"][-1] + num_new)
                per_split_puzzle_counts[split] += 1
                ds["puzzle_identifiers"].append(per_split_puzzle_counts[split])
            ds = splits[split]
            ds["group_indices"].append(len(ds["puzzle_identifiers"]))
        finally:
            con.close()
    # save
    with open(os.path.join(cfg.output_root, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab.id_to_token, f)
    for split, ds in splits.items():
        out = os.path.join(cfg.output_root, split)
        os.makedirs(out, exist_ok=True)
        inputs = np.asarray(ds["inputs"], dtype=np.int32)
        labels = np.asarray(ds["labels"], dtype=np.int32)
        pids = np.asarray(ds["puzzle_identifiers"], dtype=np.int32)
        pidx = np.asarray(ds["puzzle_indices"], dtype=np.int32)
        gidx = np.asarray(ds["group_indices"], dtype=np.int32)
        if inputs.size == 0:
            inputs = np.zeros((0, cfg.seq_len), dtype=np.int32)
            labels = np.zeros((0, cfg.seq_len), dtype=np.int32)
            pids = np.zeros((0,), dtype=np.int32)
            pidx = np.zeros((1,), dtype=np.int32)
            gidx = np.zeros((1,), dtype=np.int32)
        np.save(os.path.join(out, "all__inputs.npy"), inputs)
        np.save(os.path.join(out, "all__labels.npy"), labels)
        np.save(os.path.join(out, "all__puzzle_identifiers.npy"), pids)
        np.save(os.path.join(out, "all__puzzle_indices.npy"), pidx)
        np.save(os.path.join(out, "all__group_indices.npy"), gidx)
        meta = {
            "pad_id": 0,
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
            "vocab_size": int(len(vocab.id_to_token)),
            "seq_len": cfg.seq_len,
            "num_puzzle_identifiers": int(pids.max()) + 1 if pids.size else 1,
            "total_groups": int(gidx.size - 1),
            "mean_puzzle_examples": float((inputs.shape[0] / max(1, pids.size))) if pids.size else 0.0,
            "sets": ["all"],
        }
        with open(os.path.join(out, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
    with open(os.path.join(cfg.output_root, "identifiers.json"), "w", encoding="utf-8") as f:
        json.dump(["<blank>"], f)


