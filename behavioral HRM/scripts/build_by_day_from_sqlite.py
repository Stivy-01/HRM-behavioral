import argparse
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import numpy as np

from behavior.tokenizer import (
    Vocab,
    DurationBinner,
    CircadianConfig,
    TokenizationConfig,
    build_event_tokens,
    window_context_tokens,
)


def parse_days_list(s: str) -> Set[int]:
    return {int(x) for x in s.replace(" ", "").split(",") if x}


def compute_experiment_day(ts: Optional[str], start_hour: int) -> Optional[int]:
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts))
    except Exception:
        try:
            t = datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    # If event time is >= start_hour, it's the current calendar day; otherwise, previous day
    if t.hour >= start_hour:
        return t.day
    return (t - timedelta(hours=24)).day


def read_event_rows(con: sqlite3.Connection):
    cur = con.cursor()
    cols = (
        "idanimalA", "idanimalB", "idanimalC", "idanimalD",
        "name", "startframe", "endframe", "duration", "duration_seconds",
        "event_start_datetime", "GENOTYPE_A", "SETUP_A", "GENOTYPE_B", "SETUP_B",
        "GENOTYPE_C", "SETUP_C", "GENOTYPE_D", "SETUP_D",
    )
    q = f"SELECT {', '.join(cols)} FROM EVENT_FILTERED ORDER BY event_start_datetime"
    for row in cur.execute(q):
        yield dict(zip(cols, row))


@dataclass
class Builder:
    output_root: str
    seq_len: int
    stride: int
    start_hour: int
    include_time_tokens: bool
    train_days: Set[int]
    val_days: Set[int]
    test_days: Set[int]

    def __post_init__(self):
        self.vocab = Vocab()
        self.binner = DurationBinner()
        self.circ = CircadianConfig(lights_on_hour=self.start_hour, lights_off_hour=(self.start_hour + 12) % 24)
        self.tok_cfg = TokenizationConfig()
        self.splits = {s: {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]} for s in ("train", "val", "test")}
        self.per_split_puzzle_counts = defaultdict(int)

    def _split_for_day(self, day: int) -> Optional[str]:
        if day in self.train_days:
            return "train"
        if day in self.val_days:
            return "val"
        if day in self.test_days:
            return "test"
        return None

    def add_sqlite(self, db_path: str):
        con = sqlite3.connect(db_path)
        try:
            per_day_per_animal: Dict[int, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
            for ev in read_event_rows(con):
                # keep individual & dyadic only
                if ev["idanimalA"] is not None and ev["idanimalB"] is None and ev["idanimalC"] is None and ev["idanimalD"] is None:
                    day = compute_experiment_day(ev["event_start_datetime"], self.start_hour)
                    if day is not None:
                        per_day_per_animal[day][ev["idanimalA"]].append(ev)
                elif ev["idanimalA"] is not None and ev["idanimalB"] is not None and ev["idanimalC"] is None and ev["idanimalD"] is None:
                    day = compute_experiment_day(ev["event_start_datetime"], self.start_hour)
                    if day is not None:
                        per_day_per_animal[day][ev["idanimalA"]].append(ev)
                        per_day_per_animal[day][ev["idanimalB"]].append(ev)

            for day, per_animal in sorted(per_day_per_animal.items()):
                split = self._split_for_day(day)
                if split is None:
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
                            vocab=self.vocab,
                            event_name=ev["name"],
                            is_dyad=is_dyad,
                            is_active=is_active,
                            duration_seconds=float(ev["duration_seconds"] or 0.0),
                            partner_genotype=partner_genotype,
                            duration_binner=self.binner,
                            config=self.tok_cfg,
                        )
                        stream.extend(toks)
                    if not stream:
                        continue
                    # use day start as context hour (start_hour)
                    if self.include_time_tokens:
                        ctx = window_context_tokens(
                            vocab=self.vocab,
                            setup=str(setup or "neutral"),
                            self_genotype=str(geno or "wt"),
                            hour=self.start_hour,
                            circadian=self.circ,
                            config=self.tok_cfg,
                        )
                    else:
                        ctx = [
                            self.vocab.get_id(f"SETUP_{str(setup or 'neutral').upper()}"),
                            self.vocab.get_id(f"SELF_{str(geno or 'wt').upper()}"),
                        ]
                    seq = ctx + stream
                    L, S = self.seq_len, self.stride
                    ds = self.splits[split]
                    # init indices
                    if not ds["puzzle_indices"]:
                        ds["puzzle_indices"].append(0)
                        ds["group_indices"].append(0)
                    # windows
                    start = 0
                    while start < max(1, len(seq) - 1):
                        window = seq[start:start + L]
                        if len(window) < 64:
                            break
                        labels = window[1:] + [self.vocab.get_id("EOS")]
                        pad = L - len(window)
                        if pad > 0:
                            window += [self.vocab.get_id("PAD")] * pad
                            labels += [self.vocab.get_id("PAD")] * pad
                        ds["inputs"].append(window)
                        ds["labels"].append(labels)
                        start += S
                    # finalize puzzle indices
                    prev = ds["puzzle_indices"][-1]
                    ds["puzzle_indices"].append(len(ds["inputs"]))
                    self.per_split_puzzle_counts[split] += 1
                    ds["puzzle_identifiers"].append(self.per_split_puzzle_counts[split])
                # close group
                ds = self.splits[split]
                ds["group_indices"].append(len(ds["puzzle_identifiers"]))
        finally:
            con.close()

    def save(self):
        os.makedirs(self.output_root, exist_ok=True)
        # vocab
        import json
        with open(os.path.join(self.output_root, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab.id_to_token, f)
        for split, ds in self.splits.items():
            out = os.path.join(self.output_root, split)
            os.makedirs(out, exist_ok=True)
            inputs = np.asarray(ds["inputs"], dtype=np.int32)
            labels = np.asarray(ds["labels"], dtype=np.int32)
            pids = np.asarray(ds["puzzle_identifiers"], dtype=np.int32)
            pidx = np.asarray(ds["puzzle_indices"], dtype=np.int32)
            gidx = np.asarray(ds["group_indices"], dtype=np.int32)
            if inputs.size == 0:
                inputs = np.zeros((0, self.seq_len), dtype=np.int32)
                labels = np.zeros((0, self.seq_len), dtype=np.int32)
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
                "vocab_size": int(len(self.vocab.id_to_token)),
                "seq_len": self.seq_len,
                "num_puzzle_identifiers": int(pids.max()) + 1 if pids.size else 1,
                "total_groups": int(gidx.size - 1),
                "mean_puzzle_examples": float((inputs.shape[0] / max(1, pids.size))) if pids.size else 0.0,
                "sets": ["all"],
            }
            with open(os.path.join(out, "dataset.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
        with open(os.path.join(self.output_root, "identifiers.json"), "w", encoding="utf-8") as f:
            json.dump(["<blank>"], f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", nargs="+", required=True, help="One or more EVENT_FILTERED SQLite paths")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--start_hour", type=int, default=19, help="Experiment day starts at this hour (0-23)")
    ap.add_argument("--include_time_tokens", type=int, default=1)
    ap.add_argument("--train_days", default="14,15")
    ap.add_argument("--val_days", default="")
    ap.add_argument("--test_days", default="16")
    args = ap.parse_args()

    builder = Builder(
        output_root=args.output_root,
        seq_len=args.seq_len,
        stride=args.stride,
        start_hour=args.start_hour,
        include_time_tokens=bool(args.include_time_tokens),
        train_days=parse_days_list(args.train_days),
        val_days=parse_days_list(args.val_days),
        test_days=parse_days_list(args.test_days),
    )
    for p in args.sqlite:
        builder.add_sqlite(p)
    builder.save()
    print("DONE: built", args.output_root)


if __name__ == "__main__":
    main()


