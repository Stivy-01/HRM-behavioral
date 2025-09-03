import argparse
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from behavioral_HRM.behavior.tokenizer import (
    Vocab,
    DurationBinner,
    CircadianConfig,
    TokenizationConfig,
    build_event_tokens,
    window_context_tokens,
)


def parse_datetime(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    s = str(ts)
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def hour_in_window(t: Optional[datetime], start_hour: int, end_hour: int) -> bool:
    if t is None:
        return False
    h = t.hour
    # supports normal and wrap-around windows (e.g., 22 -> 3)
    if start_hour <= end_hour:
        return start_hour <= h < end_hour
    return h >= start_hour or h < end_hour


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
class WindowedBuilder:
    output_root: str
    seq_len: int = 256
    stride: int = 128
    hour_start: int = 19
    hour_end: int = 24
    include_time_tokens: bool = True

    def __post_init__(self):
        os.makedirs(self.output_root, exist_ok=True)
        self.vocab = Vocab()
        self.binner = DurationBinner()
        # Default circadian model (lights_on=7, lights_off=19)
        self.circ = CircadianConfig(lights_on_hour=7, lights_off_hour=19)
        self.tok_cfg = TokenizationConfig()
        self.splits = {
            s: {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
            for s in ("train", "val", "test")
        }
        self.per_split_puzzle_counts = defaultdict(int)

    def _ds(self, split: str):
        return self.splits[split]

    def add_sqlite(self, db_path: str):
        con = sqlite3.connect(db_path)
        try:
            # group by calendar date (YYYY-MM-DD), then by animal id
            per_date_per_animal: Dict[str, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))

            for ev in read_event_rows(con):
                # keep individual & dyadic only
                if ev["idanimalA"] is None:
                    continue
                if ev["idanimalC"] is not None or ev["idanimalD"] is not None:
                    continue

                t = parse_datetime(ev["event_start_datetime"])
                if not hour_in_window(t, self.hour_start, self.hour_end):
                    continue

                # decide which animals to attribute this event to
                if ev["idanimalB"] is None:
                    animal_ids = [ev["idanimalA"]]
                else:
                    animal_ids = [ev["idanimalA"], ev["idanimalB"]]

                date_key = t.strftime("%Y-%m-%d") if t else "UNKNOWN_DATE"
                for aid in animal_ids:
                    per_date_per_animal[date_key][int(aid)].append(ev)

            # simple split policy for testing: everything to train
            split = "train"

            for date_key, per_animal in sorted(per_date_per_animal.items()):
                for animal_id, events in per_animal.items():
                    events.sort(key=lambda e: (e["event_start_datetime"], e["startframe"]))
                    stream: List[int] = []
                    geno = None
                    setup = None
                    for ev in events:
                        is_dyad = ev["idanimalB"] is not None
                        is_active: Optional[bool] = None
                        partner_genotype: Optional[str] = None
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

                    # context: fix hour to window start to indicate nighttime
                    if self.include_time_tokens:
                        ctx = window_context_tokens(
                            vocab=self.vocab,
                            setup=str(setup or "neutral"),
                            self_genotype=str(geno or "wt"),
                            hour=self.hour_start,
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
                    ds = self._ds(split)
                    if not ds["puzzle_indices"]:
                        ds["puzzle_indices"].append(0)
                        ds["group_indices"].append(0)

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

                    prev = ds["puzzle_indices"][-1]
                    ds["puzzle_indices"].append(len(ds["inputs"]))
                    self.per_split_puzzle_counts[split] += 1
                    ds["puzzle_identifiers"].append(self.per_split_puzzle_counts[split])

                # close group per date
                ds = self._ds(split)
                ds["group_indices"].append(len(ds["puzzle_identifiers"]))
        finally:
            con.close()

    def save(self):
        # write vocab
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", nargs="+", required=True, help="One or more SQLite files with EVENT_FILTERED")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--hour_start", type=int, default=19)
    ap.add_argument("--hour_end", type=int, default=24)
    ap.add_argument("--include_time_tokens", type=int, default=1)
    args = ap.parse_args()

    builder = WindowedBuilder(
        output_root=args.output_root,
        seq_len=args.seq_len,
        stride=args.stride,
        hour_start=args.hour_start,
        hour_end=args.hour_end,
        include_time_tokens=bool(args.include_time_tokens),
    )
    for p in args.sqlite:
        builder.add_sqlite(p)
    builder.save()
    print("DONE: built", args.output_root)


if __name__ == "__main__":
    main()