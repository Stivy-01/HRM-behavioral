import argparse
import json
import os
from typing import List, Dict

from behavior.build_behavior_dataset import BuilderConfig, build_dataset


def make_config_from_manifest(manifest_path: str, output_root: str, seq_len: int, stride: int,
                              include_time_tokens: bool, lights_on_hour: int, lights_off_hour: int) -> BuilderConfig:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest: Dict[str, List[str]] = json.load(f)

    train_list = manifest.get("train", [])
    val_list   = manifest.get("val", [])
    test_list  = manifest.get("test", [])

    sqlite_paths: List[str] = train_list + val_list + test_list
    train_keys = [os.path.basename(p) for p in train_list]
    val_keys   = [os.path.basename(p) for p in val_list]
    test_keys  = [os.path.basename(p) for p in test_list]

    return BuilderConfig(
        sqlite_paths=sqlite_paths,
        output_root=output_root,
        seq_len=seq_len,
        stride=stride,
        lights_on_hour=lights_on_hour,
        lights_off_hour=lights_off_hour,
        include_time_tokens=include_time_tokens,
        train_keys=train_keys,
        val_keys=val_keys,
        test_keys=test_keys,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to split manifest JSON with keys train/val/test (lists of SQLite paths)")
    ap.add_argument("--output_root", required=True, help="Destination dataset root (e.g., HRM/data/behavior-v1)")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--include_time_tokens", type=int, default=1, help="1 to include DAY/HOUR_BIN tokens, 0 to drop")
    ap.add_argument("--lights_on_hour", type=int, default=7)
    ap.add_argument("--lights_off_hour", type=int, default=19)
    args = ap.parse_args()

    cfg = make_config_from_manifest(
        manifest_path=args.manifest,
        output_root=args.output_root,
        seq_len=args.seq_len,
        stride=args.stride,
        include_time_tokens=bool(args.include_time_tokens),
        lights_on_hour=args.lights_on_hour,
        lights_off_hour=args.lights_off_hour,
    )
    build_dataset(cfg)
    print("DONE: built dataset at", args.output_root)


if __name__ == "__main__":
    main()


