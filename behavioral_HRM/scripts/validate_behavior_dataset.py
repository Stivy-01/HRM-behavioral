import os
import json
import numpy as np
import sys


def main():
    root = os.environ.get("HRM_BEHAVIOR_DATA", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "behavior-v1"))
    print("DATA_ROOT", root)

    vocab_path = os.path.join(root, "vocab.json")
    vocab = []
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    print("vocab_size", len(vocab))

    for split in ["train", "val", "test"]:
        d = os.path.join(root, split)
        p_inputs = os.path.join(d, "all__inputs.npy")
        p_labels = os.path.join(d, "all__labels.npy")
        p_meta = os.path.join(d, "dataset.json")

        if not os.path.exists(p_inputs):
            print(split, "(missing)")
            continue

        inputs = np.load(p_inputs, mmap_mode="r")
        labels = np.load(p_labels, mmap_mode="r")
        meta = json.load(open(p_meta, "r", encoding="utf-8"))

        print(split, "inputs", inputs.shape, "labels", labels.shape, "seq_len(meta)", meta.get("seq_len"), "vocab(meta)", meta.get("vocab_size"))

        p_pids = os.path.join(d, "all__puzzle_identifiers.npy")
        p_pidx = os.path.join(d, "all__puzzle_indices.npy")
        p_gidx = os.path.join(d, "all__group_indices.npy")
        pids = np.load(p_pids)
        pidx = np.load(p_pidx)
        gidx = np.load(p_gidx)
        print(split, "puzzles", pids.shape, "puzzle_indices", pidx.shape, "group_indices", gidx.shape)

        if inputs.shape[0] > 0 and len(vocab):
            row = inputs[0][:32].tolist()
            toks = [vocab[t] if 0 <= t < len(vocab) else f"<unk:{t}>" for t in row]
            print(split, "preview", toks)

    # Try HRM loader
    hrm_root = os.path.dirname(os.path.dirname(__file__))
    if hrm_root not in sys.path:
        sys.path.insert(0, hrm_root)
    try:
        from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
        cfg = PuzzleDatasetConfig(seed=0, dataset_path=root, global_batch_size=32, test_set_mode=False, epochs_per_iter=1, rank=0, num_replicas=1)
        ds = PuzzleDataset(cfg, split="train")
        it = iter(ds)
        item = next(it)
        set_name, batch, global_batch_size = item
        print("loader_ok", set_name, {k: tuple(v.shape) for k, v in batch.items()}, "global_batch_size", global_batch_size)
    except StopIteration:
        print("loader_empty_train_split")
    except Exception as e:
        print("loader_error", repr(e))


if __name__ == "__main__":
    main()


