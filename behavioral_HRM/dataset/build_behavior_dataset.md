Behavior → HRM dataset contract

Inputs come from EVENT_FILTERED tables (merged bouts) and are written in the same array contract used by PuzzleDataset:

- inputs.npy: int32 [N, L] — tokenized sequences (see tokenization spec)
- labels.npy: int32 [N, L] — next-token labels, PAD positions set to ignore_label_id
- puzzle_identifiers.npy: int32 [num_puzzles] — identifier per logical "puzzle" (animal-day-condition)
- puzzle_indices.npy: int32 [num_puzzles+1] — cumulative starts into examples for each puzzle
- group_indices.npy: int32 [num_groups+1] — cumulative starts for groups (experiment-day-condition)
- dataset.json: metadata

Metadata constraints:

- pad_id: 0, ignore_label_id: 0, blank_identifier_id: 0
- vocab_size: as produced by tokenizer
- seq_len: fixed window length L (e.g., 1024)
- num_puzzle_identifiers ≥ 1
- sets: ["all"]

Labels are next-token; EOS is placed at the last valid position before padding. PAD labels are ignored by the training loop via ignore_label_id mapping.

Packing rules:

- A "puzzle" is an animal-day-condition; windows from the same animal-day share identifier.
- A "group" is experiment-day-condition; groups keep cagemates together for shuffling.

Training config:

Use config/cfg_behavior_pretrain.yaml with arch/hrm_behavior.yaml (puzzle_emb_ndim=0).

