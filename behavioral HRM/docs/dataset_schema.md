## HRM Dataset Schema (behavior)

Root: `data/behavior-v1/{train,val,test}`

Files:
- `all__inputs.npy`  int32 [N, L]
- `all__labels.npy`  int32 [N, L]
- `all__puzzle_identifiers.npy` int32 [num_puzzles]
- `all__puzzle_indices.npy`     int32 [num_puzzles+1]
- `all__group_indices.npy`      int32 [num_groups+1]
- `dataset.json` with:
  - `pad_id: 0`, `ignore_label_id: 0`, `blank_identifier_id: 0`
  - `vocab_size`, `seq_len`
  - `num_puzzle_identifiers`, `total_groups`, `mean_puzzle_examples`
  - `sets: ["all"]`

Semantics:
- Puzzle = animal-day-condition
- Group  = experiment-day-condition

