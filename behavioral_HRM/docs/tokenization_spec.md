## Tokenization Specification (EVENT_FILTERED → HRM)

Columns: `idanimalA..D`, `name`, `duration_seconds`, `event_start_datetime`, `GENOTYPE_A..D`, `SETUP_A..D`.

Vocabulary:
- Special: `PAD=0`, `EOS=1`
- Individual tokens; dyadic tokens with `_ACT`/`_PAS`
- Duration side-tokens: `DUR_BIN_[1..K]`
- Window context: `SETUP_*`, `SELF_*`, `DAY|NIGHT`, `HOUR_BIN_*`
- Optional dyad side-token: `PARTNER_[WT|TG]`

Duration bins (sec): <=0.37, <=0.70, <=1.57, <=3.80, <=9.17, <=16.57, <=64.40, >64.40.

Labels: next-token; PAD ignored.

