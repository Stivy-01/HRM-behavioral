## What this project does (in simple terms)

We turn long mouse behavior logs into sequences of tokens and train a model (HRM) that learns patterns over time, so we can see how stress and genotype affect behavior.

## The data
- Events like “Approach”, “Get away”, “Contact”, “Move”, “Stop”, “Center zone”, “Periphery zone”.
- Some events involve one mouse; some involve two (we know who is active/passive).
- Each event has a timestamp (day/night) and duration.

## What we feed the model
- Tokens per event; dyadic events get role suffix (ACT/PAS).
- Duration tags after each event (short to long bins).
- Window context (once): SETUP (baseline/stress), SELF genotype, DAY/NIGHT, HOUR_BIN.

## What the model learns
- Short patterns (e.g., “Approach → Contact → Get away”).
- Longer modes (e.g., avoidance vs exploration), and differences across baseline/stress, day/night, WT/TG.

## Outputs
- Prediction quality (perplexity, accuracy).
- Latent states and transitions.
- Posterior checks (frequencies, dwell times).

## Why zones matter
- Periphery → stress-like behavior; Center → exploration. Keeping these tokens helps the model see stress shifts.

