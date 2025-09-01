## HRM Integration Plan (Behavior Sequences)

### Pipeline (v1)
1) Tokenization: individual/dyadic with roles; duration bins; context tokens (setup, genotype, day/night, hour bin).
2) Windowing: L=1024, S=512.
3) Dataset: HRM numpy format with puzzle/group packing.
4) Training: `arch/hrm_behavior.yaml`, `cfg_behavior_pretrain.yaml`.
5) Evaluation: predictive metrics; baselines; latent analysis; PPC.

