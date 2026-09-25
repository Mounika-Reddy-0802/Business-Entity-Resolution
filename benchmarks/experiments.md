# Experiment log

Every run that changes a number gets a row. `raw/` holds the json each row came from.

`commit` is HEAD when the run started; a trailing `+` means the run included the uncommitted
changes that land in the commit adding the row. `(synthetic)` rows ran on the stand-in dataset
(docs/problems.md) and are not comparable with real-data rows.

| time (IST) | who | commit | change | block recall | cands/S1 | val F0.5 | val F0.5 US | val F0.5 IN | cross-country F0.5 | uploaded? | LB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01:08 | krishna-2-005 | 2285f9a+ | naive exact-name blocker, 4 rapidfuzz features, lgbm, t=0.5 (synthetic) | 0.5397 | 1.3660 | 0.7231 | 0.7454 | 0.7007 |  | no |  |
| 01:13 | krishna-2-005 | c776e2f+ | normalise views + keys k1-k6, cap 40 by best per-field cosine rank (synthetic) | 0.9992 | 39.9770 |  |  |  |  | no |  |
| 01:18 | krishna-2-005 | 50b2c6c+ | 71 features incl. competition, lgbm 5-fold groupkfold, t=0.5 (synthetic) | 0.9992 | 39.9770 | 0.9876 | 0.9900 | 0.9852 |  | no |  |
| 01:21 | krishna-2-005 | b60a65e+ | same blocking on harder synthetic v2 (chains in same city, city-only addresses) (synthetic) | 0.9969 | 39.9855 |  |  |  |  | no |  |
| 01:21 | krishna-2-005 | b60a65e+ | 71 features, lgbm, t=0.5 on harder synthetic v2 (synthetic) | 0.9969 | 39.9855 | 0.9791 | 0.9805 | 0.9776 |  | no |  |
| 01:24 | lahari-66 | 1dbc715+ | decision rules tuned on oof, kept if val rises: t=0.40 +0.0004 kept; per-source -0.0009; alpha 0.9 -0.0016; one-to-one +0.0004 kept; caps 4/3 -0.0001; singleton guard +0.0000 (synthetic) | 0.9969 | 39.9855 | 0.9799 | 0.9812 | 0.9786 |  | no |  |
| 01:26 | lahari-66 | a3a3373+ | train one country, score the other; feature-group ablation (synthetic) |  |  |  |  |  | 0.9663 | no |  |
| 01:31 | lahari-66 | a3a3373+ | drop is_s3 after cross-country ablation (+0.003 both directions) (synthetic) | 0.9969 | 39.9855 | 0.9809 | 0.9806 | 0.9811 |  | no |  |
| 01:35 | lahari-66 | 57859f0+ | tried: one-typo legal suffix detection (reverted, below 0.9809) (synthetic) | 0.9969 | 39.9855 | 0.9801 | 0.9803 | 0.9799 |  | no |  |
| 01:44 | lahari-66 | 3d59196+ | stage-2 lgbm on stage-1 score context (rank, margin, other-s1 best) (synthetic) | 0.9969 | 39.9855 | 0.9838 | 0.9861 | 0.9815 |  | no |  |
| 02:11 | Mounika-Reddy-0802 | 955df2a+ | tried: minilm embeddings as k7 top-20 + 3 cosine features (not kept) (synthetic) | 0.9965 | 40.0000 | 0.9811 | 0.9840 | 0.9781 |  | no |  |
| 02:14 | Mounika-Reddy-0802 | 955df2a+ | tried: minilm cosine features without k7 (not kept) (synthetic) | 0.9969 | 39.9855 | 0.9813 | 0.9859 | 0.9767 |  | no |  |
| 02:25 | Mounika-Reddy-0802 | 6086b5d+ | tried: num_leaves=15 (synthetic) | 0.9969 | 39.9855 | 0.9807 | 0.9829 | 0.9786 |  | no |  |
| 02:26 | Mounika-Reddy-0802 | 6086b5d+ | tried: num_leaves=31 (synthetic) | 0.9969 | 39.9855 | 0.9819 | 0.9828 | 0.9811 |  | no |  |
| 02:28 | Mounika-Reddy-0802 | 6086b5d+ | tried: num_leaves=127 (synthetic) | 0.9969 | 39.9855 | 0.9815 | 0.9833 | 0.9796 |  | no |  |
| 02:29 | Mounika-Reddy-0802 | 6086b5d+ | tried: min_child_samples=5 (synthetic) | 0.9969 | 39.9855 | 0.9815 | 0.9830 | 0.9800 |  | no |  |
| 02:31 | Mounika-Reddy-0802 | 6086b5d+ | tried: min_child_samples=50 (synthetic) | 0.9969 | 39.9855 | 0.9819 | 0.9840 | 0.9798 |  | no |  |
| 02:33 | Mounika-Reddy-0802 | 6086b5d+ | tried: min_child_samples=200 (synthetic) | 0.9969 | 39.9855 | 0.9793 | 0.9816 | 0.9771 |  | no |  |
| 02:34 | Mounika-Reddy-0802 | 6086b5d+ | tried: scale_pos_weight=0.5 (synthetic) | 0.9969 | 39.9855 | 0.9831 | 0.9855 | 0.9807 |  | no |  |
| 02:36 | Mounika-Reddy-0802 | 6086b5d+ | tried: scale_pos_weight=2.0 (synthetic) | 0.9969 | 39.9855 | 0.9830 | 0.9865 | 0.9795 |  | no |  |
| 02:38 | Mounika-Reddy-0802 | 6086b5d+ | tried: learning_rate=0.02 (synthetic) | 0.9969 | 39.9855 | 0.9842 | 0.9869 | 0.9814 |  | no |  |
| 02:40 | Mounika-Reddy-0802 | 6086b5d+ | tried: learning_rate=0.1 (synthetic) | 0.9969 | 39.9855 | 0.9796 | 0.9791 | 0.9800 |  | no |  |
| 02:42 | Mounika-Reddy-0802 | 6086b5d+ | tried: feature_fraction=0.5 (synthetic) | 0.9969 | 39.9855 | 0.9815 | 0.9833 | 0.9796 |  | no |  |
| 02:43 | Mounika-Reddy-0802 | 6086b5d+ | tried: feature_fraction=1.0 (synthetic) | 0.9969 | 39.9855 | 0.9802 | 0.9825 | 0.9778 |  | no |  |
| 02:45 | Mounika-Reddy-0802 | 6086b5d+ | tried: lambda_l2=0.0 (synthetic) | 0.9969 | 39.9855 | 0.9829 | 0.9862 | 0.9796 |  | no |  |
| 02:47 | Mounika-Reddy-0802 | 6086b5d+ | tried: lambda_l2=10.0 (synthetic) | 0.9969 | 39.9855 | 0.9816 | 0.9846 | 0.9787 |  | no |  |
| 02:52 | Mounika-Reddy-0802 | 6086b5d+ | learning rate 0.05 -> 0.02 (best of 14 one-change variants) (synthetic) | 0.9969 | 39.9855 | 0.9842 | 0.9869 | 0.9814 |  | no |  |
| 05:44 | Rayyan-Mohammed | f3273d9+ | 5 lightgbm seeds, fixed split: mean val f0.5, std 0.0012 (synthetic) |  |  | 0.9831 |  |  |  | no |  |
| 05:57 | Rayyan-Mohammed | 5dd63bc+ | tried: learned abbreviation maps, 5 lightgbm seeds, fixed split: mean val f0.5, std 0.0014 (vs 0.9831 without; not kept) (synthetic) |  |  | 0.9837 |  |  |  | no |  |
| 07:54 | Rayyan-Mohammed | 62c61d4+ | learning rate back to 0.05: 5 lightgbm seeds, fixed split: mean val f0.5, std 0.0007 (0.9831 at 0.02; same within noise, 3x faster) (synthetic) |  |  | 0.9829 |  |  |  | no |  |
