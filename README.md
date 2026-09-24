# Business Entity Resolution — Amazon ML Challenge 2026

Match Source 2 / Source 3 business records to the deduplicated Source 1 reference using only the
provided name, address and country fields. Scored on macro F0.5.

## Reproduce end to end

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the organiser dataset at data/raw/dataset/{train,test}/ and utils/validate_submission.py at utils/
bash scripts/run_pipeline.sh
```

Outputs land in `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Pipeline stages and file contracts: `docs/contracts.md`. Results: `benchmarks/experiments.md`.
