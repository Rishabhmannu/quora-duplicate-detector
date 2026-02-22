# Scripts

Python scripts migrated from notebooks. Run from **project root**.

## Usage

```bash
conda activate quora-project

# 1. EDA (output -> outputs/01_eda_output.txt)
python scripts/01_eda.py
python scripts/01_eda.py --plots

# 2. Baseline (output -> outputs/02_baseline_bow_output.txt)
python scripts/02_baseline_bow.py

# 3. BoW + 7 features (output -> outputs/03_bow_basic_features_output.txt)
python scripts/03_bow_basic_features.py

# 4. Full pipeline with embeddings (output -> outputs/04_train_and_save_output.txt)
python scripts/04_train_and_save.py              # 30K samples
python scripts/04_train_and_save.py --quick     # 5K samples
python scripts/04_train_and_save.py --bow       # BoW instead of TF-IDF
python scripts/04_train_and_save.py --no-embeddings  # Skip Sentence Transformers

# 5. Fine-tune DistilBERT (output -> outputs/05_train_transformer_output.txt)
python scripts/05_train_transformer.py           # 50K samples
python scripts/05_train_transformer.py --quick   # 5K samples (quick test)
python scripts/05_train_transformer.py --full    # Full 404K dataset (~2–4 hrs)

# 6. Benchmark inference times (saves models/inference_times.json)
python scripts/06_benchmark_inference.py
```

## Improvements (per project-plan.md)

- **Sentence Transformer embeddings** (all-MiniLM-L6-v2, MPS on Apple Silicon)
- **25 features**: 24 handcrafted + embedding cosine similarity
- **TF-IDF**, Stratified 5-Fold CV, full metrics
- **tqdm** progress bars for feature extraction and CV folds
- **Outputs** saved to `outputs/*.txt`

## Data

Expects `data/train.csv`. Scripts drop rows with missing question1/question2.

## Outputs

- All scripts write to `outputs/<script>_output.txt`
- `04_train_and_save.py` writes `models/model.pkl`, `models/cv.pkl`
- `05_train_transformer.py` writes `models/transformer/` (tokenizer + model)
- `06_benchmark_inference.py` writes `models/inference_times.json`
