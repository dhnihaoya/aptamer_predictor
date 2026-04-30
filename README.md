# Aptamer-Small Molecule Interaction Predictor

An ensemble learning tool for predicting whether a DNA/RNA aptamer binds to a specific small molecule target. Given an aptamer sequence and a SMILES string, the tool outputs a binding probability via 9-model ensemble voting.

## Pipeline

```
Aptamer sequence ──→ k-mer frequency features ──┐
                                                  ├─→ 9-model ensemble vote ──→ Prediction
Small molecule SMILES ──→ 209 RDKit descriptors ──┘
```

## Setup

### 1. Create a conda environment

```bash
conda create -n aptamer-pred python=3.9 -y
conda deactivate （if base is activated）
conda activate aptamer-pred
```

### 2. Install RDKit

```bash
conda install -c conda-forge rdkit=2023.9.5 -y
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note:** install the pinned RDKit version above; descriptor values are part of the trained model input schema. `rdkit` should be installed via conda, not pip. All other packages can be installed via pip.

### 4. Verify installation

```bash
python -m aptamer_predictor predict --help
python -m aptamer_predictor --tui --help
```

## Usage

### Model directory resolution

Model files are resolved in this order:

1. `--model-dir`
2. `APTAMER_MODEL_DIR`
3. `./models`

Examples:

```bash
python -m aptamer_predictor predict --model-dir ./models --aptamer "ATGC" --smiles "CCO"

export APTAMER_MODEL_DIR=/absolute/path/to/models
python -m aptamer_predictor --tui
```

### Single prediction

Predict interaction for one aptamer-SMILES pair:

```bash
python -m aptamer_predictor predict \
    --aptamer "GGGAGAAUUCCCGCGGCAGAAGCCCACCUGGCUUUGAACUCUAUGUUAUUGGGUGGGGGAAACUUAAGAAAACUACCACCCUUCAACAUUACCGCCCUUCAGCCUGCCAGCGCCCUGCAGCCCGGGAAGCUU" \
    --smiles "C1=CC=C2C(=C1)C(=O)C3=C(C2=O)C(=C(C=C3NC4=CC(=C(C=C4)S(=O)(=O)O)NC5=NC(=NC(=N5)Cl)Cl)S(=O)(=O)O)N"
```

RNA sequences are handled automatically (U is converted to T internally).

### Batch prediction

Prepare a CSV file with `sequence` and `smiles` columns (optionally `label` and `id`):

```csv
sequence,smiles
GGGAGAAUUCCCGCGG...,C1=CC=C2C(=C1)...
GUCGGCCUAUCCGACAG...,CC1=CC=C(C=C1)...
```

Run:

```bash
python -m aptamer_predictor predict \
    --input pairs.csv \
    --output predictions.csv
```

Output files:
- `predictions.csv` — summary results (`sequence`, `smiles`, `ensemble_label`)

### Interactive mutation search (TUI)

Launch the Textual interface:

```bash
python -m aptamer_predictor --tui
```

Or start it programmatically:

```python
from aptamer_predictor.tui.app import run_tui

run_tui()  # uses ./models by default
```

Workflow:
- Enter an aptamer sequence and target molecule (SMILES or resolvable molecule name via PubChem)
- Optionally set the CSV filename for exported hits
- Select mutation sites interactively
- Start exhaustive mutation prediction for the selected sites
- Stream only positive ensemble hits to CSV while showing progress, hit count, and ranked results

Notes:
- `run_tui()` and CLI now share the same default model directory resolution and both fall back to `<repo>/models`
- Molecule-name input requires network access because it resolves names through the PubChem REST API; direct SMILES input works offline
- Pressing `New Search` during an active run cancels the current background search before returning to the input screen
- The mutation search is exhaustive over the selected positions, so search space grows as `4^n`

### TUI acceleration strategy

`--tui` is optimized for exhaustive mutation search rather than generic batch prediction:

- The target molecule descriptors are computed once per run and reused for every mutant
- Mutants are enumerated in large NumPy batches instead of one sequence at a time
- k-mer features are built with vectorized batch logic (`build_feature_matrix`) rather than Python loops per candidate
- Model order is calibrated on a small sample, then the most selective models run first
- Candidates are filtered with early exit: once a mutant fails one model, later models are skipped for that mutant
- Positive hits are streamed directly to CSV (`collect_results=False`) instead of keeping the full search space in memory
- Progress updates are chunked, and the prediction loop runs in a Textual worker thread so the UI stays responsive
- If CUDA is available, PyTorch RNN models are moved to GPU and XGBoost uses its GPU prediction path when possible

For an in-depth technical explanation of each optimization, see [MUTATION_SEARCH_ACCELERATION.md](MUTATION_SEARCH_ACCELERATION.md).

### Model evaluation

Evaluate models on pre-formatted test data with labels:

```bash
python -m aptamer_predictor evaluate \
    --data-dir ./independent_data\ all\ des/ \
    --output-dir ./predictions/
```

Output:
- `all_models_metrics.csv` — AUC, Accuracy, F1, Precision, Recall, Specificity for each model
- Per-model prediction CSVs

### Feature extraction (standalone)

Extract k-mer frequency features from aptamer sequences:

```bash
python -m aptamer_predictor extract-aptamer \
    --input aptamer_sequences.csv \
    --output aptamer_features.csv \
    --k 1 2 3 4
```

Extract molecular descriptors from SMILES:

```bash
python -m aptamer_predictor extract-molecule \
    --input targets.csv \
    --output molecule_features.csv
```

## Input Format

### Batch prediction CSV

| Column     | Required | Description                              |
|------------|----------|------------------------------------------|
| `sequence` | Yes      | Aptamer sequence (RNA or DNA)            |
| `smiles`   | Yes      | Small molecule SMILES string             |
| `label`    | No       | True label (0/1), for comparison         |
| `id`       | No       | Sample identifier                        |

Column names are case-insensitive and support aliases (e.g., `aptamer`, `Sequence`, `SMILES`).

## Ensemble Models

The tool includes 9 pre-trained models. The current ensemble decision rule is strict consensus: a sample is labeled binding only when all loaded models predict binding (`ensemble_label == 1`).

| Model            | Type           | k-mer combo | Feature dim |
|------------------|----------------|-------------|-------------|
| 1mer-XGB         | XGBoost        | 1           | 213         |
| 2mer-RF          | Random Forest  | 2           | 225         |
| 4mer-RF          | Random Forest  | 4           | 465         |
| 4mer-XGB         | XGBoost        | 4           | 465         |
| 23mer-DT         | Decision Tree  | 2 + 3       | 289         |
| 24mer-XGB        | XGBoost        | 2 + 4       | 481         |
| 1234mer-RF       | Random Forest  | 1 + 2 + 3 + 4 | 549      |
| 123mer-biRNN     | PyTorch biRNN  | 1 + 2 + 3   | 293         |
| 124mer-RNN       | PyTorch RNN    | 1 + 2 + 4   | 485         |

Feature dim = sum(4^k for each k) + 209 RDKit descriptors (Ipc excluded).

## Project Structure

```
.
├── aptamer_predictor/          # CLI + TUI package
│   ├── __init__.py
│   ├── __main__.py             # python -m entry point
│   ├── cli.py                  # Argument parsing & subcommands
│   ├── features.py             # k-mer and molecular descriptor extraction
│   ├── paths.py                # Shared model path resolution
│   ├── predictor.py            # Model loading & ensemble prediction
│   └── tui/                    # Textual mutation-search interface
├── models/                     # Pre-trained models (.pkl)
│   ├── (1mer)(Dataset N1)XGB.pkl
│   ├── ...
│   └── (123mer)(Dataset N9)biRNN.pkl
├── data/                       # Training source data
│   ├── Non-redundant aptamer sequences.csv
│   └── Non-redundant target.csv
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Tested Dependency Versions

| Package      | Version |
|--------------|---------|
| Python       | 3.9.18  |
| scikit-learn | 1.5.0   |
| XGBoost      | 2.0.3   |
| PyTorch      | 1.12.1  |
| RDKit        | 2023.9.5|
| NumPy        | 1.24.3  |
