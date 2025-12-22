# One Model to Grade them All: A comparison of CEFR Classification models 
by Luan Tran
## Overview

This project implements and compares multiple machine learning approaches for automatic CEFR (Common European Framework of Reference for Languages) level classification of English learner texts. The system trains on the large-scale EFCamDAT corpus and evaluates on multiple out-of-domain datasets (Write & Improve, ICNALE, ASAG) to assess cross-corpus generalization.

### Key Features

- **Multi-corpus integration**: Processes and combines four major learner corpora (EFCamDAT, Write & Improve, ICNALE, ASAG)
- **Three model families**: Naive Bayes (traditional ML), Word2Vec (neural embeddings), and RoBERTa (transformer-based)
- **Comprehensive evaluation**: In-domain and out-of-domain testing with multiple metrics (accuracy, QWK, adjacent accuracy, per-class F1)
- **Ablation studies**: Systematic exploration of hyperparameters and architectural choices for each model type
- **Provenance tracking**: Maintains source corpus information throughout the pipeline
- **Visualization tools**: Automated generation of comparison charts, confusion matrices, and performance tables

![methodology.png](docs/images/methodology.png)
**Methodology diagrams**

### Research Questions

1. How do traditional ML, neural embedding, and transformer models compare on CEFR classification?
2. Which model generalizes best to out-of-domain learner corpora?
3. What are the optimal configurations for each model family?
4. How does performance vary across different proficiency levels and source corpora?

## Results Summary

We tested three machine learning approaches for automatically grading English learner writing by CEFR level: Naive Bayes (traditional statistics), Word2Vec (word embeddings), and RoBERTa (deep learning). We trained all models on 80,000 writing samples from EFCamDAT and tested them on the same corpus and on different datasets (Write & Improve, ICNALE, ASAG).

---

### **Main Findings:**

#### **Performance on Training Data (In-Domain)**
- **RoBERTa**: **98.5%** accuracy
- **Word2Vec**: **86.0%** accuracy  
- **Naive Bayes**: **85.0%** accuracy

#### **Performance on New Datasets (Out-of-Domain)**
-  **RoBERTa**: **26.0%** accuracy (**↓72.5%** drop)
-  **Naive Bayes**: **32.9%** accuracy (**↓52.1%** drop)
-  **Word2Vec**: **35.1%** accuracy (**↓50.9%** drop)

#### **Adjacent Accuracy (Within ±1 Level)**
When we allow predictions to be off by one level (e.g., B1 instead of B2):
- **Word2Vec**: **85.2%** on new datasets
- **Naive Bayes**: **84.4%** on new datasets
- **RoBERTa**: **80.9%** on new datasets

The dramatic drops (85-98% → 26-35%) show that complex models like RoBERTa **memorize training data** rather than learn what actually makes writing good or bad. RoBERTa essentially learned specific patterns in the training set—like particular writing prompts and types of learners—instead of general writing quality.

---

### **Key Takeaways**

- **Complex models excel on training data** but fail on new datasets  
- **Simpler models with pre-trained knowledge generalize better**  
- **Adjacent accuracy (80-85%)** is practically useful for real-world applications  
- **Middle proficiency levels (B1, B2)** are easier to classify than extremes  
- **Practical grading systems need**:
  - Training data from multiple sources
  - Focus on basic linguistic features
  - Pre-trained word representations
  - Avoiding overfitting to specific prompts/populations

# Table of Contents

1. [Getting the datasets](#getting-the-datasets)  
   1.1 [The raw datasets](#the-raw-datasets)

2. [Directory Structure](#directory-structure)

3. [Installation](#installation)  
   3.1 [Requirements](#requirements)

4. [Pipeline Overview](#pipeline-overview)

5. [Step-by-Step Execution Guide](#step-by-step-execution-guide)  
   5.1 [Step 1: Extract Individual Corpora](#step-1-extract-individual-corpora)  
   5.2 [Step 2: Combine All Corpora](#step-2-combine-all-corpora)  
   5.3 [Step 3: Create Training Splits](#step-3-create-training-splits)  
   5.4 [Step 4: Run Experiments](#step-4-run-experiments)  
       5.4.1 [Run All Baselines](#4a-run-all-baselines)  
       5.4.2 [Run Naive Bayes Ablations](#4b-run-naive-bayes-ablations)  
       5.4.3 [Run Word2Vec Ablations](#4c-run-word2vec-ablations)  
       5.4.4 [Run BERT/RoBERTa Ablations](#4d-run-bertroberta-ablations)  
   5.5 [Step 5: View output files](#step-5-view-output-files)  
   5.6 [Step 6: Visualize and Compare Results](#step-6-visualize-and-compare-results)

6. [Configuration Options](#configuration-options)  
   6.1 [Common Configuration](#common-configuration-all-models)  
   6.2 [Naive Bayes Configuration](#naive-bayes-configuration)  
   6.3 [Word2Vec Configuration](#word2vec-configuration)  
   6.4 [BERT/RoBERTa Configuration](#bertrorberta-configuration)

7. [Citation](#citation)

---
## Getting the datasets

#### The raw datasets

If you wish to train this yourself, please search the following datasets online:

1. EFCAMDAT dataset: https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html
2. ASAG Louvain dataset: https://cental.uclouvain.be/team/atack/cefr-asag/
3. ICNALE Dataset: https://language.sakura.ne.jp/icnale/download.html
4. Write & Improve Corpus: https://englishlanguageitutoring.com/datasets/write-and-improve-corpus-2024


Please unpack them in the `{project_root}/dataset/`

See the README in the `assets` directory for more information.

## Directory Structure

```
project/
├── src/
│   ├── extractor/              # Data extraction and preprocessing
│   │   ├── efcamdat.py         # EFCamDAT corpus processor
│   │   ├── write_improve_clean.py  # Write & Improve corpus processor
│   │   ├── process_icnale.py   # ICNALE corpus processor
│   │   ├── process_asag.py     # ASAG corpus processor
│   │   └── combine.py          # Corpus combiner with provenance tracking
│   │
│   ├── utils/
│   │   ├── splitter.py         # Subset splitting with stratification
│   │   ├── vizualizer.py       # Result comparison and visualization
│   │   ├── evaluation_utils.py # Metrics computation (QWK, adjacent accuracy, etc.)
│   │   ├── logger.py           # Experiment logging utilities
│   │   └── explorer.py         # Dataset exploration and statistics
│   │
│   ├── models/
│   │   ├── cefr_classifier.py      # Abstract base class for all classifiers
│   │   ├── naive_bayes_classifier.py  # Multinomial NB implementation
│   │   ├── word2vec_classifier.py  # Word2Vec + neural network implementation
│   │   ├── bert_classifier.py      # RoBERTa/BERT implementation
│   │   └── neural_network.py       # PyTorch neural network architectures
│   │
│   └── experiments/
│       ├── baseline.py         # Run all baseline experiments
│       ├── nb_ablative.py      # Naive Bayes ablation studies
│       ├── word2vec_ablative.py  # Word2Vec ablation studies
│       └── bert_ablative.py    # BERT/RoBERTa ablation studies
│
├── assets/                     # Raw corpus files (please refer to the previous section)
│   ├── EFCAMDAT/
│   ├── write-improve/
│   ├── icnale/
│   └── asag/
│
├── dataset/                    # Processed datasets
│   ├── merged/                 # All corpus files
│   └── splits/                 # Stratified and balanced split
│
└── results/                    # Experiment outputs
    ├── Experiment0_NaiveBayes_baseline/
    ├── Experiment0_Word2Vec_baseline/
    ├── Experiment0_RoBERTa_baseline/
    └── comparison_results/
```

---

## Installation

### Requirements

Create whatever virtualenv you need (ie. venv)
```bash
pip install -r requirements.txt
```

---

## Pipeline Overview

The complete pipeline follows this sequence:

**1. DATA PREPARATION**
   - Process and extract data from individual corpora (EFCamDAT, Write & Improve, ICNALE, ASAG)
     - `efcamdat.py`, 
     - `write_improve_clean.py`, 
     - `process_icnale.py`, 
     - `process_asag`.py
   - Merge into single dataset: `combine.py` → `dataset/merged/dataset_merged.csv`
   - Create stratified train/test splits

**2. TRAINING AND EVALUATION**
   - Run baseline experiments: `baseline.py`
   - Run experiments: 
     - `naive_bayes_classifier.py`, 
     - `word2vec_classifier.py`, 
     - `bert_classifier.py`

**3. RESULTS ANALYSIS**
   - Compare models results and visualize performance: `vizualizer.py`
---

## Step-by-Step Execution Guide

### Step 1: Extract Individual Corpora 
(ONLY IF RAW DATASETS ARE AVAILABLE IN `assets/`)

Each extractor processes a raw corpus and outputs a standardized CSV with columns:
`id, native_language, prompt, answer, level, raw_level`

```bash
# run in project root
chmod +x src/extractor/run_extraction.sh
./src/extractor/run_extraction.sh
```

Contents of `run_extraction.sh`
```bash
# Run from project root
if [ -d ".venv" ]; then source .venv/bin/activate; else echo ".venv not found"; fi
python -m src.extractor.process_efcamdat
python -m src.extractor.process_write_improve
python -m src.extractor.process_icnale
python -m src.extractor.process_asag
```

The extracted datsets should be viewable in `{project_root}/dataset`

---

### Step 2: Combine All Corpora 
(ONLY IF EXTRACTED FROM RAW FILES AND INDIVIDUAL .CSV FILES ARE IN `datasets/`)

Merges all extracted CSVs into a single dataset with provenance tracking.

```python
python -m src.extractor.combine
```

**Configuration** (edit in `combine.py`):
```python
input_directory = 'dataset/'
output_file = 'dataset/merged/dataset_merged.csv'
exclude_patterns = ['_all', '_native']  # Skip native speaker files and redundant samples from Write and Improve dataset
merge_c1_c2 = True  # Merge C1 and C2 into 'C1/C2'
```

**Output**: `dataset/merged/dataset_merged.csv`
- Adds `source_file` column for provenance tracking
- Merges C1 and C2 levels into 'C1/C2'

---

### Step 3: Create Training Splits

Creates stratified training sets with class-balanced sampling.

```python
python -m src.utils.splitter
```

**Configuration** (edit in `splitter.py`):
```python
INPUT_FILE = 'dataset/merged/dataset_merged.csv'
OUTPUT_DIR = 'dataset/splits/'
TRAIN_SAMPLES = 100000
TRAIN_CORPUS = 'efcamdat'  # Train only on EFCamDAT
RANDOM_STATE = 6781
```

**Outputs**:
- `dataset/splits/train_100k.csv` - Stratified training samples from EFCamDAT
- `dataset/splits/test_other_corpora.csv` - All non-EFCamDAT samples for OOD evaluation
- `dataset/splits/remaining_samples.csv` - Unused EFCamDAT samples (for Doc2Vec training)

**Insights**:
- Adds `label_numeric` column (A1→0, A2→1, B1→2, B2→3, C1/C2→4)
- Stratifies by level, source file, and topic/prompt

---

### Step 4: Run Experiments

#### 4a. Run All Baselines

```python
python -m src.experiments.baseline
```

Runs all three model baselines sequentially:
1. Naive Bayes (TF-IDF, 5000 features, unigrams+bigrams)
2. Word2Vec (GloVe-300d, mean aggregation, simple NN)
3. RoBERTa (roberta-base, 512 tokens, 4 epochs)

##### Results (please see STEP 6 for how to generate these tables)

![in_domain_test_metrics_table.png](results/comparison_results/all/baseline_comparison/in_domain_test_metrics_table.png)
![test_metrics_table.png](results/comparison_results/all/baseline_comparison/test_metrics_table.png)
![in_domain_test_f1_per_class_table.png](results/comparison_results/all/baseline_comparison/in_domain_test_f1_per_class_table.png)

#### 4b. Run Naive Bayes Ablations

```python
python -m src.experiments.nb_ablative
```

**Parameters** (edit in `nb_ablative.py`):
```python
ablations = [
    {
        'experiment_name': 'Experiment0_NaiveBayes_baseline', 
        'method': 'tfidf', 
        'max_features': 5000, 
        'ngram_range': (1,2), 
        'stop_words':'english'},
        # more ...
]
```

**Experiments (Uncomment or modifiy in file `nb_ablative.py`)**:

| Experiment | Method | Features | N-grams | Stop Words |
|------------|--------|----------|---------|------------|
| Baseline | TF-IDF | 5000 | (1,2) | English |
| No Stopwords | TF-IDF | 5000 | (1,2) | None |
| Unigrams Only | TF-IDF | 5000 | (1,1) | English |
| Bigrams Only | TF-IDF | 5000 | (2,2) | English |
| Count Vec | Count | 5000 | (1,2) | English |
| Large Vocab | TF-IDF | 15000 | (1,2) | English |

##### Results (please see STEP 6 for how to generate these tables)

![in_domain_test_f1_per_class_table.png](results/comparison_results/all/naive_bayes_comparison/in_domain_test_f1_per_class_table.png)
![in_domain_test_metrics_table.png](results/comparison_results/all/naive_bayes_comparison/in_domain_test_metrics_table.png)
![in_domain_test_metrics_table.png](results/comparison_results/all/naive_bayes_comparison/in_domain_test_metrics_table.png)

#### 4c. Run Word2Vec Ablations

```python
python -m src.experiments.word2vec_ablative
```

**Parameters** (edit in `word2vec_ablative.py`):
```python
ablations = [
    {
        'experiment_name': 'Experiment1_Word2Vec_google_news_w2v',
        'embedding_model': 'w2v',
        'embedding_name': 'word2vec-google-news-300',
        'agg_method': 'mean',
        'architecture': 'simple',
        'hidden_dim': 128,
        'epochs': 10,
        'learning_rate': 0.001,
        'batch_size': 64,
    },
    # more ...
]
```

**Experiments** (uncomment in file to enable):

| Experiment | Embeddings | Aggregation | Architecture | Epochs |
|------------|-----------|-------------|--------------|--------|
| Baseline | GloVe-300 | Mean | Simple | 10 |
| Google News | word2vec-google-news-300 | Mean | Simple | 10 |
| Doc2Vec | Trained Doc2Vec | N/A | Simple | 10 |
| Deep Network | GloVe-300 | Mean | Deep (3 layers) | 10 |
| More Epochs | GloVe-300 | Mean | Simple | 20 |

##### Results (please see STEP 6 for how to generate these tables)
![in_domain_test_f1_per_class_table.png](results/comparison_results/all/word2vec_comparison/in_domain_test_f1_per_class_table.png)

![in_domain_test_metrics_table.png](results/comparison_results/all/word2vec_comparison/in_domain_test_metrics_table.png)

![test_metrics_table.png](results/comparison_results/all/word2vec_comparison/test_metrics_table.png)

#### 4d. Run BERT/RoBERTa Ablations

```python
python -m src.experiments.bert_ablative
```

**Parameters** (edit in `bert_ablative.py`):
```python
ablations = [
    {
        'experiment_name': 'Experiment1_DistilRoBERTa',
        'description': 'Distilled model (40% faster, 40% smaller)',
        'model_name': 'distilroberta-base',
        'max_length': 512,
        'batch_size': 32,  # Can use larger batch with smaller model
        'epochs': 7,  # Compensate with more epochs
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'freeze_encoder': False,
    },
    # more ...
]
```

**Experiments** (uncomment in file to enable):

| Experiment | Model | Max Length | Batch | Epochs | LR | Frozen |
|------------|-------|-----------|-------|--------|-----|--------|
| Baseline | roberta-base | 512 | 16 | 4 | 2e-5 | No |
| DistilRoBERTa | distilroberta-base | 512 | 32 | 7 | 2e-5 | No |
| Short Sequences | roberta-base | 256 | 32 | 4 | 2e-5 | No |
| Frozen Encoder | roberta-base | 512 | 32 | 7 | 1e-3 | Yes |
| Careful Tuning | roberta-base | 512 | 16 | 10 | 1e-5 | No |

##### Results (please see STEP 6 for how to generate these tables)

![in_domain_test_f1_per_class_table.png](results/comparison_results/all/bert_comparison_results/in_domain_test_f1_per_class_table.png)

![in_domain_test_metrics_table.png](results/comparison_results/all/bert_comparison_results/in_domain_test_metrics_table.png)

![test_metrics_table.png](results/comparison_results/all/bert_comparison_results/test_metrics_table.png)

---

#### Step 5: View output files

Each experiment generates the following in its output directory:

```
results/Experiment0_ModelName_variant/
├── results.json                      # Complete results in JSON format
├── experiment_summary.txt            # Human-readable summary
├── confusion_matrix_in_domain_test.png
├── confusion_matrix_test.png
├── classification_report_in-domain test.csv
├── classification_report_in-domain test.json
├── classification_report_out-of-domain test.csv
├── classification_report_out-of-domain test.json
├── per_corpus_results.csv            # Accuracy per test corpus
├── cefr_distribution_training.png
├── cefr_distribution_in-domain test.png
└── cefr_distribution_out-of-domain test.png
```

##### Key Metrics in `results.json`

```json
{
  "in_domain_test_results": {
    "accuracy": 0.5123,
    "adjacent_accuracy": 0.8567,
    "qwk": 0.6234,
    "classification_metrics": { ... }
  },
  "test_results": {
    "accuracy": 0.3456,
    "adjacent_accuracy": 0.7234,
    "qwk": 0.4567,
    "classification_metrics": { ... }
  },
  "generalization": {
    "gap": 0.1667,
    "gap_percentage": 16.67
  },
  "per_corpus_results": {
    "write_improve": { "accuracy": 0.35, "samples": 1234 },
    "icnale_we_learners": { "accuracy": 0.42, "samples": 567 }
  }
}
```

### Step 6: Visualize and Compare Results

```python
python -m src.utils.vizualizer
```

**Configuration** (edit in `vizualizer.py`):
```python
# Compare baseline models
json_paths = [
    'results/Experiment0_NaiveBayes_baseline/results.json',
    'results/Experiment0_Word2Vec_baseline/results.json',
    'results/Experiment0_RoBERTa_baseline/results.json',
]
output_dir = 'results/comparison_results/baseline_comparison'
```

**Outputs**:
- `test_metrics_comparison.png` - Bar chart comparing OOD test metrics
- `in_domain_test_metrics_comparison.png` - Bar chart comparing in-domain metrics
- `test_f1_per_class.png` - Per-class F1 scores comparison
- `*_table.png` - Tabular versions with highlighted best values
- `*.csv` - CSV exports for further analysis

---

## Configuration Options

### Common Configuration (all models)

```python
config = {
    'experiment_name': 'Experiment0_ModelName_variant',
    'train_path': 'dataset/splits/train_100k.csv',
    'test_path': 'dataset/splits/test_other_corpora.csv',
    'output_dir': 'results/Experiment0_ModelName_variant',
    'test_size': 0.2,      # 20% of training data for in-domain validation
    'random_state': 42,    # Reproducibility seed
}
```

### Naive Bayes Configuration

```python
config = {
    'method': 'tfidf',           # 'tfidf' or 'count'
    'max_features': 5000,        # Vocabulary size
    'ngram_range': (1, 2),       # Unigrams and bigrams
    'stop_words': 'english',     # None to keep stop words
    'alpha': 1.0,                # Laplace smoothing
}
```

### Word2Vec Configuration

```python
config = {
    'embedding_model': 'w2v',     # 'w2v' or 'doc2vec'
    'embedding_name': 'glove-wiki-gigaword-300',
    'agg_method': 'mean',         # 'mean' or 'tfidf_weighted'
    'architecture': 'simple',     # 'simple' or 'deep'
    'hidden_dim': 128,
    'epochs': 10,
    'learning_rate': 0.001,
    'batch_size': 64,
    'dropout_rate': 0.3,
}
```

### BERT/RoBERTa Configuration

```python
config = {
    'model_name': 'roberta-base',  # or 'distilroberta-base', 'bert-base-uncased'
    'max_length': 512,
    'batch_size': 16,
    'epochs': 4,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'freeze_encoder': False,       # True to only train classification head
}
```

---

## Citation

**EFCamDAT:**
- Geertzen, J., Alexopoulou, T., & Korhonen, A. (2014). Automatic linguistic annotation of large scale L2 databases: The EF-Cambridge Open Language Database (EFCamDat). In R.T. Millar, K.I. Martin, C.M. Eddington, A. Henery, N.M. Miguel, & A. Tseng (Eds.), *Selected proceedings of the 2012 Second Language Research Forum* (pp. 240–254). Somerville, MA: Cascadilla Proceedings Project.


- Huang, Y., Geertzen, J., Baker, R., Korhonen, A., & Alexopoulou, T. (2017). The EF Cambridge Open Language Database (EFCAMDAT): Information for users (pp. 1–18). Retrieved from https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html


- Shatz, I. (2020). Refining and modifying the EFCAMDAT: Lessons from creating a new corpus from an existing large-scale English learner language database. *International Journal of Learner Corpus Research, 6*(2), 220-236. doi:10.1075/ijlcr.20009.sha

**Write & Improve:**
- Nicholls, D., Caines, A., & Buttery, P. (2024). The Write & Improve Corpus 2024: Error-annotated and CEFR-labelled essays by learners of English. Cambridge University Press & Assessment. https://doi.org/10.17863/CAM.112997

**ICNALE:**
- Ishikawa, S. (2023). *The ICNALE Guide: An Introduction to a Learner Corpus Study on Asian Learners*. Routledge. https://www.routledge.com/The-ICNALE-Guide-An-Introduction-to-a-Learner-Corpus-Study-on-Asian-Learners/Ishikawa/p/book/9781032180250


**ASAG:**
- Tack, A., François, T., Roekhaut, S., & Fairon, C. (2017). Human and Automated CEFR-based Grading of Short Answers. In *Proceedings of the 12th Workshop on Innovative Use of NLP for Building Educational Applications* (pp. 169–179). Copenhagen, Denmark: Association for Computational Linguistics. https://doi.org/10.18653/v1/W17-5018