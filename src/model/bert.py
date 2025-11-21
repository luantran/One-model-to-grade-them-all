"""
CEFR Level Classifier using Fine-tuned RoBERTa (HuggingFace Trainer)
Corpus-based evaluation: Train on EFCamDAT, test on other corpora for generalization
Transformer-based approach for ordinal CEFR classification (A1-C2)

Follows HuggingFace tutorial: https://huggingface.co/docs/transformers/training

Key approach:
1. Load pre-trained RoBERTa-base from HuggingFace
2. Fine-tune on EFCamDAT training data using Trainer API
3. Evaluate on in-domain validation (EFCamDAT held-out)
4. Evaluate on out-of-domain test (other corpora)
5. Analyze generalization gap
"""

import os
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import warnings

# Import common utilities
from src.utils.data_utils import load_dataset as load_csv_dataset

warnings.filterwarnings('ignore')

def log(message):
    """Print a formatted log message."""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)


def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize texts for classification.

    Parameters:
    -----------
    examples : dict
        Batch of examples with 'text' key
    tokenizer : AutoTokenizer
        Tokenizer
    max_length : int
        Maximum sequence length

    Returns:
    --------
    dict : Tokenized inputs
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)


def compute_metrics(eval_pred):
    """
    Compute metrics for Trainer evaluation.

    Parameters:
    -----------
    eval_pred : tuple
        (logits, labels)

    Returns:
    --------
    dict : Metrics dictionary
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ============================================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================================

def run_experiment(train_path, test_path, output_dir='../results',
                   model_name='roberta-base',
                   max_length=512, batch_size=16, epochs=4,
                   learning_rate=2e-5, weight_decay=0.01,
                   val_size=0.2, random_state=42):
    """
    Run complete RoBERTa CEFR classification experiment with corpus-based evaluation.

    Parameters:
    -----------
    train_path : str
        Path to training dataset CSV (EFCamDAT samples)
    test_path : str
        Path to test dataset CSV (other corpora)
    output_dir : str
        Results directory
    model_name : str
        Pre-trained model name
    max_length : int
        Maximum sequence length
    batch_size : int
        Batch size
    epochs : int
        Training epochs
    learning_rate : float
        Learning rate
    weight_decay : float
        Weight decay for optimizer
    val_size : float
        Validation set proportion (from training data)
    random_state : int
        Random seed

    Returns:
    --------
    dict : Results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ===================================================================
    # STEP 1: Load training data (EFCamDAT)
    # ===================================================================
    log("STEP 1/9: LOADING TRAINING DATA (EFCamDAT)")
    df_train_full = load_csv_dataset(train_path)
    print(f"Total training samples: {len(df_train_full):,}")
    print(f"Level distribution:")
    for level, count in df_train_full['level'].value_counts().sort_index().items():
        pct = (count / len(df_train_full)) * 100
        print(f"  {level}: {count:,} ({pct:.1f}%)")

    # ===================================================================
    # STEP 2: Split training data into train/val
    # ===================================================================
    log("STEP 2/9: SPLITTING TRAINING DATA")
    print(f"Validation size: {val_size * 100}%")

    X_full = df_train_full['answer']
    y_full = df_train_full['level']

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_full
    )

    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")

    # ===================================================================
    # STEP 3: Load test data (other corpora)
    # ===================================================================
    log("STEP 3/9: LOADING TEST DATA (Other Corpora)")
    df_test = load_csv_dataset(test_path)

    X_test = df_test['answer']
    y_test = df_test['level']

    print(f"Test set: {len(X_test):,} samples")
    print(f"Test corpora: {list(df_test['source_file'].unique())}")

    # ===================================================================
    # STEP 4: Create label mapping
    # ===================================================================
    log("STEP 4/9: PREPARING DATASETS")

    # Label mapping: CEFR → numeric (5 classes with merged C1/C2)
    label_map = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1/C2': 4}
    label_map_inv = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1/C2'}

    # Convert to numeric labels
    y_train_numeric = [label_map[label] for label in y_train.values]
    y_val_numeric = [label_map[label] for label in y_val.values]
    y_test_numeric = [label_map[label] for label in y_test.values]

    # Create Datasets (following HUggingFace tutorial pattern)
    # https://huggingface.co/docs/transformers/training#train-with-pytorch-trainer
    train_dataset = Dataset.from_dict({
        "text": X_train.tolist(),
        "label": y_train_numeric
    })

    val_dataset = Dataset.from_dict({
        "text": X_val.tolist(),
        "label": y_val_numeric
    })

    test_dataset = Dataset.from_dict({
        "text": X_test.tolist(),
        "label": y_test_numeric
    })

    print(f"✓ Datasets created")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val: {len(val_dataset):,} samples")
    print(f"  Test: {len(test_dataset):,} samples")

    # ===================================================================
    # STEP 5: Load tokenizer and tokenize datasets
    # ===================================================================
    log("STEP 5/9: TOKENIZER & PREPROCESSING")
    print(f"Model: {model_name}")
    print(f"Max length: {max_length}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize datasets (following tutorial pattern with map)
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True
    )

    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True
    )

    tokenized_test = test_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True
    )

    print("✓ Tokenization complete")

    small_train = tokenized_train.shuffle(seed=42).select(range(1000))
    small_eval = tokenized_val.shuffle(seed=42).select(range(1000))
    small_test = tokenized_test.shuffle(seed=42).select(range(1000))

    # ===================================================================
    # STEP 6: Load model
    # ===================================================================
    log("STEP 6/9: LOADING MODEL")

    # Load model with number of labels (following tutorial pattern)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model loaded")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ===================================================================
    # STEP 7: Setup training arguments
    # ===================================================================
    log("STEP 7/9: TRAINING ARGUMENTS")

    # Following tutorial pattern for TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=random_state
    )

    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  FP16: {training_args.fp16}")

    # ===================================================================
    # STEP 8: Create Trainer and train
    # ===================================================================
    log("STEP 8/9: TRAINER SETUP & TRAINING")

    # Create Trainer (following tutorial pattern)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train,
        eval_dataset=small_eval,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()
    print("✓ Training complete")

    # # ===================================================================
    # # STEP 9: Evaluate on validation set (in-domain)
    # # ===================================================================
    # log("EVALUATION ON VALIDATION SET (In-Domain EFCamDAT)")
    #
    # # Get predictions
    # val_predictions = trainer.predict(tokenized_val)
    # val_pred_numeric = np.argmax(val_predictions.predictions, axis=1)
    # val_pred_labels = np.array([label_map_inv[p] for p in val_pred_numeric])
    #
    # # Get probabilities
    # val_probs = torch.softmax(torch.tensor(val_predictions.predictions), dim=1).numpy()
    #
    # # Compute metrics
    # val_acc = accuracy_score(y_val_numeric, val_pred_numeric)
    # val_metrics = compute_all_metrics(y_val, val_pred_labels, val_probs)
    #
    # print(f"\n✓ Validation Accuracy: {val_acc:.4f}")
    # print_metrics(val_metrics, train_acc=None)
    #
    # # ===================================================================
    # # STEP 10: Evaluate on test set (out-of-domain)
    # # ===================================================================
    # log("EVALUATION ON TEST SET (Out-of-Domain Other Corpora)")
    #
    # # Get predictions
    # test_predictions = trainer.predict(tokenized_test)
    # test_pred_numeric = np.argmax(test_predictions.predictions, axis=1)
    # test_pred_labels = np.array([label_map_inv[p] for p in test_pred_numeric])
    #
    # # Get probabilities
    # test_probs = torch.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy()
    #
    # # Compute metrics
    # test_acc = accuracy_score(y_test_numeric, test_pred_numeric)
    # test_metrics = compute_all_metrics(y_test, test_pred_labels, test_probs)
    #
    # print(f"\n✓ Test Accuracy: {test_acc:.4f}")
    # print_metrics(test_metrics, train_acc=None)
    #
    # # Calculate generalization gap
    # generalization_gap = val_acc - test_acc
    # print(f"\n✓ Generalization Gap: {generalization_gap:.4f} ({generalization_gap * 100:.2f}%)")
    #
    # # ===================================================================
    # # STEP 11: Generate visualizations
    # # ===================================================================
    # log("GENERATING VISUALIZATIONS")
    #
    # # Validation confusion matrix
    # val_cm_path = os.path.join(output_dir, 'confusion_matrix_validation.png')
    # plot_confusion_matrix(y_val, val_pred_labels, output_path=val_cm_path,
    #                       title=f'Confusion Matrix - Validation ({model_name})')
    #
    # # Test confusion matrix
    # test_cm_path = os.path.join(output_dir, 'confusion_matrix_test.png')
    # plot_confusion_matrix(y_test, test_pred_labels, output_path=test_cm_path,
    #                       title=f'Confusion Matrix - Test ({model_name})')
    #
    # # ===================================================================
    # # STEP 12: Error analysis
    # # ===================================================================
    # log("ERROR ANALYSIS")
    #
    # # Validation errors
    # val_error_path = os.path.join(output_dir, 'misclassified_validation.csv')
    # df_val_for_analysis = pd.DataFrame({
    #     'answer': X_val.values,
    #     'level': y_val.values
    # })
    # val_error_df = analyze_errors(X_val, y_val, val_pred_labels,
    #                               df_val_for_analysis, output_path=val_error_path)
    #
    # # Test errors
    # test_error_path = os.path.join(output_dir, 'misclassified_test.csv')
    # df_test_for_analysis = df_test[['answer', 'level', 'source_file']].copy()
    # test_error_df = analyze_errors(X_test, y_test, test_pred_labels,
    #                                df_test_for_analysis, output_path=test_error_path)
    #
    # # ===================================================================
    # # STEP 13: Per-corpus analysis
    # # ===================================================================
    # log("PER-CORPUS ANALYSIS (Test Set)")
    #
    # corpus_results = {}
    # for corpus in df_test['source_file'].unique():
    #     corpus_mask = df_test['source_file'] == corpus
    #     corpus_y_true = y_test[corpus_mask].values
    #     corpus_y_pred = test_pred_labels[corpus_mask]
    #
    #     corpus_acc = (corpus_y_true == corpus_y_pred).mean()
    #     corpus_results[corpus] = {
    #         'accuracy': corpus_acc,
    #         'samples': len(corpus_y_true)
    #     }
    #
    #     print(f"\n{corpus}:")
    #     print(f"  Samples: {len(corpus_y_true):,}")
    #     print(f"  Accuracy: {corpus_acc:.4f}")
    #
    # # ===================================================================
    # # STEP 14: Save comprehensive results
    # # ===================================================================
    # log("SAVING RESULTS")
    #
    # summary_path = os.path.join(output_dir, 'experiment_summary.txt')
    # with open(summary_path, 'w') as f:
    #     f.write("EXPERIMENT: ROBERTA CEFR - CORPUS-BASED EVALUATION\n")
    #     f.write("=" * 70 + "\n\n")
    #
    #     f.write("DATASETS:\n")
    #     f.write(f"  Training: {train_path}\n")
    #     f.write(f"  Test: {test_path}\n\n")
    #
    #     f.write("DATA SPLITS:\n")
    #     f.write(f"  Training: {len(X_train):,} samples (EFCamDAT)\n")
    #     f.write(f"  Validation: {len(X_val):,} samples (EFCamDAT held-out)\n")
    #     f.write(f"  Test: {len(X_test):,} samples (Other corpora)\n\n")
    #
    #     f.write("MODEL:\n")
    #     f.write(f"  Architecture: {model_name}\n")
    #     f.write(f"  Total parameters: {total_params:,}\n")
    #     f.write(f"  Max sequence length: {max_length}\n\n")
    #
    #     f.write("TRAINING PARAMETERS:\n")
    #     f.write(f"  Epochs: {epochs}\n")
    #     f.write(f"  Batch size: {batch_size}\n")
    #     f.write(f"  Learning rate: {learning_rate}\n")
    #     f.write(f"  Weight decay: {weight_decay}\n")
    #     f.write(f"  Optimizer: AdamW (via Trainer)\n")
    #     f.write(f"  FP16: {training_args.fp16}\n\n")
    #
    #     f.write("=" * 70 + "\n")
    #     f.write("VALIDATION RESULTS (In-Domain EFCamDAT)\n")
    #     f.write("=" * 70 + "\n\n")
    #     f.write(f"Accuracy: {val_acc:.4f}\n")
    #     f.write(f"Adjacent Accuracy (±1): {val_metrics['adjacent_accuracy']:.4f}\n")
    #     f.write(f"MAE: {val_metrics['mae']:.4f}\n")
    #     f.write(f"QWK: {val_metrics['qwk']:.4f}\n")
    #     f.write(f"OCA: {val_metrics['oca']:.4f}\n")
    #     f.write(f"EMD: {val_metrics['emd']:.4f}\n\n")
    #
    #     f.write("=" * 70 + "\n")
    #     f.write("TEST RESULTS (Out-of-Domain Other Corpora)\n")
    #     f.write("=" * 70 + "\n\n")
    #     f.write(f"Accuracy: {test_acc:.4f}\n")
    #     f.write(f"Adjacent Accuracy (±1): {test_metrics['adjacent_accuracy']:.4f}\n")
    #     f.write(f"MAE: {test_metrics['mae']:.4f}\n")
    #     f.write(f"QWK: {test_metrics['qwk']:.4f}\n")
    #     f.write(f"OCA: {test_metrics['oca']:.4f}\n")
    #     f.write(f"EMD: {test_metrics['emd']:.4f}\n\n")
    #
    #     f.write("=" * 70 + "\n")
    #     f.write("GENERALIZATION ANALYSIS\n")
    #     f.write("=" * 70 + "\n\n")
    #     f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    #     f.write(f"Test Accuracy: {test_acc:.4f}\n")
    #     f.write(f"Generalization Gap: {generalization_gap:.4f}\n\n")
    #
    #     f.write("Per-Corpus Results:\n")
    #     for corpus, results in sorted(corpus_results.items()):
    #         f.write(f"  {corpus}: {results['accuracy']:.4f} ({results['samples']:,} samples)\n")
    #
    # print(f"\n✓ Summary saved to: {summary_path}")
    #
    # # ===================================================================
    # # STEP 15: Return results
    # # ===================================================================
    # results = {
    #     'model': model,
    #     'tokenizer': tokenizer,
    #     'trainer': trainer,
    #     'validation': {
    #         'y_true': y_val,
    #         'y_pred': val_pred_labels,
    #         'accuracy': val_acc,
    #         'metrics': val_metrics
    #     },
    #     'test': {
    #         'y_true': y_test,
    #         'y_pred': test_pred_labels,
    #         'accuracy': test_acc,
    #         'metrics': test_metrics,
    #         'corpus_results': corpus_results
    #     },
    #     'generalization_gap': generalization_gap,
    #     'parameters': {
    #         'model_name': model_name,
    #         'max_length': max_length,
    #         'epochs': epochs,
    #         'batch_size': batch_size,
    #         'learning_rate': learning_rate,
    #         'weight_decay': weight_decay
    #     }
    # }
    #
    # log("EXPERIMENT COMPLETE")
    # print(f"\nKey Results:")
    # print(f"  Validation Accuracy: {val_acc:.4f}")
    # print(f"  Test Accuracy: {test_acc:.4f}")
    # print(f"  Generalization Gap: {generalization_gap:.4f}")
    #
    # return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    samples = "150k"
    # Data paths
    TRAIN_PATH = f'../dataset/splits/train_{samples}.csv'
    TEST_PATH = '../../dataset/splits/test_other_corpora.csv'

    # Model selection
    MODEL_NAME = 'roberta-base'  # or 'roberta-large', 'distilroberta-base'

    # Model parameters
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01

    # Training parameters
    VAL_SIZE = 0.2  # 20% of training data for validation
    RANDOM_STATE = 42

    # Output directory
    OUTPUT_DIR = f'../results/RoBERTa_corpus_{samples}_EPOCH_{EPOCHS}_BS_{BATCH_SIZE}_LR_{LEARNING_RATE}'

    # ========== RUN EXPERIMENT ==========

    print("=" * 80)
    print("ROBERTA - CORPUS-BASED GENERALIZATION EXPERIMENT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Training data: {TRAIN_PATH}")
    print(f"  Test data: {TEST_PATH}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 80)

    # Run experiment
    results = run_experiment(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED SUCCESSFULLY")
    print("=" * 80)