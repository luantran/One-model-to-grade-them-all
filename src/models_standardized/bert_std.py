"""
RoBERTa CEFR Classifier - Standardized Version
===============================================
Corpus-based evaluation: Train on EFCamDAT, test on other corpora for generalization

Standardized function names:
- load_data()
- prepare_features()
- train_model()
- evaluate_model()
- run_experiment()
"""

import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

# Import common utilities
from src.utils.data_utils import load_dataset as load_csv
from src.utils.evaluation_utils import (
    compute_all_metrics, print_metrics, plot_confusion_matrix, analyze_errors
)

warnings.filterwarnings('ignore')


def log(message):
    """Print a formatted log message."""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_metrics_for_trainer(eval_pred):
    """Compute metrics for Trainer evaluation."""
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
# STANDARDIZED FUNCTIONS
# ============================================================================

def load_data(train_path, test_path, val_size=0.2, random_state=42):
    """
    Load and split data into train/val/test sets.
    Uses pre-computed numeric labels from label_numeric column.

    Parameters:
    -----------
    train_path : str
        Path to training CSV (EFCamDAT)
    test_path : str
        Path to test CSV (other corpora)
    val_size : float
        Validation set proportion
    random_state : int
        Random seed

    Returns:
    --------
    dict : Data splits
    """
    log("LOADING DATA")

    # Load datasets
    df_train_full = load_csv(train_path)
    df_test = load_csv(test_path)

    print(f"Training data: {len(df_train_full):,} samples")
    print(f"Test data: {len(df_test):,} samples")

    # Extract text and numeric labels
    X_full = df_train_full['answer']
    y_full_numeric = df_train_full['label_numeric']
    y_full_labels = df_train_full['level']

    X_test = df_test['answer']
    y_test_numeric = df_test['label_numeric']
    y_test_labels = df_test['level']

    # Split train into train/val
    X_train, X_val, y_train, y_val, y_train_labels, y_val_labels = train_test_split(
        X_full, y_full_numeric, y_full_labels,
        test_size=val_size,
        random_state=random_state,
        stratify=y_full_numeric
    )

    print(f"\nSplit sizes:")
    print(f"  Training: {len(X_train):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test: {len(X_test):,}")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test_numeric,
        'y_train_labels': y_train_labels,
        'y_val_labels': y_val_labels,
        'y_test_labels': y_test_labels,
        'df_test': df_test
    }


def prepare_features(X_train, X_val, X_test, y_train, y_val, y_test,
                     model_name='roberta-base', max_length=512):
    """
    Tokenize text and create HuggingFace datasets.

    Parameters:
    -----------
    X_train, X_val, X_test : Text data
    y_train, y_val, y_test : Numeric labels
    model_name : str
        Pre-trained model name
    max_length : int
        Maximum sequence length

    Returns:
    --------
    dict : Tokenized datasets and tokenizer
    """
    log("PREPARING FEATURES (Tokenization)")
    print(f"Model: {model_name}")
    print(f"Max length: {max_length}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create HuggingFace Datasets
    train_dataset = Dataset.from_dict({
        "text": X_train.tolist(),
        "label": y_train.tolist()
    })

    val_dataset = Dataset.from_dict({
        "text": X_val.tolist(),
        "label": y_val.tolist()
    })

    test_dataset = Dataset.from_dict({
        "text": X_test.tolist(),
        "label": y_test.tolist()
    })

    print(f"\n✓ Datasets created")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    print(f"✓ Tokenization complete")

    return {
        'tokenized_train': tokenized_train,
        'tokenized_val': tokenized_val,
        'tokenized_test': tokenized_test,
        'tokenizer': tokenizer
    }


def train_model(tokenized_train, tokenized_val, model_name='roberta-base',
                output_dir='./tmp', batch_size=16, epochs=4,
                learning_rate=2e-5, weight_decay=0.01, random_state=42):
    """
    Train RoBERTa classifier using HuggingFace Trainer.

    Parameters:
    -----------
    tokenized_train : Dataset
        Tokenized training data
    tokenized_val : Dataset
        Tokenized validation data
    model_name : str
        Pre-trained model name
    output_dir : str
        Output directory for checkpoints
    batch_size : int
        Batch size
    epochs : int
        Training epochs
    learning_rate : float
        Learning rate
    weight_decay : float
        Weight decay
    random_state : int
        Random seed

    Returns:
    --------
    Trainer : Trained Trainer object (contains model)
    """
    log("TRAINING MODEL")

    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model loaded")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Training arguments
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

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics_for_trainer
    )

    print("\nStarting training...")
    trainer.train()
    print("✓ Training complete")

    return trainer


def evaluate_model(trainer, tokenized_data, y_numeric, y_labels, dataset_name="Dataset"):
    """
    Evaluate model and return predictions with metrics.

    Parameters:
    -----------
    trainer : Trainer
        Trained Trainer object
    tokenized_data : Dataset
        Tokenized data
    y_numeric : True labels (numeric 0-4)
    y_labels : True labels (CEFR strings)
    dataset_name : str
        Name for logging

    Returns:
    --------
    dict : Evaluation results
    """
    log(f"EVALUATING MODEL ON {dataset_name.upper()}")

    # Get predictions
    predictions = trainer.predict(tokenized_data)
    y_pred_numeric = np.argmax(predictions.predictions, axis=1)

    # Get probabilities
    y_pred_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

    # Convert numeric predictions to CEFR labels
    label_map_inv = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1/C2'}
    y_pred_labels = np.array([label_map_inv[p] for p in y_pred_numeric])

    # Calculate accuracy
    accuracy = accuracy_score(y_numeric.values, y_pred_numeric)

    # Compute comprehensive metrics
    metrics = compute_all_metrics(y_labels, y_pred_labels, y_pred_proba)

    print(f"\n✓ {dataset_name} Accuracy: {accuracy:.4f}")
    print_metrics(metrics, train_acc=None)

    return {
        'y_true_numeric': y_numeric.values,
        'y_pred_numeric': y_pred_numeric,
        'y_true_labels': y_labels,
        'y_pred_labels': y_pred_labels,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'metrics': metrics
    }


def perform_error_analysis(X_data, y_true_labels, y_pred_labels, df_data, output_dir, dataset_name="validation"):
    """
    Perform error analysis and save misclassified samples.

    Parameters:
    -----------
    X_data : Text data
    y_true_labels : True CEFR labels
    y_pred_labels : Predicted CEFR labels
    df_data : DataFrame with full data
    output_dir : str
        Output directory
    dataset_name : str
        Name for output file

    Returns:
    --------
    str : Path to saved CSV
    """
    log(f"ERROR ANALYSIS - {dataset_name.upper()}")

    error_path = os.path.join(output_dir, f'misclassified_{dataset_name}.csv')
    analyze_errors(X_data, y_true_labels, y_pred_labels, df_data, output_path=error_path)

    print(f"✓ Error analysis saved to {error_path}")
    return error_path


def perform_per_corpus_analysis(df_test, y_test_labels, y_pred_labels, output_dir=None):
    """
    Analyze performance per corpus in test set.

    Parameters:
    -----------
    df_test : DataFrame
        Test dataframe with 'source_file' column
    y_test_labels : True CEFR labels
    y_pred_labels : Predicted CEFR labels
    output_dir : str, optional
        Output directory to save results

    Returns:
    --------
    dict : Per-corpus results
    """
    log("PER-CORPUS ANALYSIS (Test Set)")

    corpus_results = {}
    for corpus in df_test['source_file'].unique():
        corpus_mask = df_test['source_file'] == corpus
        corpus_y_true = y_test_labels[corpus_mask].values
        corpus_y_pred = y_pred_labels[corpus_mask]

        corpus_acc = (corpus_y_true == corpus_y_pred).mean()
        corpus_results[corpus] = {
            'accuracy': corpus_acc,
            'samples': len(corpus_y_true)
        }

        print(f"\n{corpus}:")
        print(f"  Samples: {len(corpus_y_true):,}")
        print(f"  Accuracy: {corpus_acc:.4f}")

    # Optionally save to CSV
    if output_dir:
        corpus_df = pd.DataFrame([
            {'corpus': corpus, 'accuracy': results['accuracy'], 'samples': results['samples']}
            for corpus, results in corpus_results.items()
        ])
        corpus_path = os.path.join(output_dir, 'per_corpus_results.csv')
        corpus_df.to_csv(corpus_path, index=False)
        print(f"\n✓ Per-corpus results saved to {corpus_path}")

    return corpus_results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(train_path, test_path, output_dir='../results',
                   model_name='roberta-base', max_length=512,
                   batch_size=16, epochs=4, learning_rate=2e-5,
                   weight_decay=0.01, val_size=0.2, random_state=42):
    """
    Run complete RoBERTa CEFR classification experiment.

    Parameters:
    -----------
    train_path : str
        Path to training CSV
    test_path : str
        Path to test CSV
    output_dir : str
        Output directory
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
        Weight decay
    val_size : float
        Validation proportion
    random_state : int
        Random seed

    Returns:
    --------
    dict : Complete results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Step 1: Load data
    data = load_data(train_path, test_path, val_size, random_state)

    # Step 2: Prepare features (tokenization)
    features = prepare_features(
        data['X_train'], data['X_val'], data['X_test'],
        data['y_train'], data['y_val'], data['y_test'],
        model_name=model_name,
        max_length=max_length
    )

    # Step 3: Train model
    trainer = train_model(
        features['tokenized_train'],
        features['tokenized_val'],
        model_name=model_name,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        random_state=random_state
    )

    # Step 4: Evaluate on validation set (in-domain)
    val_results = evaluate_model(
        trainer,
        features['tokenized_val'],
        data['y_val'],
        data['y_val_labels'],
        dataset_name="Validation (In-Domain)"
    )

    # Step 5: Evaluate on test set (out-of-domain)
    test_results = evaluate_model(
        trainer,
        features['tokenized_test'],
        data['y_test'],
        data['y_test_labels'],
        dataset_name="Test (Out-of-Domain)"
    )

    # Step 6: Calculate generalization gap
    generalization_gap = val_results['accuracy'] - test_results['accuracy']
    print(f"\n✓ Generalization Gap: {generalization_gap:.4f} ({generalization_gap * 100:.2f}%)")

    # Step 7: Visualizations
    log("GENERATING VISUALIZATIONS")

    val_cm_path = os.path.join(output_dir, 'confusion_matrix_validation.png')
    plot_confusion_matrix(
        data['y_val_labels'],
        val_results['y_pred_labels'],
        output_path=val_cm_path,
        title=f'Confusion Matrix - Validation ({model_name})'
    )

    test_cm_path = os.path.join(output_dir, 'confusion_matrix_test.png')
    plot_confusion_matrix(
        data['y_test_labels'],
        test_results['y_pred_labels'],
        output_path=test_cm_path,
        title=f'Confusion Matrix - Test ({model_name})'
    )

    # Step 8: Error analysis
    df_val_analysis = pd.DataFrame({
        'answer': data['X_val'].values,
        'level': data['y_val_labels'].values
    })
    perform_error_analysis(
        data['X_val'], data['y_val_labels'], val_results['y_pred_labels'],
        df_val_analysis, output_dir, dataset_name="validation"
    )
    perform_error_analysis(
        data['X_test'], data['y_test_labels'], test_results['y_pred_labels'],
        data['df_test'], output_dir, dataset_name="test"
    )

    # Step 9: Per-corpus analysis
    corpus_results = perform_per_corpus_analysis(
        data['df_test'], data['y_test_labels'], test_results['y_pred_labels'],
        output_dir=output_dir
    )

    # Step 10: Save summary
    log("SAVING RESULTS")

    summary_path = os.path.join(output_dir, 'experiment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT: ROBERTA CEFR - CORPUS-BASED EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Model: {model_name}\n")
        f.write(f"  Max length: {max_length}\n")
        f.write(f"  Epochs: {epochs}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Weight decay: {weight_decay}\n\n")

        f.write("=" * 70 + "\n")
        f.write("VALIDATION RESULTS (In-Domain)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Accuracy: {val_results['accuracy']:.4f}\n")
        f.write(f"Adjacent Accuracy: {val_results['metrics']['adjacent_accuracy']:.4f}\n")
        f.write(f"MAE: {val_results['metrics']['mae']:.4f}\n")
        f.write(f"QWK: {val_results['metrics']['qwk']:.4f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("TEST RESULTS (Out-of-Domain)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"Adjacent Accuracy: {test_results['metrics']['adjacent_accuracy']:.4f}\n")
        f.write(f"MAE: {test_results['metrics']['mae']:.4f}\n")
        f.write(f"QWK: {test_results['metrics']['qwk']:.4f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("GENERALIZATION ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generalization Gap: {generalization_gap:.4f}\n\n")

        f.write("Per-Corpus Results:\n")
        for corpus, results in sorted(corpus_results.items()):
            f.write(f"  {corpus}: {results['accuracy']:.4f} ({results['samples']:,} samples)\n")

    print(f"✓ Summary saved to {summary_path}")

    # Return complete results
    return {
        'trainer': trainer,
        'model': trainer.model,
        'tokenizer': features['tokenizer'],
        'validation': val_results,
        'test': test_results,
        'corpus_results': corpus_results,
        'generalization_gap': generalization_gap,
        'config': {
            'model_name': model_name,
            'max_length': max_length,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay
        }
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    samples = "100k"
    TRAIN_PATH = f'../dataset/splits/train_{samples}.csv'
    TEST_PATH = '../../dataset/splits/test_other_corpora.csv'

    MODEL_NAME = 'roberta-base'  # or 'roberta-large', 'distilroberta-base', 'microsoft/deberta-v3-base'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01

    VAL_SIZE = 0.2
    RANDOM_STATE = 42

    OUTPUT_DIR = f'../results/RoBERTa_corpus_{samples}_EPOCH_{EPOCHS}_BS_{BATCH_SIZE}_LR_{LEARNING_RATE}'

    print("=" * 80)
    print("ROBERTA - CORPUS-BASED GENERALIZATION EXPERIMENT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print("=" * 80)

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