from typing import Dict, Any
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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.models.cefr_classifier import CEFRClassifier


class BERTClassifier(CEFRClassifier):
    """BERT/RoBERTa classifier for CEFR classification."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_name = config.get('model_name', 'roberta-base')
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 16)
        self.epochs = config.get('epochs', 3)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.weight_decay = config.get('weight_decay', 0.01)

        self.tokenizer = None
        self.tokenized_val = None  # Store for training

    def prepare_features(self, X_train: pd.Series, X_in_test: pd.Series, X_out_test: pd.Series,
                         y_train: pd.Series, y_in_test: pd.Series, y_out_test: pd.Series) -> Dict[str, Any]:
        """Tokenize text and create HuggingFace datasets.

        Splits X_in_test (in-domain) into:
        - 50% validation (for early stopping during training)
        - 50% in-domain test (for final evaluation)
        """
        print(f"\nModel: {self.model_name}")
        print(f"Max length: {self.max_length}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Split in-domain test into validation (50%) and in-domain test (50%)
        val_size = 0.5
        random_state = self.config.get('random_state', 42)

        X_in_test_final, X_val, y_in_test_final, y_val = train_test_split(
            X_in_test, y_in_test,
            test_size=val_size,
            random_state=random_state,
            stratify=y_in_test
        )

        print(f"\nDataset splits:")
        print(f"  Training (80%): {len(X_train):,}")
        print(f"  Validation (50% of in-domain): {len(X_val):,}")
        print(f"  In-domain test (50% of in-domain): {len(X_in_test_final):,}")
        print(f"  Out-of-domain test: {len(X_out_test):,}")
        print(f"\nTokenizing datasets...")

        # Create datasets
        def create_dataset(X, y):
            return Dataset.from_dict({
                "text": X.tolist(),
                "label": y.tolist()
            })

        train_dataset = create_dataset(X_train, y_train)
        val_dataset = create_dataset(X_val, y_val)
        in_test_dataset = create_dataset(X_in_test_final, y_in_test_final)
        out_test_dataset = create_dataset(X_out_test, y_out_test)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)
        tokenized_in_test = in_test_dataset.map(tokenize_function, batched=True)
        tokenized_out_test = out_test_dataset.map(tokenize_function, batched=True)

        # Store validation dataset for training
        self.tokenized_val = tokenized_val

        print(f"✓ Tokenization complete")

        return {
            'X_train_features': tokenized_train,
            'X_in_test_features': tokenized_in_test,
            'X_out_test_features': tokenized_out_test,
            'vectorizer': self.tokenizer,
            # Return split labels - BOTH numeric and string versions
            'y_in_test': y_in_test_final,  # numeric (0-4)
            'y_in_test_labels': y_in_test_final.map({0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1/C2'}),
            # Map the SPLIT version
            'y_out_test': y_out_test,  # numeric (0-4)
            'y_out_test_labels': y_out_test.map({0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1/C2'}),
        }

    def train_model(self, X_train_features: Any, y_train: pd.Series) -> Any:

        """Train BERT/RoBERTa classifier."""
        print(f"\nTraining Configuration:")
        print(f"  Model: {self.model_name}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=5
        )

        # Freeze encoder if requested
        if self.config.get('freeze_encoder', False):
            print("\nFreezing encoder - only training classification head")
            for name, param in model.named_parameters():
                if 'classifier' not in name:  # Freeze everything except classifier
                    param.requires_grad = False

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable parameters (head only): {trainable_params:,}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel loaded")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            fp16=torch.cuda.is_available(),
            report_to="none",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.config.get('random_state', 42)
        )

        # Metrics for trainer
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted', zero_division=0
            )
            return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=X_train_features,
            eval_dataset=self.tokenized_val,
            compute_metrics=compute_metrics
        )

        print("\nStarting training...")
        trainer.train()
        print("Training complete")

        self.trainer = trainer

        return trainer


    def evaluate_model(self, model, X, y_numeric, y_labels, dataset_name="Dataset"):
        """
        Override parent's evaluate_model to handle HuggingFace Trainer predictions.
        """
        from src.utils.evaluation_utils import compute_all_metrics, print_metrics

        # Validate inputs
        if not isinstance(X, Dataset):
            raise ValueError("BERTClassifier requires tokenized Dataset")

        if 'input_ids' not in X.column_names:
            raise ValueError("Dataset must be tokenized")

        # Get predictions from trainer
        predictions = model.predict(X)

        y_pred_numeric = np.argmax(predictions.predictions, axis=1)

        # Convert numeric predictions back to CEFR labels
        label_map_inv = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1/C2'}
        y_pred_labels = np.array([label_map_inv[p] for p in y_pred_numeric])

        # Calculate accuracy
        accuracy = (y_pred_numeric == y_numeric).mean()

        # Compute comprehensive metrics
        metrics = compute_all_metrics(y_labels, y_pred_labels)

        print(f"\n✓ {dataset_name} Accuracy: {accuracy:.4f}")
        print_metrics(metrics, train_acc=None)

        return {
            'y_true_numeric': y_numeric,
            'y_pred_numeric': y_pred_numeric,
            'y_true_labels': y_labels,
            'y_pred_labels': y_pred_labels,
            'accuracy': accuracy,
            'metrics': metrics,
        }