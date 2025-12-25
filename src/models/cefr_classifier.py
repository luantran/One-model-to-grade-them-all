import os
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger import log_step, save_classification_report_structured, save_results_json, save_experiment_summary
from src.utils.splitter import load_dataset as load_csv

from src.utils.evaluation_utils import print_metrics, compute_all_metrics, plot_confusion_matrix, \
    perform_per_corpus_analysis
from src.utils.vizualizer import plot_data_distribution


class CEFRClassifier(ABC):
    """Abstract base class for CEFR classification models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize classifier with configuration.

        """
        self.config = config
        self.model = None
        self.results = {}
        self.output_dir = config['output_dir']
        self.experiment_name = config['experiment_name']
        if self.experiment_name is None:
            raise ValueError('Experiment name is required: experiment_name=...')
        if self.output_dir is None:
            raise ValueError('Output directory is required: output_dir=...')


    # Template method - defines the workflow
    def run_experiment(self) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)

        log_step(f"{self.experiment_name} - [1/11] LOADING DATA")
        data = self.load_data()
        plot_data_distribution(data, self.output_dir)

        print("✓ Data loaded")

        log_step(f"{self.experiment_name} - [2/11] PREPARING FEATURES")
        features = self.prepare_features(
            data['X_train'], data['X_in_test'], data['X_out_test'], data['y_train'], data['y_in_test'], data['y_out_test']
        )
        print("✓ Feature preparation complete")


        log_step(f"{self.experiment_name} - [3/11] TRAINING MODEL")
        model = self.train_model(features['X_train_features'], data['y_train'])
        print("✓ Model trained")

        # Get labels - use split labels from features if available, otherwise use original data labels
        y_in_test = features.get('y_in_test', data['y_in_test'])
        y_in_test_labels = features.get('y_in_test_labels', data['y_in_test_labels'])
        y_out_test = features.get('y_out_test', data['y_out_test'])
        y_out_test_labels = features.get('y_out_test_labels', data['y_out_test_labels'])

        log_step(f"{self.experiment_name} - [4/11] EVALUATING IN-DOMAIN TEST SET")
        in_test_results = self.evaluate_model(
            model,
            features['X_in_test_features'],
            y_in_test,
            y_in_test_labels,
            dataset_name="In-Domain Test"
        )
        print("✓ In-Domain evaluation complete")

        log_step(f"{self.experiment_name} - [5/11] EVALUATING OUT-OF-DOMAIN TEST SET")
        out_test_results = self.evaluate_model(
            model,
            features['X_out_test_features'],
            y_out_test,
            y_out_test_labels,
            dataset_name="Test (Out-of-Domain)"
        )
        print("✓ Out-of-Domain evaluation complete")

        log_step(f"{self.experiment_name} - [6/11] SAVING CLASSIFICATION REPORTS")
        save_classification_report_structured(
            y_in_test_labels, in_test_results['y_pred_labels'],
            self.output_dir, dataset_name="in-domain test"
        )
        save_classification_report_structured(
            y_out_test_labels, out_test_results['y_pred_labels'],
            self.output_dir, dataset_name="out-of-domain test"
        )
        print(f"✓ Classification reports saved")

        log_step(f"{self.experiment_name} - [7/11] COMPUTING GENERALIZATION GAP")
        generalization_gap = in_test_results['accuracy'] - out_test_results['accuracy']
        print(f"\n✓ Generalization Gap: {generalization_gap:.4f} ({generalization_gap * 100:.2f}%)")

        log_step(f"{self.experiment_name} - [8/11] GENERATING CONFUSION MATRICES")
        in_test_cm_path = os.path.join(self.output_dir, 'confusion_matrix_in_domain_test.png')
        plot_confusion_matrix(
            y_in_test_labels, in_test_results['y_pred_labels'],
            output_path=in_test_cm_path,
            title=f'Confusion Matrix - In-Domain Test ({self.experiment_name})'
        )
        print(f"✓ Confusion Matrix for in-domain test saved: {in_test_cm_path}")

        out_test_cm_path = os.path.join(self.output_dir, 'confusion_matrix_test.png')
        plot_confusion_matrix(
            y_out_test_labels, out_test_results['y_pred_labels'],
            output_path=out_test_cm_path,
            title=f'Confusion Matrix - Out-of-Domain Test ({self.experiment_name})'
        )
        print(f"✓  Confusion Matrix for out-of-domain test saved: {out_test_cm_path}")


        log_step(f"{self.experiment_name} - [9/11] PER-CORPUS ANALYSIS")
        corpus_results = perform_per_corpus_analysis(
            data['df_test'], y_out_test_labels, out_test_results['y_pred_labels'],
            output_dir=self.output_dir
        )
        print("✓ Per-corpus analysis complete")

        log_step(f"{self.experiment_name} - [10/11] SAVING EXPERIMENT SUMMARY")

        summary_path = save_experiment_summary(
            in_test_results, out_test_results, corpus_results, generalization_gap,
            self.config, 'Naive Bayes', self.output_dir
        )
        print(f"✓ Summary saved: {summary_path}")

        log_step(f"{self.experiment_name} - [11/11] SAVING RESULTS JSON")
        json_path = save_results_json(
            in_test_results, out_test_results, corpus_results, generalization_gap,
            self.config, self.output_dir
        )
        print(f"✓ JSON saved: {json_path}")

        log_step("✓ EXPERIMENT COMPLETE")

        return {
            'model': model,
            'vectorizer': features['vectorizer'],
            'in-domain-test': in_test_results,
            'out-domain-test': out_test_results,
            'corpus_results': corpus_results,
            'generalization_gap': generalization_gap,
            'config': self.config,
        }

    # Concrete method - common across all models
    def load_data(self) -> dict[str, Any]:
        """ Load and split data into train/test sets.
        """

        # Auto-set train_path if not specified
        train_path = self.config.get('train_path', f'dataset/splits/train_100k.csv')
        test_path = self.config.get('test_path', f'dataset/splits/test_other_corpora.csv')

        # Load datasets
        df_train_full = load_csv(train_path)
        df_test = load_csv(test_path)

        print(f"Training data: {len(df_train_full):,} samples")
        print(f"Test data: {len(df_test):,} samples")

        # Extract text and numeric labels
        X_full = df_train_full['answer']
        y_full_numeric = df_train_full['label_numeric']  # Use pre-computed numeric labels
        y_full_labels = df_train_full['level']  # Keep CEFR labels for display

        X_out_test = df_test['answer']
        y_out_test_numeric = df_test['label_numeric']
        y_out_test_labels = df_test['level']

        # Split train into train/val
        X_train, X_in_test, y_train, y_in_test, y_train_labels, y_in_test_labels = train_test_split(
            X_full, y_full_numeric, y_full_labels,
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42),
            stratify=y_full_numeric
        )

        print(f"\nSplit sizes:")
        print(f"  Training: {len(X_train):,}")
        print(f"  In-Domain Test: {len(X_in_test):,}")
        print(f"  Out-of-Domain Test: {len(X_out_test):,}")

        # Mapping from numeric to CEFR labels
        label_map_inv = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1/C2'}

        # Helper function to print distribution
        def print_distribution(y, split_name=""):
            total = len(y)
            print(f"\nLabel distribution ({split_name}):")
            for numeric in sorted(y.unique()):
                count = (y == numeric).sum()
                percentage = (count / total) * 100
                cefr = label_map_inv[numeric]
                print(f"  {numeric} ({cefr}): {count:,} ({percentage:.1f}%)")

        # Print distributions for all splits
        print_distribution(y_train, "training")
        print_distribution(y_in_test, "in-domain test")
        print_distribution(y_out_test_numeric, "out-of-domain test")

        return {
            'X_train': X_train,
            'X_in_test': X_in_test,
            'X_out_test': X_out_test,
            'y_train': y_train,
            'y_in_test': y_in_test,
            'y_out_test': y_out_test_numeric,
            'y_train_labels': y_train_labels,
            'y_in_test_labels': y_in_test_labels,
            'y_out_test_labels': y_out_test_labels,
            'df_test': df_test
        }

    # Concrete method -common across all_models
    def evaluate_model(self, model, X, y_numeric, y_labels, dataset_name="Dataset"):
        """
        Evaluate model and return predictions with metrics.
        """

        # Predict
        y_pred_numeric = model.predict(X)

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


    # Abstract methods - must be implemented by subclasses
    @abstractmethod
    def prepare_features(self, X_train, X_in_test, X_out_test, y_train, y_in_test, y_out_test) -> Dict[str, Any]:
        """
        Prepare features for the model.
        """
        pass

    @abstractmethod
    def train_model(self, X_train: Any, y_train: Any) -> Any:
        """
        Train the classification model.
        """
        pass

    @abstractmethod
    def save_model(self):
        pass