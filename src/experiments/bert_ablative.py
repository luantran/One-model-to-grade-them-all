"""
BERT/RoBERTa Ablation Experiments

0. Baseline: RoBERTa-base
1. Model comparison: DistilRoBERTa
2. Sequence length: Shorter sequences (256 tokens)
3. Fine-tuning: Frozen encoder (only train classifier head)
4. Fine-tuning: Lower learning rate with more epochs
"""

from src.models.bert_classifier import BERTClassifier

def run_bert_ablative():

    # Define ablation configurations
    ablations = [
        # {
        #     'experiment_name': 'Experiment0_RoBERTa_baseline',
        #     'description': 'RoBERTa-base with standard hyperparameters',
        #     'model_name': 'roberta-base',
        #     'max_length': 512,
        #     'batch_size': 16,
        #     'epochs': 4,
        #     'learning_rate': 2e-5,
        #     'weight_decay': 0.01,
        #     'freeze_encoder': False,
        # },
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
        {
            'experiment_name': 'Experiment2_RoBERTa_short_sequences',
            'description': 'Max length 256 (faster inference, less context)',
            'model_name': 'roberta-base',
            'max_length': 256,  # Shorter sequences
            'batch_size': 32,  # Can fit more in memory
            'epochs': 4,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'freeze_encoder': False,
        },
        {
            'experiment_name': 'Experiment3_RoBERTa_frozen_encoder',
            'description': 'Frozen encoder (only train classifier head)',
            'model_name': 'roberta-base',
            'max_length': 512,
            'batch_size': 32,  # Faster training, can use larger batch
            'epochs': 7,  # More epochs since only training head
            'learning_rate': 1e-3,  # Higher LR for classifier head
            'weight_decay': 0.01,
            'freeze_encoder': True,  # Only train classification head
        },
        {
            'experiment_name': 'Experiment4_RoBERTa_careful_tuning',
            'description': 'Lower LR with more epochs (conservative fine-tuning)',
            'model_name': 'roberta-base',
            'max_length': 512,
            'batch_size': 16,
            'epochs': 10,  # More epochs
            'learning_rate': 1e-5,  # Lower learning rate
            'weight_decay': 0.01,
            'freeze_encoder': False,
        },
    ]

    results = {}
    # Run each ablation experiment
    for ablation in ablations:
        print(f"\n{'='*80}")
        print(f"Running: {ablation['experiment_name']}")
        print(f"Description: {ablation['description']}")
        print(f"{'='*80}\n")

        # Create classifier with configuration
        classifier = BERTClassifier({
            # Experiment metadata
            'experiment_name': ablation['experiment_name'],
            'output_dir': f"results/{ablation['experiment_name']}",

            # Data paths
            'train_path': 'dataset/splits/train_100k.csv',
            'test_path': 'dataset/splits/test_other_corpora.csv',

            # Data splitting
            'test_size': 0.2,
            'random_state': 42,

            # Model configuration
            'model_name': ablation['model_name'],
            'max_length': ablation['max_length'],
            'batch_size': ablation['batch_size'],
            'epochs': ablation['epochs'],
            'learning_rate': ablation['learning_rate'],
            'weight_decay': ablation['weight_decay'],

            # Advanced features
            'freeze_encoder': ablation.get('freeze_encoder', False),
        })

        # Run experiment
        ablation_results = classifier.run_experiment()

        results[ablation['experiment_name']] = ablation_results
        print(f"\n✓ {ablation['experiment_name']} completed")
        print(f"  In-Domain Test Accuracy: {ablation_results['in-domain-test']['accuracy']:.4f}")
        print(f"  In-Domain Test QWK: {ablation_results['in-domain-test']['metrics']['qwk']:.4f}")
        print(f"  Test Accuracy: {ablation_results['out-domain-test']['accuracy']:.4f}")
        print(f"  Test QWK: {ablation_results['out-domain-test']['metrics']['qwk']:.4f}")
        print(f"  Generalization Gap: {ablation_results['generalization_gap']:.4f}")

    print("\n" + "="*80)
    print("✓ All ablation experiments completed!")
    print("="*80)
    return results

if __name__ == "__main__":
    run_bert_ablative()