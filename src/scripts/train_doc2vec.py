import argparse

from src.models.word2vec_classifier import Word2VecClassifier
from src.scripts import SAVED_MODELS_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Doc2Vec model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='Doc2Vec',
        help="Name of model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f'{SAVED_MODELS_DIR}',
        help="Output directory"
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=f'doc2vec/',
        help="Subdirectory in output directory"
    )
    args = parser.parse_args()

    # Define ablation configurations
    config = {
            'experiment_name': args.model,
            'embedding_model': 'doc2vec',
            'embedding_name': 'doc2vec-trained',
            'agg_method': 'mean',
            'architecture': 'simple',
            'doc2vec_epochs': 10,
            'doc2vec_mincount' : 1,
            'hidden_dim': 128,
            'epochs': 10,
            'learning_rate': 0.001,
            'batch_size': 64,
    }

    # Run each ablation experiment
    print(f"\n{'='*80}")
    print(f"Running: {config['experiment_name']}")
    print(f"{'='*80}\n")

    # Create classifier with configuration
    w2v_classifier = Word2VecClassifier({
        # Experiment metadata
        'experiment_name': config['experiment_name'],
        'output_dir': f"{args.output_dir}/{args.model}",

        # Data paths
        'train_path': 'dataset/splits/train_100k.csv',
        'test_path': 'dataset/splits/test_other_corpora.csv',
        'remaining_samples_path': 'dataset/splits/remaining_samples.csv',

        # Data splitting
        'test_size': 0.2,
        'random_state': 42,

        # Embedding configuration
        'embedding_model': config['embedding_model'],
        'embedding_name': config['embedding_name'],
        'agg_method': config['agg_method'],
        'stop_words': None,

        # Doc2Vec specific (if applicable)
        'doc2vec_epochs': config.get('doc2vec_epochs', 40),

        # Neural network architecture
        'hidden_dim': config['hidden_dim'],
        'batch_size': config['batch_size'],

        # Training configuration
        'epochs': config['epochs'],
        'learning_rate': config['learning_rate'],
    })

    # Run experiment
    results = w2v_classifier.run_experiment()
    w2v_classifier.save_model(args.subdir)

    print(f"\n✓ {config['experiment_name']} completed")

    print(f"  In-Domain Accuracy: {results['in-domain-test']['accuracy']:.4f}")
    print(f"  In-Domain QWK: {results['in-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Out-Domain Accuracy: {results['out-domain-test']['accuracy']:.4f}")
    print(f"  Out-Domain QWK: {results['out-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Generalization Gap: {results['generalization_gap']:.4f}")