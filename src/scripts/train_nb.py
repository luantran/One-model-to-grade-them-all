import argparse

from src.models.naive_bayes_classifier import NBClassifier
from src.scripts import SAVED_MODELS_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Naive Bayes model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='NaiveBayes',
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
        default=f'nb/',
        help="Subdirectory in output directory"
    )
    args = parser.parse_args()

    nb_config = {
        'experiment_name': args.model,
        'method': 'tfidf',
        'max_features': 15000,
        'ngram_range': (1, 2),
        'stop_words': 'english'
    }


    print(f"\n=== Running: {nb_config['experiment_name']} ===")

    nb_classifier = NBClassifier({
        'output_dir': f'{args.output_dir}/{nb_config['experiment_name']}',
        'experiment_name': nb_config['experiment_name'],
        'train_path': 'dataset/splits/train_100k.csv',
        'test_path': 'dataset/splits/test_other_corpora.csv',
        'method': nb_config['method'],
        'max_features': nb_config['max_features'],
        'ngram_range': nb_config['ngram_range'],
        'alpha': 1.0,
        'test_size': 0.2,
        'random_state': 6781,
    })
    ablation_results = nb_classifier.run_experiment()

    nb_classifier.save_model(args.subdir)

    print(f"\n✓ {nb_config['experiment_name']} completed")
    print(f"  In-Domain Accuracy: {ablation_results['in-domain-test']['accuracy']:.4f}")
    print(f"  In-Domain QWK: {ablation_results['in-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Out-Domain Accuracy: {ablation_results['out-domain-test']['accuracy']:.4f}")
    print(f"  Out-Domain QWK: {ablation_results['out-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Generalization Gap: {ablation_results['generalization_gap']:.4f}")