from src.models.naive_bayes_classifier import NBClassifier
from src.models.word2vec_classifier import Word2VecClassifier

SAVED_MODELS_DIR = 'saved_models'

def run_nb():
    nb_config = {'experiment_name': 'NaiveBayes', 'method': 'tfidf', 'max_features': 15000,
         'ngram_range': (1, 2), 'stop_words': 'english'}

    print(f"\n=== Running ablation: {nb_config['experiment_name']} ===")

    nb_classifier = NBClassifier({
        'output_dir': f'{SAVED_MODELS_DIR}/{nb_config['experiment_name']}',
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

    nb_classifier.save_model()

    print(f"\n✓ {nb_config['experiment_name']} completed")
    print(f"  In-Domain Accuracy: {ablation_results['in-domain-test']['accuracy']:.4f}")
    print(f"  In-Domain QWK: {ablation_results['in-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Out-Domain Accuracy: {ablation_results['out-domain-test']['accuracy']:.4f}")
    print(f"  Out-Domain QWK: {ablation_results['out-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Generalization Gap: {ablation_results['generalization_gap']:.4f}")

def run_word2vec():
    # Define ablation configurations
    w2v_config = {
            'experiment_name': 'Experiment2_Doc2Vec',
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
    results = {}
    # Run each ablation experiment
    print(f"\n{'='*80}")
    print(f"Running: {w2v_config['experiment_name']}")
    print(f"{'='*80}\n")

    # Create classifier with configuration
    w2v_classifier = Word2VecClassifier({
        # Experiment metadata
        'experiment_name': w2v_config['experiment_name'],
        'output_dir': f"{SAVED_MODELS_DIR}/{w2v_config['experiment_name']}",

        # Data paths
        'train_path': 'dataset/splits/train_100k.csv',
        'test_path': 'dataset/splits/test_other_corpora.csv',
        'remaining_samples_path': 'dataset/splits/remaining_samples.csv',

        # Data splitting
        'test_size': 0.2,
        'random_state': 42,

        # Embedding configuration
        'embedding_model': w2v_config['embedding_model'],
        'embedding_name': w2v_config['embedding_name'],
        'agg_method': w2v_config['agg_method'],
        'stop_words': None,

        # Doc2Vec specific (if applicable)
        'doc2vec_epochs': w2v_config.get('doc2vec_epochs', 40),

        # Neural network architecture
        'hidden_dim': w2v_config['hidden_dim'],
        'batch_size': w2v_config['batch_size'],

        # Training configuration
        'epochs': w2v_config['epochs'],
        'learning_rate': w2v_config['learning_rate'],
    })

    # Run experiment
    results = w2v_classifier.run_experiment()
    w2v_classifier.save_model()

    print(f"\n✓ {w2v_config['experiment_name']} completed")
    print(f"  In-Domain Accuracy: {results['in-domain-test']['accuracy']:.4f}")
    print(f"  In-Domain QWK: {results['in-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Out-Domain Accuracy: {results['out-domain-test']['accuracy']:.4f}")
    print(f"  Out-Domain QWK: {results['out-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Generalization Gap: {results['generalization_gap']:.4f}")

if __name__ == "__main__":
    # run_nb()
    run_word2vec()