from src.models.bert_classifier import BERTClassifier
from src.models.naive_bayes_classifier import NBClassifier
from src.models.word2vec_classifier import Word2VecClassifier

"""
Execute baseline experiments for all three model architectures (NB, W2V, BERT).

Naive Bayes Baseline (Experiment0_NaiveBayes_baseline):
- Method: TF-IDF vectorization
- Max features: 5000
- N-gram range: (1, 2) [unigrams + bigrams]
- Stop words: English
- Alpha (smoothing): 1.0

Word2Vec Baseline (Experiment0_Word2Vec_baseline):
- Embedding model: GloVe (glove-wiki-gigaword-300)
- Aggregation method: mean
- Architecture: simple (single hidden layer)
- Hidden dimension: 128
- Batch size: 64
- Epochs: 10
- Learning rate: 0.001

BERT Baseline (Experiment0_RoBERTa_baseline):
- Model: roberta-base
- Max sequence length: 512 tokens
- Batch size: 16
- Epochs: 10
- Learning rate: 2e-5
- Weight decay: 0.01
- Freeze encoder: False (full fine-tuning)


Returns: Dictionary with results for each model (accuracy, QWK, generalization gap)
"""

def run_baseline_experiments():

    # ================================================================================
    #    NAIVE BAYES BASELINE
    # ================================================================================

    results = {}

    nb_baseline_config = {
        'experiment_name': 'Experiment0_NaiveBayes_baseline', 'method': 'tfidf', 'max_features': 5000,
         'ngram_range': (1, 2), 'stop_words': 'english'}

    print("\n" + "=" * 80)
    print(f" Running NB Baseline: {nb_baseline_config['experiment_name']} ")
    print("=" * 80)

    nb_classifier = NBClassifier({
        'experiment_name': nb_baseline_config['experiment_name'],
        'train_path': 'dataset/splits/train_100k.csv',
        'test_path': 'dataset/splits/test_other_corpora.csv',
        'output_dir': f'results/{nb_baseline_config['experiment_name']}',
        'method': nb_baseline_config['method'],
        'max_features': nb_baseline_config['max_features'],
        'ngram_range': nb_baseline_config['ngram_range'],
        'alpha': 1.0,
        'test_size': 0.2,
        'random_state': 6781,
    })
    nb_results = nb_classifier.run_experiment()
    results['NB'] = nb_results

    print(f"\n✓ {nb_baseline_config['experiment_name']} completed")
    print(f"  Val Accuracy: {nb_results['in-domain-test']['accuracy']:.4f}")
    print(f"  Val QWK: {nb_results['in-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Test Accuracy: {nb_results['out-domain-test']['accuracy']:.4f}")
    print(f"  Test QWK: {nb_results['out-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Generalization Gap: {nb_results['generalization_gap']:.4f}")

    print("\n" + "=" * 80)
    print("✓ NB baseline experiment completed!")
    print("=" * 80)

    # ================================================================================
    #   WORD2VEC BASELINE
    # ================================================================================

    w2v_baseline_config = {
        'experiment_name': 'Experiment0_Word2Vec_baseline',
        'embedding_model': 'w2v',
        'embedding_name': 'glove-wiki-gigaword-300',
        'agg_method': 'mean',
        'architecture': 'simple',
        'hidden_dim': 128,
        'epochs': 10,
        'learning_rate': 0.001,
        'batch_size': 64,
    }

    print("\n" + "=" * 80)
    print(f" Running Word2Vec Baseline: {w2v_baseline_config['experiment_name']} ")
    print("=" * 80)

    # Create classifier with configuration
    w2v_classifier = Word2VecClassifier({
        # Experiment metadata
        'experiment_name': w2v_baseline_config['experiment_name'],
        'output_dir': f"results/{w2v_baseline_config['experiment_name']}",

        # Data paths
        'train_path': 'dataset/splits/train_100k.csv',
        'test_path': 'dataset/splits/test_other_corpora.csv',
        'remaining_samples_path': 'dataset/splits/remaining_samples.csv',

        # Data splitting
        'test_size': 0.2,
        'random_state': 42,

        # Embedding configuration
        'embedding_model': w2v_baseline_config['embedding_model'],
        'embedding_name': w2v_baseline_config['embedding_name'],
        'agg_method': w2v_baseline_config['agg_method'],
        'stop_words': None,

        # Doc2Vec specific (if applicable)
        'doc2vec_epochs': w2v_baseline_config.get('doc2vec_epochs', 40),

        # Neural network architecture
        'hidden_dim': w2v_baseline_config['hidden_dim'],
        'batch_size': w2v_baseline_config['batch_size'],

        # Training configuration
        'epochs': w2v_baseline_config['epochs'],
        'learning_rate': w2v_baseline_config['learning_rate'],
    })

    # Run experiment
    w2v_results = w2v_classifier.run_experiment()
    results['W2V'] = w2v_results

    print(f"\n✓ {w2v_baseline_config['experiment_name']} completed")
    print(f"  Val Accuracy: {w2v_results['in-domain-test']['accuracy']:.4f}")
    print(f"  Val QWK: {w2v_results['in-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Test Accuracy: {w2v_results['out-domain-test']['accuracy']:.4f}")
    print(f"  Test QWK: {w2v_results['out-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Generalization Gap: {w2v_results['generalization_gap']:.4f}")

    # ================================================================================
    #   BERT BASELINE
    # ================================================================================

    bert_baseline_config = {
        'experiment_name': 'Experiment0_RoBERTa_baseline',
        'description': 'RoBERTa-base with standard hyperparameters',
        'model_name': 'roberta-base',
        'max_length': 512,
        'batch_size': 16,
        'epochs': 10,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'freeze_encoder': False,
    }

    print(f"\n{'=' * 80}")
    print(f"Running: {bert_baseline_config['experiment_name']}")
    print(f"{'=' * 80}\n")

    # Create classifier with configuration
    bert_classifier = BERTClassifier({
        # Experiment metadata
        'experiment_name': bert_baseline_config['experiment_name'],
        'output_dir': f"results/{bert_baseline_config['experiment_name']}",

        # Data paths
        'train_path': 'dataset/splits/train_100k.csv',
        'test_path': 'dataset/splits/test_other_corpora.csv',

        # Data splitting
        'test_size': 0.2,
        'random_state': 42,

        # Model configuration
        'model_name': bert_baseline_config['model_name'],
        'max_length': bert_baseline_config['max_length'],
        'batch_size': bert_baseline_config['batch_size'],
        'epochs': bert_baseline_config['epochs'],
        'learning_rate': bert_baseline_config['learning_rate'],
        'weight_decay': bert_baseline_config['weight_decay'],

        # Advanced features
        'freeze_encoder': bert_baseline_config.get('freeze_encoder', False),
    })

    # Run experiment
    bert_results = bert_classifier.run_experiment()
    results['BERT'] = bert_results

    print(f"\n✓ {bert_baseline_config['experiment_name']} completed")
    print(f"  Val Accuracy: {bert_results['in-domain-test']['accuracy']:.4f}")
    print(f"  Val QWK: {bert_results['in-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Test Accuracy: {bert_results['out-domain-test']['accuracy']:.4f}")
    print(f"  Test QWK: {bert_results['out-domain-test']['metrics']['qwk']:.4f}")
    print(f"  Generalization Gap: {bert_results['generalization_gap']:.4f}")

    return results

if __name__ == "__main__":
    run_baseline_experiments()
