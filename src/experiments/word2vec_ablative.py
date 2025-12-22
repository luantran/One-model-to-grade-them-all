"""
Word2Vec Ablation Study Runner
================================

Experiments:
1. Embedding comparison: Google News Word2Vec (word2vec-google-news-300)
2. Doc2Vec approach: Trained document vectors with 10 epochs
3. Architecture depth: Deep network (multiple hidden layers)
4. Training duration: Extended training with 20 epochs

"""

from src.models.word2vec_classifier import Word2VecClassifier

def run_word2vec_ablative():
    # Define ablation configurations
    ablations = [
        # {
        #     'experiment_name': 'Experiment0_Word2Vec_baseline',
        #     'embedding_model': 'w2v',
        #     'embedding_name': 'glove-wiki-gigaword-300',
        #     'agg_method': 'mean',
        #     'architecture': 'simple',
        #     'hidden_dim': 128,
        #     'epochs': 10,
        #     'learning_rate': 0.001,
        #     'batch_size': 64,
        # },
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
        {
            'experiment_name': 'Experiment2_Doc2Vec',
            'embedding_model': 'doc2vec',
            'embedding_name': 'doc2vec-trained',
            'agg_method': 'mean',
            'architecture': 'simple',
            'doc2vec_epochs': 10,
            'hidden_dim': 128,
            'epochs': 10,
            'learning_rate': 0.001,
            'batch_size': 64,
        },
        {
            'experiment_name': 'Experiment3_Word2Vec_deeper_network',
            'embedding_model': 'w2v',
            'embedding_name': 'glove-wiki-gigaword-300',
            'agg_method': 'mean',
            'architecture': 'deep',
            'hidden_dim': 128,
            'epochs': 10,
            'learning_rate': 0.001,
            'batch_size': 64,
        },
        {
            'experiment_name': 'Experiment4_Word2Vec_more_epochs',
            'embedding_model': 'w2v',
            'embedding_name': 'glove-wiki-gigaword-300',
            'agg_method': 'mean',
            'architecture': 'simple',
            'hidden_dim': 128,
            'epochs': 20,
            'learning_rate': 0.001,
            'batch_size': 64,
        }
    ]
    results = {}
    # Run each ablation experiment
    for ablation in ablations:
        print(f"\n{'='*80}")
        print(f"Running: {ablation['experiment_name']}")
        print(f"{'='*80}\n")

        # Create classifier with configuration
        classifier = Word2VecClassifier({
            # Experiment metadata
            'experiment_name': ablation['experiment_name'],
            'output_dir': f"results/{ablation['experiment_name']}",

            # Data paths
            'train_path': 'dataset/splits/train_100k.csv',
            'test_path': 'dataset/splits/test_other_corpora.csv',
            'remaining_samples_path': 'dataset/splits/remaining_samples.csv',

            # Data splitting
            'test_size': 0.2,
            'random_state': 42,

            # Embedding configuration
            'embedding_model': ablation['embedding_model'],
            'embedding_name': ablation['embedding_name'],
            'agg_method': ablation['agg_method'],
            'stop_words': None,

            # Doc2Vec specific (if applicable)
            'doc2vec_epochs': ablation.get('doc2vec_epochs', 40),

            # Neural network architecture
            'hidden_dim': ablation['hidden_dim'],
            'batch_size': ablation['batch_size'],

            # Training configuration
            'epochs': ablation['epochs'],
            'learning_rate': ablation['learning_rate'],
        })

        # Run experiment
        ablation_results = classifier.run_experiment()
        results[ablation['experiment_name']] = ablation_results

        print(f"\n✓ {ablation['experiment_name']} completed")
        print(f"  In-Domain Accuracy: {ablation_results['in-domain-test']['accuracy']:.4f}")
        print(f"  In-Domain QWK: {ablation_results['in-domain-test']['metrics']['qwk']:.4f}")
        print(f"  Out-Domain Accuracy: {ablation_results['out-domain-test']['accuracy']:.4f}")
        print(f"  Out-Domain QWK: {ablation_results['out-domain-test']['metrics']['qwk']:.4f}")
        print(f"  Generalization Gap: {ablation_results['generalization_gap']:.4f}")

    print("\n" + "="*80)
    print("✓ All ablation experiments completed!")
    print("="*80)
    return results

if __name__ == "__main__":
    run_word2vec_ablative()