from src.models.naive_bayes_classifier import NBClassifier

"""
Execute Naive Bayes ablation experiments

Experiment1 (Stop Words Kept):
Experiment2 (Unigrams Only):
Experiment3 (Bigrams Only):
Experiment4 (Count Vectorizer):
Experiment5 (Large Vocabulary):
"""

def run_nb_ablative():
    ablations = [
        # {'experiment_name': 'Experiment0_NaiveBayes_baseline', 'method': 'tfidf', 'max_features': 5000, 'ngram_range': (1,2), 'stop_words':'english'},
        {'experiment_name': 'Experiment1_NaiveBayes_no_stopwords', 'method': 'tfidf', 'max_features': 5000, 'ngram_range': (1,2), 'stop_words':None},
        {'experiment_name': 'Experiment2_NaiveBayes_unigrams-only', 'method': 'tfidf', 'max_features': 5000, 'ngram_range': (1,1), 'stop_words':'english'},
        {'experiment_name': 'Experiment3_NaiveBayes_bigrams-only', 'method': 'tfidf', 'max_features': 5000, 'ngram_range': (2,2), 'stop_words':'english'},
        {'experiment_name': 'Experiment4_NaiveBayes_count', 'method': 'count', 'max_features': 5000, 'ngram_range': (1,2), 'stop_words':'english'},
        {'experiment_name': 'Experiment5_NaiveBayes_large_vocab', 'method': 'tfidf', 'max_features': 15000, 'ngram_range': (1,2), 'stop_words':'english'}
    ]

    results = {}

    for ablation in ablations:
        print(f"\n=== Running ablation: {ablation['experiment_name']} ===")

        nb_classifier = NBClassifier({
            'experiment_name': ablation['experiment_name'],
            'train_path': 'dataset/splits/train_100k.csv',
            'test_path': 'dataset/splits/test_other_corpora.csv',
            'output_dir': f'results/{ablation['experiment_name']}',
            'method': ablation['method'],
            'max_features': ablation['max_features'],
            'ngram_range': ablation['ngram_range'],
            'alpha': 1.0,
            'test_size': 0.2,
            'random_state': 6781,
        })
        ablation_results = nb_classifier.run_experiment()

        results[ablation['experiment_name']] = ablation_results
        print(f"\n✓ {ablation['experiment_name']} completed")
        print(f"  In-Domain Accuracy: {ablation_results['in-domain-test']['accuracy']:.4f}")
        print(f"  In-Domain QWK: {ablation_results['in-domain-test']['metrics']['qwk']:.4f}")
        print(f"  Out-Domain Accuracy: {ablation_results['out-domain-test']['accuracy']:.4f}")
        print(f"  Out-Domain QWK: {ablation_results['out-domain-test']['metrics']['qwk']:.4f}")
        print(f"  Generalization Gap: {ablation_results['generalization_gap']:.4f}")


    print("\n" + "=" * 80)
    print("✓ All ablation experiments completed!")
    print("=" * 80)
    return results

if __name__ == "__main__":
    run_nb_ablative()