import numpy as np
import pickle
import sys
from os import path
sys.path.append('..')
from fsd import FeatureShiftDetector
from fsd.divergence import ModelKS, KnnKS, FisherDivergence
from fsd.models import GaussianDensity, Knn
from fsd._utils import marginal_attack, create_graphical_model, sim_copula_data,\
                       get_detection_metrics, plot_confusion_matrix

# Experiment Parameters
n_samples = 1000  # The number of samples in p, q (thus n_samples_total = n_samples*2)
n_MB_bootstrap_runs = 250
n_Knn_bootstrap_runs = 50  # required since Knn-KS is so slow
n_expectation = 30
n_neighbors = 100
n_attacks = 100
j_attacked = 12  # attacked_sensor is fixed
alpha = 0.05  # Significance level
a = 0.5
b = 0.5
sqrtn = 5   # = sqrt(n_dim)
random_seed_list = [0, 1, 2]
mi_list = [0.2, 0.1, 0.05, 0.01]
graph_type_list = ['complete', 'grid', 'cycle', 'random']
experiment_list = ['MB-SM', 'MB-KS', 'KNN-KS']

experiment_results_dict = dict()  # the dictonary of results per experiment_graphtype_mi
for experiment in experiment_list:
    for graph_type in  graph_type_list:
        for mi in mi_list:
            detection_results_across_seeds = []
            for random_seed in random_seed_list:
                # Setting up specific experiment information
                rng = np.random.RandomState(random_seed)
                print(f'Starting: {experiment} on {graph_type} graph with {mi} MI and {random_seed} as the random seed')
                graph = create_graphical_model(sqrtn=sqrtn, kind=graph_type, target_mutual_information=mi,
                                               random_seed=random_seed, target_idx=j_attacked)
                if experiment == 'MB-SM':
                    n_bootstrap_runs = n_MB_bootstrap_runs
                    model = GaussianDensity()
                    statistic = FisherDivergence(model, n_expectation=n_expectation)
                elif experiment == 'MB-KS':
                    n_bootstrap_runs = n_MB_bootstrap_runs
                    model = GaussianDensity()
                    statistic = ModelKS(model, n_expectation=n_expectation)
                else:
                    n_bootstrap_runs = n_Knn_bootstrap_runs
                    model = Knn(n_neighbors=n_neighbors)
                    statistic = KnnKS(model, n_expectation=n_expectation)
                # Setting up recording data
                attack_log = np.zeros(shape=n_attacks*2)
                attack_log[n_attacks:] = 1
                # Detection results are [did attack happen, was it detected, the statistic score]
                detection_results = np.zeros(shape=(n_attacks*2, 3))
                # Setting up FeatureShiftDetector
                fsd = FeatureShiftDetector(statistic, bootstrap_method='simple', n_bootstrap_samples=n_bootstrap_runs)
                # since we are using data always drawn from the same distribution we only need to fit once
                X_boot, Y_boot = sim_copula_data(n_samples, n_samples, mean=np.zeros(shape=sqrtn ** 2),
                                       cov=graph['cov'], a=a, b=b, rng=rng)
                fsd.fit(X_boot, Y_boot)  # sets the detection threshold.

                for test_idx, attack_bool in enumerate(attack_log):
                    X_test, Y_test = sim_copula_data(n_samples*2, n_samples*2, mean=np.zeros(shape=sqrtn**2),
                                           cov=graph['cov'], a=a, b=b, rng=rng)
                    if attack_bool:
                        Y_test = marginal_attack(Y_test, j_attacked)
                        detection_results[test_idx, 0] = 1  # recording an attack happened

                    _, _, scores = fsd.detect_and_localize(X_test, Y_test, random_state=rng, return_scores=True)
                    detection_results[test_idx, 2] = scores[j_attacked]
                    if scores[j_attacked] > fsd.localization_thresholds_[j_attacked]:
                        detection_results[test_idx, 1] = 1
                # recording testing results for seed
                detection_results_across_seeds.append(detection_results.copy())

            # combining results across seeds
            detection_results = np.concatenate(detection_results_across_seeds, axis=0)  # combines all seed results
            experiment_results = get_detection_metrics(true_labels=detection_results[:, 0],
                                                    predicted_labels=detection_results[:, 1])
            print('Detection results:')
            print(f'Precision: {experiment_results["precision"]:.3f}; Recall: {experiment_results["recall"]:.3f}')
            plot_title = f'Detection for {experiment} on {graph_type} graph with {mi} MI'
            # Uncomment below if you would like a detection confusion matrix plotted for each experiment
            # plot_confusion_matrix(experiment_results["confusion_matrix"], title=plot_title, plot=True)  # plots cm
            experiment_results['detection_results'] = detection_results
            experiment_results_dict[f'{experiment}_{graph_type}_{mi}'] = experiment_results
            experiment_save_name = path.join('..', 'results', 'fixed-single-sensor-results-dict.pickle')
            pickle.dump(experiment_results_dict, open(experiment_save_name, 'wb'))
            print()

print('Fin!')
