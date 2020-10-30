import sys
from warnings import warn
from time import time
from os import path

import numpy as np
import pickle

sys.path.append('..')
from fsd import FeatureShiftDetector
from fsd.divergence import ModelKS, KnnKS, FisherDivergence
from fsd.models import GaussianDensity, Knn
from fsd._utils import marginal_attack, create_graphical_model, sim_copula_data,\
                       get_detection_metrics, get_localization_metrics, plot_confusion_matrix, get_confusion_tensor

# Experiment Parameters
n_samples = 10000  # The number of samples in the datastream
t_split_interval = 50
clean_percentage = 0.8  # 1-clean_percentage = % of datastream attacked
n_bootstrap_runs = 250
n_expectation = 30
n_neighbors = 100
mi = 0.5
alpha = 0.05  # Significance level
a = 0.5
b = 0.5
sqrtn = 5   # = sqrt(n_dim)
n_dim = sqrtn * sqrtn
model = GaussianDensity()
statistic = FisherDivergence(model, n_expectation=n_expectation)
random_seed_list = [0, 1, 2]
graph_type_list = ['complete', 'grid', 'cycle', 'random']
n_attacked_sensors_list = [1, 2, 3]
window_size_list = [i*100 for i in range(2, 11)]

experiment_results_dict = dict()  # the dictionary of results, saved with each experiment
for n_compromised in n_attacked_sensors_list:
    for graph_type in graph_type_list:
        for window_size in window_size_list:
            # setting up window size specific parameters
            n_tests = int(n_samples / t_split_interval)
            window_size_interval = int(window_size / t_split_interval)
            attack_point = int(n_samples * clean_percentage)
            attack_point_interval = int(attack_point / t_split_interval)
            # setting up lists for recording results over multiple seeds
            localization_results_across_seeds = []
            detection_results_across_seeds = []
            time_delays_across_seeds = []
            test_times_across_seeds = []
            print(f'Starting: {graph_type} graph with {n_compromised} compromised sensors and ' +
                  f'{window_size} as window size')
            for random_seed in random_seed_list:
                # Setting up specific experiment information
                rng = np.random.RandomState(random_seed)
                graph = create_graphical_model(sqrtn=sqrtn, kind=graph_type, target_mutual_information=mi,
                                               random_seed=random_seed, target_idx='auto')
                # Localization results are [did attack happen, was it localized, the test score] for each feature
                localization_results = np.zeros(shape=(n_dim, n_tests, 3))
                # Detection results are [did a shift happen, was it detected]
                detection_results = np.zeros(shape=(n_tests, 2))
                # Setting up attack data; in this experiment, the attack set is the same throughout the seed
                random_feature_idxs = rng.choice(n_dim, size=n_compromised, replace=False).astype(int)
                localization_results[random_feature_idxs, attack_point_interval:, 0] = 1  # recording attacks
                detection_results[attack_point_interval:, 0] = 1
                # Setting up FeatureShiftDetector
                fsd = FeatureShiftDetector(statistic, bootstrap_method='simple',
                                           n_bootstrap_samples=n_bootstrap_runs,
                                           significance_level=alpha, n_compromised=n_compromised)
                # since we are using data always drawn from the same distribution we only need to fit once
                X_boot, datastream = sim_copula_data(window_size, (window_size*2)+n_samples, mean=np.zeros(shape=sqrtn**2),
                                       cov=graph['cov'], a=a, b=b, rng=rng)
                fsd.fit(X_boot, datastream[:window_size])  # sets the detection threshold for us.
                X_fixed = X_boot.copy()  # X is fixed, and Y is set via the sliding window through the datastream
                datastream = datastream[window_size:]
                # performing the attack on the last 1-clean_percentage % of the datastream
                datastream[attack_point:] = marginal_attack(datastream[attack_point:], random_feature_idxs)
                # beginning testing
                attack_detected = False
                time_delay = 0
                for test_idx in range(n_tests):
                    # set Y_test via the sliding window through the datastream
                    window_start, window_end = test_idx * t_split_interval, test_idx * t_split_interval + window_size
                    Y_test = datastream[window_start:window_end]
                    start = time()
                    detection, attacked_features, scores = \
                        fsd.detect_and_localize(X_fixed, Y_test, random_state=rng, return_scores=True)
                    localization_results[:, test_idx, 2] = scores
                    detection_results[test_idx, 1] = detection
                    if detection:  # if a distribution shift is detected, record localization results
                        localization_results[attacked_features, test_idx, 1] = 1
                    # if an attacked sample has entered the window and an attack has not yet been detected, increment
                    # time_delay by 1
                    if window_end > attack_point:
                        if detection:
                            attack_detected = True
                        if not attack_detected:
                            time_delay += 1
                    test_times_across_seeds.append(time() - start)
                # recording testing results for seed
                if not attack_detected:
                    time_delay = np.nan  # if no attack was detected, the time_delay is infinite
                time_delays_across_seeds.append(time_delay)
                localization_results_across_seeds.append(localization_results.copy())
                detection_results_across_seeds.append(detection_results.copy())

            # recording time per test across seeds
            time_per_test = np.array(test_times_across_seeds).mean()
            time_delays_across_seeds = np.array(time_delays_across_seeds)
            print(f'Time-delay: {time_delays_across_seeds.mean():.2f} steps; Time per test: {time_per_test:.4f} sec')
            # recording detection results across seeds
            detection_results = np.concatenate(detection_results_across_seeds, axis=0)
            detection_metrics = get_detection_metrics(true_labels=detection_results[:, 0],
                                                   predicted_labels=detection_results[:, 1])
            print(f'Detection results:            Precision: {detection_metrics["precision"]:.3f};' +
                  f'       Recall: {detection_metrics["recall"]:.3f}')
            # recording localization results across seeds
            localization_results = np.concatenate(localization_results_across_seeds, axis=1)  # combines seed results
            localization_metrics = get_localization_metrics(localization_results[:, :, 0],
                                                            localization_results[:, :, 1], n_dim=n_dim)
            print('---------------------------------------------------------------------')
            print(f'Localization results:   Micro-precision: {localization_metrics["micro-precision"]:.3f};' +
                  f' Micro-recall: {localization_metrics["micro-recall"]:.3f}')
            plot_title = f'Detection on {graph_type} graph with {window_size} window size' \
                         f' and {n_compromised} compromised sensors'
            # Uncomment below if you would like a detection confusion matrix plotted for each experiment
#             plot_confusion_matrix(detection_metrics["confusion_matrix"],
#                                   title=plot_title, plot=True)  # plots cm
            # saving results
            experiment_results = {
                'detection_results': detection_results,
                'detection_metrics': detection_metrics,
                'localization_results': localization_results,
                'localization_metrics': localization_metrics,
                'time': np.array(test_times_across_seeds),
                'time-delay': time_delays_across_seeds
            }
            experiment_name = f'{graph_type}_{mi}_with_{n_compromised}_attacked-{window_size}'
            experiment_results_dict[experiment_name] = experiment_results
            experiment_save_name = path.join('..', 'results', 'time-series-simulated-dict.pickle')
            pickle.dump(experiment_results_dict, open(experiment_save_name, 'wb'))
            print()
print('Fin!')
