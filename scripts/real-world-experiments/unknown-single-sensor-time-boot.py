import sys
import numpy as np
np.seterr(over='ignore')  # suppresses errors related to overflow in numpy
np.seterr(divide='ignore')

import pandas as pd
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
import os
from pathlib import Path
from time import time, localtime, strftime
from warnings import warn
from datetime import datetime
path = os.path



# Utils
from model_all import FeatureShiftDetection


def transform_data(X, do_diff=True, do_power_transform=True):
    def diff(X, y=None):
        return np.diff(X, axis=0)

    def diff2(X, y=None):
        return diff(diff(X))

    first_diff = ('first_difference', FunctionTransformer(func=diff, validate=True))
    power_transform = ('power_transform',
                       PowerTransformer(method='yeo-johnson', standardize=True))
    pipeline = []
    if do_diff:
        pipeline.append(first_diff)
    if do_power_transform:
        pipeline.append(power_transform)

    if len(pipeline) == 0:
        Z = X
    else:
        pipeline = Pipeline(pipeline)
        Z = pipeline.fit_transform(X)
    return Z


# # Global Experiment Parameters
n_samples = 1000  # The number of samples in p, q (thus n_samples_total = n_samples*2)
n_bootstrap_runs = 500
n_conditional_expectation = 30
n_inner_expectation = n_conditional_expectation
alpha = 0.05  # Significance level
data_family = 'Copula'
a = 0.5
b = 0.5
method_list = ['score-method', 'deep_univariate_score_method']
dataset_list = ['Energy', 'Gas', 'COVID']
# dataset_list = ['Gas']
t_split_interval = 50
n_comp_sensors_list = [1]
window_size_list = [i*100 for i in range(11)]
n_comp_sensors = 1

for dataset_name in dataset_list:
    for method in method_list:
        for shuffle_data_set in [False, True]:
            rng = np.random.RandomState(42)
            torch.manual_seed(rng.randint(1000))
            # Experiment Switches
            if shuffle_data_set:
                shuffle_string = 'time axis shuffled'
                experiment_name = f'time-boot-{method}-time-axis-shuffled-on-{dataset_name}'
            else:
                shuffle_string = 'time axis unshuffled'
                experiment_name = f'time-boot-{method}-time-axis-unshuffled-on-{dataset_name}'
            print()
            print(f'Starting {method} on {dataset_name} dataset with {shuffle_string} and time boot')

            # Load in dataset and setup dataset parameters
            if dataset_name == 'Energy':
                do_diff = False
                do_power_transform = True
                dataset = pd.read_csv(path.join('..', '..', 'real-world-datasets', 'energydata.csv'))
                dataset = dataset.to_numpy()[:, 1:27].astype(np.float64)
            elif dataset_name == 'Gas':
                do_diff = False
                do_power_transform = False
                dataset = np.loadtxt(path.join('..', '..', 'real-world-datasets',
                                               'gas-sensor-array-drift.dat'), skiprows=1)
                dataset = dataset[dataset[:, 0] == 0.][:, 1:]  # gets only tests with id=0 and then drops the id
                dataset = dataset[:, 3:]  # the first 3 columns are time, gas1 pcc, gas2 pcc so irrelevant
            elif dataset_name == 'COVID':
                do_diff = True
                do_power_transform = True
                columns = ['MI', 'PA', 'IL', 'NY', 'MA', 'FL', 'TX', 'CA', 'NJ', 'NYC']
                dataset = pd.read_csv(path.join('..', '..', 'real-world-datasets',
                                                'interpolated-new-death-covid-data.csv'),
                                      index_col=0).loc[:, columns].to_numpy()
            else:
                raise NotImplementedError('Please pick a dataset name')

            ###################
            if shuffle_data_set:
                dataset = rng.permutation(dataset)
            ###################

            # splitting dataset into train/test split
            dataset_split_idx = int(dataset.shape[0] / 2)
            bootstrap_dataset = dataset[:dataset_split_idx]
            dataset = dataset[dataset_split_idx:]
            # dataset parameters
            n_dim = dataset.shape[1]
            sqrtn = int(np.floor(np.sqrt(n_dim)))
            n_dataset_samples = dataset[n_samples:].shape[0]  # to account for taking out n_samples for reference dist, p

            # getting thresholds
            bootstrap_split_range = [n_samples, bootstrap_dataset.shape[0] - n_samples]
            bootstrap_score_distribution = np.zeros(shape=(n_bootstrap_runs, n_dim))
            boot_fsd = FeatureShiftDetection(p=rng.randint(0, 100, size=(n_samples, n_dim)),
                                             q=rng.randint(0, 100, size=(n_samples, n_dim)),
                                             rng=rng, samples_generator=np.nan,
                                             detection_method=method, n_bootstrap_runs=n_bootstrap_runs,
                                             n_conditional_expectation=n_conditional_expectation,
                                             n_attacks=np.nan, alpha=alpha, threshold_vector=np.nan,
                                             j_attack=np.nan, attack_testing=False)
            nan_found = 0
            print('Bootstrapping')
            for b_idx in range(n_bootstrap_runs):
                # if b_idx % int(n_bootstrap_runs / 10) == 0:
                    # print(f'Bootstrapping: {b_idx / n_bootstrap_runs * 100:.1f}%')
                try:
                    bootstrap_split = rng.randint(*bootstrap_split_range)
                    pq = bootstrap_dataset[bootstrap_split - n_samples: bootstrap_split + n_samples]
                    pq = transform_data(pq)
                    p = pq[:int(pq.shape[0] / 2)]
                    q = pq[int(pq.shape[0] / 2):int(pq.shape[0] / 2) * 2].copy()
                    assert p.shape == q.shape, 'P must equal q'
                    bootstrap_score_distribution[b_idx] = boot_fsd.get_score(p, q)
                except Exception as e:
                    raise e
                    nan_found = 1
                    print(f'An exception has occured on {b_idx}')
                    print(e)
                    bootstrap_score_distribution[b_idx] = np.nan
            if nan_found == 1:
                rows_without_nan = ~np.isnan(bootstrap_score_distribution[:, 0])
                print(f'Bootstrap dist shape: {bootstrap_score_distribution.shape[0]}')
                bootstrap_score_distribution = bootstrap_score_distribution[rows_without_nan]
                print(f'Shape after dropping nan: {bootstrap_score_distribution.shape[0]}')
            bootstrap_score_distribution = np.sort(bootstrap_score_distribution, axis=0)
            threshold_vector = bootstrap_score_distribution[
                int(bootstrap_score_distribution.shape[0] * (1 - alpha))]  # (1-alpha)th percentile of dist
            bonferroni_threshold_vector = bootstrap_score_distribution[
                int(bootstrap_score_distribution.shape[0] * (1 - alpha / n_dim))]
            bootstrap_score_means_vector = bootstrap_score_distribution.mean(axis=0)
            bootstrap_score_std_vector = np.std(bootstrap_score_distribution, axis=0)

            n_trials = int(np.ceil((dataset.shape[0] - 2 * n_samples) / t_split_interval))
            ## Attack testing  ##
            rng = np.random.RandomState(42)
            torch.manual_seed(rng.randint(1000))

            time_list = np.zeros(n_trials)
            global_truth = np.zeros(n_trials)
            detection = np.zeros(n_trials)
            detection_results = np.zeros(shape=(n_dim, n_trials, 3))
            exception_array = np.full(shape=n_trials, fill_value=False)  # True if an exception occured

            j_attack = rng.choice(np.arange(n_dim), replace=True, size=n_trials)
            # j_attack = np.full(n_trials, 16)
            for idx, feature in enumerate(j_attack[:int(n_trials / 2)]):
                detection_results[feature, idx, 1] = 1  # recording where attacks happen
                global_truth[idx] = 1

            attack_state = 1
            print('Starting testing')
            for test_idx, split_idx in enumerate(range(0, dataset.shape[0] - 2 * n_samples, t_split_interval)):
                if test_idx >= int(n_trials / 2):
                    attack_state = 0
                start = time()
                slice1 = split_idx
                slice2 = split_idx + 2 * n_samples
                # print(f'Starting {test_idx + 1} out of {n_trials} trials. Dataset slice: {slice1}:{slice2}')
                try:
                    pq = dataset[slice1:slice2]  # Two sets of samples
                    pq = transform_data(pq)
                    p = pq[:n_samples]
                    q = pq[n_samples:n_samples * 2].copy()

                    if np.any(detection_results[:, test_idx, 1] == 1):  # attack!
                        attacked_features = j_attack[test_idx]
                        q[:, attacked_features] = rng.permutation(q[:, attacked_features])  # permutes q

                    # Bootstrap every time
                    fsd = FeatureShiftDetection(p, q, rng=rng, samples_generator=np.nan,
                                                detection_method=method,
                                                n_bootstrap_runs=n_bootstrap_runs,
                                                n_conditional_expectation=n_conditional_expectation,
                                                n_attacks=np.nan, alpha=alpha, threshold_vector=np.nan,
                                                j_attack=np.nan, attack_testing=False)
                    # now check after getting new threshold
                    score_vector = np.array(fsd.get_score(p, q))
                    detection_results[:, test_idx, 0] = score_vector
                    # predicting attack
                    if np.any(score_vector >= bonferroni_threshold_vector):
                        detection[test_idx] = 1
                        normalized_score_vector = (score_vector - bootstrap_score_means_vector) / bootstrap_score_std_vector
                        attacked_features = normalized_score_vector.argsort()[-n_comp_sensors]
                        detection_results[attacked_features, test_idx, 2] = 1
                    time_list[test_idx] = time() - start
                except Exception as e:
                    exception_array[test_idx] = True
                    print(f'An exception has occured on {test_idx}')
                    print(e)
            # \end Attack testing  ##

            if any(exception_array):
                # dropping any trials resulted in an exception
                print(f'Current detection_results size: {detection_results.shape[1]} trials')
                print('Removing trials which ran into an error')
                detection_results = detection_results[:, ~exception_array, :]
                time_list = time_list[~exception_array]
                global_truth = global_truth[~exception_array]
                detection = detection[~exception_array]
                print(f'Detection_results size after drop: {detection_results.shape[1]} trials')

            # Recording Attack Results
            confusion_tensor = np.zeros(shape=(n_dim, 2, 2))
            for feature_idx, feature_results in enumerate(detection_results):
                confusion_tensor[feature_idx] = sklearn_confusion_matrix(feature_results[:, 1],
                                                                         feature_results[:, 2],
                                                                         labels=[0, 1])

            # overall detection confusion matrix
            global_detection_confusion_matrix = sklearn_confusion_matrix(global_truth,
                                                                         detection,
                                                                         labels=[0, 1])
            # Plotting results
            # fig, axes = plt.subplots(sqrtn, sqrtn)
            # axes_flat = axes.flatten()
            # for feature, axis in enumerate(axes_flat):
            #     names = ['TN', 'FP', 'FN', 'TP']
            #     counts = confusion_tensor[feature].astype(np.int).flatten()
            #     labels = [f'{n}\n{c}' for n, c in zip(names, counts)]
            #     labels = np.array(labels).reshape(2, 2)
            #     sn.heatmap(confusion_tensor[feature].astype(np.int), annot=labels, fmt='', xticklabels=False,
            #                yticklabels=False, linewidth=.5, cbar=False, ax=axis, cmap='Blues')
            #     axis.set_title(feature + 1)
            # if shuffle_data_set:
            #     plt.suptitle(f'Localization without time dependencies',
            #                  verticalalignment='center')
            # else:
            #     plt.suptitle(f'Localization with time dependencies',
            #                  verticalalignment='center')
            # plt.tight_layout()
            # plt.show()

            full_tn, full_fp, full_fn, full_tp = confusion_tensor.sum(axis=0).flatten()
            micro_precision = full_tp / (full_tp + full_fp)
            micro_recall = full_tp / (full_tp + full_fn)

            # fig, axis = plt.subplots()
            #
            # names = ['TN', 'FP', 'FN', 'TP']
            # counts = global_detection_confusion_matrix.flatten()
            # labels = [f'{n}\n{c}' for n, c in zip(names, counts)]
            # labels = np.array(labels).reshape(2, 2)
            # sn.heatmap(global_detection_confusion_matrix.astype(np.int), annot=labels, fmt='', xticklabels=False,
            #            yticklabels=False, linewidth=.5, cbar=False, ax=axis, cmap='Blues')
            # if shuffle_data_set:
            #     plt.title(f'Detection without time dependencies')
            # else:
            #     plt.title(f'Detection with time dependencies')
            # plt.show()

            tn, fp, fn, tp = global_detection_confusion_matrix.flatten()
            detection_precision = tp / (tp + fp)
            detection_recall = tp / (tp + fn)

            print('Results for: ', experiment_name)
            print(f'Precision: {detection_precision * 100:.2f}%')
            print(f'Recall: {detection_recall * 100:.2f}%')
            print(f'Micro-precision: {micro_precision * 100:.2f}%')
            print(f'Micro-recall: {micro_recall * 100:.2f}%')
            print(f'Avg time per test: {time_list.mean():.2f} sec')
            print(f'Total time: {time_list.sum():.2f} sec')

            # Saving Score Distributions
            results_dict = {
                'detection_results': detection_results,
                'global_confusion_matrix': global_detection_confusion_matrix,
                'confusion_tensor': confusion_tensor,
                'times': time_list,
            }
            experiment_save_name = experiment_name + 'results_dict.p'
            pickle.dump(results_dict,
                        open(path.join('..', '..', 'results', experiment_save_name), 'wb'))

print(f'Experiment completed at {strftime("%a, %d %b %Y %I:%M%p", localtime())}')
