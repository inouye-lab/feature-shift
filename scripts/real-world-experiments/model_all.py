"""This is the workhorse for the real-world-dataset experiments. This exists in this repo purely to reproduce the real-world
experiments as seen in the NeurIPS 2020 paper. For more readable and understandable code, please see the fsd
package which has a more user friendly interface."""

import numpy as np
from scipy import stats
from sklearn import neighbors
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.decomposition import PCA
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

class DeepGaussianCopulaModel():
    def __init__(self, n_layers=2, **layer_kwargs):
        if layer_kwargs is None:
            layer_kwargs = {}
        self.n_layers = n_layers
        self.layer_kwargs = layer_kwargs

    def fit(self, x):
        layers = []
        for ii in range(self.n_layers):
            layer = SingleGaussianizeStep(**self.layer_kwargs)
            x = layer.fit_transform(x)
            layers.append(layer)
        self.layers_ = layers
        self._latent = x  # Just for debugging
        return self

    def log_prob(self, x, return_latent=False):
        # Transform
        log_prob = torch.zeros_like(x[:, 0])
        for layer in self.layers_:
            log_prob_layer, x = layer.log_prob(x, return_latent=True)
            log_prob += log_prob_layer

        # Base distribution probability
        if True:
            log_prob += torch.sum(self.layers_[0].standard_normal_.log_prob(x), dim=1)
        if return_latent:
            return log_prob, x
        else:
            return log_prob

    def sample(self, n_samples=None):
        if n_samples is None:
            ravel = True
            n_samples = 1
        else:
            ravel = False
        # Sample from base
        x = self.layers_[0].standard_normal_.sample([n_samples])
        for layer in np.flip(self.layers_):
            x = layer.inverse(x)
        if ravel:
            x = x.reshape(-1)
        return x


class SingleGaussianizeStep():
    def __init__(self, n_bins=10, alpha=10, lam_variance=0):
        self.n_bins = n_bins
        self.alpha = alpha
        self.lam_variance = lam_variance

    def fit(self, x):
        self.fit_transform(x)
        return self

    def fit_transform(self, x):
        all_latent = []
        # 1. PCA transform
        pca = PCA(random_state=0)
        pca.fit(x.detach().numpy())
        # assert np.isclose(np.abs(np.linalg.det(pca.components_)), 1), 'Should be close to one'
        Q_pca = torch.from_numpy(pca.components_)
        x = torch.mm(x, Q_pca.T)

        # 2. Independent normal cdf transform
        scale, loc = torch.std_mean(x, dim=0)
        ind_normal = Normal(loc, torch.sqrt(scale * scale + self.lam_variance))
        x = ind_normal.cdf(x)
        x = torch.clamp(x, 1e-10, 1 - 1e-10)

        # 3. Independent histogram transform
        if True:
            histograms = [
                TorchUnitHistogram(n_bins=self.n_bins, alpha=self.alpha).fit(x_col)
                for x_col in x.detach().T
            ]
            x = torch.cat(tuple(
                hist.cdf(x_col).reshape(-1, 1)
                for x_col, hist in zip(x.T, histograms)
            ), dim=1)
            # all_latent.append(x.detach().numpy())
            self.histograms_ = histograms

        # 4. Independent inverse standard normal transform
        if True:
            standard_normal = Normal(loc=torch.zeros_like(loc), scale=torch.ones_like(scale))
            x = standard_normal.icdf(x)
            self.standard_normal_ = standard_normal

        self.Q_pca_ = Q_pca
        self.ind_normal_ = ind_normal
        self._latent = x  # Just for debugging purposes
        return x

    def log_prob(self, x, return_latent=False):
        # 1. PCA
        log_prob = torch.zeros_like(x[:, 0])  # Orthogonal transform has logdet of 0
        x = torch.mm(x, self.Q_pca_.T)

        # 2. Ind normal
        log_prob += torch.sum(self.ind_normal_.log_prob(x), dim=1)  # Independent so sum
        x = self.ind_normal_.cdf(x)  # Transform
        x = torch.clamp(x, 1e-10, 1 - 1e-10)

        # 3. Histogram
        if True:
            log_prob += torch.sum(torch.cat(tuple(
                hist.log_prob(x_col).reshape(-1, 1)
                for x_col, hist in zip(x.T, self.histograms_)
            ), dim=1), dim=1)
            x = torch.cat(tuple(
                hist.cdf(x_col).reshape(-1, 1)
                for x_col, hist in zip(x.T, self.histograms_)
            ), dim=1)

        # 4. Inverse standard normal
        if True:
            x = self.standard_normal_.icdf(x)  # For log prob of inverse cdf must do inverse cdf first
            log_prob -= torch.sum(self.standard_normal_.log_prob(x), dim=1)  # Independent so sum

        if return_latent:
            return log_prob, x
        else:
            return log_prob

    def inverse(self, x):
        # 4. Inverse standard normal
        if True:
            x = self.standard_normal_.cdf(x)  # For log prob of inverse cdf must do inverse cdf first
            x = torch.clamp(x, 1e-10, 1 - 1e-10)

        # 3. Histogram
        if True:
            x = torch.cat(tuple(
                hist.icdf(x_col).reshape(-1, 1)
                for x_col, hist in zip(x.T, self.histograms_)
            ), dim=1)

        # 2. Ind normal
        x = self.ind_normal_.icdf(x)  # Transform

        # 1. PCA
        x = torch.mm(x, self.Q_pca_)
        return x


class TorchUnitHistogram():
    '''Assumes all data is unit norm'''

    def __init__(self, n_bins, alpha):
        self.n_bins = n_bins
        self.alpha = alpha

    def fit(self, x):
        x = x.numpy()
        # Do numpy stuff
        hist, bin_edges = np.histogram(x, bins=self.n_bins, range=[0, 1])
        hist = np.array(hist, dtype=float)  # Make float so we can add non-integer alpha
        hist += self.alpha  # Smooth histogram by alpha so no areas have 0 probability
        cum_hist = np.cumsum(hist)
        cum_hist = cum_hist / cum_hist[-1]  # Normalize cumulative histogram

        # Make torch tensors
        bin_edges = torch.from_numpy(bin_edges)
        # Makes the same length as bin_edges
        cdf_on_edges = torch.from_numpy(np.concatenate(([0], cum_hist)))

        # Compute scale and shift for every bin
        # a = (y2-y1)/(x2-y1)
        bin_scale = (
                (cdf_on_edges[1:] - cdf_on_edges[:-1])
                / (bin_edges[1:] - bin_edges[:-1])
        )
        # b = -a*x2 + y2
        bin_shift = -bin_scale * bin_edges[1:] + cdf_on_edges[1:]

        # Normalize bins by bin_edges
        self.bin_edges_ = bin_edges
        self.cdf_on_edges_ = cdf_on_edges
        self.bin_scale_ = bin_scale
        self.bin_shift_ = bin_shift
        return self

    def cdf(self, x):
        assert torch.all(torch.logical_and(x >= 0, x <= 1)), 'All inputs should be between 0 and 1'
        bin_idx = self._get_bin_idx(x)
        # Linear interpolate within the selected bin
        return self.bin_scale_[bin_idx] * x + self.bin_shift_[bin_idx]

    def icdf(self, x):
        assert torch.all(torch.logical_and(x >= 0, x <= 1)), 'All inputs should be between 0 and 1'
        bin_idx = self._get_inverse_bin_idx(x)
        # Linear interpolate within the selected bin
        return (x - self.bin_shift_[bin_idx]) / self.bin_scale_[bin_idx]

    def log_prob(self, x):
        # Find closest bin
        bin_idx = self._get_bin_idx(x)
        return torch.log(self.bin_scale_[bin_idx])

    def _get_bin_idx(self, x):
        return torch.floor(x.detach() * self.n_bins).clamp(0, self.n_bins - 1).type(torch.long)

    def _get_inverse_bin_idx(self, x):
        bin_idx = -torch.ones_like(x, dtype=torch.long)
        for ii, (left_edge, right_edge) in enumerate(zip(self.cdf_on_edges_[:-1], self.cdf_on_edges_[1:])):
            if ii == self.n_bins - 1:
                # Include right edge
                bin_idx[torch.logical_and(x >= left_edge, x <= right_edge)] = ii
            else:
                bin_idx[torch.logical_and(x >= left_edge, x < right_edge)] = ii
        assert torch.all(torch.logical_and(bin_idx >= 0, bin_idx < self.n_bins)), 'Bin indices incorrect'
        return bin_idx

class FeatureShiftDetection:

    def __init__(self, p, q, rng, detection_method, samples_generator,
                 n_bootstrap_runs, n_attacks, n_conditional_expectation, alpha, j_attack, n_neighbors=None,
                  attack_testing=True, threshold_vector=None, known_k=True):
        """
        :param p: the left split (clean data) ; n.b. this is the original p samples
        :param q: the right split (clean data) ; n.b. this is the original q samples
        :param rng: Random number generator,
        :param detection_method: 'model_based', models p and q with a Gaussian, calculates the KS stat between models
                                                for each X_-j
                                 'model_free', uses KNN to estimate X_-j directly from samples, and uses KS stat on the
                                               X_-j neighborhoods for p and q
                                  'score_method' uses the score_function to calculate distance between p and q
        :param samples_generator: a lambda function for creating p and q samples.
                                  should be of form that calling samples_generator() returns p, q

        :param n_bootstrap_runs:
        :param n_attacks: the number of attacks which will happen during testing. Also, n_no_attacks = n_attacks
        :param n_conditional_expectation: the number of samples used to approximate the X_-j conditional expectation
        :param alpha: the confidence threshold
        :param j_attack [array (shape=(n_features_attacked,)]: the features which the attack happened on
        :param n_neighbors: for model_based: the number of neighbors to use in KNN
        :param threshold_vector [array (shape=(n_dim,): the threshold for whether feature shift has
                                                        happened (score > threshold = shifted)
                                                        If this value is set in class initialization,
                                                        BS will be skipped and the set threshold_vector will be used
        """
        detection_method = detection_method.lower()
        self.detection_method = detection_method
        if detection_method == 'score_method' or detection_method == 'score-method':
            self.get_score = self.univariate_score_method
        elif detection_method == 'deep_univariate_score_method' or detection_method == 'deep_univariate_score_method':
            self.get_score = self.deep_univariate_score_method
        elif detection_method == 'model_based' or detection_method == 'model-based':
            self.get_score = self.model_based_KS
        elif detection_method == 'model_free' or detection_method == 'model-free':
            self.get_score = self.model_free_KS
            if n_neighbors is None:
                raise ValueError('Need a value for n_neighbors when performing model-free feature shift detection')
        elif detection_method == 'marginal' or detection_method == 'marginal_method':
            self.get_score = self.marginal_method

        else:
            raise NotImplementedError(f'{detection_method} type of feature shift detection method is not implemented')

        self.rng = rng
        torch.manual_seed(rng.randint(1000))  # sets pytorch random seed using passed in rng
        self.samples_generator = samples_generator
        self.n_samples = p.shape[0] + q.shape[0]
        self.n_dim = p.shape[1]
        self.n_bootstrap_runs = n_bootstrap_runs
        self.n_attacks = n_attacks
        self.n_neighbors = n_neighbors
        self.n_conditional_expectation = n_conditional_expectation
        j_attack = np.array(j_attack)
        if len(j_attack.shape) >= 2:
            j_attack = np.squeeze(j_attack)
        self.j_attack = j_attack
        self.alpha = alpha
        self.known_K = known_k
        if threshold_vector is None:
            self.threshold_vector, self.bonferroni_threshold_vector, self.bootstrap_distribution = \
                self.find_univariate_threshold(p, q)
        elif threshold_vector == 'multivariate':
            self.threshold_vector, self.bootstrap_distribution = \
            self.find_multivariate_threshold(p, q, feature_list=[*range(p.shape[1])])
        else:
            self.threshold_vector = threshold_vector

    # detection_method = 'marginal'
    def marginal_method(self, p, q):
        KS_vector = np.zeros(self.n_dim)
        for j_idx in range(self.n_dim):
            KS_vector[j_idx] += stats.ks_2samp(p[:, j_idx], q[:, j_idx])[0]
        return KS_vector

    # detection_method = 'model_based
    def model_based_KS(self, p, q):

        x_idxs = self.rng.choice(p.shape[0], size=self.n_conditional_expectation)
        p_samples = p[x_idxs]  # drawing n_cond_expect samples from p

        running_KS = np.zeros(self.n_dim)
        for sample in p_samples:
            for j_idx in range(self.n_dim):
                p_conditional_mean, p_conditional_var = self.get_conditional(p, sample, j_idx)
                q_conditional_mean, q_conditional_var = self.get_conditional(q, sample, j_idx)
                p_conditional_samples = self.rng.normal(loc=p_conditional_mean,
                                                        scale=np.sqrt(p_conditional_var), size=self.n_samples)
                q_conditional_samples = self.rng.normal(loc=q_conditional_mean,
                                                        scale=np.sqrt(q_conditional_var), size=self.n_samples)
                running_KS[j_idx] += stats.ks_2samp(p_conditional_samples, q_conditional_samples)[0]
        return running_KS / self.n_conditional_expectation

    # detection_method = 'model_free'
    def model_free_KS(self, p, q):
        if self.n_neighbors is None:
            raise ValueError('n_neighbors cannot be none')
        running_KS = np.zeros(self.n_dim)
        NN = neighbors.NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='kd_tree',
                                        metric='minkowski', p=2, n_jobs=-1)

        for j_idx in range(self.n_dim):
            # Removing j_attack from p,q
            p_not_j = np.delete(p, j_idx, axis=1)  # returns p_not_x_samp without the jth col
            q_not_j = np.delete(q, j_idx, axis=1)

            # Getting inner expectation samples
            p_samples_idxs = self.rng.choice(range(p.shape[0]),
                                             size=self.n_conditional_expectation)  # indexes of p samples
            p_samples_not_j = p_not_j[p_samples_idxs]

            # Finding Nearest neighbors
            A_idxs = NN.fit(p_not_j).kneighbors(p_samples_not_j, return_distance=False)
            B_idxs = NN.fit(q_not_j).kneighbors(p_samples_not_j, return_distance=False)
            A = p[A_idxs, j_idx]  # \hat{p}(x | x_not_j)
            B = q[B_idxs, j_idx]  # \hat{q}(x | x_not_j)

            for A_row, B_row in zip(A, B):
                running_KS[j_idx] += stats.ks_2samp(A_row, B_row)[0]

        return running_KS / self.n_conditional_expectation  # the average KS stat for n_conditional_expectation

    # detection_method = 'deep_univariate_score_method'
    def deep_univariate_score_method(self, p, q):
        '''Simplest deep density copula model'''
        # Estimate and form model for both p and q

        kwargs = dict(n_layers=2, n_bins=10, alpha=1)
        p_hat = DeepGaussianCopulaModel(**kwargs).fit(torch.from_numpy(p))
        q_hat = DeepGaussianCopulaModel(**kwargs).fit(torch.from_numpy(q))

        running_score = np.zeros(self.n_dim)
        for idx in range(self.n_conditional_expectation):
            for sample in [p_hat.sample(), q_hat.sample()]:
                sample.requires_grad_(True)

                # log_prob method only allows for n x d matrices (in this case n=1)
                log_p_sample = p_hat.log_prob(sample.reshape(1, -1))
                log_q_sample = q_hat.log_prob(sample.reshape(1, -1))
                p_grad = torch.autograd.grad(log_p_sample, sample)[0]  # grad returns a tensor inside a tuple, hence [0]
                q_grad = torch.autograd.grad(log_q_sample, sample)[0]

                score = (p_grad - q_grad).data.numpy() ** 2
                running_score += score

        return running_score / (self.n_conditional_expectation * 2)

    # detection_method = 'univaraite_score_method'
    def univariate_score_method(self, p, q):
        p_mean = p.mean(axis=0)
        p_cov = np.cov(p, rowvar=False)
        p_cov += 1e-5 * np.eye(p_cov.shape[0])
        q_mean = q.mean(axis=0)
        q_cov = np.cov(q, rowvar=False)
        q_cov += 1e-5 * np.eye(q_cov.shape[0])

        p_hat = MultivariateNormal(loc=torch.from_numpy(p_mean), covariance_matrix=torch.from_numpy(p_cov))
        q_hat = MultivariateNormal(loc=torch.from_numpy(q_mean), covariance_matrix=torch.from_numpy(q_cov))

        running_score = np.zeros(self.n_dim)
        for idx in range(self.n_conditional_expectation):
            for sample in [p_hat.sample(), q_hat.sample()]:
                sample.requires_grad_(True)

                log_p_sample = p_hat.log_prob(sample)
                log_q_sample = q_hat.log_prob(sample)

                p_grad = torch.autograd.grad(log_p_sample, sample)[0]  # grad returns a tensor inside a tuple, hence [0]
                q_grad = torch.autograd.grad(log_q_sample, sample)[0]

                score = (p_grad - q_grad).data.numpy() ** 2
                running_score += score

        return running_score / (self.n_conditional_expectation * 2)

    def multivariate_score_method(self, p, q, feature_list):
        # note: if feature list is all features, then this is just the joint score method
        p_mean = p.mean(axis=0)
        p_cov = np.cov(p, rowvar=False)
        p_cov += 1e-5 * np.eye(p_cov.shape[0])
        q_mean = q.mean(axis=0)
        q_cov = np.cov(q, rowvar=False)
        q_cov += 1e-5 * np.eye(q_cov.shape[0])

        p_hat = MultivariateNormal(loc=torch.from_numpy(p_mean), covariance_matrix=torch.from_numpy(p_cov))
        q_hat = MultivariateNormal(loc=torch.from_numpy(q_mean), covariance_matrix=torch.from_numpy(q_cov))

        running_score = 0
        for idx in range(self.n_conditional_expectation):
            for sample in [p_hat.sample(), q_hat.sample()]:
                sample.requires_grad_(True)

                log_p_sample = p_hat.log_prob(sample)
                log_q_sample = q_hat.log_prob(sample)

                p_grad = torch.autograd.grad(log_p_sample, sample)[0]  # grad returns a tensor inside a tuple, hence [0]
                q_grad = torch.autograd.grad(log_q_sample, sample)[0]

                p_score_vector = np.array(p_grad[feature_list])
                q_score_vector = np.array(q_grad[feature_list])

                score = np.sum((p_score_vector - q_score_vector)**2)
                running_score += score

        return running_score / (self.n_conditional_expectation * 2)

    def find_univariate_threshold(self, p, q):

        if self.alpha > 1 or self.alpha < 0:
            raise ValueError('alpha must be in [0,1]')

        score_matrix = np.empty(shape=(self.n_bootstrap_runs, self.n_dim))

        # BS sampling on M
        M = np.concatenate((p, q), axis=0)
        for B_idx in range(self.n_bootstrap_runs):
            X = M[self.rng.choice(M.shape[0], size=p.shape[0], replace=True)]  # p[:] samples from M with replacement
            Y = M[self.rng.choice(M.shape[0], size=q.shape[0], replace=True)]  # q[:] samples from M with replacement
            score_matrix[B_idx] = self.get_score(X, Y)

        score_matrix = np.sort(score_matrix, axis=0)
        threshold_vector = score_matrix[int(self.n_bootstrap_runs * (1-self.alpha))]  # (1-alpha)th percentile of dist
        # Bonferroni threshold is the (1-alpha/d)th percentile; for performing multiple statistical tests simultaneously
        bonferroni_threshold_vector = score_matrix[int(self.n_bootstrap_runs * (1 - self.alpha / self.n_dim))]

        return threshold_vector, bonferroni_threshold_vector, score_matrix

    def find_multivariate_threshold(self, p, q, feature_list):
        score_distribution = np.empty(self.n_bootstrap_runs)
        # BS sampling on M
        M = np.concatenate((p, q), axis=0)
        for B_idx in range(self.n_bootstrap_runs):
            X = M[self.rng.choice(M.shape[0], size=p.shape[0], replace=True)]  # p[:] samples from M with replacement
            Y = M[self.rng.choice(M.shape[0], size=q.shape[0], replace=True)]  # q[:] samples from M with replacement
            score_distribution[B_idx] = self.multivariate_score_method(X, Y, feature_list)

        score_distribution = np.sort(score_distribution)
        threshold = score_distribution[int(self.n_bootstrap_runs * (1-self.alpha))]  # (1-alpha)th percentile of dist
        return threshold, score_distribution

    @staticmethod
    def get_conditional(p, x, j_to_be_conditioned):
        """
        determines p(x_j | x_{-j})
        ref: https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf   page 40.
        :param p: The sample to be conditioned on
        :param x: The original distribution
        :param j_to_be_conditioned: The feature to condition on
        :return:  the normal mean and varaince of the univariate normal distribution: p(x_j | x_{-j})
        """
        x_nj = np.delete(x, j_to_be_conditioned)
        p_hat = p.copy()
        p_hat[:, [0, j_to_be_conditioned]] = p[:, [j_to_be_conditioned, 0]]  # makes it so j_tbc is the first column.

        p_hat_means = np.mean(p_hat, axis=0)

        p_hat_cov = np.cov(p_hat, rowvar=False)
        p_hat_cov_11 = p_hat_cov[0, 0]  # \Sigma_{11}
        p_hat_cov_12 = p_hat_cov[0, 1:]  # \Sigma_{12} = \Sigma_{21}.T
        p_hat_cov_22 = p_hat_cov[1:, 1:]  # \Sigma_{22}
        p_hat_cov_22_inv = np.linalg.inv(p_hat_cov_22)

        p_xj_given_nj_mean = p_hat_means[0] + p_hat_cov_12 @ p_hat_cov_22_inv @ (x_nj - p_hat_means[1:])
        p_xj_given_nj_cov = p_hat_cov_11 - p_hat_cov_12 @ p_hat_cov_22_inv @ p_hat_cov_12.T

        return p_xj_given_nj_mean, p_xj_given_nj_cov
