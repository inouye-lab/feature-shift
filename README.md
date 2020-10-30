# Feature Shift

Greetings! This repository is the code for the paper:

> Feature Shift Detection: Localizing Which Features Have Shifted via Conditional Distribution Tests
> Sean M. Kulinski, Saurabh Bagchi, David I. Inouye
> *Neural Information Processing Systems* (NeurIPS), 2020.

If you use this code, please do us a favor and cite this paper via:

```bibtext
@inproceedings{kulinski2020feature,
author = {Kulinski, Sean M and Bagchi, Saurabh and Inouye, David I.},
booktitle = {Neural Information Processing Systems},
title = {Feature Shift Detection: Localizing Which Features Have Shifted via Conditional Distribution Tests},
year = {2020}
}
```



## Quick Summary

> In many real world scenarios the data used by machine learning models shifts away from the distribution the models were trained on. Is there a way we can not only detect when this happens, but also localize that shift to specific features in the data?

Distribution shift is a very real and frequent problem in machine learning production environments. Recently, there has been much research looking into detecting when such a shift has happened, and our work extends this idea to not only detecting a shift, but also localizing the shift to specific problem features in the data. A simple example of this would be if a model was trained on a sensor network consisting of common sensors such as temperature, vibration, noise-levels, etc., and after a while, a couple sensors begin to malfunction and output incorrect values. When the shift is detected, if it could also be localized to these problem sensors, then the issue can be swiftly confirmed and remediated. 

Our goal for feature shift detection is to do exactly this. We use hypothesis testing to see if there is a discrepancy between the feature-wise conditional distributions of the training and query distribution. We perform this for all features and report the ones which have a discrepancy. To do this, we introduce a novel use test statistic based on the (Fisher) score function, which can compute the conditional distribution hypothesis test quite efficiently. 

## Basic Structure of  Feature Shift Detector module

The `fsd` module contain three main parts (in descending order):

1. `fsd.featureshiftdetector` - This submodule contains the main class `FeatureShiftDetector` which performs both detection and localization. Given a specified statistic instance and bootstrapping method ('time' or 'simple'), `.fit(*, *) ` to perform bootstrapping, and then `.detect_and_localize(*)` can be called to perform the shift detection and localization.
2. `fsd.divergence` - This submodule contains the various divergence methods used in the paper. This includes `FisherDivergence`, `ModelKS`, and `KnnKS`. Each divergence takes in a density model (or non-parametric method (i.e. KNN)), fits it on two data distributions (`.fit(*, *)`), and then calculates and returns the feature-wise test statistics (`.score_features(*)`). 
3. `fsd.models` - This submodule contains the models used to fit the data to a distribution (for brevity, we'll refer to KNN as model here). It includes `GaussianDensity`, `DeepDensity` (which uses iterative Gaussianization to fit a deep density model), and `Knn`. Each model has a `.fit(*)` which method which fits the model uses the provided training data, a `.sample(*)` method which samples from the model (KNN just samples from the training data), and a `.conditional_sample(*)` which performs conditional sampling using the provided point to be conditioned upon.

## Reproducing Experiments

In order to reproduce the experiments, first setup a python environment matching that seen in the `requirements.txt`. (*Note: python 3.7+ must be used.)* Then, call the desired experiment as seen in the `scripts/` folder. For example, to call the unknown-single-sensor experiment, perform the following command:

```
$ python unknown-multiple-sensors.py
```



## Contact

If you have any questions or issues, please reach out via my email:

> skulinsk AT purdue DOT edu

Cheers,
