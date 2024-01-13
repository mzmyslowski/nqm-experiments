# Is noisy quadratic model of training of deep neural networks realistic enough?
This repo contains the source code for the experiments conducted for my [bachelor's thesis](https://drive.google.com/file/d/1oWwokqQ2M9DVNyx7Ctv0KoTLOsZ1LZjQ/view?usp=sharing) done at University of Warsaw under the supervision of [dr Stanisław Jastrzębski](https://sjastrzebski.com/) from New York University (at that time). The thesis explores empirical properties of the Cross Entropy loss function during training of deep neural networks and tests them against a theoretical model of training.  

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Experiment Results](#experiment-results)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
## Introduction
The empirical risk function of a neural network is usually a highly non-convex function. Therefore, it is hard to obtain theoretical results about its optimisation that are meaningful
to practitioners without making strong assumptions. Although, it is not clear how such assumption should be made. In the bachelor's thesis, we suggest to base these assumptions on the
properties of the empirical risk function that are observed in practice. Based on a literature study, we distinguish five such properties:
1. importance of initial phase of training
2. quadratic/convex behaviour near minimum
3. tendency for wide minima to generalise better
4. correlation of learning rate and batch size with curvature of loss function
5. importance of covariance of gradient
   
The code in this repo computes the following metrics at each iteration of training of a chosen model that we use a proxies for these assumptions:
1. eigenvalues of the covariance of gradients
2. eigenvalues of the hessian matrix of the loss function
3. norm of the gradient of the loss function
4. cosine of the gradient with the eigenvalues of the hessian
5. weights norm
6. Cosine of the current weights and the initial ones
7. distance from the current weights to the initial ones
8. loss value on the train set and the test set
## Requirements
- torchvision==0.16.2
- numpy==1.26.3
- pandas==2.1.4
- scikit-learn==1.3.2
## Installation
```
pip install -r requirements.txt
```
## Usage
1. configure the experiments in `experiments_factory.json`. If you don't do it, an example will be used. Possible values can be found in the respective files in this repo (e.g. available models can be found in the `models_factory.py`.
2. Run the configured experiements:
```
python experiments_factory.py
```
3. Select a path where the experiment results should be saved.
4. After each iteration is finished the results are saved in the folder named by the experiment. 
## Experiment Results
[Quick video explainer of the main experiment results](https://www.youtube.com/watch?v=akB7jqVcsQU)
## Acknowledgments
Huge thanks to my supervisor [dr Stanisław Jastrzębski](https://sjastrzebski.com/) from New York University for patience and suggesting many exciting ideas and research papers that were used as basis for this bachelor's thesis.

