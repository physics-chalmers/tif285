#!/usr/bin/env python
# coding: utf-8

# # Exercise: Bayesian neural networks
# 
# Last revised: 25-Oct-2019 by Christian Forssén [christian.forssen@chalmers.se]

# ## A simple classification problem

# `scikit-learn` includes various random [sample generators](https://scikit-learn.org/stable/datasets/index.html#generated-datasets) that can be used to build artificial datasets of controlled size and complexity.
# 
# For example, [`make_blobs`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) generates two (or more) Gaussian distributions of data that correspond to different classes.

# In[2]:


from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split


# ### Data sets from `scikit-learn`

# In[3]:


X, t = datasets.make_blobs(n_samples=20, cluster_std = 1.0,                            centers=[(-1.0, 1.0), (1.0,-1.0)], n_features=2,random_state=2019)


# In[4]:


X_train, X_test, t_train, t_test =           train_test_split(X, t, test_size=0.5, random_state=10)


# * Plot the training data.

# ## Task 1: Logistic regression using `scikit-learn`

# Implement a logistic regression binary classifier using `scikit-learn`
# * Use an L2 regularizer with weight decay $\alpha = 1.0$.
# * Print the best-fit parameters.
# * Create a grid in the $(x_1, x_2)$-plane and plot the decision boundary ($y=0.5$) for the binary classifier together with both the training and the test data.
# * Add also levels that correspond to the activation $a=\pm1,\pm2$ (what class probabilities do these activations correspond to?).

# ## Task 2: Bayesian logistic regression using MCMC sampling

# Implement instead a Bayesian binary classifier by considering the probability distributions for the three parameters of the single neuron (bias $w_0$ and weights $w_1,w_2$).
# 1. You will need to define the single neuron as a function that takes data $\boldsymbol{x}$ and parameters $\boldsymbol{w}$ as input and returns the output $y$.
# 1. You will also need to define the log prior for the parameters (use a pdf that is consistent with the choice of an L2-regularizer in the logistic regression implementation) and a log-likelihood for the data.
# 1. Use an MCMC sampler to make draws from the posterior distribution of the neuron parameters and make a corner plot.
# 1. Use a subset of the samples ($\sim 50$) to make predictions on the $x_1,x_2$-grid that was created in task 1. 
#   - Extract the mean and the standard deviation of the predictions for these sampled neurons on the grid (remember that each sample correspond to a neuron with those specific parameters).
#   - Plot the decision boundaries for ~ten of those samples. 
# 1. Finally, compare the predictions (mean and standard deviation) of your Bayesian binary classifier with those from the logistic regression approach.

# ## Task 3: Bayesian logistic regression using Variational Inference

# To be added.

# In[ ]:




