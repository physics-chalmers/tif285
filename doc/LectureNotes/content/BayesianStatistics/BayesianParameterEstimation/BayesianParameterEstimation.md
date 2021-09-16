
<!-- !split -->
# Inference With Parametric Models
Inductive inference with parametric models is a very important tool in the natural sciences.
* Consider $N$ different models $M_i$ ($i = 1, \ldots, N$), each with a parameter vector $\boldsymbol{\theta}_i$. The number of parameters (length of $\boldsymbol{\theta}_i$) might be different for different models. Each of them implies a sampling distribution for possible data

$$

p(D|\boldsymbol{\theta}_i, M_i)

$$

* The likelihood function is the pdf of the actual, observed data ($D_\mathrm{obs}$) given a set of parameters $\boldsymbol{\theta}_i$:

$$

\mathcal{L}_i (\boldsymbol{\theta}_i) \equiv p(D_\mathrm{obs}|\boldsymbol{\theta}_i, M_i)

$$
* We may be uncertain about $M_i$ (model uncertainty),
* or uncertain about $\boldsymbol{\theta}_i$ (parameter uncertainty).



<!-- !split -->
```{Admonition} Parameter Estimation:
  :class: tip
  Premise: We have chosen a model (say $M_1$)
  
  $\Rightarrow$ What can we say about its parameters $\boldsymbol{\theta}_1$?
  ```
```{Admonition} Model comparison:
  :class: tip
  Premise: We have a set of different models $\{M_i\}$
  
  $\Rightarrow$ How do they compare with each other? Do we have evidence to say that, e.g. $M_1$, is better than $M_2$?
  ```
```{Admonition} Model adequacy:
  :class: tip
  Premise: We have a model $M_1$
  
  $\Rightarrow$ Is $M_1$ adequate?
  ```
```{Admonition} Hybrid Uncertainty:
  :class: tip
  Premise: Models share some common parameters: $\boldsymbol{\theta}_i = \{ \boldsymbol{\varphi}, \boldsymbol{\eta}_i\}$
  
  $\Rightarrow$ What can we say about $\boldsymbol{\varphi}$? (Systematic error is an example)
```


<!-- !split -->
## Parameter estimation
Overview comments:
* In general terms, "parameter estimation" in physics means obtaining values for parameters (constants) that appear in a theoretical model which describes data (exceptions to this general definition exist of course).
* Conventionally this process is known as "parameter fitting" and very often the goal is just to find the "best fit".
* We will interpret this task from our Bayesian point of view.
* In particular, our ambition is larger as we realize that the strength of inference is best expressed via probabilities (or pdf:s).
* We will also see how familiar ideas like "least-squares optimization" show up from a Bayesian perspective.




<!-- !split -->
### Bayesian parameter estimation
We will now consider the Bayesian approach to the very important task of model parameter estimation using statistical inference. 

Let us first remind ourselves what can go wrong in a fit. We have encountered both **underfitting** (model is not complex enough to describe the variability in the data) and **overfitting** (model tunes to data fluctuations, or terms are underdetermined causing them playing off each other). Bayesian methods can prevent/identify both these situations.

<img src="./figs/m1m2.png" width=600><p><em>Joint pdf for the masses of two black holes merging obtained from the data analysis of a gravitational wave signal. This representation of a joint pdf is known as a corner plot.  <div id="fig-gw"></div></em></p> 


<!-- !split -->
<!-- ===== Example: Measured flux from a star (single parameter) ===== -->
## Example: Measured flux from a star (single parameter)
Adapted from the blog [Pythonic Perambulations](http://jakevdp.github.io) by Jake VanderPlas.

Imagine that we point our telescope to the sky, and observe the light coming from a single star. Our physics model will be that the star's true flux is constant with time, i.e. that  it has a fixed value $F_\mathrm{true}$ (we'll also ignore effects like sky noise and other sources of systematic error). Thus, we have a single model parameter: $F_\mathrm{true}$.

We'll assume that we perform a series of $N$ measurements with our telescope, where the i:th measurement reports an observed photon flux $F_i$ and is accompanied by an error model given by $e_i$[^errors].
The question is, given this set of measurements $D = \{F_i\}_{i=0}^{N-1}$, and the statistical model $F_i = F_\mathrm{true} + e_i$, what is our best estimate of the true flux $F_\mathrm{true}$?

[^errors]: We'll make the reasonable assumption that errors are Gaussian. In a Frequentist perspective, $e_i$ is the standard deviation of the results of a single measurement event in the limit of repetitions of *that event*. In the Bayesian perspective, $e_i$ is the standard deviation of the (Gaussian) probability distribution describing our knowledge of that particular measurement given its observed value.

Because the measurements are number counts, a Poisson distribution is a good approximation to the measurement process:


```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import emcee
```

```python
np.random.seed(1)      # for repeatability
F_true = 1000          # true flux, say number of photons measured in 1 second
N = 50                 # number of measurements
F = stats.poisson(F_true).rvs(N)
                       # N measurements of the flux
e = np.sqrt(F)         # errors on Poisson counts estimated via square root
```

Now let's make a simple visualization of the "observed" data, see Fig. [fig:flux](#fig:flux).


```python
fig, ax = plt.subplots()
ax.errorbar(F, np.arange(N), xerr=e, fmt='ok', ecolor='gray', alpha=0.5)
ax.vlines([F_true], 0, N, linewidth=5, alpha=0.2)
ax.set_xlabel("Flux");ax.set_ylabel("measurement number");
```

<!-- <img src="fig/BayesianParameterEstimation/singlephotoncount_fig_1.png" width=400><p><em>Single photon counts (flux measurements). <div id="fig:flux"></div></em></p> -->
![<p><em>Single photon counts (flux measurements). <div id="fig:flux"></div></em></p>](./figs/singlephotoncount_fig_1.png)

These measurements each have a different error $e_i$ which is estimated from Poisson statistics using the standard square-root rule. In this toy example we know the true flux that was used to generate the data, but the question is this: given our measurements and statistical model, what is our best estimate of $F_\mathrm{true}$?

Let's take a look at the frequentist and Bayesian approaches to solving this.

### Simple Photon Counts: Frequentist Approach

We'll start with the classical frequentist maximum likelihood approach. Given a single observation $D_i = F_i$, we can compute the probability distribution of the measurement given the true flux $F_\mathrm{true}$ and our assumption of Gaussian errors

\begin{equation}
p(D_i | F_\mathrm{true}, I) = \frac{1}{\sqrt{2\pi e_i^2}} \exp \left( \frac{-(F_i-F_\mathrm{true})^2}{2e_i^2} \right).
\end{equation}

This should be read "the probability of $D_i$ given $F_\mathrm{true}$
equals ...". You should recognize this as a normal distribution with mean $F_\mathrm{true}$ and standard deviation $e_i$.

We construct the *likelihood function* by computing the product of the probabilities for each data point

\begin{equation}
\mathcal{L}(F_\mathrm{true}) = \prod_{i=1}^N p(D_i | F_\mathrm{true}, I),
\end{equation}

here $D = \{D_i\}$ represents the entire set of measurements. Because the value of the likelihood can become very small, it is often more convenient to instead compute the log-likelihood. 

*Notice.* 
In the following we will use $\log$ to denote the natural logarithm. We will write $\log_{10}$ if we specifically mean the logarithm with base 10.



Combining the previous two equations and computing the log, we have

\begin{equation}
\log\mathcal{L} = -\frac{1}{2} \sum_{i=1}^N \left[ \log(2\pi e_i^2) +  \frac{(F_i-F_\mathrm{true})^2}{e_i^2} \right].
\end{equation}

In this approach we will determine $F_\mathrm{true}$ such that the likelihood is maximized. At this pont we can note that that problem of maximizing the likelihood is equivalent to the minimization of the sum

\begin{equation}
\sum_{i=1}^N \frac{(F_i-F_\mathrm{true})^2}{e_i^2},
\end{equation}

which you should recognize as the chi-squared function encountered in the linear regression model.

Therefore, it is not surprising that this particular maximization problem can be solved analytically (i.e. by setting $d\log\mathcal{L}/d F_\mathrm{true} = 0$). This results in the following observed estimate of $F_\mathrm{true}$

\begin{equation}
F_\mathrm{est} = \frac{ \sum_{i=1}^N w_i F_i }{ \sum_{i=1}^N w_i}, \quad w_i = 1/e_i^2.
\end{equation}

Notice that in the special case of all errors $e_i$ being equal, this reduces to

\begin{equation}
F_\mathrm{est} = \frac{1}{N} \sum_{i=1} F_i.
\end{equation}

That is, in agreement with intuition, $F_\mathrm{est}$ is simply the mean of the observed data when errors are equal.

We can go further and ask what the error of our estimate is. In the frequentist approach, this can be accomplished by fitting a Gaussian approximation to the likelihood curve at maximum; in this simple case this can also be solved analytically (the sum of Gaussians is also a Gaussian). It can be shown that the standard deviation of this Gaussian approximation is $\sigma_\mathrm{est}$, which is given by

\begin{equation}
\frac{ 1 } {\sigma_\mathrm{est}^2} = \sum_{i=1}^N w_i .
\end{equation}

These results are fairly simple calculations; let's evaluate them for our toy dataset:


```
w=1./e**2
print(f"""
F_true = {F_true}
F_est = {(w * F).sum() / w.sum():.0f} +/- { w.sum() ** -0.5:.0f} (based on {N} measurements) """)
```

`F_true = 1000` <br/>
`F_est = 998 +/- 4 (based on 50 measurements)` <br/>

We find that for 50 measurements of the flux, our estimate has an error of about 0.4% and is consistent with the input value.


### Simple Photon Counts: Bayesian Approach

The Bayesian approach, as you might expect, begins and ends with probabilities. Our hypothesis is that the star has a constant flux $F_\mathrm{true}$. It recognizes that what we fundamentally want to compute is our knowledge of the parameter in question given the data and other information (such as our knowledge of uncertainties for the observed values), i.e. in this case, $p(F_\mathrm{true} | D,I)$.
Note that this formulation of the problem is fundamentally contrary to the frequentist philosophy, which says that probabilities have no meaning for model parameters like $F_\mathrm{true}$. Nevertheless, within the Bayesian philosophy this is perfectly acceptable.

To compute this pdf, Bayesians next apply Bayes' Theorem.
If we set the prior $p(F_\mathrm{true}|I) \propto 1$ (a flat prior), we find
$p(F_\mathrm{true}|D,I) \propto p(D | F_\mathrm{true},I) \equiv \mathcal{L}(F_\mathrm{true})$
and the Bayesian probability is maximized at precisely the same value as the frequentist result! So despite the philosophical differences, we see that (for this simple problem at least) the Bayesian and frequentist point estimates are equivalent.

### A note about priors

The prior allows inclusion of other information into the computation, which becomes very useful in cases where multiple measurement strategies are being combined to constrain a single model. The necessity to specify a prior, however, is one of the more controversial pieces of Bayesian analysis.
A frequentist will point out that the prior is problematic when no true prior information is available. Though it might seem straightforward to use a noninformative prior like the flat prior mentioned above, there are some surprising
[subtleties](https://normaldeviate.wordpress.com/2013/07/13/lost-causes-in-statistics-ii-noninformative-priors/comment-page-1/)
involved. It turns out that in many situations, a truly noninformative prior does not exist! Frequentists point out that the subjective choice of a prior which necessarily biases your result has no place in statistical data analysis.
A Bayesian would counter that frequentism doesn't solve this problem, but simply skirts the question. Frequentism can often be viewed as simply a special case of the Bayesian approach for some (implicit) choice of the prior: a Bayesian would say that it's better to make this implicit choice explicit, even if the choice might include some subjectivity.

### Simple Photon Counts: Bayesian approach in practice

Leaving these philosophical debates aside for the time being, let's address how Bayesian results are generally computed in practice. For a one parameter problem like the one considered here, it's as simple as computing the posterior probability $p(F_\mathrm{true} | D,I)$ as a function of $F_\mathrm{true}$: this is the distribution reflecting our knowledge of the parameter $F_\mathrm{true}$.
But as the dimension of the model grows, this direct approach becomes increasingly intractable. For this reason, Bayesian calculations often depend on sampling methods such as Markov Chain Monte Carlo (MCMC). For this practical example, let us apply an MCMC approach using Dan Foreman-Mackey's [emcee](http://dan.iel.fm/emcee/current/) package. Keep in mind here that the goal is to generate a set of points drawn from the posterior probability distribution, and to use those points to determine the answer we seek.
To perform this MCMC, we start by defining Python functions for the prior $p(F_\mathrm{true} | I)$, the likelihood $p(D | F_\mathrm{true},I)$, and the posterior $p(F_\mathrm{true} | D,I)$, noting that none of these need be properly normalized. Our model here is one-dimensional, but to handle multi-dimensional models we'll define the model in terms of an array of parameters $\boldsymbol{\theta}$, which in this case is $\boldsymbol{\theta} = [F_\mathrm{true}]$


```python
def log_prior(theta):
    if theta>0 and theta<10000:
        return 0 # flat prior
    else:
        return -np.inf

def log_likelihood(theta, F, e):
    return -0.5 * np.sum(np.log(2 * np.pi * e ** 2) \ 
                             + (F - theta[0]) ** 2 / e ** 2)
                             
def log_posterior(theta, F, e):
    return log_prior(theta) + log_likelihood(theta, F, e)
```

Now we set up the problem, including generating some random starting guesses for the multiple chains of points.


```python
ndim = 1      # number of parameters in the model
nwalkers = 50 # number of MCMC walkers
nwarm = 1000  # "warm-up" period to let chains stabilize
nsteps = 2000 # number of MCMC steps to take
# we'll start at random locations between 0 and 2000
starting_guesses = 2000 * np.random.rand(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[F,e])
sampler.run_mcmc(starting_guesses, nsteps)
# Shape of sampler.chain  = (nwalkers, nsteps, ndim)
# Flatten the sampler chain and discard warm-in points:
samples = sampler.chain[:, nwarm:, :].reshape((-1, ndim))
```

If this all worked correctly, the array sample should contain a series of 50,000 points drawn from the posterior. Let's plot them and check. See results in Fig. [fig:flux-bayesian](#fig:flux-bayesian).


```python
fig, ax = plt.subplots()
ax.hist(samples, bins=50, histtype="stepfilled", alpha=0.3, density=True)
ax.set_xlabel(r'$F_\mathrm{est}$')
ax.set_ylabel(r'$p(F_\mathrm{est}|D,I)$');
```

<img src="./figs/singlephotoncount_fig_2.png" width=600><p><em>Bayesian posterior pdf (represented by a histogram of MCMC samples) from flux measurements.<div id="fig:flux-bayesian"></div></em></p> 

<!-- !split -->
## Best estimates and credible intervals
The posterior distribution from our Bayesian data analysis is the key quantity that encodes our inference about the values of the model parameters, given the data and the relevant background information. Often, however, we wish to summarize this result with just a few numbers: the best estimate and a measure of its reliability. 

There are a few different options for this. The choice of the most appropriate one depends mainly on the shape of the posterior distribution:

<!-- !split -->
### Symmetric posterior pdfs

Since the probability (density) associated with any particular value of the parameter is a measure of how much we believe that it lies in the neighbourhood of that point, our best estimate is given by the maximum of the posterior pdf. If we denote the quantity of interest by $\theta$, with a posterior pdf $P =p(\theta|D,I)$, then the best estimate of its value $\theta_0$ is given by the condition $dP/d\theta|_{\theta=\theta_0}=0$. Strictly speaking, we should also check the sign of the second derivative to ensure that $\theta_0$ represents a maximum.

For a pdf that is symmetric around the mode $\theta_0$ we can find a positive number $\Delta \theta$ such that

$$
p(\theta_0-\Delta\theta < \theta < \theta_0+\Delta\theta | D,I) = \int_{\theta_0-\Delta\theta}^{\theta_0+\Delta\theta} p(\theta|D,I) d\theta = m.
$$

The single error bar $\Delta\theta$, or $\theta_0 \pm \Delta\theta$, then defines an $100 \times m\%$ *credible interval*.

<!-- !split -->
### Asymmetric posterior pdfs

While the maximum (mode) of the posterior ($\theta_0$) can still be regarded as giving the best estimate, the integrated probability mass is larger on one side than the other. Alternatively one can compute the mean value, $\langle \theta \rangle = \int \theta p(\theta|D,I) d\theta$, although this tends to overemphasise very long tails. The best option is probably a compromise that can be employed when having access to a large sample from the posterior (as provided by an MCMC), namely to give the median of this ensemble.

Furthermore, the concept of a single error bar does not seem appropriate in this case, as it implicitly entails the idea of symmetry. A good way of expressing the reliability with which a parameter can be inferred, for an asymmetric posterior pdf, is rather through specifying the *credible interval*. Since the area under the posterior pdf between $[\theta_1,\theta_2]$ is proportional to how much we believe that $\theta$ lies in that range, the shortest interval that encloses $m$ probability mass represents an $100 \times m\%$ credible interval for the estimate. Obviously we can choose to provide any degree-of-belief intervals that we think are relevant for the case at hand. Assuming that the posterior pdf has been normalized, to have unit area, we need to find $\theta_1$ and $\theta_2$ such that: 

$$
p(\theta_1 < \theta < \theta_2 | D,I) = \int_{\theta_1}^{\theta_2} p(\theta|D,I) d\theta = m.
$$

Note that oen can come up with other choices for the interval $[\theta_1,\theta_2]$ that still gives $m$ integrated probability mass. The choice for which the distance $\theta_2 - \theta_1$ is as small as possible is usually called the highest posterior density (HPD) interval.  

<!-- !split -->
### Multimodal posterior pdfs

We can sometimes obtain posteriors which are multimodal; i.e. contains several disconnected regions with large probabilities. There is no difficulty when one of the maxima is very much larger than the others: we can simply ignore the subsidiary solutions, to a good approximation, and concentrate on the global maximum. The problem arises when there are several maxima of comparable magnitude. What do we now mean by a best estimate, and how should we quantify its reliability? The idea of a best estimate and an error-bar, or even a credible interval, is merely an attempt to summarize the posterior with just two or three numbers; sometimes this just can’t be done, and so these concepts are not valid. For the bimodal case we might be able to characterize the posterior in terms of a few numbers: two best estimates and their associated error-bars, or disjoint credible intervals. For a general multimodal pdf, the most honest thing we can do is just display the posterior itself.

Two options for assigning credible intervals to asymmetric and multimodal pdfs:
* Equal-tailed interval: the probability area above and below the interval are equal.
* Highest posterior density (HPD) interval: The posterior density for any point within the interval is larger than the posterior density for any point outside the interval.



<!-- !split -->
### Different views on credible/confidence intervals

A Bayesian credible interval, or degree-of-belief (DOB) interval, is the following: 

```{admonition} Bayesian credible interval
  *Given this data and other information there is $d \%$ probability that this interval contains the true value of the parameter.*
  ```

E.g. a 95% DOB interval implies that the Baysian data analyser would bet 20-to-1 that the true result is inside the interval.



<!-- !split -->
A frequentist $d \%$ *confidence interval* should be understood as follows: 

```{admonition} Frequentist confidence interval
  *There is a $d \%$ probability that when I compute a confidence interval from data of this sort that he true value of the parameter will fall within the (hypothetical) space of observations*
  ```

So the parameter is fixed (no pdf) and the confidence interval is based on random sampling of data. 

Let's try again to understand this for the special case of a 95% confidence interval: If we make a large number of repeated samples, then 95% of the intervals extracted in this way will include the true value of the parameter.



### Simple Photon Counts: Best estimates and credible intervals

To compute these numbers for our example, you would run:


```python
sampper=np.percentile(samples, [2.5, 16.5, 50, 83.5, 97.5],axis=0).flatten()
print(f"""
F_true = {F_true}
Based on {N} measurements the posterior point estimates are:
...F_est = { np.mean(samples):.0f} +/- { np.std(samples):.0f}
or using credibility intervals:
...F_est = {sampper[2]:.0f}          (posterior median) 
...F_est in [{sampper[1]:.0f}, {sampper[3]:.0f}] (67% credibility interval) 
...F_est in [{sampper[0]:.0f}, {sampper[4]:.0f}] (95% credibility interval) """)
```

`F_true = 1000` <br/>
`Based on 50 measurements the posterior point estimates are:` <br/>
`...F_est = 998 +/- 4` <br/>
`or using credibility intervals:` <br/>
`...F_est = 998          (posterior median)`  <br/>
`...F_est in [993, 1002] (67% credibility interval)`  <br/>
`...F_est in [989, 1006] (95% credibility interval)`  <br/>

In this particular example, the posterior pdf is actually a Gaussian (since it is constructed as a product of Gaussians), and the mean and variance from the quadratic approximation will agree exactly with the frequentist approach.

From this final result you might come away with the impression that the Bayesian method is unnecessarily complicated, and in this case it certainly is. Using an MCMC sampler to characterize a one-dimensional normal distribution is a bit like using the Death Star to destroy a beach ball, but we did this here because it demonstrates an approach that can scale to complicated posteriors in many, many dimensions, and can provide nice results in more complicated situations where an analytic likelihood approach is not possible.

Furthermore, as data, prior information, and models grow in complexity, the two approaches can diverge greatly. 



<!-- !split -->
## Example: Gaussian noise and averages
The example in the demonstration notebook is from Sivia's book. How do we infer the mean and standard deviation from $M$ measurements $D \in \{ x_k \}_{k=0}^{M-1}$ that should be distributed according to a normal distribution $p( D | \mu,\sigma,I)$?

<!-- !split -->
Start from Bayes theorem

$$
p(\mu,\sigma | D, I) = \frac{p(D|\mu,\sigma,I) p(\mu,\sigma|I)}{p(D|I)}
$$

* Remind yourself about the names of the different terms.
* It should become intuitive what the different probabilities (pdfs) describe.
* Bayes theorem tells you how to flip from (hard-to-compute) $p(\mu,\sigma | D, I) \Leftrightarrow p(D|\mu,\sigma,I)$ (easier-to-compute).

<!-- !split -->
Aside on the denominator, which is known as the "data probability" or "marginalized likelihood" or "evidence". 
* With $\theta$ denoting a general vector of parameters we must have

$$
p(D|I) = \int d\theta p(D|\theta,I) p(\theta|I).
$$

* This integration (or marginalization) over all parameters is often difficult to perform.
* Fortunately, for **parameter estimation** we don't need $p(D|I)$ since it doesn't depend on $\theta$. We usually only need relative probabilities, or we can determine the normalization $N$ after we have computed the unnormalized posterior 

$$
p(\theta | D,I) = \frac{1}{N} p(D|\theta,I) p(\theta|I).
$$

<!-- !split -->
If we use a uniform prior $p(\theta | I ) \propto 1$ (in a finite volume), then the posterior is proportional to the **likelihood**

$$
p(\theta | D,I) \propto p(D|\theta,I) = \mathcal{L}(\theta)
$$

In this particular situation, the mode of the likelihood (which would correspond to the point estimate of maximum likelihood) is equivalent to the mode of the posterior pdf in the Bayesian analysis.

<!-- !split -->
The real use of the prior, however, is to include into the analysis any additional information that you might have. The prior statement makes such additional assumptions and information very explicit.

But how do we actually compute the posterior in practice. Most often we won't be able to get an analytical expression, but we can sample the distribution using a method known as Markov Chain Monte Carlo (MCMC).

<!-- !split -->
## Example: Fitting a straight line
The next example that we will study is the well known fit of a straight line.

* Here the theoretical model is

$$
y_\mathrm{th}(x; \theta) = m x + b,
$$

with parameters $\theta = [b,m]$. The theoretical model is related to reality via the statistical model

$$
y_{i} = y_{\mathrm{th},i} + \varepsilon_i, 
$$

where we often assume that the experimental errors are independent and normally distributed (with a standard deviation $e_i$) so that

$$
y_i \sim \mathcal{N} \left( y_\mathrm{th}(x_i; \theta), e_i^2 \right).
$$


* The statistical model for the data is

$$
y_{\mathrm{exp},i} = y_{i} + \delta y_{\mathrm{exp},i},
$$

* Are independent errors always a good approximation?

<!-- !split -->
### Linear regression revisited
At this point it is instructive to revisit the linear regression method that we started out with. It corresponds to models that are linear in the parameters such that

$$
y_\mathrm{th} = \sum_{j=0}^{p-1} \theta_j g_j(x),
$$

with $p$ parameters and $g_j(x)$ denoting the basis functions.

With a likelihood as before

$$
p(D|\theta,I) = \prod_{i=0}^{N-1} \exp \left[ -\frac{\left(y_i - y_\mathrm{th}(x_i;\theta) \right)^2}{2\sigma_i^2} \right],
$$

and assuming a Gaussian prior with a single width $\sigma_\theta$ on the parameters

$$
p(\theta|I) \propto \prod_{j=0}^{p-1} \exp \left[ -\frac{\theta_j^2}{2\sigma_\theta^2} \right].
$$

We note that the prior can be written $\exp\left( -|\theta|^2 / 2 \sigma_\theta^2\right)$, such that the log (unnormalized) posterior becomes

$$
\log \left[ p(\theta|D,I) \right] = -\frac{1}{2} \left[ \sum_{i=0}^{N-1} \left( \frac{ y_i - y_\mathrm{th}(x_i;\theta)}{\sigma_i}\right)^2 + \frac{|\theta|^2}{\sigma_\theta^2} \right].
$$

The mode of the posterior pdf occurs at the minimum of this log-posterior function. You might recognise it as the modified cost function that we introduced in a rather *ad hoc* fashion when implementing linear regression with Ridge regularisation.  From our Bayesian perspective, linear regression with Ridge regularisation corresponds to the maximum a posteriori (MAP) estimate with a Gaussian prior on the parameters.


<!-- !split -->
### Why normal distributions?
Let us give a quick motivation why Gaussian distributions show up so often. In fact, most distributions look like a Gaussian one in the vicinity of a mode (as will be shown below). Replacing a general pdf by a Gaussian one is called the *Laplace approximation*. It helps by providing analytical expressions for various quantities, but it is not always a very good approximations as you move further away from the mode.

#### One-dimensional normal distributions
The expression for a one-dimensional normal distribution is

$$
p(\theta) = \frac{1}{N} \exp \left[ -\frac{1}{2} \frac{(\theta-\theta_0)^2}{\sigma^2} \right],
$$

where the normalization is $N=\sqrt{2\pi\sigma^2}$.

The mode of this distribution is at $\theta=\theta_0$. The probability density at the mode is $p(\theta_0) = 1/N$, while it is a factor $e^{-1/2}$ smaller a distance $\sigma$ away. Furthermore, the integral

$$
p(\theta_0-\sigma < \theta < \theta_0+\sigma | D,I) = \int_{\theta_0-\sigma}^{\theta_0+\sigma} p(\theta|D,I) d\theta \approx 0.68,
$$

which implies that the interval $[\theta_0-\sigma, \theta_0+\sigma]$ is a Bayesian 68% credible interval.



To obtain a measure of the reliability of this best estimate, we need to look at the width or spread of the posterior pdf about $\theta_0$. The linear term is zero at the maximum and the quadratic term is often the dominating one determining the width of the posterior pdf. Ignoring all the higher-order terms we arrive at the Gaussian approximation (see more details below)

\begin{equation}
p(\theta|D,I) \approx \frac{1}{\sigma\sqrt{2\pi}} \exp \left[ -\frac{(\theta-\mu)^2}{2\sigma^2} \right],
\end{equation}

where the mean $\mu = \theta_0$ and the variance $\sigma = \left( - \left. \frac{d^2L}{d\theta^2} \right|_{\theta_0} \right)^{-1/2}$, where $L$ is the logarithm of the posterior $P$. Our inference about the quantity of interest is conveyed very concisely, therefore, by the 67% Bayesian credible interval $\theta = \theta_0 \pm \sigma$, and 

$$
p(\theta_0-\sigma < \theta < \theta_0+\sigma | D,I) = \int_{\theta_0-\sigma}^{\theta_0+\sigma} p(\theta|D,I) d\theta \approx 0.67.
$$


#### The Laplace approximation

Say that we have a general pdf $p(\theta | D,I)$ with a mode at $\theta = \theta_0$ where

$$ 
\left. 
\frac{ \partial p }{ \partial \theta }
\right|_{\theta=\theta_0} = 0, \qquad
\left. \frac{ \partial^2 p }{ \partial \theta^2 }
\right|_{\theta=\theta_0} < 0.
$$

The distribution usually varies very rapidly so we study $L(\theta) \equiv \log \left( p(\theta) \right)$ instead.

When considering the behaviour of any function in the neighbourhood of a particular point, it is often helpful to carry out a Taylor series expansion; this is simply a standard tool for (locally) approximating a complicated function by a low-order polynomial. Near the peak, our pdf behaves as

$$
L(\theta) = L(\theta_0) + \frac{1}{2} \left. \frac{\partial^2 L}{\partial \theta^2} \right|_{\theta_0} \left( \theta - \theta_0 \right)^2 + \ldots,
$$

where the first-order term is zero since we are expanding around a maximum and $\partial L / \partial\theta = 0$. The second term is negative since the curvature is negative at the peak,which we indicate by writing $-\frac{1}{2} \left| \frac{\partial^2 L}{\partial \theta^2} \right|_{\theta=\theta_0}$. Furthermore, higher order terms can be neglected **if**

$$
(\theta-\theta_0)^k \frac{ \partial^{2+k} L }{ \partial \theta^{2+k} } \ll \frac{ \partial^2 L }{ \partial \theta^2 },
$$

for $k=1,2,\ldots$. This will often be true for small distances $\theta-\theta_0$.

<!-- !split -->
Consequently, if we neglect higher-order terms we find that 

$$
p(\theta|D,I) \approx A \exp \left[ -\frac{1}{2} \left| \frac{\partial^2 L}{\partial \theta^2} \right|_{\theta=\theta_0} \left( \theta - \theta_0 \right)^2  \right],
$$

which is a Gaussian $\mathcal{N}(\mu,\sigma^2)$ with

$$
\mu = \theta_0, \qquad \frac{1}{\sigma^2} = \left| \frac{\partial^2 L}{\partial \theta^2} \right|_{\theta=\theta_0} .
$$

<!-- !split -->
### Correlations
In the "fitting a straight-line" example you should find that the joint pdf for the slope and the intercept $[m, b]$ corresponds to a slanted ellipse. That result implies that the model parameters are **correlated**.

* Try to understand the correlation that you find in this example.

Let us explore correlations by studying the behavior of a bivariate pdf near the maximum where we employ the Laplace approximation (neglecting terms beyond the quadratic one in a Taylor expansion). We start by considering two independent parameters $x$ and $y$, before studying the dependent case.

#### Two independent parameters

Independence implies that $p(x,y) = p_x(x) p_y(y)$. We will again consider the log-pdf $L(x,y) = \log\left( p(x,y) \right)$ which will then be

$$
L(x,y) = L_x(x) + L_y(y).
$$

At the mode we will have $\partial p / \partial x = \partial p_x / \partial x = \partial L_x / \partial x = 0$, and similarly $\partial L_y / \partial y = 0$.

The second derivatives will be

$$
A \equiv \left. \frac{\partial^2 L_x}{\partial x^2} \right|_{x=x_0} < 0, \quad
B \equiv \left. \frac{\partial^2 L_y}{\partial y^2} \right|_{y=y_0} < 0, \quad
C \equiv \left. \frac{\partial L(x,y)}{\partial x \partial y} \right|_{x=x_0,y=y_0} = 0.
$$

such that our approximated (log) pdf near the mode will be

$$
L(x,y) = L(x_0, y_0) - \frac{1}{2} |A| (x-x_0)^2 - \frac{1}{2} |B| (y-y_0)^2.
$$

We could visualize this bivariate pdf by plotting iso-probability contours. Or, equivalently, iso-log-probability contours which correspond to

$$
|A| (x-x_0)^2 + |B| (y-y_0)^2 = \mathrm{constant}.
$$

This you should recognize as the equation for an ellipse with its center in $(x_0, y_0)$, the principal axes corresponding to the $x$ and $y$ directions, and with the width and height parameters $\sigma_x = 1/\sqrt{|A|}$ and $\sigma_y = 1/\sqrt{|B|}$, respectively.

#### Two dependent parameters

For two dependent parameters we cannot separate $p(x,y)$ into a product of one-dimensional pdf:s. Instead, the Taylor expansion for the bivariate log-pdf $L(x,y)$ around the mode $(x_0,y_0)$ gives

$$
L(x,y) \approx L(x_0,y_0) + \frac{1}{2} \begin{pmatrix} x-x_0 & y-y_0 \end{pmatrix}
H
\begin{pmatrix} x-x_0 \\ y-y_0 \end{pmatrix},
$$

where $H$ is the symmetric Hessian matrix

$$
\begin{pmatrix}
A & C \\ C & B
\end{pmatrix}, 
$$

with elements

$$
A = \left. \frac{\partial^2 L}{\partial x^2} \right|_{x_0,y_0} < 0, \quad
B = \left. \frac{\partial^2 L}{\partial y^2} \right|_{x_0,y_0} < 0, \quad
C = \left. \frac{\partial^2 L}{\partial x \partial y} \right|_{x_0,y_0} \neq 0.
$$

<!-- !split -->
* So in this quadratic approximation the contour is an ellipse centered at $(x_0,y_0)$ with orientation and eccentricity determined by $A,B,C$.
* The principal axes are found from the eigenvectors of $H$.
* Depending on the skewness of the ellipse, the parameters are either (i) not correlated, (ii) correlated, or (iii) anti-correlated.
* Take a minute to consider what that implies.

Let us be explicit. The Hessian can be diagonalized (we will also change sign)

$$
-H = U \begin{pmatrix} a & 0 \\ 0 & b \end{pmatrix} U^{-1},
$$

where $a$, $b$ are the (positive) eigenvalues of $-H$, and $U = \begin{pmatrix} a_x & b_x \\ a_y & b_y \end{pmatrix}$ is constructed from the eigenvectors. Defining a new set of translated and linearly combined parameters

$$
x' = a_x (x - x_0) + a_y (y - y_0) \\
y' = b_x (x - x_0) + b_y (y - y_0) 
$$

we find that the pdf becomes independent in this new pair of parameters

$$
L(x',y') = L(0,0) - \frac{1}{2} \begin{pmatrix} x' & y' \end{pmatrix}
\begin{pmatrix} a & 0 \\ 0 & b \end{pmatrix}
\begin{pmatrix} x' \\ y' \end{pmatrix} \\
\qquad = L(0, 0) - \frac{1}{2} a (x')^2 - \frac{1}{2} b (y')^2.
$$

* Take a minute to consider what has been achieved by this change of variables.
