<!-- !split -->
# Markov Chain Monte Carlo sampling
We have been emphasizing that everything is a pdf in the Bayesian approach. In particular, we studied parameter estimation for which we were interested in the posterior pdf of parameters $\boldsymbol{\theta}$ in a model $M$ given data $D$ and other information $I$

$$
p(\boldsymbol{\theta} | D, I) \equiv p(\boldsymbol{\theta}).
$$

<!-- !split -->
Suppose that this parametrized model can make predictions for some quantity $y = f(\boldsymbol{\theta})$ that was not part of the original data set $D$ used to constrain the model. The result of such a prediction is best represented by a posterior *predictive distribution** (ppd)

$$
\{f(\boldsymbol{\theta}) : \boldsymbol{\theta} \sim p(\boldsymbol{\theta} | D, I) \}.
$$

The ppd is the set of all predictions computed over likely values of the model parameters, i.e., drawing from the posterior pdf for $\boldsymbol{\theta}$. Let us express the expectation value of this prediction, which turns into a multidimensional integral

$$
\langle f(\boldsymbol{\theta}) \rangle = \int f(\boldsymbol{\theta}) p(\boldsymbol{\theta} | D,I) d \boldsymbol{\theta} \equiv \int g( \boldsymbol{\theta} ) d\boldsymbol{\theta}.
$$ 

<!-- !split -->
Note that this is much more involved than traditional calculations in which we would use a single vector of parameters, e.g., denoted $\boldsymbol{\theta}^*$, that we might have found by maximizing a likelihood. Instead, $\langle f( \boldsymbol{\theta} ) \rangle$ means that we do a multidimensional integral over the full range of possible $\boldsymbol{\theta}$ values, weighted by the probability density function, $p(\boldsymbol{\theta} |D,I)$ that we have inferred.

* This is a lot more work!
* The same sort of multidimensional integrals appear when we want to marginalize over a subset of parameters $\boldsymbol{\theta}_B$ to find a pdf for the rest, $\boldsymbol{\theta}_A$ 

$$
p(\boldsymbol{\theta}_A | D, I) = \int p(\boldsymbol{\theta}_A, \boldsymbol{\theta}_B | D, I) d\boldsymbol{\theta}_B.
$$

* An example of such a marginalization procedure would be the inference of the masses of binary black holes from gravitational-wave signals. In such a data analysis there are many (nuisance) parameters that characterize, e.g., background noise which should be integrated out.
* Therefore, in the Bayesian approach we will frequently encounter these multidimensional integrals. However, conventional methods for low dimensions (Gaussian quadrature or Simpson's rule) become inadequate rapidly with the increase of dimension.
* In particular, the integrals are problematic because the posterior pdfs are usually very small in much of the integration volume so that the relevant region has a very complicated shape.

<!-- !split -->
## Monte Carlo integration
To approximate such integrals one turns to Monte Carlo (MC) methods. The straight and naive version of MC integration evaluates the integral by randomly distributing $n$ points in the multidimensional volume $V$ of possible parameter values $\boldsymbol{\theta}$. These points have to cover the regions where $p( \boldsymbol{\theta} |D,I)$ is significantly different from zero. Then

<!-- !split -->

$$

\langle f( \boldsymbol{\theta} ) \rangle = \int_V g( \boldsymbol{\theta} ) d\boldsymbol{\theta} \approx V \langle g( \boldsymbol{\theta} ) \rangle 
\pm V \sqrt{ \frac{\langle g^2( \boldsymbol{\theta} ) \rangle - \langle g( \boldsymbol{\theta} ) \rangle^2 }{n} },

$$

where

$$

\langle g( \boldsymbol{\theta} ) \rangle = \frac{1}{n} \sum_{i=0}^{n-1} g(\boldsymbol{\theta}_i ), \qquad

\langle g^2( \boldsymbol{\theta} ) \rangle = \frac{1}{n} \sum_{i=0}^{n-1} g^2(\boldsymbol{\theta}_i )

$$

<!-- !split -->
### Example: One-dimensional integration

The average of a function $g(\theta)$ on $\theta \in [a,b]$ is

$$

\overline{g(\theta)} = \frac{1}{b-a} \int_a^b g(\theta) d\theta,

$$

from calculus. However, we can estimate $\overline{g(\theta)}$ by averaging over a set of random samples

$$

\overline{g(\theta)} \approx \frac{1}{n} \sum_{i=0}^{n-1} g(\theta_i).

$$

Let us consider the integral

$$

\langle f(\theta) \rangle = \int_a^b g(\theta) d\theta \approx 
\frac{b-a}{n} \sum_{i=0}^{n-1} g(\theta_i),

$$

where $b-a$ is the volume $V$.

<!-- !split -->
### Slow convergence

The main uncertainty lies in assuming that a Gaussian approximation is valid. Note the dependence on $a/\sqrt{n}$, which means that you can get a more precise answer by increasing $n$. However, the result only gets better very slowly. Each additional decimal point accuracy costs you a factor of 100 in $n$.

<!-- !split -->
The key problem is that too much time is wasted in sampling regions where $p( \boldsymbol{\theta} |D,I )$ is very small. Consider a situation in which the significant region of the pdf is concentrated in a $10^{-1}$ fraction of the full range for one of the parameters. With such a concentration in $m$ dimensions, the significant fraction of the total volume would be $10^{-m}$! This situation necessitates *importance sampling* which reweighs the integrand to more appropriately distribute points (e.g. the [VEGAS algorithm](https://en.wikipedia.org/wiki/VEGAS_algorithm)), but this is difficult to accomplish.

<!-- !split -->
The bottom line is that its not feasible to draw a series of independent random samples from $p ( \boldsymbol{\theta} | D,I )$ from large $\boldsymbol{\theta}$ dimensions. Independence means if $\boldsymbol{\theta}_0, \boldsymbol{\theta}_1, \ldots$ is the series, knowing $\boldsymbol{\theta}_1$ doesn't tell us anything about $\boldsymbol{\theta}_2$.

<!-- !split -->
However, the samples don't actually need to be independent. they just need to generate a distribution that is proportional to $p ( \boldsymbol{\theta} |D,I)$. E.g., a histogram of the samples should approximate the true distribution.

<!-- !split -->
## Markov Chain Monte Carlo
A solution is therefore to do a *random walk* in the parameter space of $\boldsymbol{\theta}$ such that the probability for being in a region is proportional to the value of $p( \boldsymbol{\theta} | D,I)$ in that region.
* The position $\boldsymbol{\theta}_{i+1}$ follows from $\boldsymbol{\theta}_i$ by a transition probability (kernel) $t ( \boldsymbol{\theta}_{i+1} | \boldsymbol{\theta}_i )$.
* The transition probability is *time independent*, which means that $t ( \boldsymbol{\theta}_{i+1} | \boldsymbol{\theta}_i )$ is always the same.

A sequence of points generated according to these rules is called a *Markov Chain* and the method is called Markov Chain Monte Carlo (MCMC).

<!-- !split -->
Before describing the most basic implementation of the MCMC, namely the Metropolis and Metropolis-Hastings algorithms, let us list a few state-of-the-art implementations and packages that are available in Python (and often other languages)

```{admonition} emcee:
  [emcee](https://emcee.readthedocs.io/en/latest/) {cite}`Foreman_Mackey_2013` is an MIT licensed pure-Python implementation of Goodman & Weare’s [Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler](http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml) {cite}`Goodman2010`
  ```
  
```{admonition} PyMC3:
  [PyMC3](https://docs.pymc.io/) is a Python package for Bayesian statistical modeling and probabilistic machine learning which focuses on advanced Markov chain Monte Carlo and variational fitting algorithms.
  ```
  
```{admonition} PyStan:
  [PyStan](https://pystan.readthedocs.io/en/latest/) provides an interface to [Stan](http://mc-stan.org/), a package for Bayesian inference using the No-U-Turn sampler, a variant of Hamiltonian Monte Carlo.
  ```
  
```{admonition} PyMultiNest:
  [PyMultiNest](https://johannesbuchner.github.io/PyMultiNest/) interacts with [MultiNest](https://github.com/farhanferoz/MultiNest) {cite}`Feroz2009`, a Nested Sampling Monte Carlo library.
  ```

We have been using emcee extensively in this course. It is based on ensemble samplers (many MCMC walkers) with affine-invariance. For more details, there is the paper (see above) and some [lecture notes](http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-16.html)


<!-- !split -->
## The Metropolis Hastings algorithm
The basic structure of the Metropolis (and Metropolis-Hastings) algorithm is the following:

1. Initialize the sampling by choosing a starting point $\boldsymbol{\theta}_0$.
2. Collect samples by repeating the following:
   1. Given $\boldsymbol{\theta}_i$, *propose* a new point $\boldsymbol{\phi}$, sampled from a proposal distribution $q( \boldsymbol{\phi} | \boldsymbol{\theta}_i )$. This proposal distribution could take many forms. However, for concreteness you can imagine it as a multivariate normal with mean given by $\boldsymbol{\theta}_i$ and variance $\boldsymbol{\sigma}^2$ specified by the user.
      * The transition density will (usually) give a smaller probability for visiting positions that are far from the current position.
      * The width $\boldsymbol{\sigma}$ determines the average step size and is known as the proposal width.
   2. Compute the Metropolis(-Hastings) ratio $r$ (defined below). Note that the second factor is equal to one if the proposal distribution is symmetric. It is then known as the Metropolis algorithm.
   3. Decide whether or not to accept candidate $\boldsymbol{\phi}$ for $\boldsymbol{\theta}_{i+1}$. 
      * If $r \geq 1$: accept the proposal position and set $\boldsymbol{\theta}_{i+1} = \boldsymbol{\phi}$.
      * If $r < 1$: accept the position with probability $r$ by sampling a uniform $\mathrm{U}(0,1)$ distribution (note that now we have $0 \leq r < 1$). If $u \sim \mathrm{U}(0,1) \leq r$, then $\boldsymbol{\theta}_{i+1} = \boldsymbol{\phi}$ (accept); else $\boldsymbol{\theta}_{i+1} = \boldsymbol{\theta}_i$ (reject). Note that the chain always grows since you add the current position again if the proposed step is rejected.
   4. Iterate until the chain has reached a predetermined length or passes some convergence tests.


<!-- !split -->
The Metropolis(-Hastings) ratio is

$$
    
    r = \frac{p( \boldsymbol{\phi} | D,I)}{p( \boldsymbol{\theta}_i | D,I)}
    \times \frac{q( \boldsymbol{\theta}_i | \boldsymbol{\phi} )}{q( \boldsymbol{\phi} | \boldsymbol{\theta}_i )}.
    
$$

* The Metropolis algorithm dates back to the 1950s in physics, but didn't become widespread in statistics until almost 1980.
* It enabled Bayesian methods to become feasible.
* Note, however, that nowadays there are much more sophisticated samplers than the original Metropolis one.

<!-- !split -->
## Visualizations of MCMC
* There are excellent javascript visualizations of MCMC sampling on the internet.
* A particularly useful set of interactive demos was created by Chi Feng, and is available on the github page: [The Markov-chain Monte Carlo Interactive Gallery](https://chi-feng.github.io/mcmc-demo/)
* An accessible introduction to MCMC, with simplified versions of Feng's visualizations, was created by Richard McElreath. It promotes Hamiltonian Monte Carlo and is available in a blog entry called [Markov Chains: Why Walk When You Can Flow?](http://elevanth.org/blog/2017/11/28/build-a-better-markov-chain/) 

<!-- !split -->
## Challenges in MCMC sampling
There is much to be written about challenges in performing MCMC sampling and diagnostics that should be made to ascertain that your Markov chain has converged (although it is not really possible to be 100% certain except in special cases.)

We will not focus on these issues here, but just list a few problematic pdfs:
* Correlated distributions that are very narrow in certain directions. (scaled parameters needed)
* Donut or banana shapes. (very low acceptance ratios)
* Multimodal distributions. (might easily get stuck in local region of high probability and completely miss other regions.)
