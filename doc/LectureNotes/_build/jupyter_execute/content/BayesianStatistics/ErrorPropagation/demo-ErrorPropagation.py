#!/usr/bin/env python
# coding: utf-8

# # Learning from Data: Error propagation and nuisance parameters (demonstration)

# ## Import of modules

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy import special

# Not really needed, but nicer plots
import seaborn as sns
sns.set()
sns.set_context("talk")


# ## Nuisance parameters (I)

# ### A Bayesian billiard game
# Adapted from the blog post [Frequentism and Bayesianism II: When Results Differ](http://jakevdp.github.io/blog/2014/06/06/frequentism-and-bayesianism-2-when-results-differ/) 
# 
# This example of nuisance parameters dates all the way back to the posthumous [1763 paper](http://www.stat.ucla.edu/history/essay.pdf) written by Thomas Bayes himself. The particular version of this problem used here is borrowed from [Eddy 2004](ftp://selab.janelia.org/pub/publications/Eddy-ATG3/Eddy-ATG3-reprint.pdf).
# 
# The setting is a rather contrived game in which Alice and Bob bet on the outcome of a process they can't directly observe:
# 
# Alice and Bob enter a room. Behind a curtain there is a billiard table, which they cannot see, but their friend Carol can. Carol rolls a ball down the table, and marks where it lands. Once this mark is in place, Carol begins rolling new balls down the table. If the ball lands to the left of the mark, Alice gets a point; if it lands to the right of the mark, Bob gets a point.  We can assume for the sake of example that Carol's rolls are unbiased: that is, the balls have an equal chance of ending up anywhere on the table.  The first person to reach **six points** wins the game.
# 
# Here the location of the mark (determined by the first roll) can be considered a nuisance parameter: it is unknown, and not of immediate interest, but it clearly must be accounted for when predicting the outcome of subsequent rolls. If the first roll settles far to the right, then subsequent rolls will favor Alice. If it settles far to the left, Bob will be favored instead.

# Given this setup, here is the question we ask of ourselves:
# 
# > In a particular game, after eight rolls, Alice has five points and Bob has three points. What is the probability that Bob will go on to win the game?
# 
# Intuitively, you probably realize that because Alice received five of the eight points, the marker placement likely favors her. And given this, it's more likely that the next roll will go her way as well. And she has three opportunities to get a favorable roll before Bob can win; she seems to have clinched it.  But, **quantitatively**, what is the probability that Bob will squeak-out a win?

# #### A Naive Frequentist Approach
# Someone following a classical frequentist approach might reason as follows:
# 
# To determine the result, we need an intermediate estimate of where the marker sits. We'll quantify this marker placement as a probability $\alpha$ that any given roll lands in Alice's favor.  Because five balls out of eight fell on Alice's side of the marker, we can quickly show that the maximum likelihood estimate of $\alpha$ is given by:
# 
# $$
# \hat{\alpha} = 5/8
# $$
# 
# (This result follows in a straightforward manner from the [binomial likelihood](http://en.wikipedia.org/wiki/Binomial_distribution)). Assuming this maximum likelihood estimate, we can compute the probability that Bob will win, which is given by:

# $$
# p(B) = (1 - \hat{\alpha})^3
# $$

# That is, he needs to win three rolls in a row. Thus, we find that the following estimate of the probability:

# In[2]:


alpha_hat = 5. / 8.
freq_prob = (1 - alpha_hat) ** 3
print(f"Naive frequentist probability of Bob winning: {freq_prob:.2f}")
print(f"or\nOdds against Bob winning: {(1. - freq_prob) / freq_prob:.0f} to 1")


# So we've estimated using frequentist ideas that Alice will win about 17 times for each time Bob wins. Let's try a Bayesian approach next.

# #### Bayesian approach

# We can also approach this problem from a Bayesian standpoint. This is slightly more involved, and requires us to first define some notation.
# 
# We'll consider the following random variables:
# 
# - $B$ = Bob Wins;
# - $D$ = observed data, i.e. $D = (n_A, n_B) = (5, 3)$;
# - $I$ = other information that we have, e.g. concerning the rules of the game;
# - $\alpha$ = unknown probability that a ball lands on Alice's side during the current game.
# 
# We want to compute $p(B~|~D,I)$; that is, the probability that Bob wins given our observation that Alice currently has five points to Bob's three.
# 
# In the following, we will not explicitly include $I$ in the expressions for conditional probabilities. However, it should be assumed to be part of the known propositions, e.g. 
# $$p(B~|~D)\equiv p(B~|~D,I),$$
# $$p(\alpha) \equiv p(\alpha~|~I),$$ etc. 

# The general Bayesian method of treating nuisance parameters is *marginalization*, or integrating the joint probability over the entire range of the nuisance parameter. In this case, that means that we will first calculate the joint distribution
# 
# $$
# p(B,\alpha~|~D)
# $$
# 
# and then marginalize over $\alpha$ using the following identity:
# 
# $$
# p(B~|~D) \equiv \int_{-\infty}^\infty p(B,\alpha~|~D) {\mathrm d}\alpha
# $$
# 
# This identity follows from the definition of conditional probability, and the law of total probability: that is, it is a fundamental consequence of probability axioms and will always be true. Even a frequentist would recognize this; they would simply disagree with our interpretation of $p(\alpha|I)$ (appearing below) as being a measure of uncertainty of our own knowledge.

# ##### Building our Bayesian Expression

# To compute this result, we will manipulate the above expression for $p(B~|~D)$ until we can express it in terms of other quantities that we can compute.

# We'll start by applying the following definition of [conditional probability](http://en.wikipedia.org/wiki/Conditional_probability#Definition) to expand the term $p(B,\alpha~|~D)$:
# 
# $$
# p(B~|~D) = \int P(B~|~\alpha, D) P(\alpha~|~D) \mathrm{d}\alpha
# $$

# Next we use [Bayes' rule](http://en.wikipedia.org/wiki/Bayes%27_theorem) to rewrite $p(\alpha~|~D)$:
# 
# $$
# p(B~|~D) = \int p(B~|~\alpha, D) \frac{p(D~|~\alpha)p(\alpha)}{p(D)} \mathrm{d}\alpha
# $$

# Finally, using the same probability identity we started with, we can expand $p(D)$ in the denominator to find:
# 
# $$
# p(B~|~D) = \frac{\int p(B~|~\alpha,D) p(D~|~\alpha) p(\alpha) \mathrm{d}\alpha}{\int p(D~|~\alpha)p(\alpha) \mathrm{d}\alpha}
# $$

# Now the desired probability is expressed in terms of three quantities that we can compute. Let's look at each of these in turn:
# 
# - $p(B~|~\alpha,D)$: This term is exactly the frequentist likelihood we used above. In words: given a marker placement $\alpha$ and the fact that Alice has won 5 times and Bob 3 times, what is the probability that Bob will go on to six wins?  Bob needs three wins in a row, i.e. $p(B~|~\alpha,D) = (1 - \alpha) ^ 3$.
# - $p(D~|~\alpha)$: this is another easy-to-compute term. In words: given a probability $\alpha$, what is the likelihood of exactly 5 positive outcomes out of eight trials? The answer comes from the well-known [Binomial distribution](http://en.wikipedia.org/wiki/Binomial_distribution): in this case $p(D~|~\alpha) \propto \alpha^5 (1-\alpha)^3$
# - $p(\alpha)$: this is our prior on the probability $\alpha$. By the problem definition, we can assume that $\alpha$ is evenly drawn between 0 and 1.  That is, $p(\alpha)$ is a uniform probability distribution in the range from 0 to 1.

# Putting this all together, canceling some terms, and simplifying a bit, we find
# $$
# p(B~|~D) = \frac{\int_0^1 (1 - \alpha)^6 \alpha^5 \mathrm{d}\alpha}{\int_0^1 (1 - \alpha)^3 \alpha^5 \mathrm{d}\alpha}
# $$
# where both integrals are evaluated from 0 to 1.

# These integrals are special cases of the [Beta Function](http://en.wikipedia.org/wiki/Beta_function):
# $$
# \beta(n, m) = \int_0^1 (1 - t)^{n - 1} t^{m - 1} dt
# $$
# The Beta function can be further expressed in terms of gamma functions (i.e. factorials), but for simplicity we'll compute them directly using Scipy's beta function implementation:

# In[3]:


from scipy.special import beta
bayes_prob = beta(6 + 1, 5 + 1) / beta(3 + 1, 5 + 1)

print(f"p(B|D) = {bayes_prob:.2f}")
print(f"or\nBayesian odds against Bob winning: {(1. - bayes_prob) / bayes_prob:.0f} to 1")


# So we see that the Bayesian result gives us 10 to 1 odds, which is quite different than the 17 to 1 odds found using the frequentist approach. So which one is correct?

# #### Brute-force (Monte Carlo) approach

# For this type of well-defined and simple setup, it is actually relatively easy to use a Monte Carlo simulation to determine the correct answer. This is essentially a brute-force tabulation of possible outcomes: we generate a large number of random games, and simply count the fraction of relevant games that Bob goes on to win. The current problem is especially simple because so many of the random variables involved are uniformly distributed.  We can use the ``numpy`` package to do this as follows:

# In[4]:


np.random.seed(0)

# play 100000 games with randomly-drawn p, between 0 and 1
p = np.random.random(100000)

# each game needs at most 11 rolls for one player to reach 6 wins
rolls = np.random.random((11, len(p)))

# count the cumulative wins for Alice and Bob at each roll
Alice_count = np.cumsum(rolls < p, 0)
Bob_count = np.cumsum(rolls >= p, 0)

# sanity check: total number of wins should equal number of rolls
total_wins = Alice_count + Bob_count
assert np.all(total_wins.T == np.arange(1, 12))
print("(Sanity check passed)")


# In[5]:


# determine number of games which meet our criterion of (A wins, B wins)=(5, 3)
# this means Bob's win count at eight rolls must equal 3
good_games = Bob_count[7] == 3
print(f"Number of suitable games: {good_games.sum()} (out of {len(p)} simulated ones)")

# truncate our results to consider only these games
Alice_count = Alice_count[:, good_games]
Bob_count = Bob_count[:, good_games]

# determine which of these games Bob won.
# to win, he must reach six wins after 11 rolls.
bob_won = np.sum(Bob_count[10] == 6)
print("Number of these games Bob won: {0}".format(bob_won.sum()))

# compute the probability
mc_prob = bob_won.sum() * 1. / good_games.sum()
print(f"Monte Carlo Probability of Bob winning: {mc_prob:.2f}")
print(f"MC Odds against Bob winning: {(1. - mc_prob) / mc_prob:.0f} to 1")


# The Monte Carlo approach gives 10-to-1 odds on Bob, which agrees with the Bayesian result. Apparently, our naive frequentist approach above was flawed.

# #### Discussion

# This example shows different approaches to dealing with the presence of a nuisance parameter $\alpha$. The Monte Carlo simulation gives us a close brute-force estimate of the true probability (assuming the validity of our assumptions), which the Bayesian approach matches. The naive frequentist approach, by utilizing a single maximum likelihood estimate of the nuisance parameter $\alpha$, arrives at the wrong result.
# 
# We should emphasize that **this does not imply frequentism itself is incorrect**. The incorrect result above is more a matter of the approach being "naive" than it being "frequentist". There certainly exist frequentist methods for handling this sort of nuisance parameter – for example, it is theoretically possible to apply a transformation and conditioning of the data to isolate the dependence on $\alpha$ – but it's hard to find any approach to this particular problem that does not somehow take advantage of Bayesian-like marginalization over $\alpha$.

# Another potential point of contention is that the question itself is posed in a way that is perhaps unfair to the classical, frequentist approach. A frequentist might instead hope to give the answer in terms of null tests or confidence intervals: that is, they might devise a procedure to construct limits which would provably bound the correct answer in $100\times(1 - p)$ percent of similar trials, for some value of $p$ – say, 0.05. This might be classically accurate, but it doesn't quite answer the question at hand. 
# 
# In contrast, Bayesianism provides a better approach for this sort of problem: by simple algebraic manipulation of a few well-known axioms of probability within a Bayesian framework, we can straightforwardly arrive at the correct answer without need for other special expertise.

# ## Error propagation: prior information

# ### Example 3.6.2 in Sivia
# * Consider a Bragg peak amplitude that is proportional to the square of a complex structure function: $A = f^2$.
# * The amplitude is measured with an uncertainty $A = A_0 \pm \sigma_A$ from a least-squares fit to experimental data.
# * What is $f = f_0 \pm \sigma_f$?

# In[6]:


def A_posterior(A,A0,sigA):
    pA = np.exp(-(A-A0)**2/(2*sigA**2))
    return pA/np.max(pA)

# Wrong analysis
def f_likelihood(f,A0,sigA):
    sigf = sigA / (2*np.sqrt(A0))
    pf = np.exp(-(f-np.sqrt(A0))**2/(2*sigf**2))
    return pf/np.max(pf)

# Correct error propagation
def f_posterior(f,A0,sigA):
    pf = f*np.exp(-(f**2-A0)**2/(2*sigA**2))
    return pf/np.max(pf)


# In[7]:


for (A0,sigA) in [(9,1),(1,9),(-20,9)]:
    maxA = max(2*A0,3*sigA)
    A_arr = np.linspace(0.01,maxA)
    f_arr = np.sqrt(A_arr)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(A_arr,A_posterior(A_arr,A0,sigA))
    ax[1].plot(f_arr,f_posterior(f_arr,A0,sigA),label='Bayesian')
    if A0>0:
        ax[1].plot(f_arr,f_likelihood(f_arr,A0,sigA),'--',label='Naive')
    ax[0].set(xlabel='A',ylabel='p(A|D,I)')
    plt.text(0.55,0.8,f'$A={A0}$, $\sigma_A={sigA}$', transform=ax[0].transAxes,fontsize=16)
    ax[1].set(xlabel='f',ylabel='p(f|D,I)')
    ax[1].legend(loc='best')
    plt.tight_layout()
    fig.savefig(f'error_square_root_{A0}_{sigA}.png')


# In[8]:


fig.savefig('error_square_root.png')


# In[ ]:




