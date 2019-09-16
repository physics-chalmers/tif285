# start import modules
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import emcee
# end import modules

savefig=True

# start generate data
np.random.seed(1)      # for repeatability
F_true = 1000          # true flux, say number of photons measured in 1 second
N = 50                 # number of measurements
F = stats.poisson(F_true).rvs(N)
                       # N measurements of the flux
e = np.sqrt(F)         # errors on Poisson counts estimated via square root
# end generate data

# start visualize data
fig, ax = plt.subplots()
ax.errorbar(F, np.arange(N), xerr=e, fmt='ok', ecolor='gray', alpha=0.5)
ax.vlines([F_true], 0, N, linewidth=5, alpha=0.2)
ax.set_xlabel("Flux");ax.set_ylabel("measurement number");
# end visualize data

if savefig:
    fig.savefig('../fig/singlephotoncount_fig_1.png')

# start frequentist
w=1./e**2
print(f"""
F_true = {F_true}
F_est = {(w * F).sum() / w.sum():.0f} +/- { w.sum() ** -0.5:.0f} (based on {N} measurements) """)
# end frequentist

# start bayesian setup
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
# end bayesian setup

# start bayesian mcmc
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
# end bayesian mcmc

# start visualize bayesian
fig, ax = plt.subplots()
ax.hist(samples, bins=50, histtype="stepfilled", alpha=0.3, density=True)
ax.set_xlabel(r'$F_\mathrm{est}$')
ax.set_ylabel(r'$p(F_\mathrm{est}|D,I)$');
# end visualize bayesian

if savefig:
    fig.savefig('../fig/singlephotoncount_fig_2.png')

# plot a best-fit Gaussian
F_est = np.linspace(975, 1025)
pdf = stats.norm(np.mean(samples), np.std(samples)).pdf(F_est)
ax.plot(F_est, pdf, '-k')

# start bayesian CI
sampper=np.percentile(samples, [2.5, 16.5, 50, 83.5, 97.5],axis=0).flatten()
print(f"""
F_true = {F_true}
Based on {N} measurements the posterior point estimates are:
...F_est = { np.mean(samples):.0f} +/- { np.std(samples):.0f}
or using credibility intervals:
...F_est = {sampper[2]:.0f}          (posterior median) 
...F_est in [{sampper[1]:.0f}, {sampper[3]:.0f}] (67% credibility interval) 
...F_est in [{sampper[0]:.0f}, {sampper[4]:.0f}] (95% credibility interval) """)
# end bayesian CI

if not savefig:
    plt.show()

