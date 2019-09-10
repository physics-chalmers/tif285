# start import modules
import numpy as np
import matplotlib.pyplot as plt
# end import modules

savefig=True

# start simulation
np.random.seed(999)         # for reproducibility
pH=0.6                       # biased coin
flips=np.random.rand(2**12) # simulates 4096 coin flips
heads=flips<pH              # boolean array, heads[i]=True if flip i is heads
# end simulation

# start bayesian setup
def prior(pH):
    p=np.zeros_like(pH)
    p[(0<=pH)&(pH<=1)]=1      # allowed range: 0<=pH<=1
    return p                # uniform prior
def likelihood(pH,data):
    N = len(data)
    no_of_heads = sum(data)
    no_of_tails = N - no_of_heads
    return pH**no_of_heads * (1-pH)**no_of_tails
def posterior(pH,data):
    p=prior(pH)*likelihood(pH,data)
    norm=np.trapz(p,pH)
    return p/norm
# end bayesian setup

# start plotting
pH=np.linspace(0,1,1000)
fig, axs = plt.subplots(nrows=4,ncols=3,sharex=True,sharey='row',figsize=(14,14))
axs_vec=np.reshape(axs,-1)
axs_vec[0].plot(pH,prior(pH))
for ndouble in range(11):
    ax=axs_vec[1+ndouble]
    ax.plot(pH,posterior(pH,heads[:2**ndouble]))
    ax.text(0.1, 0.8, '$N={0}$'.format(2**ndouble), transform=ax.transAxes)
for row in range(4): axs[row,0].set_ylabel('$p(p_H|D_\mathrm{obs},I)$')
for col in range(3): axs[-1,col].set_xlabel('$p_H$')
# end plotting

if savefig:
    fig.savefig('../fig/coinflipping_fig_1.png')

if not savefig:
    plt.show()

