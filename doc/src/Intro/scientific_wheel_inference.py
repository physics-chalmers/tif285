import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon,Ellipse, Arc
from matplotlib.collections import PatchCollection

import matplotlib
from matplotlib import rc
matplotlib.rc('font', size = 18)
matplotlib.rc('savefig', bbox ='tight', dpi=300)
#matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
#matplotlib.rc('text', usetex=True)

lfont = 14
mfont = 10
# XKCD style
plt.xkcd()

figdir='fig/'

fig = plt.figure(1, figsize=(3,3))
ax = plt.gca()
#ax.grid()
plt.axis('off')

mainbox = dict(boxstyle="round4", fc="w", lw=2)

ax.annotate("Observation",
            xy=(0.88, 0.37), xycoords='data',
            xytext=(0.5, 0.85), textcoords='data',
            size=lfont, va="center", ha="center",
            bbox=mainbox, zorder=200,
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=-0.5",
                            relpos=(1,0.5)) 
            )

ax.annotate("Theory",
            xy=(0.18, 0.25), xycoords='data',
            xytext=(0.85, 0.3), textcoords='data',
            size=lfont, va="center", ha="center",
            bbox=mainbox, zorder=200)

ax.annotate("",
            xy=(0.18, 0.22), xycoords='data',
            xytext=(0.85, 0.3), textcoords='data',
            size=lfont, va="center", ha="center",
            bbox=mainbox,
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=-0.6")
            )
ax.annotate("Modeling",
            xy=(0.26, 0.84), xycoords='data',
            xytext=(0.15, 0.3), textcoords='data',
            size=lfont, va="center", ha="center",
            bbox=mainbox,zorder=200)

ax.annotate("",
            xy=(0.24, 0.84), xycoords='data',
            xytext=(0.15, 0.3), textcoords='data',
            size=lfont, va="center", ha="center",
            bbox=mainbox,
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=-0.52")
            )

#plt.show()
plt.savefig(figdir+'scientific_wheel.png',bbox_inches='tight',
            transparent=True, pad_inches=0,
            edgecolor='none')

# Keep adding text boxes to make it Nuclear Physics specific

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', edgecolor='green', facecolor='white',lw=1)
props_exp = dict(boxstyle='round', edgecolor='blue', facecolor='white',lw=1)

# Sample data and theory with errors

#ax_inset = fig.add_axes([0.16,0.49,0.2,0.2], frameon=False, axisbg='w')
ax_inset = fig.add_axes([0.35,0.45,0.2,0.2], frameon=False)

# example data
x = np.arange(0.2, 2, 0.5)
x_th = np.linspace(0.,1.9,100)
y = np.exp(-x)
# example error bar values that vary with x-position
error = 0.1 + 0.2 * x
# error bar values w/ different -/+ errors
lower_error = 0.4 * error
upper_error = error
asymmetric_error = [lower_error, upper_error]
y_th = np.exp(-2.*x_th)
y_th_lo = y_th-0.1
y_th_hi = y_th+0.1

ax_inset.fill_between(x_th,y_th_lo,y_th_hi, alpha=0.5,
                      facecolor='green', interpolate=True)
ax_inset.errorbar(x, y, yerr=error, fmt='o')


ax_inset.get_xaxis().set_visible(False)
ax_inset.get_yaxis().set_visible(False)

#plt.show()
plt.savefig(figdir+'scientific_wheel_data.png',bbox_inches='tight',
            transparent=True, pad_inches=0,
            edgecolor='none')

fig2, ax2 = plt.subplots(1,1, figsize=(6,1))
plt.axis('off')

mainbox = dict(boxstyle="round4", fc="w", lw=2)

ax2.annotate("Data",
            xy=(0.35, 0.5), xycoords='data',
            xytext=(0.1, 0.5), textcoords='data',
            size=lfont, va="center", ha="center",
            bbox=mainbox, zorder=200,
            arrowprops=dict(arrowstyle="fancy",
                            relpos=(1,0.5)) 
            )

ax2.annotate("Learning",
            xy=(0.7, 0.5), xycoords='data',
            xytext=(0.45, 0.5), textcoords='data',
            size=lfont, va="center", ha="center",
            bbox=mainbox, zorder=200,
            arrowprops=dict(arrowstyle="fancy",
                            relpos=(1,0.5)) 
            )

ax2.text(0.83,0.5,"Conclusions", 
            fontsize=lfont, va="center", ha="center",
            bbox=mainbox,zorder=200)

#plt.show()
fig2.savefig(figdir+'inference.png',bbox_inches='tight',
            transparent=True, pad_inches=0,
            edgecolor='none')


fig3, ax3 = plt.subplots(1,1, figsize=(6,1))
plt.axis('off')

mainbox = dict(boxstyle="round4", fc="w", lw=2)

ax3.annotate("Data",
            xy=(0.25, 0.5), xycoords='data',
            xytext=(0.1, 0.5), textcoords='data',
            size=lfont, va="center", ha="center",
            bbox=mainbox, zorder=200,
            arrowprops=dict(arrowstyle="fancy",
                            relpos=(1,0.5)) 
            )

ax3.annotate("Machine Learning\nAlgorithm",
            xy=(0.72, 0.5), xycoords='data',
            xytext=(0.45, 0.5), textcoords='data',
            size=lfont, va="center", ha="center",
            bbox=mainbox, zorder=200,
            arrowprops=dict(arrowstyle="fancy",
                            relpos=(1,0.5)) 
            )

ax3.text(0.85,0.5,"Predict/\nClassify/...", 
            fontsize=lfont, va="center", ha="center",
            bbox=mainbox,zorder=200)

#plt.show()
fig3.savefig(figdir+'MLinference.png',bbox_inches='tight',
            transparent=True, pad_inches=0,
            edgecolor='none')
