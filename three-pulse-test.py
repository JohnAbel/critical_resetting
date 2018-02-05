# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:03:42 2017

@author: abel

This file tries using a three pulses to see how slight changes in period affect 
the nearness of the oscillator to the singularity.
"""
from __future__ import division

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from LocalImports import LimitCycle as lc
from LocalImports import PlotOptions as plo

from LocalModels.kronauer_model import model, param, EqCount, y0in, null_I

# pulse
def I_pulse(t):
    """ returns just max """
    return 9500

def threecycle_stimulus(tau):
    """
    Applies the 3-cycle stimulus protocol for a given tau

    Arguments
    ----------
    tau : float
        period of oscillation
    """

    # pre-pulse
    y0 = [ 0.99996874,  0.10023398, 0]
    param[3]=tau
    kron = lc.Oscillator(model(null_I), param, y0)
    kron.intoptions['constraints']=None
    dsol_pre = kron.int_odes(10.4)
    dts_pre = kron.ts
    y0_pulse = dsol_pre[-1]

    dsol_pulses = []
    dts_pulses = []
    for i in range(3):
        # during pulse
        kron = lc.Oscillator(model(I_pulse), param, y0_pulse)
        kron.intoptions['constraints']=None
        dsol_dur = kron.int_odes(5)
        dts_dur = kron.ts
        
        # after pulse
        kron = lc.Oscillator(model(null_I), param, dsol_dur[-1])
        kron.intoptions['constraints']=None
        dsol_post = kron.int_odes(19)
        dts_post = kron.ts
        
        y0_pulse = dsol_post[-1]
        dts_pulses.append(dts_dur)
        dts_pulses.append(dts_post)
        dsol_pulses.append(dsol_dur)
        dsol_pulses.append(dsol_post)

    final_dist = np.sqrt(np.sum(dsol_dur[-1][:2]**2))
    dsol_pulses = np.vstack(dsol_pulses)
    dts_pulses = np.hstack(dts_pulses)

    return dts_pulses, dsol_pulses, final_dist

dts_pulses, dsol_pulses, final_dist = threecycle_stimulus(24.2)

# comparison lc
y0 = [ 0.99996874,  0.10023398, 0]
kron = lc.Oscillator(model(null_I), param, y0)
kron.intoptions['constraints']=None
dsol_lc = kron.int_odes(25)
dts_lc = kron.ts




plt.figure()

# limit cycle plot
plt.plot(dsol_lc[:,0], dsol_lc[:,1], 'k', label ='LC')
plt.plot(dsol_pre[:,0], dsol_pre[:,1], 'r--', label ='Pre-Stim')
plt.plot(dsol_pulses[:,0], dsol_pulses[:,1], 'b--', label ='During')
plt.plot(dsol_post[:,0], dsol_post[:,1], 'g', label ='Post')

plt.xlabel('$x$')
plt.ylabel('$x_c$')
plt.ylim([-1.4,1.1])
plt.xlim([-1.5,1.6])
plt.legend()
plt.title(r'$\tau$='+str(param[3]))
plt.tight_layout(**plo.layout_pad)


# plotting how the distance from the singularity scales with tau
final_distances = []
taus = np.arange(23,26,0.1)
for tau in taus:
    final_distances.append(threecycle_stimulus(tau)[2])

plt.figure()

# limit cycle plot
plt.plot(taus, final_distances, 'k')

plt.xlabel(r'$\tau$')
plt.ylabel('Distance to Critical Pt')
plt.tight_layout(**plo.layout_pad)




