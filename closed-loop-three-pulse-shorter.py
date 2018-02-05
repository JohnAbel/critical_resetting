# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:03:42 2017

@author: abel

This file tries using a simple feedback controller to time the application of 
three shorter than five-hour pulses.
"""
from __future__ import division

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from LocalImports import LimitCycle as lc
from LocalImports import PlotOptions as plo

from LocalModels.kronauer_model import model, param, EqCount, y0in, null_I
import LocalModels.simplified_kronauer_model as skm


# set-up way to get the phase of a point using the simpler model
skron = lc.Oscillator(skm.model(), skm.param, skm.y0in)
skron.intoptions['constraints']=None
skron.calc_y0()
skron.limit_cycle()

# figure out phase at which the pulse occurs for the 24.6h case, since that 
# one is the most effective
y0 = [ 0.99996874,  0.10023398, 0]
param[3]=24.6
kron = lc.Oscillator(model(null_I), param, y0)
kron.intoptions['constraints']=None
dsol_pre = kron.int_odes(10.4)
dts_pre = kron.ts
y0_pulse = dsol_pre[-1]
pulse_phase = skron.phase_of_point(y0_pulse[:-1]) # should give ~2.66

# pulse function formula
def I_pulse(t):
    """ returns just max """
    return 9500

# targeted stimulus formula
def threecycle_targeted_stimulus(tau, phase_target, duration):
    """
    Applies the 3-cycle stimulus protocol for a given tau

    Arguments
    ----------
    tau : float
        period of oscillation
    phase_target : float
        phase at which we want to apply the pulse
    """

    # set up parameters for first time
    y0 = [ 0.99996874,  0.10023398, 0]
    param[3]=tau

    dsol_total = [[y0]]
    dts_total = [[0]]
    for pulsecount in range(3):

        # set up oscillator
        kron = lc.Oscillator(model(null_I), param, y0)
        kron.intoptions['constraints']=None
        phase = skron.phase_of_point(y0[:-1])

        # move forward when it's before the pulse time, or after the pulse
        while np.logical_or(phase < phase_target, 
                            phase > phase_target+1/tau*2*np.pi):
            # perform the integration
            dsol_pre = kron.int_odes(0.25, numsteps=100)
            dts_pre = kron.ts
            yend = dsol_pre[-1]
            phase = skron.phase_of_point(yend[:-1])
            kron.y0 = yend

            # append the integration
            dsol_total.append(dsol_pre)
            dts_total.append(dts_pre+dts_total[-1][-1])

        # when it's time, apply the pulse
        kron = lc.Oscillator(model(I_pulse), param, yend)
        kron.intoptions['constraints']=None
        dsol_dur = kron.int_odes(duration)
        dts_dur = kron.ts
        
        y0 = dsol_dur[-1]
        dts_total.append(dts_dur+dts_total[-1][-1])
        dsol_total.append(dsol_dur)

    # residual 19h period after pulse
    kron = lc.Oscillator(model(null_I), param, dsol_dur[-1])
    kron.intoptions['constraints']=None
    dsol_post = kron.int_odes(24-duration)
    dts_post = kron.ts
    
    dts_total.append(dts_post+dts_total[-1][-1])
    dsol_total.append(dsol_post)

    final_dist = np.sqrt(np.sum(dsol_dur[-1][:2]**2))
    dsol_total = np.vstack(dsol_total)
    dts_total = np.hstack(dts_total)

    return dts_total, dsol_total, final_dist


dts5, dsol5, final_dist5 = threecycle_targeted_stimulus(24.2, pulse_phase, 5)
dts3, dsol3, final_dist3 = threecycle_targeted_stimulus(24.2, 2.65, 5)


# comparison lc
y0 = [ 0.99996874,  0.10023398, 0]
kron = lc.Oscillator(model(null_I), param, y0)
kron.intoptions['constraints']=None
dsol_lc = kron.int_odes(25)
dts_lc = kron.ts


plt.figure()
# dynamics plot
plt.plot(dts5, dsol5, 'b', label ='5h stim')
plt.plot(dts3, dsol3, 'r--', label ='3h stim')
plt.xlabel('Time (h)')
plt.ylabel('$States$')
plt.legend()
plt.title(r'$\tau$='+str(param[3]))
plt.tight_layout(**plo.layout_pad)


plt.figure()
# limit cycle plot
plt.plot(dsol_lc[:,0], dsol_lc[:,1], 'k', label ='LC')
plt.plot(dsol3[:,0], dsol3[:,1], 'r--', label ='3h stim')
plt.plot(dsol5[:,0], dsol5[:,1], 'b:', label ='5h stim')
plt.xlabel('$x$')
plt.ylabel('$x_c$')
plt.ylim([-1.4,1.1])
plt.xlim([-1.5,1.6])
plt.legend()
plt.title(r'$\tau$='+str(param[3]))
plt.tight_layout(**plo.layout_pad)





# plotting how the distance from the singularity scales with tau (~15min)
final_distance_5h = []
final_distance_45h = []
final_distance_4h = []
final_distance_35h = []
final_distance_3h = []
phases = np.arange(2,3,0.05)
for  phase in phases:
    final_distance_5h.append(threecycle_targeted_stimulus(24.2, phase, 5)[2])
    final_distance_45h.append(threecycle_targeted_stimulus(24.2, phase, 4.5)[2])
    final_distance_4h.append(threecycle_targeted_stimulus(24.2, phase, 4)[2])
    final_distance_35h.append(threecycle_targeted_stimulus(24.2, phase, 3.5)[2])
    final_distance_3h.append(threecycle_targeted_stimulus(24.2, phase, 3)[2])




plt.figure()
# distances plot
plt.plot(phases, final_distance_5h, 'b', label='5h stim')
plt.plot(phases, final_distance_4h, 'g', label='4h stim')
plt.plot(phases, final_distance_3h, 'r', label='3h stim')
plt.legend()
plt.xlabel(r'$\phi_{stim}$')
plt.ylabel('Distance to Critical Pt')
plt.tight_layout(**plo.layout_pad)




