# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:03:42 2017

@author: abel

This file tries using a simple feedback controller to time the application of
the three five-hour pulses.
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
    """ returns just max lux """
    return 9500

# targeted stimulus formula
def threecycle_targeted_stimulus(tau, phase_target):
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
                            phase > phase_target+3/tau*2*np.pi):
            # perform the integration
            dsol_pre = kron.int_odes(0.5, numsteps=100)
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
        dsol_dur = kron.int_odes(5)
        dts_dur = kron.ts

        y0 = dsol_dur[-1]
        dts_total.append(dts_dur+dts_total[-1][-1])
        dsol_total.append(dsol_dur)

    # residual 15h period after pulse - no need to try to target here
    kron = lc.Oscillator(model(null_I), param, dsol_dur[-1])
    kron.intoptions['constraints']=None
    dsol_post = kron.int_odes(15)
    dts_post = kron.ts

    dts_total.append(dts_post+dts_total[-1][-1])
    dsol_total.append(dsol_post)

    final_dist = np.sqrt(np.sum(dsol_dur[-1][:2]**2))
    dsol_total = np.vstack(dsol_total)
    dts_total = np.hstack(dts_total)

    return dts_total, dsol_total, final_dist

# untargeted stimulus formula
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

    dsol_pulses = [dsol_pre]
    dts_pulses = [dts_pre]
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
        dts_pulses.append(dts_dur+dts_pulses[-1][-1])
        dts_pulses.append(dts_post+dts_pulses[-1][-1])
        dsol_pulses.append(dsol_dur)
        dsol_pulses.append(dsol_post)

    final_dist = np.sqrt(np.sum(dsol_dur[-1][:2]**2))
    dsol_pulses = np.vstack(dsol_pulses)
    dts_pulses = np.hstack(dts_pulses)

    return dts_pulses, dsol_pulses, final_dist

dts, dsol, final_dist = threecycle_targeted_stimulus(23.8, pulse_phase)

dtsu, dsolu, final_distu = threecycle_stimulus(23.8)

# comparison lc
y0 = [ 0.99996874,  0.10023398, 0]
kron = lc.Oscillator(model(null_I), param, y0)
kron.intoptions['constraints']=None
dsol_lc = kron.int_odes(25)
dts_lc = kron.ts


plt.figure()
# dynamics plot
plt.plot(dtsu, dsolu, 'b', label ='Blind')
plt.plot(dts, dsol, 'r--', label ='Controlled')
plt.xlabel('Time (h)')
plt.ylabel('$States$')
plt.legend()
plt.title(r'$\tau$='+str(param[3]))
plt.tight_layout(**plo.layout_pad)


plt.figure()
# limit cycle plot
plt.plot(dsol_lc[:,0], dsol_lc[:,1], 'k', label ='LC')
plt.plot(dsol[:,0], dsol[:,1], 'r--', label ='Controlled')
plt.plot(dsolu[:,0], dsolu[:,1], 'b:', label ='Blind')
plt.xlabel('$x$')
plt.ylabel('$x_c$')
plt.ylim([-1.4,1.1])
plt.xlim([-1.5,1.6])
plt.legend()
plt.title(r'$\tau$='+str(param[3]))
plt.tight_layout(**plo.layout_pad)





# plotting how the distance from the singularity scales with tau (~15min)
final_distances_blind = []
final_distances_control = []
taus = np.arange(23,26,0.2) # shoose the periods
for tau in taus:
    final_distances_blind.append(threecycle_stimulus(tau)[2])
    final_distances_control.append(
                threecycle_targeted_stimulus(tau, pulse_phase)[2])




plt.figure()
# distances plot
plt.plot(taus, final_distances_blind, 'b', label='Blind')
plt.plot(taus, final_distances_control, 'r', label='Controlled')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel('Distance to Critical Pt')
plt.tight_layout(**plo.layout_pad)




