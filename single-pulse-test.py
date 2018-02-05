# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:03:42 2017

@author: abel

This file tries using a single pulse to see how close to the singularity we can
get, full pulse form 10-15h.
"""
from __future__ import division

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from LocalImports import LimitCycle as lc
from LocalImports import PlotOptions as plo

from LocalModels.kronauer_model import model, param, EqCount, y0in, null_I

# pre-pulse
y0 = [ 0.99996874,  0.10023398, 0]
kron = lc.Oscillator(model(null_I), param, y0)
kron.intoptions['constraints']=None
dsol_pre = kron.int_odes(3)
dts_pre = kron.ts


# during pulse
def I_pulse(t):
    """ returns just max """
    return 9500
kron = lc.Oscillator(model(I_pulse), param, dsol_pre[-1])
kron.intoptions['constraints']=None
dsol_dur = kron.int_odes(5)
dts_dur = kron.ts


# after pulse
kron = lc.Oscillator(model(null_I), param, dsol_dur[-1])
kron.intoptions['constraints']=None
dsol_post = kron.int_odes(19.2)
dts_post = kron.ts


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
plt.plot(dsol_dur[:,0], dsol_dur[:,1], 'b--', label ='During')
plt.plot(dsol_post[:,0], dsol_post[:,1], 'g--', label ='Post')

plt.xlabel('$x$')
plt.ylabel('$x_c$')
plt.ylim([-1.5,1.1])
plt.xlim([-1.5,1.6])
plt.legend()
plt.tight_layout(**plo.layout_pad)