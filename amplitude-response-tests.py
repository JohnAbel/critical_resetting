# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:03:42 2017

@author: abel

This file plots the ARC of the Kronauer model toward driving it to 0.
"""
from __future__ import division

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from LocalImports import LimitCycle as lc
from LocalImports import PlotOptions as plo

from LocalModels.simplified_kronauer_model import model, param, EqCount, y0in

kron = lc.Oscillator(model(), param, y0in)
kron.intoptions['constraints']=None

# get the PRC and ARC for this model
kron.calc_y0()
kron.limit_cycle()
kron.find_prc()
kron.findARC_whole()

# plot PRC, ARC

plt.figure()
plt.plot(kron.arc_ts, kron.pPRC_interp(kron.arc_ts)[:,4])
plt.vlines(12.9, -0.4,0.3,'k')
plt.xlabel('time (h), CBTmin = 12.9')
plt.ylabel('pPRC $\hat B$')
plt.tight_layout(**plo.layout_pad)


plt.figure()
plt.plot(kron.arc_ts, kron.pARC[:,:,4])
plt.vlines(12.9, -4, 5, 'k')
plt.xlabel('time (h), CBTmin = 12.9')
plt.ylabel('pARC $\hat B$')
plt.tight_layout(**plo.layout_pad)
