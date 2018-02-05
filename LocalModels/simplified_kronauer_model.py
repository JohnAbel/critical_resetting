# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:34:29 2016

@author: abel

Model from Kronauer et al. 1999 in JBR, "Quantifying human circadian response
to light."

This model is simplified. Instead of simulating light input, we have a
parameter B, for which we can get a parameter PRC
"""

import casadi as cs
import numpy as np

# Constants and Equation Setup
EqCount = 2
ParamCount = 5

param_P = [0.13, 0.55, 1./3, 24.2]
param_L = [0.16, 0.013, 19.875, 0.6, 9500] #alpha0, beta, G, p, I0

param = param_P+[0.0]
y0in = [ 0.99996874,  0.10023398]
period = param_P[-1]

#ref_min
x_min = 12.100
phi_ref = 0.8
CBT_min = x_min + phi_ref

def model():
    """
    Function for the L-P process model of human circadian rhythms. Calculation
    of phase shifts must be done via the
    """
    #==================================================================
    #setup of symbolics
    #==================================================================
    x  = cs.SX.sym("x")
    xc = cs.SX.sym("xc")

    sys = cs.vertcat([x,xc])

    #===================================================================
    #Parameter definitions
    #===================================================================

    mu  = cs.SX.sym("mu")
    k   = cs.SX.sym("k")
    q   = cs.SX.sym("q")
    taux= cs.SX.sym("taux")

    alpha0 = cs.SX.sym("alpha0")
    beta   = cs.SX.sym("beta")
    G      = cs.SX.sym("G")
    p = cs.SX.sym("p")
    I0 = cs.SX.sym("I0")

    Bhat = cs.SX.sym("B")
    paramset = cs.vertcat([mu, k, q, taux, Bhat])

    # Time
    t = cs.SX.sym("t")


    #===================================================================
    # set up the ode system
    #===================================================================
    def B(bhat):
        return (1-0.4*x)*(1-0.4*xc)*bhat

    ode = [[]]*EqCount #initializes vector
    ode[0] = (cs.pi/12)*(xc +mu*(x/3. + (4/3.)*x**3 - (256/105.)*x**7)+B(Bhat))
    ode[1] = (cs.pi/12)*(q*B(Bhat)*xc - ((24/(0.99729*taux))**2 + k*B(Bhat))*x)
    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=sys, p=paramset),
                        cs.daeOut(ode=ode))

    fn.setOption("name","kronauer_P")

    return fn


from scipy.interpolate import UnivariateSpline
from scipy.integrate import odeint
# output drive onto the pacemaker for the prompt response model
def process_L(I, param_L):
    """
    The L process of the model. This returns a (spline?) for B:
    I : function defining the light input
    t : np.ndarray of time points
    param_L : parameters for the L process
    """

    alpha0, beta, G, p, I0 = param_L
    def alpha(t):
        return alpha0*(I(t)/I0)**p

    def ndot(nt, t0):
        t = nt[0]
        n = nt[1]
        return [1, 60*(alpha(t)*(1-n)-beta*n)]

    # switch to casadi object that will be instantaneous-ish
    def bhat_func(ts, y0):
        nn = odeint(ndot, y0, ts)[:,1]
        # hmax helps prevent the solver from ignoring step functions, which it
        # otherwise tends to do
        bhat = G*np.asarray([alpha(t) for t in ts])*(1-nn)
        return bhat,nn

    return bhat_func
    '''
    Bhat = np.asarray([G*alpha(ti)*(1-nn[i]) for i,ti in enumerate(t)])

    # circadian modulation of photic sensitivity
    Bhat_spline = UnivariateSpline(t, Bhat, k=3, ext=0)
    return Bhat_spline
    '''



if __name__ == "__main__":

    import circadiantoolbox as ctb

    # test the full model with light input--does it integrate?
    kron = ctb.Oscillator(kronauer(), param, y0in)
    kron.intoptions['constraints']=None

    # integrating with the test light input
    dsol = kron.int_odes(100,numsteps=1000)
    dts = kron.ts
    plt.plot(dts,dsol)

    kron.calc_y0()
    kron.limit_cycle()


    kron.first_order_sensitivity()
    kron.find_prc()


