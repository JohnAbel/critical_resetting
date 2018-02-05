# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:34:29 2016

@author: abel

Model from Kronauer et al. 1999 in JBR, "Quantifying human circadian response
to light."

We will define this model to take cs-usable light inputs. This probest he
underlying oscillator, not so much process L.
"""

import casadi as cs
import numpy as np


# Constants and Equation Setup
EqCount = 3
ParamCount = 9

param_P = [0.13, 0.55, 1./3, 24.2]
param_L = [0.16, 0.013, 19.875, 0.6, 9500] #alpha0, beta, G, p, I0

param = param_P+param_L
y0in = [ -0.17,  -1.22, 0.5]
period = param_P[-1]

def model(I):
    """
    Function for the L-P process model of human circadian rhythms. Takes
    function I(t), light intensity, as its input. I(t) must be able to be
    evaluated by casadi.

    Some LC information cannot be computed here because n does not use a limit
    cycle. Instead, use kronauer_LC for limit cycle calcs.
    """
    #==================================================================
    #setup of symbolics
    #==================================================================
    x  = cs.SX.sym("x")
    xc = cs.SX.sym("xc")
    n  = cs.SX.sym("n")

    sys = cs.vertcat([x,xc,n])

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

    paramset = cs.vertcat([mu, k, q, taux, alpha0, beta, G, p, I0])

    # Time
    t = cs.SX.sym("t")


    #===================================================================
    # set up the ode system
    #===================================================================

    # light input
    def alpha(t):
        return alpha0*(I(t)/I0)**p

        # output drive onto the pacemaker for the prompt response model
    def Bhat(t):
        return G*alpha(t)*(1-n)

    # circadian modulation of photic sensitivity
    def B(t):
        return (1-0.4*x)*(1-0.4*xc)*Bhat(t)


    ode = [[]]*EqCount #initializes vector
    ode[0] = (cs.pi/12)*(xc +mu*(x/3. + (4/3.)*x**3 - (256/105.)*x**7)+B(t))
    ode[1] = (cs.pi/12)*(q*B(t)*xc - ((24/(0.99729*taux))**2 + k*B(t))*x)
    ode[2] = 60*(alpha(t)*(1-n)-beta*n)
    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=sys, p=paramset),
                        cs.daeOut(ode=ode))

    fn.setOption("name","kronauer_LP")

    return fn


def null_I(t):
    """
    Test light input function. No light.
    """
    return 0





if __name__ == "__main__":

    import circadiantoolbox as ctb
    import light_functions as lf

    # test the full model with light input--does it integrate?
    krona = ctb.Oscillator(kronauer(null_I, param, y0in))
    krona.intoptions['constraints']=None

    # integrating with the test light input
    dsol = krona.int_odes(48,numsteps=4801)
    dts = krona.ts
    y0_dawn = dsol[800]


# parts that do not work
#kron.limit_cycle() # because it's discontinuous
#kron.first_order_sensitivity()
#kron.find_prc()

