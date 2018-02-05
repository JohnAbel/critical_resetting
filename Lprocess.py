# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:57:09 2016

@author: abel
"""

from __future__ import division

import scipy as sp
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import pdb

from scipy import integrate
from scipy.interpolate import UnivariateSpline

import circadiantoolbox as ctb

class Lprocess(object):
    """
    Class to represent L-process for the Kronauer model. 
    
    """
    def __init__(self, P_process, L_process_empty,
                 param_L=None, LP_process_func=None):
        """
        Initializes the object
        ----
        Pprocess : Oscillator object, the internal oscillator
        L_process_empty(param_L) : function, which returns Lprocess(I(t)), the 
                    parameterized L-process ready for taking I(t)
        param_L : optional iterable fed to the empty L_process
        LP_process_func : optional function used to compare the exact and 
                    approximated responses 
        ----
        """
        # get the oscillator running
        P_process.calc_y0()
        P_process.limit_cycle()
        P_process.find_prc()
        P_process.roots()
        self.P_process = P_process
        
        # get the Lprocess, parameterize it
        self.L_process_empty = L_process_empty
        if param_L is not None:
            self.param_L = param_L
            def L_process(I):
                return self.L_process_empty(I, param_L)
            self.L_process = L_process
    
    def parameterize(self, param_L):
        """
        Re-parameterizes Lprocess with the provided parameters.
        """
        self.param_L = param_L
        def L_process(I):
            return self.L_process_empty(I, param_L)
        self.L_process = L_process
        
    def attach_full_model(self, LP_process_func, param_LP, y0_LP):
        """
        Attach a full model
        ----
        LP_process_func : Oscillator object which has the Lprocess baked in, so
                        that results can be compared between the exact
                        and approximate responses. Note that this function
                        is not parameterized
        ----
        """
        self.LP_process_func = LP_process_func
        self.param_LP = param_LP
        self.y0_LP = y0_LP
    
    def parameterize_full_model(self, I, param_LP=None, y0_LP=None,
                                        LP_process_func=None):
        """Re-parameterizes the full model.
        ----
        I : casadi-evaluable light input
        param_LP : parameters for the full L and P model
        
        """
        if LP_process_func is not None:
            self.LP_process_func = LP_process_func
        elif self.LP_process_func==None:
            raise Exception("No model provided to parameterize.")
        # creates the Oscillator
        if param_LP is not None:
            self.param_LP = param_LP
        if y0_LP is not None:
            self.y0_LP = y0_LP
        
        self.LP_process = ctb.Oscillator(self.LP_process_func(I), self.param_LP,
                                            self.y0_LP)
    
    def exact_response(self, I, parameterization_args=None, tf=None,
                       full_output=False, discontinuities=[]):
        """
        Uses the full L and P processes to find the exact response of the model
        to a perturbation through I. This returns the final phase in comparison
        to the final phase with no I input.
        ----
        parameterization_args : list [I, param_LP, y0_LP, LP_process_func]
        discontinuities : np.ndarray of points where integration must stop and
            restart to ensure Lipschitz continuity
        full_output : bool, defaults to false. if true, will return trajectory
            results in addition to the relative phase
        ----
        """
        if tf==None:
            #get a timespan well beyond where perturbation dies out
            tf = 3*self.P_process.T
        if parameterization_args is not None:
            # optional parameterization of the model here
            param_LP, y0_LP, LP_process_func = parameterization_args
            self.parameterize_full_model(I, param_LP, y0_LP, LP_process_func)
        else:
            self.parameterize_full_model(I)
        
        tfs = np.asarray(list(discontinuities)+list([tf])) 
        #stop and restart solver where not lipscitz continuous
        solution = []
        times = []
        y0=self.LP_process.y0; t0=0 # set up initial conditions
        for tf in tfs:
            test_sol = self.LP_process.int_odes(tf, y0=y0, ts=t0)
            test_ts = self.LP_process.ts
            times+=list(test_ts[:-1])
            solution+=list(test_sol[:-1])
            y0=test_sol[-1]
            t0=test_ts[-1]
        final_phase = self.P_process.phase_of_point(
                                solution[-1][:self.P_process.neq])
        ref_phase = self.P_process.phase_of_point(
                        self.P_process.int_odes(tf, 
                                y0=self.LP_process.y0[:self.P_process.neq],
                                return_endpt=True))
        exact_response = final_phase - ref_phase
        
        # wrap to phase regions
        exact_response = (exact_response + np.pi) % (2 * np.pi ) - np.pi
        if full_output==True:
            return exact_response, times, np.asarray(solution)
        else: return exact_response
    
    def approximate_response_pPRC(self, I, full_output=False, xs_test=None,
                                  light_param=-1, discontinuities=[],
                                    param_L=None, res=500, n0=0, phi_start=0):
        """
        Calculates the phase response using the pPRC, and treating light input
        as a change in parameter value.
        ----
        I(t) : numpy evaluable function for light input
        light_param : int index of the parameter used to input light
        xs_test : np.ndarray of points for which we are solving the input 
            equation
        discontinuities : np.ndarray of points where integration must stop and
            restart to ensure Lipscitz continuity
        full_output : bool, defaults to false. if true, will return trajectory
            results
        res : int number of points for each region discretization
        n0  : float [0,1], initial fraction of used
        phi_start  : float [0,T], the phase (h) at which the simulation starts. 
              defaults to 0, the min of X
              
        Returns : approx_pPRC (units:)
        
        Notes:
        Longer perturbations (over multiple cycles) will shift the clock each 
        cycle forcing updating of the phase at each calculation of dphi to 
        correct the phase at which the perturbation hits
        ----
        """
        if param_L is not None:
            self.parameterize(param_L)
        if xs_test is None:
            # create a spline fit of Bhat only where it is nonzero
            xs_test = np.linspace(0,self.P_process.T*5, 500)
        xmin = xs_test[np.max([0, np.min(np.where(I(xs_test)>1E-3))-1])]
        xmax = xs_test[np.max(np.where(I(xs_test)>1E-3))+1]
        xmaxs = list(discontinuities)+list([xmax])
        
        Bhat_func = self.L_process(I)
        
        xmini = xmin; n0=n0; init=0 # initial conditions
        xs=[]
        bhat=[]
        approx_pPRC=[]
        pPRC_interp = self.P_process.pPRC_interp.splines[light_param]
        for xmaxi in xmaxs:
            xsi = np.linspace(xmini,xmaxi,res)
            xs +=list(xsi)
            bhati, nn = Bhat_func(xsi,[xmini, n0])
            bhat +=list(bhati)
            
            bhatspl = UnivariateSpline(xsi,bhati,s=0,k=3)
            
            def pPRC_times_bhat(phit_array,t): #does this work for realtime updating the shift?
                delphi=phit_array
                result = pPRC_interp(t+delphi+phi_start)*bhatspl(t)
                return result
                
                
            delta_phi = integrate.odeint(pPRC_times_bhat, init, xsi)
            
            approx_pPRC+=list(delta_phi)
            xmini=xmaxi; n0=nn[-1]; init=delta_phi[-1]
            
           
        #pprc_xs_B = self.P_process.pPRC_interp(xs)[:,light_param]
        #prB = bhat*pprc_xs_B
        #approx_pPRC = integrate.cumtrapz(prB,xs)[-1]
        
        if full_output:
            return (approx_pPRC, xs, 
                        UnivariateSpline(xs, approx_pPRC, s=0, k=3, ext=3))
        else:
            return approx_pPRC[-1]
        
    def simulate_full_model(self, I, parameterization_args=None, tf=None,
                       discontinuities=[]):
        if tf==None:
            #get a timespan well beyond where perturbation dies out
            tf = 10*self.P_process.T
        
        if parameterization_args is not None:
            # optional parameterization of the model here
            param_LP, y0_LP, LP_process_func = parameterization_args
            self.parameterize_full_model(I, param_LP, y0_LP, LP_process_func)
        else:
            self.parameterize_full_model(I)
        
        tfs = np.asarray(list(discontinuities)+list([tf])) 
        #stop and restart solver where not lipscitz continuous
        solution = []
        times = []
        y0=self.LP_process.y0; t0=0 # set up initial conditions
        for tf in tfs:
            test_sol = self.LP_process.int_odes(tf, y0=y0, ts=t0)
            test_ts = self.LP_process.ts
            times+=list(test_ts[:-1])
            solution+=list(test_sol[:-1])
            y0=test_sol[-1]
            t0=test_ts[-1]
        return np.asarray(times), np.asarray(solution)
                       
    def compare_phase_responses(self, Ipulse, Iref, method='exact', 
                                full_output=False):
        """
        Comparison of exact response with the pPRC response.
        ----
        method : 'exact'- uses the exact simulation for each
                 'pPRC' - not yet implemented
        """
        
        if method=='exact':
            self.exact_phase_response(Ipulse, Iref, full_output)
            


def _I_null_np(t):
    return 0

def npheaviside(t):
    """ A numpy formulation of the heaviside step function"""
    return 0.5*(np.sign(t)+1)












if __name__=="__main__":
    import PlotOptions as plo
    import matplotlib.pyplot as plt
    import jha_utilities as jha
    
    # reproduce the comparison from compare_kronauer_responses.py
    from LocalModels import kronauer_model as km
    from LocalModels import simplified_kronauer_model as skm
    
    #set up Pprocess oscillator
    Pprocess = ctb.Oscillator(skm.kronauer(), skm.param, skm.y0in)
    Pprocess.intoptions['constraints']=None
    
    #set up the Lprocess object
    lproc = Lprocess(Pprocess, skm.process_L, skm.param_L)
    
    # let's see how it responds to a perturbation of 1.0h at different times
    end_ts=np.linspace(1.1,24.,100)
    start_ts=np.linspace(0.1,23.,100)
    
    errs = []
    exact_shifts = []
    approximated_shifts = []
    
    timer = jha.laptimer()
    for i,startt in enumerate(start_ts):
        endt = end_ts[i]
        def test_Inp(t):
            """ Test numpy light input function. 9500lux from t=5 to 5.5 """
            return 9500*npheaviside(t-startt) - 9500*npheaviside(t-endt)
        approx_response_pPRC = lproc.approximate_response_pPRC(test_Inp,
                                discontinuities=[startt,endt])
        
        
        # we can compare this with a full model
        def test_Ics(t):
            """ Test casadi light input function. 9500lux from t=5 to 5.5 """
            return 9500*cs.heaviside(t-startt) - 9500*cs.heaviside(t-endt)
            
        lproc.parameterize_full_model(test_Ics, km.param, km.y0in,
                        LP_process_func=km.kronauer)
        pshift = lproc.exact_response(discontinuities=[startt,endt])
        
        err = round((pshift-approx_response_pPRC)/pshift, 3)
        #print "Error = "+str(err*100)+"%."
        errs.append(err)
        approximated_shifts.append(approx_response_pPRC)
        exact_shifts.append(pshift)
        
    print timer()

    exact_shifts = np.asarray(exact_shifts)
    approximated_shifts = np.asarray(approximated_shifts)
    
    plo.PlotOptions()
    ax=plt.subplot()
    ax.plot(start_ts, exact_shifts, label='Exact Shifts')
    ax.plot(start_ts, approximated_shifts, label='Approximated Shifts')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Phase Response ($\phi$)')
    ax.legend()
    plo.hide_spines(ax)
    plt.tight_layout(**plo.layout_pad)
