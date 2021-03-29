import numpy as np
from numba import njit, prange

def setup(model):
    """ setup simulation """

    par = model.par
    sim = model.sim

    assert par.simT%12 == 0, 'simulation must be in whole years'

    np.random.seed(model.par.seed_sim)
    
    # a. allocate
    size = (par.simN,par.simT)
    sim.y = np.zeros(size)
    sim.p = np.zeros(size)
    sim.z = np.zeros(size)

    # b. uniform shocks
    sim.p_psi = np.reshape(np.repeat(np.linspace(0,1,par.simN),par.simT),(par.simN,par.simT))
    sim.p_xi = sim.p_psi.copy()
    sim.p_phi = sim.p_psi.copy()
    sim.p_eta = sim.p_psi.copy()
    for t in range(par.simT):
        np.random.shuffle(sim.p_psi[:,t])
        np.random.shuffle(sim.p_xi[:,t])
        np.random.shuffle(sim.p_phi[:,t])
        np.random.shuffle(sim.p_eta[:,t])

    # c. normal shocks
    sim.psi_raw = np.random.normal(size=size)
    sim.xi_raw = np.random.normal(size=size)
    sim.phi_raw = np.random.normal(size=size)
    sim.eta_raw = np.random.normal(size=size)
    sim.epsilon_raw = np.random.normal(size=size)
    sim.p0_raw = np.random.normal(size=par.simN)

    # d. growth rates
    size = (par.kmax,par.simN,par.simT)
    
    if par.do_d1ky: 

        sim.d1ky = np.nan*np.ones(size)
        sim.d1kyw = np.nan*np.ones(size)
        sim.d1ky_lag = np.nan*np.ones(size)
    
    if par.do_d12ky: 

        sim.d12ky = np.nan*np.ones(size)
        sim.d12kyw = np.nan*np.ones(size)
        sim.d12ky_lag = np.nan*np.ones(size)

@njit(parallel=True)
def main(sim,par):
    """ main simulation algorithm """

    # a. initial std
    if par.rho < 1-1e-4:
        sigma_p0 = np.sqrt(par.sigma_psi**2/(1-par.rho**2))
    else:
        sigma_p0 = 0

    # b. unpack
    y = sim.y
    p = sim.p
    z = sim.z

    # c. time loop
    for i in prange(par.simN):
        for t in range(par.simT):
            
            # i. lagged
            if t == 0:
                p_lag = sigma_p0*sim.p0_raw[i]
                z_lag = 0             
            else:
                p_lag = p[i,t-1]
                z_lag = z[i,t-1]

            # ii. permanent and persistent
            if sim.p_phi[i,t] < par.p_phi:
                z[i,t] = z_lag + par.sigma_phi*sim.phi_raw[i,t] + par.mu_phi
            else:
                z[i,t] = z_lag

            if sim.p_psi[i,t] < par.p_psi:
                p[i,t] = par.rho*p_lag + par.sigma_psi*sim.psi_raw[i,t]
            else:
                p[i,t] = p_lag

            y[i,t] = p[i,t] + z[i,t]

            # iii. infrequent transitory
            if sim.p_xi[i,t] < par.p_xi:
                y[i,t] += par.sigma_xi*sim.xi_raw[i,t] + par.mu_xi
            
            if sim.p_eta[i,t] < par.p_eta:
                y[i,t] += par.sigma_eta*sim.eta_raw[i,t] + par.mu_eta
            
            # v. ever-present transitory
            if par.sigma_epsilon > 0:
                y[i,t] += par.sigma_epsilon*sim.epsilon_raw[i,t]

def all(model):
    """ simulation and calculation of growth rates """
    
    par = model.par
    sim = model.sim
    
    # a. main
    main(sim,par)

    # b. calculate growth rates
    if par.do_d1ky:

        for k in range(1,par.kmax+1):

            sim.d1ky[k-1,:,k:] = sim.y[:,k:] - sim.y[:,:-k]
            if k > 0: sim.d1ky_lag[k-1,:,k+1:] = sim.d1ky[k-1,:,k:-1]
            sim.d1kyw[k-1] = sim.d1ky[k-1]

    if par.do_d12ky:

        for k in range(1,par.kmax+1):
        
            sim.d12ky[k-1,:,12*k:] = sim.y[:,12*k:] - sim.y[:,:-12*k]
            if k > 0: sim.d12ky_lag[k-1,:,12*(k+k):] = sim.d12ky[k-1,:,12*k:-12*k]
            sim.d12kyw[k-1] = sim.d12ky[k-1]