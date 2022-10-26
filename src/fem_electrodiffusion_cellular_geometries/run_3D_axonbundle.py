#!/usr/bin/python3

""" Solve KNP-EMI and EMI with Hodgkin Huxley membrane dynamics on idealized 3D
    axon bundle mesh. Six different simulation are run:
        (1a) KNP-EMI where axon A is stimulated
        (1b) KNP-EMI where axons B and C are stimulated
        (2a) EMI where axon A is stimulated
        (2b) EMI where axons B and C are stimulated
        (3a) EMI with lower conductivity (sigma) where axon A is stimulated
        (3b) EMI with lower conductivity (sigma) where axons B and C are stimulated
    All six experiments are run with the same physical parameters and initial
    conditions.
"""

from dolfin import *

import os
import sys
from time import sleep

import solver_knpemi as solver_knpemi
import solver_emi as solver_emi

def g_syn_A(g_syn_bar, a_syn, t):
    """ stimulate axon A """
    g_syn = Expression('g_syn_bar*exp(-fmod(t,0.02)/a_syn)*(x[0]<25e-6)*\
                        (x[1]>0.85e-6)*(x[1]<1.15e-6)*\
                        (x[2]>0.85e-6)*(x[2]<1.15e-6)',
                        g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)
    return g_syn

def g_syn_BC(g_syn_bar, a_syn, t):
    """ stimulate axons B and C """
    g_syn = Expression('g_syn_bar*exp(-fmod(t,0.02)/a_syn)*(x[0]<25e-6)*\
                        (1 - (x[1]>0.85e-6)*(x[1]<1.15e-6)*\
                        (x[2]>0.85e-6)*(x[2]<1.15e-6))',
                        g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)
    return g_syn

if __name__ == "__main__":
    resolution = '0'   # resolution factor mesh
    L = '400'        # length (x-dir) of domain

    # time variable
    dt = 1.0e-4      # global time step (s)
    Tstop = 5.0e-2   # global end time (s)
    n_steps_ode = 25 # number of steps for ODE solver

    # physical parameters
    C_M = 0.02                        # capacitance
    temperature = 300                 # temperature (K)
    F = 96485                         # Faraday's constant (C/mol)
    R = 8.314                         # Gas Constant (J/(K*mol))
    g_Na_bar = 1200                   # Na max conductivity (S/m**2)
    g_K_bar = 360                     # K max conductivity (S/m**2)
    g_Na_leak = Constant(2.0*0.5)     # Na leak conductivity (S/m**2)
    g_K_leak = Constant(8.0*0.5)      # K leak conductivity (S/m**2)
    g_Cl_leak = Constant(0.0)         # Cl leak conductivity (S/m**2)
    a_syn = 0.002                     # synaptic time constant (s)
    g_syn_bar = 40                    # synaptic conductivity (S/m**2)
    D_Na = Constant(1.33e-9)          # diffusion coefficients Na (m/s)
    D_K = Constant(1.96e-9)           # diffusion coefficients K (m/s)
    D_Cl = Constant(2.03e-9)          # diffusion coefficients Cl (m/s)

    # EMI specific parameters
    sigma_i = 2.011202                # intracellular conductivity
    sigma_e = 1.31365                 # extracellular conductivity
    V_rest = -0.065                   # resting membrane potential
    E_Na = 54.8e-3                    # Nernst potential sodium
    E_K = -88.98e-3                   # Nernst potential potassium
    g_Na_leak_emi = Constant(2.0*0.5) # Na leak conductivity (S/m**2)
    g_K_leak_emi = Constant(8.0*0.5)  # K leak conductivity (S/m**2)
    # lower EMI conductivities (from Bokil et al)
    sigma_i_low = 1.0                 # intracellular conductivity
    #sigma_e_low = 0.1                 # extracellular conductivity
    sigma_e_low = 0.05                # extracellular conductivity

    # initial values
    Na_i_init = Constant(12)                # Intracellular Na concentration
    Na_e_init = Constant(100)               # extracellular Na concentration
    K_i_init = Constant(125)                # intracellular K concentration
    K_e_init = Constant(4)                  # extracellular K concentration
    Cl_i_init = Constant(137)               # intracellular Cl concentration
    Cl_e_init = Constant(104)               # extracellular CL concentration
    phi_M_init = Constant(-0.0677379636231) # membrane potential (V)
    n_init = 0.27622914792                  # gating variable n
    m_init = 0.0379183462722                # gating variable m
    h_init = 0.688489218108                 # gating variable h

    # gather parameters
    params = {'dt':dt, 'Tstop':Tstop,
              'C_M':C_M, 'temperature':temperature, 'R':R, 'F':F,
              'phi_M_init':phi_M_init,
              'n_init':n_init, 'm_init':m_init,'h_init':h_init,
              'g_K_bar':g_K_bar, 'g_Na_bar':g_Na_bar,
              'sigma_i':sigma_i, 'sigma_e':sigma_e, 'V_rest':V_rest,
              'g_K_leak':g_K_leak_emi, 'g_Na_leak':g_Na_leak_emi,
              'E_Na':E_Na, 'E_K':E_K}

    # create ions with attributes
    Na = {'Di':D_Na, 'De':D_Na, 'ki_init':Na_i_init,
          'ke_init':Na_e_init, 'z':1.0, 'name':'Na'}
    K = {'Di':D_K, 'De':D_K, 'ki_init':K_i_init,
         'ke_init':K_e_init, 'z':1.0, 'name':'K'}
    Cl = {'Di':D_Cl, 'De':D_Cl, 'ki_init':Cl_i_init,
          'ke_init':Cl_e_init, 'z':-1.0, 'name':'Cl'}
    # create ion list
    ion_list = [Na, K, Cl]

    # get path to mesh, subdomains, surfaces
    here = os.path.abspath(os.path.dirname(__file__))
    mesh_prefix = os.path.join(here, 'meshes/' + L + '_axonbundle_3d/')
    mesh = mesh_prefix + 'mesh_' + resolution + '.xml'
    subdomains = mesh_prefix + 'subdomains_' + resolution + '.xml'
    surfaces = mesh_prefix + 'surfaces_' + resolution + '.xml'
    # generate mesh if it does not exist
    if not os.path.isfile(mesh):
        script = 'make_mesh_axonbundle_3D.py '                # script
        os.system('python3 ' + script + ' ' + resolution)      # run script

    ##########################################################################
    # Run (1a) with KNP-EMI - stimulate 1 axon (A)
    sys.stdout.write("\nRunning KNP-EMI stimulating axon A (1a)\n")

    t_1a = Constant(0.0)                                   # time constant
    fname_1a = 'results/data/axonbundle_stimuli_A_knpemi/res_' + resolution + '/' # file for results
    # set conductivity and synaptic current
    ion_list[0]['g_k'] = g_Na_leak + g_syn_A(g_syn_bar, a_syn, t_1a) # Na
    ion_list[1]['g_k'] = g_K_leak                                    # K
    ion_list[2]['g_k'] = g_Cl_leak                                   # Cl
    # KNP-EMI solver
    S_1a = solver_knpemi.Solver(ion_list, t_1a, **params) # create solver
    S_1a.setup_domain(mesh, subdomains, surfaces)         # setup domain
    S_1a.solve_system_HH(n_steps_ode, filename=fname_1a)  # solve

    ##########################################################################
    # Run (1b) with KNP-EMI - stimulate 8 axons (BC)
    sys.stdout.write("\nRunning KNP-EMI stimulating axons B&C (1b)\n")

    t_1b = Constant(0.0)                                    # time constant
    fname_1b = 'results/data/axonbundle_stimuli_BC_knpemi/res_' + resolution + '/' # file for results
    # set conductivity and synaptic current
    ion_list[0]['g_k'] = g_Na_leak + g_syn_BC(g_syn_bar, a_syn, t_1b) # Na
    ion_list[1]['g_k'] = g_K_leak                                     # K
    ion_list[2]['g_k'] = g_Cl_leak                                    # Cl
    # KNP-EMI solver
    S_1b = solver_knpemi.Solver(ion_list, t_1b, **params) # create solver
    S_1b.setup_domain(mesh, subdomains, surfaces)         # setup domain
    S_1b.solve_system_HH(n_steps_ode, filename=fname_1b)  # solve

    ##########################################################################
    # Run (2a) with EMI - stimulate 1 axon (A)
    sys.stdout.write("\nRunning EMI stimulating axon A (2a)\n")

    t_2a = Constant(0.0)                                # time constant
    fname_2a = 'results/data/axonbundle_stimuli_A_emi/res_' + resolution + '/' # file for results
    # set synaptic current
    params['g_ch_syn'] = g_syn_A(g_syn_bar, a_syn, t_2a)
    # EMI solver
    S_2a = solver_emi.Solver(t_2a, **params)             # create solver
    S_2a.setup_domain(mesh, subdomains, surfaces)        # setup domain
    S_2a.solve_system_HH(n_steps_ode, filename=fname_2a) # solve the system

    ##########################################################################
    # Run (2b) with EMI - stimulate 8 axon (BC)
    sys.stdout.write("\nRunning EMI stimulating axons B&C (2b)\n")

    t_2b = Constant(0.0)                                 # time constant
    fname_2b = 'results/data/axonbundle_stimuli_BC_emi/res_' + resolution + '/' # file for results
    # set synaptic current
    params['g_ch_syn'] = g_syn_BC(g_syn_bar, a_syn, t_2b)
    # EMI solver
    S_2b = solver_emi.Solver(t_2b, **params)             # create solver
    S_2b.setup_domain(mesh, subdomains, surfaces)        # setup domain
    S_2b.solve_system_HH(n_steps_ode, filename=fname_2b) # solve

    ##########################################################################
    # Run (3a) with EMI low conductivity - stimulate 1 axon (A)
    sys.stdout.write("\nRunning EMI stimulating axon A with low sigma (3a)\n")

    t_3a = Constant(0.0)                                          # time constant
    fname_3a = 'results/data/axonbundle_stimuli_A_emi_low_sigma/res_' + resolution + '/' # file for results
    # set synaptic current
    params['g_ch_syn'] = g_syn_A(g_syn_bar, a_syn, t_3a)
    # set lower conductivities from Bokil
    params['sigma_i'] = sigma_i_low # intracellular
    params['sigma_e'] = sigma_e_low # extracellular
    # EMI solver
    S_3a = solver_emi.Solver(t_3a, **params)             # create solver
    S_3a.setup_domain(mesh, subdomains, surfaces)        # setup domain
    S_3a.solve_system_HH(n_steps_ode, filename=fname_3a) # solve

    ##########################################################################
    # Run (3b) with EMI low conductivity - stimulate 8 axons (BC)
    sys.stdout.write("\nRunning EMI stimulating axons BC with low sigma (3b)\n")

    t_3b = Constant(0.0)                                           # time constant
    fname_3b = 'results/data/axonbundle_stimuli_BC_emi_low_sigma/res_' + resolution + '/' # file for results
    # set synaptic current
    params['g_ch_syn'] = g_syn_BC(g_syn_bar, a_syn, t_3b)
    # set lower conductivity from Bokil
    params['sigma_i'] = sigma_i_low # intracellular
    params['sigma_e'] = sigma_e_low # extracellular
    # EMI solver
    S_3b = solver_emi.Solver(t_3b, **params)             # create solver
    S_3b.setup_domain(mesh, subdomains, surfaces)        # setup domain
    S_3b.solve_system_HH(n_steps_ode, filename=fname_3b) # solve
