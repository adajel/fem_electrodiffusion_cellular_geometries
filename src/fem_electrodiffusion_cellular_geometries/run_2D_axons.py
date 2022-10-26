#!/usr/bin/python3

"""
Solve KNP-EMI and EMI on with passive membrane mechanisms and stimuli input
current. Both KNP and EMI are run on 3 different two-dimensional meshes:
    (1a) one cell (model C1) KNP-EMI
    (1b) one cell (model C1) EMI
    (2a) two cells (model C2 / 3 um distance in y-direction between cells) KNP-EMI
    (2b) two cells (model C2 / 3 um distance in y-direction between cells) EMI
    (3a) two cells (model C3 / 10 um distance in y-direction between cells) KNP-EMI
    (3b) two cells (model C3 / 10 um distance in y-direction between cells) EMI
"""

from dolfin import *

import os
import sys
from time import sleep

import solver_knpemi as solver_knpemi
import solver_emi as solver_emi

if __name__=='__main__':
    # resolution factor of mesh
    resolution = 1

    # time variables (seconds)
    dt = 1.0e-5                      # global time step (s)
    Tstop = 1.0e-2                   # global end time (s)

    # physical parameters
    C_M = 0.02                       # capacitance (F)
    temperature = 300                # temperature (K)
    F = 96485                        # Faraday's constant (C/mol)
    R = 8.314                        # Gas constant (J/(K*mol))
    g_Na_leak = Constant(30*0.2)     # Na leak membrane conductivity (S/(m^2))
    g_K_leak = Constant(30*0.8)      # K leak membrane conductivity (S/(m^2))
    g_Cl_leak = Constant(0.0)        # Cl leak membrane conductivity (S/(m^2))
    g_syn_bar = 1.25e3               # Na synaptic membrane conductivity (S/(m^2))
    D_Na = Constant(1.33e-9)         # Na diffusion coefficient (m^2/s)
    D_K = Constant(1.96e-9)          # K diffusion coefficient (m^2/s)
    D_Cl = Constant(2.03e-9)         # Cl diffusion coefficient (m^2/s)

    # EMI specific parameters
    sigma_i = 2.01202                # intracellular conductivity
    sigma_e = 1.31365                # extracellular conductivity
    E_Na = 54.8e-3                   # reversal potential Na (V)
    E_K = -88.98e-3                  # reversal potential K (V)
    g_Na_leak_emi = Constant(30*0.2) # Na leak membrane conductivity (S/(m^2))
    g_K_leak_emi = Constant(30*0.8)  # K leak membrane conductivity (S/(m^2))

    # initial conditions
    phi_M_init = Constant(-60e-3)    # membrane potential (V)
    Na_i_init = Constant(12)         # intracellular Na concentration (mol/m^3)
    Na_e_init = Constant(100)        # extracellular Na concentration (mol/m^3)
    K_i_init = Constant(125)         # intracellular K concentration (mol/m^3)
    K_e_init = Constant(4)           # extracellular K concentration (mol/m^3)
    Cl_i_init = Constant(137)        # intracellular Cl concentration (mol/m^3)
    Cl_e_init = Constant(104)        # extracellular Cl concentration (mol/m^3)

    # set parameters
    params = {'dt':dt, 'Tstop':Tstop,
              'temperature':temperature, 'R':R, 'F':F, 'C_M':C_M,
              'phi_M_init':phi_M_init,
              'sigma_i':sigma_i, 'sigma_e':sigma_e,
              'g_K_leak':g_K_leak_emi,
              'g_Na_leak':g_Na_leak_emi,
              'E_Na':E_Na, 'E_K':E_K}

    # create ions (Na conductivity is set below for each model)
    Na = {'Di':D_Na, 'De':D_Na, 'ki_init':Na_i_init,
          'ke_init':Na_e_init, 'z':1.0, 'name':'Na'}
    K = {'Di':D_K, 'De':D_K, 'ki_init':K_i_init,
         'ke_init':K_e_init, 'z':1.0, 'name':'K'}
    Cl = {'Di':D_Cl, 'De':D_Cl, 'ki_init':Cl_i_init,
          'ke_init':Cl_e_init, 'z':-1.0, 'name':'Cl'}
    # create ion list
    ion_list = [Na, K, Cl]

    #####################################################################
    # Setup for model C1: one neuron
    # get mesh, subdomains, surfaces paths
    here = os.path.abspath(os.path.dirname(__file__))
    mesh_prefix_C1 = os.path.join(here, 'meshes/one_neuron_2d/')
    mesh_C1 = mesh_prefix_C1 + 'mesh_' + str(resolution) + '.xml'
    subdomains_C1 = mesh_prefix_C1 + 'subdomains_' + str(resolution) + '.xml'
    surfaces_C1 = mesh_prefix_C1 + 'surfaces_' + str(resolution) + '.xml'
    # generate mesh if it does not exist
    if not os.path.isfile(mesh_C1):
        script_C1 = 'make_mesh_one_neuron_2D.py '                # script
        os.system('python3 ' + script_C1 + ' ' + str(resolution)) # run script

    # synaptic current
    g_syn_C1 = Expression('g_syn_bar*(x[0] <= 40e-6)', g_syn_bar=g_syn_bar, degree=4)

    # Run (1a) with model C1 with KNP-EMI
    sys.stdout.write("\nRunning KNP-EMI using model C1 (one neuron)\n")

    t_1a = Constant(0.0)                                        # time constant
    fname_1a = 'results/data/one_neuron/res_' + str(resolution) # filename for results
    # set conductivity and synaptic current
    ion_list[0]['g_k'] = g_Na_leak + g_syn_C1 # Na
    ion_list[1]['g_k'] = g_K_leak             # K
    ion_list[2]['g_k'] = g_Cl_leak            # Cl
    # solve system
    S_1a = solver_knpemi.Solver(ion_list, t_1a, **params)  # create solver
    S_1a.setup_domain(mesh_C1, subdomains_C1, surfaces_C1) # setup meshes
    S_1a.solve_system_passive(filename=fname_1a)           # solve

    # Run (1b) with model C1 with EMI
    sys.stdout.write("\nRunning EMI using model C1 (one neuron)\n")

    t_1b = Constant(0.0)                                            # time constant
    fname_1b = 'results/data/one_neuron_emi/res_' + str(resolution) # filename for results
    # set synaptic current
    params['g_ch_syn'] = g_syn_C1
    # solve system
    S_1b = solver_emi.Solver(t_1b, **params)               # create solver
    S_1b.setup_domain(mesh_C1, subdomains_C1, surfaces_C1) # setup meshes
    S_1b.solve_system_passive(filename=fname_1b)           # solve

    #####################################################################
    # Setup for model C2: two neurons with 3 um distance between neurons
    # get mesh, subdomains, surfaces paths
    here = os.path.abspath(os.path.dirname(__file__))
    mesh_prefix_C2 = os.path.join(here, 'meshes/two_neurons_2d/')
    mesh_C2 = mesh_prefix_C2 + 'mesh_' + str(resolution) + '.xml'
    subdomains_C2 = mesh_prefix_C2 + 'subdomains_' + str(resolution) + '.xml'
    surfaces_C2 = mesh_prefix_C2 + 'surfaces_' + str(resolution) + '.xml'
    # generate mesh if it does not exist
    if not os.path.isdir(mesh_C2):
        script_C2 = 'make_mesh_two_neurons_2D.py '               # script
        os.system('python3 ' + script_C2 + ' ' + str(resolution)) # run script

    # synaptic current C2
    g_syn_C2 = Expression('g_syn_bar*((x[0] >= 55e-6)* \
                          (x[0] <= 60e-6)*(x[1] <= 60e-6) + \
                          (x[0] >= 60e-6)*(x[0] <= 65e-6)*(x[1] >= 60e-6))',
                           g_syn_bar=g_syn_bar, degree=4)

    # Run (2a) with model C2 with KNP-EMI
    sys.stdout.write("\nRunning KNP-EMI using model C2 (two neurons)\n")

    t_2a = Constant(0.0)                                         # time constant
    fname_2a = 'results/data/two_neurons/res_' + str(resolution) # filename for results
    # set conductivity and synaptic current
    ion_list[0]['g_k'] = g_Na_leak + g_syn_C2 # Na
    ion_list[1]['g_k'] = g_K_leak             # K
    ion_list[2]['g_k'] = g_Cl_leak            # Cl
    # solve system
    S_2a = solver_knpemi.Solver(ion_list, t_2a, **params)  # create solver
    S_2a.setup_domain(mesh_C2, subdomains_C2, surfaces_C2) # setup meshes
    S_2a.solve_system_passive(filename=fname_2a)           # solve

    # Run (2b) with model C2 with EMI
    sys.stdout.write("\nRunning EMI using model C2 (two neurons)\n")

    t_2b = Constant(0.0)                                             # time constant
    fname_2b = 'results/data/two_neurons_emi/res_' + str(resolution) # filename for results
    # set synaptic current
    params['g_ch_syn'] = g_syn_C2
    # solve system
    S_2b = solver_emi.Solver(t_2b, **params)               # create solver
    S_2b.setup_domain(mesh_C2, subdomains_C2, surfaces_C2) # setup meshes
    S_2b.solve_system_passive(filename=fname_2b)           # solve

    #####################################################################
    # Setup for model C3: two neurons with 10 um distance between neurons
    # get mesh, subdomains, surfaces paths
    here = os.path.abspath(os.path.dirname(__file__))
    mesh_prefix_C3 = os.path.join(here, 'meshes/two_neurons_wide_2d/')
    mesh_C3 = mesh_prefix_C3 + 'mesh_' + str(resolution) + '.xml'
    subdomains_C3 = mesh_prefix_C3 + 'subdomains_' + str(resolution) + '.xml'
    surfaces_C3 = mesh_prefix_C3 + 'surfaces_' + str(resolution) + '.xml'
    # generate mesh if it does not exist
    if not os.path.isdir(mesh_C3):
        script_C3 = 'make_mesh_two_neurons_wide_2D.py '          # script
        os.system('python3 ' + script_C3 + ' ' + str(resolution)) # run script

    # synaptic current
    g_syn_C3 = Expression('g_syn_bar*((x[0] >= 55e-6)*(x[0] <= 60e-6)*\
                          (x[1] <= 60e-6) + (x[0] >= 60e-6)*(x[0] <= 65e-6)*\
                          (x[1] >= 60e-6))', g_syn_bar=g_syn_bar, degree=4)

    # Run (3a) with model C3 with KNP-EMI
    sys.stdout.write("\nRunning KNP-EMI using model C3 (two neurons wide)\n")

    t_3a = Constant(0.0)                                              # time constant
    fname_3a = 'results/data/two_neurons_wide/res_' + str(resolution) # filename for results
    # set conductivity and synaptic current
    ion_list[0]['g_k'] = g_Na_leak + g_syn_C3 # Na
    ion_list[1]['g_k'] = g_K_leak             # K
    ion_list[2]['g_k'] = g_Cl_leak            # Cl
    # solve system
    S_3a = solver_knpemi.Solver(ion_list, t_3a, **params)  # create solver
    S_3a.setup_domain(mesh_C3, subdomains_C3, surfaces_C3) # setup meshes
    S_3a.solve_system_passive(filename=fname_3a)           # solve

    # Run (3b) with model C3 with EMI
    sys.stdout.write("\nRunning EMI using model C3 (two neurons wide)\n")

    t_3b = Constant(0.0)                                                  # time constant
    fname_3b = 'results/data/two_neurons_wide_emi/res_' + str(resolution) # filename for results
    # set synaptic current
    params['g_ch_syn'] = g_syn_C3
    # solve system
    S_3b = solver_emi.Solver(t_3b, **params)               # create solver
    S_3b.setup_domain(mesh_C3, subdomains_C3, surfaces_C3) # setup meshes
    S_3b.solve_system_passive(filename=fname_3b)           # solve
