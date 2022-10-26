#!/usr/bin/python3

"""
Solve KNP-EMI with passive membrane mechanisms for a 2D box mesh, refined 3
times through 4 iterations.
"""

import os
import sys
from time import sleep

from dolfin import *
import solver_knpemi as solver_knpemi

if __name__=='__main__':
    # time variables
    dt = 1.0e-5              # time step (s)
    Tstop = 1.0e-2           # global end time (s)
    # physical parameters
    C_M = 0.02               # capacitance (F)
    temperature = 300        # temperature (K)
    F = 96485                # Faraday's constant (C/mol)
    R = 8.314                # Gas constant (J/(K*mol))
    D_Na = Constant(1.33e-9) # diffusion coefficients Na (m/s)
    D_K = Constant(1.96e-9)  # diffusion coefficients K (m/s)
    D_Cl = Constant(2.03e-9) # diffusion coefficients Cl (m/s)

    # initial conditions
    phi_M_init = Constant(-60e-3) # membrane potential (V)
    Na_i_init = Constant(12)      # intracellular Na concentration (mol/m^3)
    Na_e_init = Constant(100)     # extracellular Na concentration (mol/m^3)
    K_i_init = Constant(125)      # intracellular K concentration (mol/m^3)
    K_e_init = Constant(4)        # extracellular K concentration (mol/m^3)
    Cl_i_init = Constant(137)     # intracellular Cl concentration (mol/m^3)
    Cl_e_init = Constant(104)     # extracellular Cl concentration (mol/m^3)

    # set parameters
    params = {'dt':dt, 'Tstop':Tstop, 'C_M':C_M, 'temperature':temperature,
              'R':R, 'F':F, 'phi_M_init':phi_M_init}

    # synaptic current
    g_syn_bar = 1.25e3  # synaptic conductance (S/(m**2))
    g_syn = Expression('g_syn_bar*(x[0] < 11.0e-6)', g_syn_bar=g_syn_bar, degree=4)
    # membrane conductivities (S/(m**2))
    g_Na = Constant(0.2*30) + g_syn
    g_K = Constant(0.8*30)
    g_Cl = Constant(0.0)

    # create ions
    Na = {'Di':D_Na, 'De':D_Na, 'ki_init':Na_i_init, 'ke_init':Na_e_init,
          'z':1.0, 'g_k':g_Na, 'name':'Na'}
    K = {'Di':D_K, 'De':D_K, 'ki_init':K_i_init, 'ke_init':K_e_init,
         'z':1.0, 'g_k':g_K, 'name':'K'}
    Cl = {'Di':D_Cl, 'De':D_Cl, 'ki_init':Cl_i_init, 'ke_init':Cl_e_init,
          'z':-1.0, 'g_k':g_Cl, 'name':'Cl'}

    # create ion list
    ion_list = [Na, K, Cl]

    # solve for N different meshes (2D block mesh refined N-1 times
    for resolution in range(4):
        sys.stdout.write("Running refinement test - resolution %d" % resolution)
        sleep(0.25)

        # get path to mesh, subdomains, surfaces
        here = os.path.abspath(os.path.dirname(__file__))
        mesh_prefix = os.path.join(here, 'meshes/refinement_test/')
        mesh = mesh_prefix + 'mesh_' + str(resolution) + '.xml'
        subdomains = mesh_prefix + 'subdomains_' + str(resolution) + '.xml'
        surfaces = mesh_prefix + 'surfaces_' + str(resolution) + '.xml'

        # generate mesh if mesh does not exist
        if not os.path.isfile(mesh):
            script = 'make_mesh_refinement_test.py '              # script
            os.system('python3 ' + script + ' ' + str(resolution)) # run script

        t = Constant(0.0)                                      # time constant (s)
        S = solver_knpemi.Solver(ion_list, t, **params)        # create solver
        S.setup_domain(mesh, subdomains, surfaces)             # setup meshes
        fname = 'results/data/refinement_test/res_' + str(resolution) # result file
        S.solve_system_passive(fname)                          # solve
