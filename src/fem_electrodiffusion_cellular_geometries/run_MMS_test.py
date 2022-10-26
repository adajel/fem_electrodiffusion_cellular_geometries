#!/usr/bin/python3

"""
Solve KNP-EMI with passive membrane mechanisms for a 2D box mesh, refined N-1
times through N iterations.
"""

from dolfin import *

import os
import sys
import numpy as np
from time import sleep

import solver_knpemi as solver_knpemi
from utils import setup_MMS

if __name__=='__main__':
    # time variables
    dt_0 = 1.0e-5/64
    Tstop = dt_0*2      # end time

    # physical parameters
    C_M = 1.0           # capacitance F)
    temperature = 1.0   # temperature (K)
    F = 1.0             # Faraday's constant (C/mol)
    R = 1.0             # Gas constant (J/(K*mol))
    # membrane conductivities
    g_Na = Constant(1.0)
    g_K = Constant(1.0)
    g_Cl = Constant(1.0)

    # create directory for saving results, if it does not exist
    here = os.path.abspath(os.path.dirname(__file__))
    results_dir = os.path.join(here, 'results/data/convergence_rates')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    ## create files for saving results in Latex table format
    title_f1 = results_dir + '/data_Na.txt'
    title_f2 = results_dir + '/data_K.txt'
    title_f3 = results_dir + '/data_Cl.txt'
    title_f4 = results_dir + '/data_phi.txt'
    title_f5 = results_dir + '/data_JM.txt'

    ## open files
    f1 = open(title_f1, 'w+')
    f2 = open(title_f2, 'w+')
    f3 = open(title_f3, 'w+')
    f4 = open(title_f4, 'w+')
    f5 = open(title_f5, 'w+')

    i = 0
    # create UnitSquareMesh(n,n) with n = 2**x, x = {3, 4, 5, ...} and solve
    # system with using method of manufactured solutions (MMS)
    for resolution in [3, 4, 5, 6, 7, 8]:
        sys.stdout.write("Running MMS test with n=%d" % resolution)
        sleep(0.25)

        t = Constant(0.0) # time constant (s)
        n = 2**resolution # number of cells
        dt = dt_0/(4**i)  # time step

        # get MMS terms and exact solutions
        M = setup_MMS()
        src_terms, exact_sols, init_conds, bndry_terms, subdomains_MMS = M.get_MMS_terms_KNPEMI(t)

        # initial values
        phi_M_init = init_conds['phi_M'] # membrane potential (V)

        # create ions
        Na = {'Di':1.0, 'De':1.0, 'z':1.0,
              'ki_init':init_conds['Na_i'],
              'ke_init':init_conds['Na_e'],
              'g_k':g_Na,
              'f_k_i':src_terms['f_Na_i'],
              'f_k_e':src_terms['f_Na_e'],
              'J_k_e':bndry_terms['J_Na_e'],
              'phi_i_e':exact_sols['phi_i_e'],
              'f_phi_i':src_terms['f_phi_i'],
              'f_phi_e':src_terms['f_phi_e'],
              'f_g_M':src_terms['f_g_M'],
              'f_I_M':src_terms['f_I_M'],
              'name':'Na'}

        K = {'Di':1.0, 'De':1.0, 'z':1.0,
             'ki_init':init_conds['K_i'],
             'ke_init':init_conds['K_e'],
             'g_k':g_K,
             'f_k_i':src_terms['f_K_i'],
             'f_k_e':src_terms['f_K_e'],
             'J_k_e':bndry_terms['J_K_e'],
             'phi_i_e':exact_sols['phi_i_e'],
             'f_phi_i':src_terms['f_phi_i'],
             'f_phi_e':src_terms['f_phi_e'],
             'f_g_M':src_terms['f_g_M'],
             'f_I_M':src_terms['f_I_M'],
              'name':'K'}

        Cl = {'Di':1.0, 'De':1.0, 'z':-1.0,
              'ki_init':init_conds['Cl_i'],
              'ke_init':init_conds['Cl_e'],
              'g_k':g_Cl,
              'f_k_i':src_terms['f_Cl_i'],
              'f_k_e':src_terms['f_Cl_e'],
              'J_k_e':bndry_terms['J_Cl_e'],
              'phi_i_e':exact_sols['phi_i_e'],
              'f_phi_i':src_terms['f_phi_i'],
              'f_phi_e':src_terms['f_phi_e'],
              'f_g_M':src_terms['f_g_M'],
              'f_I_M':src_terms['f_I_M'],
              'name':'Cl'}

        # create ion list
        ion_list = [Na, K, Cl]

        # set parameters
        params = {'dt':dt,
                  'Tstop':Tstop,
                  'C_M':C_M,
                  'temperature':temperature,
                  'R':R,
                  'F':F,
                  'phi_M_init':phi_M_init}

        # get mesh, subdomains, surfaces path
        here = os.path.abspath(os.path.dirname(__file__))
        mesh_prefix = os.path.join(here, 'meshes/MMS/')
        mesh_path = mesh_prefix + 'mesh_' + str(resolution) + '.xml'
        subdomains_path = mesh_prefix + 'subdomains_' + str(resolution) + '.xml'
        surfaces_path = mesh_prefix + 'surfaces_' + str(resolution) + '.xml'

        # generate mesh if it does not exist
        if not os.path.isfile(mesh_path):
            script = 'make_mesh_MMS.py '                          # script
            os.system('python3 ' + script + ' ' + str(resolution)) # run script

        # create solver
        S = solver_knpemi.Solver(ion_list, t, M, **params)
        # setup meshes in solver
        S.setup_domain(mesh_path, subdomains_path, surfaces_path)
        # solver system
        fname = 'results/data/tmp/tmp'
        S.solve_system_passive(filename=fname)

        ## Extract solutions
        # Intracellular
        Na_i = S.wh.sub(0, deepcopy=True).sub(0)     # Na concentration
        K_i = S.wh.sub(0, deepcopy=True).sub(1)      # K concentration
        Cl_i = S.wh.sub(0, deepcopy=True).sub(2)     # Cl concentration
        phi_i = S.wh.sub(0, deepcopy=True).sub(3)    # potential
        # Extracellular
        Na_e = S.wh.sub(1, deepcopy=True).sub(0)     # Na concentration
        K_e = S.wh.sub(1, deepcopy=True).sub(1)      # K concentration
        Cl_e = S.wh.sub(1, deepcopy=True).sub(2)     # Cl concentration
        phi_e = S.wh.sub(1, deepcopy=True).sub(3)    # potential
        # Membrane
        I_M = S.wh.sub(2, deepcopy=True)             # potential

        # function space for exact solutions
        VI = FiniteElement('CG', S.interior_mesh.ufl_cell(), 4)   # define element
        VI = FunctionSpace(S.interior_mesh, VI)                   # define function space
        VE = FiniteElement('CG', S.exterior_mesh.ufl_cell(), 4)   # define element
        VE = FunctionSpace(S.exterior_mesh, VE)                   # define function space
        VG = FiniteElement('CG', S.gamma_mesh.ufl_cell(), 4)      # define element
        VG = FunctionSpace(S.gamma_mesh, VG)                      # define function space

        # interpolate exact solutions into spaces defined above
        Na_i_e = interpolate(exact_sols['Na_i_e'], VI)         # Na intracellular
        Na_e_e = interpolate(exact_sols['Na_e_e'], VE)         # Na extracellular
        K_i_e = interpolate(exact_sols['K_i_e'], VI)           # K intracellular
        K_e_e = interpolate(exact_sols['K_e_e'], VE)           # K extracellular
        Cl_i_e = interpolate(exact_sols['Cl_i_e'], VI)         # Cl intracellular
        Cl_e_e = interpolate(exact_sols['Cl_e_e'], VE)         # Cl extracellular
        phi_i_e = interpolate(exact_sols['phi_i_e'], VI)       # phi intracellular
        phi_e_e = interpolate(exact_sols['phi_e_e'], VE)       # phi extracellular

        # get error L2
        Nai_L2 = errornorm(Na_i_e, Na_i, 'L2', degree_rise=4)
        Nae_L2 = errornorm(Na_e_e, Na_e, 'L2', degree_rise=4)
        Ki_L2 = errornorm(K_i_e, K_i, 'L2', degree_rise=4)
        Ke_L2 = errornorm(K_e_e, K_e, 'L2', degree_rise=4)
        Cli_L2 = errornorm(Cl_i_e, Cl_i, 'L2', degree_rise=4)
        Cle_L2 = errornorm(Cl_e_e, Cl_e, 'L2', degree_rise=4)
        phii_L2 = errornorm(phi_i_e, phi_i, 'L2', degree_rise=4)
        phie_L2 = errornorm(phi_e_e, phi_e, 'L2', degree_rise=4)
        JM_L2 = M.broken_L2_norm(exact_sols['I_M_e'], I_M, subdomains_MMS[1])

        # get error H1
        Nai_H1 = errornorm(Na_i_e, Na_i, 'H1', degree_rise=4)
        Nae_H1 = errornorm(Na_e_e, Na_e, 'H1', degree_rise=4)
        Ki_H1 = errornorm(K_i_e, K_i, 'H1', degree_rise=4)
        Ke_H1 = errornorm(K_e_e, K_e, 'H1', degree_rise=4)
        Cli_H1 = errornorm(Cl_i_e, Cl_i, 'H1', degree_rise=4)
        Cle_H1 = errornorm(Cl_e_e, Cl_e, 'H1', degree_rise=4)
        phii_H1 = errornorm(phi_i_e, phi_i, 'H1', degree_rise=4)
        phie_H1 = errornorm(phi_e_e, phi_e, 'H1', degree_rise=4)

        # mesh minimum "diameter"
        hi = S.interior_mesh.hmin()
        he = S.exterior_mesh.hmin()
        hg = S.gamma_mesh.hmin()

        if i == 0:
            # write to file - L2/H1 err - alpha
            f1.write('%g & %.2E(---) & %.2E(---) & %.2E(---) & %.2E(---) \\\\' % (n,\
                     Nai_L2,  Nae_L2,
                     Nai_H1,  Nae_H1))
            # write to file - L2/H1 err - alpha
            f2.write('%g & %.2E(---) & %.2E(---) & %.2E(---) & %.2E(---) \\\\' % (n,\
                     Ki_L2, Ke_L2,
                     Ki_H1, Ke_H1))
            # write to file - L2/H1 err - alpha
            f3.write('%g & %.2E(---) & %.2E(---) & %.2E(---) & %.2E(---) \\\\' % (n,\
                     Cli_L2, Cle_L2,
                     Cli_H1, Cle_H1))
            # write to file - L2/H1 err - alpha
            f4.write('%g & %.2E(---) & %.2E(---) & %.2E(---) & %.2E(---) \\\\' % (n,\
                     phii_L2, phie_L2,
                     phii_H1, phie_H1))
            # write to file - L2/H1 err - alpha
            f5.write('%g & %.2E(---) \\\\' % (n, JM_L2))

        else:
            # calculate L2 rates
            r_Nai_L2 = np.log(Nai_L2/Nai_L2_0)/np.log(hi/hi0)
            r_Nae_L2 = np.log(Nae_L2/Nae_L2_0)/np.log(he/he0)
            r_Ki_L2 = np.log(Ki_L2/Ki_L2_0)/np.log(hi/hi0)
            r_Ke_L2 = np.log(Ke_L2/Ke_L2_0)/np.log(he/he0)
            r_Cli_L2 = np.log(Cli_L2/Cli_L2_0)/np.log(hi/hi0)
            r_Cle_L2 = np.log(Cle_L2/Cle_L2_0)/np.log(he/he0)
            r_phii_L2 = np.log(phii_L2/phii_L2_0)/np.log(hi/hi0)
            r_phie_L2 = np.log(phie_L2/phie_L2_0)/np.log(he/he0)
            r_JM_L2 = np.log(JM_L2/JM_L2_0)/np.log(hg/hg0)

            # calculate H1 rates
            r_Nai_H1 = np.log(Nai_H1/Nai_H1_0)/np.log(hi/hi0)
            r_Nae_H1 = np.log(Nae_H1/Nae_H1_0)/np.log(he/he0)
            r_Ki_H1 = np.log(Ki_H1/Ki_H1_0)/np.log(hi/hi0)
            r_Ke_H1 = np.log(Ke_H1/Ke_H1_0)/np.log(he/he0)
            r_Cli_H1 = np.log(Cli_H1/Cli_H1_0)/np.log(hi/hi0)
            r_Cle_H1 = np.log(Cle_H1/Cle_H1_0)/np.log(he/he0)
            r_phii_H1 = np.log(phii_H1/phii_H1_0)/np.log(hi/hi0)
            r_phie_H1 = np.log(phie_H1/phie_H1_0)/np.log(he/he0)

            # write to file - L2/H1 err and rate - alpha
            f1.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (n,\
                     Nai_L2, r_Nai_L2, Nae_L2, r_Nae_L2,
                     Nai_H1, r_Nai_H1, Nae_H1, r_Nae_H1))
            # write to file - L2/H1 err and rate - alpha
            f2.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (n,\
                     Ki_L2, r_Ki_L2, Ke_L2, r_Ke_L2,
                     Ki_H1, r_Ki_H1, Ke_H1, r_Ke_H1))
            # write to file - L2/H1 err and rate - alpha
            f3.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (n,\
                     Cli_L2, r_Cli_L2, Cle_L2, r_Cle_L2,
                     Cli_H1, r_Cli_H1, Cle_H1, r_Cle_H1))
            # write to file - L2/H1 err and rate - alpha
            f4.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (n,\
                     phii_L2, r_phii_L2, phie_L2, r_phie_L2,
                     phii_H1, r_phii_H1, phie_H1, r_phie_H1))
            # write to file - L2/H1 err and rate - alpha
            f5.write('%g & %.2E(%.2f) \\\\' % (n,\
                     JM_L2, r_JM_L2))

        # update prev h
        hi0, he0, hg0 = hi, he, hg

        # update prev L2
        Nai_L2_0, Nae_L2_0 = Nai_L2, Nae_L2
        Ki_L2_0, Ke_L2_0 = Ki_L2, Ke_L2
        Cli_L2_0, Cle_L2_0 = Cli_L2, Cle_L2
        phii_L2_0, phie_L2_0 = phii_L2, phie_L2
        JM_L2_0 = JM_L2

        # update prev H1
        Nai_H1_0, Nae_H1_0 = Nai_H1, Nae_H1
        Ki_H1_0, Ke_H1_0 = Ki_H1, Ke_H1
        Cli_H1_0, Cle_H1_0 = Cli_H1, Cle_H1
        phii_H1_0, phie_H1_0 = phii_H1, phie_H1

        i += 1

    # close files
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
