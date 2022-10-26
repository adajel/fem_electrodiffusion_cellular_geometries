import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from dolfin import * 
import string

# set font & text parameters
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 13}

plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'

path = 'results/data/'

def get_plottable_ECS_function(h5_fname, c_range, n, i, cticks=None, \
                               clip=True, mean_normalize=True, scale=1.):
    # get plottable function of extracellular concentration or potential
    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5 = HDF5File(MPI.comm_world, h5_fname, 'r')
    hdf5.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e6
    hdf5.read(subdomains, '/subdomains')
    hdf5.read(surfaces, '/surfaces')

    interior_mesh = SubMesh(mesh, subdomains, 1)
    exterior_mesh = SubMesh(mesh, subdomains, 0)
    P1 = FiniteElement('CG', triangle, 1)
    R = FiniteElement('R', triangle, 0)
    Wi = FunctionSpace(interior_mesh, MixedElement(4*[P1] + [R]))
    We = FunctionSpace(exterior_mesh, MixedElement(4*[P1]))
    Vi = FunctionSpace(interior_mesh, P1)
    Ve = FunctionSpace(exterior_mesh, P1)

    ui = Function(Wi)
    ue = Function(We)
    fi = Function(Vi)
    fe = Function(Ve)

    if i != None:
        hdf5.read(ue, "/exterior_solution/vector_" + str(n))
        assign(fe, ue.sub(i))
    else:
        hdf5.read(fe, "/exterior_solution/vector_" + str(n))
 
    if mean_normalize: 
        fe.vector()[:] -= np.mean(fe.vector().get_local())
    fe.vector()[:] = scale*fe.vector().get_local()
    if clip:
        fe.vector()[:] = np.clip(fe.vector().get_local(), c_range[0], c_range[1])
    return fe

def get_time_series_ECS(dt, T, fname, x, y, z, EMI=False):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, 'r')

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e6
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    exterior_mesh = SubMesh(mesh, subdomains, 0)
    P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    We = FunctionSpace(exterior_mesh, MixedElement(4*[P1]))
    Ve = FunctionSpace(exterior_mesh, P1)

    ue = Function(We)

    f_phi_e = Function(Ve)
    f_Na_e = Function(Ve)
    f_K_e = Function(Ve)

    Na_e = []
    K_e = []
    phi_e = []

    for n in range(1, int(T/dt)):
        if EMI:
            # read file and append membrane potential
            hdf5file.read(f_phi_e, "/exterior_solution/vector_" + str(n))
            phi_e.append(1.0e3*f_phi_e(x, y, z))
        else:
            # read file
            hdf5file.read(ue, "/exterior_solution/vector_" + str(n))

            # potential
            assign(f_phi_e, ue.sub(3))
            phi_e.append(1.0e3*f_phi_e(x, y, z)) 

            # Na concentrations
            assign(f_Na_e, ue.sub(0))
            Na_e.append(f_Na_e(x, y, z))

            # K concentrations
            assign(f_K_e, ue.sub(1))
            K_e.append(f_K_e(x, y, z))
    return Na_e, K_e, phi_e

def get_time_series_ICS(dt, T, fname, x, y, z):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, 'r')

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e6
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    interior_mesh = SubMesh(mesh, subdomains, 1)
    P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    R = FiniteElement('R', mesh.ufl_cell(), 0)
    Wi = FunctionSpace(interior_mesh, MixedElement(4*[P1] + [R]))
    Vi = FunctionSpace(interior_mesh, P1)

    ui = Function(Wi)

    f_Na_i = Function(Vi)
    f_K_i = Function(Vi)
 
    Na_i = []
    K_i = []

    for n in range(1, int(T/dt)):
        # read file
        hdf5file.read(ui, "/interior_solution/vector_" + str(n))

        # Na concentration
        assign(f_Na_i, ui.sub(0))
        Na_i.append(f_Na_i(x, y, z))

        # K concentration
        assign(f_K_i, ui.sub(1))
        K_i.append(f_K_i(x, y, z))
    return Na_i, K_i

def get_time_series_gamma(dt, T, fname, x, y, z): 
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, 'r')

    # Membrane potentail
    gamma_mesh = Mesh()
    hdf5file.read(gamma_mesh, '/gamma_mesh', False)
    P1 = FiniteElement('P', gamma_mesh.ufl_cell(), 1)
    Vg = FunctionSpace(gamma_mesh, P1)
    gamma_mesh.coordinates()[:] *= 1e6

    f_phi_M = Function(Vg)
    phi_M = []

    for n in range(1, int(T/dt)):
        # read file
        hdf5file.read(f_phi_M, "/membrane_potential/vector_" + str(n))

        # membrane potential
        phi_M.append(1.0e3*f_phi_M(x, y, z))
    return phi_M


def plot_refinement_test_potential():
    # Refinement test - plot of extracellular potential
    fname_0 = path + 'refinement_test/res_0/results.h5'
    fname_1 = path + 'refinement_test/res_1/results.h5'
    fname_2 = path + 'refinement_test/res_2/results.h5'
    fname_3 = path + 'refinement_test/res_3/results.h5'

    # data parameters
    crange = [-0.2, 0.07]   # range to clip
    n = 1000                # time step to evaluate solution
    i = 3                   # get potential

    fe_0 = get_plottable_ECS_function(fname_0, crange, n, i, scale=1e3)
    fe_1 = get_plottable_ECS_function(fname_1, crange, n, i, scale=1e3)
    fe_2 = get_plottable_ECS_function(fname_2, crange , n, i, scale=1e3)
    fe_3 = get_plottable_ECS_function(fname_3, crange, n, i, scale=1e3)

    # plotting parameters
    cticks=[-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1]
    clabel=r'$\phi_e$ (mV)'
    xrange = [0,60]
    yrange = [15,45]

    # create plot
    fig = plt.figure(figsize=(9.5,5))
    ax = plt.gca()

    # subplot number 1
    ax1 = fig.add_subplot(2,2,1, xlim=xrange,ylim=yrange)
    plt.title(r'$\Delta x = 2 \, \mu$m')
    plt.yticks([20, 30, 40])
    plt.xticks([0, 20, 40, 60])
    plt.ylabel(r'$y$-position ($\mu$m)')
    c1 = plot(fe_0)

    # subplot number 2
    ax2 = fig.add_subplot(2,2,2, xlim=xrange, ylim=yrange)
    plt.title(r'$\Delta x = 1 \, \mu$m')
    plt.yticks([20, 30, 40])
    plt.xticks([0, 20, 40, 60])
    c2 = plot(fe_1)

    # subplot number 3
    ax3 = fig.add_subplot(2,2,3, xlim=xrange, ylim=yrange)
    plt.title(r'$\Delta x = 0.5 \, \mu$m')
    plt.yticks([20, 30, 40])
    plt.xticks([0, 20, 40, 60])
    plt.xlabel(r'$x$-position ($\mu$m)')
    plt.ylabel(r'$y$-position ($\mu$m)')
    c3 = plot(fe_2)

    # subplot number 4
    ax4 = fig.add_subplot(2,2,4, xlim=xrange, ylim=yrange)
    plt.title(r'$\Delta x = 0.25 \, \mu$m')
    plt.yticks([20, 30, 40])
    plt.xticks([0, 20, 40, 60])
    plt.xlabel(r'$x$-position ($\mu$m)')
    c4 = plot(fe_3)

    # add colorbar
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(c4, ticks=cticks, label=clabel, cax=cax)

    # make pretty
    fig.subplots_adjust(wspace=-0.2,hspace=0.1)
    plt.tight_layout()

    # add numbering for the subplots (A, B, C etc)
    letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}']
    for n, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.text(-0.1, 1.1, letters[n], transform=ax.transAxes)

    # save figure to file
    plt.savefig('results/figures/refinement_potential.svg', format='svg')
    return

def plot_refinement_concentration():
    # Refinement test - plot of extracellular sodium (Na)
    fname_0 = path + 'refinement_test/res_0/results.h5'
    fname_1 = path + 'refinement_test/res_1/results.h5'
    fname_2 = path + 'refinement_test/res_2/results.h5'
    fname_3 = path + 'refinement_test/res_3/results.h5'

    # data parameters
    crange = [99.7, 100]    # range to clip
    n = 1000                # time step to evaluate solution
    i = 0                   # get Na concentration

    fe_0 = get_plottable_ECS_function(fname_0, crange, n, i, clip=True, mean_normalize=False)
    fe_1 = get_plottable_ECS_function(fname_1, crange, n, i, clip=True, mean_normalize=False)
    fe_2 = get_plottable_ECS_function(fname_2, crange, n, i, clip=True, mean_normalize=False)
    fe_3 = get_plottable_ECS_function(fname_3, crange, n, i, clip=True, mean_normalize=False)

    # plotting parameters
    xrange = [0,60]
    yrange = [15,45]
    cticks=[99.7, 99.75, 99.8, 99.85, 99.9, 99.95, 100]
    clabel=r'[Na]$_e$ (mM)'

    # create plot
    fig = plt.figure(figsize=(9.5,5))
    ax = plt.gca()

    # subplot number 1
    ax1 = fig.add_subplot(2,2,1, xlim=xrange,ylim=yrange)
    plt.title(r'$\Delta x = 2 \, \mu$m')
    plt.yticks([20, 30, 40])
    plt.xticks([0, 20, 40, 60])
    plt.ylabel(r'$y$-position ($\mu$m)')
    c1 = plot(fe_0)

    # subplot number 2
    ax2 = fig.add_subplot(2,2,2, xlim=xrange, ylim=yrange)
    plt.title(r'$\Delta x = 1 \, \mu$m')
    plt.yticks([20, 30, 40])
    plt.xticks([0, 20, 40, 60])
    c2 = plot(fe_1)

    # subplot number 2
    ax3 = fig.add_subplot(2,2,3, xlim=xrange, ylim=yrange)
    plt.title(r'$\Delta x = 0.5 \, \mu$m')
    plt.yticks([20, 30, 40])
    plt.xticks([0, 20, 40, 60])
    plt.xlabel(r'$x$-position ($\mu$m)')
    plt.ylabel(r'$y$-position ($\mu$m)')
    c3 = plot(fe_2)

    # subplot number 2
    ax4 = fig.add_subplot(2,2,4, xlim=xrange, ylim=yrange)
    plt.title(r'$\Delta x = 0.25 \, \mu$m')
    plt.yticks([20, 30, 40])
    plt.xticks([0, 20, 40, 60])
    plt.xlabel(r'$x$-position ($\mu$m)')
    c4 = plot(fe_3)

    # add colorbar
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(c4, ticks=cticks, label=clabel, cax=cax)

    # make pretty
    fig.subplots_adjust(wspace=-0.2,hspace=0.1)
    plt.tight_layout()

    # add numbering for the subplots (A, B, C etc)
    letters = [r'\textbf{E}', r'\textbf{F}', r'\textbf{G}', r'\textbf{H}']
    for n, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.text(-0.1, 1.1, letters[n], transform=ax.transAxes)

    # save figure to file
    plt.savefig("results/figures/refinement_concentration.svg", format='svg')
    return

def plot_one_neuron(res):
    # one neuron
    fname = path + 'one_neuron/res_' + res + '/results.h5'         # KNP-EMI
    fname_emi = path + 'one_neuron_emi/res_' + res + '/results.h5' # EMI

    # data parameters
    c_range = [-0.13, 0.05]
    n = 1000
    i = 3

    fe_0 = get_plottable_ECS_function(fname, c_range, n, i, scale=1e3)
    fe_1 = get_plottable_ECS_function(fname_emi, c_range, n, None, scale=1e3)
    fe_2 = get_plottable_ECS_function(fname, c_range, n, i, clip=False, mean_normalize=False, scale=1e3)
    fe_3 = get_plottable_ECS_function(fname_emi, c_range, n, None, clip=False, mean_normalize=False, scale=1e3)

    y_pos = 65
    x_pos = np.linspace(35, 85, n)
    phi = np.zeros(n)
    phi_emi = np.zeros(n)
    for i in range(n):
        phi[i] = fe_2(x_pos[i], y_pos)
        phi_emi[i] = fe_3(x_pos[i], y_pos)

    # plotting parameters
    xrange = [30, 90]
    yrange = [40, 80]
    clabel=r'$\phi_e$ (mV)'

    # create plot
    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()

    # subplot number 1 - extracellular potential KNP-EMI
    ax1 = fig.add_subplot(2,2,2, xlim=xrange, ylim=yrange)
    plt.title(r'$\phi_e$ (KNP-EMI)')
    plt.ylabel(r'$y$-position ($\mu$m)')
    plt.xticks([40, 60, 80])
    plt.yticks([45, 60, 75])
    c1 = plot(fe_0)

    # subplot number 2 - extracellular potentail EMI
    ax2 = fig.add_subplot(2,2,4, xlim=xrange, ylim=yrange)
    plt.title(r'$\phi_e$ (EMI)')
    plt.ylabel(r'$y$-position ($\mu$m)')
    plt.xlabel(r'$x$-position ($\mu$m)')
    plt.xticks([40, 60, 80])
    plt.yticks([45, 60, 75])
    c2 = plot(fe_1)

    # subplot number 3 - comparisom ECS potentials
    ax3 = fig.add_subplot(1,2,1)
    plt.plot(x_pos, phi-phi[0], label='KNP-EMI', linewidth=3)
    plt.plot(x_pos, phi_emi-phi_emi[0], '--', label='EMI', linewidth=3)
    plt.title(r'2 $\mu$m above the cell')
    plt.ylabel(r'$\phi_e - \phi_{ref}$ (mV)')
    plt.xlabel(r'$x$-position ($\mu$m)')
    plt.xlim(35,85)
    plt.yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    plt.xticks([40, 50, 60, 70, 80])
    plt.legend()

    # add colorbar
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(c1, label=clabel, cax=cax)

    # make pretty
    fig.subplots_adjust(wspace=0.1,hspace=0.0)
    plt.tight_layout()

    # add numbering for the subplots (A, B, C etc)
    letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}']
    ax3.text(-0.04, 1.02, letters[0], transform=ax.transAxes)
    for n, ax in enumerate([ax1, ax2]):
        ax.text(-0.12, 1.06, letters[n+1], transform=ax.transAxes)

    # save figure to file
    plt.savefig("results/figures/one_neuron.svg", format='svg')
    return

def plot_two_neurons(res):
    # Two neurons - narrow
    fname = path + 'two_neurons/res_' + res + '/results.h5'         # KNP-EMI
    fname_emi = path + 'two_neurons_emi/res_' + res + '/results.h5' # EMI

    # data parameters
    c_range = [-0.4, 0.1]
    n = 1000
    i = 3

    fe_0 = get_plottable_ECS_function(fname, c_range, n, i, scale=1e3)
    fe_1 = get_plottable_ECS_function(fname_emi, c_range, n, None, scale=1e3)
    fe_2 = get_plottable_ECS_function(fname, c_range, n, i, clip=False, mean_normalize=False, scale=1e3)
    fe_3 = get_plottable_ECS_function(fname_emi, c_range, n, None, clip=False, mean_normalize=False, scale=1e3)

    y_pos = 60
    x_pos = np.linspace(35.001, 84.999, n)
    phi = np.zeros(n)
    phi_emi = np.zeros(n)
    for i in range(n):
        phi[i] = fe_2(x_pos[i], y_pos)
        phi_emi[i] = fe_3(x_pos[i], y_pos)
 
    # plotting parameters
    xrange = [30, 90]
    yrange = [40, 80]
    cticks = np.linspace(-0.4, 0.1, 11)  # ticks colorbar
    clevels = np.linspace(-0.4, 0.1, 32) # range values colorbar
    clabel = r'$\phi_e$ (mV)'            # label colorbar

    # create plot
    fig = plt.figure(figsize=(9,5))
    ax = plt.gca()

    # subplot number 1 - extracellular potential KNP-EMI
    ax1 = fig.add_subplot(2,2,2)
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.title(r'$\phi_e$ (KNP-EMI)')
    plt.ylabel(r'$y$-position ($\mu$m)')
    plt.xticks([40, 60, 80])
    plt.yticks([45, 60, 75])
    c1 = plot(fe_0, levels=clevels)

    # subplot number 2 - extracellular potential EMI
    ax2 = fig.add_subplot(2,2,4)
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.title(r'$\phi_e$ (EMI)')
    plt.xlabel(r'$x$-position ($\mu$m)')
    plt.ylabel(r'$y$-position ($\mu$m)')
    plt.xticks([40, 60, 80])
    plt.yticks([45, 60, 75])
    c2 = plot(fe_1, levels=clevels)

    # subplot number 3 - comparison membrane potentials
    ax3 = fig.add_subplot(1,2,1)
    plt.plot(x_pos, phi-phi[0], label='KNP-EMI', linewidth=3)
    plt.plot(x_pos, phi_emi-phi_emi[0], '--', label='EMI', linewidth=3)
    plt.ylabel(r'$\phi_e - \phi_{ref}$ (mV)')
    plt.xlabel(r'$x$-position ($\mu$m)')
    plt.xlim([35,85])
    plt.ylim([-0.345, 0.01])
    plt.yticks([-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0])
    plt.title(r'Between cells')
    plt.legend()

    # add colorbar
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(c1, ticks=cticks, label=clabel, cax=cax)

    # make pretty
    fig.subplots_adjust(wspace=0.1, hspace=0.0)
    plt.tight_layout()

    # add numbering for the subplots (A, B, C etc)
    letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}']
    ax3.text(-0.04, 1.02, letters[0], transform=ax.transAxes)
    for n, ax in enumerate([ax1, ax2]):
        ax.text(-0.12, 1.06, letters[n+1], transform=ax.transAxes)

    # save figure to file
    plt.savefig("results/figures/two_neurons.svg", format='svg')
    return

def plot_two_neurons_wide(res):
    # Two neurons - wide
    fname = path + 'two_neurons_wide/res_' + res + '/results.h5'     # KNP-EMI
    fname_emi = path + 'two_neurons_wide_emi/res_' + res + '/results.h5' # EMI

    # define parameters
    xrange = [30, 90]
    yrange = [40, 80]
    c_range = [-0.4, 0.1]
    cticks = np.linspace(-0.4, 0.1, 11)  # ticks colorbar
    clevels = np.linspace(-0.4, 0.1, 32) # range values colorbar
    clabel = r'$\phi_e$ (mV)'            # label colorbar

    n = 1000
    i = 3

    fe_0 = get_plottable_ECS_function(fname, c_range, n, i, scale=1e3)
    fe_1 = get_plottable_ECS_function(fname_emi, c_range, n, None, scale=1e3)
    fe_2 = get_plottable_ECS_function(fname, c_range, n, i, clip=False, mean_normalize=False, scale=1e3)
    fe_3 = get_plottable_ECS_function(fname_emi, c_range, n, None, clip=False, mean_normalize=False, scale=1e3)

    y_pos = 60
    x_pos = np.linspace(35.001, 84.999, n)
    phi = np.zeros(n)
    phi_emi = np.zeros(n)
    for i in range(n):
        phi[i] = fe_2(x_pos[i], y_pos)
        phi_emi[i] = fe_3(x_pos[i], y_pos)

    # create plot
    fig = plt.figure(figsize=(9,5))
    ax = plt.gca()

    # subplot number 1 - extracellular potential KNP-EMI
    ax1 = fig.add_subplot(2,2,2)
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.title(r'$\phi_e$ (KNP-EMI)')
    plt.ylabel(r'$y$-position ($\mu$m)')
    plt.xticks([40, 60, 80])
    plt.yticks([45, 60, 75])
    c1 = plot(fe_0, levels=clevels)

    # subplot number 2 - extracellular potential EMI
    ax2 = fig.add_subplot(2,2,4)
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.title(r'$\phi_e$ (EMI)')
    plt.xlabel(r'$x$-position ($\mu$m)')
    plt.ylabel(r'$y$-position ($\mu$m)')
    plt.xticks([40, 60, 80])
    plt.yticks([45, 60, 75])
    c2 = plot(fe_1, levels=clevels)

    # subplot number 3 - comparison membrane potentials
    ax3 = fig.add_subplot(1,2,1)
    plt.plot(x_pos, phi-phi[0], label='KNP-EMI', linewidth=3)
    plt.plot(x_pos, phi_emi-phi_emi[0], '--', label='EMI', linewidth=3)
    plt.ylabel(r'$\phi_e - \phi_{ref}$ (mV)')
    plt.xlabel(r'$x$-position ($\mu$m)')
    plt.title(r'Between cells')
    plt.xlim(35,85)
    plt.ylim([-0.345, 0.01])
    plt.yticks([-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0])
    plt.legend()

    # add colorbar
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(c1, ticks=cticks, label=clabel, cax=cax)

    # make pretty
    fig.subplots_adjust(wspace=0.1, hspace=0.0)
    plt.tight_layout()

    # add numbering for the subplots (A, B, C etc)
    letters = [r'\textbf{D}', r'\textbf{E}', r'\textbf{F}']
    ax3.text(-0.04, 1.02, letters[0], transform=ax.transAxes)
    for n, ax in enumerate([ax1, ax2]):
        ax.text(-0.12, 1.06, letters[n+1], transform=ax.transAxes)

    # save figure to file
    plt.savefig("results/figures/two_neurons_wide.svg", format='svg')
    return

def plot_axonbundle_potential(res):

    dt = 1.0e-4
    T = 5.0e-2

    time = 1.0e3*np.arange(0, T-dt, dt)

    # at membrane of axon A (gamma)
    x_M_A = 200; y_M_A = 1.1; z_M_A = 1.1
    # at membrane of axon B (gamma)
    x_M_B = 200; y_M_B = 0.9; z_M_B = 0.6
    # at membrane of axon C (gamma)
    x_M_C = 200; y_M_C = 0.6; z_M_C = 0.6

    # 0.05 um above axon A (ECS)
    x_e_A = 200; y_e_A = 1.1 + 0.05; z_e_A = 1.1

    #################################################
    # get data when axon A stimulated (KNP-EMI & EMI)
    fname = path + 'axonbundle_stimuli_A_knpemi/res_' + res + '/results.h5'
    fname_emi = path + 'axonbundle_stimuli_A_emi/res_' + res + '/results.h5'

    # get membrane potential at axon A
    phi_M_A = get_time_series_gamma(dt, T, fname, x_M_A, y_M_A, z_M_A)
    phi_M_A_emi = get_time_series_gamma(dt, T, fname_emi, x_M_A, y_M_A, z_M_A)

    # get membrane potential at axon B
    phi_M_B = get_time_series_gamma(dt, T, fname, x_M_B, y_M_B, z_M_B) 
    phi_M_B_emi = get_time_series_gamma(dt, T, fname_emi, x_M_B, y_M_B, z_M_B)

    # get ECS potential above axon A
    _, _, phi_e = get_time_series_ECS(dt, T, fname, x_e_A, y_e_A, z_e_A)
    _, _, phi_e_emi = get_time_series_ECS(dt, T, fname_emi, x_e_A, y_e_A, z_e_A, EMI=True)

    ###################################################
    # get data when axons BC stimulated (KNP-EMI & EMI)
    fname_sBC = path + 'axonbundle_stimuli_BC_knpemi/res_' + res + '/results.h5'
    fname_sBC_emi = path + 'axonbundle_stimuli_BC_emi/res_' + res + '/results.h5'

    # get membrane potential at axon A
    phi_M_A_sBC = get_time_series_gamma(dt, T, fname_sBC, x_M_A, y_M_A, z_M_A)
    phi_M_A_sBC_emi = get_time_series_gamma(dt, T, fname_sBC_emi, x_M_A, y_M_A, z_M_A)
    # get membrane potential at axon B
    phi_M_B_sBC = get_time_series_gamma(dt, T, fname_sBC, x_M_B, y_M_B, z_M_B)
    phi_M_B_sBC_emi = get_time_series_gamma(dt, T, fname_sBC_emi, x_M_B, y_M_B, z_M_B)
    # get membrane potential at axon C
    phi_M_B_sBC = get_time_series_gamma(dt, T, fname_sBC, x_M_B, y_M_B, z_M_B)
    phi_M_B_sBC_emi = get_time_series_gamma(dt, T, fname_sBC_emi, x_M_B, y_M_B, z_M_B)

    ###################################################
    # get data when axon(s) A or BC stimulated (EMI)
    fname_sA_h_sigma = path + 'axonbundle_stimuli_A_emi_low_sigma/res_' + res + '/results.h5'
    fname_sBC_h_sigma = path + 'axonbundle_stimuli_BC_emi_low_sigma/res_' + res +'/results.h5'

    # get membrane potential at axon A and B when A stimulated
    phi_M_A_sA_h_sigma = get_time_series_gamma(dt, T, fname_sA_h_sigma, x_M_A, y_M_A, z_M_A)
    phi_M_B_sA_h_sigma = get_time_series_gamma(dt, T, fname_sA_h_sigma, x_M_B, y_M_B, z_M_B)
    # get membrane potential at axon A and B when BC stimulated 
    phi_M_A_sBC_h_sigma = get_time_series_gamma(dt, T, fname_sBC_h_sigma, x_M_A, y_M_A, z_M_A)
    phi_M_B_sBC_h_sigma = get_time_series_gamma(dt, T, fname_sBC_h_sigma, x_M_B, y_M_B, z_M_B)

    ###################################################
    # Plot potentials
    fig = plt.figure(figsize=(9*0.9,12*0.9))
    ax = plt.gca()

    # subplot 1: Membrane potential in axon A when axon A stimulated (KNP-EMI and EMI)
    ax1 = fig.add_subplot(3,2,1)
    plt.ylabel(r'$\phi_M$ (mV)')
    plt.plot(time, phi_M_A, linewidth=3, label='KNPEMI')
    plt.plot(time, phi_M_A_emi, '--', linewidth=3, label='EMI')
    plt.title(r'Membrane potential in axon A')
    plt.ylim(-95, 45)
    plt.yticks([-75, -50, -25, 0, 25])
    plt.xticks([0, 25, 50])
    plt.legend()

    # subplot 2: ECS potential above axon A when axon A stimulated (KNP-EMI and EMI)
    ax2 = fig.add_subplot(3,2,2)
    plt.ylabel(r'$\phi_e$ (mV)')
    plt.plot(time, phi_e, linewidth=3, label='KNPEMI')
    plt.plot(time, phi_e_emi, '--',  linewidth=3, label='EMI')
    plt.title(r'ECS potential above axon A')
    plt.ylim(59.5, 70.5)
    plt.yticks([61, 63, 65, 67, 69])
    plt.xticks([0, 25, 50])

    # subplot 3: Membrane potential in axon B when axon A stimulated (KNP-EMI and EMI)
    ax3 = fig.add_subplot(3,2,3)
    plt.ylabel(r'$\phi_M$ (mV)')
    plt.plot(time, phi_M_B, linewidth=3, label='KNPEMI')
    plt.plot(time, phi_M_B_emi, '--', linewidth=3, label='EMI')
    plt.ylim(-74.0, -58.0)
    plt.yticks([-72, -69, -66, -63, -60])
    plt.xticks([0, 25, 50])
    plt.title(r'1 active neighbour')

    # subplot 4: Membrane potential in axon A when axons B and C stimulated (KNP-EMI and EMI)
    ax4 = fig.add_subplot(3,2,4)
    plt.ylabel(r'$\phi_M$ (mV)')
    plt.plot(time, phi_M_A_sBC, linewidth=3, label='KNPEMI')
    plt.plot(time, phi_M_A_sBC_emi, '--', linewidth=3, label='EMI')
    plt.ylim(-74.0, -58.0)
    plt.yticks([-72, -69, -66, -63, -60])
    plt.xticks([0, 25, 50])
    plt.title(r'8 active neighbours')

    # subplot 3: Membrane potential in axon A when axon A and BC stimulated (EMI)
    ax5 = fig.add_subplot(3,2,5)
    plt.ylabel(r'$\phi_M$ (mV)')
    plt.title(r'1 active neighbour (EMI)')
    plt.plot(time, phi_M_A_sA_h_sigma, '--', linewidth=3, label='stimulated')
    plt.plot(time, phi_M_B_sA_h_sigma, '--', linewidth=3, label='non-stimulated')
    plt.xlabel(r'time (ms)')
    plt.ylim(-95, 45)
    plt.yticks([-75, -50, -25, 0, 25])
    plt.xticks([0, 25, 50])
    plt.legend()

    # subplot 3: Membrane potential in axon B when axon A and BC stimulated (EMI)
    ax6 = fig.add_subplot(3,2,6)
    plt.ylabel(r'$\phi_M$ (mV)')
    plt.title(r'8 active neighbours (EMI)')
    plt.plot(time, phi_M_B_sBC_h_sigma, '--', linewidth=3, label='stimulated')
    plt.plot(time, phi_M_A_sBC_h_sigma, '--', linewidth=3, label='non-stimulated')
    plt.ylim(-95, 45)
    plt.yticks([-75, -50, -25, 0, 25])
    plt.xticks([0, 25, 50])
    plt.xlabel(r'time (ms)')

    # make plot pretty
    ax.axis('off')
    plt.tight_layout()

    # mark each subplot (A, B, C, etc)
    letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', 
               r'\textbf{D}', r'\textbf{E}', r'\textbf{F}']
    for n, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        ax.text(-0.09, 1.035, letters[n], transform=ax.transAxes)

    # save figure to file
    plt.savefig('results/figures/axon_bundle_potential.svg', format='svg')
    return

def plot_axonbundle_concentration(res):

    dt = 1.0e-4
    T = 5.0e-2

    temperature = 300 # temperature (K)
    F = 96485         # Faraday's constant (C/mol)
    R = 8.314         # Gas constant (J/(K*mol))

    time = 1.0e3*np.arange(0, T-dt, dt)

    # at membrane of axon A (gamma)
    x_M_A = 200; y_M_A = 1.1; z_M_A = 1.1
    # 0.05 um above axon A (ECS)
    x_e_A = 200; y_e_A = 1.1 + 0.05; z_e_A = 1.1
    # mid point inside axon A (ICS)
    x_i_A = 200; y_i_A = 1.0; z_i_A = 1.0

    #################################################################
    # get data axon A is stimulated
    fname_A = path + 'axonbundle_stimuli_A_knpemi/res_' + res + '/results.h5'

    # bulk concentrations
    Na_e_A, K_e_A, _ = get_time_series_ECS(dt, T, fname_A, x_e_A, y_e_A, z_e_A)
    Na_i_A, K_i_A = get_time_series_ICS(dt, T, fname_A, x_i_A, y_i_A, z_i_A)

    # trace concentrations
    Na_e_A_tr, K_e_A_tr, _ = get_time_series_ECS(dt, T, fname_A, x_M_A, y_M_A, z_M_A)
    Na_i_A_tr, K_i_A_tr = get_time_series_ICS(dt, T, fname_A, x_M_A, y_M_A, z_M_A)

    #################################################################
    # get data axons BC are stimulated
    fname_BC = path + 'axonbundle_stimuli_BC_knpemi/res_' + res + '/results.h5'

    # bulk concentrations
    Na_e_BC, K_e_BC, _ = get_time_series_ECS(dt, T, fname_BC, x_e_A, y_e_A, z_e_A)
    Na_i_BC, K_i_BC = get_time_series_ICS(dt, T, fname_BC, x_i_A, y_i_A, z_i_A)

    # trace concentrations
    Na_e_BC_tr, K_e_BC_tr, _ = get_time_series_ECS(dt, T, fname_BC, x_M_A, y_M_A, z_M_A)
    Na_i_BC_tr, K_i_BC_tr = get_time_series_ICS(dt, T, fname_BC, x_M_A, y_M_A, z_M_A)

    # calculate Nernst potentials
    E_Na_A = []
    E_K_A = []
    E_Na_BC = []
    E_K_BC = []

    for i in range(int(T/dt-1)):
        # Nernst potentials in axon A
        E_Na_A.append(1.0e3*R*temperature/F*np.log(Na_e_A_tr[i]/Na_i_A_tr[i]))
        E_K_A.append(1.0e3*R*temperature/F*np.log(K_e_A_tr[i]/K_i_A_tr[i]))

        # Nernst potentials in axons B/C
        E_Na_BC.append(1.0e3*R*temperature/F*np.log(Na_e_BC_tr[i]/Na_i_BC_tr[i]))
        E_K_BC.append(1.0e3*R*temperature/F*np.log(K_e_BC_tr[i]/K_i_BC_tr[i]))

    # Concentration plots
    fig = plt.figure(figsize=(9*0.9,12*0.9))
    ax = plt.gca()

    ax1 = fig.add_subplot(3,2,1)
    plt.title(r'Na$^+$ concentration (ECS)')
    plt.ylabel(r'[Na]$_e$ (mM)')
    plt.plot(time, Na_e_A, label='1 active axon', linewidth=3)
    plt.plot(time, Na_e_BC, label='8 active axons', linewidth=3)
    plt.xticks([0, 25, 50])
    plt.ylim(97.8, 100.2)
    plt.yticks([98, 98.5, 99, 99.5, 100.0])
    plt.legend()

    ax3 = fig.add_subplot(3,2,2)
    plt.title(r'K$^+$ concentration (ECS)')
    plt.ylabel(r'[K]$_e$ (mM)')
    plt.plot(time, K_e_A, label='1 active axon', linewidth=3)
    plt.plot(time, K_e_BC, label='8 active axons', linewidth=3)
    plt.xticks([0, 25, 50])
    plt.ylim(3.8, 6.2)
    plt.yticks([4.0, 4.5, 5.0, 5.5, 6.0])

    ax2 = fig.add_subplot(3,2,3)
    plt.title(r'Na$^+$ concentration (ICS)')
    plt.ylabel(r'[Na]$_i$ (mM)')
    plt.plot(time, Na_i_A, label='1 active axon', linewidth=3)
    plt.plot(time, Na_i_BC, label='8 active axons', linewidth=3)
    plt.xticks([0, 25, 50])
    plt.ylim(11.5, 22.5)
    plt.yticks([13, 15, 17, 19, 21])

    ax4 = fig.add_subplot(3,2,4)
    plt.title(r'K$^+$ concentration (ICS)')
    plt.ylabel(r'[K]$_i$ (mM)')
    plt.plot(time, K_i_A, label='1 active axon', linewidth=3)
    plt.plot(time, K_i_BC, label='8 active axon', linewidth=3)
    plt.xticks([0, 25, 50])
    plt.ylim(114, 126)
    plt.yticks([116, 118, 120, 122, 124])

    ax5 = fig.add_subplot(3,2,5)
    plt.title(r'Na$^+$ reversal potential')
    plt.ylabel(r'E$_{Na}$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(time, E_Na_A, label='1 active axon', linewidth=3)
    plt.plot(time, E_Na_BC, label='8 active axons', linewidth=3)
    plt.xticks([0, 25, 50])
    plt.ylim(37, 57)
    plt.yticks([39, 43, 47, 51, 55])

    ax6 = fig.add_subplot(3,2,6)
    plt.title(r'K$^+$ reversal potential')
    plt.ylabel(r'E$_K$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(time, E_K_A, label='1 active axon', linewidth=3)
    plt.plot(time, E_K_BC, label='8 active axons', linewidth=3)
    plt.xticks([0, 25, 50])
    plt.ylim(-90, -76)
    plt.yticks([-89, -86, -83, -80, -77])

    # make pretty
    ax.axis('off')
    plt.tight_layout()

    # mark each subplot (A, B, C, etc)
    letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', 
               r'\textbf{D}', r'\textbf{E}', r'\textbf{F}']
    for n, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        ax.text(-0.09, 1.035, letters[n], transform=ax.transAxes)

    # save figure to file
    plt.savefig('results/figures/axon_bundle_concentration.svg', format='svg')
    return

# create directory for figures
if not os.path.isdir('results/figures'):
    os.mkdir('results/figures')

# create figures
sys.stdout.write("\nGenerating figures\n")
#plot_refinement_test_potential()
#plot_refinement_concentration()
res_2D = '1' # mesh resolution for 2D axons
plot_one_neuron(res_2D)
plot_two_neurons(res_2D)
plot_two_neurons_wide(res_2D)
#res_3D = '0' # mesh resolution for 3D axon bundle
#plot_axonbundle_potential(res_3D)
#plot_axonbundle_concentration(res_3D)
