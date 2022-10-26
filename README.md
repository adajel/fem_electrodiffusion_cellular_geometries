# Supplementary material (code) for Finite element simulation of ionic electrodiffusion in cellular geometries
_________________

[![GPL-3.0](https://img.shields.io/github/license/adajel/fem_electrodiffusion_cellular_geometries)](LICENSE)
[Read Latest Documentation](https://adajel.github.io/fem_electrodiffusion_cellular_geometries/)
_________________

------------------------------------------------------------------------------
Project description
------------------------------------------------------------------------------
This directory contains an implementation of a mortar element FEM method for
the solving the KNP-EMI model and the EMI model, and code for reproducing
results from Ellingsrud, Ada J., et al. "Finite element simulation of ionic
electrodiffusion in cellular geometries." Frontiers in Neuroinformatics 14
(2020): 11.

------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------
To get the environment needed (all dependencies etc.) to run the code, download
the docker container by running:


```bash
docker pull ghcr.io/adajel/fem_electrodiffusion_cellular_geometries:v0.1.1
docker run --rm -v $PWD:/home/shared -w /home/shared  -it ghcr.io/adajel/fem_electrodiffusion_cellular_geometries:v0.1.1
```

------------------------------------------------------------------------------
Running the code
------------------------------------------------------------------------------
To run all the numerical experiments, execute:

```python
$ python3 main.py
```

All meshes are generates automatically in the code where they are used, if they
do not already exist. Each numerical experiments can be run by the run_*.py
files as follows:

```python

# run method of manufactured solutions (MMS) test on unit square
$ python3 run_MMM_test.py

# run convergence test using refined 2D meshes with one neuron
$ run_refinement_test.py:

# run simulations to compare the EMI and the KNP-EMI model on mesh with one
# and two intracellular compartment
$ run_2D_axons.py:

# run physiological simulation to explore ephaptic coupling in idealized
# 3D neuron bundle mesh
$ run_3D_axonbundle.py:
```

------------------------------------------------------------------------------
Generate figures
------------------------------------------------------------------------------
To generate the figures (.svg format) presented in the paper, execute:

```python
$ python3 make_figures.py
```
.. and to convert the figures to .pdf and .pdf_latex format execute:

```bash
$ convert.sh
```
