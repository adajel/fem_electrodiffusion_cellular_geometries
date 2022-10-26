#!/usr/bin/python3

import os

# refinement test KNP-EMI (2D)
os.system('python3 run_refinement_test.py')
# MMS test (2D)
os.system('python3 run_MMS_test.py')
# comparison KNP-EMI and EMI on one neuron (2D) and two neurons (2D)
os.system('python3 run_2D_axons.py')
# KNP-EMI and EMI on axon bundle (3D)
os.system('python3 run_3D_axonbundle.py')
