"""
This script generates a 2-D mesh consisting of a box with an embedded neuron
(a smaller box). The dimensions are modified from
https://www.frontiersin.org/articles/10.3389/fncom.2017.00027/full
to make a coarser mesh. The outer block as lengths
Lx = 120 um
Ly = 120 um
while the inner block has lengths
lx = 50 um
ly = 6 um
"""

from dolfin import *
import sys

class Boundary(SubDomain):
    # define exterior boundary
    def inside(self, x, on_boundary):
        return on_boundary

def add_rectangle(mesh, subdomains, surfaces, a, b):
    # define interior domain
    in_interior = """ (x[0] >= %g && x[0] <= %g &&
    x[1] >= %g && x[1] <= %g)""" \
    % (a[0], b[0], a[1], b[1])

    interior = CompiledSubDomain(in_interior)

    # mark interior and exterior domain
    for cell in cells(mesh):
        x = cell.midpoint().array()
        subdomains[cell] += int(interior.inside(x, False))
    assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        side_1 = (near(x[0], a[0]) and a[1] <= x[1] <= b[1])
        side_2 = (near(x[0], b[0]) and a[1] <= x[1] <= b[1])
        side_3 = (near(x[1], a[1]) and a[0] <= x[0] <= b[0])
        side_4 = (near(x[1], b[1]) and a[0] <= x[0] <= b[0])
        surfaces[facet] += side_1 or side_2 or side_3 or side_4

# if no input argument, set resolution factor to default = 0
if len(sys.argv) == 1:
    resolution_factor = 0
else:
    resolution_factor = int(sys.argv[1])

nx = 120*2**resolution_factor
ny = 120*2**resolution_factor

# box mesh
mesh = RectangleMesh(Point(0, 0), Point(120, 120), nx, ny)
subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)

a = Point(35, 49)   # bottom left of interior domain
b = Point(85, 55)   # top right of interior domain
add_rectangle(mesh, subdomains, surfaces, a, b)

a = Point(35, 65)   # bottom left of interior domain
b = Point(85, 71)   # top right of interior domain
add_rectangle(mesh, subdomains, surfaces, a, b)

# mark exterior boundary
Boundary().mark(surfaces, 2)

# convert mesh to unit meter (m)
mesh.coordinates()[:,:] *= 1e-6

# path to directory where mesh files are saved
dir_path = 'meshes/two_neurons_wide_2d/'

# save .xml files
meshfile = File(dir_path + 'mesh_' + str(resolution_factor) + '.xml')
meshfile << mesh

subdomains_file = File(dir_path + 'subdomains_' + str(resolution_factor) + '.xml')
subdomains_file << subdomains

surfaces_file = File(dir_path + 'surfaces_' + str(resolution_factor) + '.xml')
surfaces_file << surfaces

# save .pvd files
subdomainsplot = File(dir_path + 'subdomains_' + str(resolution_factor) + '.pvd')
subdomainsplot << subdomains

interfaceplot = File(dir_path + 'interface_' + str(resolution_factor) + '.pvd')
interfaceplot << surfaces
