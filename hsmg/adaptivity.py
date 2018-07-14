#!/usr/bin/env python

# Copyright (C) 2016 Imperial College London and others.
#
# Please see the AUTHORS file in the main source directory for a
# full list of copyright holders.
#
# Gerard Gorman
# Applied Modelling and Computation Group
# Department of Earth Science and Engineering
# Imperial College London
#
# g.gorman@imperial.ac.uk
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# Many thanks to:
# Patrick Farrell    for mesh based metric used for coarsening.
# James Maddinson    for the original version of Dolfin interface.
# Davide Longoni     for p-norm function.
# Kristian E. Jensen for ellipse function, test cases, vectorization opt., 3D glue
# Matteo Croci       for updating/rewriting/optimising the interface

"""@package PRAgMaTIc
The python interface to PRAgMaTIc (Parallel anisotRopic Adaptive Mesh
ToolkIt) provides anisotropic mesh adaptivity for meshes of
simplexes. The target applications are finite element and finite
volume methods although the it can also be used as a lossy compression
algorithm for data (e.g. image compression). It takes as its input the
mesh and a metric tensor field which encodes desired mesh element size
anisotropically.
"""

import ctypes, ctypes.util, numpy, scipy.sparse, scipy.sparse.linalg, collections
from numpy import array, zeros, ones, any, arange, isnan
from numpy.linalg import eigh as pyeig
from itertools import combinations
from dolfin import *

__all__ = ["_libpragmatic",
           "InvalidArgumentException",
           "LibraryException",
           "NotImplementedException",
           "ParameterException",
           "adapt",
           "adapt_boundary_regions",
           "detect_colinearity",
           "edge_lengths",
           "mesh_metric",
           "refine_metric",
           "metric_pnorm"]

class InvalidArgumentException(TypeError):
  pass
class LibraryException(SystemError):
  pass
class NotImplementedException(Exception):
  pass
class ParameterException(Exception):
  pass

try:
  path = "/usr/local/lib/libpragmatic.so"
  _libpragmatic = ctypes.cdll.LoadLibrary(path)
except:
  raise LibraryException("Failed to load libpragmatic in %s" % path)

def refine_metric(M, factor):
  M2 = M.copy(deepcopy=True)
  M2.assign(factor**2*M)
  name = "mesh_metric_refined_x%.6g" % factor
  M2.rename(name, name)

  return M2

def edge_lengths(M):
  class EdgeLengthExpression(Expression):
    def eval(self, value, x):
      mat = M(x)
      mat.shape = (2, 2)
      evals, evecs = numpy.linalg.eig(mat)
      value[:] = 1.0 / numpy.sqrt(evals)
      return
    def value_shape(self):
      return (2,)
  e = interpolate(EdgeLengthExpression(), VectorFunctionSpace(M.function_space().mesh(), "CG", 1))
  name = "%s_edge_lengths" % M.name()
  e.rename(name, name)

  return e

def set_mesh(n_xy, n_enlist, mesh=None, dx=None, debugon=False):
  #this function generates a mesh 2D or 3D DOLFIN mesh given coordinates (nxy) and cells(n_enlist).
  #it is used in the adaptation, but can also be used in the context of debugging, i.e. if one
  #one saves the mesh coordinates and cells using pickle between iterations.
  #INPUT : n_xy.shape = (2,N) or n_xy.shape = (3,N) for 2D and 3D, respectively.
  #INPUT : n_enlist.shape = (3*M,) or n_enlist.shape = (4*M,)
  #INPUT : mesh is the oldmesh used for checking area/volume conservation
  #INPUT : dx, operator !? for arae/volume conservation check
  #INPUT : debugon flag for checking area/volume preservation,
  #       should be off for curved geometries.
  #OUTOUT: DOLFIN MESH
  startTime = time()
  nvtx = n_xy.shape[1]
  n_mesh = Mesh()
  ed = MeshEditor()
  dim = len(n_xy)
  ed.open(n_mesh, str({2: triangle}[dim]), dim, dim)
  ed.init_vertices(nvtx) #n_NNodes.value
  if len(n_xy) == 1:
   for i in range(nvtx):
    ed.add_vertex(i, n_xy[0,i])
   ed.init_cells(int(len(n_enlist)/2))
   for i in range(int(len(n_enlist)/2)): #n_NElements.value
    ed.add_cell(i, n_enlist[i * 2], n_enlist[i * 2 + 1])
  elif len(n_xy) == 2:
   for i in range(nvtx): #n_NNodes.value
     ed.add_vertex(i, n_xy[0,i], n_xy[1,i])
   ed.init_cells(int(len(n_enlist)/3)) #n_NElements.value
   for i in range(int(len(n_enlist)/3)): #n_NElements.value
     ed.add_cell(i, n_enlist[i * 3], n_enlist[i * 3 + 1], n_enlist[i * 3 + 2])
  else: #3D
   for i in range(nvtx): #n_NNodes.value
     ed.add_vertex(i, n_xy[0,i], n_xy[1,i], n_xy[2,i])
   ed.init_cells(int(len(n_enlist)/4)) #n_NElements.value
   for i in range(int(len(n_enlist)/4)): #n_NElements.value
     ed.add_cell(i, n_enlist[i * 4], n_enlist[i * 4 + 1], n_enlist[i * 4 + 2], n_enlist[i * 4 + 3])
  ed.close()
  info("mesh definition took %0.1fs (not vectorized)" % (time()-startTime))
  if debugon==True and dx is not None:
    # Sanity check to be deleted or made optional
    area = assemble(interpolate(Constant(1.0),FunctionSpace(mesh,'DG',0)) * dx)
    n_area = assemble(interpolate(Constant(1.0),FunctionSpace(n_mesh,'DG',0)) * dx)
    err = abs(area - n_area)
    info("Donor mesh area : %.17e" % area)
    info("Target mesh area: %.17e" % n_area)
    info("Change          : %.17e" % err)
    info("Relative change : %.17e" % (err / area))

    #assert(err < 2.0e-11 * area)
  return n_mesh


def impose_maxN(metric, maxN):
    #enforces complexity constraint on the
    #INPUT : metric, DOLFIN SPD TENSOR VARIABLE
    #INPUT : maxN upper complexity (~ number of nodes) constraint
    #OUTPUT: metric, DOLFIN SPD TENSOR VARIABLE
    #OUTPUT: fak, factor with which the metric was coarsened - usefull for
    #throwing warnings
    gdim = metric.function_space().ufl_element().cell().geometric_dimension()
    targetN = assemble(sqrt(det(metric))*dx)
    fak = 1.
    if targetN > maxN:
      fak = (targetN/maxN)**(gdim/2)
      metric.vector().set_local(metric.vector().get_local()/fak)
      info('metric coarsened to meet target node number')
    return [metric,fak]

def patchwise_projection(dg_metric, mesh):
    # Patchwise projection of a DG 0 metric to CG 1.
    # This is done by setting each vertex value to be the
    # average of the values of the DG metric over the cells
    # in the vertex patch. The average is weighted by the patch
    # cell volume.
    # Implementation in C++ for speed

    code = r'''
    #include <dolfin/mesh/Vertex.h>
    #include <dolfin/fem/fem_utils.h>
    namespace dolfin
    {
      void project(const Function M, GenericVector& u, const Mesh mesh, const FunctionSpace space)
      {
        const std::size_t dim = mesh.geometry().dim();
        const std::vector<int> vertex_map = vertex_to_dof_map(space);
        double init_vals[dim*dim];
        memset( init_vals, 0, dim*dim*sizeof(double) );
        Array<double> vals(dim*dim, init_vals);
        double denom = 0.0;
        double volume = 0.0;
        unsigned count = 0;
        unsigned len = u.local_size();
        std::vector<double> values(len,0.0);
        for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
        {
            for (CellIterator patch_cell(*vertex); !patch_cell.end(); ++patch_cell)
            {
                Array<double> p(dim,patch_cell->midpoint().coordinates());
                M.eval(vals,p);
                volume = patch_cell->volume();
                for (unsigned i=0;i<dim*dim;i++)
                {
                values[vertex_map[count + i]] += vals[i]*volume;
                }
                denom += volume;
            }
            for (unsigned i=0;i<dim*dim;i++)
            {
            values[vertex_map[count + i]] = values[vertex_map[count + i]]/denom;
            }
            denom = 0.0;
            count += dim*dim;
        }
        u.set_local(values);
      }
    }'''

    # compile C++ code
    project_jit =  compile_extension_module(code=code, cppargs=["-fpermissive"]).project

    # define new metric
    CG = TensorFunctionSpace(mesh, 'CG', 1)
    cg_metric = Function(CG)

    # projection step
    project_jit(dg_metric, cg_metric.vector(), mesh, CG)

    return cg_metric

def detect_colinearity(mesh, tol = 30):
    # detects colinear boundary facets for boundary coarsening/refinement.
    # mesh is the mesh to be adapted, tol is the angle tolerance in degrees,
    # i.e. if two facets have an angle between them less than tol, they are
    # considered as colinear.
    # This function assigns each colinear group of boundary facets a different
    # colour. The colouring is stored in a FacetFunction which is returned as an output.
    #
    # The colouring is performed by assigning a new colour to an uncoloured boundary facet
    # and by checking if the angle between the facet normal and the normals of
    # the facet neighbours is below the prescribed tolerance. This operation is repeated
    # for each of the neighbour facets until no colinear neighbours are detected.
    # A new colour is then assigned to the next uncoloured facet and the operation
    # is repeated until all facets are coloured.

    def angle_between(v1, v2):
        """ Returns the angle in radians between unit vectors 'v1' and 'v2':
                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        return numpy.arccos(numpy.clip(numpy.dot(v1, v2), -1.0, 1.0))

    # convert from degrees to radiants
    tol = tol*numpy.pi/180.

    dim = mesh.topology().dim()

    mesh.init(dim-1)

    # compute boundary mesh and initialise connectivity
    bmesh = BoundaryMesh(mesh, 'exterior')
    bmesh.init(dim-2, dim-1)
    bmesh.init(dim-1, dim-2)

    # compute connectivity from facets to edges/vertices
    # and vice versa
    bmt = bmesh.topology()
    onetwo = bmt(dim-2,dim-1)
    twoone = bmt(dim-1,dim-2)

    normalmap = collections.defaultdict()

    # compute the index of the adjacent facets
    def facetmap(facet_ind):
        return [elem[elem!=facet_ind][0] for elem in [onetwo(item) for item in twoone(facet_ind)]]

    # compute all the normals and store them as a numpy array
    for f in cells(bmesh):
        n = Facet(mesh, bmesh.entity_map(dim-1)[f.index()]).normal()
        normalmap[f.index()] = numpy.array([n.x(), n.y(), n.z()])

    # initialise a boolean array to mark visited cells and the colormap array
    notvisited = numpy.ones((bmesh.num_cells(),), dtype=bool)
    colormap = numpy.zeros((bmesh.num_cells(),), dtype = 'int32')

    # first colour to use is 1, then update it when a new colour is needed.
    next_col = 1
    while True:
        # get the next uncoloured facet, if there are none, break.
        try: origin = numpy.argwhere(notvisited)[0][0]
        except IndexError: break
        # assign the colour
        colormap[origin] = next_col
        # find the group of the adjacent colinear facets which have not been visited yet
        nextgroup = [item for item in facetmap(origin) if notvisited[item]]
        nextgroup = [item for item in nextgroup if angle_between(normalmap[origin], normalmap[item]) < tol]
        notvisited[origin] = False
        # keep going through the adjacent facets until no more colinear facets are found
        while len(nextgroup) > 0:
            temp = []
            for nxt in nextgroup:
                # find the group of the adjacent colinear facets which have not been visited yet
                temp2 = [item for item in facetmap(nxt) if notvisited[item] and item not in temp]
                temp += [item for item in temp2 if angle_between(normalmap[origin], normalmap[item]) < tol]
                notvisited[nxt] = False
                colormap[nxt] = next_col
            nextgroup = [item for item in temp if notvisited[item]]

        # update next colour
        next_col += 1

    # define FacetFunction on the original mesh and map the boundary mesh colormap
    # to the full mesh FacetFunction
    bfaces_func = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    bfaces_func.set_all(0)
    for facet in cells(bmesh):
        bfaces_func[bmesh.entity_map(dim-1)[facet.index()]] = colormap[facet.index()]

    return bfaces_func

def adapt_boundary_regions(original_mesh, adapted_mesh, original_regions):
    # boundary region IDs are currently `lost' (not computed)
    # when a mesh is adapted by using pragmatic_adapt.
    # This routine takes the orginal mesh and boundary IDs before adaptation and the
    # output mesh adapted by pragmatic_adapt and computes consistent boundary region IDs
    # for the adapted mesh.
    #
    # INPUT: original_mesh,    a dolfin mesh object containing the mesh before adaptation
    #        adapted_mesh,     a dolfin mesh object containing the adapted mesh (output of adapt)
    #        original_regions, a dolfin FacetFunction containing the boundary region IDs of
    #                          the original_mesh before adaptation.
    #
    # OUTPUT: adapted_regions, a dolfin FacetFunction containing the adapted boundary region IDs
    #                          of the adapted mesh.
    #
    # We loop over all the adapted_mesh boundary facets and compute each facet midpoint.
    # For each midpoint, we find the original_mesh boundary facet that includes (or is closest)
    # to the midpoint and we assign the same region ID as the original_mesh facet to the adapted_mesh facet.
    # NOTE: this function is deprecated and it will be removed soon.

    # compute geometric dimension
    dim = original_mesh.geometry().dim()

    # extract boundary meshes for efficient looping over the facets
    original_bmesh = BoundaryMesh(original_mesh, 'exterior')
    adapted_bmesh = BoundaryMesh(adapted_mesh, 'exterior')

    # use the original_regions FacetFunction to compute the boundary region IDs over the corresponding boundary mesh
    original_boundary_regions = numpy.zeros((original_bmesh.num_cells(),), dtype='int32')
    for facet in cells(original_bmesh):
        original_boundary_regions[facet.index()] = original_regions[original_bmesh.entity_map(dim-1)[facet.index()]]

    # initialise adapted_bmesh FacetFunction values
    adapted_boundary_regions = numpy.zeros((adapted_bmesh.num_cells(),), dtype='int32')

    # initialise bounding box tree for point search queries
    tree = original_bmesh.bounding_box_tree()
    # loop over the facets of the adapted mesh
    for cell_facet in cells(adapted_bmesh):
        # compute facet midpoint
        p = cell_facet.midpoint()
        # find in which original mesh facet the midpoint belongs to
        original_facet_ID = tree.compute_first_entity_collision(p)
        # NOTE: sometimes compute_first_entity_collision fails, we add a call to
        # compute_closest_entity for robustness.
        if original_facet_ID > original_bmesh.num_cells() or original_facet_ID < 0:
            original_facet_ID = tree.compute_closest_entity(p)[0]

        # compute adapted mesh facet index
        adapted_facet_ID = cell_facet.index()
        # assign boundary region ID to the corresponding adapted mesh facet
        adapted_boundary_regions[adapted_facet_ID] = original_boundary_regions[original_facet_ID]

    # create an adapted mesh FacetFunction and assign the relative region ID values from adapted_boundary_regions
    adapted_regions = MeshFunction('size_t', adapted_mesh, adapted_mesh.topology().dim()-1, 0)
    adapted_regions.set_all(0)
    for facet in cells(adapted_bmesh):
        adapted_regions[adapted_bmesh.entity_map(dim-1)[facet.index()]] = adapted_boundary_regions[facet.index()]

    return adapted_regions

def adapt(metric, bfaces=None, bfaces_func=None, colinearity_tol = 30, debugon=True, eta=1e-2, grada=None, maxN=None, coarsen=False):
  #this is the actual adapt function. It currently works with vertex
  #numbers rather than DOF numbers, but as of DOLFIN.__VERSION__ >= "1.3.0",
  #there is no difference.
  # INPUT : metric is a DG0 or CG1 SPD DOLFIN TENSOR VARIABLE or
  #         a DOLFIN SCALAR VARIABLE. In the latter case metric_pnorm is called
  #         to calculate a DOLFIN CG1 TENSOR VARIABLE
  # INPUT : bfaces.shape = (n,2) or bfaces.shape = (n,3) is a list of edges or
  #        faces for the mesh boundary. If not specified, it will be calculated.
  # INPUT : bfaces_func is a FacetFunction that gives each edge and face and ID, so
  #         that corners are implicitly specified in 2D and edges in 3D.
  #         All corners can be specified this way in 3D, but it can require
  #         definition of dummy IDs, if the corner is in the middle of face and
  #         thus only related to two IDs. The ID value 0 is reserved for the interior
  #         of the domain.
  # INPUT : colinearity_tol is the angle tolerance for colinearity detection
  #         (see the function detect_colinearity).
  # INPUT : debugon=True (default) checks for conservation of area/volume
  # INPUT : eta is the scaling factor used, if the metric input is a
  #         SCALAR DOLFIN FUNCTION
  # INPUT : grada enables gradation of the input metric, (1 for slight gradation,
  #         2 for more etc... off by default)
  # INPUT : maxN facilitates rescaling of the input metric to meet a
  #         mesh complexity constraint (~number of nodes). This can prevent
  #         OUT OF MEMORY ERRORS in the context of direct solvers, but it can
  #         also be headache for convergence analysis, which is why a warning
  #         is thrown if the constraint is active
  # OUTPUT: DOLFIN MESH
  mesh = metric.function_space().mesh()

  #check if input is not a metric
  if metric.function_space().ufl_element().num_sub_elements() == 0:
     metric = metric_pnorm(metric, eta=eta, CG1out=True)

  if metric.function_space().ufl_element().degree() == 0 and metric.function_space().ufl_element().family()[0] == 'D':
      # patchwise projection to CG 1 for speed
      metric = patchwise_projection(metric, mesh)
  if grada is not None:
      metric = gradate(metric,grada)
  if maxN is not None:
      [metric,fak] = impose_maxN(metric, maxN)

  # warn before generating huge mesh
  targetN = assemble(sqrt(det(metric))*dx)
  if targetN < 1e6:
    info("target mesh has %0.0f nodes" % targetN)
  else:
    warning("target mesh has %0.0f nodes" % targetN)

  space = metric.function_space() #FunctionSpace(mesh, "CG", 1)
  element = space.ufl_element()

  # Sanity checks
  if not (mesh.geometry().dim() == 2 or mesh.geometry().dim() == 3)\
        or not (element.cell().geometric_dimension() == 2 \
        or element.cell().geometric_dimension() == 3) \
        or not (element.cell().topological_dimension() == 2 \
        or element.cell().topological_dimension() == 3) \
        or not element.family() == "Lagrange" \
        or not element.degree() == 1:
    raise InvalidArgumentException("Require 2D P1 function space for metric tensor field")

  gdim = element.cell().geometric_dimension()

  nodes = array(range(0,mesh.num_vertices()),dtype=numpy.intc)
  cells_array = mesh.cells().astype(numpy.intc)
  coords = mesh.coordinates()
  # create boundary mesh and associated list of co-linear edges
  if bfaces is None:
      mesh.init(1,2)
      bfaces_list = []
      [bfaces_list.append(f) for f in facets(mesh) if f.exterior()]
      bfaces = numpy.array([tuple(f.entities(0)) for f in bfaces_list])

  if bfaces_func is None:
      # detect colinear facets
      info("Colinearity detection ...")
      bfaces_func = detect_colinearity(mesh, tol = colinearity_tol)
  else:
      for f in facets(mesh):
          if f.exterior():
              if bfaces_func[f] == 0:
                  raise ValueError("Zero value specified as a boundary tag in bfaces_func: the tag value 0 is reserved for interior facets.")

  bfaces_IDs = numpy.array([bfaces_func[f.index()] for f in bfaces_list], dtype = numpy.intc)

  x = coords[nodes,0]
  y = coords[nodes,1]
  if gdim == 3:
    z = coords[nodes,2]

  # Dolfin stores the tensor as:
  # |dxx dxy|
  # |dyx dyy|
    ## THE (CG1-)DOF NUMBERS ARE DIFFERENT FROM THE VERTEX NUMBERS (and we wish to work with the former)
  if dolfin.__version__ != '1.2.0':
      dof2vtx = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
  else:
      dof2vtx = FunctionSpace(mesh,'CG',1).dofmap().vertex_to_dof_map(mesh).argsort()

  metric_arr = numpy.empty(metric.vector().get_local().size, dtype = numpy.float64)
  if gdim == 2:
    metric_arr[range(0,metric.vector().get_local().size,4)] = metric.vector().get_local()[arange(0,metric.vector().get_local().size,4)[dof2vtx]]
    metric_arr[range(1,metric.vector().get_local().size,4)] = metric.vector().get_local()[arange(2,metric.vector().get_local().size,4)[dof2vtx]]
    metric_arr[range(2,metric.vector().get_local().size,4)] = metric.vector().get_local()[arange(2,metric.vector().get_local().size,4)[dof2vtx]]
    metric_arr[range(3,metric.vector().get_local().size,4)] = metric.vector().get_local()[arange(3,metric.vector().get_local().size,4)[dof2vtx]]
  else:
    metric_arr[range(0,metric.vector().get_local().size,9)] = metric.vector().get_local()[arange(0,metric.vector().get_local().size,9)[dof2vtx]]
    metric_arr[range(1,metric.vector().get_local().size,9)] = metric.vector().get_local()[arange(3,metric.vector().get_local().size,9)[dof2vtx]]
    metric_arr[range(2,metric.vector().get_local().size,9)] = metric.vector().get_local()[arange(6,metric.vector().get_local().size,9)[dof2vtx]]
    metric_arr[range(3,metric.vector().get_local().size,9)] = metric.vector().get_local()[arange(3,metric.vector().get_local().size,9)[dof2vtx]]
    metric_arr[range(4,metric.vector().get_local().size,9)] = metric.vector().get_local()[arange(4,metric.vector().get_local().size,9)[dof2vtx]]
    metric_arr[range(5,metric.vector().get_local().size,9)] = metric.vector().get_local()[arange(7,metric.vector().get_local().size,9)[dof2vtx]]
    metric_arr[range(6,metric.vector().get_local().size,9)] = metric.vector().get_local()[arange(6,metric.vector().get_local().size,9)[dof2vtx]]
    metric_arr[range(7,metric.vector().get_local().size,9)] = metric.vector().get_local()[arange(7,metric.vector().get_local().size,9)[dof2vtx]]
    metric_arr[range(8,metric.vector().get_local().size,9)] = metric.vector().get_local()[arange(8,metric.vector().get_local().size,9)[dof2vtx]]
  info("Beginning PRAgMaTIc adapt")
  info("Initialising PRAgMaTIc ...")
  NNodes = ctypes.c_int(x.shape[0])

  NElements = ctypes.c_int(cells_array.shape[0])

  if gdim == 2:
      _libpragmatic.pragmatic_2d_init(ctypes.byref(NNodes),
                                  ctypes.byref(NElements),
                                  cells_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                  x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
  else:
      _libpragmatic.pragmatic_3d_init(ctypes.byref(NNodes),
                                  ctypes.byref(NElements),
                                  cells_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                  x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
  info("Setting surface ...")
  nfacets = ctypes.c_int(len(bfaces))
  facets_array = array(bfaces.flatten(),dtype=numpy.intc)

  _libpragmatic.pragmatic_set_boundary(ctypes.byref(nfacets),
                                       facets_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                       bfaces_IDs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

  info("Setting metric tensor field ...")
  _libpragmatic.pragmatic_set_metric(metric_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

  info("Entering adapt ...")
  startTime = time()
  if coarsen:
      _libpragmatic.pragmatic_coarsen(ctypes.c_int(1))
  else:
      _libpragmatic.pragmatic_adapt(ctypes.c_int(1))

  info("adapt took %0.1fs" % (time()-startTime))
  n_NNodes = ctypes.c_int()
  n_NElements = ctypes.c_int()
  n_NSElements = ctypes.c_int()

  info("Querying output ...")
  _libpragmatic.pragmatic_get_info(ctypes.byref(n_NNodes),
                                   ctypes.byref(n_NElements),
                                   ctypes.byref(n_NSElements))

  n_enlist = numpy.empty((gdim+1) * n_NElements.value, numpy.intc)

  info("Extracting output ...")
  n_x = numpy.empty(n_NNodes.value)
  n_y = numpy.empty(n_NNodes.value)
  if gdim == 3:
      n_z = numpy.empty(n_NNodes.value)
      _libpragmatic.pragmatic_get_coords_3d(n_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                            n_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                            n_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
  else:
      _libpragmatic.pragmatic_get_coords_2d(n_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                        n_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

  _libpragmatic.pragmatic_get_elements(n_enlist.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

  if gdim == 2:
      n_mesh = set_mesh(array([n_x,n_y]),n_enlist,mesh=mesh,dx=dx,debugon=debugon)
  else:
      n_mesh = set_mesh(array([n_x,n_y,n_z]),n_enlist,mesh=mesh,dx=dx,debugon=debugon)

  # extracting boundary tags: pragmatic_get_boundaryTags returns an
  # array of the same size as n_enlist.
  # For each vertex, boundary_tags contains the tag of the opposite facet in the element.
  # boundary_tags[3*i+k] gives the tag of the facet opposite to the k-th vertex of the i-th triangle

  # extract boundary_tags from pragmatic
  ptr = ctypes.POINTER(ctypes.c_int*(gdim + 1)*n_NElements.value)()
  _libpragmatic.pragmatic_get_boundaryTags(ctypes.byref(ptr))
  boundary_tags = numpy.frombuffer(ptr.contents, dtype = numpy.intc)

  # define a FacetFunction with the boundary tags. We need to loop over all boundary cells
  # and find which boundary facet is opposite to which vertex.
  n_bfaces_func = MeshFunction("size_t", n_mesh, n_mesh.topology().dim()-1, 0)
  # FIXME: if the mesh is being coarsened, some boundary tags can get lost.
  # Keep track of lost boundary tags in wrong tags.
  wrong_tags = []
  for cell in cells(n_mesh):
      for facet in facets(cell):
          if facet.exterior():
              # if the cell facet is a boundary facet, get the cell vertex indices from n_enlist
              # and find the index of the vertex which is opposite to the boundary facet of the cell.
              # this vertex will be the only vertex which is not a vertex of the boundary facet.
              nodes_indices = n_enlist[(gdim+1)*cell.index():(gdim+1)*(cell.index()+1)].tolist()
              opposite_vertex = (set(nodes_indices) - set(facet.entities(0))).pop()
              # once we know the index of the vertex opposite to the boundary facet assign
              # the correct tag to the correct facet.
              pos = nodes_indices.index(opposite_vertex)
              n_bfaces_func[facet.index()] = boundary_tags[(gdim+1)*cell.index() + pos]
              if n_bfaces_func[facet.index()] == 0:
                  wrong_tags.append(facet.index())

  # FIXME: if the mesh is being coarsened, some boundary tags can get lost.
  #        the following is a temporary fix and it will be removed soon.
  while len(wrong_tags) > 0:
      still_wrong = []
      n_mesh.init(gdim-2,gdim-1) # initialise connectivity
      for wrong_tag in wrong_tags:
          # find tags of adjacent boundary facets. First get the vertices of the wrong tag facet, then the exterior facets
          # these vertices belong to.
          if gdim == 2:
              adjacent_tags = numpy.array([n_bfaces_func[ff] for ff in [Facet(n_mesh, index) for index in                         \
                              numpy.unique(numpy.concatenate([v.entities(gdim-1) for v in vertices(Facet(n_mesh, wrong_tag))]))]  \
                              if ff.exterior() and ff.index() != wrong_tag])
          else:
              adjacent_tags = numpy.array([n_bfaces_func[ff] for ff in [Facet(n_mesh, index) for index in                         \
                              numpy.unique(numpy.concatenate([e.entities(gdim-1) for e in    edges(Facet(n_mesh, wrong_tag))]))]  \
                              if ff.exterior() and ff.index() != wrong_tag])
          # replace the wrong tag with any of the adjacent correct tags.
          if (adjacent_tags != 0).max() == True:
              n_bfaces_func[wrong_tag] = adjacent_tags[adjacent_tags != 0][0]
          else:
              still_wrong.append(wrong_tag)

      wrong_tags = numpy.array(still_wrong)

  info("Finalising PRAgMaTIc ...")
  _libpragmatic.pragmatic_finalize()
  info("PRAgMaTIc adapt complete")

  return n_mesh, n_bfaces_func

def consistent_interpolation(mesh, fields=[]):
  if not isinstance(fields, list):
    return consistent_interpolation(mesh, [fields])

  n_space = FunctionSpace(n_mesh, "CG", 1)
  n_fields = []
  for field in fields:
    n_field = Function(n_space)
    n_field.rename(field.name(), field.name())
    val = numpy.empty(1)
    coord = numpy.empty(2)
    nx = interpolate(Expression("x[0]"), n_space).vector().get_local()
    ny = interpolate(Expression("x[1]"), n_space).vector().get_local()
    n_field_arr = numpy.empty(n_NNodes.value)
    for i in range(n_NNodes.value):
      coord[0] = nx[i]
      coord[1] = ny[i]
      field.eval(val, coord)
      n_field_arr[i] = val
    n_field.vector().set_local(n_field_arr)
    n_field.vector().apply("insert")
    n_fields.append(n_field)

  if len(n_fields) > 0:
    return n_fields
  else:
    return n_mesh

def fix_CG1_metric(Mp):
 #makes the eigenvalues of a metric positive (this property can be lost during
 #the projection step)
 #INPUT and OUTPUT: DOLFIN TENSOR VARIABLE
 [H,cell2dof] = get_dofs(Mp)
 [eigL,eigR] = analytic_eig(H)
# if any(lambda1<zeros(len(lambda2))) or any(lambda2<zeros(len(lambda2))):
#  warning('negative eigenvalue in metric fixed')
 eigL = numpy.abs(eigL)
 H = analyt_rot(fulleig(eigL),eigR)
 out = sym2asym(H).transpose().flatten()
 Mp.vector().set_local(out)
 return Mp

def metric_pnorm(f, eta, max_edge_length=None, min_edge_length=None, max_edge_ratio=10, p=2, CG1out=False, CG0H=3):
  # p-norm scaling to the metric, as in Chen, Sun and Xu, Mathematics of
  # Computation, Volume 76, Number 257, January 2007, pp. 179-204.
  # INPUT : f, SCALAR DOLFIN VARIABLE
  # INPUT : eta, scaling factor (0.04-0.005 are good values for engineering tolerances)
  # INPUT : max_edge_length is an optional lower bound on the metric eigenvalues
  # INPUT : min_edge_length is an optional upper bound on the metric eigenvalues
  # INPUT : max_edge_ratio is an optional local lower bound on the metric eigenvalues,
  #         which enforce a maximum ratio between the smaller and large eigenvalue
  # INPUT : p is the interpolation norm to be minimised (default to 2)
  # INPUT : CG1out enables projection of Hessian to CG1 space, such that this
   #        projection does not have to be performed at a later stage
  # INPUT : CG0H controls how a DG0 Hessian is extracted from a SCALAR DOLFIN CG2 VARIABLE
  # OUTPUT: DOLFIN (CG1 or DG0) SPD TENSOR VARIABLE
  mesh = f.function_space().mesh()
  # Sanity checks
  if max_edge_ratio is not None and max_edge_ratio < 1.0:
    raise InvalidArgumentException("The maximum edge ratio must be greater greater or equal to 1")
  else:
    if max_edge_ratio is not None:
     max_edge_ratio = max_edge_ratio**2 # ie we are going to be looking at eigenvalues

  n = mesh.geometry().dim()

  if f.function_space().ufl_element().degree() == 2 and f.function_space().ufl_element().family() == 'Lagrange':
     if CG0H == 0:
        S = VectorFunctionSpace(mesh,'DG',1) #False and
        A = assemble(inner(TrialFunction(S), TestFunction(S))*dx)
        b = assemble(inner(grad(f), TestFunction(S))*dx)
        ones_ = Function(S)
        ones_.vector()[:] = 1
        A_diag = A * ones_.vector()
        A_diag.set_local(1.0/A_diag.get_local())
        gradf = Function(S)
        gradf.vector()[:] = b * A_diag

        S = TensorFunctionSpace(mesh,'DG',0)
        A = assemble(inner(TrialFunction(S), TestFunction(S))*dx)
        b = assemble(inner(grad(gradf), TestFunction(S))*dx)
        ones_ = Function(S)
        ones_.vector()[:] = 1
        A_diag = A * ones_.vector()
        A_diag.set_local(1.0/A_diag.get_local())
     elif CG0H == 1:
        S = TensorFunctionSpace(mesh,'DG',0)
        A = assemble(inner(TrialFunction(S), TestFunction(S))*dx)
        b = assemble(inner(grad(grad(f)), TestFunction(S))*dx)
        ones_ = Function(S)
        ones_.vector()[:] = 1
        A_diag = A * ones_.vector()
        A_diag.set_local(1.0/A_diag.get_local())
        H = Function(S)
        H.vector()[:] = b * A_diag
     else:
        H = project(grad(grad(f)), TensorFunctionSpace(mesh, "DG", 0))
  else:
    gradf = project(grad(f), VectorFunctionSpace(mesh, "CG", 1))
    H = project(sym(grad(gradf)), TensorFunctionSpace(mesh, "DG", 0))

  if CG1out or dolfin.__version__ >= '1.4.0':
   H = project(H,TensorFunctionSpace(mesh,'CG',1))
  # EXTRACT HESSIAN
  [HH,cell2dof] = get_dofs(H)
  # add DOLFIN_EPS on the diagonal to avoid zero eigenvalues
  HH[0,:] += DOLFIN_EPS
  HH[2,:] += DOLFIN_EPS
  if n==3: #3D
   HH[5,:] += DOLFIN_EPS

  # CALCULATE EIGENVALUES
  [eigL,eigR] = analytic_eig(HH)

  # Make H positive definite and calculate the p-norm.
  #enforce hardcoded min and max contraints
  min_eigenvalue = 1e-20; max_eigenvalue = 1e20
  onesC = ones(eigL.shape)
  eigL = array([numpy.abs(eigL),onesC*min_eigenvalue]).max(0)
  eigL = array([numpy.abs(eigL),onesC*max_eigenvalue]).min(0)
  #enforce constraint on aspect ratio
  if max_edge_ratio is not None:
   RR = arange(HH.shape[1])
   CC = eigL.argmax(0)
   I_ = array([False]).repeat(array(eigL.shape).prod())
   I_[CC+(RR-1)*eigL.shape[0]] = True
   I_ = I_.reshape(eigL.shape)
   eigL[I_==False] = array([eigL[I_==False],eigL[I_].repeat(eigL.shape[0]-1)/max_edge_ratio]).max(0)

  #check (will not trigger with min_eigenvalue > 0)
  det = eigL.prod(0)
  if any(det==0):
    raise FloatingPointError("Eigenvalues are zero")

  #compute metric
  exponent = -1.0/(2*p + n)
  eigL *= 1./eta*(det**exponent).repeat(eigL.shape[0]).reshape([eigL.shape[1],eigL.shape[0]]).T

#  HH = analyt_rot(fulleig(eigL),eigR)
#  HH *= 1./eta*det**exponent
#  [eigL,eigR] = analytic_eig(HH)

  #enforce min and max contraints
  if max_edge_length is not None:
    min_eigenvalue = 1.0/max_edge_length**2
    if eigL.flatten().min()<min_eigenvalue:
     info('upper bound on element edge length is active')
  if min_edge_length is not None:
    max_eigenvalue = 1.0/min_edge_length**2
    if eigL.flatten().max()>max_eigenvalue:
     info('lower bound on element edge length is active')
  eigL = array([eigL,onesC*min_eigenvalue]).max(0)
  eigL = array([eigL,onesC*max_eigenvalue]).min(0)
  HH = analyt_rot(fulleig(eigL),eigR)

  Hfinal = sym2asym(HH)
  cbig=zeros((H.vector().get_local()).size)
  cbig[cell2dof.flatten()] = Hfinal.transpose().flatten()
  H.vector().set_local(cbig)
  return H

def metric_ellipse(H1, H2, method='in', qualtesting=False):
  # calculates the inner or outer ellipse (depending on the value of the method input)
  # of two the two input metrics.
  # INPUT : H1 is a DOLFIN SPD TENSOR VARIABLE (CG1 or DG0)
  # INPUT : H2 is a DOLFIN SPD TENSOR VARIABLE (CG1 or DG0)
  # INPUT : method determines calculation method, 'in' for inner ellipse (default)
  # INPUT : qualtesting flag that can be used to to trigger return of scalar a variable
  #         that indicates if one ellipse is entirely within the other (-1,1) or if they
  #         intersect (0)
  # OUTPUT: H1, DOLFIN SPD TENSOR VARIABLE (CG1 or DG0)
  [HH1,cell2dof] = get_dofs(H1)
  [HH2,cell2dof] = get_dofs(H2)
  cbig = zeros((H1.vector().get_local()).size)

  # CALCULATE EIGENVALUES using analytic expression numpy._version__>1.8.0 can do this more elegantly
  [eigL1,eigR1] = analytic_eig(HH1)
  # convert metric2 to metric1 space
  tmp = analyt_rot(HH2, transpose_eigR(eigR1))
  tmp = prod_eig(tmp, 1/eigL1)
  [eigL2,eigR2] = analytic_eig(tmp)
  # enforce inner or outer ellipse
  if method == 'in':
    if qualtesting:
     HH = Function(FunctionSpace(H1.function_space().mesh(),'DG',0))
     HH.vector().set_local((eigL2<ones(eigL2.shape)).sum(0)-ones(eigL2.shape[1]))
     return HH
    else:
     eigL2 = array([eigL2 ,ones(eigL2.shape)]).max(0)
  else:
    eigL2 = array([eigL2, ones(eigL2.shape)]).min(0)

  #convert metric2 back to original space
  tmp = analyt_rot(fulleig(eigL2), eigR2)
  tmp = prod_eig(tmp, eigL1)
  HH = analyt_rot(tmp,eigR1)
  HH = sym2asym(HH)
  #set metric
  cbig[cell2dof.flatten()] = HH.transpose().flatten()
  H1.vector().set_local(cbig)
  return H1

def get_dofs(H):
  #converts a DOLFIN SPD TENSOR VARIABLE to a numpy array, see sym2asym for storage convention
  #OUTPUT: argout.shape = (3,N) or argout.shape = (6,N) for 2D and 3D, respectively.
  #INPUT : DOLFIN TENSOR VARIABLE
  mesh = H.function_space().mesh()
  n = mesh.geometry().dim()
  if H.function_space().ufl_element().degree() == 0 and H.function_space().ufl_element().family()[0] == 'D':
      cell2dof = c_cell_dofs(mesh,H.function_space())
      cell2dof = cell2dof.reshape([mesh.num_cells(),n**2])
  else: #CG1 metric
      cell2dof = arange(mesh.num_vertices()*n**2)
      cell2dof = cell2dof.reshape([mesh.num_vertices(),n**2])
  if n == 2:
   H11 = H.vector().get_local()[cell2dof[:,0]]
   H12 = H.vector().get_local()[cell2dof[:,1]] #;H21 = H.vector().get_local()[cell2dof[:,2]]
   H22 = H.vector().get_local()[cell2dof[:,3]]
   return [array([H11,H12,H22]),cell2dof]
  else: #n==3
   H11 = H.vector().get_local()[cell2dof[:,0]]
   H12 = H.vector().get_local()[cell2dof[:,1]] #;H21 = H.vector().get_local()[cell2dof[:,3]]
   H13 = H.vector().get_local()[cell2dof[:,2]] #;H31 = H.vector().get_local()[cell2dof[:,6]]
   H22 = H.vector().get_local()[cell2dof[:,4]]
   H23 = H.vector().get_local()[cell2dof[:,5]] #H32 = H.vector().get_local()[cell2dof[:,7]]
   H33 = H.vector().get_local()[cell2dof[:,8]]
   return [array([H11,H12,H22,H13,H23,H33]),cell2dof]

def transpose_eigR(eigR):
    #transposes a rotation matrix (eigenvectors)
    #INPUT and OUTPUT: .shape = (4,N) or .shape = (9,N)
    if eigR.shape[0] == 4:
     return array([eigR[0,:],eigR[2,:],\
                   eigR[1,:],eigR[3,:]])
    else: #3D
     return array([eigR[0,:],eigR[3,:],eigR[6,:],\
                   eigR[1,:],eigR[4,:],eigR[7,:],\
                   eigR[2,:],eigR[5,:],eigR[8,:]])

def sym2asym(HH):
    #converts between upper diagonal storage and full storage of a
    #SPD tensor
    #INPUT : HH.shape = (3,N) or HH.shape(6,N) for 2D and 3D, respectively.
    #OUTPUT: argout.shape = (4,N) or outarg.shape(9,N) for 2D and 3D, respectively.
    if HH.shape[0] == 3:
        return array([HH[0,:],HH[1,:],\
                      HH[1,:],HH[2,:]])
    else:
        return array([HH[0,:],HH[1,:],HH[3,:],\
                      HH[1,:],HH[2,:],HH[4,:],\
                      HH[3,:],HH[4,:],HH[5,:]])

def fulleig(eigL):
    #creates a diagonal tensor from a vector
    #INPUT : eigL.shape = (2,N) or eigL.shape = (3,N) for 2D and 3D, respetively.
    #OUTPUT: outarg.shape = (3,N) or outarg.shape = (6,N) for 2D and 3D, respectively.
    zeron = zeros(eigL.shape[1])
    if eigL.shape[0] == 2:
        return array([eigL[0,:],zeron,eigL[1,:]])
    else: #3D
        return array([eigL[0,:],zeron,eigL[1,:],zeron,zeron,eigL[2,:]])

def analyt_rot(H,eigR):
  #rotates a symmetric matrix, i.e. it calculates the tensor product
  # R*h*R.T, where
  #INPUT : H.shape = (3,N) or H.shape = (6,N) for 2D and 3D matrices, respectively.
  #INPUT : eigR.shape = (2,N) or eigR.shape = (3,N) for 2D and 3D, respetively.
  #OUTPUT: A.shape = (3,N) or A.shape = (6,N) for 2D and 3D, respectively
  if H.shape[0] == 3: #2D
   inds  = array([[0,1],[1,2]])
   indA = array([[0,1],[2,3]])
  else: #3D
   inds  = array([[0,1,3],[1,2,4],[3,4,5]])
   indA = array([[0,1,2],[3,4,5],[6,7,8]])
  indB = indA.T
  A = zeros(H.shape)
  for i in range(len(inds)):
    for j in range(len(inds)):
      for m in range(len(inds)):
        for n in range(len(inds)):
          if i<n:
           continue
          A[inds[i,n],:] += eigR[indB[i,j],:]*H[inds[j,m],:]*eigR[indA[m,n],:]
  return A

def prod_eig(H, eigL):
    #calculates the tensor product of H and diag(eigL), where
    #H is a tensor and eigL is a vector (diag(eigL) is a diagonal tensor).
    #INPUT : H.shape = (3,N) or H.shape(6,N) for 2D and 3D, respectively and
    #INPUT : eigL.shape = (2,N) or eigL.shape = (3,N) for 2D and 3D, respectively
    #OUTPUT: argout.shape = (3,N) or argout.shape = (6,N) for 2D and 3Dm respectively
    if H.shape[0] == 3:
        return array([H[0,:]*eigL[0,:], H[1,:]*numpy.sqrt(eigL[0,:]*eigL[1,:]), \
                                        H[2,:]*eigL[1,:]])
    else:
        return array([H[0,:]*eigL[0,:], H[1,:]*numpy.sqrt(eigL[0,:]*eigL[1,:]), H[2,:]*eigL[1,:], \
                                        H[3,:]*numpy.sqrt(eigL[0,:]*eigL[2,:]), H[4,:]*numpy.sqrt(eigL[2,:]*eigL[1,:]),\
                                        H[5,:]*eigL[2,:]])

def analytic_eig(H, tol=1e-12):
  #calculates the eigenvalues and eigenvectors using explicit analytical
  #expression for an array of 2x2 and a 3x3 symmetric matrices.
  #if numpy.__version__ >= "1.8.0", the vectorisation functionality of numpy.linalg.eig is used
  #INPUT: H.shape = (3,N) or H.shape = (6,N) for 2x2 and 3x3, respectively. Refer to sym2asym for ordering convention
  #       tol, is an optinal numerical tolerance for identifying diagonal matrices
  #OUTPUT: eigL.shape = (2,N) or eigL.shape = (3,N) for 2x2 and 3x3, resptively.
  #OUTPUT: eigR.shape = (4,N) or eigr.shape = (9,N) for 2x2 and 3x3, resptively. Refer to transpose_eigR for ordering convention
  H11 = H[0,:]
  H12 = H[1,:]
  H22 = H[2,:]
  onesC = ones(len(H11))
  if H.shape[0] == 3:
      if numpy.__version__ < "1.8.0":
          lambda1 = 0.5*(H11+H22-numpy.sqrt((H11-H22)**2+4*H12**2))
          lambda2 = 0.5*(H11+H22+numpy.sqrt((H11-H22)**2+4*H12**2))
          v1x = ones(len(H11)); v1y = zeros(len(H11))
          #identical eigenvalues
          I2 = numpy.abs(lambda1-lambda2)<onesC*tol;
          #diagonal matrix
          I1 = numpy.abs(H12)<onesC*tol
          lambda1[I1] = H11[I1]
          lambda2[I1] = H22[I1]
          #general case
          nI = (I1==False)*(I2==False)
          v1x[nI] = -H12[nI]
          v1y[nI] = H11[nI]-lambda1[nI]
          L1 = numpy.sqrt(v1x**2+v1y**2)
          v1x /= L1
          v1y /= L1
          eigL = array([lambda1,lambda2])
          eigR = array([v1x,v1y,-v1y,v1x])
      else:
          Hin = zeros([len(H11),2,2])
          Hin[:,0,0] = H11; Hin[:,0,1] = H12
          Hin[:,1,0] = H12; Hin[:,1,1] = H22
  else: #3D
      H13 = H[3,:]
      H23 = H[4,:]
      H33 = H[5,:]
      if numpy.__version__ < "1.8.0":
          p1 = H12**2 + H13**2 + H23**2
          zeroC = zeros(len(H11))
          eig1 = array(H11); eig2 = array(H22); eig3 = array(H33) #do not modify input
          v1 = array([onesC, zeroC, zeroC])
          v2 = array([zeroC, onesC, zeroC])
          v3 = array([zeroC, zeroC, onesC])
          # A is not diagonal.
          nI = (numpy.abs(p1) > tol**2)
          p1 = p1[nI]
          H11 = H11[nI]; H12 = H12[nI]; H22 = H22[nI];
          H13 = H13[nI]; H23 = H23[nI]; H33 = H33[nI];
          q = array((H11+H22+H33)/3.)
#          H11 /= q; H12 /= q; H22 /= q; H13 /= q; H23 /= q; H33 /= q
#          p1 /= q**2; qold = q; q = ones(len(H11))
          p2 = (H11-q)**2 + (H22-q)**2 + (H33-q)**2 + 2.*p1
          p = numpy.sqrt(p2 / 6.)
          I = array([onesC,zeroC,onesC,zeroC,zeroC,onesC])#I = array([1., 0., 1., 0., 0., 1.]).repeat(len(H11)).reshape(6,len(H11)) #identity matrix
          HH = array([H11,H12,H22,H13,H23,H33])
          B = (1./p) * (HH-q.repeat(6).reshape(len(H11),6).T*I[:,nI])
          #detB = B11*B22*B33+2*(B12*B23*B13)-B13*B22*B13-B12*B12*B33-B11*B23*B23
          detB = B[0,:]*B[2,:]*B[5,:]+2*(B[1,:]*B[4,:]*B[3,:])-B[3,:]*B[2,:]*B[3,:]-B[1,:]*B[1,:]*B[5,:]-B[0,:]*B[4,:]*B[4,:]

          #calc r
          r = detB / 2.
          rsmall = r<=-1.
          rbig   = r>= 1.
          rgood = (rsmall==False)*(rbig==False)
          phi = zeros(len(H11))
          phi[rsmall] = pi / 3.
          phi[rbig]   = 0.
          phi[rgood]  = numpy.arccos(r[rgood]) / 3.

          eig1[nI] = q + 2.*p*numpy.cos(phi)
          eig3[nI] = q + 2.*p*numpy.cos(phi + (2.*pi/3.))
          eig2[nI] = array(3.*q - eig1[nI] - eig3[nI])
#          eig1[nI] *= qold; eig2[nI] *= qold; eig3[nI] *= qold
          v1[0,nI] = H22*H33 - H23**2 + eig1[nI]*(eig1[nI]-H33-H22)
          v1[1,nI] = H12*(eig1[nI]-H33)+H13*H23
          v1[2,nI] = H13*(eig1[nI]-H22)+H12*H23
          v2[0,nI] = H12*(eig2[nI]-H33)+H23*H13
          v2[1,nI] = H11*H33 - H13**2 + eig2[nI]*(eig2[nI]-H11-H33)
          v2[2,nI] = H23*(eig2[nI]-H11)+H12*H13
          v3[0,nI] = H13*(eig3[nI]-H22)+H23*H12
          v3[1,nI] = H23*(eig3[nI]-H11)+H13*H12
          v3[2,nI] = H11*H22 - H12**2 + eig3[nI]*(eig3[nI]-H11-H22)
          L1 = numpy.sqrt((v1[:,nI]**2).sum(0))
          L2 = numpy.sqrt((v2[:,nI]**2).sum(0))
          L3 = numpy.sqrt((v3[:,nI]**2).sum(0))
          v1[:,nI] /= L1.repeat(3).reshape(len(L1),3).T
          v2[:,nI] /= L2.repeat(3).reshape(len(L1),3).T
          v3[:,nI] /= L3.repeat(3).reshape(len(L1),3).T
          eigL = array([eig1,eig2,eig3])
          eigR = array([v1[0,:],v1[1,:],v1[2,:],\
                        v2[0,:],v2[1,:],v2[2,:],\
                        v3[0,:],v3[1,:],v3[2,:]])
          bad = (numpy.abs(analyt_rot(fulleig(eigL),eigR)-H).sum(0) > tol) | isnan(eigR).any(0) | isnan(eigL).any(0)
          if any(bad):
           log(INFO,'%0.0f problems in eigendecomposition' % bad.sum())
           for i in numpy.where(bad)[0]:
               [eigL_,eigR_] = pyeig(array([[H[0,i],H[1,i],H[3,i]],\
                                            [H[1,i],H[2,i],H[4,i]],\
                                            [H[3,i],H[4,i],H[5,i]]]))
               eigL[:,i] = eigL_
               eigR[:,i] = eigR_.T.flatten()
      else:
          Hin = zeros([len(H11),3,3])
          Hin[:,0,0] = H11; Hin[:,0,1] = H12; Hin[:,0,2] = H13
          Hin[:,1,0] = H12; Hin[:,1,1] = H22; Hin[:,1,2] = H23
          Hin[:,2,0] = H13; Hin[:,2,1] = H23; Hin[:,2,2] = H33
  if numpy.__version__ >= "1.8.0":
          [eigL,eigR] = pyeig(Hin)
          eigL = eigL.T
          eigR = eigR.transpose([0,2,1]).reshape([len(H11),array(Hin.shape[1:3]).prod()]).T
  return [eigL,eigR]

def logexpmetric(Mp,logexp='log'):
    #calculates various tensor transformations in the principal frame
    #INPUT : DOLFIN TENSOR VARIABLE
    #INPUT : logexp is an optinal argument specifying the transformation,
    #        valid values are:
    #        'log'    , natural logarithm (default)
    #        'exp'    , exponential
    #        'inv'    , inverse
    #        'sqr'    , square
    #        'sqrt'   , square root
    #        'sqrtinv', inverse square root
    #        'sqrinv' , inverse square

    #OUTPUT: DOLFIN TENSOR VARIABLE
    [H,cell2dof] = get_dofs(Mp)
    [eigL,eigR] = analytic_eig(H)
    if logexp=='log':
      eigL = numpy.log(eigL)
    elif logexp=='sqrt':
      eigL = numpy.sqrt(eigL)
    elif logexp=='inv':
      eigL = 1./eigL
    elif logexp=='sqr':
      eigL = eigL**2
    elif logexp=='sqrtinv':
      eigL = numpy.sqrt(1./eigL)
    elif logexp=='sqrinv':
      eigL = 1./eigL**2
    elif logexp=='exp':
      eigL = numpy.exp(eigL)
    else:
      error('logexp='+logexp+' is an invalid value')
    HH = analyt_rot(fulleig(eigL),eigR)
    out = sym2asym(HH).transpose().flatten()
    Mp.vector().set_local(out)
    return Mp

def minimum_eig(Mp):
    # calculates the minimum eigenvalue of a DOLFIN TENSOR VARIABLE
    # INPUT : DOLFIN TENSOR VARIABLE
    # OUTPUT: DOLFIN SCALAR VARIABLE
    mesh = Mp.function_space().mesh()
    element = Mp.function_space().ufl_element()
    [H,cell2dof] = get_dofs(Mp)
    [eigL,eigR] = analytic_eig(H)
    out = Function(FunctionSpace(mesh,element.family(),element.degree()))
    out.vector().set_local(eigL.min(0))
    return out

def get_rot(Mp):
    mesh = Mp.function_space().mesh()
    element = Mp.function_space().ufl_element()
    [H,cell2dof] = get_dofs(Mp)
    [eigL,eigR] = analytic_eig(H)
    out = Function(TensorFunctionSpace(mesh,element.family(),element.degree()))
    out.vector().set_local(eigR.transpose().flatten())
    return out

def logproject(Mp):
    # provides projection to a CG1 tensor space in log-space.
    # That is,
    # #1 An eigen decomposition is calculated for the input tensor
    # #2 This is used to calculate the tensor logarithm
    # #3 which is the projected onto the CG1 tensor space
    # #4 Finally, the inverse operation, a tensor exponential is performed.
    # This approach requires SPD input, but also preserves this attribute.
    # INPUT : DOLFIN SPD TENSOR VARIABLE
    # OUTPUT: DOLFIN SPD CG1 TENSOR VARIABLE
    mesh = Mp.function_space().mesh()
    logMp = project(logexpmetric(Mp),TensorFunctionSpace(mesh,'CG',1))
    return logexpmetric(logMp,logexp='exp')

def mesh_metric(mesh):
    # calculates a mesh metric (that is it has unit of squared inverse length,
    # use mesh_metric2 to get units of length)
    # On each element, the edge i is delimited by two points of coordinate vectors x_i and y_i.
    # Define the edge vector Li to be given by ri = yi - xi and define the element mesh metric tensor M such that
    #
    # ri^T M ri = 1, for each i .
    #
    # To compute M, we use the exact formula given in https://doi.org/10.1016/j.crma.2016.11.007
    #
    # M = (gdim + 1)/2*(\sum_{i<j}ri*ri^T)^{-1}
    #
    # The matrix inverse in the formula is computed using the exact formula for a symmetric matrix inverse in 2D and 3D,
    # the implementation is done so that the code is vectorised.
    #
    # INPUT : DOLFIN MESH
    # OUTPUT: DOLFIN DG0 SPD TENSOR VARIABLE

    # geometric dimension and its square
    gdim = mesh.geometry().dim()
    sqdim = gdim**2

    # extract cells and mesh coordinates
    cells_array = mesh.cells()
    coords = mesh.coordinates()

    # the following routine takes an m-by-n matrix r and computes the outer product of each row vector s[i,:,:] = outer(r[i,:], r[i,:])
    # in a vectorised way, where s[i,:,:] is a n-by-n rank 1 matrix. Once all of the outer products have been computed,
    # the tensor s is flattened into a 1D array, which is made of the flattened s[i,:,:] stacked one after the other.
    # We use this routine to compute the outer products between the edge vectors efficiently.
    def row_wise_outer_product(r):
        return numpy.einsum('ij,ik->ijk',r,r).flatten()

    # get all edge vectors
    p = [coords[cells_array[:,i],:] for i in range(gdim+1)]
    # compute the inverse of M on each cell
    invM = 2.0/(gdim + 1)*sum([row_wise_outer_product(p[i]-p[j]) for i,j in combinations(range(gdim+1), 2)])

    M_val = invM.copy()

    # in the following we compute a vectorised exact inverse of invM for each cell so as to obtain M
    if gdim == 2:
        # exact formula in 2D

        # determinants
        determinants = (invM[::sqdim]*invM[3::sqdim] - invM[1::sqdim]**2).repeat(sqdim)

        # swap the matrix diagonal entries
        M_val[::sqdim]  = invM[3::sqdim].copy()
        M_val[3::sqdim] = invM[ ::sqdim].copy()

        # change the sign of off-diagonal entries
        M_val[1::sqdim] *= -1
        M_val[2::sqdim] *= -1

        # divide by determinants
        M_val /= determinants
    else:
        # exact formula in 3D, not that nice

        # determinants
        determinants = (invM[8::sqdim]*invM[1::sqdim]**2 + invM[4::sqdim]*invM[2::sqdim]**2 + invM[::sqdim]*invM[5::sqdim]**2 \
                  - 2.0*invM[1::sqdim]*invM[2::sqdim]*invM[5::sqdim] - invM[::sqdim]*invM[4::sqdim]*invM[8::sqdim]).repeat(sqdim)

        # diagonal entries
        M_val[::sqdim]  = invM[5::sqdim]**2 - invM[4::sqdim]*invM[8::sqdim]
        M_val[4::sqdim] = invM[2::sqdim]**2 - invM[::sqdim]*invM[8::sqdim]
        M_val[8::sqdim] = invM[1::sqdim]**2 - invM[::sqdim]*invM[4::sqdim]

        # off-diagonal entries
        M_val[1::sqdim] = invM[1::sqdim]*invM[8::sqdim] - invM[2::sqdim]*invM[5::sqdim]
        M_val[2::sqdim] = invM[2::sqdim]*invM[4::sqdim] - invM[1::sqdim]*invM[5::sqdim]
        M_val[5::sqdim] = invM[::sqdim]*invM[5::sqdim] - invM[1::sqdim]*invM[2::sqdim]

        # impose symmetry
        M_val[3::sqdim] = M_val[1::sqdim]
        M_val[6::sqdim] = M_val[2::sqdim]
        M_val[7::sqdim] = M_val[5::sqdim]

        # divide by determinants
        M_val /= determinants

    # define dolfin metric tensor and assign the computed values
    M = Function(TensorFunctionSpace(mesh, "DG", 0))
    M.vector().set_local(M_val)

    return M

def mesh_metric1(mesh):
  #this is just the inverse of mesh_metric2
  M = mesh_metric(mesh)
  #M = logexpmetric(M,logexp='sqrt')
  [MM,cell2dof] = get_dofs(M)
  [eigL,eigR] = analytic_eig(MM)
  eigL = numpy.sqrt(eigL)
  MM = analyt_rot(fulleig(eigL),eigR)
  MM = sym2asym(MM).transpose().flatten()
  M.vector().set_local(MM[cell2dof.flatten()])
  return M

def mesh_metric2(mesh):
  #calculates a metric field, which when divided by sqrt(3) corresponds to the steiner
  #ellipse for the individual elements, see the test case mesh_metric2_example
  #the sqrt(3) ensures that the unit element maps to the identity tensor
  M = mesh_metric(mesh)
  #M = logexpmetric(M,logexp='sqrtinv')
  [MM,cell2dof] = get_dofs(M)
  [eigL,eigR] = analytic_eig(MM)
  eigL = numpy.sqrt(1./eigL)
  MM = analyt_rot(fulleig(eigL),eigR)
  MM = sym2asym(MM).transpose().flatten()
  M.vector().set_local(MM[cell2dof.flatten()])
  return M

def gradate(H, grada, itsolver=False):
    # provides anisotropic Helm-holtz smoothing on the logarithm
    #of a metric based on the metric of the mesh times a scaling factor(grada)
    if itsolver:
        solverp = {"linear_solver": "cg", "preconditioner": "ilu"}
    else:
        solverp = {"linear_solver": "lu"}
    mesh = H.function_space().mesh()
    grada = Constant(grada)
    mm2 = mesh_metric2(mesh)
    mm2sq = dot(grada*mm2,grada*mm2)
    Hold = Function(H); H = logexpmetric(H) #avoid logexpmetric side-effect
    V = TensorFunctionSpace(mesh,'CG',1); H_trial = TrialFunction(V); H_test = TestFunction(V); Hnew=Function(V)
    a = (inner(grad(H_test),dot(mm2sq,grad(H_trial)))+inner(H_trial,H_test))*dx
    L = inner(H,H_test)*dx
    solve(a==L,Hnew,[], solver_parameters=solverp)
    Hnew = metric_ellipse(logexpmetric(Hnew,logexp='exp'), Hold)
    return Hnew


def c_cell_dofs(mesh,V):
  #returns the degree of free numbers in each cell (for DG0 input) input or a each
  # vertex (CG1 input).
  # INPUT : DOLFIN TENSOR VARIABLE (CG1 or DG0)
  # OUTPUT: outarg.shape = (4*N,) or outarg.shape = (9*N,)
  # The DOLFIN storage CONVENTION was greatly simplified at 1.3.0 :
  if dolfin.__version__ >= '1.3.0':
   if V.ufl_element().is_cellwise_constant():
    return arange(mesh.num_cells()*mesh.geometry().dim()**2)
   else:
    return arange(mesh.num_vertices()*mesh.geometry().dim()**2)
  else:
      #returns the degrees of freedom numbers in a cell
      code = """
      void cell_dofs(boost::shared_ptr<GenericDofMap> dofmap,
                     const std::vector<std::size_t>& cell_indices,
                     std::vector<std::size_t>& dofs)
      {
        assert(dofmap);
        std::size_t local_dof_size = dofmap->cell_dofs(0).size();
        const std::size_t size = cell_indices.size()*local_dof_size;
        dofs.resize(size);
        for (std::size_t i=0; i<cell_indices.size(); i++)
           for (std::size_t j=0; j<local_dof_size;j++)
               dofs[i*local_dof_size+j] = dofmap->cell_dofs(cell_indices[i])[j];
      }
      """
      module = compile_extension_module(code)
      return module.cell_dofs(V.dofmap(), arange(mesh.num_cells(), dtype=numpy.uintp))


if __name__=="__main__":
 testcase = 3
 if testcase == 0:
   from minimal_example import minimal_example
   minimal_example(width=5e-2)
 elif testcase == 1:
   from minimal_example_minell import check_metric_ellipse
   check_metric_ellipse(width=2e-2)
 elif testcase == 2:
   from play_multigrid import test_refine_metric
   test_refine_metric()
 elif testcase == 3:
   from mesh_metric2_example import test_mesh_metric
   test_mesh_metric()
 elif testcase == 4:
   from circle_convergence import circle_convergence
   circle_convergence()
 elif testcase == 5:
   from maximal_example import maximal_example
   maximal_example()
