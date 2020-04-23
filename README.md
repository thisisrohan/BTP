## Incremental Algorithms for Delaunay Meshing in two and three dimesnions

This is a part of my final year B.Tech. Project, and these files contain a python implementation of the Bowyer-Watson algorithm using Numba for improved performance performance.

---
### Status as of 23-April-2020
The unconstrained 2D and 3D triangulators work well, but the 3D BRIO implementation doesn't seem to offer much of a speed bumb. This is being investigated. Ruppert's algorithm seems to run into trouble with smaller triangles because of floating point arithmetic, for this I'm presently working on incorporating Shewchuk's adaptive precision predicates.
