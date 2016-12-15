Hopscotch concurrent hash table [1] implementation with a few minor bug fixes
from the implementation from
https://sites.google.com/site/cconcurrencypackage/hopscotch-hashing

This implementation has been used in HYPRE sparse linear solver library [2] (http://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/software) and Intel optimized high performance conjugate gradient (HPCG) benchmark [3] (https://software.intel.com/en-us/articles/intel-mkl-benchmarks-suite).

The code is very simple with 1 header file and 1 source file.
It has been only tested with x86 processors and Intel compiler.
It should be easy to compile with GCC after minimal changes.

The test case contained here models a process for preparing MPI communication
to perform distributed sparse matrix operations like sparse matrix dense
vector multiplication (SpMV).
The test case models one MPI rank scenario of one MPI rank receiving matrix
rows belong to other MPI ranks and are connected to the rows belong to the
given rank.
The rows will be received with their global ids, and the test case eliminates
duplicated global ids and assigns contiguous local ids to them so that later
the data received from other MPI ranks can be copied to contiguous locations.
This is a very common pre-processing step for distributed sparse linear
algebra operations, and concurrent hash table is very useful to take
advantage of multiple cores available to accelerate this pre-processing step.
The test input file provided captures the case of 96^3 27-pt 3D Laplacian
parallelized with 16 MPI ranks.

[1] Maurice Herlihy, Nir Shavit, and Moran Tzafrir, Hopscotch Hashing, International Symposium on Distributed Computing (DISC), 2008

[2] Jongsoo Park, Mikhail Smelyanskiy, Ulrike Meier Yang, Dheevatsa Mudigere, and Pradeep Dubey, High-Performance Algebraic Multigrid Solver Optimized for Multi-Core Based Distributed Parallel Systems, The International Conference for High Performance Computing, Networking, Storage, and Analysis (SC), 2015

[3] Jongsoo Park, Mikhail Smelyanskiy, Karthikeyan Vaidyanathan, Alexander Heinecke, Dhiraj D. Kalamkar, Xing Liu, Md. Mostofa Ali Patwary, Yutong Lu, and Pradeep Dubey, Efficient Shared-Memory Implementation of High-Performance Conjugate Gradient Benchmark and Its Application to Unstructured Matrices, The International Conference for High Performance Computing, Networking, Storage, and Analysis (SC), 2014
