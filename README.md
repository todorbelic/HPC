# HPC

1. Sekvencijalna verzija
   
gcc -O3 -o abc_sequential abc_openmp.c -lm

2. OPENMP verzija

gcc -O3 -fopenmp -DUSE_OPENMP -o abc_openmp abc_openmp_original.c -lm

3. MPI verzija

mpicc -O3 -o abc_mpi abc_mpi_original.c -lm
