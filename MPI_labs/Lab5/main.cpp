#include <iostream>
#include <mpi.h>
#include <stdlib.h>

// ! mpiexec -n 4 main.exe 

int main(int argc, char** argv) {
    int rank, size;
    int n = 300000000, count = 0;
    double x, y, pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    srand(1 + rank);

    int local_n = n / size;

    // Генерация случайных точек и подсчёт попавших в круг
    for (int i = 0; i < local_n; i++) {
        x = (double)rand() / RAND_MAX; 
        y = (double)rand() / RAND_MAX; 
        if (x * x + y * y <= 1) {
            count++;
        }
    }

    int total_count;
    MPI_Reduce(&count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi = 4.0 * total_count / n;
        printf("Approximation of Pi: %.4lf\n", pi);
        printf("Time taken: %lf seconds", (MPI_Wtime() - start_time));
    }

    MPI_Finalize();
    return 0;
}
