#include <iostream>
#include <mpi.h>

#define SIZE 10

// ! mpiexec -n 4 main.exe 

int main(int argc, char** argv) {

    // Инициализация MPI
    MPI_Init(&argc, &argv);

    // Получаем число и ранг процессов
    int rank, size;
    int array[SIZE];
    int local_max = 0, global_max;
    int local_sum = 0, global_sum;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string message;
    if (rank == 0) {
        for (int i = 0; i < SIZE; ++i) {
            array[i] = i + 1;
        }
    }

    MPI_Bcast(array, SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        for (int i = 0; i < SIZE; i++) {
            array[i] = array[i] * rank;
        }

        for (int i = 0; i < SIZE; i++) {
            local_sum += array[i];
        }

        for (int i = 0; i < SIZE; i++) {
            if (array[i] > local_max) {
                local_max = array[i];
            }
        }
    }

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sum: %d\n", global_sum);
        printf("Max: %d\n", global_max);
    }

    MPI_Finalize();

    return 0;
}
