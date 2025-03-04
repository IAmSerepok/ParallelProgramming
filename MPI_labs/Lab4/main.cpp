#include <iostream>
#include <mpi.h>
#include <time.h>

// ! mpiexec -n 4 main.exe 

int main(int argc, char** argv) {

    // Инициализация MPI
    MPI_Init(&argc, &argv);

    // Получаем число и ранг процессов
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int data[4 * size];
    int local_data[4]; 
    int gathered_data[4 * size]; 

    MPI_Request request;

    // Синхронщина
    double start = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < 4 * size; i++) {
            data[i] = i + 1; 
        }
        // printf("Process 0 scatter data: ");
        // for (int i = 0; i < 4 * size; i++) {
        //     printf("%d ", data[i]);
        // }
        // printf("\n");
    }

    MPI_Scatter(data, 4, MPI_INT, local_data, 4, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < 4; i++) {
        local_data[i] *= rank;
    }

    MPI_Gather(local_data, 4, MPI_INT, gathered_data, 4, MPI_INT, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Synchronous data exchange took: %lf sec.\n", end - start);
        printf("Resulting array of data (synchronous): ");
        for (int i = 0; i < 4 * size; i++) {
            printf("%d ", gathered_data[i]);
        }
        printf("\n");
    }

    // Асинхронщина
    start = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < 4 * size; i++) {
            data[i] = i + 1; 
        }
        // printf("Process 0 scatter data: ");
        // for (int i = 0; i < 4 * size; i++) {
        //     printf("%d ", data[i]);
        // }
        // printf("\n");
    }

    MPI_Iscatter(data, 4, MPI_INT, local_data, 4, MPI_INT, 0, MPI_COMM_WORLD, &request);

    for (int i = 0; i < 4; i++) {
        local_data[i] *= rank;
    }

    MPI_Wait(&request, MPI_STATUS_IGNORE);

    MPI_Igather(local_data, 4, MPI_INT, gathered_data, 4, MPI_INT, 0, MPI_COMM_WORLD, &request);

    end = MPI_Wtime();

    if (rank == 0) {
        printf("Asynchronous data exchange took: %lf sec.\n", end - start);
        printf("Resulting array of data (asynchronous): ");
        for (int i = 0; i < 4 * size; i++) {
            printf("%d ", gathered_data[i]);
        }
        printf("\n");
    }

    MPI_Finalize();

    return 0;
}
