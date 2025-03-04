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
    int local_sum = 0, total_sum = 0;

    if (rank == 0) {
        for (int i = 0; i < 4 * size; i++) {
            data[i] = i + 1; 
        }
        printf("Process 0 scatter data: ");
        for (int i = 0; i < 4 * size; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    MPI_Scatter(data, 4, MPI_INT, local_data, 4, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < 4; i++) {
        local_data[i] *= rank;
        local_sum += local_data[i];
    }

    MPI_Gather(local_data, 4, MPI_INT, gathered_data, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Process 0 gathered data: ");
        for (int i = 0; i < 4 * size; i++) {
            printf("%d ", gathered_data[i]);
        }
        printf("\n");
    }

    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Вывод локальной и глобальной суммы
    printf("Process %d got total sum of all multiplied elements: %d\n", rank, total_sum);

    MPI_Finalize();

    return 0;
}
