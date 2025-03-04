#include <iostream>
#include <mpi.h>

// ! mpiexec -n 4 main.exe 

int main(int argc, char** argv) {

    // Инициализация MPI
    MPI_Init(&argc, &argv);

    // Получаем число и ранг процессов
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string message;
    if (rank == 0) {
        // Процесс 0 отправляет сообщения
        for (int i = 1; i < size; ++i) {
            std::cout << "I am process number 0 send message to process number " << std::to_string(i) << std::endl;
            message = "I am process number 0 send message to process number " + std::to_string(i);
            MPI_Send(message.c_str(), message.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // Другие процессы принимают сообщения
        char buffer[256];
        MPI_Recv(buffer, sizeof(buffer), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process " << rank << " receive message: " << buffer << std::endl;
    }

    // Завершение работы MPI
    MPI_Finalize();

    return 0;
}
