#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <vector>

#define N 500

// Function to initialize matrices
void initialize_matrices(std::vector<int>& matrixA, std::vector<int>& matrixB) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Filling matrices as i + j, stored as a 1D array
            matrixA[i * N + j] = i + j;
            matrixB[i * N + j] = i + j;
        }
    }
}

// Function to multiply matrices (serial)
void multiply_matrices(const std::vector<int>& matrixA, const std::vector<int>& matrixB, std::vector<int>& result) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                result[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
            }
        }
    }
}

// Function to multiply matrices in parallel
void multiply_matrices_parallel(int rank, int size, const std::vector<int>& matrixA, const std::vector<int>& matrixB, std::vector<int>& local_result, int rows_for_process) {
    for (int i = 0; i < rows_for_process; i++) {
        for (int j = 0; j < N; j++) {
            local_result[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                local_result[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
            }
        }
    }
}

void print_matrix(const std::vector<int>& matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time;
    std::vector<int> matrixA(N * N);
    std::vector<int> matrixB(N * N);
    std::vector<int> result(N * N);
    std::vector<int> result_parallel(N * N);
    
    // Initialize matrices on rank 0
    if (rank == 0) {
        initialize_matrices(matrixA, matrixB);
    }

    // Measure time for sequential multiplication
    if (rank == 0) {
        start_time = MPI_Wtime();
        multiply_matrices(matrixA, matrixB, result);
        double end_time = MPI_Wtime();
        printf("Sequential multiplication took %lf seconds\n", end_time - start_time);
        // print_matrix(result);
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(matrixB.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    
    int rows_per_process = N / size;
    int extra_rows = N % size;
    int rows_for_process = rows_per_process + (rank < extra_rows ? 1 : 0);
    
    // Allocate local matrices
    std::vector<int> A_sub(rows_for_process * N);
    std::vector<int> C_local(rows_for_process * N); // Local result matrix

    // Scatter matrix A
    int sendcounts[size];
    int displacements[size];
    for (int i = 0; i < size; i++) {
        sendcounts[i] = rows_per_process * N;
        if (i < extra_rows) {
            sendcounts[i] += N; // Give extra rows to the first 'extra_rows' processes
        }
        displacements[i] = (i * rows_per_process + (i < extra_rows ? i : extra_rows)) * N;
    }

    start_time = MPI_Wtime();
    MPI_Scatterv(matrixA.data(), sendcounts, displacements, MPI_INT, A_sub.data(), rows_for_process * N, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Perform local multiplication
    multiply_matrices_parallel(rank, size, A_sub, matrixB, C_local, rows_for_process);
    
    // Gather results
    MPI_Gatherv(C_local.data(), rows_for_process * N, MPI_INT, result_parallel.data(), sendcounts, displacements, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // End time measurement
    if (rank == 0) {
        printf("Parallel multiplication took %lf seconds\n", end_time - start_time);
        // print_matrix(result_parallel);
    }

    MPI_Finalize();
    return 0;
}
