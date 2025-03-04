#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <vector>
#include <chrono>

#define N 300000000

// ! mpiexec -n 4 main.exe 

void initialize_array(std::vector<int>& arr) {
    for (int j = 0; j < arr.size(); j++) {
        arr[j] = rand() % 500; 
    }
}

void print_array(std::vector<int>& arr) {
    for (int i = 0; i < N; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n\n";
}

int find_max(const std::vector<int>& arr) {
    int max_element = arr[0];  
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] > max_element) {
            max_element = arr[i];
        }
    }
    return max_element;
}


int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(42); 

    std::vector<int> arr(N);

    if (rank == 0) {
        initialize_array(arr);
        // print_array(arr);

        auto start_time_seq = std::chrono::high_resolution_clock::now();
        
        int max_element = find_max(arr);

        auto end_time_seq = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> seq_duration = end_time_seq - start_time_seq;

        std::cout << "Sequential Global max = " << max_element << std::endl;
        std::cout << "Sequential execution time: " << seq_duration.count() << " seconds" << std::endl;
    }

    int base_size = N / size;        
    int remainder = N % size;    

    int elements_for_process = base_size + (rank < remainder ? 1 : 0);
    std::vector<int> arr_sub(elements_for_process);

    std::vector<int> sendcounts(size, base_size);
    std::vector<int> displs(size, 0);
    

    for (int i = 0; i < remainder; i++) {
        sendcounts[i] += 1;
    }

    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + sendcounts[i - 1]; 
    }

    auto start_time_par = std::chrono::high_resolution_clock::now();

    MPI_Scatterv(arr.data(), sendcounts.data(), displs.data(), MPI_INT,
                  arr_sub.data(), elements_for_process, MPI_INT, 0, MPI_COMM_WORLD);

    int local_max = find_max(arr_sub);

    int global_max;

    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        auto end_time_par = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> par_duration = end_time_par - start_time_par;

        std::cout << "Global max = " << global_max << std::endl;
        std::cout << "Parallel execution time: " << par_duration.count() << " seconds" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
