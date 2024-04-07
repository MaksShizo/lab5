#include <iostream>
#include <mpi.h>
#include <chrono>

const int N = 100;
std::chrono::high_resolution_clock::time_point startTime, finishTime;

double A[N][N];
double B[N][N];
double C[N][N];
double D[N][N];

void printMatrix(double A[N][N], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        startTime = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = i + 1;
                B[i][j] = j + 1;
                C[i][j] = 0;
                D[i][j] = 0;
            }
        }
    }

    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int blockSize = N / size;
    int remainder = N % size;
    int start = rank * blockSize;
    int end = start + blockSize;

    if (rank == size - 1) {
        end += remainder;
    }

    for (int i = start; i < end; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    MPI_Reduce(C, D, N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Resulting Matrix:" << std::endl;
        printMatrix(D, N, N);
    }

    MPI_Finalize();
    return 0;
}
