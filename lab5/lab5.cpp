#include <iostream>
#include <mpi.h>

const int N = 2000; // Размер матрицы
double** A; // Двумерный массив для расширенной матрицы системы
double X[N]; // Решение
int countError = 0; // Количество ошибок

void deletearr() {
    for (int i = 0; i < N; ++i) {
        delete[] A[i];
    }
    delete[] A;
}

void BackGauss() {
    for (int i = N - 1; i >= 0; --i) {
        X[i] = A[i][N];
        for (int j = i + 1; j < N; ++j) {
            X[i] -= A[i][j] * X[j];
        }
        X[i] /= A[i][i];
    }
}

void CountError() {
    double sum = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sum += X[j] * A[i][j];
        }
        if (std::abs(sum - A[i][N]) > 1e-6) {
            countError++;
        }
        sum = 0;
    }
    std::cout << "Error count:" << countError << std::endl;
}

int main(int argc, char** argv) {
    A = new double* [N];
    for (int i = 0; i < N; ++i) {
        A[i] = new double[N + 1];
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 10 + 1;
        }
        A[i][N] = rand() % N;
    }
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int blockSize = N / (size - 1);
    if (N % (size - 1) != 0) {
        std::cout << "Error count process. Take (Matrix size % (count process - 1) == 0)" << std::endl;
        deletearr();
        MPI_Finalize();
        return -1;
    }
    if (rank == 0) {
        double startT, endT;
        startT = MPI_Wtime();
        double* temp_a = new double[blockSize * (N + 1)];
        for (int proc = 1; proc < size; ++proc) {
            for (int i = 0; i < blockSize; ++i)
                for (int j = 0; j < N + 1; ++j)
                {
                    temp_a[i] = A[i][j];
                }
            MPI_Send(temp_a, blockSize * (N + 1), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
        }
        for (int proc = 1; proc < size; ++proc) {
            MPI_Recv(temp_a, blockSize * (N + 1), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < blockSize; ++i)
                for (int j = 0; j < N + 1; ++j) {
                    A[i][j] = temp_a[i];
                }
        }
        delete[] temp_a;
        BackGauss();
        endT = MPI_Wtime();
        std::cout << "Time work " << endT - startT << " s" << std::endl;
        CountError();
    }
    else {
        int start = blockSize * (rank - 1);
        int finish = rank * blockSize - 1;
        double* temp_a = new double[blockSize * (N + 1)];
        double* temp_b = new double[blockSize * (N + 1)];
        MPI_Recv(temp_a, blockSize * (N + 1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        for (int proc = 1; proc < rank; ++proc) {
            MPI_Recv(temp_b, blockSize * (N + 1), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            for (int i = 0; i < blockSize; ++i) {
                for (int j = 0; j < blockSize; ++j) {
                    double ratio = temp_b[i * N + j + blockSize * (proc - 1)];
                    for (int k = N + 1; k >= j; --k) {
                        temp_a[i * N + k] -= ratio * temp_b[j * N + k];
                    }
                }
            }
        }
        delete[] temp_b;
        int counter = 0;
        for (int i = 0; i < blockSize; ++i) {
            double ratio = temp_a[i * N + i + start + counter];
            for (int j = N; j >= start + counter; --j) {
                temp_a[i * N + j] /= ratio;
            }
            for (int j = i + 1; j < blockSize; ++j) {
                ratio = temp_a[j * N + start + counter];
                for (int k = N; k >= start + counter; --k) {
                    temp_a[j * N + k] -= ratio * temp_a[i * N + k];
                }
            }
        }
        for (int proc = rank + 1; proc < size; ++proc) {
            MPI_Send(temp_a, blockSize * (N + 1), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
        }
        MPI_Send(temp_a, blockSize * (N + 1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        delete[] temp_a;
    }

    deletearr();
    MPI_Finalize();
    return 0;
}
