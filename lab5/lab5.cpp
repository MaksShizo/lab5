#include <iostream>
#include <mpi.h>
#include <chrono>

const int N = 10;

void Gauss(double** a, double* x, int n) {
	/* Прямой ход*/
	unsigned int start = clock();
	for (int k = 1; k < n; k++) {
#pragma omp parallel for
		for (int j = k; j < n; j++) {
			double d = a[j][k - 1] / a[k - 1][k - 1];
			for (int i = 0; i <= n; i++) {
				a[j][i] = a[i][i] - d * a[k - 1][i];
			}
		}
	}

	/*Обратный ход*/
	for (int i = n - 1; i >= 0; i--) {
		double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
		for (int j = i + 1; j < n; j++) {
			sum += a[i][j] * x[j]; // / a[i][i];
		}
		x[i] = (x[i] - sum) / a[i][i];
	}
	std::cout << "Гаусс " << clock() - start << std::endl;
}

void GaussMPI(double* a, double* x, int n, int size, int rank) {

	int blockSize = n / size;
	int remainder = n % size;

	if (rank == 0)
	{
		int start = 1;
		int end = start + blockSize;

		double* temp_a = new double[blockSize * (N + 1)];
		int index = 0;
		for (int num = 1; num < size; num++)
		{

			for (int i = 0; i < blockSize * (N + 1); i++)
			{
				temp_a[i] = a[index];
				index++;
			}
			MPI_Send(temp_a, blockSize * (N + 1), MPI_DOUBLE, num, 0, MPI_COMM_WORLD);
		}
		index = 0;
		for (int num = 1; num < size; num++)
		{
			MPI_Recv(temp_a, blockSize * (N + 1), MPI_DOUBLE, num, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			for (int i = 0; i < blockSize * (N + 1); i++)
			{
				a[index] = temp_a[i];
				index++;
			}

		}
		delete[] temp_a;
		/*Обратный ход*/
		for (int i = n - 1; i >= 0; i--) {
			x[i] = a[i * N + N];
			for (int j = i + 1; j < N; j++) {
				x[i] -= a[i * N + j] * x[j];
			}
			x[i] /= a[i * N + i];
		}
	}
	else
	{
		int start = rank * (blockSize - 1);
		int end = start + blockSize;



		double* temp_a = new double[blockSize * (N + 1)];
		double* temp_b = new double[blockSize * (N + 1)];

		MPI_Recv(temp_a, blockSize * (N + 1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


		for (int num = 1; num < rank; num++)
		{

			MPI_Recv(temp_b, blockSize * (N + 1), MPI_DOUBLE, num, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
			for (int i = 0; i < blockSize; i++) {
				for (int j = 0; j < blockSize; j++) {
					double ratio = temp_b[i * N + j + blockSize * (num - 1)];
					for (int k = N + 1; k >= j; k--) {
						temp_a[i * N + k] -= ratio * temp_b[j * N + k];
					}
				}
			}

		}
		delete[] temp_b;
		int counter = 0;
		for (int i = 0; i < blockSize; i++) {
			double ratio = temp_a[i * N + i + start + counter];

			for (int j = (N + 1); j >= start + counter; j--)
				temp_a[i * N + j] /= ratio;
			for (int j = i + 1; j < blockSize; j++) {
				ratio = temp_a[j * N + start + counter];
				for (int k = N + 1; k >= start + counter; k--)
					temp_a[j * N + k] -= ratio * temp_a[i * N + k];
			}
		}
		for (int num = rank + 1; num < size; num++)
		{
			MPI_Send(temp_a, blockSize * (N + 1), MPI_DOUBLE, num, 0, MPI_COMM_WORLD);
		}
		MPI_Send(temp_a, blockSize * (N + 1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		delete[] temp_a;

	}


}




int main(int argc, char** argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	double* a = new double[N * (N + 1)];
	double* x = new double[N];
	double* x1 = new double[N];

	if (rank == 0) {
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				a[i * N + j] = i + j + 1;
			}
			a[i * N + N] = i + 5;
		}

	}
	GaussMPI(a, x, N, size, rank);
	// Print the result in the root process
	if (rank == 0) {
		std::cout << "Solution:" << std::endl;
		for (int i = 0; i < N; ++i) {
			std::cout << "x[" << i << "] = " << x[i] << std::endl;
		}
	}

	MPI_Finalize();
	// Освобождение выделенной памяти

	return 0;
}
