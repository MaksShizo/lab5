#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int size, rank;
const int N = 1000;
double a[N][N], x[N], x1[N], x_check[N], a_check[N][N];
int count_error = 0;

void BackGauss()
{
	for (int i = N - 1; i >= 0; i--)
	{
		int count = 0;
		x1[i] = x[i];
		for (int j = i + 1; j < N; j++)
		{
			x1[i] -= a[i][j] * x1[j];

		}
		x1[i] = x1[i] / a[i][i];
	}
}

void CheckCountError() {
	double sum = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			sum += x1[j] * a_check[i][j];
		}
		if (std::abs(sum - x_check[i]) > 1e-6)
			count_error++;
		sum = 0;
	}
}

void GaussMPI(int N)
{
	MPI_Request request;
	MPI_Status status;
	float mp;
	MPI_Barrier(MPI_COMM_WORLD);
	for (int k = 0; k < N - 1; k++)
	{
		MPI_Bcast(&a[k][0], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&x[k], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (rank == 0)
		{
			for (int p = 1; p < size; p++)
			{
				for (int i = k + 1 + p; i < N; i += size)
				{
					MPI_Isend(&a[i], N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					MPI_Isend(&x[i], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
				}
			}
			for (int i = k + 1; i < N; i += size)
			{
				mp = a[i][k] / a[k][k];
				for (int j = k; j < N; j++)
				{
					a[i][j] -= a[k][j] * mp;
				}
				x[i] -= x[k] * mp;
			}
			for (int p = 1; p < size; p++)
			{
				for (int i = k + 1 + p; i < N; i += size)
				{
					MPI_Recv(&a[i], N, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
					MPI_Recv(&x[i], 1, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
				}
			}
		}
		else
		{
			for (int i = k + 1 + rank; i < N; i += size)
			{
				MPI_Recv(&a[i], N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&x[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
				mp = a[i][k] / a[k][k];
				for (int j = k; j < N; j++)
				{
					a[i][j] -= a[k][j] * mp;
				}
				x[i] -= x[k] * mp;
				MPI_Isend(&a[i], N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				MPI_Isend(&x[i], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	unsigned long long usecstart, usecstop;
	double startTime, endTime;
	if (rank == 0)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				a[i][j] = rand() % 100000 / 100.0;
				a_check[i][j] = a[i][j];
			}
			x[i] = rand() % 100000 / 100.0;
			x_check[i] = x[i];
		}
	}
	startTime = MPI_Wtime();
	GaussMPI(N);//implementing the gaussian elimination
	if (rank == 0)
	{
		BackGauss();
		endTime = MPI_Wtime();
		CheckCountError();
		printf("Time gaussMPI  = %f s Count Error %d", endTime - startTime, count_error);
	}
	MPI_Finalize(); //Finalizing the MPI
	return 0;
}