#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#define FILTER_DIM 3
#define FILTER_RADIUS 1
__constant__ int filter_dev[FILTER_DIM * FILTER_DIM];

extern "C" {
    int conv_dev(int nprocs, int my_rank, int my_work, int *h_in_image, int *h_out_image, int height, int width, int filter_dim, int *filter_cpu);
}


__global__ void conv_dev_cuda(int my_rank, int my_work, int height, int width, int *input, int *output) {
    int sum = 0;
    int filter_row, filter_col, in_row, in_col;

    int out_row = blockIdx.y * blockDim.y + threadIdx.y + my_rank * my_work;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row < height && out_col < width) {
        for (filter_row = -FILTER_RADIUS; filter_row <= FILTER_RADIUS; filter_row++) {
            for (filter_col = -FILTER_RADIUS; filter_col <= FILTER_RADIUS; filter_col++) {
                in_row = out_row + filter_row;
                in_col = out_col + filter_col;

                if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                    int input_index = in_row * width + in_col;
                    int filter_index = (filter_row + FILTER_RADIUS) * FILTER_DIM + (filter_col + FILTER_RADIUS);
                    sum += input[input_index] * filter_dev[filter_index];
                }
            }
        }
        output[out_row * width + out_col] = sum;
    }
}

void conv_host_cpu(int my_rank, int my_work, int* input, int* output, unsigned int height, unsigned int width, int *filter_cpu) {
    int out_row, out_col, sum = 0;
    int filter_row, filter_col, in_row, in_col;

    int offset = my_rank * my_work;
    int start_row = offset, end_row = offset + my_work;
    int cnt = 0;

    for (out_row = start_row; out_row < end_row; out_row++) {
        for (out_col = 0; out_col < width; out_col++) {
            sum = 0;
            for (filter_row = -FILTER_RADIUS; filter_row <= FILTER_RADIUS; filter_row++) {
                for (filter_col = -FILTER_RADIUS; filter_col <= FILTER_RADIUS; filter_col++) {
                    in_row = out_row + filter_row;
                    in_col = out_col + filter_col;

                    if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                        int input_index = in_row * width + in_col;
                        int filter_index = (filter_row + FILTER_RADIUS) * FILTER_DIM + (filter_col + FILTER_RADIUS);
                        sum += input[input_index] * filter_cpu[filter_index];
                    }
                }
            }
            output[cnt++] = sum;
        }
    }
}

int compare_cpu(int my_rank, int my_work, int *host, int *dev, int height, int width) {
    int i, j, idx, flag = 1;

    for (i = 0; i < my_work; i++) {
        for (j = 0; j < width; j++) {
            idx = i * width + j;
            if (dev[idx] != host[idx]) {
                printf("DIFFERENT: rank=%d: dev[%d][%d]=%d != host[%d][%d]=%d\n", my_rank, i, j, dev[idx], i, j, host[idx]);
                flag = 0;
                return flag;
            }
        }
    }

    return flag;
}

int conv_dev(int nprocs, int my_rank, int my_work, int *h_in_image, int *h_out_image, int height, int width, int filter_dim, int *filter_cpu) {
    int in_size = height * width;
    int out_size = in_size;
    int filter_size_bytes = filter_dim * filter_dim * sizeof(int);

    int *d_in_image = nullptr;
    int *d_out_image = nullptr;

    cudaError_t err;

    // Allocate memory for input image
    err = cudaMalloc((void**)&d_in_image, in_size * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for d_in_image on rank %d: %s\n", my_rank, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate memory for output image
    err = cudaMalloc((void**)&d_out_image, out_size * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for d_out_image on rank %d: %s\n", my_rank, cudaGetErrorString(err));
        cudaFree(d_in_image);  // Free previously allocated memory
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_in_image, h_in_image, in_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(filter_dev, filter_cpu, filter_size_bytes);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (my_work + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conv_dev_cuda<<<blocksPerGrid, threadsPerBlock>>>(my_rank, my_work, height, width, d_in_image, d_out_image);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_image, d_out_image, out_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in_image);
    cudaFree(d_out_image);

    return 0;
}

void print_lst_cpu(int name, int rank, int n, int *l) {
    int i = 0;
    printf("CPU rank=%d: %d: size=%d:: ", rank, name, n);
    for (i = 0; i < n; i++) printf("%x ", l[i]);
    printf("\n");
}

void print_filter_cpu(int name, int rank, int n, int *buf) {
    int i = 0, j;
    printf("CPU rank=%d: %d: size=%d:: ", rank, name, n);
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) printf("%x ", *buf++);
    printf("\n");
}

void init_filter(int *buf) {
    int i, j, cnt = 0;
    for (i = 0; i < FILTER_DIM; i++)
        for (j = 0; j < FILTER_DIM; j++) *buf++ = cnt++;
}