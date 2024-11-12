// dot_prod_cuda.h

#ifndef DOT_PROD_CUDA_H
#define DOT_PROD_CUDA_H

// Function declarations
extern "C" int dot_product_cuda(int my_rank, int my_work, int *h_A, int *h_B);
extern "C" int sum(int size, int *data);

#endif // DOT_PROD_CUDA_H
