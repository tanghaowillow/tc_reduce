#include <assert.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include<time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cuda_profiler_api.h>
#include <cub/cub.cuh> //the CUDA unbound library unbrella head file

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


#define SHMEM_SIZE 32*1024
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 16
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure\nError: " << cudaGetErrorString(status); \
      std::stringstream _where, _message;                                \
      _where << __FILE__ << ':' << __LINE__;                             \
      _message << _error.str() + "\n" << __FILE__ << ':' << __LINE__;\
      std::cerr << _message.str() << "\nAborting...\n";                  \
      cudaDeviceReset();                                                 \
      exit(EXIT_FAILURE);                                                \
    }                                                                  \
}

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)



using namespace nvcuda;

template<int BLOCKS_PER_GIRD>
__global__ void compute_reductions256N_grid(half *input, float *output, size_t N){


  const unsigned int blockId = blockIdx.x;
  const unsigned int warpId = threadIdx.x / WARP_SIZE;



  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> P_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> PT_frag;
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> A_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> Vn_frag;
  wmma::fill_fragment(C_frag, 0.0f);
  wmma::fill_fragment(Vn_frag, 0.0f);
  wmma::fill_fragment(PT_frag, 1.0f);
  wmma::fill_fragment(P_frag, 1.0f);


  unsigned int i=WARPS_PER_BLOCK * blockId+warpId;
  wmma::load_matrix_sync(A_frag, input, 16);
#pragma unroll
  //if(warpId ==0){
    while(i < N){
      //wmma::load_matrix_sync(A_frag, input, 16);
      wmma::mma_sync(Vn_frag, A_frag, PT_frag, Vn_frag);
      //i += (WARPS_PER_BLOCK * BLOCKS_PER_GIRD);
      i++;
    }
  //}
  
}



template<int BLOCKS_PER_GIRD>
 float sum_hybrid(const half *input, const size_t input_size){
  cudaStream_t stream1, stream2;
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);


  half *input_d;
  float *res_seq;
  float *res_d;
  float res_h[2] = {0,0};
  //float *cub_temp = NULL;
  //size_t cub_temp_byte = 0;
  float *cub_temp1 = NULL;
  size_t cub_temp_byte1 = 0;

  checkCudaErrors(cudaMalloc(&input_d, 2*input_size));
  checkCudaErrors(cudaMalloc(&res_d, 2*sizeof(float)));
  checkCudaErrors(cudaMalloc(&res_seq, BLOCKS_PER_GIRD*sizeof(float)));

  checkCudaErrors(cudaMemcpy(input_d, input, 2*input_size, cudaMemcpyHostToDevice));

  checkKernelErrors( (compute_reductions256N_grid<BLOCKS_PER_GIRD><<<BLOCKS_PER_GIRD, THREADS_PER_BLOCK, 0, stream1>>>(input_d, res_seq, 1<<24 ) ));
  checkKernelErrors( (compute_reductions256N_grid<BLOCKS_PER_GIRD><<<BLOCKS_PER_GIRD, THREADS_PER_BLOCK, 0, stream1>>>(input_d, res_seq, 1<<24 ) ));
  cub::DeviceReduce::Sum(cub_temp1, cub_temp_byte1, input_d+(input_size/2), res_d+1, (input_size/2), stream2);
  cudaMalloc(&cub_temp1, cub_temp_byte1);
  cub::DeviceReduce::Sum(cub_temp1, cub_temp_byte1, input_d+(input_size/2), res_d+1, (input_size/2), stream2);
  


  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&res_h, res_d, 2*sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(input_d));
  checkCudaErrors(cudaFree(res_d));
  checkCudaErrors(cudaFree(cub_temp1));
  checkCudaErrors(cudaFree(res_seq));

  return res_h[0]+res_h[1];
 }

__host__ void init_input(half *input_half, float *input_float,size_t size){
  srand((int)time(0));
  for(int i=0;i<size;i++){
    input_float[i] = (float)(rand() % 100);
    //input_half[i] = __float2half(((float)(input_float[i])-1.0f+0.25f));
    input_half[i] = __float2half(((float)(input_float[i]))/100000.0f);
  }
}

int main(){ 

  int N = 1<<21;
  size_t input_size = N*256;

  half *input_h = (half*)malloc(2*input_size);
  float *input_h_cub = (float*)malloc(4*input_size);
  init_input(input_h, input_h_cub, input_size);


  sum_hybrid<40>(input_h, input_size);
  
  free(input_h);
  free(input_h_cub);
  return 0;
}