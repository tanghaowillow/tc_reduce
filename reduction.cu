#include <assert.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>


#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define SHMEM_SIZE 32*1024
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define CONST_BYTES 16*16*2
#define SKEW_HALF 8                      //offset for avoding bank conflict
#define SHMEM_STRIDE 16+SKEW_HALF
#define INPUT_STORE_POINT WMMA_M

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

// malloc gpu constant memory
__constant__  half P_d[16*16];
//__constant__  half PT_d[16*16];

half P_h[16*16]={1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                };



__global__ void compute_reductions16N_warp(const half *input, float *output, int input_size){

     extern __shared__ half shmem[][16 + SKEW_HALF];

    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

 

    if(warpId==0){

      typedef int4 copy_t;//vector pointer for fast copy
      //load P matrix to shared memory
      int shmem_row = laneId/2;
      //just remember this fixed copy approach!!
      copy_t *lane_ptr = (copy_t *)(P_d+laneId*8);          //one thread copy a int4 = 16bytes = 8 fp16.
      *((copy_t *)&shmem[shmem_row][0]+laneId%2) = *lane_ptr;

      //load input
      lane_ptr = (copy_t *)(input+laneId*8);
      *((copy_t *)&shmem[INPUT_STORE_POINT+shmem_row][0]+laneId%2) = *lane_ptr;
       __syncthreads();  


      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> P_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> PT_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> A_frag;
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
      wmma::fill_fragment(C_frag, 0.0f);
    

      
      wmma::load_matrix_sync(P_frag, &shmem[0][0], SHMEM_STRIDE);
      wmma::load_matrix_sync(PT_frag, &shmem[0][0], SHMEM_STRIDE);
      wmma::load_matrix_sync(A_frag, &shmem[INPUT_STORE_POINT][0], SHMEM_STRIDE);

      wmma::mma_sync(C_frag, P_frag, A_frag, C_frag);
      
      wmma::store_matrix_sync(output, C_frag, 16, wmma::mem_row_major);
      
      
    }




    //printf("%f ,", (float)P_d[threadIdx.x]);
    __syncthreads();
    if(threadIdx.x==0)
      printf("kernel complete!\n");
}



__host__ void init_input(half *input){
  for(float i=0.0;i<256;i++){
    *(input+(int)i) = (half)i;
  }
}


int main(){

    half *input_h;
    int input_size = 256;
    float *output_h;
    half *input_d;
    float *output_d;

    output_h = (float*)malloc(2*CONST_BYTES);
    input_h = (half*)malloc(2*CONST_BYTES);
    init_input(input_h);
    //malloc GPU and copy contant data to constant memory
    checkCudaErrors(cudaMalloc(&input_d, CONST_BYTES));
    checkCudaErrors(cudaMalloc(&output_d, CONST_BYTES*2));
    checkCudaErrors(cudaMemcpyToSymbol(P_d, P_h, CONST_BYTES));

    //copy input to gpu
    checkCudaErrors(cudaMemcpy(input_d, input_h, CONST_BYTES, cudaMemcpyHostToDevice));

    //launch kernel 
    checkKernelErrors( (compute_reductions16N_warp<<<1, THREADS_PER_BLOCK, SHMEM_SIZE>>>(input_d, output_d, input_size)) );
    checkCudaErrors(cudaDeviceSynchronize());

    //copy result back to cpu
    checkCudaErrors(cudaMemcpy(output_h, output_d, 2*CONST_BYTES, cudaMemcpyDeviceToHost) );

    //check the computing result
    for(int i=0;i<16;++i){
        for(int j=0;j<16;++j){
            std::cout<<output_h[16*i+j]<<",";
        }
        std::cout<<std::endl;
    }
    

    std::cout<<std::endl<<"all complete!"<<std::endl;



    //free host&device memory
    free(output_h);
    checkCudaErrors(cudaFree(input_d));
    cudaFree((output_d));

    return 0;
}