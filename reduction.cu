#include <assert.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_profiler_api.h>
#include <cub/cub.cuh> 

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
#define FREE_USE INPUT_STORE_POINT+16

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

half P_h[16*16]={1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
                };



__global__ void compute_reductions16N_warp(const half *input, float *output, int N){

    extern __shared__ half shmem[][16 + SKEW_HALF];
    half *free_use = (half*)&shmem[FREE_USE][0];

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
      if(laneId < N<<1){
        lane_ptr = (copy_t *)(input+laneId*8);
        *((copy_t *)&shmem[INPUT_STORE_POINT+shmem_row][0]+laneId%2) = *lane_ptr;
      }
      else{
        *((copy_t *)&shmem[INPUT_STORE_POINT+shmem_row][0]+laneId%2) = make_int4(0,0,0,0);//padding with 0;
      }

       __syncthreads();  


      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> P_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> PT_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> A_frag;
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_frag;
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> Vn_frag;
      wmma::fill_fragment(C_frag, 0.0f);

      
      wmma::load_matrix_sync(P_frag, &shmem[0][0], SHMEM_STRIDE);
      wmma::load_matrix_sync(PT_frag, &shmem[0][0], SHMEM_STRIDE);
      //wmma::load_matrix_sync(A_frag, &shmem[INPUT_STORE_POINT][0], SHMEM_STRIDE);
      wmma::load_matrix_sync(A_frag, &shmem[INPUT_STORE_POINT][0], SHMEM_STRIDE);

      wmma::mma_sync(Vn_frag, P_frag, A_frag, C_frag);//perform V = P x A

      wmma::store_matrix_sync(free_use, Vn_frag, 16, wmma::mem_row_major);//store Vn to shared memory
      wmma::load_matrix_sync(V_frag, free_use, 16);//load V from Vn

      wmma::mma_sync(C_frag, V_frag, PT_frag, C_frag);//perform output = V x PT 
      
      wmma::store_matrix_sync(output, C_frag, 16, wmma::mem_row_major);
    }

    __syncthreads();
    if(threadIdx.x==0)
      printf("kernel complete!\n");
}

/************************
 * WARP-LEVEL REDUCTION *
 ************************/
__global__ void compute_reductions256N_warp(const half *input, float *output, int N){

  extern __shared__ half shmem[][16 + SKEW_HALF];
  half *free_use = (half*)&shmem[FREE_USE][0];
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;


  if(warpId==0){

    typedef int4 copy_t;//vector pointer for fast copy
    //load P matrix to shared memory
    int shmem_row = laneId/2;
    //just remember this fixed copy approach!!
    copy_t *lane_ptr = (copy_t *)(P_d+laneId*8);          //one thread copy a int4 = 16bytes = 8 fp16.
    *((copy_t *)&shmem[shmem_row][0]+laneId%2) = *lane_ptr;

     __syncthreads();  

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> P_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> PT_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> A_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> Vn_frag;
    wmma::fill_fragment(C_frag, 0.0f);
    //wmma::fill_fragment(Vn_frag, 0.0f);
    
    wmma::load_matrix_sync(P_frag, &shmem[0][0], SHMEM_STRIDE);
    wmma::load_matrix_sync(PT_frag, &shmem[0][0], SHMEM_STRIDE);
    //load input to frags
    for(int i=0;i<N;i++){
      wmma::load_matrix_sync(A_frag, input+i*256, 16);  
      wmma::mma_sync(Vn_frag, P_frag, A_frag, Vn_frag);//perform Vn = P x An+Vn-1
    }

    wmma::store_matrix_sync(free_use, Vn_frag, 16, wmma::mem_row_major);//store Vn to shared memory, because as an accumulator frag Vn cannot be used for computing multiplication
    wmma::load_matrix_sync(V_frag, free_use, 16);//load V from Vn as a matrix_a type 
  
    wmma::mma_sync(C_frag, V_frag, PT_frag, C_frag);//perform output = V x PT 
    
    wmma::store_matrix_sync(output, C_frag, 16, wmma::mem_row_major);

  }

  
  //if(threadIdx.x==0)
    //printf("%f kernel complete!\n", (float)input[N*256-1]);
}


/*************************
 * BLOCK-LEVEL REDUCTION *
 *************************/
__global__ void compute_reductions256N_block(const half *input, float *output, N){
  extern __shared__ half shmem[][16 + SKEW_HALF];
  half *free_use = (half*)&shmem[FREE_USE][0];
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  if(warpId==0){
    typedef int4 copy_t;//vector pointer for fast copy
    //load P matrix to shared memory
    int shmem_row = laneId/2;
    //just remember this fixed copy approach!!
    copy_t *lane_ptr = (copy_t *)(P_d+laneId*8);          //one thread copy a int4 = 16bytes = 8 fp16.
    *((copy_t *)&shmem[shmem_row][0]+laneId%2) = *lane_ptr;

     __syncthreads();
  }

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> P_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> PT_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> A_frag;
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> Vn_frag;
  wmma::fill_fragment(C_frag, 0.0f);
  //wmma::fill_fragment(Vn_frag, 0.0f);
  
  wmma::load_matrix_sync(P_frag, &shmem[0][0], SHMEM_STRIDE);
  wmma::load_matrix_sync(PT_frag, &shmem[0][0], SHMEM_STRIDE);
  unsigned int i=0;
  while(warpId+i){
    wmma::load_matrix_sync(A_frag, input+i*256, 16);
    wmma::mma_sync(Vn_frag, P_frag, A_frag, Vn_frag);//perform Vn = P x An+Vn-1
    i+=WARPS_PER_BLOCK;
  }

}




//test the memory transfer speed
__global__ void shared_to_frag(const half *input, float *output, const int N){
  extern __shared__ half shmem[][16 + SKEW_HALF];
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  if(warpId==0){
    typedef int4 copy_t;//vector pointer for fast copy
    //load P matrix to shared memory
    int shmem_row = laneId/2;
    copy_t *lane_ptr = (copy_t *)(P_d+laneId*8);       
    *((copy_t *)&shmem[shmem_row][0]+laneId%2) = *lane_ptr;
    
    __syncthreads();  
    //load input
    for(int i=0;i<100;i++){
      if(laneId < N<<1){
        copy_t *lane_ptr = (copy_t *)(input+laneId*8);
        *((copy_t *)&shmem[INPUT_STORE_POINT+shmem_row][0]+laneId%2) = *lane_ptr;
      }
      else{
        *((copy_t *)&shmem[INPUT_STORE_POINT+shmem_row][0]+laneId%2) = make_int4(0,0,0,0);//padding with 0;
      }
    }
     __syncthreads();  
     
    //wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> A_frag[32];

    //for(int i=0;i<1000;i++)
    //  wmma::load_matrix_sync(A_frag[0], &shmem[INPUT_STORE_POINT][0], 16);
    
  } 
  
}
/*
事实证明global->shared->frag和global->frag速度没什么却别
计算一次mma的时间约为load一个frag或从constant读到shared的10倍
*/





__host__ void init_input(half *input, int size){
  for(float i=0.0;i<size;i++){
    *(input+(int)i) = (half)(1.10);
  }
}


int main(){

    half *input_h;
    int N = (2<<10);
    int input_size = N*256;

    float *output_h;
    half *input_d;
    float *output_d;

    output_h = (float*)malloc(2*CONST_BYTES);
    input_h = (half*)malloc(2*input_size);
    init_input(input_h, input_size);

    //malloc GPU and copy contant data to constant memory
    checkCudaErrors(cudaMalloc(&input_d, 2*input_size));
    checkCudaErrors(cudaMalloc(&output_d, CONST_BYTES*2));
    checkCudaErrors(cudaMemcpyToSymbol(P_d, P_h, CONST_BYTES));

    //copy input to gpu
    checkCudaErrors(cudaMemcpy(input_d, input_h, 2*input_size, cudaMemcpyHostToDevice));

    //launch kernel
    //for(int i=0;i<100;i++)
    //checkKernelErrors( (compute_reductions16N_warp<<<1, THREADS_PER_BLOCK, SHMEM_SIZE>>>(input_d, output_d, 15)) );
    checkKernelErrors( (compute_reductions256N_warp<<<1, THREADS_PER_BLOCK, SHMEM_SIZE>>>(input_d, output_d, N)) );
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
   
    std::cout<<std::endl<<"all complete!"<<(float)input_h[N*256-1]<<std::endl;

    //free host&device memory
    free(output_h);
    checkCudaErrors(cudaFree(input_d));
    cudaFree((output_d));

    return 0;
}

//遗留问题 256N超过一个256就不对
//block-level reduction