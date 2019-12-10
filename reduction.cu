#include <assert.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include<time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

#define CONST_BYTES (16*16*2)
#define SKEW_HALF 8                      //offset for avoding bank conflict
#define SHMEM_STRIDE (16+SKEW_HALF)
#define INPUT_STORE_POINT WMMA_M
#define FREE_USE (INPUT_STORE_POINT+16)

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
#define frag_c wmma::fragment<wmma::accumulator, 16, 16, 16, half>
//using frag_b = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
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

    //if(threadIdx.x==0)
      //printf("kernel complete!\n");
}

/************************
 * WARP-LEVEL REDUCTION *
 ************************/
__global__ void compute_reductions256N_warp(const half *input, float *output, int N){

  extern __shared__ half shmem[][16 + SKEW_HALF];
  half *free_use = (half*)&shmem[FREE_USE][0];
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  //const unsigned int laneId = threadIdx.x % WARP_SIZE;


  if(warpId==0){

     __syncthreads();  

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> P_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> PT_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> A_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> Vn_frag;
    wmma::fill_fragment(C_frag, 0.0f);
    wmma::fill_fragment(Vn_frag, 0.0f);
    wmma::fill_fragment(P_frag, 1.0f);
    wmma::fill_fragment(PT_frag, 1.0f);


#pragma unroll
    for(int i=0;i<N;i++){
      wmma::load_matrix_sync(A_frag, input+i*256, 16);  
      wmma::mma_sync(Vn_frag, P_frag, A_frag, Vn_frag);                 //perform Vn = P x An+Vn-1
    }

    wmma::store_matrix_sync(free_use, Vn_frag, 16, wmma::mem_row_major);//store Vn to shared memory, because as an accumulator frag Vn cannot be used for computing multiplication
    wmma::load_matrix_sync(V_frag, free_use, 16);                       //load V from Vn as a matrix_a type   
    wmma::mma_sync(C_frag, V_frag, PT_frag, C_frag);                    //perform output = V x PT 
    
    if(threadIdx.x == 0)
      *output = C_frag.x[0];
  }
}


/*************************
 * BLOCK-LEVEL REDUCTION *
 *************************/
__global__ void compute_reductions256N_block(const half *input, float *output, int N){
  extern __shared__ half shmem[][16 + SKEW_HALF];
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;
  half *free_use = (half*)&shmem[FREE_USE+warpId*16][0];//shared memory buffer start point for each warp
  float *partial_sums = (float *)&shmem[0][0];
  
  if(warpId==0){
    typedef int4 copy_t;                                  //vector pointer for fast copy
    //load P matrix to shared memory
    int shmem_row = laneId/2;
    //just remember this fixed copy approach!!
    copy_t *lane_ptr = (copy_t *)(P_d+laneId*8);          //one thread copy a int4 = 16bytes = 8 fp16
    *((copy_t *)&shmem[shmem_row][0]+laneId%2) = *lane_ptr;  
  }
  __syncthreads();                                        //如果这里不加, 其他warp就会先执行下面的语句以至于读不到shared memory里的数据
  
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> P_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> PT_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> A_frag;
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> Vn_frag;
  wmma::fill_fragment(C_frag, 0.0f);
  wmma::fill_fragment(Vn_frag, 0.0f);
  wmma::load_matrix_sync(P_frag, &shmem[0][0], SHMEM_STRIDE);
  wmma::load_matrix_sync(PT_frag, &shmem[0][0], SHMEM_STRIDE);

  unsigned int i=0;
  while(warpId+i < N){
    wmma::load_matrix_sync(A_frag, input+(warpId+i)*256, 16);
    wmma::mma_sync(Vn_frag, P_frag, A_frag, Vn_frag);                 //perform Vn = P x An+Vn-1
    i+=WARPS_PER_BLOCK;
  } 

  wmma::store_matrix_sync(free_use, Vn_frag, 16, wmma::mem_row_major);//store Vn to shared memory, because as an accumulator frag Vn cannot be used for computing multiplication
  wmma::load_matrix_sync(V_frag, free_use, 16);                       //load V from Vn as a matrix_a type
  wmma::mma_sync(C_frag, V_frag, PT_frag, C_frag);                    //perform output = V x PT
  __syncthreads();                                                    

  
  if(laneId == 0){
    partial_sums[warpId] = C_frag.x[0];
  }     
  __syncthreads();
  
  if(warpId == 0){
    float mysum = 0.0f;
    if(threadIdx.x < 16){
      mysum = partial_sums[threadIdx.x];
#pragma unroll 
      for(int offset = 8; offset > 0; offset >>= 1)
        mysum += __shfl_down_sync(0xffffffff, mysum, offset, 8);
      //printf("%f, ", mysum);
    }
    if(threadIdx.x == 0)
      *output = mysum;
  }

}

/*******************************************
 * BLOCK LEVEL REDCUTION WITH OPTIMIZATION *
 *******************************************/
__global__ void compute_reductions256N_block_opt(const half *input, float *output, int N){

  __shared__ half res_warps[16*256];
  __shared__ float partial_sums[256];

  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  //const unsigned int laneId = threadIdx.x % WARP_SIZE;
  half *res_ptr = &(res_warps[warpId*256]);

  
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


  unsigned int i=0;
#pragma unroll
  while(warpId+i < N){
    wmma::load_matrix_sync(A_frag, input+(warpId+i)*256, 16);
    wmma::mma_sync(Vn_frag, A_frag, PT_frag, Vn_frag);
    i += WARPS_PER_BLOCK;
  }

  wmma::store_matrix_sync(res_ptr, Vn_frag, 16, wmma::mem_col_major);                //store Vn to shared memory, because as an accumulator frag Vn cannot be used for computing multiplication
  __syncthreads();
  if(warpId == 0){
    wmma::load_matrix_sync(V_frag, res_ptr, 256);
    /*
    for(int i = 0;i<V_frag.num_elements;i++)
      printf("%.2f, ", __half2float(V_frag.x[i]));
    /*
    if(threadIdx.x == 0)
      for(int i = 0;i<256;i++)
        printf("%.2f, ", __half2float(res_ptr[256+i]));
    */
    wmma::mma_sync(C_frag, P_frag, V_frag, C_frag);
    wmma::store_matrix_sync(partial_sums, C_frag, 16, wmma::mem_row_major);

  //__syncthreads();
    float mysum = 0.0f;
    if(threadIdx.x < 16)
      mysum = partial_sums[threadIdx.x];
#pragma unroll 
    for(int offset = 8; offset > 0; offset >>= 1)
      mysum += __shfl_down_sync(0xffffffff, mysum, offset, 16);
      //printf("%f, ", mysum);

    if(threadIdx.x == 0)
      *output = mysum;
  }

}

/******************************************
 * BLOCK LEVEL REDUCTION WITH CUB LIBRARY *
 ******************************************/
 template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    cub::BlockReduceAlgorithm    ALGORITHM,
    class T>
__global__ void BlockSumKernel(
    T             *d_in,          // Tile of input
    float         *d_out)         // Tile aggregate
{
    // Specialize BlockReduce type for our thread block
    typedef cub::BlockReduce<T, BLOCK_THREADS, ALGORITHM> BlockReduceT;
    // Shared memory
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    // Per-thread tile data
    T data[ITEMS_PER_THREAD];
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_in, data);
    // Compute sum
    float aggregate = BlockReduceT(temp_storage).Sum(data);
    // Store aggregate
    if (threadIdx.x == 0)
    {
        *d_out = aggregate;
    }
}


/*******************************
 * GRID LEVEL REDUCTION KERNEL *
 *******************************/
template<int BLOCKS_PER_GIRD>
__global__ void compute_reductions256N_grid(half *input, float *output, int N){


  const unsigned int blockId = blockIdx.x;
  const unsigned int warpId = threadIdx.x / WARP_SIZE;

  __shared__ half res_warps[16*256];
  __shared__ float partial_sums[256];
  half *res_ptr = &(res_warps[warpId*256]);

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
  
#pragma unroll
  while(i < N){
    wmma::load_matrix_sync(A_frag, input+i*256, 16);
    wmma::mma_sync(Vn_frag, A_frag, PT_frag, Vn_frag);
    i += (WARPS_PER_BLOCK * BLOCKS_PER_GIRD);
  }

  wmma::store_matrix_sync(res_ptr, Vn_frag, 16, wmma::mem_col_major);                //store Vn to shared memory, because as an accumulator frag Vn cannot be used for computing multiplication  
  __syncthreads();

  if(warpId == 0){
    wmma::load_matrix_sync(V_frag, res_ptr, 256);
    wmma::mma_sync(C_frag, P_frag, V_frag, C_frag);
    wmma::store_matrix_sync(partial_sums, C_frag, 16, wmma::mem_row_major);

  //__syncthreads();
    float mysum = 0.0f;
    if(threadIdx.x < 16)
      mysum = partial_sums[threadIdx.x];
#pragma unroll 
    for(int offset = 8; offset > 0; offset >>= 1)
      mysum += __shfl_down_sync(0xffffffff, mysum, offset, 16);
      //printf("%f, ", mysum);

    if(threadIdx.x == 0)
      output[blockId] = mysum;
  }

}



/**********************************
 * TEST MEMORY TRANSFER SPEED *
 **********************************/
__global__ void mem_test(const half *input){
  extern __shared__ half shmem[][16 + SKEW_HALF];
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  //const unsigned int laneId = threadIdx.x % WARP_SIZE;

  if(warpId==0){

    typedef int4 copy_t;                                              //vector pointer for fast copy
    //load P matrix to shared memory
    //int shmem_row = laneId/2;
    //copy_t *lane_ptr = (copy_t *)(P_d+laneId*8);
    //copy_t *lane_ptr = (copy_t *)(input+laneId*8);                  //one thread copy a int4 = 16bytes = 8 fp16

    //constant memory ----> shared memory      
    //*((copy_t *)&shmem[shmem_row][0]+laneId%2) = *lane_ptr;

    //global memory ----> shared memory
    //*((copy_t *)&shmem[shmem_row][0]+laneId%2) = *lane_ptr;
    
    
  }  
  //directly set the shared memory
  //if(threadIdx.x < 16)
    //*(&shmem[0][0] + threadIdx.x) = __float2half(1.0f);
  //else
    //*(&shmem[0][0] + threadIdx.x) = __float2half(0.0f);

  //__syncthreads();
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> P_frag;
  //load from shared memory
  //wmma::load_matrix_sync(P_frag, &shmem[0][0], SHMEM_STRIDE);
  //directly load from global memory
  //wmma::load_matrix_sync(P_frag, input, SHMEM_STRIDE);
  //directly fill the frag
  wmma::fill_fragment(P_frag, 1.0f);

}




__host__ void init_input(half *input_half, float *input_float,int size){
  srand((int)time(0));
  for(int i=0;i<size;i++){
    input_float[i] = (float)(rand() % 100);
    //input_half[i] = __float2half(((float)(input_float[i])-1.0f+0.25f));
    input_half[i] = __float2half(((float)(input_float[i]))/100000.0f);
  }
}


 template<int BLOCKS_PER_GIRD>
 float sum_device(const half *input, const int input_size){
 
   half *input_d;
   float *res_seq;
   float *res_d;
   float res_h = 0;

   checkCudaErrors(cudaMalloc(&input_d, 2*input_size));
   checkCudaErrors(cudaMalloc(&res_d, sizeof(float)));
   checkCudaErrors(cudaMalloc(&res_seq, BLOCKS_PER_GIRD*sizeof(float)));
 
   checkCudaErrors(cudaMemcpy(input_d, input, 2*input_size, cudaMemcpyHostToDevice));
 
   checkKernelErrors( (compute_reductions256N_grid<BLOCKS_PER_GIRD><<<BLOCKS_PER_GIRD, THREADS_PER_BLOCK>>>(input_d, res_seq, input_size/256)) );
   checkCudaErrors(cudaDeviceSynchronize());
   BlockSumKernel<BLOCKS_PER_GIRD, 1, cub::BLOCK_REDUCE_RAKING, float><<<1, BLOCKS_PER_GIRD>>>(res_seq, res_d);
  
   checkCudaErrors(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));

   checkCudaErrors(cudaFree(input_d));
   checkCudaErrors(cudaFree(res_d));
   checkCudaErrors(cudaFree(res_seq));

   return res_h;
 }


int sum_wmma(half *input, int input_size){
  float res_h = 0.0;
  float *res_d;

  half *input_d;
  //malloc GPU and copy contant data to constant memory
  checkCudaErrors(cudaMalloc(&input_d, 2*input_size));
  checkCudaErrors(cudaMalloc(&res_d, sizeof(float)));
  checkCudaErrors(cudaMemcpyToSymbol(P_d, P_h, CONST_BYTES));

  //copy input to gpu
  checkCudaErrors(cudaMemcpy(input_d, input, 2*input_size, cudaMemcpyHostToDevice));

  //launch kernel
  checkKernelErrors( (compute_reductions256N_block<<<1, THREADS_PER_BLOCK, SHMEM_SIZE>>>(input_d, res_d, input_size/256)) );
  //checkKernelErrors( (compute_reductions256N_block_opt<<<1, THREADS_PER_BLOCK>>>(input_d, res_d, input_size/256)) );
  //checkKernelErrors( (compute_reductions256N_warp<<<1, 32, SHMEM_SIZE>>>(input_d, res_d, input_size/256)) );

  //checkCudaErrors(cudaDeviceSynchronize());
  //checkKernelErrors( (mem_test<<<1, THREADS_PER_BLOCK, SHMEM_SIZE>>>(input_d)) );

  //copy result back to cpu
  checkCudaErrors(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));

  //free gpu memory and return
  checkCudaErrors(cudaFree(input_d));
  checkCudaErrors(cudaFree(res_d));
  return res_h;
}

template<class T>
int sum_cub(T *input, int input_size){
  float res_h = 0;
  float *res_d;
  T *input_d;
  checkCudaErrors(cudaMalloc(&input_d, sizeof(T)*input_size));
  checkCudaErrors(cudaMalloc(&res_d, sizeof(float)));
  checkCudaErrors(cudaMemcpy(input_d, input, sizeof(T)*input_size, cudaMemcpyHostToDevice));
  BlockSumKernel<THREADS_PER_BLOCK, 1<<2, cub::BLOCK_REDUCE_RAKING, T><<<1, THREADS_PER_BLOCK>>>(input_d, res_d);
  checkCudaErrors(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(input_d));
  checkCudaErrors(cudaFree(res_d));
  return res_h;
}

int main(){

    int N = 1<<16;
    int input_size = N*256;

    half *input_h = (half*)malloc(2*input_size);
    float *input_h_cub = (float*)malloc(4*input_size);
    init_input(input_h, input_h_cub, input_size);

    //float res_cub = sum_cub<float>(input_h_cub, input_size);
    //checkCudaErrors(cudaDeviceSynchronize());
    //float res = sum_wmma(input_h, input_size);
    float res_device = sum_device<80>(input_h, input_size);
    

    float res_cpu = (half)0.0;
    for(int i=0;i<input_size;i++)
        res_cpu+=__half2float(input_h[i]);
    std::cout<<"<-----------------------computing result----------------------->"<<std::endl;
    //std::cout<<"result of reduction with tensor core: "<<res<<std::endl;
    std::cout<<"result of reduction with whole device: "<<res_device<<std::endl;
    //std::cout<<"result of reduction with CUB: "<<res_cub<<std::endl;
    std::cout<<"result of reduction with CPU: "<<res_cpu<<std::endl;
    std::cout<<std::endl<<"<-----------------------all complete----------------------->"<<std::endl;

    free(input_h);
    free(input_h_cub);
    return 0;
}

