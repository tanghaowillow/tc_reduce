#pragma once
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
#include "Error.h"

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


using namespace nvcuda;

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

/*******************************************
 * BLOCK LEVEL REDCUTION WITH OPTIMIZATION2 *
 *******************************************/
 __global__ void compute_reductions256N_block_opt2(const half *input, float *output, int N){

  __shared__ half res_warps[16*256];
  __shared__ float partial_sums[256];

  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  //const unsigned int laneId = threadIdx.x % WARP_SIZE;
  half *res_ptr = &(res_warps[warpId*256]);

  
  wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> P_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> PT_frag;
  wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::col_major> A_frag;
  wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> V_frag;
  wmma::fragment<wmma::accumulator, 8, 32, 16, float> C_frag;
  wmma::fragment<wmma::accumulator, 32, 8, 16, half> Vn_frag;
  wmma::fill_fragment(C_frag, 0.0f);
  wmma::fill_fragment(Vn_frag, 0.0f);
  wmma::fill_fragment(PT_frag, 1.0f);
  wmma::fill_fragment(P_frag, 1.0f);


  unsigned int i=0;
#pragma unroll
  while(warpId+i < N){
    wmma::load_matrix_sync(A_frag, input+(warpId+i)*256, 16);
    wmma::mma_sync(Vn_frag, A_frag, PT_frag, Vn_frag);
    i += WARPS_PER_BLOCK*2;
  }

  wmma::store_matrix_sync(res_ptr, Vn_frag, 32, wmma::mem_col_major);                //store Vn to shared memory, because as an accumulator frag Vn cannot be used for computing multiplication  
  __syncthreads();

  if(warpId == 0){
    wmma::load_matrix_sync(V_frag, res_ptr, 256);
    wmma::mma_sync(C_frag, P_frag, V_frag, C_frag);
    wmma::store_matrix_sync(partial_sums, C_frag, 32, wmma::mem_row_major);

  //__syncthreads();
    float mysum = 0.0f;
    if(threadIdx.x < 32)
      mysum = partial_sums[threadIdx.x];
#pragma unroll 
    for(int offset = 16; offset > 0; offset >>= 1)
      mysum += __shfl_down_sync(0xffffffff, mysum, offset, 32);
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