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



using namespace nvcuda;
#define frag_c wmma::fragment<wmma::accumulator, 16, 16, 16, half>
//using frag_b = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;


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


//something's still wrong with this kernel:testing prefetching
template<int BLOCKS_PER_GIRD>
__global__ void compute_reductions256N_grid_opt(half *input, float *output, int N){

  const unsigned int blockId = blockIdx.x;
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;
  __shared__ half res_warps[16*256];
  __shared__ float partial_sums[256];
  
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

  unsigned int i = 2 * (WARPS_PER_BLOCK * blockId + warpId);
  

  while(i < N){
    wmma::load_matrix_sync(A_frag, input+i*256, 32);
    wmma::mma_sync(Vn_frag, A_frag, PT_frag, Vn_frag);
    i += 2 * (WARPS_PER_BLOCK * BLOCKS_PER_GIRD);                                      //now a warp compute 2*256 numbers
  }

  wmma::store_matrix_sync(res_ptr, Vn_frag, 32, wmma::mem_col_major);                //store Vn to shared memory, because as an accumulator frag Vn cannot be used for computing multiplication  
  __syncthreads();

  if(warpId == 0){
    wmma::load_matrix_sync(V_frag, res_ptr, 256);
    wmma::mma_sync(C_frag, P_frag, V_frag, C_frag);
    wmma::store_matrix_sync(partial_sums, C_frag, 32, wmma::mem_row_major);

    float mysum = 0.0f;
    if(threadIdx.x < 32)
      mysum = partial_sums[threadIdx.x];
#pragma unroll 
    for(int offset = 16; offset > 0; offset >>= 1)
      mysum += __shfl_down_sync(0xffffffff, mysum, offset, 32);
      //printf("%f, ", mysum);

    if(threadIdx.x == 0)
      //output[blockId] = mysum;
      atomicAdd(output, (float)mysum);
  }
 /*
  if(warpId != 0){                                                                          //test removing tail effect
    for(int i=0;i<1000;i++){
    wmma::load_matrix_sync(V_frag, input+(blockId*BLOCKS_PER_GIRD+warpId), 16);
    wmma::mma_sync(C_frag, P_frag, V_frag, C_frag);
    }
    //if(laneId == 0)
      //output[blockId*WARPS_PER_BLOCK+warpId] = C_frag.x[32];
  }*/
  /*
  cooperative_groups::this_grid().sync();                                            //gird level sync

  if(blockId == 0){
    __shared__ float collect[256];
    float mysum = 0;                                                                    //collect results from blocks
    collect[threadIdx.x] = threadIdx.x < BLOCKS_PER_GIRD? 1.0f:0.0f;
    mysum = collect[threadIdx.x];

#pragma unroll
    for(unsigned int offset = 128;offset>32;offset>>=1){
      if(threadIdx.x<offset){
        collect[threadIdx.x] = mysum = mysum + collect[threadIdx.x+offset];
      }
    }
    __syncthreads();
    //if(threadIdx.x == 0)
      //for(int i=0;i<256;i++)
        //printf("%.2f, ", collect[i]);

    if(warpId == 0){
      collect[threadIdx.x] += collect[threadIdx.x+32];
      mysum = collect[threadIdx.x];
      for (int offset = WARP_SIZE/2; offset > 0; offset>>=1) 
        mysum += __shfl_down_sync(0xffffffff, mysum, offset, 32);
        
      if(threadIdx.x == 0)
        output[0] = mysum;
    }
  }*/
}

template<int BLOCKS_PER_GIRD>
__global__ void compute_reductions256N_grid_prefetch16(half *input, float *output, int N){
  const unsigned int blockId = blockIdx.x;
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;
  __shared__ half res_warps[16*256];
  __shared__ float partial_sums[256]; 
  __shared__ half prefetch_buff[16 * 256 * 2];

  half *res_ptr = &(res_warps[warpId*256]);
  int4 *warp_prefetch_ptr = (int4*)(&prefetch_buff[warpId * 256]);
  int4 *lane_prefetch_ptr = warp_prefetch_ptr + laneId;
  
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> P_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> PT_frag;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> V_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> C_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> Vn_frag;
  wmma::fill_fragment(C_frag, 0.0f);
  wmma::fill_fragment(Vn_frag, 0.0f);
  wmma::fill_fragment(PT_frag, 1.0f);
  wmma::fill_fragment(P_frag, 1.0f);
  
  unsigned int i = (WARPS_PER_BLOCK * blockId + warpId);
  int sign = 0;//512 = 256 * 16 / 8, 
  int4 *lane_src_ptr = (int4 *)(input + laneId * 8);
  *(lane_prefetch_ptr + 512) = *(lane_src_ptr + i * 32);//32 = 256 / 8
  i += (WARPS_PER_BLOCK * BLOCKS_PER_GIRD);

  while(i < N){
    lane_prefetch_ptr = warp_prefetch_ptr + laneId + sign;//一个lane复制8个half
    *lane_prefetch_ptr = *(lane_src_ptr + i * 32);//32 = 256 / 8

    wmma::load_matrix_sync(A_frag, (half*)(warp_prefetch_ptr + 512 - sign), 16);//先计算+512
    wmma::mma_sync(Vn_frag, A_frag, PT_frag, Vn_frag);
    sign = 512 - sign;
    i += (WARPS_PER_BLOCK * BLOCKS_PER_GIRD);                                      //now a warp compute 2*256 numbers
  }
  wmma::load_matrix_sync(A_frag, (half*)(warp_prefetch_ptr + 512 - sign), 16);
  wmma::mma_sync(Vn_frag, A_frag, PT_frag, Vn_frag);

  wmma::store_matrix_sync(res_ptr, Vn_frag, 16, wmma::mem_col_major);                //store Vn to shared memory, because as an accumulator frag Vn cannot be used for computing multiplication  
  __syncthreads();

  if(warpId == 0){
    wmma::load_matrix_sync(V_frag, res_ptr, 256);
    wmma::mma_sync(C_frag, P_frag, V_frag, C_frag);
    wmma::store_matrix_sync(partial_sums, C_frag, 16, wmma::mem_row_major);

    float mysum = 0.0f;
    if(threadIdx.x < 16)
      mysum = partial_sums[threadIdx.x];
#pragma unroll 
    for(int offset = 8; offset > 0; offset >>= 1)
      mysum += __shfl_down_sync(0xffffffff, mysum, offset, 32);
      //printf("%f, ", mysum);

    if(threadIdx.x == 0)
      //output[blockId] = mysum;
      atomicAdd(output, (float)mysum);
  }
}

template<int BLOCKS_PER_GIRD>
__global__ void compute_reductions256N_grid_prefetch32(half *input, float *output, int N){
  const unsigned int blockId = blockIdx.x;
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;
  __shared__ half res_warps[16*256];
  __shared__ float partial_sums[256]; 
  __shared__ half prefetch_buff[16 * 512 * 2];

  half *res_ptr = &(res_warps[warpId*256]);
  int4 *warp_prefetch_ptr = (int4*)(&prefetch_buff[warpId * 512]);
  int4 *lane_prefetch_ptr = warp_prefetch_ptr + laneId;
  

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
  
  unsigned int i = 2 * (WARPS_PER_BLOCK * blockId + warpId);
  int sign = 0;//1024 = 512 * 16 / 8, 
  int4 *lane_src_ptr = (int4 *)(input + laneId * 8);
  *(lane_prefetch_ptr + 1024) = *(lane_src_ptr + i * 32);//32 = 256 / 8
  *(lane_prefetch_ptr + 1056) = *(lane_src_ptr + i * 32 + 32);//32 = 256 / 8
  i += 2 * (WARPS_PER_BLOCK * BLOCKS_PER_GIRD);

#pragma unroll
  while(i < N){
    lane_prefetch_ptr = warp_prefetch_ptr + laneId + sign;//一个lane复制16个half
    *lane_prefetch_ptr = *(lane_src_ptr + i * 32);//32 = 256 / 8
    *(lane_prefetch_ptr + 32) = *(lane_src_ptr + i * 32 + 32);//32 = 256 / 8

    wmma::load_matrix_sync(A_frag, (half*)(warp_prefetch_ptr + 1024 - sign), 32);//先计算+1024
    wmma::mma_sync(Vn_frag, A_frag, PT_frag, Vn_frag);
    sign = 1024 - sign;
    i += 2 * (WARPS_PER_BLOCK * BLOCKS_PER_GIRD);                                      //now a warp compute 2*256 numbers
  }
  wmma::load_matrix_sync(A_frag, (half*)(warp_prefetch_ptr + 1024 - sign), 32);
  wmma::mma_sync(Vn_frag, A_frag, PT_frag, Vn_frag);

  wmma::store_matrix_sync(res_ptr, Vn_frag, 32, wmma::mem_col_major);                //store Vn to shared memory, because as an accumulator frag Vn cannot be used for computing multiplication  
  __syncthreads();

  if(warpId == 0){
    wmma::load_matrix_sync(V_frag, res_ptr, 256);
    wmma::mma_sync(C_frag, P_frag, V_frag, C_frag);
    wmma::store_matrix_sync(partial_sums, C_frag, 32, wmma::mem_row_major);

    float mysum = 0.0f;
    if(threadIdx.x < 32)
      mysum = partial_sums[threadIdx.x];
#pragma unroll 
    for(int offset = 16; offset > 0; offset >>= 1)
      mysum += __shfl_down_sync(0xffffffff, mysum, offset, 32);
      //printf("%f, ", mysum);

    if(threadIdx.x == 0)
      //output[blockId] = mysum;
      atomicAdd(output, (float)mysum);
  }
}

/**********************************
 * TEST MEMORY TRANSFER SPEED *
 **********************************/
__global__ void mem_test(const half *input){
  extern __shared__ half shmem[][16 + SKEW_HALF];
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;


  //directly set the shared memory
  /*
  if(threadIdx.x < 16)
    *(&shmem[0][0] + threadIdx.x) = __float2half(1.0f);
  else
    *(&shmem[0][0] + threadIdx.x) = __float2half(0.0f);
  __syncthreads();
  */
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> P_frag;
  wmma::load_matrix_sync(P_frag, &shmem[0][0], SHMEM_STRIDE);
  

  //directly load from global memory
  //wmma::load_matrix_sync(P_frag, input, 16);

  //directly fill the frag
  //wmma::fill_fragment(P_frag, 1.0f);

}



 template<int BLOCKS_PER_GIRD>
 float sum_wmma_device(const half *input, const size_t input_size, int alg){
 
   half *input_d;
   float *res_seq;
   float *res_d;
   float res_h = 0;
   float *cub_temp = NULL;
   size_t cub_temp_byte=0;

   checkCudaErrors(cudaMalloc(&input_d, 2*input_size));
   checkCudaErrors(cudaMalloc(&res_d, sizeof(float)));
   checkCudaErrors(cudaMalloc(&res_seq, WARPS_PER_BLOCK*BLOCKS_PER_GIRD*sizeof(float)));
 
   checkCudaErrors(cudaMemcpy(input_d, input, sizeof(half)*input_size, cudaMemcpyHostToDevice));
 
  if(alg == 0)
      checkKernelErrors( (compute_reductions256N_grid<BLOCKS_PER_GIRD><<<BLOCKS_PER_GIRD, THREADS_PER_BLOCK>>>(input_d, res_seq, input_size/256)) );
  if(alg == 1)
      checkKernelErrors( (compute_reductions256N_grid_opt<BLOCKS_PER_GIRD><<<BLOCKS_PER_GIRD, THREADS_PER_BLOCK>>>(input_d, res_seq, input_size/256)) );
  if(alg == 2)
      checkKernelErrors( (compute_reductions256N_grid_prefetch16<BLOCKS_PER_GIRD><<<BLOCKS_PER_GIRD, THREADS_PER_BLOCK>>>(input_d, res_seq, input_size/256)) );
  if(alg == 3)
      checkKernelErrors( (compute_reductions256N_grid_prefetch32<BLOCKS_PER_GIRD><<<BLOCKS_PER_GIRD, THREADS_PER_BLOCK>>>(input_d, res_seq, input_size/256)) );

   //cub::DeviceReduce::Sum(cub_temp, cub_temp_byte, res_seq, res_d, BLOCKS_PER_GIRD);
   //cudaMalloc(&cub_temp, cub_temp_byte);
   //cub::DeviceReduce::Sum(cub_temp, cub_temp_byte, res_seq, res_d, BLOCKS_PER_GIRD);
   checkCudaErrors(cudaDeviceSynchronize());

   checkCudaErrors(cudaMemcpy(&res_h, res_seq, sizeof(float), cudaMemcpyDeviceToHost));

   checkCudaErrors(cudaFree(input_d));
   checkCudaErrors(cudaFree(res_d));
   checkCudaErrors(cudaFree(res_seq));
   checkCudaErrors(cudaFree(cub_temp));

   return res_h;
 }


float sum_wmma(half *input, int input_size){
  float res_h = 0.0;
  float *res_d;

  half *input_d;
  //malloc GPU and copy contant data to constant memory
  checkCudaErrors(cudaMalloc(&input_d, 2*input_size));
  checkCudaErrors(cudaMalloc(&res_d, sizeof(float)));
  //checkCudaErrors(cudaMemcpyToSymbol(P_d, P_h, CONST_BYTES));

  //copy input to gpu
  checkCudaErrors(cudaMemcpy(input_d, input, 2*input_size, cudaMemcpyHostToDevice));

  //launch kernel
  //checkKernelErrors( (compute_reductions256N_block<<<1, THREADS_PER_BLOCK, SHMEM_SIZE>>>(input_d, res_d, input_size/256)) );
  checkKernelErrors( (compute_reductions256N_block_opt<<<1, THREADS_PER_BLOCK>>>(input_d, res_d, input_size/256)) );
  //checkKernelErrors( (compute_reductions256N_block_opt2<<<1, THREADS_PER_BLOCK>>>(input_d, res_d, input_size/256)) );
  //checkKernelErrors( (compute_reductions256N_warp<<<1, 32, SHMEM_SIZE>>>(input_d, res_d, input_size/256)) );

  checkCudaErrors(cudaDeviceSynchronize());

  //copy result back to cpu
  checkCudaErrors(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));

  //free gpu memory and return
  checkCudaErrors(cudaFree(input_d));
  checkCudaErrors(cudaFree(res_d));
  return res_h;
}

template<class T>
int sum_cub(T *input, size_t input_size){
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

float sum_cub_device(const half *input, const size_t input_size){
  half *input_d;
  float *res_d;
  float res_h = 0;
  float *cub_temp = NULL;
  size_t cub_temp_byte=0;

  checkCudaErrors(cudaMalloc(&input_d, sizeof(half)*input_size));
  checkCudaErrors(cudaMalloc(&res_d, sizeof(float)));
  checkCudaErrors(cudaMemcpy(input_d, input, 2*input_size, cudaMemcpyHostToDevice));
  cub::DeviceReduce::Sum(cub_temp, cub_temp_byte, input_d, res_d, input_size);//default stream
  cudaMalloc(&cub_temp, cub_temp_byte);
  cub::DeviceReduce::Sum(cub_temp, cub_temp_byte, input_d, res_d, input_size);
  checkCudaErrors(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(input_d));
  checkCudaErrors(cudaFree(res_d));
  checkCudaErrors(cudaFree(cub_temp));

  return res_h;
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
  float *cub_temp = NULL;
  size_t cub_temp_byte = 0;
  float *cub_temp1 = NULL;
  size_t cub_temp_byte1 = 0;

  checkCudaErrors(cudaMalloc(&input_d, 2*input_size));
  checkCudaErrors(cudaMalloc(&res_d, 2*sizeof(float)));
  checkCudaErrors(cudaMalloc(&res_seq, BLOCKS_PER_GIRD*sizeof(float)));

  checkCudaErrors(cudaMemcpy(input_d, input, 2*input_size, cudaMemcpyHostToDevice));

  checkKernelErrors( (compute_reductions256N_grid<BLOCKS_PER_GIRD><<<BLOCKS_PER_GIRD, THREADS_PER_BLOCK, 0, stream1>>>(input_d, res_seq, (input_size/2)/256)) );
  cub::DeviceReduce::Sum(cub_temp1, cub_temp_byte1, input_d+(input_size/2), res_d+1, (input_size/2), stream2);
  cub::DeviceReduce::Sum(cub_temp, cub_temp_byte, res_seq, res_d, BLOCKS_PER_GIRD, stream1);
  cudaMalloc(&cub_temp, cub_temp_byte);
  cub::DeviceReduce::Sum(cub_temp, cub_temp_byte, res_seq, res_d, BLOCKS_PER_GIRD, stream1);
  //BlockSumKernel<BLOCKS_PER_GIRD, 1, cub::BLOCK_REDUCE_RAKING, float><<<1, BLOCKS_PER_GIRD, 0, stream1>>>(res_seq, res_d);
  
  //cudaMalloc(&cub_temp1, cub_temp_byte1);
  //cub::DeviceReduce::Sum(cub_temp1, cub_temp_byte1, input_d+(input_size/2), res_d+1, (input_size/2), stream2);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&res_h, res_d, 2*sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(input_d));
  checkCudaErrors(cudaFree(res_d));
  checkCudaErrors(cudaFree(cub_temp));
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


