#include "block_reduction.cuh"
#include "grid_reduction.cuh"


int main(){

    int N = 1<<22;
    size_t input_size = N*256;

    half *input_h = (half*)malloc(2*input_size);
    float *input_h_cub = (float*)malloc(4*input_size);
    init_input(input_h, input_h_cub, input_size);

    //float res_cub = sum_cub<float>(input_h_cub, input_size);
    //checkCudaErrors(cudaDeviceSynchronize());
    //float res_block = sum_wmma(input_h, input_size);
    float res_device = 0.0f;
    float res_cub = 0.0f;
    //res_device = sum_hybrid<240>(input_h, input_size);
    for(int i=0; i < 100; ++i){
        res_cub = sum_cub_device(input_h, input_size);
        res_device = sum_wmma_device<640>(input_h, input_size, 0);//16*16
        res_device = sum_wmma_device<240>(input_h, input_size, 1);//32*8
        res_device = sum_wmma_device<240>(input_h, input_size, 2);//prefetch 16*16
        res_device = sum_wmma_device<160>(input_h, input_size, 3);//prefetch 32*8
    }

    float res_cpu = 0.0;
    for(size_t i=0;i<input_size;i++)
        res_cpu+=__half2float(input_h[i]);
    std::cout<<"<-----------------------computing result----------------------->"<<std::endl;
    //std::cout<<"result of reduction with tensor core-block: "<<res_block<<std::endl;
    std::cout<<"result of reduction with device: "<<res_device<<std::endl;
    std::cout<<"result of reduction with CUB: "<<res_cub<<std::endl;
    std::cout<<"result of reduction with CPU: "<<res_cpu<<std::endl;
    std::cout<<std::endl<<"<-----------------------all complete----------------------->"<<std::endl;

    free(input_h);
    free(input_h_cub);
    return 0;
}