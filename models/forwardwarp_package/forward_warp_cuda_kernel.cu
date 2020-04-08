#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

// Define CUDA_NUM_THREAS and GET_BLOCKS
const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N){ return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;}

// Define CUDA_KERNEL_LOOP
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)



static __forceinline__ __device__ 
int get_im_index(
        const int bb,
        const int cc,
        const int hh,
        const int ww,
        const size_t C,
        const size_t H,
        const size_t W) {
    return bb*C*H*W + cc*H*W + hh*W + ww;
}

template <typename scalar_t>
__global__ void forward_warp_cuda_forward_kernel(
    const int total_step,
    const scalar_t* im0,
    const scalar_t* flow,
    scalar_t* im1,
    const int B,
    const int C,
    const int H,
    const int W) {
    
    CUDA_KERNEL_LOOP(index, total_step-1) {
        const int bb = index / (H * W);
        const int hh = (index - bb*H*W) / W;
        const int ww = index % W;
        const scalar_t x = (scalar_t)ww + flow[index * 2 + 0];
        const scalar_t y = (scalar_t)hh + flow[index * 2 + 1];
        const int x_f = static_cast<int>(::floor(x));
        const int y_f = static_cast<int>(::floor(y));
        const int x_c = x_f + 1;
        const int y_c = y_f + 1;

        if(x_f>=0 && x_c<W && y_f>=0 && y_c<H){
            const scalar_t nw_k = (x_c - x) * (y_c - y);
            const scalar_t ne_k = (x - x_f) * (y_c - y);
            const scalar_t sw_k = (x_c - x) * (y - y_f);
            const scalar_t se_k = (x - x_f) * (y - y_f);
            const scalar_t* im0_p = im0 + get_im_index(bb, 0, hh, ww, C, H, W);
            scalar_t* im1_p = im1 + get_im_index(bb, 0, y_f, x_f, C, H, W);
            for (int cc = 0; cc < C; ++cc, im0_p += H*W, im1_p += H*W){
                atomicAdd(im1_p,     nw_k*(*im0_p));
                atomicAdd(im1_p+1,   ne_k*(*im0_p));
                atomicAdd(im1_p+W,   sw_k*(*im0_p));
                atomicAdd(im1_p+W+1, se_k*(*im0_p));
            }
        }   
    }
}

template <typename scalar_t>
__global__ void forward_warp_cuda_backward_kernel(
    const int total_step,
    const scalar_t* grad_output,
    const scalar_t* im0,
    const scalar_t* flow,
    scalar_t* im0_grad,
    scalar_t* flow_grad,
    const int B,
    const int C,
    const int H,
    const int W) {

    CUDA_KERNEL_LOOP(index, total_step) {
        const int bb = index / (H * W);
        const int hh = (index-bb*H*W) / W;
        const int ww = index % W;
        const scalar_t x = (scalar_t)ww + flow[index * 2 + 0];
        const scalar_t y = (scalar_t)hh + flow[index * 2 + 1];

        const int x_f = static_cast<int>(::floor(x));
        const int y_f = static_cast<int>(::floor(y));
        const int x_c = x_f + 1;
        const int y_c = y_f + 1;

        if(x_f>=0 && x_c<W && y_f>=0 && y_c<H){

            const scalar_t nw_k = (x_c - x) * (y_c - y);
            const scalar_t sw_k = (x_c - x) * (y - y_f);
            const scalar_t ne_k = (x - x_f) * (y_c - y);
            const scalar_t se_k = (x - x_f) * (y - y_f);
            scalar_t flow_grad_x = 0;
            scalar_t flow_grad_y = 0;
            scalar_t* im0_grad_p = im0_grad + get_im_index(bb, 0, hh, ww, C, H, W);
            for (int cc = 0; cc < C; ++cc, im0_grad_p += H*W){
                const scalar_t nw_grad = grad_output[get_im_index(bb, cc, y_f, x_f, C, H, W)];
                const scalar_t ne_grad = grad_output[get_im_index(bb, cc, y_f, x_c, C, H, W)];
                const scalar_t sw_grad = grad_output[get_im_index(bb, cc, y_c, x_f, C, H, W)];
                const scalar_t se_grad = grad_output[get_im_index(bb, cc, y_c, x_c, C, H, W)];
                const scalar_t p = im0[get_im_index(bb, cc, hh, ww, C, H, W)];
                atomicAdd(im0_grad_p, nw_k*nw_grad);
                atomicAdd(im0_grad_p, ne_k*ne_grad);
                atomicAdd(im0_grad_p, sw_k*sw_grad);
                atomicAdd(im0_grad_p, se_k*se_grad);
                flow_grad_x -= (y_c-y)*p*nw_grad;
                flow_grad_y -= (x_c-x)*p*nw_grad;
                flow_grad_x += (y_c-y)*p*ne_grad;
                flow_grad_y -= (x-x_f)*p*ne_grad;
                flow_grad_x -= (y-y_f)*p*sw_grad;
                flow_grad_y += (x_c-x)*p*sw_grad;
                flow_grad_x += (y-y_f)*p*se_grad;
                flow_grad_y += (x-x_f)*p*se_grad;
            }
            flow_grad[index*2 + 0] = flow_grad_x;
            flow_grad[index*2 + 1] = flow_grad_y;
        }     
    }
}

int forward_warp_cuda_forward(
    const at::Tensor& im0, 
    const at::Tensor& flow,
    at::Tensor& im1) {
    
    const int B = im0.size(0);
    const int C = im0.size(1);
    const int H = im0.size(2);
    const int W = im0.size(3);
    const int total_step = B * H * W;
    
    AT_DISPATCH_FLOATING_TYPES(im0.scalar_type(), "forward_warp_forward_cuda", ([&] {
    forward_warp_cuda_forward_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
        total_step,
        im0.data<scalar_t>(),
        flow.data<scalar_t>(),
        im1.data<scalar_t>(),
        B, C, H, W);
    }));

    cudaError_t err = cudaGetLastError();

    // check for errors
    if (err != cudaSuccess) {
        printf("error in Forwardwarp : forward_cuda_kernel: %s\n", cudaGetErrorString(err));
        return 0;
    }

    return 1;
}

int forward_warp_cuda_backward(
    const at::Tensor& grad_output,
    const at::Tensor& im0, 
    const at::Tensor& flow, 
    at::Tensor& im0_grad, 
    at::Tensor& flow_grad) {

    const int B = im0.size(0);
    const int C = im0.size(1);
    const int H = im0.size(2);
    const int W = im0.size(3);
    const int total_step = B * H * W;

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "forward_warp_backward_cuda", ([&] {
    forward_warp_cuda_backward_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
        total_step,
        grad_output.data<scalar_t>(),
        im0.data<scalar_t>(),
        flow.data<scalar_t>(),
        im0_grad.data<scalar_t>(),
        flow_grad.data<scalar_t>(),
        B, C, H, W);
    }));

    cudaError_t err = cudaGetLastError();

    // check for errors
    if (err != cudaSuccess) {
        printf("error in Forwardwarp : forward_cuda_kernel: %s\n", cudaGetErrorString(err));
        return 0;
    }

    return 1;
}
