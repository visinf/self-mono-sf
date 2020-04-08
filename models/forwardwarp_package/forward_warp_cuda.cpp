// #include <torch/torch.h>
#include <torch/extension.h>

int forward_warp_cuda_forward(const at::Tensor& im0, const at::Tensor& flow, at::Tensor& im1);
int forward_warp_cuda_backward(const at::Tensor& grad_output, const at::Tensor& im0, const at::Tensor& flow, at::Tensor& im0_grad, at::Tensor& flow_grad);

// Because of the incompatible of Pytorch 1.0 && Pytorch 0.4, we have to annotation this.
#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int forward_warp_forward(const at::Tensor& im0, const at::Tensor& flow, at::Tensor& im1){
    CHECK_INPUT(im0);
    CHECK_INPUT(flow);

	// im1.resize_({im0.size(0), im0.size(1), im0.size(2), im0.size(3)});
	// im1.fill_(0);
    int success = forward_warp_cuda_forward(im0, flow, im1);

	if (!success) {
		AT_ERROR("CUDA call failed");
	}
	return 1;
}

int forward_warp_backward(const at::Tensor& grad_output, const at::Tensor& im0, const at::Tensor& flow, at::Tensor& im0_grad, at::Tensor& flow_grad){
    CHECK_INPUT(grad_output);
    CHECK_INPUT(im0);
    CHECK_INPUT(flow);

 	// im0_grad.resize_({im0.size(0), im0.size(1), im0.size(2), im0.size(3)});
 	// flow_grad.resize_({flow.size(0), flow.size(1), flow.size(2), flow.size(3)});
	// im0_grad.fill_(0);
	// flow_grad.fill_(0);

    int success = forward_warp_cuda_backward(grad_output, im0, flow, im0_grad, flow_grad);

    if (!success) {
		AT_ERROR("CUDA call failed");
	}
	return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &forward_warp_forward, "forward warp forward (CUDA)");
    m.def("backward", &forward_warp_backward, "forward warp backward (CUDA)");
}
