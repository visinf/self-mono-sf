import torch
from torch.nn import Module, Parameter
from torch.autograd import Function

import forward_warp_cuda

class forward_warp_function(Function):

    @staticmethod
    def forward(ctx, im0, flow):
        '''
        im0: the first image with shape [B, C, H, W]
        flow: the optical flow with shape [B, H, W, 2] (different to grid_sample, it's range is from [-W, -H] to [W, H])
        '''
        assert(len(im0.shape) == len(flow.shape) == 4)
        assert(im0.shape[0] == flow.shape[0])
        assert(im0.shape[-2:] == flow.shape[1:3])
        assert(flow.shape[3] == 2)
        
        im0 = im0.contiguous()
        flow = flow.contiguous()
        ctx.save_for_backward(im0, flow)

        im1 = torch.zeros(im0.size(), dtype=im0.dtype, layout=im0.layout, device=im0.device)

        # with torch.cuda.device_of(im0):
        forward_warp_cuda.forward(im0, flow, im1)

        return im1

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.contiguous()
        im0, flow = ctx.saved_variables
        im0_grad = torch.zeros(im0.size(), dtype=im0.dtype, layout=im0.layout, device=im0.device)
        flow_grad = torch.zeros(flow.size(), dtype=flow.dtype, layout=flow.layout, device=flow.device)

        #with torch.cuda.device_of(im0):
        forward_warp_cuda.backward(grad_output, im0, flow, im0_grad, flow_grad)

        return im0_grad, flow_grad


class forward_warp(Module):

    def __init__(self):
        super(forward_warp, self).__init__()
        
    def forward(self, im0, flow):

        _, _, h, w = im0.size()
        flow = torch.clamp(flow, -2*w, 2*w)

        return forward_warp_function.apply(im0, flow)
