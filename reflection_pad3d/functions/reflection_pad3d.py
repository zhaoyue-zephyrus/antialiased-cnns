from torch.autograd import Function

from .. import reflection_pad3d_cuda

class ReflectionPad3dFunction(Function):

    @staticmethod
    def forward(ctx, input, pad):
        assert isinstance(pad, tuple) or isinstance(pad, list)
        assert len(pad) == 6

        ctx.pad = pad

        ctx.pad_l, ctx.pad_r = pad[0:2]
        ctx.pad_t, ctx.pad_b = pad[2:4]
        ctx.pad_f, ctx.pad_bk = pad[4:6]
        ctx.save_for_backward(input)

        pad_l = ctx.pad_l
        pad_r = ctx.pad_r
        pad_t = ctx.pad_t
        pad_b = ctx.pad_b
        pad_f = ctx.pad_f
        pad_bk = ctx.pad_bk

        if len(input.shape) == 4:
            nplane = input.size(0)
            input_t = input.size(1)
            input_h = input.size(2)
            input_w = input.size(3)
            output = input.new_zeros((nplane, input_t + pad_f + pad_bk,
                                      input_h + pad_t + pad_b,
                                      input_w + pad_l + pad_r))
        elif len(input.shape) == 5:
            nbatch = input.size(0)
            nplane = input.size(1)
            input_t = input.size(2)
            input_h = input.size(3)
            input_w = input.size(4)
            output = input.new_zeros((nbatch, nplane,
                                      input_t + pad_f + pad_bk,
                                      input_h + pad_t + pad_b,
                                      input_w + pad_l + pad_r))
        if input.is_cuda:
            reflection_pad3d_cuda.forward(output, input, pad_l, pad_r,
                                          pad_t, pad_b, pad_f, pad_bk)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        pad_l = ctx.pad_l
        pad_r = ctx.pad_r
        pad_t = ctx.pad_t
        pad_b = ctx.pad_b
        pad_f = ctx.pad_f
        pad_bk = ctx.pad_bk
        input = ctx.saved_tensors[0]

        if len(grad_output.shape) == 4:
            nplane = grad_output.size(0)
            grad_output_t = grad_output.size(1)
            grad_output_h = grad_output.size(2)
            grad_output_w = grad_output.size(3)
            grad_input = grad_output.new_zeros((nplane, grad_output_t - pad_f - pad_bk,
                                                grad_output_h - pad_t - pad_b,
                                                grad_output_w - pad_l - pad_r))
        elif len(input.shape) == 5:
            nbatch = grad_output.size(0)
            nplane = grad_output.size(1)
            grad_output_t = grad_output.size(2)
            grad_output_h = grad_output.size(3)
            grad_output_w = grad_output.size(4)
            grad_input = grad_output.new_zeros((nbatch, nplane,
                                                grad_output_t - pad_f - pad_bk,
                                                grad_output_h - pad_t - pad_b,
                                                grad_output_w - pad_l - pad_r))
        if ctx.needs_input_grad[0]:
            reflection_pad3d_cuda.backward(grad_input, grad_output, input,
                                           pad_l, pad_r, pad_t, pad_b, pad_f, pad_bk)

        return grad_input, None

reflection_pad3d = ReflectionPad3dFunction.apply
