#include <torch/extension.h>

void reflection_pad3d_out_template(at::Tensor &output,
                                   at::Tensor &input_,
                                   int64_t pad_l, int64_t pad_r,
                                   int64_t pad_t, int64_t pad_b,
                                   int64_t pad_f, int64_t pad_bk);

void reflection_pad3d_backward_out_template(at::Tensor &grad_input,
                                            at::Tensor &grad_output_,
                                            at::Tensor &input,
                                            int64_t pad_l, int64_t pad_r,
                                            int64_t pad_t, int64_t pad_b,
                                            int64_t pad_f, int64_t pad_bk);

int reflection_pad3d_forward_cuda(at::Tensor &output,
    at::Tensor &input,
    int64_t pad_l, int64_t pad_r,
    int64_t pad_t, int64_t pad_b,
    int64_t pad_f, int64_t pad_bk) {
  reflection_pad3d_out_template(output, input,
                                pad_l, pad_r,
                                pad_t, pad_b,
                                pad_f, pad_bk);
  return 1;
}

int reflection_pad3d_backward_cuda(at::Tensor &grad_input,
    at::Tensor &grad_output,
    at::Tensor &input,
    int64_t pad_l, int64_t pad_r,
    int64_t pad_t, int64_t pad_b,
    int64_t pad_f, int64_t pad_bk) {
  reflection_pad3d_backward_out_template(
    grad_input, grad_output, input,
    pad_l, pad_r,
    pad_t, pad_b,
    pad_f, pad_bk);
  return 1;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &reflection_pad3d_forward_cuda, "Reflection_Pad forward (CUDA)");
  m.def("backward", &reflection_pad3d_backward_cuda, "Reflection_Pad backward (CUDA)");
}
