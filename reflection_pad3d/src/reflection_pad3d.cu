#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include <thrust/pair.h>

__device__
inline thrust::pair<int64_t, int64_t> get_index_mapping3d(
    int64_t input_dim_x, int64_t input_dim_y, int64_t input_dim_z,
    int64_t output_dim_x, int64_t output_dim_y, int64_t output_dim_z,
    int64_t pad_f, int64_t pad_bk,
    int64_t pad_t, int64_t pad_b,
    int64_t pad_l, int64_t pad_r,
    int64_t output_xyz) {
  auto input_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * input_dim_x * input_dim_y * input_dim_z;
  auto output_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * output_dim_x * output_dim_y * output_dim_z;

  auto output_x = output_xyz % output_dim_x;
  auto output_y = output_xyz / output_dim_x % output_dim_y;
  auto output_z = output_xyz / (output_dim_x * output_dim_y);

  auto i_start_x = ::max(int64_t(0), -pad_l);
  auto i_start_y = ::max(int64_t(0), -pad_t);
  auto i_start_z = ::max(int64_t(0), -pad_f);
  auto o_start_x = ::max(int64_t(0), pad_l);
  auto o_start_y = ::max(int64_t(0), pad_t);
  auto o_start_z = ::max(int64_t(0), pad_f);
  

  auto input_x = ::abs(output_x - pad_l)
                 - ::abs(output_x - (input_dim_x + pad_l - 1))
                 - output_x
                 + 2 * pad_l + input_dim_x - 1
                 - o_start_x + i_start_x;
  auto input_y = ::abs(output_y - pad_t)
                 - ::abs(output_y - (input_dim_y + pad_t - 1))
                 - output_y
                 + 2 * pad_t + input_dim_y - 1
                 - o_start_y + i_start_y;
  auto input_z = ::abs(output_z - pad_f)
                 - ::abs(output_z - (input_dim_z + pad_f - 1))
                 - output_z
                 + 2 * pad_f + input_dim_z - 1
                 - o_start_z + i_start_z;
 
  return thrust::make_pair<int64_t, int64_t>(
    input_offset
    + input_z * input_dim_y * input_dim_x
    + input_y * input_dim_x + input_x,
    output_offset
    + output_z * output_dim_y * output_dim_x
    + output_y * output_dim_x + output_x
  );
}

template <typename scalar_t>
__global__ void reflection_pad3d_out_kernel(
    scalar_t *input, scalar_t *output,
    int64_t input_dim_x, int64_t input_dim_y,
    int64_t input_dim_z,
    int pad_f, int pad_bk,
    int pad_t, int pad_b,
    int pad_l, int pad_r) {
  auto output_xyz = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_dim_x = input_dim_x + pad_l + pad_r;
  auto output_dim_y = input_dim_y + pad_t + pad_b;
  auto output_dim_z = input_dim_z + pad_f + pad_bk;

  if (output_xyz < output_dim_x * output_dim_y * output_dim_z) {
    auto index_pair = get_index_mapping3d(
      input_dim_x, input_dim_y, input_dim_z,
      output_dim_x, output_dim_y, output_dim_z,
      pad_f, pad_bk, pad_t, pad_b, pad_l, pad_r,
      output_xyz);

    output[index_pair.second] = input[index_pair.first];
  }
}

template <typename scalar_t>
__global__ void reflection_pad3d_backward_out_kernel(
    scalar_t *grad_input, scalar_t *grad_output,
    int64_t input_dim_x, int64_t input_dim_y,
    int64_t input_dim_z,
    int pad_f, int pad_bk,
    int pad_t, int pad_b,
    int pad_l, int pad_r) {
  auto output_xyz = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_dim_x = input_dim_x + pad_l + pad_r;
  auto output_dim_y = input_dim_y + pad_t + pad_b;
  auto output_dim_z = input_dim_z + pad_f + pad_bk;

  if (output_xyz < output_dim_x * output_dim_y * output_dim_z) {
    auto index_pair = get_index_mapping3d(
      input_dim_x, input_dim_y, input_dim_z,
      output_dim_x, output_dim_y, output_dim_z,
      pad_f, pad_bk, pad_t, pad_b, pad_l, pad_r,
      output_xyz);

    atomicAdd(&grad_input[index_pair.first], grad_output[index_pair.second]);
  }
}

void reflection_pad3d_out_template(
     at::Tensor &output, at::Tensor &input_,
     int64_t pad_l, int64_t pad_r,
     int64_t pad_t, int64_t pad_b,
     int64_t pad_f, int64_t pad_bk) {
  int plane_dim = 0;
  int dim_t = 1;
  int dim_h = 2;
  int dim_w = 3;
  int nbatch = 1;

  if (input_.ndimension() == 5) {
    nbatch = input_.size(0);
    plane_dim++;
    dim_t++;
    dim_h++;
    dim_w++;
  }

  int nplane = input_.size(plane_dim);
  int input_t = input_.size(dim_t);
  int input_h = input_.size(dim_h);
  int input_w = input_.size(dim_w);

  if (pad_l >= input_w || pad_r >= input_w) {
    printf("Padding size should be less than the corresponding input dimension");
    return ;
  }
  if (pad_t >= input_h || pad_b >= input_h) {
    printf("Padding size should be less than the corresponding input dimension");
    return ;
  }
  if (pad_f >= input_t || pad_bk >= input_t) {
    printf("Padding size should be less than the corresponding input dimension");
    return ;  
  }

  int output_t = input_t + pad_f + pad_bk;
  int output_h = input_h + pad_t + pad_b;
  int output_w = input_w + pad_l + pad_r;

  if (input_.ndimension() == 4){
    output.resize_({nplane, output_t, output_h, output_w});
  } else {
    output.resize_({nbatch, nplane, output_t, output_h, output_w});
  }
  at::Tensor input = input_.contiguous();

  int output_plane_size = output_t * output_h * output_w;
  dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);
  dim3 grid_size(
    (int) std::ceil(output_plane_size / 256.0), nplane, nbatch);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.type(), "reflection_pad3d_out_template", [&]{
      reflection_pad3d_out_kernel<<<
        grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
          input.data<scalar_t>(), output.data<scalar_t>(),
          input_w, input_h, input_t,
          pad_f, pad_bk, pad_t, pad_b, pad_l, pad_r);
    }
  );

}

void reflection_pad3d_backward_out_template(
     at::Tensor &grad_input, at::Tensor &grad_output_,
     at::Tensor &input,
     int64_t pad_l, int64_t pad_r,
     int64_t pad_t, int64_t pad_b,
     int64_t pad_f, int64_t pad_bk) {
  int plane_dim = 0;
  int dim_t = 1;
  int dim_h = 2;
  int dim_w = 3;
  int nbatch = 1;

  if (input.ndimension() == 5) {
    nbatch = input.size(0);
    plane_dim++;
    dim_t++;
    dim_h++;
    dim_w++;
  }

  int nplane = input.size(plane_dim);
  int input_t = input.size(dim_t);
  int input_h = input.size(dim_h);
  int input_w = input.size(dim_w);

  int output_t = input_t + pad_f + pad_bk;
  int output_h = input_h + pad_t + pad_b;
  int output_w = input_w + pad_l + pad_r;

  if (output_w != grad_output_.size(dim_w)) {
    printf("grad_output width unexpected.");
    return ;
  }
  if (output_h != grad_output_.size(dim_h)) {
    printf("grad_output height unexpected.");
    return ;
  }
  if (output_t != grad_output_.size(dim_t)) {
    printf("grad_output depth unexpected.");
    return ;
  }

  at::Tensor grad_output = grad_output_.contiguous();

  int output_plane_size = output_t * output_h * output_w;
  dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);
  dim3 grid_size(
    (int) std::ceil(output_plane_size / 256.0), nplane, nbatch);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.type(), "reflection_pad3d_backward_out_template", [&]{
      reflection_pad3d_backward_out_kernel<<<
        grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_input.data<scalar_t>(), grad_output.data<scalar_t>(),
          input_w, input_h, input_t,
          pad_f, pad_bk, pad_t, pad_b, pad_l, pad_r);
    }
  );

}
