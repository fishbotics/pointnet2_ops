#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "cuda_utils.h"

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
template<typename scalar_t>
__global__ void group_points_kernel(int b, int c, int n, int npoints,
                                    int nsample,
                                    const scalar_t *__restrict__ points,
                                    const int *__restrict__ idx,
                                    scalar_t *__restrict__ out) {
  int batch_index = blockIdx.x;
  points += batch_index * n * c;
  idx += batch_index * npoints * nsample;
  out += batch_index * npoints * nsample * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      out[(l * npoints + j) * nsample + k] = points[l * n + ii];
    }
  }
}

at::Tensor group_points_kernel_wrapper(
    int b,
    int c,
    int n,
    int npoints,
    int nsample,
    const at::Tensor points,
    const at::Tensor idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor out = torch::zeros(
    {points.size(0), points.size(1), idx.size(1), idx.size(2)},
    at::device(points.device()).dtype(points.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    points.scalar_type(), "group_points_cuda", ([&] {
      group_points_kernel<scalar_t><<<b, opt_block_config(npoints, c), 0, stream>>>(
          b,
          c,
          n,
          npoints,
          nsample,
          points.data_ptr<scalar_t>(),
          idx.data_ptr<int>(),
          out.data_ptr<scalar_t>());
  }));
  CUDA_CHECK_ERRORS();
  return out;
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
template<typename scalar_t>
__global__ void group_points_grad_kernel(
    int b,
    int c,
    int n,
    int npoints,
    int nsample,
    const scalar_t *__restrict__ grad_out,
    const int *__restrict__ idx,
    scalar_t *__restrict__ grad_points) {
  int batch_index = blockIdx.x;
  grad_out += batch_index * npoints * nsample * c;
  idx += batch_index * npoints * nsample;
  grad_points += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      atomicAdd(grad_points + l * n + ii,
                grad_out[(l * npoints + j) * nsample + k]);
    }
  }
}

at::Tensor group_points_grad_kernel_wrapper(
    int b,
    int c,
    int n,
    int npoints,
    int nsample,
    const at::Tensor grad_out,
    const at::Tensor idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor grad_points = torch::zeros(
      {grad_out.size(0), grad_out.size(1), n},
      at::device(grad_out.device()).dtype(grad_out.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_out.scalar_type(), "group_points_grad_cuda", ([&] {
      group_points_grad_kernel<scalar_t><<<b, opt_block_config(npoints, c), 0, stream>>>(
        b,
        c,
        n,
        npoints,
        nsample,
        grad_out.data_ptr<scalar_t>(),
        idx.data_ptr<int>(),
        grad_points.data_ptr<scalar_t>());
  }));
  CUDA_CHECK_ERRORS();
  return grad_points;
}
