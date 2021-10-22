#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "cuda_utils.h"

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
template<typename scalar_t>
__global__ void gather_points_kernel(int b, int c, int n, int m,
                                     const scalar_t *__restrict__ points,
                                     const int *__restrict__ idx,
                                     scalar_t *__restrict__ out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

at::Tensor gather_points_kernel_wrapper(
    int b,
    int c,
    int n,
    int npoints,
    const at::Tensor points,
    const at::Tensor idx) {
  at::Tensor out = torch::zeros(
      {points.size(0), points.size(1), idx.size(1)},
      at::device(points.device()).dtype(points.scalar_type()));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "gather_points_kernel_cuda", ([&] {
        gather_points_kernel<scalar_t><<<dim3(b, c, 1), opt_n_threads(npoints), 0, at::cuda::getCurrentCUDAStream()>>>(
          b,
          c,
          n,
          npoints,
          points.data_ptr<scalar_t>(),
          idx.data_ptr<int>(),
          out.data_ptr<scalar_t>());
  }));
  CUDA_CHECK_ERRORS();
  return out;
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
template<typename scalar_t>
__global__ void gather_points_grad_kernel(
    int b,
    int c,
    int n,
    int m,
    const scalar_t *__restrict__ grad_out,
    const int *__restrict__ idx,
    scalar_t *__restrict__ grad_points) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        atomicAdd(grad_points + (i * c + l) * n + a,
                  grad_out[(i * c + l) * m + j]);
      }
    }
  }
}

at::Tensor gather_points_grad_kernel_wrapper(
    int b,
    int c,
    int n,
    int npoints,
    const at::Tensor grad_out,
    const at::Tensor idx) {
  at::Tensor grad_points =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.scalar_type(), "gather_points_grad_cuda", ([&] {
        gather_points_grad_kernel<scalar_t><<<dim3(b, c, 1), opt_n_threads(npoints), 0, at::cuda::getCurrentCUDAStream()>>>(
          b,
          c,
          n,
          npoints,
          grad_out.data_ptr<scalar_t>(),
          idx.data_ptr<int>(),
          grad_points.data_ptr<scalar_t>());
  }));
  CUDA_CHECK_ERRORS();
  return grad_points;
}

template<typename scalar_t>
__device__ void __update(scalar_t *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const scalar_t v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idx (b, m)
template <typename scalar_t, unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b,
    int n,
    int m,
    const scalar_t *__restrict__ dataset,
    scalar_t *__restrict__ temp,
    int *__restrict__ idx) {
  if (m <= 0) return;
  __shared__ scalar_t dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  temp += batch_index * n;
  idx += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) idx[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    scalar_t best = -1;
    scalar_t x1 = dataset[old * 3 + 0];
    scalar_t y1 = dataset[old * 3 + 1];
    scalar_t z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      scalar_t x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      scalar_t mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      if (mag <= 1e-3) continue;

      scalar_t d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

      scalar_t d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 512) {
      if (tid < 256) {
        __update<scalar_t>(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update<scalar_t>(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update<scalar_t>(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update<scalar_t>(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update<scalar_t>(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update<scalar_t>(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update<scalar_t>(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update<scalar_t>(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update<scalar_t>(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    __syncthreads();
    if (tid == 0) idx[j] = old;
  }
}

at::Tensor furthest_point_sampling_kernel_wrapper(
    int b,
    int n,
    int m,
    at::Tensor points) {
  unsigned int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor idx =
      torch::zeros({points.size(0), m},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  // Setting a number close to the maximum a half can be
  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 65e3,
                  at::device(points.device()).dtype(points.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "furthest_point_sampling_cuda", ([&] {
        switch (n_threads) {
          case 512:
            furthest_point_sampling_kernel<scalar_t, 512><<<b, n_threads, 0, stream>>>(
                      b,
                      n,
                      m,
                      points.data_ptr<scalar_t>(),
                      tmp.data_ptr<scalar_t>(),
                      idx.data_ptr<int>());
            break;
          case 256:
            furthest_point_sampling_kernel<scalar_t, 256><<<b, n_threads, 0, stream>>>(
                b,
                n,
                m,
                points.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                idx.data_ptr<int>());
            break;
          case 128:
            furthest_point_sampling_kernel<scalar_t, 128><<<b, n_threads, 0, stream>>>(
                b,
                n,
                m,
                points.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                idx.data_ptr<int>());
            break;
          case 64:
            furthest_point_sampling_kernel<scalar_t, 64><<<b, n_threads, 0, stream>>>(
                b,
                n,
                m,
                points.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                idx.data_ptr<int>());
            break;
          case 32:
            furthest_point_sampling_kernel<scalar_t, 32><<<b, n_threads, 0, stream>>>(
                b,
                n,
                m,
                points.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                idx.data_ptr<int>());
            break;
          case 16:
            furthest_point_sampling_kernel<scalar_t, 16><<<b, n_threads, 0, stream>>>(
                b,
                n,
                m,
                points.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                idx.data_ptr<int>());
            break;
          case 8:
            furthest_point_sampling_kernel<scalar_t, 8><<<b, n_threads, 0, stream>>>(
                b,
                n,
                m,
                points.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                idx.data_ptr<int>());
            break;
          case 4:
            furthest_point_sampling_kernel<scalar_t, 4><<<b, n_threads, 0, stream>>>(
                b,
                n,
                m,
                points.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                idx.data_ptr<int>());
            break;
          case 2:
            furthest_point_sampling_kernel<scalar_t, 2><<<b, n_threads, 0, stream>>>(
                b,
                n,
                m,
                points.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                idx.data_ptr<int>());
            break;
          case 1:
            furthest_point_sampling_kernel<scalar_t, 1><<<b, n_threads, 0, stream>>>(
                b,
                n,
                m,
                points.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                idx.data_ptr<int>());
            break;
          default:
            furthest_point_sampling_kernel<scalar_t, 512><<<b, n_threads, 0, stream>>>(b, n, m, points.data_ptr<scalar_t>(), tmp.data_ptr<scalar_t>(), idx.data_ptr<int>());
        }
    }));
  CUDA_CHECK_ERRORS();
  return idx;
}
