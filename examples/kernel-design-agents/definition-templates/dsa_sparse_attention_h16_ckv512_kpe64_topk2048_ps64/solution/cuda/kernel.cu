#include <torch/extension.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>
#include <cmath>
#include <tuple>

namespace {

constexpr int64_t kNumQoHeads = 16;
constexpr int64_t kHeadDimCkv = 512;
constexpr int64_t kHeadDimKpe = 64;
constexpr int64_t kPageSize = 64;
constexpr int64_t kTopK = 2048;

inline at::Tensor ensure_contiguous(const at::Tensor& tensor) {
  return tensor.defined() ? tensor.contiguous() : tensor;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> kernel(const at::Tensor& q_nope,
                                          const at::Tensor& q_pe,
                                          const at::Tensor& ckv_cache,
                                          const at::Tensor& kpe_cache,
                                          const at::Tensor& sparse_indices,
                                          double sm_scale) {
  TORCH_CHECK(q_nope.defined(), "q_nope must be defined");
  TORCH_CHECK(q_pe.defined(), "q_pe must be defined");
  TORCH_CHECK(ckv_cache.defined(), "ckv_cache must be defined");
  TORCH_CHECK(kpe_cache.defined(), "kpe_cache must be defined");
  TORCH_CHECK(sparse_indices.defined(), "sparse_indices must be defined");

  c10::cuda::CUDAGuard device_guard(q_nope.device());

  TORCH_CHECK(q_nope.is_cuda(), "q_nope must be a CUDA tensor");
  TORCH_CHECK(q_pe.is_cuda(), "q_pe must be a CUDA tensor");
  TORCH_CHECK(ckv_cache.is_cuda(), "ckv_cache must be a CUDA tensor");
  TORCH_CHECK(kpe_cache.is_cuda(), "kpe_cache must be a CUDA tensor");
  TORCH_CHECK(sparse_indices.is_cuda(), "sparse_indices must be a CUDA tensor");

  auto q_nope_c = ensure_contiguous(q_nope);
  auto q_pe_c = ensure_contiguous(q_pe);
  auto ckv_cache_c = ensure_contiguous(ckv_cache);
  auto kpe_cache_c = ensure_contiguous(kpe_cache);
  auto sparse_indices_c = ensure_contiguous(sparse_indices);

  TORCH_CHECK(
      q_nope_c.dim() == 3 && q_nope_c.size(1) == kNumQoHeads && q_nope_c.size(2) == kHeadDimCkv,
      "q_nope must have shape [num_tokens, 16, 512]");
  TORCH_CHECK(
      q_pe_c.dim() == 3 && q_pe_c.size(0) == q_nope_c.size(0) && q_pe_c.size(1) == kNumQoHeads &&
          q_pe_c.size(2) == kHeadDimKpe,
      "q_pe must have shape [num_tokens, 16, 64]");
  TORCH_CHECK(
      ckv_cache_c.dim() == 3 && ckv_cache_c.size(1) == kPageSize && ckv_cache_c.size(2) == kHeadDimCkv,
      "ckv_cache must have shape [num_pages, 64, 512]");
  TORCH_CHECK(
      kpe_cache_c.dim() == 3 && kpe_cache_c.sizes() == torch::IntArrayRef({ckv_cache_c.size(0), kPageSize, kHeadDimKpe}),
      "kpe_cache must have shape [num_pages, 64, 64]");
  TORCH_CHECK(
      sparse_indices_c.dim() == 2 && sparse_indices_c.size(0) == q_nope_c.size(0) &&
          sparse_indices_c.size(1) == kTopK,
      "sparse_indices must have shape [num_tokens, 2048]");

  const int64_t num_tokens = q_nope_c.size(0);

  auto qn_all = q_nope_c.to(at::kFloat);
  auto qp_all = q_pe_c.to(at::kFloat);
  auto kc_all = ckv_cache_c.reshape({-1, kHeadDimCkv}).to(at::kFloat);
  auto kp_all = kpe_cache_c.reshape({-1, kHeadDimKpe}).to(at::kFloat);
  auto sparse_i32 = sparse_indices_c.to(at::kInt);

  auto output = at::zeros(
      {num_tokens, kNumQoHeads, kHeadDimCkv},
      at::TensorOptions().dtype(at::kBFloat16).device(q_nope.device()));
  auto lse = at::full(
      {num_tokens, kNumQoHeads},
      -std::numeric_limits<float>::infinity(),
      at::TensorOptions().dtype(at::kFloat).device(q_nope.device()));

  const double log2_scale = std::log(2.0);

  for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    auto indices = sparse_i32[token_idx];
    auto valid_mask = indices.ne(-1);
    auto valid_indices = indices.masked_select(valid_mask);

    if (valid_indices.numel() == 0) {
      output[token_idx].zero_();
      continue;
    }

    auto token_offsets = valid_indices.to(at::kLong);
    auto kc = kc_all.index_select(0, token_offsets);
    auto kp = kp_all.index_select(0, token_offsets);
    auto qn = qn_all[token_idx];
    auto qp = qp_all[token_idx];

    auto logits = at::matmul(qn, kc.transpose(0, 1)) + at::matmul(qp, kp.transpose(0, 1));
    auto logits_scaled = logits * sm_scale;

    lse.index_put_({token_idx}, at::logsumexp(logits_scaled, /*dim=*/1) / log2_scale);

    auto attn = at::softmax(logits_scaled, /*dim=*/1);
    auto out = at::matmul(attn, kc);
    output.index_put_({token_idx}, out.to(at::kBFloat16));
  }

  return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel", &kernel, "DSA sparse attention kernel");
}
