#include <torch/extension.h>

#include <algorithm>
#include <vector>
#include <c10/cuda/CUDAGuard.h>

namespace {

using torch::indexing::Slice;

torch::Tensor dequantize_k_cache(const torch::Tensor& k_index_cache_fp8) {
  auto k_uint8 = k_index_cache_fp8.contiguous().view(torch::kUInt8);
  const auto num_pages = k_uint8.size(0);
  const auto page_size = k_uint8.size(1);
  const auto head_dim_with_scale = k_uint8.size(3);
  const auto head_dim = head_dim_with_scale - 4;

  auto flat = k_uint8.view({num_pages, page_size * head_dim_with_scale});
  auto fp8_bytes = flat.index({Slice(), Slice(0, page_size * head_dim)}).contiguous();
  auto scale_bytes =
      flat.index({Slice(), Slice(page_size * head_dim, page_size * head_dim_with_scale)}).contiguous();

  auto fp8 = fp8_bytes.view({num_pages, page_size, head_dim}).to(torch::kFloat8_e4m3fn);
  auto fp8_f32 = fp8.to(torch::kFloat32);
  auto scale = scale_bytes.view({num_pages, page_size, 4}).view(torch::kFloat32);
  return fp8_f32 * scale;
}

torch::Tensor kernel(torch::Tensor q_index_fp8,
                     torch::Tensor k_index_cache_fp8,
                     torch::Tensor weights,
                     torch::Tensor seq_lens,
                     torch::Tensor block_table) {
  c10::cuda::CUDAGuard device_guard(q_index_fp8.device());

  TORCH_CHECK(q_index_fp8.is_cuda(), "q_index_fp8 must be CUDA");
  TORCH_CHECK(k_index_cache_fp8.is_cuda(), "k_index_cache_fp8 must be CUDA");
  TORCH_CHECK(weights.is_cuda(), "weights must be CUDA");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be CUDA");
  TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA");

  const auto batch_size = q_index_fp8.size(0);
  const auto num_index_heads = q_index_fp8.size(1);
  const auto index_head_dim = q_index_fp8.size(2);
  const auto topk = int64_t{2048};
  const auto page_size = k_index_cache_fp8.size(1);

  TORCH_CHECK(num_index_heads == 64, "expected 64 index heads");
  TORCH_CHECK(index_head_dim == 128, "expected 128-dim heads");
  TORCH_CHECK(page_size == 64, "expected page size 64");

  auto q = q_index_fp8.to(torch::kFloat32);
  auto k_all = dequantize_k_cache(k_index_cache_fp8);
  auto w = weights.to(torch::kFloat32);
  auto output = torch::full(
      {batch_size, topk},
      -1,
      torch::TensorOptions().dtype(torch::kInt32).device(q_index_fp8.device()));

  for (int64_t b = 0; b < batch_size; ++b) {
    const auto seq_len = seq_lens[b].item<int32_t>();
    if (seq_len <= 0) {
      continue;
    }

    const auto num_pages_for_seq = (seq_len + page_size - 1) / page_size;
    auto page_indices = block_table.index({b, Slice(0, num_pages_for_seq)}).to(torch::kInt64);
    auto k_paged = k_all.index_select(0, page_indices);
    auto k_tokens = k_paged.reshape({-1, index_head_dim}).narrow(0, 0, seq_len);
    auto q_b = q[b];

    auto scores = torch::matmul(q_b, k_tokens.transpose(0, 1));
    auto scores_relu = torch::relu(scores);
    auto final_scores = (scores_relu * w[b].unsqueeze(1)).sum(/*dim=*/0);

    const auto actual_topk = std::min<int64_t>(topk, seq_len);
    auto token_offsets = torch::arange(
        page_size, torch::TensorOptions().dtype(torch::kInt64).device(q_index_fp8.device()));
    auto token_pages = page_indices.unsqueeze(-1) * page_size + token_offsets;
    auto token_indices = token_pages.reshape({-1}).narrow(0, 0, seq_len);
    auto tie_break = token_indices.to(torch::kFloat32) * 1.0e-7f;
    auto adjusted_scores = final_scores - tie_break;
    auto topk_idx = std::get<1>(
        adjusted_scores.topk(actual_topk, /*dim=*/0, /*largest=*/true, /*sorted=*/true));
    auto topk_tokens = token_indices.index_select(0, topk_idx).to(torch::kInt32);
    output.index_put_({b, Slice(0, actual_topk)}, topk_tokens);
  }

  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel", &kernel, "DSA top-k indexer kernel");
}
