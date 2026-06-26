#include <torch/extension.h>

#include <ATen/ATen.h>

#include <limits>
#include <vector>

namespace {

constexpr int64_t kHiddenSize = 7168;
constexpr int64_t kIntermediateSize = 2048;
constexpr int64_t kNumExpertsGlobal = 256;
constexpr int64_t kNumLocalExperts = 32;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kNumHiddenBlocks = kHiddenSize / kBlockSize;
constexpr int64_t kNumIntermediateBlocks = kIntermediateSize / kBlockSize;
constexpr int64_t kNumGemm1OutBlocks = (2 * kIntermediateSize) / kBlockSize;
constexpr int64_t kTopK = 8;
constexpr int64_t kNumGroups = 8;
constexpr int64_t kTopKGroups = 4;

at::Tensor dequantize_hidden_states(
    const at::Tensor& hidden_states,
    const at::Tensor& hidden_states_scale) {
  auto hidden_f32 = hidden_states.to(at::kFloat).contiguous();
  auto scale_f32 = hidden_states_scale.to(at::kFloat).permute({1, 0}).contiguous();
  auto expanded = scale_f32.unsqueeze(-1).repeat({1, 1, kBlockSize}).reshape({
      hidden_states.size(0),
      kHiddenSize,
  });
  return hidden_f32 * expanded;
}

at::Tensor dequantize_gemm1_weight_expert(
    const at::Tensor& gemm1_weights,
    const at::Tensor& gemm1_weights_scale,
    int64_t local_expert) {
  auto w_f32 = gemm1_weights[local_expert].to(at::kFloat).contiguous();
  auto s_f32 = gemm1_weights_scale[local_expert].to(at::kFloat).contiguous();
  auto expanded = at::repeat_interleave(s_f32, kBlockSize, /*dim=*/0);
  expanded = at::repeat_interleave(expanded, kBlockSize, /*dim=*/1);
  return w_f32 * expanded;
}

at::Tensor dequantize_gemm2_weight_expert(
    const at::Tensor& gemm2_weights,
    const at::Tensor& gemm2_weights_scale,
    int64_t local_expert) {
  auto w_f32 = gemm2_weights[local_expert].to(at::kFloat).contiguous();
  auto s_f32 = gemm2_weights_scale[local_expert].to(at::kFloat).contiguous();
  auto expanded = at::repeat_interleave(s_f32, kBlockSize, /*dim=*/0);
  expanded = at::repeat_interleave(expanded, kBlockSize, /*dim=*/1);
  return w_f32 * expanded;
}

}  // namespace

at::Tensor kernel(
    const at::Tensor& routing_logits,
    const at::Tensor& routing_bias,
    const at::Tensor& hidden_states,
    const at::Tensor& hidden_states_scale,
    const at::Tensor& gemm1_weights,
    const at::Tensor& gemm1_weights_scale,
    const at::Tensor& gemm2_weights,
    const at::Tensor& gemm2_weights_scale,
    int64_t local_expert_offset,
    double routed_scaling_factor) {
  const int64_t seq_len = routing_logits.size(0);
  const int64_t num_experts = routing_logits.size(1);
  const int64_t num_local_experts = gemm1_weights.size(0);

  TORCH_CHECK(num_experts == kNumExpertsGlobal, "expected 256 experts");
  TORCH_CHECK(num_local_experts == kNumLocalExperts, "expected 32 local experts");
  TORCH_CHECK(hidden_states.sizes() == torch::IntArrayRef({seq_len, kHiddenSize}), "hidden_states shape mismatch");
  TORCH_CHECK(hidden_states_scale.sizes() == torch::IntArrayRef({kNumHiddenBlocks, seq_len}), "hidden_states_scale shape mismatch");
  TORCH_CHECK(gemm1_weights.sizes() == torch::IntArrayRef({kNumLocalExperts, 2 * kIntermediateSize, kHiddenSize}), "gemm1_weights shape mismatch");
  TORCH_CHECK(gemm1_weights_scale.sizes() == torch::IntArrayRef({kNumLocalExperts, kNumGemm1OutBlocks, kNumHiddenBlocks}), "gemm1_weights_scale shape mismatch");
  TORCH_CHECK(gemm2_weights.sizes() == torch::IntArrayRef({kNumLocalExperts, kHiddenSize, kIntermediateSize}), "gemm2_weights shape mismatch");
  TORCH_CHECK(gemm2_weights_scale.sizes() == torch::IntArrayRef({kNumLocalExperts, kNumHiddenBlocks, kNumIntermediateBlocks}), "gemm2_weights_scale shape mismatch");
  TORCH_CHECK(routing_bias.numel() == kNumExpertsGlobal, "routing_bias shape mismatch");

  auto hidden_f32 = dequantize_hidden_states(hidden_states, hidden_states_scale);

  auto logits = routing_logits.to(at::kFloat).contiguous();
  auto bias = routing_bias.to(at::kFloat).reshape({kNumExpertsGlobal}).contiguous();
  auto s = at::sigmoid(logits);
  auto s_with_bias = s + bias;

  const int64_t group_size = kNumExpertsGlobal / kNumGroups;
  auto grouped = s_with_bias.view({seq_len, kNumGroups, group_size});
  auto top2_vals = std::get<0>(grouped.topk(2, /*dim=*/2, /*largest=*/true, /*sorted=*/false));
  auto group_scores = top2_vals.sum(/*dim=*/2);
  auto group_idx = std::get<1>(group_scores.topk(kTopKGroups, /*dim=*/1, /*largest=*/true, /*sorted=*/false));

  auto group_mask = at::zeros_like(group_scores);
  group_mask.scatter_(1, group_idx, 1.0);
  auto score_mask = group_mask.unsqueeze(2).expand({seq_len, kNumGroups, group_size}).reshape({seq_len, kNumExpertsGlobal});

  const double neg_inf = std::numeric_limits<float>::lowest();
  auto scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf);
  auto topk_idx = std::get<1>(scores_pruned.topk(kTopK, /*dim=*/1, /*largest=*/true, /*sorted=*/false));

  auto topk_weights = s.gather(1, topk_idx);
  auto topk_weights_sum = topk_weights.sum(/*dim=*/1, /*keepdim=*/true) + 1e-20;
  auto normalized_topk_weights = (topk_weights / topk_weights_sum) * routed_scaling_factor;

  auto output = at::zeros(
      {seq_len, kHiddenSize},
      at::TensorOptions().dtype(at::kFloat).device(hidden_states.device()));

  for (int64_t local_expert = 0; local_expert < kNumLocalExperts; ++local_expert) {
    const int64_t global_expert = local_expert_offset + local_expert;
    if (global_expert < 0 || global_expert >= kNumExpertsGlobal) {
      continue;
    }

    auto selected_positions = at::eq(topk_idx, global_expert);
    if (!selected_positions.any().item<bool>()) {
      continue;
    }

    auto matched = at::nonzero(selected_positions);
    auto token_idx = matched.select(1, 0);
    auto topk_slot_idx = matched.select(1, 1);
    auto hidden_e = hidden_f32.index_select(0, token_idx);
    auto w13_e = dequantize_gemm1_weight_expert(gemm1_weights, gemm1_weights_scale, local_expert);
    auto w2_e = dequantize_gemm2_weight_expert(gemm2_weights, gemm2_weights_scale, local_expert);

    auto g1 = at::matmul(hidden_e, w13_e.transpose(0, 1));
    auto x1 = g1.narrow(1, 0, kIntermediateSize);
    auto x2 = g1.narrow(1, kIntermediateSize, kIntermediateSize);
    auto c = at::silu(x2) * x1;
    auto out_e = at::matmul(c, w2_e.transpose(0, 1));

    auto w_tok = normalized_topk_weights.index_select(0, token_idx)
                     .gather(1, topk_slot_idx.unsqueeze(1))
                     .squeeze(1);
    output.index_add_(0, token_idx, out_e * w_tok.unsqueeze(1));
  }

  return output.to(at::kBFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel", &kernel, "FlashInfer MoE FP8 phase-1 baseline kernel");
}
