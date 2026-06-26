#include <torch/extension.h>

#include <ATen/ATen.h>
#include <torch/types.h>

#include <cmath>
#include <vector>

namespace {

inline at::Tensor ensure_contiguous(const at::Tensor& t) {
  return t.defined() ? t.contiguous() : t;
}

at::Tensor repeat_heads(const at::Tensor& x, int64_t repeat_factor) {
  if (repeat_factor == 1) {
    return x;
  }
  return x.repeat_interleave(repeat_factor, /*dim=*/1);
}

}  // namespace

std::vector<at::Tensor> kernel(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& state,
    const at::Tensor& A_log,
    const at::Tensor& a,
    const at::Tensor& dt_bias,
    const at::Tensor& b,
    const at::Tensor& cu_seqlens,
    const double scale) {
  TORCH_CHECK(q.defined(), "q must be defined");
  TORCH_CHECK(k.defined(), "k must be defined");
  TORCH_CHECK(v.defined(), "v must be defined");
  TORCH_CHECK(A_log.defined(), "A_log must be defined");
  TORCH_CHECK(a.defined(), "a must be defined");
  TORCH_CHECK(dt_bias.defined(), "dt_bias must be defined");
  TORCH_CHECK(b.defined(), "b must be defined");
  TORCH_CHECK(cu_seqlens.defined(), "cu_seqlens must be defined");

  auto qc = ensure_contiguous(q);
  auto kc = ensure_contiguous(k);
  auto vc = ensure_contiguous(v);
  auto statec = state.defined() ? ensure_contiguous(state) : state;
  auto A_logc = ensure_contiguous(A_log);
  auto ac = ensure_contiguous(a);
  auto dt_biasc = ensure_contiguous(dt_bias);
  auto bc = ensure_contiguous(b);
  auto cu_seqlensc = ensure_contiguous(cu_seqlens);

  TORCH_CHECK(qc.dim() == 3, "q must have shape [total_seq_len, num_q_heads, head_size]");
  TORCH_CHECK(kc.dim() == 3, "k must have shape [total_seq_len, num_k_heads, head_size]");
  TORCH_CHECK(vc.dim() == 3, "v must have shape [total_seq_len, num_v_heads, head_size]");
  TORCH_CHECK(ac.dim() == 2, "a must have shape [total_seq_len, num_v_heads]");
  TORCH_CHECK(bc.dim() == 2, "b must have shape [total_seq_len, num_v_heads]");
  TORCH_CHECK(A_logc.dim() == 1, "A_log must have shape [num_v_heads]");
  TORCH_CHECK(dt_biasc.dim() == 1, "dt_bias must have shape [num_v_heads]");
  TORCH_CHECK(cu_seqlensc.dim() == 1, "cu_seqlens must be a 1D prefix-sum array");

  const auto total_seq_len = qc.size(0);
  const auto num_q_heads = qc.size(1);
  const auto head_size = qc.size(2);
  const auto num_v_heads = vc.size(1);
  const auto num_k_heads = kc.size(1);
  const auto num_seqs = cu_seqlensc.size(0) - 1;

  TORCH_CHECK(kc.size(0) == total_seq_len, "k must match q in sequence length");
  TORCH_CHECK(vc.size(0) == total_seq_len, "v must match q in sequence length");
  TORCH_CHECK(kc.size(2) == head_size, "q and k must have the same head size");
  TORCH_CHECK(vc.size(2) == head_size, "q and v must have the same head size");
  TORCH_CHECK(ac.size(0) == total_seq_len, "a must match q in sequence length");
  TORCH_CHECK(bc.size(0) == total_seq_len, "b must match q in sequence length");
  TORCH_CHECK(ac.size(1) == num_v_heads, "a must match v head count");
  TORCH_CHECK(bc.size(1) == num_v_heads, "b must match v head count");
  TORCH_CHECK(A_logc.size(0) == num_v_heads, "A_log must match v head count");
  TORCH_CHECK(dt_biasc.size(0) == num_v_heads, "dt_bias must match v head count");

  const int64_t q_repeat = num_v_heads / num_q_heads;
  const int64_t k_repeat = num_v_heads / num_k_heads;
  TORCH_CHECK(q_repeat * num_q_heads == num_v_heads, "num_v_heads must be divisible by num_q_heads");
  TORCH_CHECK(k_repeat * num_k_heads == num_v_heads, "num_v_heads must be divisible by num_k_heads");

  auto q_exp = repeat_heads(qc, q_repeat).to(at::kFloat);
  auto k_exp = repeat_heads(kc, k_repeat).to(at::kFloat);
  auto v_f = vc.to(at::kFloat);
  auto a_f = ac.to(at::kFloat);
  auto b_f = bc.to(at::kFloat);
  auto A_log_f = A_logc.to(at::kFloat);
  auto dt_bias_f = dt_biasc.to(at::kFloat);

  const double scale_value = scale == 0.0 ? 1.0 / std::sqrt(static_cast<double>(head_size)) : scale;

  auto output = at::zeros({total_seq_len, num_v_heads, head_size}, q.options().dtype(at::kBFloat16));
  auto new_state = at::zeros(
      {num_seqs, num_v_heads, head_size, head_size},
      state.defined() ? state.options() : at::TensorOptions().dtype(at::kFloat).device(q.device()));

  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t seq_start = cu_seqlensc[seq_idx].item<int64_t>();
    const int64_t seq_end = cu_seqlensc[seq_idx + 1].item<int64_t>();
    const int64_t seq_len = seq_end - seq_start;
    if (seq_len <= 0) {
      continue;
    }

    at::Tensor state_hkv;
    if (state.defined()) {
      state_hkv = statec[seq_idx].to(at::kFloat).transpose(-1, -2);
    } else {
      state_hkv = at::zeros({num_v_heads, head_size, head_size}, at::TensorOptions().dtype(at::kFloat).device(q.device()));
      state_hkv = state_hkv.transpose(-1, -2);
    }

    for (int64_t t = 0; t < seq_len; ++t) {
      const int64_t idx = seq_start + t;
      auto q_t = q_exp[idx];
      auto k_t = k_exp[idx];
      auto v_t = v_f[idx];
      auto a_t = a_f[idx];
      auto b_t = b_f[idx];

      auto x = a_t + dt_bias_f;
      auto g = torch::exp(-torch::exp(A_log_f) * at::softplus(x));
      auto beta = torch::sigmoid(b_t);

      auto out_t = at::empty({num_v_heads, head_size}, at::TensorOptions().dtype(at::kFloat).device(q.device()));
      for (int64_t h = 0; h < num_v_heads; ++h) {
        auto state_hv = state_hkv[h].unsqueeze(0);
        auto q_h1k = q_t[h].unsqueeze(0).unsqueeze(1).to(at::kFloat);
        auto k_h1k = k_t[h].unsqueeze(0).unsqueeze(1).to(at::kFloat);
        auto v_h1v = v_t[h].unsqueeze(0).unsqueeze(1).to(at::kFloat);
        auto g_h11 = g[h].unsqueeze(0).unsqueeze(1).unsqueeze(2);
        auto beta_h11 = beta[h].unsqueeze(0).unsqueeze(1).unsqueeze(2);

        auto old_state_hkv = g_h11 * state_hv;
        auto old_v_h1v = at::matmul(k_h1k, old_state_hkv);
        auto new_v_h1v = beta_h11 * v_h1v + (1.0 - beta_h11) * old_v_h1v;
        auto state_remove = at::einsum("hkl,hlv->hkv", {k_h1k.transpose(-1, -2), old_v_h1v});
        auto state_update = at::einsum("hkl,hlv->hkv", {k_h1k.transpose(-1, -2), new_v_h1v});
        state_hv = old_state_hkv - state_remove + state_update;
        state_hkv[h].copy_(state_hv.squeeze(0));

        auto out_h = scale_value * at::matmul(q_h1k, state_hv).squeeze(0).squeeze(0);
        out_t[h].copy_(out_h);
      }

      output.index_put_({idx}, out_t.to(at::kBFloat16));
    }

    new_state.index_put_({seq_idx}, state_hkv.transpose(-1, -2).to(new_state.dtype()));
  }

  return {output, new_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel", &kernel, "GDN prefill kernel");
}
