#include <torch/extension.h>

#include <ATen/ATen.h>

#include <cmath>
#include <tuple>

namespace {

inline at::Tensor ensure_contiguous(const at::Tensor& t) {
  return t.defined() ? t.contiguous() : t;
}

inline at::Tensor repeat_heads(const at::Tensor& x, int64_t repeat_factor) {
  if (repeat_factor == 1) {
    return x;
  }
  return x.repeat_interleave(repeat_factor, /*dim=*/1);
}

inline float load_scale(const pybind11::object& scale_obj) {
  if (scale_obj.is_none()) {
    return 0.0f;
  }

  if (pybind11::isinstance<torch::Tensor>(scale_obj)) {
    auto scale_tensor = scale_obj.cast<torch::Tensor>();
    if (!scale_tensor.defined() || scale_tensor.numel() == 0) {
      return 0.0f;
    }
    return static_cast<float>(scale_tensor.item<double>());
  }

  return scale_obj.cast<float>();
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> kernel(const at::Tensor& q,
                                          const at::Tensor& k,
                                          const at::Tensor& v,
                                          const at::Tensor& state,
                                          const at::Tensor& A_log,
                                          const at::Tensor& a,
                                          const at::Tensor& dt_bias,
                                          const at::Tensor& b,
                                          pybind11::object scale_obj) {
  TORCH_CHECK(q.defined(), "q must be defined");
  TORCH_CHECK(k.defined(), "k must be defined");
  TORCH_CHECK(v.defined(), "v must be defined");
  TORCH_CHECK(A_log.defined(), "A_log must be defined");
  TORCH_CHECK(a.defined(), "a must be defined");
  TORCH_CHECK(dt_bias.defined(), "dt_bias must be defined");
  TORCH_CHECK(b.defined(), "b must be defined");
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
  TORCH_CHECK(A_log.is_cuda(), "A_log must be a CUDA tensor");
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(dt_bias.is_cuda(), "dt_bias must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");

  auto qc = ensure_contiguous(q);
  auto kc = ensure_contiguous(k);
  auto vc = ensure_contiguous(v);
  auto statec = ensure_contiguous(state);
  auto A_logc = ensure_contiguous(A_log);
  auto ac = ensure_contiguous(a);
  auto dt_biasc = ensure_contiguous(dt_bias);
  auto bc = ensure_contiguous(b);

  TORCH_CHECK(qc.dim() == 4, "q must have shape [B, 1, num_q_heads, head_size]");
  TORCH_CHECK(kc.dim() == 4, "k must have shape [B, 1, num_k_heads, head_size]");
  TORCH_CHECK(vc.dim() == 4, "v must have shape [B, 1, num_v_heads, head_size]");
  TORCH_CHECK(statec.dim() == 4, "state must have shape [B, num_v_heads, head_size, head_size]");
  TORCH_CHECK(ac.dim() == 3, "a must have shape [B, 1, num_v_heads]");
  TORCH_CHECK(bc.dim() == 3, "b must have shape [B, 1, num_v_heads]");
  TORCH_CHECK(A_logc.dim() == 1, "A_log must have shape [num_v_heads]");
  TORCH_CHECK(dt_biasc.dim() == 1, "dt_bias must have shape [num_v_heads]");

  const auto batch_size = qc.size(0);
  const auto seq_len = qc.size(1);
  const auto num_q_heads = qc.size(2);
  const auto head_size = qc.size(3);
  const auto num_k_heads = kc.size(2);
  const auto num_v_heads = vc.size(2);
  const auto value_size = vc.size(3);

  TORCH_CHECK(seq_len == 1, "gdn_decode_qk4_v8_d128_k_last expects seq_len == 1");
  TORCH_CHECK(num_q_heads == 4, "expected 4 query heads");
  TORCH_CHECK(num_k_heads == 4, "expected 4 key heads");
  TORCH_CHECK(num_v_heads == 8, "expected 8 value heads");
  TORCH_CHECK(head_size == 128 && value_size == 128, "expected head size 128");
  TORCH_CHECK(
      statec.sizes() == torch::IntArrayRef({batch_size, num_v_heads, value_size, head_size}),
      "state must have shape [B, 8, 128, 128]");

  const int64_t q_repeat = num_v_heads / num_q_heads;
  const int64_t k_repeat = num_v_heads / num_k_heads;
  TORCH_CHECK(q_repeat * num_q_heads == num_v_heads, "num_v_heads must be divisible by num_q_heads");
  TORCH_CHECK(k_repeat * num_k_heads == num_v_heads, "num_v_heads must be divisible by num_k_heads");

  const float scale_value = [&]() {
    const float scale = load_scale(scale_obj);
    if (scale == 0.0f) {
      return 1.0f / std::sqrt(static_cast<float>(head_size));
    }
    return scale;
  }();

  auto q_exp = repeat_heads(qc.squeeze(1), q_repeat).to(at::kFloat);
  auto k_exp = repeat_heads(kc.squeeze(1), k_repeat).to(at::kFloat);
  auto v_f = vc.squeeze(1).to(at::kFloat);
  auto state_f = statec.to(at::kFloat);
  auto x = ac.squeeze(1).to(at::kFloat) + dt_biasc.to(at::kFloat);
  auto g = at::exp(-at::exp(A_logc.to(at::kFloat)) * at::softplus(x));
  auto beta = at::sigmoid(bc.squeeze(1).to(at::kFloat));

  auto output_f = at::zeros({batch_size, num_v_heads, value_size}, at::TensorOptions().dtype(at::kFloat).device(q.device()));
  auto new_state_f = at::zeros_like(state_f);

  for (int64_t b_idx = 0; b_idx < batch_size; ++b_idx) {
    auto q_b = q_exp[b_idx];
    auto k_b = k_exp[b_idx];
    auto v_b = v_f[b_idx];
    auto state_hkv = state_f[b_idx].clone().transpose(-1, -2);
    auto g_b = g[b_idx].unsqueeze(-1).unsqueeze(-1);
    auto beta_b = beta[b_idx].unsqueeze(-1);

    auto old_state_hkv = g_b * state_hkv;
    auto old_v = at::matmul(k_b.unsqueeze(1), old_state_hkv).squeeze(1);
    auto new_v = beta_b * v_b + (1.0 - beta_b) * old_v;
    auto state_remove = at::matmul(k_b.unsqueeze(2), old_v.unsqueeze(1));
    auto state_update = at::matmul(k_b.unsqueeze(2), new_v.unsqueeze(1));
    auto state_next_hkv = old_state_hkv - state_remove + state_update;
    auto out_b = scale_value * at::matmul(q_b.unsqueeze(1), state_next_hkv).squeeze(1);

    output_f[b_idx].copy_(out_b);
    new_state_f[b_idx].copy_(state_next_hkv.transpose(-1, -2));
  }

  return {
      output_f.unsqueeze(1).to(q.dtype()),
      new_state_f.to(state.dtype()),
  };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel", &kernel, "GDN decode kernel");
}
