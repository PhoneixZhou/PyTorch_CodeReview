#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <torch/library.h>


#include <ATen/RedispatchFunctions.h>

// @generated from tools/autograd/templates/InplaceOrViewType.cpp


using namespace at;
using torch::autograd::CreationMeta;
using torch::autograd::as_view;
using torch::autograd::increment_version;

namespace torch {

namespace InplaceOrView {

namespace {
Tensor & _add_relu__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_add_relu_(ks & c10::after_InplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & _add_relu_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_add_relu_outf(ks & c10::after_InplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
Tensor & _bmm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat2, bool deterministic, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_bmm_outf(ks & c10::after_InplaceOrView_keyset, self, mat2, deterministic, out);
  }
  increment_version(out);
  return out;
}
Tensor & _cat_out_out(c10::DispatchKeySet ks, TensorList tensors, int64_t dim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_cat_outf(ks & c10::after_InplaceOrView_keyset, tensors, dim, out);
  }
  increment_version(out);
  return out;
}
Tensor & _cumprod_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_cumprod_outf(ks & c10::after_InplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
Tensor & _cumsum_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_cumsum_outf(ks & c10::after_InplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
Tensor & _fft_c2r_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_fft_c2r_outf(ks & c10::after_InplaceOrView_keyset, self, dim, normalization, last_dim_size, out);
  }
  increment_version(out);
  return out;
}
Tensor & _fft_r2c_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, int64_t normalization, bool onesided, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_fft_r2c_outf(ks & c10::after_InplaceOrView_keyset, self, dim, normalization, onesided, out);
  }
  increment_version(out);
  return out;
}
Tensor & _index_copy_(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_index_copy_(ks & c10::after_InplaceOrView_keyset, self, dim, index, source);
  }
  increment_version(self);
  return self;
}
Tensor _indices(c10::DispatchKeySet ks, const Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_indices(ks & c10::after_InplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
Tensor & _mkldnn_transpose_(c10::DispatchKeySet ks, Tensor & self, int64_t dim0, int64_t dim1) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_mkldnn_transpose_(ks & c10::after_InplaceOrView_keyset, self, dim0, dim1);
  }
  increment_version(self);
  return self;
}
Tensor & _stack_out_out(c10::DispatchKeySet ks, TensorList tensors, int64_t dim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_stack_outf(ks & c10::after_InplaceOrView_keyset, tensors, dim, out);
  }
  increment_version(out);
  return out;
}
Tensor & adaptive_avg_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::adaptive_avg_pool3d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & adaptive_avg_pool3d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::adaptive_avg_pool3d_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, out);
  }
  increment_version(out);
  return out;
}
Tensor & adaptive_max_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & indices, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::adaptive_max_pool3d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, Tensor & out, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::adaptive_max_pool3d_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
}
Tensor & add__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::add_(ks & c10::after_InplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & add__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::add_(ks & c10::after_InplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & add_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::add_outf(ks & c10::after_InplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
Tensor & addbmm_(c10::DispatchKeySet ks, Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addbmm_(ks & c10::after_InplaceOrView_keyset, self, batch1, batch2, beta, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & addbmm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addbmm_outf(ks & c10::after_InplaceOrView_keyset, self, batch1, batch2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
Tensor & addcdiv_(c10::DispatchKeySet ks, Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addcdiv_(ks & c10::after_InplaceOrView_keyset, self, tensor1, tensor2, value);
  }
  increment_version(self);
  return self;
}
Tensor & addcdiv_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addcdiv_outf(ks & c10::after_InplaceOrView_keyset, self, tensor1, tensor2, value, out);
  }
  increment_version(out);
  return out;
}
Tensor & addmm_(c10::DispatchKeySet ks, Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addmm_(ks & c10::after_InplaceOrView_keyset, self, mat1, mat2, beta, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & addmm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addmm_outf(ks & c10::after_InplaceOrView_keyset, self, mat1, mat2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
Tensor & addr_(c10::DispatchKeySet ks, Tensor & self, const Tensor & vec1, const Tensor & vec2, const Scalar & beta, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addr_(ks & c10::after_InplaceOrView_keyset, self, vec1, vec2, beta, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & addr_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & vec1, const Tensor & vec2, const Scalar & beta, const Scalar & alpha, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addr_outf(ks & c10::after_InplaceOrView_keyset, self, vec1, vec2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
Tensor alias(c10::DispatchKeySet ks, const Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::alias(ks & c10::after_InplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::alias(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & all_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::all_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
Tensor & amax_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::amax_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
Tensor & amin_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::amin_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
Tensor & angle_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::angle_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor as_strided(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::as_strided(ks & c10::after_InplaceOrView_keyset, self, size, stride, storage_offset);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    auto size_vec = size.vec();
    auto stride_vec = stride.vec();
    auto storage_offset_val = storage_offset.value_or(0);
    func = [=](const at::Tensor& input_base) {
      return at::as_strided(input_base, size_vec, stride_vec, storage_offset_val);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & asin_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::asin_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & asin_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::asin_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & asinh_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::asinh_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & asinh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::asinh_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & avg_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::avg_pool2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & avg_pool2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::avg_pool2d_outf(ks & c10::after_InplaceOrView_keyset, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
  }
  increment_version(out);
  return out;
}
Tensor & baddbmm_(c10::DispatchKeySet ks, Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::baddbmm_(ks & c10::after_InplaceOrView_keyset, self, batch1, batch2, beta, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & baddbmm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::baddbmm_outf(ks & c10::after_InplaceOrView_keyset, self, batch1, batch2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
Tensor & batch_norm_elemt_out_out(c10::DispatchKeySet ks, const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const Tensor & mean, const Tensor & invstd, double eps, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::batch_norm_elemt_outf(ks & c10::after_InplaceOrView_keyset, input, weight, bias, mean, invstd, eps, out);
  }
  increment_version(out);
  return out;
}
Tensor & bitwise_and_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bitwise_and_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & bitwise_and_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bitwise_and_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & bitwise_not_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bitwise_not_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & bitwise_or_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bitwise_or_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & bitwise_or_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bitwise_or_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & bitwise_xor_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bitwise_xor_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & bitwise_xor_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bitwise_xor_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & cauchy_(c10::DispatchKeySet ks, Tensor & self, double median, double sigma, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cauchy_(ks & c10::after_InplaceOrView_keyset, self, median, sigma, generator);
  }
  increment_version(self);
  return self;
}
Tensor & ceil_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ceil_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & ceil_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ceil_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & celu_(c10::DispatchKeySet ks, Tensor & self, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::celu_(ks & c10::after_InplaceOrView_keyset, self, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & cholesky_inverse_out_out(c10::DispatchKeySet ks, const Tensor & self, bool upper, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cholesky_inverse_outf(ks & c10::after_InplaceOrView_keyset, self, upper, out);
  }
  increment_version(out);
  return out;
}
Tensor & clamp_(c10::DispatchKeySet ks, Tensor & self, const c10::optional<Scalar> & min, const c10::optional<Scalar> & max) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::clamp_(ks & c10::after_InplaceOrView_keyset, self, min, max);
  }
  increment_version(self);
  return self;
}
Tensor & clamp_out_out(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & min, const c10::optional<Scalar> & max, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::clamp_outf(ks & c10::after_InplaceOrView_keyset, self, min, max, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &,Tensor &> conv_depthwise3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::conv_depthwise3d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, dilation, grad_input, grad_weight, grad_bias);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
Tensor & copy_sparse_to_sparse_(c10::DispatchKeySet ks, Tensor & self, const Tensor & src, bool non_blocking) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::copy_sparse_to_sparse_(ks & c10::after_InplaceOrView_keyset, self, src, non_blocking);
  }
  increment_version(self);
  return self;
}
Tensor & cos_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cos_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & cos_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cos_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & cosh_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cosh_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & cosh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cosh_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> cummax_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cummax_outf(ks & c10::after_InplaceOrView_keyset, self, dim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> cummin_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cummin_outf(ks & c10::after_InplaceOrView_keyset, self, dim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
Tensor & deg2rad_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::deg2rad_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & deg2rad_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::deg2rad_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & diag_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t diagonal, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::diag_outf(ks & c10::after_InplaceOrView_keyset, self, diagonal, out);
  }
  increment_version(out);
  return out;
}
Tensor & div__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::div_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & div__Tensor_mode(c10::DispatchKeySet ks, Tensor & self, const Tensor & other, std::string rounding_mode) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::div_(ks & c10::after_InplaceOrView_keyset, self, other, rounding_mode);
  }
  increment_version(self);
  return self;
}
Tensor & div__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::div_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & div__Scalar_mode(c10::DispatchKeySet ks, Tensor & self, const Scalar & other, std::string rounding_mode) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::div_(ks & c10::after_InplaceOrView_keyset, self, other, rounding_mode);
  }
  increment_version(self);
  return self;
}
Tensor & div_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::div_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & div_out_out_mode(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, std::string rounding_mode, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::div_outf(ks & c10::after_InplaceOrView_keyset, self, other, rounding_mode, out);
  }
  increment_version(out);
  return out;
}
Tensor & dot_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & tensor, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::dot_outf(ks & c10::after_InplaceOrView_keyset, self, tensor, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> eig_out_e(c10::DispatchKeySet ks, const Tensor & self, bool eigenvectors, Tensor & e, Tensor & v) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::eig_outf(ks & c10::after_InplaceOrView_keyset, self, eigenvectors, e, v);
  }
  increment_version(e);
  increment_version(v);
  return std::forward_as_tuple(e, v);
}
Tensor & embedding_renorm_(c10::DispatchKeySet ks, Tensor & self, const Tensor & indices, double max_norm, double norm_type) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::embedding_renorm_(ks & c10::after_InplaceOrView_keyset, self, indices, max_norm, norm_type);
  }
  increment_version(self);
  return self;
}
Tensor & erf_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::erf_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & erf_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::erf_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & exp2_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::exp2_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & exp2_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::exp2_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & exp_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::exp_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & exp_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::exp_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor expand(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef size, bool implicit) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::expand(ks & c10::after_InplaceOrView_keyset, self, size, implicit);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    auto size_vec = size.vec();
    func = [=](const at::Tensor& input_base) {
      return input_base.expand(size_vec, implicit);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & expm1_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::expm1_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & expm1_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::expm1_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & exponential_(c10::DispatchKeySet ks, Tensor & self, double lambd, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::exponential_(ks & c10::after_InplaceOrView_keyset, self, lambd, generator);
  }
  increment_version(self);
  return self;
}
Tensor & eye_out_out(c10::DispatchKeySet ks, int64_t n, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::eye_outf(ks & c10::after_InplaceOrView_keyset, n, out);
  }
  increment_version(out);
  return out;
}
Tensor & eye_out_m_out(c10::DispatchKeySet ks, int64_t n, int64_t m, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::eye_outf(ks & c10::after_InplaceOrView_keyset, n, m, out);
  }
  increment_version(out);
  return out;
}
Tensor & fill__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & value) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fill_(ks & c10::after_InplaceOrView_keyset, self, value);
  }
  increment_version(self);
  return self;
}
Tensor & fill__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & value) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fill_(ks & c10::after_InplaceOrView_keyset, self, value);
  }
  increment_version(self);
  return self;
}
Tensor & fractional_max_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fractional_max_pool2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, kernel_size, output_size, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out_output(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples, Tensor & output, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fractional_max_pool2d_outf(ks & c10::after_InplaceOrView_keyset, self, kernel_size, output_size, random_samples, output, indices);
  }
  increment_version(output);
  increment_version(indices);
  return std::forward_as_tuple(output, indices);
}
std::tuple<Tensor &,Tensor &> frexp_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & mantissa, Tensor & exponent) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::frexp_outf(ks & c10::after_InplaceOrView_keyset, self, mantissa, exponent);
  }
  increment_version(mantissa);
  increment_version(exponent);
  return std::forward_as_tuple(mantissa, exponent);
}
Tensor & gather_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::gather_outf(ks & c10::after_InplaceOrView_keyset, self, dim, index, sparse_grad, out);
  }
  increment_version(out);
  return out;
}
Tensor & geometric_(c10::DispatchKeySet ks, Tensor & self, double p, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::geometric_(ks & c10::after_InplaceOrView_keyset, self, p, generator);
  }
  increment_version(self);
  return self;
}
std::tuple<Tensor &,Tensor &> geqrf_out_a(c10::DispatchKeySet ks, const Tensor & self, Tensor & a, Tensor & tau) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::geqrf_outf(ks & c10::after_InplaceOrView_keyset, self, a, tau);
  }
  increment_version(a);
  increment_version(tau);
  return std::forward_as_tuple(a, tau);
}
Tensor & gt__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::gt_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & gt__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::gt_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & gt_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::gt_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & gt_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::gt_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & hardsigmoid_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::hardsigmoid_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & hardsigmoid_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::hardsigmoid_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & hardswish_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::hardswish_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & hardswish_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::hardswish_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & histc_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t bins, const Scalar & min, const Scalar & max, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::histc_outf(ks & c10::after_InplaceOrView_keyset, self, bins, min, max, out);
  }
  increment_version(out);
  return out;
}
Tensor & hspmm_out_out(c10::DispatchKeySet ks, const Tensor & mat1, const Tensor & mat2, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::hspmm_outf(ks & c10::after_InplaceOrView_keyset, mat1, mat2, out);
  }
  increment_version(out);
  return out;
}
Tensor & i0_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::i0_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & i0_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::i0_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & igammac_(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::igammac_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & igammac_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::igammac_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & index_select_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, const Tensor & index, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::index_select_outf(ks & c10::after_InplaceOrView_keyset, self, dim, index, out);
  }
  increment_version(out);
  return out;
}
Tensor & isneginf_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::isneginf_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & isposinf_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::isposinf_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & l1_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::l1_loss_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, target, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & l1_loss_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::l1_loss_outf(ks & c10::after_InplaceOrView_keyset, self, target, reduction, out);
  }
  increment_version(out);
  return out;
}
Tensor & le__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::le_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & le__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::le_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & le_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::le_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & le_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::le_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & leaky_relu_(c10::DispatchKeySet ks, Tensor & self, const Scalar & negative_slope) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::leaky_relu_(ks & c10::after_InplaceOrView_keyset, self, negative_slope);
  }
  increment_version(self);
  return self;
}
Tensor & leaky_relu_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & negative_slope, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::leaky_relu_outf(ks & c10::after_InplaceOrView_keyset, self, negative_slope, out);
  }
  increment_version(out);
  return out;
}
Tensor & lerp__Scalar(c10::DispatchKeySet ks, Tensor & self, const Tensor & end, const Scalar & weight) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lerp_(ks & c10::after_InplaceOrView_keyset, self, end, weight);
  }
  increment_version(self);
  return self;
}
Tensor & lerp__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & end, const Tensor & weight) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lerp_(ks & c10::after_InplaceOrView_keyset, self, end, weight);
  }
  increment_version(self);
  return self;
}
Tensor & lerp_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & end, const Scalar & weight, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lerp_outf(ks & c10::after_InplaceOrView_keyset, self, end, weight, out);
  }
  increment_version(out);
  return out;
}
Tensor & lerp_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & end, const Tensor & weight, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lerp_outf(ks & c10::after_InplaceOrView_keyset, self, end, weight, out);
  }
  increment_version(out);
  return out;
}
Tensor & lgamma_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lgamma_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & lgamma_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lgamma_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> linalg_eigh_out_eigvals(c10::DispatchKeySet ks, const Tensor & self, std::string UPLO, Tensor & eigvals, Tensor & eigvecs) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::linalg_eigh_outf(ks & c10::after_InplaceOrView_keyset, self, UPLO, eigvals, eigvecs);
  }
  increment_version(eigvals);
  increment_version(eigvecs);
  return std::forward_as_tuple(eigvals, eigvecs);
}
Tensor & linalg_eigvalsh_out_out(c10::DispatchKeySet ks, const Tensor & self, std::string UPLO, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::linalg_eigvalsh_outf(ks & c10::after_InplaceOrView_keyset, self, UPLO, out);
  }
  increment_version(out);
  return out;
}
Tensor & linalg_inv_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::linalg_inv_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> linalg_qr_out_out(c10::DispatchKeySet ks, const Tensor & self, std::string mode, Tensor & Q, Tensor & R) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::linalg_qr_outf(ks & c10::after_InplaceOrView_keyset, self, mode, Q, R);
  }
  increment_version(Q);
  increment_version(R);
  return std::forward_as_tuple(Q, R);
}
Tensor & linalg_solve_out_out(c10::DispatchKeySet ks, const Tensor & input, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::linalg_solve_outf(ks & c10::after_InplaceOrView_keyset, input, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & linspace_out_out(c10::DispatchKeySet ks, const Scalar & start, const Scalar & end, c10::optional<int64_t> steps, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::linspace_outf(ks & c10::after_InplaceOrView_keyset, start, end, steps, out);
  }
  increment_version(out);
  return out;
}
Tensor & log10_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::log10_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & log10_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::log10_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & log1p_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::log1p_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & log1p_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::log1p_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & log_sigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & buffer, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::log_sigmoid_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, buffer, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, Tensor & output, Tensor & buffer) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::log_sigmoid_forward_outf(ks & c10::after_InplaceOrView_keyset, self, output, buffer);
  }
  increment_version(output);
  increment_version(buffer);
  return std::forward_as_tuple(output, buffer);
}
Tensor & logcumsumexp_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logcumsumexp_outf(ks & c10::after_InplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
Tensor & logical_and_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logical_and_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & logical_not_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logical_not_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & logical_or_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logical_or_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & logical_xor_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logical_xor_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & logit_(c10::DispatchKeySet ks, Tensor & self, c10::optional<double> eps) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logit_(ks & c10::after_InplaceOrView_keyset, self, eps);
  }
  increment_version(self);
  return self;
}
Tensor & logit_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, c10::optional<double> eps, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logit_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, eps, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & logit_out_out(c10::DispatchKeySet ks, const Tensor & self, c10::optional<double> eps, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logit_outf(ks & c10::after_InplaceOrView_keyset, self, eps, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> lstsq_out_X(c10::DispatchKeySet ks, const Tensor & self, const Tensor & A, Tensor & X, Tensor & qr) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lstsq_outf(ks & c10::after_InplaceOrView_keyset, self, A, X, qr);
  }
  increment_version(X);
  increment_version(qr);
  return std::forward_as_tuple(X, qr);
}
Tensor & lu_solve_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lu_solve_outf(ks & c10::after_InplaceOrView_keyset, self, LU_data, LU_pivots, out);
  }
  increment_version(out);
  return out;
}
Tensor & masked_fill__Scalar(c10::DispatchKeySet ks, Tensor & self, const Tensor & mask, const Scalar & value) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::masked_fill_(ks & c10::after_InplaceOrView_keyset, self, mask, value);
  }
  increment_version(self);
  return self;
}
Tensor & masked_fill__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & mask, const Tensor & value) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::masked_fill_(ks & c10::after_InplaceOrView_keyset, self, mask, value);
  }
  increment_version(self);
  return self;
}
Tensor & max_pool3d_with_indices_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::max_pool3d_with_indices_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
std::tuple<Tensor &,Tensor &> max_pool3d_with_indices_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, Tensor & out, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::max_pool3d_with_indices_outf(ks & c10::after_InplaceOrView_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
}
Tensor & max_unpool3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::max_unpool3d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, indices, output_size, stride, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & max_unpool3d_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::max_unpool3d_outf(ks & c10::after_InplaceOrView_keyset, self, indices, output_size, stride, padding, out);
  }
  increment_version(out);
  return out;
}
Tensor & mean_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::mean_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> mode_out_values(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::mode_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
Tensor & mse_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::mse_loss_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, target, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & mse_loss_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::mse_loss_outf(ks & c10::after_InplaceOrView_keyset, self, target, reduction, out);
  }
  increment_version(out);
  return out;
}
Tensor & multinomial_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t num_samples, bool replacement, c10::optional<Generator> generator, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::multinomial_outf(ks & c10::after_InplaceOrView_keyset, self, num_samples, replacement, generator, out);
  }
  increment_version(out);
  return out;
}
Tensor & mv_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & vec, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::mv_outf(ks & c10::after_InplaceOrView_keyset, self, vec, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> nanmedian_out_dim_values(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::nanmedian_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
Tensor & narrow_copy_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, int64_t start, int64_t length, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::narrow_copy_outf(ks & c10::after_InplaceOrView_keyset, self, dim, start, length, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &,Tensor &> native_batch_norm_out_out(c10::DispatchKeySet ks, const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double momentum, double eps, Tensor & out, Tensor & save_mean, Tensor & save_invstd) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::native_batch_norm_outf(ks & c10::after_InplaceOrView_keyset, input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
  }
  increment_version(out);
  increment_version(save_mean);
  increment_version(save_invstd);
  return std::forward_as_tuple(out, save_mean, save_invstd);
}
Tensor & ne__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ne_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & ne__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ne_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & ne_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ne_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & ne_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ne_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & nextafter_(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::nextafter_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & nextafter_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::nextafter_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & nonzero_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::nonzero_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & normal_(c10::DispatchKeySet ks, Tensor & self, double mean, double std, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::normal_(ks & c10::after_InplaceOrView_keyset, self, mean, std, generator);
  }
  increment_version(self);
  return self;
}
Tensor & normal_out_Tensor_float_out(c10::DispatchKeySet ks, const Tensor & mean, double std, c10::optional<Generator> generator, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::normal_outf(ks & c10::after_InplaceOrView_keyset, mean, std, generator, out);
  }
  increment_version(out);
  return out;
}
Tensor & normal_out_float_Tensor_out(c10::DispatchKeySet ks, double mean, const Tensor & std, c10::optional<Generator> generator, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::normal_outf(ks & c10::after_InplaceOrView_keyset, mean, std, generator, out);
  }
  increment_version(out);
  return out;
}
Tensor & normal_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const Tensor & mean, const Tensor & std, c10::optional<Generator> generator, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::normal_outf(ks & c10::after_InplaceOrView_keyset, mean, std, generator, out);
  }
  increment_version(out);
  return out;
}
Tensor & ormqr_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ormqr_outf(ks & c10::after_InplaceOrView_keyset, self, input2, input3, left, transpose, out);
  }
  increment_version(out);
  return out;
}
Tensor permute(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dims) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::permute(ks & c10::after_InplaceOrView_keyset, self, dims);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    auto dims_vec = dims.vec();
    func = [=](const at::Tensor& input_base) {
      return input_base.permute(dims_vec);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & polygamma_out_out(c10::DispatchKeySet ks, int64_t n, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::polygamma_outf(ks & c10::after_InplaceOrView_keyset, n, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & prod_out_int_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::prod_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
Tensor & put_(c10::DispatchKeySet ks, Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::put_(ks & c10::after_InplaceOrView_keyset, self, index, source, accumulate);
  }
  increment_version(self);
  return self;
}
Tensor & rad2deg_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::rad2deg_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & rad2deg_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::rad2deg_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & random__from(c10::DispatchKeySet ks, Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::random_(ks & c10::after_InplaceOrView_keyset, self, from, to, generator);
  }
  increment_version(self);
  return self;
}
Tensor & random__to(c10::DispatchKeySet ks, Tensor & self, int64_t to, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::random_(ks & c10::after_InplaceOrView_keyset, self, to, generator);
  }
  increment_version(self);
  return self;
}
Tensor & random_(c10::DispatchKeySet ks, Tensor & self, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::random_(ks & c10::after_InplaceOrView_keyset, self, generator);
  }
  increment_version(self);
  return self;
}
Tensor & randperm_out_generator_out(c10::DispatchKeySet ks, int64_t n, c10::optional<Generator> generator, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::randperm_outf(ks & c10::after_InplaceOrView_keyset, n, generator, out);
  }
  increment_version(out);
  return out;
}
Tensor & range_out_out(c10::DispatchKeySet ks, const Scalar & start, const Scalar & end, const Scalar & step, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::range_outf(ks & c10::after_InplaceOrView_keyset, start, end, step, out);
  }
  increment_version(out);
  return out;
}
Tensor & reflection_pad2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::reflection_pad2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & reflection_pad2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::reflection_pad2d_outf(ks & c10::after_InplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
Tensor & remainder__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::remainder_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & remainder__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::remainder_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & remainder_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::remainder_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & remainder_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::remainder_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & renorm_(c10::DispatchKeySet ks, Tensor & self, const Scalar & p, int64_t dim, const Scalar & maxnorm) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::renorm_(ks & c10::after_InplaceOrView_keyset, self, p, dim, maxnorm);
  }
  increment_version(self);
  return self;
}
Tensor & renorm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & p, int64_t dim, const Scalar & maxnorm, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::renorm_outf(ks & c10::after_InplaceOrView_keyset, self, p, dim, maxnorm, out);
  }
  increment_version(out);
  return out;
}
Tensor & replication_pad1d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::replication_pad1d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & replication_pad1d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::replication_pad1d_outf(ks & c10::after_InplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
Tensor & replication_pad3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::replication_pad3d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & replication_pad3d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::replication_pad3d_outf(ks & c10::after_InplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
Tensor & searchsorted_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::searchsorted_outf(ks & c10::after_InplaceOrView_keyset, sorted_sequence, self, out_int32, right, out);
  }
  increment_version(out);
  return out;
}
Tensor select_int(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, int64_t index) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::select(ks & c10::after_InplaceOrView_keyset, self, dim, index);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::select(input_base, dim, index);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & sign_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sign_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & sign_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sign_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & silu_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::silu_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & silu_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::silu_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & sinc_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sinc_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & sinc_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sinc_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor slice_Tensor(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::slice(ks & c10::after_InplaceOrView_keyset, self, dim, start, end, step);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    auto start_val = start.value_or(0);
    auto end_val = end.value_or(0);
    func = [=](const at::Tensor& input_base) {
      return at::slice(input_base, dim, start_val, end_val, step);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::slow_conv3d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, grad_input, grad_weight, grad_bias);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv3d_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, Tensor & output, Tensor & finput, Tensor & fgrad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::slow_conv3d_forward_outf(ks & c10::after_InplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output, finput, fgrad_input);
  }
  increment_version(output);
  increment_version(finput);
  increment_version(fgrad_input);
  return std::forward_as_tuple(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv_transpose3d_backward_out_grad_output(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input, Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::slow_conv_transpose3d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, grad_input, grad_weight, grad_bias);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
Tensor & slow_conv_transpose3d_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::slow_conv_transpose3d_outf(ks & c10::after_InplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
  }
  increment_version(out);
  return out;
}
Tensor & soft_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::soft_margin_loss_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, target, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & soft_margin_loss_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::soft_margin_loss_outf(ks & c10::after_InplaceOrView_keyset, self, target, reduction, out);
  }
  increment_version(out);
  return out;
}
Tensor & softshrink_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Scalar & lambd, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::softshrink_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, lambd, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & softshrink_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & lambd, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::softshrink_outf(ks & c10::after_InplaceOrView_keyset, self, lambd, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> solve_out_solution(c10::DispatchKeySet ks, const Tensor & self, const Tensor & A, Tensor & solution, Tensor & lu) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::solve_outf(ks & c10::after_InplaceOrView_keyset, self, A, solution, lu);
  }
  increment_version(solution);
  increment_version(lu);
  return std::forward_as_tuple(solution, lu);
}
Tensor & sparse_resize_(c10::DispatchKeySet ks, Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sparse_resize_(ks & c10::after_InplaceOrView_keyset, self, size, sparse_dim, dense_dim);
  }
  increment_version(self);
  return self;
}
Tensor & sparse_resize_and_clear_(c10::DispatchKeySet ks, Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sparse_resize_and_clear_(ks & c10::after_InplaceOrView_keyset, self, size, sparse_dim, dense_dim);
  }
  increment_version(self);
  return self;
}
Tensor & special_entr_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::special_entr_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
std::vector<Tensor> split_Tensor(c10::DispatchKeySet ks, const Tensor & self, int64_t split_size, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::split(ks & c10::after_InplaceOrView_keyset, self, split_size, dim);
  })();
  as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* creation_meta */ CreationMeta::MULTI_OUTPUT_SAFE);
  auto result = std::move(_tmp);
  return result;
}
std::vector<Tensor> split_with_sizes(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef split_sizes, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::split_with_sizes(ks & c10::after_InplaceOrView_keyset, self, split_sizes, dim);
  })();
  as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* creation_meta */ CreationMeta::MULTI_OUTPUT_SAFE);
  auto result = std::move(_tmp);
  return result;
}
Tensor & square_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::square_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor squeeze(c10::DispatchKeySet ks, const Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::squeeze(ks & c10::after_InplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::squeeze(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor squeeze_dim(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::squeeze(ks & c10::after_InplaceOrView_keyset, self, dim);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::squeeze(input_base, dim);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & sspaddmm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sspaddmm_outf(ks & c10::after_InplaceOrView_keyset, self, mat1, mat2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
Tensor & std_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::std_outf(ks & c10::after_InplaceOrView_keyset, self, dim, unbiased, keepdim, out);
  }
  increment_version(out);
  return out;
}
Tensor & sum_out_IntList_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sum_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
Tensor t(c10::DispatchKeySet ks, const Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::t(ks & c10::after_InplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::t(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & take_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & index, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::take_outf(ks & c10::after_InplaceOrView_keyset, self, index, out);
  }
  increment_version(out);
  return out;
}
Tensor & tan_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::tan_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & tan_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::tan_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & tanh_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::tanh_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & tanh_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & output, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::tanh_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, output, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & tanh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::tanh_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::thnn_conv2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, grad_input, grad_weight, grad_bias);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, Tensor & output, Tensor & finput, Tensor & fgrad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::thnn_conv2d_forward_outf(ks & c10::after_InplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output, finput, fgrad_input);
  }
  increment_version(output);
  increment_version(finput);
  increment_version(fgrad_input);
  return std::forward_as_tuple(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor & grad_input, Tensor & grad_weight) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::thnn_conv_depthwise2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, dilation, grad_input, grad_weight);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  return std::forward_as_tuple(grad_input, grad_weight);
}
Tensor & thnn_conv_depthwise2d_forward_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::thnn_conv_depthwise2d_forward_outf(ks & c10::after_InplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, dilation, out);
  }
  increment_version(out);
  return out;
}
Tensor & threshold_(c10::DispatchKeySet ks, Tensor & self, const Scalar & threshold, const Scalar & value) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::threshold_(ks & c10::after_InplaceOrView_keyset, self, threshold, value);
  }
  increment_version(self);
  return self;
}
Tensor & threshold_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & threshold, const Scalar & value, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::threshold_outf(ks & c10::after_InplaceOrView_keyset, self, threshold, value, out);
  }
  increment_version(out);
  return out;
}
Tensor & transpose_(c10::DispatchKeySet ks, Tensor & self, int64_t dim0, int64_t dim1) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::transpose_(ks & c10::after_InplaceOrView_keyset, self, dim0, dim1);
  }
  increment_version(self);
  return self;
}
std::tuple<Tensor &,Tensor &> triangular_solve_out_X(c10::DispatchKeySet ks, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular, Tensor & X, Tensor & M) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::triangular_solve_outf(ks & c10::after_InplaceOrView_keyset, self, A, upper, transpose, unitriangular, X, M);
  }
  increment_version(X);
  increment_version(M);
  return std::forward_as_tuple(X, M);
}
Tensor & tril_(c10::DispatchKeySet ks, Tensor & self, int64_t diagonal) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::tril_(ks & c10::after_InplaceOrView_keyset, self, diagonal);
  }
  increment_version(self);
  return self;
}
Tensor & tril_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t diagonal, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::tril_outf(ks & c10::after_InplaceOrView_keyset, self, diagonal, out);
  }
  increment_version(out);
  return out;
}
std::vector<Tensor> unbind_int(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::unbind(ks & c10::after_InplaceOrView_keyset, self, dim);
  })();
  as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* creation_meta */ CreationMeta::MULTI_OUTPUT_NODE);
  auto result = std::move(_tmp);
  return result;
}
Tensor unfold(c10::DispatchKeySet ks, const Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::unfold(ks & c10::after_InplaceOrView_keyset, self, dimension, size, step);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return input_base.unfold(dimension, size, step);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & unsqueeze_(c10::DispatchKeySet ks, Tensor & self, int64_t dim) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::unsqueeze_(ks & c10::after_InplaceOrView_keyset, self, dim);
  }
  increment_version(self);
  return self;
}
Tensor & upsample_bicubic2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_bicubic2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & upsample_bicubic2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_bicubic2d_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, align_corners, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
Tensor & upsample_nearest1d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_nearest1d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, output_size, input_size, scales, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & upsample_nearest1d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_nearest1d_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, scales, out);
  }
  increment_version(out);
  return out;
}
Tensor & upsample_nearest3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_nearest3d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & upsample_nearest3d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_nearest3d_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, scales_d, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
Tensor & upsample_trilinear3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_trilinear3d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & upsample_trilinear3d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_trilinear3d_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, align_corners, scales_d, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
Tensor values(c10::DispatchKeySet ks, const Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::values(ks & c10::after_InplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return input_base.values();
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & var_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::var_outf(ks & c10::after_InplaceOrView_keyset, self, dim, unbiased, keepdim, out);
  }
  increment_version(out);
  return out;
}
Tensor & vdot_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::vdot_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & xlogy__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::xlogy_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & xlogy__Scalar_Other(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::xlogy_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & xlogy_out_OutTensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::xlogy_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & xlogy_out_OutScalar_Self(c10::DispatchKeySet ks, const Scalar & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::xlogy_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & xlogy_out_OutScalar_Other(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::xlogy_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
}  // namespace
}  // namespace InplaceOrView

namespace {

TORCH_LIBRARY_IMPL(aten, InplaceOrView, m) {
  m.impl("_add_relu_.Tensor",
         TORCH_FN(InplaceOrView::_add_relu__Tensor)
  );
  m.impl("_add_relu.out",
         TORCH_FN(InplaceOrView::_add_relu_out_out)
  );
  m.impl("_bmm.out",
         TORCH_FN(InplaceOrView::_bmm_out_out)
  );
  m.impl("_cat.out",
         TORCH_FN(InplaceOrView::_cat_out_out)
  );
  m.impl("_cumprod.out",
         TORCH_FN(InplaceOrView::_cumprod_out_out)
  );
  m.impl("_cumsum.out",
         TORCH_FN(InplaceOrView::_cumsum_out_out)
  );
  m.impl("_fft_c2r.out",
         TORCH_FN(InplaceOrView::_fft_c2r_out_out)
  );
  m.impl("_fft_r2c.out",
         TORCH_FN(InplaceOrView::_fft_r2c_out_out)
  );
  m.impl("_index_copy_",
         TORCH_FN(InplaceOrView::_index_copy_)
  );
  m.impl("_indices",
         TORCH_FN(InplaceOrView::_indices)
  );
  m.impl("_mkldnn_transpose_",
         TORCH_FN(InplaceOrView::_mkldnn_transpose_)
  );
  m.impl("_stack.out",
         TORCH_FN(InplaceOrView::_stack_out_out)
  );
  m.impl("adaptive_avg_pool3d_backward.grad_input",
         TORCH_FN(InplaceOrView::adaptive_avg_pool3d_backward_out_grad_input)
  );
  m.impl("adaptive_avg_pool3d.out",
         TORCH_FN(InplaceOrView::adaptive_avg_pool3d_out_out)
  );
  m.impl("adaptive_max_pool3d_backward.grad_input",
         TORCH_FN(InplaceOrView::adaptive_max_pool3d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool3d.out",
         TORCH_FN(InplaceOrView::adaptive_max_pool3d_out_out)
  );
  m.impl("add_.Tensor",
         TORCH_FN(InplaceOrView::add__Tensor)
  );
  m.impl("add_.Scalar",
         TORCH_FN(InplaceOrView::add__Scalar)
  );
  m.impl("add.out",
         TORCH_FN(InplaceOrView::add_out_out)
  );
  m.impl("addbmm_",
         TORCH_FN(InplaceOrView::addbmm_)
  );
  m.impl("addbmm.out",
         TORCH_FN(InplaceOrView::addbmm_out_out)
  );
  m.impl("addcdiv_",
         TORCH_FN(InplaceOrView::addcdiv_)
  );
  m.impl("addcdiv.out",
         TORCH_FN(InplaceOrView::addcdiv_out_out)
  );
  m.impl("addmm_",
         TORCH_FN(InplaceOrView::addmm_)
  );
  m.impl("addmm.out",
         TORCH_FN(InplaceOrView::addmm_out_out)
  );
  m.impl("addr_",
         TORCH_FN(InplaceOrView::addr_)
  );
  m.impl("addr.out",
         TORCH_FN(InplaceOrView::addr_out_out)
  );
  m.impl("alias",
         TORCH_FN(InplaceOrView::alias)
  );
  m.impl("all.out",
         TORCH_FN(InplaceOrView::all_out_out)
  );
  m.impl("amax.out",
         TORCH_FN(InplaceOrView::amax_out_out)
  );
  m.impl("amin.out",
         TORCH_FN(InplaceOrView::amin_out_out)
  );
  m.impl("angle.out",
         TORCH_FN(InplaceOrView::angle_out_out)
  );
  m.impl("as_strided",
         TORCH_FN(InplaceOrView::as_strided)
  );
  m.impl("asin_",
         TORCH_FN(InplaceOrView::asin_)
  );
  m.impl("asin.out",
         TORCH_FN(InplaceOrView::asin_out_out)
  );
  m.impl("asinh_",
         TORCH_FN(InplaceOrView::asinh_)
  );
  m.impl("asinh.out",
         TORCH_FN(InplaceOrView::asinh_out_out)
  );
  m.impl("avg_pool2d_backward.grad_input",
         TORCH_FN(InplaceOrView::avg_pool2d_backward_out_grad_input)
  );
  m.impl("avg_pool2d.out",
         TORCH_FN(InplaceOrView::avg_pool2d_out_out)
  );
  m.impl("baddbmm_",
         TORCH_FN(InplaceOrView::baddbmm_)
  );
  m.impl("baddbmm.out",
         TORCH_FN(InplaceOrView::baddbmm_out_out)
  );
  m.impl("batch_norm_elemt.out",
         TORCH_FN(InplaceOrView::batch_norm_elemt_out_out)
  );
  m.impl("bitwise_and.Tensor_out",
         TORCH_FN(InplaceOrView::bitwise_and_out_Tensor_out)
  );
  m.impl("bitwise_and.Scalar_out",
         TORCH_FN(InplaceOrView::bitwise_and_out_Scalar_out)
  );
  m.impl("bitwise_not.out",
         TORCH_FN(InplaceOrView::bitwise_not_out_out)
  );
  m.impl("bitwise_or.Tensor_out",
         TORCH_FN(InplaceOrView::bitwise_or_out_Tensor_out)
  );
  m.impl("bitwise_or.Scalar_out",
         TORCH_FN(InplaceOrView::bitwise_or_out_Scalar_out)
  );
  m.impl("bitwise_xor.Tensor_out",
         TORCH_FN(InplaceOrView::bitwise_xor_out_Tensor_out)
  );
  m.impl("bitwise_xor.Scalar_out",
         TORCH_FN(InplaceOrView::bitwise_xor_out_Scalar_out)
  );
  m.impl("cauchy_",
         TORCH_FN(InplaceOrView::cauchy_)
  );
  m.impl("ceil_",
         TORCH_FN(InplaceOrView::ceil_)
  );
  m.impl("ceil.out",
         TORCH_FN(InplaceOrView::ceil_out_out)
  );
  m.impl("celu_",
         TORCH_FN(InplaceOrView::celu_)
  );
  m.impl("cholesky_inverse.out",
         TORCH_FN(InplaceOrView::cholesky_inverse_out_out)
  );
  m.impl("clamp_",
         TORCH_FN(InplaceOrView::clamp_)
  );
  m.impl("clamp.out",
         TORCH_FN(InplaceOrView::clamp_out_out)
  );
  m.impl("conv_depthwise3d_backward.grad_input",
         TORCH_FN(InplaceOrView::conv_depthwise3d_backward_out_grad_input)
  );
  m.impl("copy_sparse_to_sparse_",
         TORCH_FN(InplaceOrView::copy_sparse_to_sparse_)
  );
  m.impl("cos_",
         TORCH_FN(InplaceOrView::cos_)
  );
  m.impl("cos.out",
         TORCH_FN(InplaceOrView::cos_out_out)
  );
  m.impl("cosh_",
         TORCH_FN(InplaceOrView::cosh_)
  );
  m.impl("cosh.out",
         TORCH_FN(InplaceOrView::cosh_out_out)
  );
  m.impl("cummax.out",
         TORCH_FN(InplaceOrView::cummax_out_out)
  );
  m.impl("cummin.out",
         TORCH_FN(InplaceOrView::cummin_out_out)
  );
  m.impl("deg2rad_",
         TORCH_FN(InplaceOrView::deg2rad_)
  );
  m.impl("deg2rad.out",
         TORCH_FN(InplaceOrView::deg2rad_out_out)
  );
  m.impl("diag.out",
         TORCH_FN(InplaceOrView::diag_out_out)
  );
  m.impl("div_.Tensor",
         TORCH_FN(InplaceOrView::div__Tensor)
  );
  m.impl("div_.Tensor_mode",
         TORCH_FN(InplaceOrView::div__Tensor_mode)
  );
  m.impl("div_.Scalar",
         TORCH_FN(InplaceOrView::div__Scalar)
  );
  m.impl("div_.Scalar_mode",
         TORCH_FN(InplaceOrView::div__Scalar_mode)
  );
  m.impl("div.out",
         TORCH_FN(InplaceOrView::div_out_out)
  );
  m.impl("div.out_mode",
         TORCH_FN(InplaceOrView::div_out_out_mode)
  );
  m.impl("dot.out",
         TORCH_FN(InplaceOrView::dot_out_out)
  );
  m.impl("eig.e",
         TORCH_FN(InplaceOrView::eig_out_e)
  );
  m.impl("embedding_renorm_",
         TORCH_FN(InplaceOrView::embedding_renorm_)
  );
  m.impl("erf_",
         TORCH_FN(InplaceOrView::erf_)
  );
  m.impl("erf.out",
         TORCH_FN(InplaceOrView::erf_out_out)
  );
  m.impl("exp2_",
         TORCH_FN(InplaceOrView::exp2_)
  );
  m.impl("exp2.out",
         TORCH_FN(InplaceOrView::exp2_out_out)
  );
  m.impl("exp_",
         TORCH_FN(InplaceOrView::exp_)
  );
  m.impl("exp.out",
         TORCH_FN(InplaceOrView::exp_out_out)
  );
  m.impl("expand",
         TORCH_FN(InplaceOrView::expand)
  );
  m.impl("expm1_",
         TORCH_FN(InplaceOrView::expm1_)
  );
  m.impl("expm1.out",
         TORCH_FN(InplaceOrView::expm1_out_out)
  );
  m.impl("exponential_",
         TORCH_FN(InplaceOrView::exponential_)
  );
  m.impl("eye.out",
         TORCH_FN(InplaceOrView::eye_out_out)
  );
  m.impl("eye.m_out",
         TORCH_FN(InplaceOrView::eye_out_m_out)
  );
  m.impl("fill_.Scalar",
         TORCH_FN(InplaceOrView::fill__Scalar)
  );
  m.impl("fill_.Tensor",
         TORCH_FN(InplaceOrView::fill__Tensor)
  );
  m.impl("fractional_max_pool2d_backward.grad_input",
         TORCH_FN(InplaceOrView::fractional_max_pool2d_backward_out_grad_input)
  );
  m.impl("fractional_max_pool2d.output",
         TORCH_FN(InplaceOrView::fractional_max_pool2d_out_output)
  );
  m.impl("frexp.Tensor_out",
         TORCH_FN(InplaceOrView::frexp_out_Tensor_out)
  );
  m.impl("gather.out",
         TORCH_FN(InplaceOrView::gather_out_out)
  );
  m.impl("geometric_",
         TORCH_FN(InplaceOrView::geometric_)
  );
  m.impl("geqrf.a",
         TORCH_FN(InplaceOrView::geqrf_out_a)
  );
  m.impl("gt_.Scalar",
         TORCH_FN(InplaceOrView::gt__Scalar)
  );
  m.impl("gt_.Tensor",
         TORCH_FN(InplaceOrView::gt__Tensor)
  );
  m.impl("gt.Scalar_out",
         TORCH_FN(InplaceOrView::gt_out_Scalar_out)
  );
  m.impl("gt.Tensor_out",
         TORCH_FN(InplaceOrView::gt_out_Tensor_out)
  );
  m.impl("hardsigmoid_",
         TORCH_FN(InplaceOrView::hardsigmoid_)
  );
  m.impl("hardsigmoid.out",
         TORCH_FN(InplaceOrView::hardsigmoid_out_out)
  );
  m.impl("hardswish_",
         TORCH_FN(InplaceOrView::hardswish_)
  );
  m.impl("hardswish.out",
         TORCH_FN(InplaceOrView::hardswish_out_out)
  );
  m.impl("histc.out",
         TORCH_FN(InplaceOrView::histc_out_out)
  );
  m.impl("hspmm.out",
         TORCH_FN(InplaceOrView::hspmm_out_out)
  );
  m.impl("i0_",
         TORCH_FN(InplaceOrView::i0_)
  );
  m.impl("i0.out",
         TORCH_FN(InplaceOrView::i0_out_out)
  );
  m.impl("igammac_",
         TORCH_FN(InplaceOrView::igammac_)
  );
  m.impl("igammac.out",
         TORCH_FN(InplaceOrView::igammac_out_out)
  );
  m.impl("index_select.out",
         TORCH_FN(InplaceOrView::index_select_out_out)
  );
  m.impl("isneginf.out",
         TORCH_FN(InplaceOrView::isneginf_out_out)
  );
  m.impl("isposinf.out",
         TORCH_FN(InplaceOrView::isposinf_out_out)
  );
  m.impl("l1_loss_backward.grad_input",
         TORCH_FN(InplaceOrView::l1_loss_backward_out_grad_input)
  );
  m.impl("l1_loss.out",
         TORCH_FN(InplaceOrView::l1_loss_out_out)
  );
  m.impl("le_.Scalar",
         TORCH_FN(InplaceOrView::le__Scalar)
  );
  m.impl("le_.Tensor",
         TORCH_FN(InplaceOrView::le__Tensor)
  );
  m.impl("le.Scalar_out",
         TORCH_FN(InplaceOrView::le_out_Scalar_out)
  );
  m.impl("le.Tensor_out",
         TORCH_FN(InplaceOrView::le_out_Tensor_out)
  );
  m.impl("leaky_relu_",
         TORCH_FN(InplaceOrView::leaky_relu_)
  );
  m.impl("leaky_relu.out",
         TORCH_FN(InplaceOrView::leaky_relu_out_out)
  );
  m.impl("lerp_.Scalar",
         TORCH_FN(InplaceOrView::lerp__Scalar)
  );
  m.impl("lerp_.Tensor",
         TORCH_FN(InplaceOrView::lerp__Tensor)
  );
  m.impl("lerp.Scalar_out",
         TORCH_FN(InplaceOrView::lerp_out_Scalar_out)
  );
  m.impl("lerp.Tensor_out",
         TORCH_FN(InplaceOrView::lerp_out_Tensor_out)
  );
  m.impl("lgamma_",
         TORCH_FN(InplaceOrView::lgamma_)
  );
  m.impl("lgamma.out",
         TORCH_FN(InplaceOrView::lgamma_out_out)
  );
  m.impl("linalg_eigh.eigvals",
         TORCH_FN(InplaceOrView::linalg_eigh_out_eigvals)
  );
  m.impl("linalg_eigvalsh.out",
         TORCH_FN(InplaceOrView::linalg_eigvalsh_out_out)
  );
  m.impl("linalg_inv.out",
         TORCH_FN(InplaceOrView::linalg_inv_out_out)
  );
  m.impl("linalg_qr.out",
         TORCH_FN(InplaceOrView::linalg_qr_out_out)
  );
  m.impl("linalg_solve.out",
         TORCH_FN(InplaceOrView::linalg_solve_out_out)
  );
  m.impl("linspace.out",
         TORCH_FN(InplaceOrView::linspace_out_out)
  );
  m.impl("log10_",
         TORCH_FN(InplaceOrView::log10_)
  );
  m.impl("log10.out",
         TORCH_FN(InplaceOrView::log10_out_out)
  );
  m.impl("log1p_",
         TORCH_FN(InplaceOrView::log1p_)
  );
  m.impl("log1p.out",
         TORCH_FN(InplaceOrView::log1p_out_out)
  );
  m.impl("log_sigmoid_backward.grad_input",
         TORCH_FN(InplaceOrView::log_sigmoid_backward_out_grad_input)
  );
  m.impl("log_sigmoid_forward.output",
         TORCH_FN(InplaceOrView::log_sigmoid_forward_out_output)
  );
  m.impl("logcumsumexp.out",
         TORCH_FN(InplaceOrView::logcumsumexp_out_out)
  );
  m.impl("logical_and.out",
         TORCH_FN(InplaceOrView::logical_and_out_out)
  );
  m.impl("logical_not.out",
         TORCH_FN(InplaceOrView::logical_not_out_out)
  );
  m.impl("logical_or.out",
         TORCH_FN(InplaceOrView::logical_or_out_out)
  );
  m.impl("logical_xor.out",
         TORCH_FN(InplaceOrView::logical_xor_out_out)
  );
  m.impl("logit_",
         TORCH_FN(InplaceOrView::logit_)
  );
  m.impl("logit_backward.grad_input",
         TORCH_FN(InplaceOrView::logit_backward_out_grad_input)
  );
  m.impl("logit.out",
         TORCH_FN(InplaceOrView::logit_out_out)
  );
  m.impl("lstsq.X",
         TORCH_FN(InplaceOrView::lstsq_out_X)
  );
  m.impl("lu_solve.out",
         TORCH_FN(InplaceOrView::lu_solve_out_out)
  );
  m.impl("masked_fill_.Scalar",
         TORCH_FN(InplaceOrView::masked_fill__Scalar)
  );
  m.impl("masked_fill_.Tensor",
         TORCH_FN(InplaceOrView::masked_fill__Tensor)
  );
  m.impl("max_pool3d_with_indices_backward.grad_input",
         TORCH_FN(InplaceOrView::max_pool3d_with_indices_backward_out_grad_input)
  );
  m.impl("max_pool3d_with_indices.out",
         TORCH_FN(InplaceOrView::max_pool3d_with_indices_out_out)
  );
  m.impl("max_unpool3d_backward.grad_input",
         TORCH_FN(InplaceOrView::max_unpool3d_backward_out_grad_input)
  );
  m.impl("max_unpool3d.out",
         TORCH_FN(InplaceOrView::max_unpool3d_out_out)
  );
  m.impl("mean.out",
         TORCH_FN(InplaceOrView::mean_out_out)
  );
  m.impl("mode.values",
         TORCH_FN(InplaceOrView::mode_out_values)
  );
  m.impl("mse_loss_backward.grad_input",
         TORCH_FN(InplaceOrView::mse_loss_backward_out_grad_input)
  );
  m.impl("mse_loss.out",
         TORCH_FN(InplaceOrView::mse_loss_out_out)
  );
  m.impl("multinomial.out",
         TORCH_FN(InplaceOrView::multinomial_out_out)
  );
  m.impl("mv.out",
         TORCH_FN(InplaceOrView::mv_out_out)
  );
  m.impl("nanmedian.dim_values",
         TORCH_FN(InplaceOrView::nanmedian_out_dim_values)
  );
  m.impl("narrow_copy.out",
         TORCH_FN(InplaceOrView::narrow_copy_out_out)
  );
  m.impl("native_batch_norm.out",
         TORCH_FN(InplaceOrView::native_batch_norm_out_out)
  );
  m.impl("ne_.Scalar",
         TORCH_FN(InplaceOrView::ne__Scalar)
  );
  m.impl("ne_.Tensor",
         TORCH_FN(InplaceOrView::ne__Tensor)
  );
  m.impl("ne.Scalar_out",
         TORCH_FN(InplaceOrView::ne_out_Scalar_out)
  );
  m.impl("ne.Tensor_out",
         TORCH_FN(InplaceOrView::ne_out_Tensor_out)
  );
  m.impl("nextafter_",
         TORCH_FN(InplaceOrView::nextafter_)
  );
  m.impl("nextafter.out",
         TORCH_FN(InplaceOrView::nextafter_out_out)
  );
  m.impl("nonzero.out",
         TORCH_FN(InplaceOrView::nonzero_out_out)
  );
  m.impl("normal_",
         TORCH_FN(InplaceOrView::normal_)
  );
  m.impl("normal.Tensor_float_out",
         TORCH_FN(InplaceOrView::normal_out_Tensor_float_out)
  );
  m.impl("normal.float_Tensor_out",
         TORCH_FN(InplaceOrView::normal_out_float_Tensor_out)
  );
  m.impl("normal.Tensor_Tensor_out",
         TORCH_FN(InplaceOrView::normal_out_Tensor_Tensor_out)
  );
  m.impl("ormqr.out",
         TORCH_FN(InplaceOrView::ormqr_out_out)
  );
  m.impl("permute",
         TORCH_FN(InplaceOrView::permute)
  );
  m.impl("polygamma.out",
         TORCH_FN(InplaceOrView::polygamma_out_out)
  );
  m.impl("prod.int_out",
         TORCH_FN(InplaceOrView::prod_out_int_out)
  );
  m.impl("put_",
         TORCH_FN(InplaceOrView::put_)
  );
  m.impl("rad2deg_",
         TORCH_FN(InplaceOrView::rad2deg_)
  );
  m.impl("rad2deg.out",
         TORCH_FN(InplaceOrView::rad2deg_out_out)
  );
  m.impl("random_.from",
         TORCH_FN(InplaceOrView::random__from)
  );
  m.impl("random_.to",
         TORCH_FN(InplaceOrView::random__to)
  );
  m.impl("random_",
         TORCH_FN(InplaceOrView::random_)
  );
  m.impl("randperm.generator_out",
         TORCH_FN(InplaceOrView::randperm_out_generator_out)
  );
  m.impl("range.out",
         TORCH_FN(InplaceOrView::range_out_out)
  );
  m.impl("reflection_pad2d_backward.grad_input",
         TORCH_FN(InplaceOrView::reflection_pad2d_backward_out_grad_input)
  );
  m.impl("reflection_pad2d.out",
         TORCH_FN(InplaceOrView::reflection_pad2d_out_out)
  );
  m.impl("remainder_.Scalar",
         TORCH_FN(InplaceOrView::remainder__Scalar)
  );
  m.impl("remainder_.Tensor",
         TORCH_FN(InplaceOrView::remainder__Tensor)
  );
  m.impl("remainder.Scalar_out",
         TORCH_FN(InplaceOrView::remainder_out_Scalar_out)
  );
  m.impl("remainder.Tensor_out",
         TORCH_FN(InplaceOrView::remainder_out_Tensor_out)
  );
  m.impl("renorm_",
         TORCH_FN(InplaceOrView::renorm_)
  );
  m.impl("renorm.out",
         TORCH_FN(InplaceOrView::renorm_out_out)
  );
  m.impl("replication_pad1d_backward.grad_input",
         TORCH_FN(InplaceOrView::replication_pad1d_backward_out_grad_input)
  );
  m.impl("replication_pad1d.out",
         TORCH_FN(InplaceOrView::replication_pad1d_out_out)
  );
  m.impl("replication_pad3d_backward.grad_input",
         TORCH_FN(InplaceOrView::replication_pad3d_backward_out_grad_input)
  );
  m.impl("replication_pad3d.out",
         TORCH_FN(InplaceOrView::replication_pad3d_out_out)
  );
  m.impl("searchsorted.Tensor_out",
         TORCH_FN(InplaceOrView::searchsorted_out_Tensor_out)
  );
  m.impl("select.int",
         TORCH_FN(InplaceOrView::select_int)
  );
  m.impl("sign_",
         TORCH_FN(InplaceOrView::sign_)
  );
  m.impl("sign.out",
         TORCH_FN(InplaceOrView::sign_out_out)
  );
  m.impl("silu_",
         TORCH_FN(InplaceOrView::silu_)
  );
  m.impl("silu.out",
         TORCH_FN(InplaceOrView::silu_out_out)
  );
  m.impl("sinc_",
         TORCH_FN(InplaceOrView::sinc_)
  );
  m.impl("sinc.out",
         TORCH_FN(InplaceOrView::sinc_out_out)
  );
  m.impl("slice.Tensor",
         TORCH_FN(InplaceOrView::slice_Tensor)
  );
  m.impl("slow_conv3d_backward.grad_input",
         TORCH_FN(InplaceOrView::slow_conv3d_backward_out_grad_input)
  );
  m.impl("slow_conv3d_forward.output",
         TORCH_FN(InplaceOrView::slow_conv3d_forward_out_output)
  );
  m.impl("slow_conv_transpose3d_backward.grad_output",
         TORCH_FN(InplaceOrView::slow_conv_transpose3d_backward_out_grad_output)
  );
  m.impl("slow_conv_transpose3d.out",
         TORCH_FN(InplaceOrView::slow_conv_transpose3d_out_out)
  );
  m.impl("soft_margin_loss_backward.grad_input",
         TORCH_FN(InplaceOrView::soft_margin_loss_backward_out_grad_input)
  );
  m.impl("soft_margin_loss.out",
         TORCH_FN(InplaceOrView::soft_margin_loss_out_out)
  );
  m.impl("softshrink_backward.grad_input",
         TORCH_FN(InplaceOrView::softshrink_backward_out_grad_input)
  );
  m.impl("softshrink.out",
         TORCH_FN(InplaceOrView::softshrink_out_out)
  );
  m.impl("solve.solution",
         TORCH_FN(InplaceOrView::solve_out_solution)
  );
  m.impl("sparse_resize_",
         TORCH_FN(InplaceOrView::sparse_resize_)
  );
  m.impl("sparse_resize_and_clear_",
         TORCH_FN(InplaceOrView::sparse_resize_and_clear_)
  );
  m.impl("special_entr.out",
         TORCH_FN(InplaceOrView::special_entr_out_out)
  );
  m.impl("split.Tensor",
         TORCH_FN(InplaceOrView::split_Tensor)
  );
  m.impl("split_with_sizes",
         TORCH_FN(InplaceOrView::split_with_sizes)
  );
  m.impl("square.out",
         TORCH_FN(InplaceOrView::square_out_out)
  );
  m.impl("squeeze",
         TORCH_FN(InplaceOrView::squeeze)
  );
  m.impl("squeeze.dim",
         TORCH_FN(InplaceOrView::squeeze_dim)
  );
  m.impl("sspaddmm.out",
         TORCH_FN(InplaceOrView::sspaddmm_out_out)
  );
  m.impl("std.out",
         TORCH_FN(InplaceOrView::std_out_out)
  );
  m.impl("sum.IntList_out",
         TORCH_FN(InplaceOrView::sum_out_IntList_out)
  );
  m.impl("t",
         TORCH_FN(InplaceOrView::t)
  );
  m.impl("take.out",
         TORCH_FN(InplaceOrView::take_out_out)
  );
  m.impl("tan_",
         TORCH_FN(InplaceOrView::tan_)
  );
  m.impl("tan.out",
         TORCH_FN(InplaceOrView::tan_out_out)
  );
  m.impl("tanh_",
         TORCH_FN(InplaceOrView::tanh_)
  );
  m.impl("tanh_backward.grad_input",
         TORCH_FN(InplaceOrView::tanh_backward_out_grad_input)
  );
  m.impl("tanh.out",
         TORCH_FN(InplaceOrView::tanh_out_out)
  );
  m.impl("thnn_conv2d_backward.grad_input",
         TORCH_FN(InplaceOrView::thnn_conv2d_backward_out_grad_input)
  );
  m.impl("thnn_conv2d_forward.output",
         TORCH_FN(InplaceOrView::thnn_conv2d_forward_out_output)
  );
  m.impl("thnn_conv_depthwise2d_backward.grad_input",
         TORCH_FN(InplaceOrView::thnn_conv_depthwise2d_backward_out_grad_input)
  );
  m.impl("thnn_conv_depthwise2d_forward.out",
         TORCH_FN(InplaceOrView::thnn_conv_depthwise2d_forward_out_out)
  );
  m.impl("threshold_",
         TORCH_FN(InplaceOrView::threshold_)
  );
  m.impl("threshold.out",
         TORCH_FN(InplaceOrView::threshold_out_out)
  );
  m.impl("transpose_",
         TORCH_FN(InplaceOrView::transpose_)
  );
  m.impl("triangular_solve.X",
         TORCH_FN(InplaceOrView::triangular_solve_out_X)
  );
  m.impl("tril_",
         TORCH_FN(InplaceOrView::tril_)
  );
  m.impl("tril.out",
         TORCH_FN(InplaceOrView::tril_out_out)
  );
  m.impl("unbind.int",
         TORCH_FN(InplaceOrView::unbind_int)
  );
  m.impl("unfold",
         TORCH_FN(InplaceOrView::unfold)
  );
  m.impl("unsqueeze_",
         TORCH_FN(InplaceOrView::unsqueeze_)
  );
  m.impl("upsample_bicubic2d_backward.grad_input",
         TORCH_FN(InplaceOrView::upsample_bicubic2d_backward_out_grad_input)
  );
  m.impl("upsample_bicubic2d.out",
         TORCH_FN(InplaceOrView::upsample_bicubic2d_out_out)
  );
  m.impl("upsample_nearest1d_backward.grad_input",
         TORCH_FN(InplaceOrView::upsample_nearest1d_backward_out_grad_input)
  );
  m.impl("upsample_nearest1d.out",
         TORCH_FN(InplaceOrView::upsample_nearest1d_out_out)
  );
  m.impl("upsample_nearest3d_backward.grad_input",
         TORCH_FN(InplaceOrView::upsample_nearest3d_backward_out_grad_input)
  );
  m.impl("upsample_nearest3d.out",
         TORCH_FN(InplaceOrView::upsample_nearest3d_out_out)
  );
  m.impl("upsample_trilinear3d_backward.grad_input",
         TORCH_FN(InplaceOrView::upsample_trilinear3d_backward_out_grad_input)
  );
  m.impl("upsample_trilinear3d.out",
         TORCH_FN(InplaceOrView::upsample_trilinear3d_out_out)
  );
  m.impl("values",
         TORCH_FN(InplaceOrView::values)
  );
  m.impl("var.out",
         TORCH_FN(InplaceOrView::var_out_out)
  );
  m.impl("vdot.out",
         TORCH_FN(InplaceOrView::vdot_out_out)
  );
  m.impl("xlogy_.Tensor",
         TORCH_FN(InplaceOrView::xlogy__Tensor)
  );
  m.impl("xlogy_.Scalar_Other",
         TORCH_FN(InplaceOrView::xlogy__Scalar_Other)
  );
  m.impl("xlogy.OutTensor",
         TORCH_FN(InplaceOrView::xlogy_out_OutTensor)
  );
  m.impl("xlogy.OutScalar_Self",
         TORCH_FN(InplaceOrView::xlogy_out_OutScalar_Self)
  );
  m.impl("xlogy.OutScalar_Other",
         TORCH_FN(InplaceOrView::xlogy_out_OutScalar_Other)
  );;
}

}  // namespace
} // namespace torch

