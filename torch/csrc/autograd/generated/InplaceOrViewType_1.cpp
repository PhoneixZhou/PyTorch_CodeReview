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
Tensor & __ilshift___Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::__ilshift__(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & __ilshift___Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::__ilshift__(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & __irshift___Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::__irshift__(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & __irshift___Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::__irshift__(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & _addmv_impl_(c10::DispatchKeySet ks, Tensor & self, const Tensor & self2, const Tensor & mat, const Tensor & vec, const Scalar & beta, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_addmv_impl_(ks & c10::after_InplaceOrView_keyset, self, self2, mat, vec, beta, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & _coalesced_(c10::DispatchKeySet ks, Tensor & self, bool coalesced) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_coalesced_(ks & c10::after_InplaceOrView_keyset, self, coalesced);
  }
  increment_version(self);
  return self;
}
Tensor & _compute_linear_combination_out_out(c10::DispatchKeySet ks, const Tensor & input, const Tensor & coefficients, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_compute_linear_combination_outf(ks & c10::after_InplaceOrView_keyset, input, coefficients, out);
  }
  increment_version(out);
  return out;
}
Tensor & _fft_c2c_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, int64_t normalization, bool forward, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_fft_c2c_outf(ks & c10::after_InplaceOrView_keyset, self, dim, normalization, forward, out);
  }
  increment_version(out);
  return out;
}
Tensor & _index_put_impl_(c10::DispatchKeySet ks, Tensor & self, const c10::List<c10::optional<Tensor>> & indices, const Tensor & values, bool accumulate, bool unsafe) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_index_put_impl_(ks & c10::after_InplaceOrView_keyset, self, indices, values, accumulate, unsafe);
  }
  increment_version(self);
  return self;
}
Tensor & _linalg_inv_out_helper_(c10::DispatchKeySet ks, Tensor & self, Tensor & infos_lu, Tensor & infos_getri) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_linalg_inv_out_helper_(ks & c10::after_InplaceOrView_keyset, self, infos_lu, infos_getri);
  }
  increment_version(self);
  return self;
}
Tensor & _linalg_solve_out_helper_(c10::DispatchKeySet ks, Tensor & self, Tensor & other, Tensor & infos) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_linalg_solve_out_helper_(ks & c10::after_InplaceOrView_keyset, self, other, infos);
  }
  increment_version(self);
  return self;
}
Tensor & _logcumsumexp_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_logcumsumexp_outf(ks & c10::after_InplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> _mode_out_values(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_mode_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
Tensor _values(c10::DispatchKeySet ks, const Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_values(ks & c10::after_InplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
Tensor & abs_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::abs_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & abs_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::abs_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & acos_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::acos_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & acos_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::acos_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & acosh_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::acosh_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & acosh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::acosh_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & adaptive_avg_pool2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::adaptive_avg_pool2d_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, out);
  }
  increment_version(out);
  return out;
}
Tensor & adaptive_max_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & indices, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::adaptive_max_pool2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, Tensor & out, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::adaptive_max_pool2d_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
}
Tensor & addcmul_(c10::DispatchKeySet ks, Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addcmul_(ks & c10::after_InplaceOrView_keyset, self, tensor1, tensor2, value);
  }
  increment_version(self);
  return self;
}
Tensor & addcmul_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addcmul_outf(ks & c10::after_InplaceOrView_keyset, self, tensor1, tensor2, value, out);
  }
  increment_version(out);
  return out;
}
Tensor & addmv_(c10::DispatchKeySet ks, Tensor & self, const Tensor & mat, const Tensor & vec, const Scalar & beta, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addmv_(ks & c10::after_InplaceOrView_keyset, self, mat, vec, beta, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & addmv_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat, const Tensor & vec, const Scalar & beta, const Scalar & alpha, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::addmv_outf(ks & c10::after_InplaceOrView_keyset, self, mat, vec, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
Tensor & any_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::any_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
Tensor & arange_out_start_out(c10::DispatchKeySet ks, const Scalar & start, const Scalar & end, const Scalar & step, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::arange_outf(ks & c10::after_InplaceOrView_keyset, start, end, step, out);
  }
  increment_version(out);
  return out;
}
Tensor & argmax_out_out(c10::DispatchKeySet ks, const Tensor & self, c10::optional<int64_t> dim, bool keepdim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::argmax_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
Tensor & argmin_out_out(c10::DispatchKeySet ks, const Tensor & self, c10::optional<int64_t> dim, bool keepdim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::argmin_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
Tensor & as_strided_(c10::DispatchKeySet ks, Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::as_strided_(ks & c10::after_InplaceOrView_keyset, self, size, stride, storage_offset);
  }
  increment_version(self);
  return self;
}
Tensor & atan2_(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::atan2_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & atan2_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::atan2_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & atan_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::atan_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & atan_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::atan_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & atanh_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::atanh_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & atanh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::atanh_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & avg_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::avg_pool3d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & avg_pool3d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::avg_pool3d_outf(ks & c10::after_InplaceOrView_keyset, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
  }
  increment_version(out);
  return out;
}
Tensor & bernoulli__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & p, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bernoulli_(ks & c10::after_InplaceOrView_keyset, self, p, generator);
  }
  increment_version(self);
  return self;
}
Tensor & bernoulli__float(c10::DispatchKeySet ks, Tensor & self, double p, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bernoulli_(ks & c10::after_InplaceOrView_keyset, self, p, generator);
  }
  increment_version(self);
  return self;
}
Tensor & bernoulli_out_out(c10::DispatchKeySet ks, const Tensor & self, c10::optional<Generator> generator, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bernoulli_outf(ks & c10::after_InplaceOrView_keyset, self, generator, out);
  }
  increment_version(out);
  return out;
}
Tensor & binary_cross_entropy_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::binary_cross_entropy_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, target, weight, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & binary_cross_entropy_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::binary_cross_entropy_outf(ks & c10::after_InplaceOrView_keyset, self, target, weight, reduction, out);
  }
  increment_version(out);
  return out;
}
Tensor & bmm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat2, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bmm_outf(ks & c10::after_InplaceOrView_keyset, self, mat2, out);
  }
  increment_version(out);
  return out;
}
Tensor & bucketize_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & boundaries, bool out_int32, bool right, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::bucketize_outf(ks & c10::after_InplaceOrView_keyset, self, boundaries, out_int32, right, out);
  }
  increment_version(out);
  return out;
}
Tensor & cat_out_out(c10::DispatchKeySet ks, TensorList tensors, int64_t dim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cat_outf(ks & c10::after_InplaceOrView_keyset, tensors, dim, out);
  }
  increment_version(out);
  return out;
}
Tensor & cholesky_out_out(c10::DispatchKeySet ks, const Tensor & self, bool upper, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cholesky_outf(ks & c10::after_InplaceOrView_keyset, self, upper, out);
  }
  increment_version(out);
  return out;
}
Tensor & cholesky_solve_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & input2, bool upper, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cholesky_solve_outf(ks & c10::after_InplaceOrView_keyset, self, input2, upper, out);
  }
  increment_version(out);
  return out;
}
Tensor & clamp_max_(c10::DispatchKeySet ks, Tensor & self, const Scalar & max) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::clamp_max_(ks & c10::after_InplaceOrView_keyset, self, max);
  }
  increment_version(self);
  return self;
}
Tensor & clamp_max_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & max, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::clamp_max_outf(ks & c10::after_InplaceOrView_keyset, self, max, out);
  }
  increment_version(out);
  return out;
}
Tensor & clamp_min_(c10::DispatchKeySet ks, Tensor & self, const Scalar & min) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::clamp_min_(ks & c10::after_InplaceOrView_keyset, self, min);
  }
  increment_version(self);
  return self;
}
Tensor & clamp_min_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & min, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::clamp_min_outf(ks & c10::after_InplaceOrView_keyset, self, min, out);
  }
  increment_version(out);
  return out;
}
Tensor & col2im_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::col2im_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, kernel_size, dilation, padding, stride, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & col2im_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::col2im_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, kernel_size, dilation, padding, stride, out);
  }
  increment_version(out);
  return out;
}
Tensor & complex_out_out(c10::DispatchKeySet ks, const Tensor & real, const Tensor & imag, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::complex_outf(ks & c10::after_InplaceOrView_keyset, real, imag, out);
  }
  increment_version(out);
  return out;
}
Tensor & conj_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::conj_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & copysign__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::copysign_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & copysign__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::copysign_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & copysign_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::copysign_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & cross_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, c10::optional<int64_t> dim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cross_outf(ks & c10::after_InplaceOrView_keyset, self, other, dim, out);
  }
  increment_version(out);
  return out;
}
Tensor & cumprod_(c10::DispatchKeySet ks, Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cumprod_(ks & c10::after_InplaceOrView_keyset, self, dim, dtype);
  }
  increment_version(self);
  return self;
}
Tensor & cumprod_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cumprod_outf(ks & c10::after_InplaceOrView_keyset, self, dim, dtype, out);
  }
  increment_version(out);
  return out;
}
Tensor & cumsum_(c10::DispatchKeySet ks, Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cumsum_(ks & c10::after_InplaceOrView_keyset, self, dim, dtype);
  }
  increment_version(self);
  return self;
}
Tensor & cumsum_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::cumsum_outf(ks & c10::after_InplaceOrView_keyset, self, dim, dtype, out);
  }
  increment_version(out);
  return out;
}
Tensor diagonal(c10::DispatchKeySet ks, const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::diagonal(ks & c10::after_InplaceOrView_keyset, self, offset, dim1, dim2);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::diagonal(input_base, offset, dim1, dim2);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & digamma_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::digamma_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & digamma_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::digamma_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & elu_(c10::DispatchKeySet ks, Tensor & self, const Scalar & alpha, const Scalar & scale, const Scalar & input_scale) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::elu_(ks & c10::after_InplaceOrView_keyset, self, alpha, scale, input_scale);
  }
  increment_version(self);
  return self;
}
Tensor & elu_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & alpha, const Scalar & scale, const Scalar & input_scale, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::elu_outf(ks & c10::after_InplaceOrView_keyset, self, alpha, scale, input_scale, out);
  }
  increment_version(out);
  return out;
}
Tensor & eq__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::eq_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & eq__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::eq_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & eq_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::eq_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & eq_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::eq_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & erfc_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::erfc_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & erfc_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::erfc_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & erfinv_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::erfinv_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & erfinv_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::erfinv_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & floor_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::floor_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & floor_divide__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::floor_divide_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & floor_divide_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::floor_divide_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & floor_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::floor_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & fmax_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fmax_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & fmin_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fmin_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & fmod__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fmod_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & fmod__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fmod_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & fmod_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fmod_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & fmod_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fmod_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & frac_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::frac_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & frac_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::frac_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & fractional_max_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fractional_max_pool3d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, kernel_size, output_size, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
std::tuple<Tensor &,Tensor &> fractional_max_pool3d_out_output(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples, Tensor & output, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::fractional_max_pool3d_outf(ks & c10::after_InplaceOrView_keyset, self, kernel_size, output_size, random_samples, output, indices);
  }
  increment_version(output);
  increment_version(indices);
  return std::forward_as_tuple(output, indices);
}
Tensor & gcd_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::gcd_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & ge__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ge_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & ge__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ge_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & ge_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ge_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & ge_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ge_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & ger_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & vec2, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::ger_outf(ks & c10::after_InplaceOrView_keyset, self, vec2, out);
  }
  increment_version(out);
  return out;
}
Tensor & glu_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, int64_t dim, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::glu_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, dim, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & glu_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::glu_outf(ks & c10::after_InplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
Tensor & hardtanh_(c10::DispatchKeySet ks, Tensor & self, const Scalar & min_val, const Scalar & max_val) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::hardtanh_(ks & c10::after_InplaceOrView_keyset, self, min_val, max_val);
  }
  increment_version(self);
  return self;
}
Tensor & hardtanh_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Scalar & min_val, const Scalar & max_val, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::hardtanh_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, min_val, max_val, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & hardtanh_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & min_val, const Scalar & max_val, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::hardtanh_outf(ks & c10::after_InplaceOrView_keyset, self, min_val, max_val, out);
  }
  increment_version(out);
  return out;
}
Tensor & heaviside_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & values, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::heaviside_outf(ks & c10::after_InplaceOrView_keyset, self, values, out);
  }
  increment_version(out);
  return out;
}
Tensor & huber_loss_backward_out_out(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double delta, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::huber_loss_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, target, reduction, delta, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & huber_loss_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction, double delta, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::huber_loss_outf(ks & c10::after_InplaceOrView_keyset, self, target, reduction, delta, out);
  }
  increment_version(out);
  return out;
}
Tensor & hypot_(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::hypot_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & hypot_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::hypot_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & igamma_(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::igamma_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & igamma_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::igamma_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & im2col_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::im2col_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, input_size, kernel_size, dilation, padding, stride, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & im2col_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::im2col_outf(ks & c10::after_InplaceOrView_keyset, self, kernel_size, dilation, padding, stride, out);
  }
  increment_version(out);
  return out;
}
Tensor & index_add_(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::index_add_(ks & c10::after_InplaceOrView_keyset, self, dim, index, source);
  }
  increment_version(self);
  return self;
}
Tensor & index_copy_(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::index_copy_(ks & c10::after_InplaceOrView_keyset, self, dim, index, source);
  }
  increment_version(self);
  return self;
}
Tensor & index_fill__int_Scalar(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Scalar & value) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::index_fill_(ks & c10::after_InplaceOrView_keyset, self, dim, index, value);
  }
  increment_version(self);
  return self;
}
Tensor & index_fill__int_Tensor(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::index_fill_(ks & c10::after_InplaceOrView_keyset, self, dim, index, value);
  }
  increment_version(self);
  return self;
}
Tensor & index_put_(c10::DispatchKeySet ks, Tensor & self, const c10::List<c10::optional<Tensor>> & indices, const Tensor & values, bool accumulate) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::index_put_(ks & c10::after_InplaceOrView_keyset, self, indices, values, accumulate);
  }
  increment_version(self);
  return self;
}
Tensor indices(c10::DispatchKeySet ks, const Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::indices(ks & c10::after_InplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
Tensor & inverse_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::inverse_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> kthvalue_out_values(c10::DispatchKeySet ks, const Tensor & self, int64_t k, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::kthvalue_outf(ks & c10::after_InplaceOrView_keyset, self, k, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
Tensor & lcm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lcm_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & linalg_cholesky_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::linalg_cholesky_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & linalg_householder_product_out_out(c10::DispatchKeySet ks, const Tensor & input, const Tensor & tau, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::linalg_householder_product_outf(ks & c10::after_InplaceOrView_keyset, input, tau, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> linalg_slogdet_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & sign, Tensor & logabsdet) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::linalg_slogdet_outf(ks & c10::after_InplaceOrView_keyset, self, sign, logabsdet);
  }
  increment_version(sign);
  increment_version(logabsdet);
  return std::forward_as_tuple(sign, logabsdet);
}
Tensor & linalg_vector_norm_out_out(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::linalg_vector_norm_outf(ks & c10::after_InplaceOrView_keyset, self, ord, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
Tensor & log2_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::log2_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & log2_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::log2_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & log_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::log_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & log_normal_(c10::DispatchKeySet ks, Tensor & self, double mean, double std, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::log_normal_(ks & c10::after_InplaceOrView_keyset, self, mean, std, generator);
  }
  increment_version(self);
  return self;
}
Tensor & log_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::log_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & logaddexp2_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logaddexp2_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & logaddexp_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logaddexp_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & logspace_out_out(c10::DispatchKeySet ks, const Scalar & start, const Scalar & end, c10::optional<int64_t> steps, double base, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logspace_outf(ks & c10::after_InplaceOrView_keyset, start, end, steps, base, out);
  }
  increment_version(out);
  return out;
}
Tensor & logsumexp_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::logsumexp_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
Tensor & lt__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lt_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & lt__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lt_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & lt_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lt_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & lt_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::lt_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & masked_scatter_(c10::DispatchKeySet ks, Tensor & self, const Tensor & mask, const Tensor & source) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::masked_scatter_(ks & c10::after_InplaceOrView_keyset, self, mask, source);
  }
  increment_version(self);
  return self;
}
Tensor & masked_select_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mask, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::masked_select_outf(ks & c10::after_InplaceOrView_keyset, self, mask, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> max_out_dim_max(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & max, Tensor & max_values) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::max_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, max, max_values);
  }
  increment_version(max);
  increment_version(max_values);
  return std::forward_as_tuple(max, max_values);
}
Tensor & max_pool2d_with_indices_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::max_pool2d_with_indices_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, Tensor & out, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::max_pool2d_with_indices_outf(ks & c10::after_InplaceOrView_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
}
Tensor & max_unpool2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::max_unpool2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, indices, output_size, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & max_unpool2d_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & indices, IntArrayRef output_size, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::max_unpool2d_outf(ks & c10::after_InplaceOrView_keyset, self, indices, output_size, out);
  }
  increment_version(out);
  return out;
}
Tensor & maximum_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::maximum_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> median_out_dim_values(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::median_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> min_out_dim_min(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & min, Tensor & min_indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::min_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, min, min_indices);
  }
  increment_version(min);
  increment_version(min_indices);
  return std::forward_as_tuple(min, min_indices);
}
Tensor & minimum_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::minimum_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & mm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat2, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::mm_outf(ks & c10::after_InplaceOrView_keyset, self, mat2, out);
  }
  increment_version(out);
  return out;
}
Tensor & mul__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::mul_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & mul__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::mul_(ks & c10::after_InplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
Tensor & mul_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::mul_outf(ks & c10::after_InplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
Tensor & multi_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Scalar & p, const Scalar & margin, const c10::optional<Tensor> & weight, int64_t reduction, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::multi_margin_loss_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, target, p, margin, weight, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & multi_margin_loss_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const Scalar & p, const Scalar & margin, const c10::optional<Tensor> & weight, int64_t reduction, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::multi_margin_loss_outf(ks & c10::after_InplaceOrView_keyset, self, target, p, margin, weight, reduction, out);
  }
  increment_version(out);
  return out;
}
Tensor & multilabel_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::multilabel_margin_loss_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, target, reduction, is_target, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
std::tuple<Tensor &,Tensor &> multilabel_margin_loss_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & output, Tensor & is_target) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::multilabel_margin_loss_forward_outf(ks & c10::after_InplaceOrView_keyset, self, target, reduction, output, is_target);
  }
  increment_version(output);
  increment_version(is_target);
  return std::forward_as_tuple(output, is_target);
}
Tensor & mvlgamma_(c10::DispatchKeySet ks, Tensor & self, int64_t p) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::mvlgamma_(ks & c10::after_InplaceOrView_keyset, self, p);
  }
  increment_version(self);
  return self;
}
Tensor & nan_to_num_(c10::DispatchKeySet ks, Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::nan_to_num_(ks & c10::after_InplaceOrView_keyset, self, nan, posinf, neginf);
  }
  increment_version(self);
  return self;
}
Tensor & nan_to_num_out_out(c10::DispatchKeySet ks, const Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::nan_to_num_outf(ks & c10::after_InplaceOrView_keyset, self, nan, posinf, neginf, out);
  }
  increment_version(out);
  return out;
}
Tensor & nansum_out_IntList_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::nansum_outf(ks & c10::after_InplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
Tensor & neg_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::neg_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & neg_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::neg_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & nll_loss2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::nll_loss2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
std::tuple<Tensor &,Tensor &> nll_loss2d_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::nll_loss2d_forward_outf(ks & c10::after_InplaceOrView_keyset, self, target, weight, reduction, ignore_index, output, total_weight);
  }
  increment_version(output);
  increment_version(total_weight);
  return std::forward_as_tuple(output, total_weight);
}
Tensor & nll_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::nll_loss_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
std::tuple<Tensor &,Tensor &> nll_loss_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::nll_loss_forward_outf(ks & c10::after_InplaceOrView_keyset, self, target, weight, reduction, ignore_index, output, total_weight);
  }
  increment_version(output);
  increment_version(total_weight);
  return std::forward_as_tuple(output, total_weight);
}
Tensor & norm_out_dtype_out(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, ScalarType dtype, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::norm_outf(ks & c10::after_InplaceOrView_keyset, self, p, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
Tensor & norm_out_out(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::norm_outf(ks & c10::after_InplaceOrView_keyset, self, p, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
Tensor & polar_out_out(c10::DispatchKeySet ks, const Tensor & abs, const Tensor & angle, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::polar_outf(ks & c10::after_InplaceOrView_keyset, abs, angle, out);
  }
  increment_version(out);
  return out;
}
Tensor & pow__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & exponent) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::pow_(ks & c10::after_InplaceOrView_keyset, self, exponent);
  }
  increment_version(self);
  return self;
}
Tensor & pow__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & exponent) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::pow_(ks & c10::after_InplaceOrView_keyset, self, exponent);
  }
  increment_version(self);
  return self;
}
Tensor & pow_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & exponent, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::pow_outf(ks & c10::after_InplaceOrView_keyset, self, exponent, out);
  }
  increment_version(out);
  return out;
}
Tensor & pow_out_Scalar_out(c10::DispatchKeySet ks, const Scalar & self, const Tensor & exponent, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::pow_outf(ks & c10::after_InplaceOrView_keyset, self, exponent, out);
  }
  increment_version(out);
  return out;
}
Tensor & pow_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & exponent, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::pow_outf(ks & c10::after_InplaceOrView_keyset, self, exponent, out);
  }
  increment_version(out);
  return out;
}
Tensor & reciprocal_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::reciprocal_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & reciprocal_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::reciprocal_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & reflection_pad1d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::reflection_pad1d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & reflection_pad1d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::reflection_pad1d_outf(ks & c10::after_InplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
Tensor & relu_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::relu_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & replication_pad2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::replication_pad2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & replication_pad2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::replication_pad2d_outf(ks & c10::after_InplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
Tensor & round_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::round_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & round_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::round_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & rrelu_with_noise_(c10::DispatchKeySet ks, Tensor & self, const Tensor & noise, const Scalar & lower, const Scalar & upper, bool training, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::rrelu_with_noise_(ks & c10::after_InplaceOrView_keyset, self, noise, lower, upper, training, generator);
  }
  increment_version(self);
  return self;
}
Tensor & rrelu_with_noise_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & noise, const Scalar & lower, const Scalar & upper, bool training, c10::optional<Generator> generator, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::rrelu_with_noise_outf(ks & c10::after_InplaceOrView_keyset, self, noise, lower, upper, training, generator, out);
  }
  increment_version(out);
  return out;
}
Tensor & rsqrt_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::rsqrt_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & rsqrt_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::rsqrt_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & scatter__src(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::scatter_(ks & c10::after_InplaceOrView_keyset, self, dim, index, src);
  }
  increment_version(self);
  return self;
}
Tensor & scatter__value(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Scalar & value) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::scatter_(ks & c10::after_InplaceOrView_keyset, self, dim, index, value);
  }
  increment_version(self);
  return self;
}
Tensor & scatter__reduce(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, std::string reduce) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::scatter_(ks & c10::after_InplaceOrView_keyset, self, dim, index, src, reduce);
  }
  increment_version(self);
  return self;
}
Tensor & scatter__value_reduce(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, std::string reduce) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::scatter_(ks & c10::after_InplaceOrView_keyset, self, dim, index, value, reduce);
  }
  increment_version(self);
  return self;
}
Tensor & scatter_add_(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::scatter_add_(ks & c10::after_InplaceOrView_keyset, self, dim, index, src);
  }
  increment_version(self);
  return self;
}
Tensor & set__source_Storage(c10::DispatchKeySet ks, Tensor & self, Storage source) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::set_(ks & c10::after_InplaceOrView_keyset, self, source);
  }
  increment_version(self);
  return self;
}
Tensor & set__source_Storage_storage_offset(c10::DispatchKeySet ks, Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::set_(ks & c10::after_InplaceOrView_keyset, self, source, storage_offset, size, stride);
  }
  increment_version(self);
  return self;
}
Tensor & set__source_Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & source) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::set_(ks & c10::after_InplaceOrView_keyset, self, source);
  }
  increment_version(self);
  return self;
}
Tensor & set_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::set_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & sgn_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sgn_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & sgn_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sgn_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & sigmoid_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sigmoid_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & sigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & output, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sigmoid_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, output, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & sigmoid_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sigmoid_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & signbit_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::signbit_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & sin_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sin_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & sin_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sin_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & sinh_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sinh_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & sinh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sinh_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv_transpose2d_backward_out_grad_output(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::slow_conv_transpose2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, grad_input, grad_weight, grad_bias);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
Tensor & slow_conv_transpose2d_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::slow_conv_transpose2d_outf(ks & c10::after_InplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
  }
  increment_version(out);
  return out;
}
Tensor & smooth_l1_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double beta, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::smooth_l1_loss_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, target, reduction, beta, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & smooth_l1_loss_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction, double beta, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::smooth_l1_loss_outf(ks & c10::after_InplaceOrView_keyset, self, target, reduction, beta, out);
  }
  increment_version(out);
  return out;
}
Tensor & softplus_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Scalar & beta, const Scalar & threshold, const Tensor & output, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::softplus_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, self, beta, threshold, output, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & softplus_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & beta, const Scalar & threshold, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::softplus_outf(ks & c10::after_InplaceOrView_keyset, self, beta, threshold, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> sort_out_values(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sort_outf(ks & c10::after_InplaceOrView_keyset, self, dim, descending, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> sort_out_values_stable(c10::DispatchKeySet ks, const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sort_outf(ks & c10::after_InplaceOrView_keyset, self, stable, dim, descending, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
Tensor & sqrt_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sqrt_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & sqrt_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sqrt_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & squeeze_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::squeeze_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & squeeze__dim(c10::DispatchKeySet ks, Tensor & self, int64_t dim) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::squeeze_(ks & c10::after_InplaceOrView_keyset, self, dim);
  }
  increment_version(self);
  return self;
}
Tensor & stack_out_out(c10::DispatchKeySet ks, TensorList tensors, int64_t dim, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::stack_outf(ks & c10::after_InplaceOrView_keyset, tensors, dim, out);
  }
  increment_version(out);
  return out;
}
Tensor & sub__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sub_(ks & c10::after_InplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & sub__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other, const Scalar & alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sub_(ks & c10::after_InplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
Tensor & sub_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::sub_outf(ks & c10::after_InplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> symeig_out_e(c10::DispatchKeySet ks, const Tensor & self, bool eigenvectors, bool upper, Tensor & e, Tensor & V) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::symeig_outf(ks & c10::after_InplaceOrView_keyset, self, eigenvectors, upper, e, V);
  }
  increment_version(e);
  increment_version(V);
  return std::forward_as_tuple(e, V);
}
Tensor & t_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::t_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & tensordot_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::tensordot_outf(ks & c10::after_InplaceOrView_keyset, self, other, dims_self, dims_other, out);
  }
  increment_version(out);
  return out;
}
std::tuple<Tensor &,Tensor &> topk_out_values(c10::DispatchKeySet ks, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::topk_outf(ks & c10::after_InplaceOrView_keyset, self, k, dim, largest, sorted, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
Tensor transpose_int(c10::DispatchKeySet ks, const Tensor & self, int64_t dim0, int64_t dim1) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::transpose(ks & c10::after_InplaceOrView_keyset, self, dim0, dim1);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::transpose(input_base, dim0, dim1);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & triu_(c10::DispatchKeySet ks, Tensor & self, int64_t diagonal) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::triu_(ks & c10::after_InplaceOrView_keyset, self, diagonal);
  }
  increment_version(self);
  return self;
}
Tensor & triu_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t diagonal, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::triu_outf(ks & c10::after_InplaceOrView_keyset, self, diagonal, out);
  }
  increment_version(out);
  return out;
}
Tensor & trunc_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::trunc_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
Tensor & trunc_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::trunc_outf(ks & c10::after_InplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
Tensor & uniform_(c10::DispatchKeySet ks, Tensor & self, double from, double to, c10::optional<Generator> generator) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::uniform_(ks & c10::after_InplaceOrView_keyset, self, from, to, generator);
  }
  increment_version(self);
  return self;
}
Tensor unsqueeze(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::unsqueeze(ks & c10::after_InplaceOrView_keyset, self, dim);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::unsqueeze(input_base, dim);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & upsample_bilinear2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_bilinear2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & upsample_bilinear2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_bilinear2d_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, align_corners, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
Tensor & upsample_linear1d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_linear1d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & upsample_linear1d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_linear1d_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, align_corners, scales, out);
  }
  increment_version(out);
  return out;
}
Tensor & upsample_nearest2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & grad_input) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_nearest2d_backward_outf(ks & c10::after_InplaceOrView_keyset, grad_output, output_size, input_size, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
Tensor & upsample_nearest2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & out) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::upsample_nearest2d_outf(ks & c10::after_InplaceOrView_keyset, self, output_size, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
Tensor view(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef size) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::view(ks & c10::after_InplaceOrView_keyset, self, size);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    auto size_vec = size.vec();
    func = [=](const at::Tensor& input_base) {
      return input_base.view(size_vec);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor view_dtype(c10::DispatchKeySet ks, const Tensor & self, ScalarType dtype) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::view(ks & c10::after_InplaceOrView_keyset, self, dtype);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
Tensor view_as_complex(c10::DispatchKeySet ks, const Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::view_as_complex(ks & c10::after_InplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (true || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::view_as_complex(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor view_as_real(c10::DispatchKeySet ks, const Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::view_as_real(ks & c10::after_InplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (true || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::view_as_real(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE);
  return result;
}
Tensor & zero_(c10::DispatchKeySet ks, Tensor & self) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::zero_(ks & c10::after_InplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
}  // namespace
}  // namespace InplaceOrView

namespace {

TORCH_LIBRARY_IMPL(aten, InplaceOrView, m) {
  m.impl("__ilshift__.Scalar",
         TORCH_FN(InplaceOrView::__ilshift___Scalar)
  );
  m.impl("__ilshift__.Tensor",
         TORCH_FN(InplaceOrView::__ilshift___Tensor)
  );
  m.impl("__irshift__.Scalar",
         TORCH_FN(InplaceOrView::__irshift___Scalar)
  );
  m.impl("__irshift__.Tensor",
         TORCH_FN(InplaceOrView::__irshift___Tensor)
  );
  m.impl("_addmv_impl_",
         TORCH_FN(InplaceOrView::_addmv_impl_)
  );
  m.impl("_coalesced_",
         TORCH_FN(InplaceOrView::_coalesced_)
  );
  m.impl("_compute_linear_combination.out",
         TORCH_FN(InplaceOrView::_compute_linear_combination_out_out)
  );
  m.impl("_fft_c2c.out",
         TORCH_FN(InplaceOrView::_fft_c2c_out_out)
  );
  m.impl("_index_put_impl_",
         TORCH_FN(InplaceOrView::_index_put_impl_)
  );
  m.impl("_linalg_inv_out_helper_",
         TORCH_FN(InplaceOrView::_linalg_inv_out_helper_)
  );
  m.impl("_linalg_solve_out_helper_",
         TORCH_FN(InplaceOrView::_linalg_solve_out_helper_)
  );
  m.impl("_logcumsumexp.out",
         TORCH_FN(InplaceOrView::_logcumsumexp_out_out)
  );
  m.impl("_mode.values",
         TORCH_FN(InplaceOrView::_mode_out_values)
  );
  m.impl("_values",
         TORCH_FN(InplaceOrView::_values)
  );
  m.impl("abs_",
         TORCH_FN(InplaceOrView::abs_)
  );
  m.impl("abs.out",
         TORCH_FN(InplaceOrView::abs_out_out)
  );
  m.impl("acos_",
         TORCH_FN(InplaceOrView::acos_)
  );
  m.impl("acos.out",
         TORCH_FN(InplaceOrView::acos_out_out)
  );
  m.impl("acosh_",
         TORCH_FN(InplaceOrView::acosh_)
  );
  m.impl("acosh.out",
         TORCH_FN(InplaceOrView::acosh_out_out)
  );
  m.impl("adaptive_avg_pool2d.out",
         TORCH_FN(InplaceOrView::adaptive_avg_pool2d_out_out)
  );
  m.impl("adaptive_max_pool2d_backward.grad_input",
         TORCH_FN(InplaceOrView::adaptive_max_pool2d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool2d.out",
         TORCH_FN(InplaceOrView::adaptive_max_pool2d_out_out)
  );
  m.impl("addcmul_",
         TORCH_FN(InplaceOrView::addcmul_)
  );
  m.impl("addcmul.out",
         TORCH_FN(InplaceOrView::addcmul_out_out)
  );
  m.impl("addmv_",
         TORCH_FN(InplaceOrView::addmv_)
  );
  m.impl("addmv.out",
         TORCH_FN(InplaceOrView::addmv_out_out)
  );
  m.impl("any.out",
         TORCH_FN(InplaceOrView::any_out_out)
  );
  m.impl("arange.start_out",
         TORCH_FN(InplaceOrView::arange_out_start_out)
  );
  m.impl("argmax.out",
         TORCH_FN(InplaceOrView::argmax_out_out)
  );
  m.impl("argmin.out",
         TORCH_FN(InplaceOrView::argmin_out_out)
  );
  m.impl("as_strided_",
         TORCH_FN(InplaceOrView::as_strided_)
  );
  m.impl("atan2_",
         TORCH_FN(InplaceOrView::atan2_)
  );
  m.impl("atan2.out",
         TORCH_FN(InplaceOrView::atan2_out_out)
  );
  m.impl("atan_",
         TORCH_FN(InplaceOrView::atan_)
  );
  m.impl("atan.out",
         TORCH_FN(InplaceOrView::atan_out_out)
  );
  m.impl("atanh_",
         TORCH_FN(InplaceOrView::atanh_)
  );
  m.impl("atanh.out",
         TORCH_FN(InplaceOrView::atanh_out_out)
  );
  m.impl("avg_pool3d_backward.grad_input",
         TORCH_FN(InplaceOrView::avg_pool3d_backward_out_grad_input)
  );
  m.impl("avg_pool3d.out",
         TORCH_FN(InplaceOrView::avg_pool3d_out_out)
  );
  m.impl("bernoulli_.Tensor",
         TORCH_FN(InplaceOrView::bernoulli__Tensor)
  );
  m.impl("bernoulli_.float",
         TORCH_FN(InplaceOrView::bernoulli__float)
  );
  m.impl("bernoulli.out",
         TORCH_FN(InplaceOrView::bernoulli_out_out)
  );
  m.impl("binary_cross_entropy_backward.grad_input",
         TORCH_FN(InplaceOrView::binary_cross_entropy_backward_out_grad_input)
  );
  m.impl("binary_cross_entropy.out",
         TORCH_FN(InplaceOrView::binary_cross_entropy_out_out)
  );
  m.impl("bmm.out",
         TORCH_FN(InplaceOrView::bmm_out_out)
  );
  m.impl("bucketize.Tensor_out",
         TORCH_FN(InplaceOrView::bucketize_out_Tensor_out)
  );
  m.impl("cat.out",
         TORCH_FN(InplaceOrView::cat_out_out)
  );
  m.impl("cholesky.out",
         TORCH_FN(InplaceOrView::cholesky_out_out)
  );
  m.impl("cholesky_solve.out",
         TORCH_FN(InplaceOrView::cholesky_solve_out_out)
  );
  m.impl("clamp_max_",
         TORCH_FN(InplaceOrView::clamp_max_)
  );
  m.impl("clamp_max.out",
         TORCH_FN(InplaceOrView::clamp_max_out_out)
  );
  m.impl("clamp_min_",
         TORCH_FN(InplaceOrView::clamp_min_)
  );
  m.impl("clamp_min.out",
         TORCH_FN(InplaceOrView::clamp_min_out_out)
  );
  m.impl("col2im_backward.grad_input",
         TORCH_FN(InplaceOrView::col2im_backward_out_grad_input)
  );
  m.impl("col2im.out",
         TORCH_FN(InplaceOrView::col2im_out_out)
  );
  m.impl("complex.out",
         TORCH_FN(InplaceOrView::complex_out_out)
  );
  m.impl("conj.out",
         TORCH_FN(InplaceOrView::conj_out_out)
  );
  m.impl("copysign_.Tensor",
         TORCH_FN(InplaceOrView::copysign__Tensor)
  );
  m.impl("copysign_.Scalar",
         TORCH_FN(InplaceOrView::copysign__Scalar)
  );
  m.impl("copysign.out",
         TORCH_FN(InplaceOrView::copysign_out_out)
  );
  m.impl("cross.out",
         TORCH_FN(InplaceOrView::cross_out_out)
  );
  m.impl("cumprod_",
         TORCH_FN(InplaceOrView::cumprod_)
  );
  m.impl("cumprod.out",
         TORCH_FN(InplaceOrView::cumprod_out_out)
  );
  m.impl("cumsum_",
         TORCH_FN(InplaceOrView::cumsum_)
  );
  m.impl("cumsum.out",
         TORCH_FN(InplaceOrView::cumsum_out_out)
  );
  m.impl("diagonal",
         TORCH_FN(InplaceOrView::diagonal)
  );
  m.impl("digamma_",
         TORCH_FN(InplaceOrView::digamma_)
  );
  m.impl("digamma.out",
         TORCH_FN(InplaceOrView::digamma_out_out)
  );
  m.impl("elu_",
         TORCH_FN(InplaceOrView::elu_)
  );
  m.impl("elu.out",
         TORCH_FN(InplaceOrView::elu_out_out)
  );
  m.impl("eq_.Scalar",
         TORCH_FN(InplaceOrView::eq__Scalar)
  );
  m.impl("eq_.Tensor",
         TORCH_FN(InplaceOrView::eq__Tensor)
  );
  m.impl("eq.Scalar_out",
         TORCH_FN(InplaceOrView::eq_out_Scalar_out)
  );
  m.impl("eq.Tensor_out",
         TORCH_FN(InplaceOrView::eq_out_Tensor_out)
  );
  m.impl("erfc_",
         TORCH_FN(InplaceOrView::erfc_)
  );
  m.impl("erfc.out",
         TORCH_FN(InplaceOrView::erfc_out_out)
  );
  m.impl("erfinv_",
         TORCH_FN(InplaceOrView::erfinv_)
  );
  m.impl("erfinv.out",
         TORCH_FN(InplaceOrView::erfinv_out_out)
  );
  m.impl("floor_",
         TORCH_FN(InplaceOrView::floor_)
  );
  m.impl("floor_divide_.Tensor",
         TORCH_FN(InplaceOrView::floor_divide__Tensor)
  );
  m.impl("floor_divide.out",
         TORCH_FN(InplaceOrView::floor_divide_out_out)
  );
  m.impl("floor.out",
         TORCH_FN(InplaceOrView::floor_out_out)
  );
  m.impl("fmax.out",
         TORCH_FN(InplaceOrView::fmax_out_out)
  );
  m.impl("fmin.out",
         TORCH_FN(InplaceOrView::fmin_out_out)
  );
  m.impl("fmod_.Scalar",
         TORCH_FN(InplaceOrView::fmod__Scalar)
  );
  m.impl("fmod_.Tensor",
         TORCH_FN(InplaceOrView::fmod__Tensor)
  );
  m.impl("fmod.Scalar_out",
         TORCH_FN(InplaceOrView::fmod_out_Scalar_out)
  );
  m.impl("fmod.Tensor_out",
         TORCH_FN(InplaceOrView::fmod_out_Tensor_out)
  );
  m.impl("frac_",
         TORCH_FN(InplaceOrView::frac_)
  );
  m.impl("frac.out",
         TORCH_FN(InplaceOrView::frac_out_out)
  );
  m.impl("fractional_max_pool3d_backward.grad_input",
         TORCH_FN(InplaceOrView::fractional_max_pool3d_backward_out_grad_input)
  );
  m.impl("fractional_max_pool3d.output",
         TORCH_FN(InplaceOrView::fractional_max_pool3d_out_output)
  );
  m.impl("gcd.out",
         TORCH_FN(InplaceOrView::gcd_out_out)
  );
  m.impl("ge_.Scalar",
         TORCH_FN(InplaceOrView::ge__Scalar)
  );
  m.impl("ge_.Tensor",
         TORCH_FN(InplaceOrView::ge__Tensor)
  );
  m.impl("ge.Scalar_out",
         TORCH_FN(InplaceOrView::ge_out_Scalar_out)
  );
  m.impl("ge.Tensor_out",
         TORCH_FN(InplaceOrView::ge_out_Tensor_out)
  );
  m.impl("ger.out",
         TORCH_FN(InplaceOrView::ger_out_out)
  );
  m.impl("glu_backward.grad_input",
         TORCH_FN(InplaceOrView::glu_backward_out_grad_input)
  );
  m.impl("glu.out",
         TORCH_FN(InplaceOrView::glu_out_out)
  );
  m.impl("hardtanh_",
         TORCH_FN(InplaceOrView::hardtanh_)
  );
  m.impl("hardtanh_backward.grad_input",
         TORCH_FN(InplaceOrView::hardtanh_backward_out_grad_input)
  );
  m.impl("hardtanh.out",
         TORCH_FN(InplaceOrView::hardtanh_out_out)
  );
  m.impl("heaviside.out",
         TORCH_FN(InplaceOrView::heaviside_out_out)
  );
  m.impl("huber_loss_backward.out",
         TORCH_FN(InplaceOrView::huber_loss_backward_out_out)
  );
  m.impl("huber_loss.out",
         TORCH_FN(InplaceOrView::huber_loss_out_out)
  );
  m.impl("hypot_",
         TORCH_FN(InplaceOrView::hypot_)
  );
  m.impl("hypot.out",
         TORCH_FN(InplaceOrView::hypot_out_out)
  );
  m.impl("igamma_",
         TORCH_FN(InplaceOrView::igamma_)
  );
  m.impl("igamma.out",
         TORCH_FN(InplaceOrView::igamma_out_out)
  );
  m.impl("im2col_backward.grad_input",
         TORCH_FN(InplaceOrView::im2col_backward_out_grad_input)
  );
  m.impl("im2col.out",
         TORCH_FN(InplaceOrView::im2col_out_out)
  );
  m.impl("index_add_",
         TORCH_FN(InplaceOrView::index_add_)
  );
  m.impl("index_copy_",
         TORCH_FN(InplaceOrView::index_copy_)
  );
  m.impl("index_fill_.int_Scalar",
         TORCH_FN(InplaceOrView::index_fill__int_Scalar)
  );
  m.impl("index_fill_.int_Tensor",
         TORCH_FN(InplaceOrView::index_fill__int_Tensor)
  );
  m.impl("index_put_",
         TORCH_FN(InplaceOrView::index_put_)
  );
  m.impl("indices",
         TORCH_FN(InplaceOrView::indices)
  );
  m.impl("inverse.out",
         TORCH_FN(InplaceOrView::inverse_out_out)
  );
  m.impl("kthvalue.values",
         TORCH_FN(InplaceOrView::kthvalue_out_values)
  );
  m.impl("lcm.out",
         TORCH_FN(InplaceOrView::lcm_out_out)
  );
  m.impl("linalg_cholesky.out",
         TORCH_FN(InplaceOrView::linalg_cholesky_out_out)
  );
  m.impl("linalg_householder_product.out",
         TORCH_FN(InplaceOrView::linalg_householder_product_out_out)
  );
  m.impl("linalg_slogdet.out",
         TORCH_FN(InplaceOrView::linalg_slogdet_out_out)
  );
  m.impl("linalg_vector_norm.out",
         TORCH_FN(InplaceOrView::linalg_vector_norm_out_out)
  );
  m.impl("log2_",
         TORCH_FN(InplaceOrView::log2_)
  );
  m.impl("log2.out",
         TORCH_FN(InplaceOrView::log2_out_out)
  );
  m.impl("log_",
         TORCH_FN(InplaceOrView::log_)
  );
  m.impl("log_normal_",
         TORCH_FN(InplaceOrView::log_normal_)
  );
  m.impl("log.out",
         TORCH_FN(InplaceOrView::log_out_out)
  );
  m.impl("logaddexp2.out",
         TORCH_FN(InplaceOrView::logaddexp2_out_out)
  );
  m.impl("logaddexp.out",
         TORCH_FN(InplaceOrView::logaddexp_out_out)
  );
  m.impl("logspace.out",
         TORCH_FN(InplaceOrView::logspace_out_out)
  );
  m.impl("logsumexp.out",
         TORCH_FN(InplaceOrView::logsumexp_out_out)
  );
  m.impl("lt_.Scalar",
         TORCH_FN(InplaceOrView::lt__Scalar)
  );
  m.impl("lt_.Tensor",
         TORCH_FN(InplaceOrView::lt__Tensor)
  );
  m.impl("lt.Scalar_out",
         TORCH_FN(InplaceOrView::lt_out_Scalar_out)
  );
  m.impl("lt.Tensor_out",
         TORCH_FN(InplaceOrView::lt_out_Tensor_out)
  );
  m.impl("masked_scatter_",
         TORCH_FN(InplaceOrView::masked_scatter_)
  );
  m.impl("masked_select.out",
         TORCH_FN(InplaceOrView::masked_select_out_out)
  );
  m.impl("max.dim_max",
         TORCH_FN(InplaceOrView::max_out_dim_max)
  );
  m.impl("max_pool2d_with_indices_backward.grad_input",
         TORCH_FN(InplaceOrView::max_pool2d_with_indices_backward_out_grad_input)
  );
  m.impl("max_pool2d_with_indices.out",
         TORCH_FN(InplaceOrView::max_pool2d_with_indices_out_out)
  );
  m.impl("max_unpool2d_backward.grad_input",
         TORCH_FN(InplaceOrView::max_unpool2d_backward_out_grad_input)
  );
  m.impl("max_unpool2d.out",
         TORCH_FN(InplaceOrView::max_unpool2d_out_out)
  );
  m.impl("maximum.out",
         TORCH_FN(InplaceOrView::maximum_out_out)
  );
  m.impl("median.dim_values",
         TORCH_FN(InplaceOrView::median_out_dim_values)
  );
  m.impl("min.dim_min",
         TORCH_FN(InplaceOrView::min_out_dim_min)
  );
  m.impl("minimum.out",
         TORCH_FN(InplaceOrView::minimum_out_out)
  );
  m.impl("mm.out",
         TORCH_FN(InplaceOrView::mm_out_out)
  );
  m.impl("mul_.Tensor",
         TORCH_FN(InplaceOrView::mul__Tensor)
  );
  m.impl("mul_.Scalar",
         TORCH_FN(InplaceOrView::mul__Scalar)
  );
  m.impl("mul.out",
         TORCH_FN(InplaceOrView::mul_out_out)
  );
  m.impl("multi_margin_loss_backward.grad_input",
         TORCH_FN(InplaceOrView::multi_margin_loss_backward_out_grad_input)
  );
  m.impl("multi_margin_loss.out",
         TORCH_FN(InplaceOrView::multi_margin_loss_out_out)
  );
  m.impl("multilabel_margin_loss_backward.grad_input",
         TORCH_FN(InplaceOrView::multilabel_margin_loss_backward_out_grad_input)
  );
  m.impl("multilabel_margin_loss_forward.output",
         TORCH_FN(InplaceOrView::multilabel_margin_loss_forward_out_output)
  );
  m.impl("mvlgamma_",
         TORCH_FN(InplaceOrView::mvlgamma_)
  );
  m.impl("nan_to_num_",
         TORCH_FN(InplaceOrView::nan_to_num_)
  );
  m.impl("nan_to_num.out",
         TORCH_FN(InplaceOrView::nan_to_num_out_out)
  );
  m.impl("nansum.IntList_out",
         TORCH_FN(InplaceOrView::nansum_out_IntList_out)
  );
  m.impl("neg_",
         TORCH_FN(InplaceOrView::neg_)
  );
  m.impl("neg.out",
         TORCH_FN(InplaceOrView::neg_out_out)
  );
  m.impl("nll_loss2d_backward.grad_input",
         TORCH_FN(InplaceOrView::nll_loss2d_backward_out_grad_input)
  );
  m.impl("nll_loss2d_forward.output",
         TORCH_FN(InplaceOrView::nll_loss2d_forward_out_output)
  );
  m.impl("nll_loss_backward.grad_input",
         TORCH_FN(InplaceOrView::nll_loss_backward_out_grad_input)
  );
  m.impl("nll_loss_forward.output",
         TORCH_FN(InplaceOrView::nll_loss_forward_out_output)
  );
  m.impl("norm.dtype_out",
         TORCH_FN(InplaceOrView::norm_out_dtype_out)
  );
  m.impl("norm.out",
         TORCH_FN(InplaceOrView::norm_out_out)
  );
  m.impl("polar.out",
         TORCH_FN(InplaceOrView::polar_out_out)
  );
  m.impl("pow_.Scalar",
         TORCH_FN(InplaceOrView::pow__Scalar)
  );
  m.impl("pow_.Tensor",
         TORCH_FN(InplaceOrView::pow__Tensor)
  );
  m.impl("pow.Tensor_Tensor_out",
         TORCH_FN(InplaceOrView::pow_out_Tensor_Tensor_out)
  );
  m.impl("pow.Scalar_out",
         TORCH_FN(InplaceOrView::pow_out_Scalar_out)
  );
  m.impl("pow.Tensor_Scalar_out",
         TORCH_FN(InplaceOrView::pow_out_Tensor_Scalar_out)
  );
  m.impl("reciprocal_",
         TORCH_FN(InplaceOrView::reciprocal_)
  );
  m.impl("reciprocal.out",
         TORCH_FN(InplaceOrView::reciprocal_out_out)
  );
  m.impl("reflection_pad1d_backward.grad_input",
         TORCH_FN(InplaceOrView::reflection_pad1d_backward_out_grad_input)
  );
  m.impl("reflection_pad1d.out",
         TORCH_FN(InplaceOrView::reflection_pad1d_out_out)
  );
  m.impl("relu_",
         TORCH_FN(InplaceOrView::relu_)
  );
  m.impl("replication_pad2d_backward.grad_input",
         TORCH_FN(InplaceOrView::replication_pad2d_backward_out_grad_input)
  );
  m.impl("replication_pad2d.out",
         TORCH_FN(InplaceOrView::replication_pad2d_out_out)
  );
  m.impl("round_",
         TORCH_FN(InplaceOrView::round_)
  );
  m.impl("round.out",
         TORCH_FN(InplaceOrView::round_out_out)
  );
  m.impl("rrelu_with_noise_",
         TORCH_FN(InplaceOrView::rrelu_with_noise_)
  );
  m.impl("rrelu_with_noise.out",
         TORCH_FN(InplaceOrView::rrelu_with_noise_out_out)
  );
  m.impl("rsqrt_",
         TORCH_FN(InplaceOrView::rsqrt_)
  );
  m.impl("rsqrt.out",
         TORCH_FN(InplaceOrView::rsqrt_out_out)
  );
  m.impl("scatter_.src",
         TORCH_FN(InplaceOrView::scatter__src)
  );
  m.impl("scatter_.value",
         TORCH_FN(InplaceOrView::scatter__value)
  );
  m.impl("scatter_.reduce",
         TORCH_FN(InplaceOrView::scatter__reduce)
  );
  m.impl("scatter_.value_reduce",
         TORCH_FN(InplaceOrView::scatter__value_reduce)
  );
  m.impl("scatter_add_",
         TORCH_FN(InplaceOrView::scatter_add_)
  );
  m.impl("set_.source_Storage",
         TORCH_FN(InplaceOrView::set__source_Storage)
  );
  m.impl("set_.source_Storage_storage_offset",
         TORCH_FN(InplaceOrView::set__source_Storage_storage_offset)
  );
  m.impl("set_.source_Tensor",
         TORCH_FN(InplaceOrView::set__source_Tensor)
  );
  m.impl("set_",
         TORCH_FN(InplaceOrView::set_)
  );
  m.impl("sgn_",
         TORCH_FN(InplaceOrView::sgn_)
  );
  m.impl("sgn.out",
         TORCH_FN(InplaceOrView::sgn_out_out)
  );
  m.impl("sigmoid_",
         TORCH_FN(InplaceOrView::sigmoid_)
  );
  m.impl("sigmoid_backward.grad_input",
         TORCH_FN(InplaceOrView::sigmoid_backward_out_grad_input)
  );
  m.impl("sigmoid.out",
         TORCH_FN(InplaceOrView::sigmoid_out_out)
  );
  m.impl("signbit.out",
         TORCH_FN(InplaceOrView::signbit_out_out)
  );
  m.impl("sin_",
         TORCH_FN(InplaceOrView::sin_)
  );
  m.impl("sin.out",
         TORCH_FN(InplaceOrView::sin_out_out)
  );
  m.impl("sinh_",
         TORCH_FN(InplaceOrView::sinh_)
  );
  m.impl("sinh.out",
         TORCH_FN(InplaceOrView::sinh_out_out)
  );
  m.impl("slow_conv_transpose2d_backward.grad_output",
         TORCH_FN(InplaceOrView::slow_conv_transpose2d_backward_out_grad_output)
  );
  m.impl("slow_conv_transpose2d.out",
         TORCH_FN(InplaceOrView::slow_conv_transpose2d_out_out)
  );
  m.impl("smooth_l1_loss_backward.grad_input",
         TORCH_FN(InplaceOrView::smooth_l1_loss_backward_out_grad_input)
  );
  m.impl("smooth_l1_loss.out",
         TORCH_FN(InplaceOrView::smooth_l1_loss_out_out)
  );
  m.impl("softplus_backward.grad_input",
         TORCH_FN(InplaceOrView::softplus_backward_out_grad_input)
  );
  m.impl("softplus.out",
         TORCH_FN(InplaceOrView::softplus_out_out)
  );
  m.impl("sort.values",
         TORCH_FN(InplaceOrView::sort_out_values)
  );
  m.impl("sort.values_stable",
         TORCH_FN(InplaceOrView::sort_out_values_stable)
  );
  m.impl("sqrt_",
         TORCH_FN(InplaceOrView::sqrt_)
  );
  m.impl("sqrt.out",
         TORCH_FN(InplaceOrView::sqrt_out_out)
  );
  m.impl("squeeze_",
         TORCH_FN(InplaceOrView::squeeze_)
  );
  m.impl("squeeze_.dim",
         TORCH_FN(InplaceOrView::squeeze__dim)
  );
  m.impl("stack.out",
         TORCH_FN(InplaceOrView::stack_out_out)
  );
  m.impl("sub_.Tensor",
         TORCH_FN(InplaceOrView::sub__Tensor)
  );
  m.impl("sub_.Scalar",
         TORCH_FN(InplaceOrView::sub__Scalar)
  );
  m.impl("sub.out",
         TORCH_FN(InplaceOrView::sub_out_out)
  );
  m.impl("symeig.e",
         TORCH_FN(InplaceOrView::symeig_out_e)
  );
  m.impl("t_",
         TORCH_FN(InplaceOrView::t_)
  );
  m.impl("tensordot.out",
         TORCH_FN(InplaceOrView::tensordot_out_out)
  );
  m.impl("topk.values",
         TORCH_FN(InplaceOrView::topk_out_values)
  );
  m.impl("transpose.int",
         TORCH_FN(InplaceOrView::transpose_int)
  );
  m.impl("triu_",
         TORCH_FN(InplaceOrView::triu_)
  );
  m.impl("triu.out",
         TORCH_FN(InplaceOrView::triu_out_out)
  );
  m.impl("trunc_",
         TORCH_FN(InplaceOrView::trunc_)
  );
  m.impl("trunc.out",
         TORCH_FN(InplaceOrView::trunc_out_out)
  );
  m.impl("uniform_",
         TORCH_FN(InplaceOrView::uniform_)
  );
  m.impl("unsqueeze",
         TORCH_FN(InplaceOrView::unsqueeze)
  );
  m.impl("upsample_bilinear2d_backward.grad_input",
         TORCH_FN(InplaceOrView::upsample_bilinear2d_backward_out_grad_input)
  );
  m.impl("upsample_bilinear2d.out",
         TORCH_FN(InplaceOrView::upsample_bilinear2d_out_out)
  );
  m.impl("upsample_linear1d_backward.grad_input",
         TORCH_FN(InplaceOrView::upsample_linear1d_backward_out_grad_input)
  );
  m.impl("upsample_linear1d.out",
         TORCH_FN(InplaceOrView::upsample_linear1d_out_out)
  );
  m.impl("upsample_nearest2d_backward.grad_input",
         TORCH_FN(InplaceOrView::upsample_nearest2d_backward_out_grad_input)
  );
  m.impl("upsample_nearest2d.out",
         TORCH_FN(InplaceOrView::upsample_nearest2d_out_out)
  );
  m.impl("view",
         TORCH_FN(InplaceOrView::view)
  );
  m.impl("view.dtype",
         TORCH_FN(InplaceOrView::view_dtype)
  );
  m.impl("view_as_complex",
         TORCH_FN(InplaceOrView::view_as_complex)
  );
  m.impl("view_as_real",
         TORCH_FN(InplaceOrView::view_as_real)
  );
  m.impl("zero_",
         TORCH_FN(InplaceOrView::zero_)
  );;
}

}  // namespace
} // namespace torch

