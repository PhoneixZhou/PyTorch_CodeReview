#include "torch/csrc/autograd/VariableTypeUtils.h"
#include "torch/csrc/autograd/FunctionsManual.h"

#include <ATen/RedispatchFunctions.h>
#include <torch/library.h>


// @generated from tools/autograd/templates/VariableType.cpp

// NOTE [Sharded File]: on this file's split-into-shards state
//
// Back in the good old days, VariableType.cpp was generated as one
// file with every function in it, and everything was great and
// simple.
//
// However, this file was also very large (over 36,000 lines), and
// compiling it was very slow, and in fact was a significant
// bottleneck for incremental rebuilds. To address this, we now
// generate the file split across multiple shards, named
// VariableType_0.cpp and so on, which can be compiled in parallel.
//
// For ease of inspection and debugging, so that it's not necessary to
// go rooting around in multiple files, we also generate all the
// functions together in VariableTypeEverything.cpp. This generated
// file is only for convenience; it's not actually used in the
// build. If the file you're looking at now is one of the shards, you
// may want to switch over to the Everything variant to make you
// grepping smoother.

using namespace at;
using namespace torch::autograd::generated;
using namespace torch::autograd::generated::details;

namespace torch { namespace autograd {

namespace VariableType {
namespace{
  void reset_grad_accumulator(Variable & self) {
    AutogradMeta* meta = torch::autograd::impl::get_autograd_meta(self);
    if (meta != nullptr) {
      meta->grad_accumulator_.reset();
    }
  }
}

namespace {
Tensor & __irshift___Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::__irshift__(ks & c10::after_autograd_keyset, self_, other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return self;
}
Tensor & __irshift___Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::__irshift__(ks & c10::after_autograd_keyset, self_, other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  return self;
}
Tensor __rshift___Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::__rshift__(ks & c10::after_autograd_keyset, self_, other);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor __rshift___Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::__rshift__(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  return result;
}
Tensor _adaptive_avg_pool2d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AdaptiveAvgPool2DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AdaptiveAvgPool2DBackward>(new AdaptiveAvgPool2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_adaptive_avg_pool2d(ks & c10::after_autograd_keyset, self_, output_size);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_adaptive_avg_pool2d");
  return result;
}
Tensor & _add_relu_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_add_relu");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_add_relu");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_add_relu_outf(ks & c10::after_autograd_keyset, self_, other_, alpha, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor _amp_update_scale(c10::DispatchKeySet ks, Tensor & growth_tracker, const Tensor & current_scale, const Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  auto& growth_tracker_ = unpack(growth_tracker, "growth_tracker", 0);
  auto& current_scale_ = unpack(current_scale, "current_scale", 1);
  auto& found_inf_ = unpack(found_inf, "found_inf", 2);
  auto _any_requires_grad = compute_requires_grad( growth_tracker, current_scale, found_inf );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_amp_update_scale"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( growth_tracker, current_scale, found_inf ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> growth_tracker__storage_saved =
    growth_tracker_.has_storage() ? c10::optional<Storage>(growth_tracker_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> growth_tracker__impl_saved;
  if (growth_tracker_.defined()) growth_tracker__impl_saved = growth_tracker_.getIntrusivePtr();
  c10::optional<Storage> current_scale__storage_saved =
    current_scale_.has_storage() ? c10::optional<Storage>(current_scale_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> current_scale__impl_saved;
  if (current_scale_.defined()) current_scale__impl_saved = current_scale_.getIntrusivePtr();
  c10::optional<Storage> found_inf__storage_saved =
    found_inf_.has_storage() ? c10::optional<Storage>(found_inf_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> found_inf__impl_saved;
  if (found_inf_.defined()) found_inf__impl_saved = found_inf_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_amp_update_scale(ks & c10::after_autograd_keyset, growth_tracker_, current_scale_, found_inf_, scale_growth_factor, scale_backoff_factor, growth_interval);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (growth_tracker__storage_saved.has_value())
    AT_ASSERT(growth_tracker__storage_saved.value().is_alias_of(growth_tracker_.storage()));
  if (growth_tracker__impl_saved) AT_ASSERT(growth_tracker__impl_saved == growth_tracker_.getIntrusivePtr());
  if (current_scale__storage_saved.has_value())
    AT_ASSERT(current_scale__storage_saved.value().is_alias_of(current_scale_.storage()));
  if (current_scale__impl_saved) AT_ASSERT(current_scale__impl_saved == current_scale_.getIntrusivePtr());
  if (found_inf__storage_saved.has_value())
    AT_ASSERT(found_inf__storage_saved.value().is_alias_of(found_inf_.storage()));
  if (found_inf__impl_saved) AT_ASSERT(found_inf__impl_saved == found_inf_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_amp_update_scale");
  return result;
}
Tensor _bmm(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat2, bool deterministic) {
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  auto _any_requires_grad = compute_requires_grad( self, mat2 );
  (void)_any_requires_grad;
  std::shared_ptr<BmmBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<BmmBackward1>(new BmmBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    grad_fn->deterministic = deterministic;
    if (grad_fn->should_compute_output(0)) {
      grad_fn->mat2_ = SavedVariable(mat2, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_bmm(ks & c10::after_autograd_keyset, self_, mat2_, deterministic);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_bmm");
  return result;
}
Tensor & _cat_out_out(c10::DispatchKeySet ks, TensorList tensors, int64_t dim, Tensor & out) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( tensors )) {
    throw_error_out_requires_grad("_cat");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_cat");
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_cat_outf(ks & c10::after_autograd_keyset, tensors_, dim, out_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor _cdist_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & x1, const Tensor & x2, double p, const Tensor & cdist) {
  auto& grad_ = unpack(grad, "grad", 0);
  auto& x1_ = unpack(x1, "x1", 1);
  auto& x2_ = unpack(x2, "x2", 2);
  auto& cdist_ = unpack(cdist, "cdist", 4);
  auto _any_requires_grad = compute_requires_grad( grad, x1, x2, cdist );
  (void)_any_requires_grad;
  std::shared_ptr<CdistBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CdistBackwardBackward>(new CdistBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad, x1, x2, cdist ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  c10::optional<Storage> x1__storage_saved =
    x1_.has_storage() ? c10::optional<Storage>(x1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> x1__impl_saved;
  if (x1_.defined()) x1__impl_saved = x1_.getIntrusivePtr();
  c10::optional<Storage> x2__storage_saved =
    x2_.has_storage() ? c10::optional<Storage>(x2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> x2__impl_saved;
  if (x2_.defined()) x2__impl_saved = x2_.getIntrusivePtr();
  c10::optional<Storage> cdist__storage_saved =
    cdist_.has_storage() ? c10::optional<Storage>(cdist_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> cdist__impl_saved;
  if (cdist_.defined()) cdist__impl_saved = cdist_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_cdist_backward(ks & c10::after_autograd_keyset, grad_, x1_, x2_, p, cdist_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (x1__storage_saved.has_value())
    AT_ASSERT(x1__storage_saved.value().is_alias_of(x1_.storage()));
  if (x1__impl_saved) AT_ASSERT(x1__impl_saved == x1_.getIntrusivePtr());
  if (x2__storage_saved.has_value())
    AT_ASSERT(x2__storage_saved.value().is_alias_of(x2_.storage()));
  if (x2__impl_saved) AT_ASSERT(x2__impl_saved == x2_.getIntrusivePtr());
  if (cdist__storage_saved.has_value())
    AT_ASSERT(cdist__storage_saved.value().is_alias_of(cdist_.storage()));
  if (cdist__impl_saved) AT_ASSERT(cdist__impl_saved == cdist_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_cdist_backward");
  return result;
}
Tensor _cholesky_helper(c10::DispatchKeySet ks, const Tensor & self, bool upper) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cholesky_helper"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_cholesky_helper(ks & c10::after_autograd_keyset, self_, upper);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_cholesky_helper");
  return result;
}
Tensor & _coalesced_(c10::DispatchKeySet ks, Tensor & self, bool coalesced) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_coalesced_(ks & c10::after_autograd_keyset, self_, coalesced);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return self;
}
Tensor _compute_linear_combination(c10::DispatchKeySet ks, const Tensor & input, const Tensor & coefficients) {
  auto& input_ = unpack(input, "input", 0);
  auto& coefficients_ = unpack(coefficients, "coefficients", 1);
  auto _any_requires_grad = compute_requires_grad( input, coefficients );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_compute_linear_combination"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, coefficients ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> coefficients__storage_saved =
    coefficients_.has_storage() ? c10::optional<Storage>(coefficients_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> coefficients__impl_saved;
  if (coefficients_.defined()) coefficients__impl_saved = coefficients_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_compute_linear_combination(ks & c10::after_autograd_keyset, input_, coefficients_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (coefficients__storage_saved.has_value())
    AT_ASSERT(coefficients__storage_saved.value().is_alias_of(coefficients_.storage()));
  if (coefficients__impl_saved) AT_ASSERT(coefficients__impl_saved == coefficients_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_compute_linear_combination");
  return result;
}
Tensor _conj(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ConjBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ConjBackward>(new ConjBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_conj(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> _cudnn_rnn_backward(c10::DispatchKeySet ks, const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const c10::optional<Tensor> & cx, const Tensor & output, const c10::optional<Tensor> & grad_output, const c10::optional<Tensor> & grad_hy, const c10::optional<Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const c10::optional<Tensor> & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) {
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack(weight, "weight", 1);
  auto& weight_buf_ = unpack(weight_buf, "weight_buf", 3);
  auto& hx_ = unpack(hx, "hx", 4);
  auto& output_ = unpack(output, "output", 6);
  auto& reserve_ = unpack(reserve, "reserve", 20);
  auto _any_requires_grad = compute_requires_grad( input, weight, hx, cx, output, grad_output, grad_hy, grad_cy );
  (void)_any_requires_grad;
  check_no_requires_grad(weight_buf, "weight_buf");
  check_no_requires_grad(reserve, "reserve");
  std::shared_ptr<CudnnRnnBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CudnnRnnBackwardBackward>(new CudnnRnnBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, hx, cx, output, grad_output, grad_hy, grad_cy ));
    grad_fn->weight_size_ = weight.size();
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  std::vector<Tensor> result3;
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  std::vector<c10::optional<Storage>> weight__storage_saved(weight_.size());
  for (const Tensor& tensor : weight_)
    weight__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> weight__impl_saved(weight_.size());
  for (size_t i=0; i<weight_.size(); i++)
    if (weight_[i].defined()) weight__impl_saved[i] = weight_[i].getIntrusivePtr();
  c10::optional<Storage> weight_buf__storage_saved =
    weight_buf_.has_storage() ? c10::optional<Storage>(weight_buf_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight_buf__impl_saved;
  if (weight_buf_.defined()) weight_buf__impl_saved = weight_buf_.getIntrusivePtr();
  c10::optional<Storage> hx__storage_saved =
    hx_.has_storage() ? c10::optional<Storage>(hx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> hx__impl_saved;
  if (hx_.defined()) hx__impl_saved = hx_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> reserve__storage_saved =
    reserve_.has_storage() ? c10::optional<Storage>(reserve_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> reserve__impl_saved;
  if (reserve_.defined()) reserve__impl_saved = reserve_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_cudnn_rnn_backward(ks & c10::after_autograd_keyset, input_, weight_, weight_stride0, weight_buf_, hx_, cx, output_, grad_output, grad_hy, grad_cy, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve_, output_mask);
  })();
  std::tie(result0, result1, result2, result3) = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  for (size_t i=0; i<weight_.size(); i++) {
    if (weight__storage_saved[i].has_value())
      AT_ASSERT(weight__storage_saved[i].value().is_alias_of(weight_[i].storage()));
  }
  for (size_t i=0; i<weight_.size(); i++) {
    if (weight__impl_saved[i])
      AT_ASSERT(weight__impl_saved[i] == weight_[i].getIntrusivePtr());
  }
  if (weight_buf__storage_saved.has_value())
    AT_ASSERT(weight_buf__storage_saved.value().is_alias_of(weight_buf_.storage()));
  if (weight_buf__impl_saved) AT_ASSERT(weight_buf__impl_saved == weight_buf_.getIntrusivePtr());
  if (hx__storage_saved.has_value())
    AT_ASSERT(hx__storage_saved.value().is_alias_of(hx_.storage()));
  if (hx__impl_saved) AT_ASSERT(hx__impl_saved == hx_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (reserve__storage_saved.has_value())
    AT_ASSERT(reserve__storage_saved.value().is_alias_of(reserve_.storage()));
  if (reserve__impl_saved) AT_ASSERT(reserve__impl_saved == reserve_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2, result3 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_cudnn_rnn_backward");
  throw_error_for_complex_autograd(result1, "_cudnn_rnn_backward");
  throw_error_for_complex_autograd(result2, "_cudnn_rnn_backward");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
void _cummax_helper(c10::DispatchKeySet ks, const Tensor & self, Tensor & values, Tensor & indices, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto& values_ = unpack(values, "values", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::_cummax_helper(ks & c10::after_autograd_keyset, self_, values_, indices_, dim);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
}
Tensor & _cumprod_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_cumprod");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_cumprod");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_cumprod_outf(ks & c10::after_autograd_keyset, self_, dim, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor _cumsum(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cumsum"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_cumsum(ks & c10::after_autograd_keyset, self_, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_cumsum");
  return result;
}
Tensor _dirichlet_grad(c10::DispatchKeySet ks, const Tensor & x, const Tensor & alpha, const Tensor & total) {
  auto& x_ = unpack(x, "x", 0);
  auto& alpha_ = unpack(alpha, "alpha", 1);
  auto& total_ = unpack(total, "total", 2);
  auto _any_requires_grad = compute_requires_grad( x, alpha, total );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_dirichlet_grad"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( x, alpha, total ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> x__storage_saved =
    x_.has_storage() ? c10::optional<Storage>(x_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> x__impl_saved;
  if (x_.defined()) x__impl_saved = x_.getIntrusivePtr();
  c10::optional<Storage> alpha__storage_saved =
    alpha_.has_storage() ? c10::optional<Storage>(alpha_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> alpha__impl_saved;
  if (alpha_.defined()) alpha__impl_saved = alpha_.getIntrusivePtr();
  c10::optional<Storage> total__storage_saved =
    total_.has_storage() ? c10::optional<Storage>(total_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> total__impl_saved;
  if (total_.defined()) total__impl_saved = total_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_dirichlet_grad(ks & c10::after_autograd_keyset, x_, alpha_, total_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (x__storage_saved.has_value())
    AT_ASSERT(x__storage_saved.value().is_alias_of(x_.storage()));
  if (x__impl_saved) AT_ASSERT(x__impl_saved == x_.getIntrusivePtr());
  if (alpha__storage_saved.has_value())
    AT_ASSERT(alpha__storage_saved.value().is_alias_of(alpha_.storage()));
  if (alpha__impl_saved) AT_ASSERT(alpha__impl_saved == alpha_.getIntrusivePtr());
  if (total__storage_saved.has_value())
    AT_ASSERT(total__storage_saved.value().is_alias_of(total_.storage()));
  if (total__impl_saved) AT_ASSERT(total__impl_saved == total_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_dirichlet_grad");
  return result;
}
std::vector<Tensor> _foreach_ceil(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_ceil"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_ceil(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
void _foreach_ceil_(c10::DispatchKeySet ks, TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_ceil_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
}
std::vector<Tensor> _foreach_div_Scalar(c10::DispatchKeySet ks, TensorList tensors, const Scalar & scalar) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_div"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_div(ks & c10::after_autograd_keyset, tensors_, scalar);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
std::vector<Tensor> _foreach_div_List(c10::DispatchKeySet ks, TensorList tensors1, TensorList tensors2) {
  auto tensors1_ = unpack(tensors1, "tensors1", 0);
  auto tensors2_ = unpack(tensors2, "tensors2", 1);
  auto _any_requires_grad = compute_requires_grad( tensors1, tensors2 );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_div"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors1, tensors2 ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors1__storage_saved(tensors1_.size());
  for (const Tensor& tensor : tensors1_)
    tensors1__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors1__impl_saved(tensors1_.size());
  for (size_t i=0; i<tensors1_.size(); i++)
    if (tensors1_[i].defined()) tensors1__impl_saved[i] = tensors1_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensors2__storage_saved(tensors2_.size());
  for (const Tensor& tensor : tensors2_)
    tensors2__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors2__impl_saved(tensors2_.size());
  for (size_t i=0; i<tensors2_.size(); i++)
    if (tensors2_[i].defined()) tensors2__impl_saved[i] = tensors2_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_div(ks & c10::after_autograd_keyset, tensors1_, tensors2_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors1_.size(); i++) {
    if (tensors1__storage_saved[i].has_value())
      AT_ASSERT(tensors1__storage_saved[i].value().is_alias_of(tensors1_[i].storage()));
  }
  for (size_t i=0; i<tensors1_.size(); i++) {
    if (tensors1__impl_saved[i])
      AT_ASSERT(tensors1__impl_saved[i] == tensors1_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensors2_.size(); i++) {
    if (tensors2__storage_saved[i].has_value())
      AT_ASSERT(tensors2__storage_saved[i].value().is_alias_of(tensors2_[i].storage()));
  }
  for (size_t i=0; i<tensors2_.size(); i++) {
    if (tensors2__impl_saved[i])
      AT_ASSERT(tensors2__impl_saved[i] == tensors2_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
std::vector<Tensor> _foreach_div_ScalarList(c10::DispatchKeySet ks, TensorList tensors, ArrayRef<Scalar> scalars) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_div"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_div(ks & c10::after_autograd_keyset, tensors_, scalars);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
void _foreach_div__Scalar(c10::DispatchKeySet ks, TensorList self, const Scalar & scalar) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_div_(ks & c10::after_autograd_keyset, self_, scalar);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
}
void _foreach_div__List(c10::DispatchKeySet ks, TensorList self, TensorList other) {
  auto self_ = unpack(self, "self", 0);
  auto other_ = unpack(other, "other", 1);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> other__storage_saved(other_.size());
  for (const Tensor& tensor : other_)
    other__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> other__impl_saved(other_.size());
  for (size_t i=0; i<other_.size(); i++)
    if (other_[i].defined()) other__impl_saved[i] = other_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_div_(ks & c10::after_autograd_keyset, self_, other_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<other_.size(); i++) {
    if (other__storage_saved[i].has_value())
      AT_ASSERT(other__storage_saved[i].value().is_alias_of(other_[i].storage()));
  }
  for (size_t i=0; i<other_.size(); i++) {
    if (other__impl_saved[i])
      AT_ASSERT(other__impl_saved[i] == other_[i].getIntrusivePtr());
  }
  #endif
}
void _foreach_div__ScalarList(c10::DispatchKeySet ks, TensorList self, ArrayRef<Scalar> scalars) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_div_(ks & c10::after_autograd_keyset, self_, scalars);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
}
std::vector<Tensor> _foreach_exp(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_exp"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_exp(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
void _foreach_exp_(c10::DispatchKeySet ks, TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_exp_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
}
std::vector<Tensor> _foreach_lgamma(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_lgamma"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_lgamma(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
void _foreach_lgamma_(c10::DispatchKeySet ks, TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_lgamma_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
}
std::vector<Tensor> _foreach_log1p(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_log1p"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_log1p(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
void _foreach_log1p_(c10::DispatchKeySet ks, TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_log1p_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
}
std::vector<Tensor> _foreach_sigmoid(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_sigmoid"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_sigmoid(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
void _foreach_sigmoid_(c10::DispatchKeySet ks, TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_sigmoid_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
}
std::vector<Tensor> _foreach_sqrt(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_sqrt"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_sqrt(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
void _foreach_sqrt_(c10::DispatchKeySet ks, TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_sqrt_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
}
std::vector<Tensor> _foreach_tan(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_tan"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_tan(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
void _foreach_tan_(c10::DispatchKeySet ks, TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_tan_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
}
void _foreach_zero_(c10::DispatchKeySet ks, TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_zero_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
}
std::tuple<Tensor,Tensor> _fused_dropout(c10::DispatchKeySet ks, const Tensor & self, double p, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<FusedDropoutBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FusedDropoutBackward>(new FusedDropoutBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->p = p;
  }
  Tensor result0;
  Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_fused_dropout(ks & c10::after_autograd_keyset, self_, p, generator);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  throw_error_for_complex_autograd(result0, "_fused_dropout");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor _logcumsumexp(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_logcumsumexp"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_logcumsumexp(ks & c10::after_autograd_keyset, self_, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_logcumsumexp");
  return result;
}
Tensor _make_per_tensor_quantized_tensor(c10::DispatchKeySet ks, const Tensor & self, double scale, int64_t zero_point) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_make_per_tensor_quantized_tensor"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_make_per_tensor_quantized_tensor(ks & c10::after_autograd_keyset, self_, scale, zero_point);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_make_per_tensor_quantized_tensor");
  return result;
}
std::tuple<Tensor,Tensor> _mode(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_mode"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_mode(ks & c10::after_autograd_keyset, self_, dim, keepdim);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_mode");
  throw_error_for_complex_autograd(result1, "_mode");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor _nnpack_spatial_convolution(c10::DispatchKeySet ks, const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef stride) {
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( input, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<NnpackSpatialConvolutionBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NnpackSpatialConvolutionBackward>(new NnpackSpatialConvolutionBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_argsize_2 = weight.size(2);
    grad_fn->weight_argsize_3 = weight.size(3);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_nnpack_spatial_convolution(ks & c10::after_autograd_keyset, input_, weight_, bias, padding, stride);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_nnpack_spatial_convolution");
  return result;
}
Tensor _softmax_backward_data(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  auto& self_ = unpack(self, "self", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<SoftmaxBackwardDataBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SoftmaxBackwardDataBackward>(new SoftmaxBackwardDataBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->dim = dim;
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_softmax_backward_data(ks & c10::after_autograd_keyset, grad_output_, output_, dim, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_softmax_backward_data");
  return result;
}
Tensor _sparse_log_softmax(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool half_to_float) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SparseLogSoftmaxBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SparseLogSoftmaxBackward>(new SparseLogSoftmaxBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_sparse_log_softmax(ks & c10::after_autograd_keyset, self_, dim, half_to_float);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "_sparse_log_softmax");
  return result;
}
Tensor _sparse_sum_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & self, IntArrayRef dim) {
  auto& grad_ = unpack(grad, "grad", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_sparse_sum_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad, self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_sparse_sum_backward(ks & c10::after_autograd_keyset, grad_, self_, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_sparse_sum_backward");
  return result;
}
Tensor _std(c10::DispatchKeySet ks, const Tensor & self, bool unbiased) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_std"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_std(ks & c10::after_autograd_keyset, self_, unbiased);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_std");
  return result;
}
Tensor _test_optional_floatlist(c10::DispatchKeySet ks, const Tensor & values, c10::optional<ArrayRef<double>> addends) {
  auto& values_ = unpack(values, "values", 0);
  auto _any_requires_grad = compute_requires_grad( values );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_test_optional_floatlist"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( values ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_test_optional_floatlist(ks & c10::after_autograd_keyset, values_, addends);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_test_optional_floatlist");
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_gru_cell_backward(c10::DispatchKeySet ks, const Tensor & grad_hy, const Tensor & workspace, bool has_bias) {
  auto& grad_hy_ = unpack(grad_hy, "grad_hy", 0);
  auto& workspace_ = unpack(workspace, "workspace", 1);
  auto _any_requires_grad = compute_requires_grad( grad_hy, workspace );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_fused_gru_cell_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_hy, workspace ));
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  Tensor result4;
  #ifndef NDEBUG
  c10::optional<Storage> grad_hy__storage_saved =
    grad_hy_.has_storage() ? c10::optional<Storage>(grad_hy_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_hy__impl_saved;
  if (grad_hy_.defined()) grad_hy__impl_saved = grad_hy_.getIntrusivePtr();
  c10::optional<Storage> workspace__storage_saved =
    workspace_.has_storage() ? c10::optional<Storage>(workspace_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> workspace__impl_saved;
  if (workspace_.defined()) workspace__impl_saved = workspace_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_thnn_fused_gru_cell_backward(ks & c10::after_autograd_keyset, grad_hy_, workspace_, has_bias);
  })();
  std::tie(result0, result1, result2, result3, result4) = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_hy__storage_saved.has_value())
    AT_ASSERT(grad_hy__storage_saved.value().is_alias_of(grad_hy_.storage()));
  if (grad_hy__impl_saved) AT_ASSERT(grad_hy__impl_saved == grad_hy_.getIntrusivePtr());
  if (workspace__storage_saved.has_value())
    AT_ASSERT(workspace__storage_saved.value().is_alias_of(workspace_.storage()));
  if (workspace__impl_saved) AT_ASSERT(workspace__impl_saved == workspace_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2, result3, result4 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_thnn_fused_gru_cell_backward");
  throw_error_for_complex_autograd(result1, "_thnn_fused_gru_cell_backward");
  throw_error_for_complex_autograd(result2, "_thnn_fused_gru_cell_backward");
  throw_error_for_complex_autograd(result3, "_thnn_fused_gru_cell_backward");
  throw_error_for_complex_autograd(result4, "_thnn_fused_gru_cell_backward");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
Tensor _values(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoNonVariableTypeMode guard;
    return at::redispatch::_values(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface_backward(c10::DispatchKeySet ks, const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) {
  auto& grad_w_ = unpack(grad_w, "grad_w", 0);
  auto& saved_v_ = unpack(saved_v, "saved_v", 1);
  auto& saved_g_ = unpack(saved_g, "saved_g", 2);
  auto& saved_norms_ = unpack(saved_norms, "saved_norms", 3);
  auto _any_requires_grad = compute_requires_grad( grad_w, saved_v, saved_g, saved_norms );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_weight_norm_cuda_interface_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_w, saved_v, saved_g, saved_norms ));
  }
  Tensor result0;
  Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> grad_w__storage_saved =
    grad_w_.has_storage() ? c10::optional<Storage>(grad_w_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_w__impl_saved;
  if (grad_w_.defined()) grad_w__impl_saved = grad_w_.getIntrusivePtr();
  c10::optional<Storage> saved_v__storage_saved =
    saved_v_.has_storage() ? c10::optional<Storage>(saved_v_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> saved_v__impl_saved;
  if (saved_v_.defined()) saved_v__impl_saved = saved_v_.getIntrusivePtr();
  c10::optional<Storage> saved_g__storage_saved =
    saved_g_.has_storage() ? c10::optional<Storage>(saved_g_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> saved_g__impl_saved;
  if (saved_g_.defined()) saved_g__impl_saved = saved_g_.getIntrusivePtr();
  c10::optional<Storage> saved_norms__storage_saved =
    saved_norms_.has_storage() ? c10::optional<Storage>(saved_norms_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> saved_norms__impl_saved;
  if (saved_norms_.defined()) saved_norms__impl_saved = saved_norms_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_weight_norm_cuda_interface_backward(ks & c10::after_autograd_keyset, grad_w_, saved_v_, saved_g_, saved_norms_, dim);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_w__storage_saved.has_value())
    AT_ASSERT(grad_w__storage_saved.value().is_alias_of(grad_w_.storage()));
  if (grad_w__impl_saved) AT_ASSERT(grad_w__impl_saved == grad_w_.getIntrusivePtr());
  if (saved_v__storage_saved.has_value())
    AT_ASSERT(saved_v__storage_saved.value().is_alias_of(saved_v_.storage()));
  if (saved_v__impl_saved) AT_ASSERT(saved_v__impl_saved == saved_v_.getIntrusivePtr());
  if (saved_g__storage_saved.has_value())
    AT_ASSERT(saved_g__storage_saved.value().is_alias_of(saved_g_.storage()));
  if (saved_g__impl_saved) AT_ASSERT(saved_g__impl_saved == saved_g_.getIntrusivePtr());
  if (saved_norms__storage_saved.has_value())
    AT_ASSERT(saved_norms__storage_saved.value().is_alias_of(saved_norms_.storage()));
  if (saved_norms__impl_saved) AT_ASSERT(saved_norms__impl_saved == saved_norms_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_weight_norm_cuda_interface_backward");
  throw_error_for_complex_autograd(result1, "_weight_norm_cuda_interface_backward");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & acos_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("acos");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("acos");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::acos_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor acosh(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AcoshBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AcoshBackward0>(new AcoshBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::acosh(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & acosh_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<AcoshBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AcoshBackward1>(new AcoshBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::acosh_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & adaptive_avg_pool3d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::adaptive_avg_pool3d_outf(ks & c10::after_autograd_keyset, self_, output_size, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor adaptive_max_pool3d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<AdaptiveMaxPool3DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AdaptiveMaxPool3DBackwardBackward>(new AdaptiveMaxPool3DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::adaptive_max_pool3d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, indices_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "adaptive_max_pool3d_backward");
  return result;
}
Tensor & add_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("add");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("add");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::add_outf(ks & c10::after_autograd_keyset, self_, other_, alpha, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor addr(c10::DispatchKeySet ks, const Tensor & self, const Tensor & vec1, const Tensor & vec2, const Scalar & beta, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& vec1_ = unpack(vec1, "vec1", 1);
  auto& vec2_ = unpack(vec2, "vec2", 2);
  auto _any_requires_grad = compute_requires_grad( self, vec1, vec2 );
  (void)_any_requires_grad;
  std::shared_ptr<AddrBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddrBackward>(new AddrBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, vec1, vec2 ));
    grad_fn->beta = beta;
    if (grad_fn->should_compute_output(1)) {
      grad_fn->vec2_ = SavedVariable(vec2, false);
    }
    grad_fn->alpha = alpha;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->vec1_ = SavedVariable(vec1, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> vec1__storage_saved =
    vec1_.has_storage() ? c10::optional<Storage>(vec1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec1__impl_saved;
  if (vec1_.defined()) vec1__impl_saved = vec1_.getIntrusivePtr();
  c10::optional<Storage> vec2__storage_saved =
    vec2_.has_storage() ? c10::optional<Storage>(vec2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec2__impl_saved;
  if (vec2_.defined()) vec2__impl_saved = vec2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::addr(ks & c10::after_autograd_keyset, self_, vec1_, vec2_, beta, alpha);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec1__storage_saved.has_value())
    AT_ASSERT(vec1__storage_saved.value().is_alias_of(vec1_.storage()));
  if (vec1__impl_saved) AT_ASSERT(vec1__impl_saved == vec1_.getIntrusivePtr());
  if (vec2__storage_saved.has_value())
    AT_ASSERT(vec2__storage_saved.value().is_alias_of(vec2_.storage()));
  if (vec2__impl_saved) AT_ASSERT(vec2__impl_saved == vec2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & addr_(c10::DispatchKeySet ks, Tensor & self, const Tensor & vec1, const Tensor & vec2, const Scalar & beta, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& vec1_ = unpack(vec1, "vec1", 1);
  auto& vec2_ = unpack(vec2, "vec2", 2);
  auto _any_requires_grad = compute_requires_grad( self, vec1, vec2 );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<AddrBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddrBackward>(new AddrBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, vec1, vec2 ));
    grad_fn->beta = beta;
    if (grad_fn->should_compute_output(1)) {
      grad_fn->vec2_ = SavedVariable(vec2, false);
    }
    grad_fn->alpha = alpha;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->vec1_ = SavedVariable(vec1, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> vec1__storage_saved =
    vec1_.has_storage() ? c10::optional<Storage>(vec1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec1__impl_saved;
  if (vec1_.defined()) vec1__impl_saved = vec1_.getIntrusivePtr();
  c10::optional<Storage> vec2__storage_saved =
    vec2_.has_storage() ? c10::optional<Storage>(vec2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec2__impl_saved;
  if (vec2_.defined()) vec2__impl_saved = vec2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::addr_(ks & c10::after_autograd_keyset, self_, vec1_, vec2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec1__storage_saved.has_value())
    AT_ASSERT(vec1__storage_saved.value().is_alias_of(vec1_.storage()));
  if (vec1__impl_saved) AT_ASSERT(vec1__impl_saved == vec1_.getIntrusivePtr());
  if (vec2__storage_saved.has_value())
    AT_ASSERT(vec2__storage_saved.value().is_alias_of(vec2_.storage()));
  if (vec2__impl_saved) AT_ASSERT(vec2__impl_saved == vec2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor affine_grid_generator(c10::DispatchKeySet ks, const Tensor & theta, IntArrayRef size, bool align_corners) {
  auto& theta_ = unpack(theta, "theta", 0);
  auto _any_requires_grad = compute_requires_grad( theta );
  (void)_any_requires_grad;
  std::shared_ptr<AffineGridGeneratorBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AffineGridGeneratorBackward>(new AffineGridGeneratorBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( theta ));
    grad_fn->size = size.vec();
    grad_fn->align_corners = align_corners;
  }
  #ifndef NDEBUG
  c10::optional<Storage> theta__storage_saved =
    theta_.has_storage() ? c10::optional<Storage>(theta_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> theta__impl_saved;
  if (theta_.defined()) theta__impl_saved = theta_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::affine_grid_generator(ks & c10::after_autograd_keyset, theta_, size, align_corners);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (theta__storage_saved.has_value())
    AT_ASSERT(theta__storage_saved.value().is_alias_of(theta_.storage()));
  if (theta__impl_saved) AT_ASSERT(theta__impl_saved == theta_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "affine_grid_generator");
  return result;
}
Tensor amin(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AminBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AminBackward>(new AminBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim.vec();
    grad_fn->keepdim = keepdim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::amin(ks & c10::after_autograd_keyset, self_, dim, keepdim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "amin");
  return result;
}
Tensor & arange_out_start_out(c10::DispatchKeySet ks, const Scalar & start, const Scalar & end, const Scalar & step, Tensor & out) {
  auto& out_ = unpack(out, "out", 3);
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::arange_outf(ks & c10::after_autograd_keyset, start, end, step, out_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  return out;
}
Tensor & asin_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("asin");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("asin");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::asin_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor asinh(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AsinhBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AsinhBackward0>(new AsinhBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::asinh(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & asinh_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<AsinhBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AsinhBackward1>(new AsinhBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::asinh_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor avg_pool2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<AvgPool2DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AvgPool2DBackwardBackward>(new AvgPool2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
    grad_fn->divisor_override = divisor_override;
    grad_fn->self_info = self;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::avg_pool2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "avg_pool2d_backward");
  return result;
}
Tensor avg_pool3d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AvgPool3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AvgPool3DBackward>(new AvgPool3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
    grad_fn->divisor_override = divisor_override;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::avg_pool3d(ks & c10::after_autograd_keyset, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "avg_pool3d");
  return result;
}
Tensor & avg_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 8);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("avg_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("avg_pool3d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::avg_pool3d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  return grad_input;
}
Tensor baddbmm(c10::DispatchKeySet ks, const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  auto _any_requires_grad = compute_requires_grad( self, batch1, batch2 );
  (void)_any_requires_grad;
  std::shared_ptr<BaddbmmBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<BaddbmmBackward>(new BaddbmmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->batch2_ = SavedVariable(batch2, false);
    }
    grad_fn->alpha = alpha;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->batch1_ = SavedVariable(batch1, false);
    }
    grad_fn->beta = beta;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> batch1__storage_saved =
    batch1_.has_storage() ? c10::optional<Storage>(batch1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch1__impl_saved;
  if (batch1_.defined()) batch1__impl_saved = batch1_.getIntrusivePtr();
  c10::optional<Storage> batch2__storage_saved =
    batch2_.has_storage() ? c10::optional<Storage>(batch2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch2__impl_saved;
  if (batch2_.defined()) batch2__impl_saved = batch2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::baddbmm(ks & c10::after_autograd_keyset, self_, batch1_, batch2_, beta, alpha);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (batch1__storage_saved.has_value())
    AT_ASSERT(batch1__storage_saved.value().is_alias_of(batch1_.storage()));
  if (batch1__impl_saved) AT_ASSERT(batch1__impl_saved == batch1_.getIntrusivePtr());
  if (batch2__storage_saved.has_value())
    AT_ASSERT(batch2__storage_saved.value().is_alias_of(batch2_.storage()));
  if (batch2__impl_saved) AT_ASSERT(batch2__impl_saved == batch2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & baddbmm_(c10::DispatchKeySet ks, Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  auto _any_requires_grad = compute_requires_grad( self, batch1, batch2 );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<BaddbmmBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<BaddbmmBackward>(new BaddbmmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->batch2_ = SavedVariable(batch2, false);
    }
    grad_fn->alpha = alpha;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->batch1_ = SavedVariable(batch1, false);
    }
    grad_fn->beta = beta;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> batch1__storage_saved =
    batch1_.has_storage() ? c10::optional<Storage>(batch1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch1__impl_saved;
  if (batch1_.defined()) batch1__impl_saved = batch1_.getIntrusivePtr();
  c10::optional<Storage> batch2__storage_saved =
    batch2_.has_storage() ? c10::optional<Storage>(batch2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch2__impl_saved;
  if (batch2_.defined()) batch2__impl_saved = batch2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::baddbmm_(ks & c10::after_autograd_keyset, self_, batch1_, batch2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (batch1__storage_saved.has_value())
    AT_ASSERT(batch1__storage_saved.value().is_alias_of(batch1_.storage()));
  if (batch1__impl_saved) AT_ASSERT(batch1__impl_saved == batch1_.getIntrusivePtr());
  if (batch2__storage_saved.has_value())
    AT_ASSERT(batch2__storage_saved.value().is_alias_of(batch2_.storage()));
  if (batch2__impl_saved) AT_ASSERT(batch2__impl_saved == batch2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> batch_norm_backward_reduce(c10::DispatchKeySet ks, const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & weight, bool input_g, bool weight_g, bool bias_g) {
  auto& grad_out_ = unpack(grad_out, "grad_out", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& mean_ = unpack(mean, "mean", 2);
  auto& invstd_ = unpack(invstd, "invstd", 3);
  auto _any_requires_grad = compute_requires_grad( grad_out, input, mean, invstd, weight );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_backward_reduce"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_out, input, mean, invstd, weight ));
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  #ifndef NDEBUG
  c10::optional<Storage> grad_out__storage_saved =
    grad_out_.has_storage() ? c10::optional<Storage>(grad_out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_out__impl_saved;
  if (grad_out_.defined()) grad_out__impl_saved = grad_out_.getIntrusivePtr();
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> invstd__storage_saved =
    invstd_.has_storage() ? c10::optional<Storage>(invstd_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> invstd__impl_saved;
  if (invstd_.defined()) invstd__impl_saved = invstd_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::batch_norm_backward_reduce(ks & c10::after_autograd_keyset, grad_out_, input_, mean_, invstd_, weight, input_g, weight_g, bias_g);
  })();
  std::tie(result0, result1, result2, result3) = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_out__storage_saved.has_value())
    AT_ASSERT(grad_out__storage_saved.value().is_alias_of(grad_out_.storage()));
  if (grad_out__impl_saved) AT_ASSERT(grad_out__impl_saved == grad_out_.getIntrusivePtr());
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (invstd__storage_saved.has_value())
    AT_ASSERT(invstd__storage_saved.value().is_alias_of(invstd_.storage()));
  if (invstd__impl_saved) AT_ASSERT(invstd__impl_saved == invstd_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2, result3 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "batch_norm_backward_reduce");
  throw_error_for_complex_autograd(result1, "batch_norm_backward_reduce");
  throw_error_for_complex_autograd(result2, "batch_norm_backward_reduce");
  throw_error_for_complex_autograd(result3, "batch_norm_backward_reduce");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
Tensor & bernoulli_out_out(c10::DispatchKeySet ks, const Tensor & self, c10::optional<Generator> generator, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("bernoulli");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bernoulli");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::bernoulli_outf(ks & c10::after_autograd_keyset, self_, generator, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor binary_cross_entropy_with_logits(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & pos_weight, int64_t reduction) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self, target );
  (void)_any_requires_grad;
  check_no_requires_grad(weight, "weight");
  check_no_requires_grad(pos_weight, "pos_weight");
  std::shared_ptr<BinaryCrossEntropyWithLogitsBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<BinaryCrossEntropyWithLogitsBackward>(new BinaryCrossEntropyWithLogitsBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->pos_weight_ = SavedVariable(pos_weight, false);
    grad_fn->reduction = reduction;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::binary_cross_entropy_with_logits(ks & c10::after_autograd_keyset, self_, target_, weight, pos_weight, reduction);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "binary_cross_entropy_with_logits");
  return result;
}
Tensor bincount(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Tensor> & weights, int64_t minlength) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self, weights );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("bincount"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weights ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::bincount(ks & c10::after_autograd_keyset, self_, weights, minlength);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "bincount");
  return result;
}
Tensor bmm(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat2) {
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  auto _any_requires_grad = compute_requires_grad( self, mat2 );
  (void)_any_requires_grad;
  std::shared_ptr<BmmBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<BmmBackward0>(new BmmBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    if (grad_fn->should_compute_output(0)) {
      grad_fn->mat2_ = SavedVariable(mat2, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::bmm(ks & c10::after_autograd_keyset, self_, mat2_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor bucketize_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & boundaries, bool out_int32, bool right) {
  auto& self_ = unpack(self, "self", 0);
  auto& boundaries_ = unpack(boundaries, "boundaries", 1);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> boundaries__storage_saved =
    boundaries_.has_storage() ? c10::optional<Storage>(boundaries_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> boundaries__impl_saved;
  if (boundaries_.defined()) boundaries__impl_saved = boundaries_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::bucketize(ks & c10::after_autograd_keyset, self_, boundaries_, out_int32, right);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (boundaries__storage_saved.has_value())
    AT_ASSERT(boundaries__storage_saved.value().is_alias_of(boundaries_.storage()));
  if (boundaries__impl_saved) AT_ASSERT(boundaries__impl_saved == boundaries_.getIntrusivePtr());
  #endif
  return result;
}
Tensor bucketize_Scalar(c10::DispatchKeySet ks, const Scalar & self, const Tensor & boundaries, bool out_int32, bool right) {
  auto& boundaries_ = unpack(boundaries, "boundaries", 1);
  #ifndef NDEBUG
  c10::optional<Storage> boundaries__storage_saved =
    boundaries_.has_storage() ? c10::optional<Storage>(boundaries_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> boundaries__impl_saved;
  if (boundaries_.defined()) boundaries__impl_saved = boundaries_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::bucketize(ks & c10::after_autograd_keyset, self, boundaries_, out_int32, right);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (boundaries__storage_saved.has_value())
    AT_ASSERT(boundaries__storage_saved.value().is_alias_of(boundaries_.storage()));
  if (boundaries__impl_saved) AT_ASSERT(boundaries__impl_saved == boundaries_.getIntrusivePtr());
  #endif
  return result;
}
Tensor & cat_out_out(c10::DispatchKeySet ks, TensorList tensors, int64_t dim, Tensor & out) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( tensors )) {
    throw_error_out_requires_grad("cat");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cat");
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::cat_outf(ks & c10::after_autograd_keyset, tensors_, dim, out_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor cholesky(c10::DispatchKeySet ks, const Tensor & self, bool upper) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<CholeskyBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CholeskyBackward>(new CholeskyBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->upper = upper;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::cholesky(ks & c10::after_autograd_keyset, self_, upper);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor clamp_max(c10::DispatchKeySet ks, const Tensor & self, const Scalar & max) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ClampMaxBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ClampMaxBackward>(new ClampMaxBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->max = max;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::clamp_max(ks & c10::after_autograd_keyset, self_, max);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "clamp_max");
  return result;
}
Tensor & clamp_max_(c10::DispatchKeySet ks, Tensor & self, const Scalar & max) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<ClampMaxBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ClampMaxBackward>(new ClampMaxBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->max = max;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::clamp_max_(ks & c10::after_autograd_keyset, self_, max);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor coalesce(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<CoalesceBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CoalesceBackward>(new CoalesceBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::coalesce(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "coalesce");
  return result;
}
Tensor & col2im_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 6);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("col2im");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("col2im");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::col2im_outf(ks & c10::after_autograd_keyset, self_, output_size, kernel_size, dilation, padding, stride, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor conv_tbc(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& bias_ = unpack(bias, "bias", 2);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<ConvTbcBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ConvTbcBackward>(new ConvTbcBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->bias_ = SavedVariable(bias, false);
    grad_fn->pad = pad;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::conv_tbc(ks & c10::after_autograd_keyset, self_, weight_, bias_, pad);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "conv_tbc");
  return result;
}
Tensor copysign_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<CopysignBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CopysignBackward0>(new CopysignBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    if (grad_fn->should_compute_output(0)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::copysign(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "copysign");
  return result;
}
Tensor copysign_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<CopysignBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CopysignBackward1>(new CopysignBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::copysign(ks & c10::after_autograd_keyset, self_, other);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "copysign");
  return result;
}
Tensor & copysign__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<CopysignBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CopysignBackward0>(new CopysignBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    if (grad_fn->should_compute_output(0)) {
      grad_fn->self_ = SavedVariable(self.clone(), false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::copysign_(ks & c10::after_autograd_keyset, self_, other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
Tensor & copysign__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<CopysignBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CopysignBackward1>(new CopysignBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::copysign_(ks & c10::after_autograd_keyset, self_, other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
Tensor cudnn_convolution_backward_weight(c10::DispatchKeySet ks, IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_backward_weight"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::cudnn_convolution_backward_weight(ks & c10::after_autograd_keyset, weight_size, grad_output_, self_, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "cudnn_convolution_backward_weight");
  return result;
}
std::tuple<Tensor,Tensor> cummax(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<CummaxBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CummaxBackward>(new CummaxBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  Tensor values;
  Tensor indices;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::cummax(ks & c10::after_autograd_keyset, self_, dim);
  })();
  std::tie(values, indices) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  throw_error_for_complex_autograd(values, "cummax");
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor & cumprod_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cumprod");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cumprod");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::cumprod_outf(ks & c10::after_autograd_keyset, self_, dim, dtype, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor cumsum(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<CumsumBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CumsumBackward>(new CumsumBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->dim = dim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::cumsum(ks & c10::after_autograd_keyset, self_, dim, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & cumsum_(c10::DispatchKeySet ks, Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<CumsumBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CumsumBackward>(new CumsumBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->dim = dim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::cumsum_(ks & c10::after_autograd_keyset, self_, dim, dtype);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
int64_t dense_dim(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::dense_dim(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = _tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor diagonal(c10::DispatchKeySet ks, const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<DiagonalBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DiagonalBackward>(new DiagonalBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->offset = offset;
    grad_fn->dim1 = dim1;
    grad_fn->dim2 = dim2;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoNonVariableTypeMode guard;
    return at::redispatch::diagonal(ks & c10::after_autograd_keyset, self_, offset, dim1, dim2);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor dist(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & p) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<DistBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DistBackward>(new DistBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
    grad_fn->p = p;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::dist(ks & c10::after_autograd_keyset, self_, other_, p);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "dist");
  return result;
}
Tensor & dot_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & tensor, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& tensor_ = unpack(tensor, "tensor", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, tensor );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, tensor )) {
    throw_error_out_requires_grad("dot");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("dot");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor__storage_saved =
    tensor_.has_storage() ? c10::optional<Storage>(tensor_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor__impl_saved;
  if (tensor_.defined()) tensor__impl_saved = tensor_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::dot_outf(ks & c10::after_autograd_keyset, self_, tensor_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor__storage_saved.has_value())
    AT_ASSERT(tensor__storage_saved.value().is_alias_of(tensor_.storage()));
  if (tensor__impl_saved) AT_ASSERT(tensor__impl_saved == tensor_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor elu(c10::DispatchKeySet ks, const Tensor & self, const Scalar & alpha, const Scalar & scale, const Scalar & input_scale) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<EluBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<EluBackward0>(new EluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->alpha = alpha;
    grad_fn->scale = scale;
    grad_fn->input_scale = input_scale;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::elu(ks & c10::after_autograd_keyset, self_, alpha, scale, input_scale);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "elu");
  return result;
}
Tensor & elu_(c10::DispatchKeySet ks, Tensor & self, const Scalar & alpha, const Scalar & scale, const Scalar & input_scale) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<EluBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<EluBackward1>(new EluBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->alpha = alpha;
    grad_fn->scale = scale;
    grad_fn->input_scale = input_scale;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::elu_(ks & c10::after_autograd_keyset, self_, alpha, scale, input_scale);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
Tensor & embedding_renorm_(c10::DispatchKeySet ks, Tensor & self, const Tensor & indices, double max_norm, double norm_type) {
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<EmbeddingRenormBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<EmbeddingRenormBackward>(new EmbeddingRenormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::embedding_renorm_(ks & c10::after_autograd_keyset, self_, indices_, max_norm, norm_type);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
bool equal(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::equal(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = _tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  return result;
}
Tensor & erf_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("erf");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("erf");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::erf_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor erfc(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ErfcBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ErfcBackward>(new ErfcBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::erfc(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "erfc");
  return result;
}
Tensor & erfc_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<ErfcBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ErfcBackward>(new ErfcBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::erfc_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor expm1(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Expm1Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Expm1Backward>(new Expm1Backward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::expm1(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "expm1");
  return result;
}
Tensor & expm1_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<Expm1Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Expm1Backward>(new Expm1Backward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::expm1_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
Tensor & exponential_(c10::DispatchKeySet ks, Tensor & self, double lambd, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<ExponentialBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ExponentialBackward>(new ExponentialBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::exponential_(ks & c10::after_autograd_keyset, self_, lambd, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor floor(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<FloorBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FloorBackward>(new FloorBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::floor(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "floor");
  return result;
}
Tensor & floor_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<FloorBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FloorBackward>(new FloorBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::floor_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor fmin(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<FminBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FminBackward>(new FminBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::fmin(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "fmin");
  return result;
}
Tensor & fmod_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("fmod");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("fmod");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::fmod_outf(ks & c10::after_autograd_keyset, self_, other, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & fmod_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("fmod");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("fmod");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::fmod_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & frac_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("frac");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("frac");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::frac_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out_output(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples, Tensor & output, Tensor & indices) {
  auto& self_ = unpack(self, "self", 0);
  auto& random_samples_ = unpack(random_samples, "random_samples", 3);
  auto& output_ = unpack(output, "output", 4);
  auto& indices_ = unpack(indices, "indices", 5);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, random_samples )) {
    throw_error_out_requires_grad("fractional_max_pool2d");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("fractional_max_pool2d");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> random_samples__storage_saved =
    random_samples_.has_storage() ? c10::optional<Storage>(random_samples_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> random_samples__impl_saved;
  if (random_samples_.defined()) random_samples__impl_saved = random_samples_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::fractional_max_pool2d_outf(ks & c10::after_autograd_keyset, self_, kernel_size, output_size, random_samples_, output_, indices_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (random_samples__storage_saved.has_value())
    AT_ASSERT(random_samples__storage_saved.value().is_alias_of(random_samples_.storage()));
  if (random_samples__impl_saved) AT_ASSERT(random_samples__impl_saved == random_samples_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  return std::forward_as_tuple(output, indices);
}
Tensor & gcd_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("gcd");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("gcd");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::gcd_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor hardshrink_backward(c10::DispatchKeySet ks, const Tensor & grad_out, const Tensor & self, const Scalar & lambd) {
  auto& grad_out_ = unpack(grad_out, "grad_out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_out, self );
  (void)_any_requires_grad;
  std::shared_ptr<HardshrinkBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HardshrinkBackwardBackward>(new HardshrinkBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_out, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->lambd = lambd;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_out__storage_saved =
    grad_out_.has_storage() ? c10::optional<Storage>(grad_out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_out__impl_saved;
  if (grad_out_.defined()) grad_out__impl_saved = grad_out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::hardshrink_backward(ks & c10::after_autograd_keyset, grad_out_, self_, lambd);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_out__storage_saved.has_value())
    AT_ASSERT(grad_out__storage_saved.value().is_alias_of(grad_out_.storage()));
  if (grad_out__impl_saved) AT_ASSERT(grad_out__impl_saved == grad_out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "hardshrink_backward");
  return result;
}
Tensor & hardtanh_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & min_val, const Scalar & max_val, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("hardtanh");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("hardtanh");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::hardtanh_outf(ks & c10::after_autograd_keyset, self_, min_val, max_val, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & im2col_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 5);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("im2col");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("im2col");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::im2col_outf(ks & c10::after_autograd_keyset, self_, kernel_size, dilation, padding, stride, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor index_Tensor(c10::DispatchKeySet ks, const Tensor & self, const c10::List<c10::optional<Tensor>> & indices) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<IndexBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<IndexBackward>(new IndexBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_options = self.options();
    grad_fn->indices_ = make_saved_variable_list(indices);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  std::vector<c10::optional<Storage>> indices_storage_saved(indices.size());
  for (const c10::optional<Tensor>& tensor : indices)
    indices_storage_saved.push_back(
      tensor.has_value() && tensor->has_storage() ? c10::optional<Storage>(tensor->storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> indices_impl_saved(indices.size());
  for (size_t i=0; i<indices.size(); i++) {
    c10::optional<Tensor> t = indices[i];
    if (t.has_value() && t->defined()) indices_impl_saved[i] = t->getIntrusivePtr();
  }
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::index(ks & c10::after_autograd_keyset, self_, indices);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  for (size_t i=0; i<indices.size(); i++) {
    if (indices_storage_saved[i].has_value())
      AT_ASSERT(indices_storage_saved[i].value().is_alias_of(
          static_cast<c10::optional<Tensor>>(indices[i])->storage()));
  }
  for (size_t i=0; i<indices.size(); i++) {
    if (indices_impl_saved[i])
      AT_ASSERT(indices_impl_saved[i] == static_cast<c10::optional<Tensor>>(indices[i])->getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "index");
  return result;
}
Tensor & index_put_(c10::DispatchKeySet ks, Tensor & self, const c10::List<c10::optional<Tensor>> & indices, const Tensor & values, bool accumulate) {
  auto& self_ = unpack(self, "self", 0);
  auto& values_ = unpack(values, "values", 2);
  auto _any_requires_grad = compute_requires_grad( self, values );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<IndexPutBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<IndexPutBackward>(new IndexPutBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, values ));
    grad_fn->indices_ = make_saved_variable_list(indices);
    grad_fn->values_info = values;
    grad_fn->accumulate = accumulate;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  std::vector<c10::optional<Storage>> indices_storage_saved(indices.size());
  for (const c10::optional<Tensor>& tensor : indices)
    indices_storage_saved.push_back(
      tensor.has_value() && tensor->has_storage() ? c10::optional<Storage>(tensor->storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> indices_impl_saved(indices.size());
  for (size_t i=0; i<indices.size(); i++) {
    c10::optional<Tensor> t = indices[i];
    if (t.has_value() && t->defined()) indices_impl_saved[i] = t->getIntrusivePtr();
  }
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::index_put_(ks & c10::after_autograd_keyset, self_, indices, values_, accumulate);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  for (size_t i=0; i<indices.size(); i++) {
    if (indices_storage_saved[i].has_value())
      AT_ASSERT(indices_storage_saved[i].value().is_alias_of(
          static_cast<c10::optional<Tensor>>(indices[i])->storage()));
  }
  for (size_t i=0; i<indices.size(); i++) {
    if (indices_impl_saved[i])
      AT_ASSERT(indices_impl_saved[i] == static_cast<c10::optional<Tensor>>(indices[i])->getIntrusivePtr());
  }
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor index_select(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, const Tensor & index) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<IndexSelectBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<IndexSelectBackward>(new IndexSelectBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::index_select(ks & c10::after_autograd_keyset, self_, dim, index_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
bool is_coalesced(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::is_coalesced(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = _tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor l1_loss(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self, target );
  (void)_any_requires_grad;
  std::shared_ptr<L1LossBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<L1LossBackward>(new L1LossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::l1_loss(ks & c10::after_autograd_keyset, self_, target_, reduction);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "l1_loss");
  return result;
}
Tensor & l1_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& grad_input_ = unpack(grad_input, "grad_input", 4);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, target );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("l1_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("l1_loss_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::l1_loss_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, target_, reduction, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  return grad_input;
}
Tensor & linalg_cholesky_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("linalg_cholesky");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("linalg_cholesky");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::linalg_cholesky_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & linalg_eigvalsh_out_out(c10::DispatchKeySet ks, const Tensor & self, std::string UPLO, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::linalg_eigvalsh_outf(ks & c10::after_autograd_keyset, self_, UPLO, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  return out;
}
Tensor linalg_householder_product(c10::DispatchKeySet ks, const Tensor & input, const Tensor & tau) {
  auto& input_ = unpack(input, "input", 0);
  auto& tau_ = unpack(tau, "tau", 1);
  auto _any_requires_grad = compute_requires_grad( input, tau );
  (void)_any_requires_grad;
  std::shared_ptr<LinalgHouseholderProductBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LinalgHouseholderProductBackward>(new LinalgHouseholderProductBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, tau ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->tau_ = SavedVariable(tau, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> tau__storage_saved =
    tau_.has_storage() ? c10::optional<Storage>(tau_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tau__impl_saved;
  if (tau_.defined()) tau__impl_saved = tau_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::linalg_householder_product(ks & c10::after_autograd_keyset, input_, tau_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (tau__storage_saved.has_value())
    AT_ASSERT(tau__storage_saved.value().is_alias_of(tau_.storage()));
  if (tau__impl_saved) AT_ASSERT(tau__impl_saved == tau_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & linspace_out_out(c10::DispatchKeySet ks, const Scalar & start, const Scalar & end, c10::optional<int64_t> steps, Tensor & out) {
  auto& out_ = unpack(out, "out", 3);
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::linspace_outf(ks & c10::after_autograd_keyset, start, end, steps, out_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  return out;
}
Tensor & log2_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log2");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("log2");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::log2_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & log_normal_(c10::DispatchKeySet ks, Tensor & self, double mean, double std, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<LogNormalBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogNormalBackward>(new LogNormalBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::log_normal_(ks & c10::after_autograd_keyset, self_, mean, std, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & log_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("log");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::log_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor log_sigmoid_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& buffer_ = unpack(buffer, "buffer", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  check_no_requires_grad(buffer, "buffer");
  std::shared_ptr<LogSigmoidBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogSigmoidBackwardBackward>(new LogSigmoidBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->buffer_ = SavedVariable(buffer, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> buffer__storage_saved =
    buffer_.has_storage() ? c10::optional<Storage>(buffer_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> buffer__impl_saved;
  if (buffer_.defined()) buffer__impl_saved = buffer_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::log_sigmoid_backward(ks & c10::after_autograd_keyset, grad_output_, self_, buffer_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (buffer__storage_saved.has_value())
    AT_ASSERT(buffer__storage_saved.value().is_alias_of(buffer_.storage()));
  if (buffer__impl_saved) AT_ASSERT(buffer__impl_saved == buffer_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "log_sigmoid_backward");
  return result;
}
std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, Tensor & output, Tensor & buffer) {
  auto& self_ = unpack(self, "self", 0);
  auto& output_ = unpack(output, "output", 1);
  auto& buffer_ = unpack(buffer, "buffer", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log_sigmoid_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("log_sigmoid_forward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> buffer__storage_saved =
    buffer_.has_storage() ? c10::optional<Storage>(buffer_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> buffer__impl_saved;
  if (buffer_.defined()) buffer__impl_saved = buffer_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::log_sigmoid_forward_outf(ks & c10::after_autograd_keyset, self_, output_, buffer_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (buffer__storage_saved.has_value())
    AT_ASSERT(buffer__storage_saved.value().is_alias_of(buffer_.storage()));
  if (buffer__impl_saved) AT_ASSERT(buffer__impl_saved == buffer_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  return std::forward_as_tuple(output, buffer);
}
Tensor & logaddexp2_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("logaddexp2");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("logaddexp2");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::logaddexp2_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & logaddexp_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("logaddexp");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("logaddexp");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::logaddexp_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor logcumsumexp(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LogcumsumexpBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogcumsumexpBackward>(new LogcumsumexpBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::logcumsumexp(ks & c10::after_autograd_keyset, self_, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "logcumsumexp");
  return result;
}
Tensor logsumexp(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LogsumexpBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogsumexpBackward>(new LogsumexpBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim.vec();
    grad_fn->keepdim = keepdim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::logsumexp(ks & c10::after_autograd_keyset, self_, dim, keepdim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "logsumexp");
  return result;
}
std::tuple<Tensor &,Tensor &> lstsq_out_X(c10::DispatchKeySet ks, const Tensor & self, const Tensor & A, Tensor & X, Tensor & qr) {
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  auto& X_ = unpack(X, "X", 2);
  auto& qr_ = unpack(qr, "qr", 3);
  auto _any_requires_grad = compute_requires_grad( self, A );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, A )) {
    throw_error_out_requires_grad("lstsq");
  }
  if (compute_requires_grad( X, qr )) {
    throw_error_out_requires_grad("lstsq");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  c10::optional<Storage> X__storage_saved =
    X_.has_storage() ? c10::optional<Storage>(X_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> X__impl_saved;
  if (X_.defined()) X__impl_saved = X_.getIntrusivePtr();
  c10::optional<Storage> qr__storage_saved =
    qr_.has_storage() ? c10::optional<Storage>(qr_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> qr__impl_saved;
  if (qr_.defined()) qr__impl_saved = qr_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::lstsq_outf(ks & c10::after_autograd_keyset, self_, A_, X_, qr_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  if (X__storage_saved.has_value())
    AT_ASSERT(X__storage_saved.value().is_alias_of(X_.storage()));
  if (X__impl_saved) AT_ASSERT(X__impl_saved == X_.getIntrusivePtr());
  if (qr__storage_saved.has_value())
    AT_ASSERT(qr__storage_saved.value().is_alias_of(qr_.storage()));
  if (qr__impl_saved) AT_ASSERT(qr__impl_saved == qr_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( X, qr ), grad_fn);
  }
  return std::forward_as_tuple(X, qr);
}
std::tuple<Tensor,Tensor> max_dim(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MaxBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaxBackward0>(new MaxBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor values;
  Tensor indices;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::max(ks & c10::after_autograd_keyset, self_, dim, keepdim);
  })();
  std::tie(values, indices) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  throw_error_for_complex_autograd(values, "max");
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor max(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MaxBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaxBackward1>(new MaxBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::max(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "max");
  return result;
}
std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, Tensor & out, Tensor & indices) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 6);
  auto& indices_ = unpack(indices, "indices", 7);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_pool2d_with_indices");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("max_pool2d_with_indices");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::max_pool2d_with_indices_outf(ks & c10::after_autograd_keyset, self_, kernel_size, stride, padding, dilation, ceil_mode, out_, indices_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return std::forward_as_tuple(out, indices);
}
Tensor max_unpool2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<MaxUnpool2DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaxUnpool2DBackwardBackward>(new MaxUnpool2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->output_size = output_size.vec();
    grad_fn->self_info = self;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::max_unpool2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, indices_, output_size);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "max_unpool2d_backward");
  return result;
}
Tensor max_unpool3d(c10::DispatchKeySet ks, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MaxUnpool3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaxUnpool3DBackward>(new MaxUnpool3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->output_size = output_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::max_unpool3d(ks & c10::after_autograd_keyset, self_, indices_, output_size, stride, padding);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "max_unpool3d");
  return result;
}
Tensor & max_unpool3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  auto& grad_input_ = unpack(grad_input, "grad_input", 6);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, indices );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("max_unpool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_unpool3d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::max_unpool3d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, indices_, output_size, stride, padding, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  return grad_input;
}
Tensor maximum(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<MaximumBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaximumBackward>(new MaximumBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::maximum(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "maximum");
  return result;
}
Tensor & mean_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("mean");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("mean");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::mean_outf(ks & c10::after_autograd_keyset, self_, dim, keepdim, dtype, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
std::tuple<Tensor &,Tensor &> median_out_dim_values(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
  auto& self_ = unpack(self, "self", 0);
  auto& values_ = unpack(values, "values", 3);
  auto& indices_ = unpack(indices, "indices", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("median");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("median");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::median_outf(ks & c10::after_autograd_keyset, self_, dim, keepdim, values_, indices_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values ), grad_fn);
  }
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm(c10::DispatchKeySet ks, const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( input, weight, bias );
  (void)_any_requires_grad;
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<MiopenBatchNormBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MiopenBatchNormBackward>(new MiopenBatchNormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->training = training;
    grad_fn->epsilon = epsilon;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::miopen_batch_norm(ks & c10::after_autograd_keyset, input_, weight_, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  throw_error_for_complex_autograd(result0, "miopen_batch_norm");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor,Tensor> miopen_convolution_transpose_backward(c10::DispatchKeySet ks, const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( self, grad_output, weight );
  (void)_any_requires_grad;
  std::shared_ptr<MiopenConvolutionTransposeBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MiopenConvolutionTransposeBackwardBackward>(new MiopenConvolutionTransposeBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::miopen_convolution_transpose_backward(ks & c10::after_autograd_keyset, self_, grad_output_, weight_, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "miopen_convolution_transpose_backward");
  throw_error_for_complex_autograd(result1, "miopen_convolution_transpose_backward");
  throw_error_for_complex_autograd(result2, "miopen_convolution_transpose_backward");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor miopen_convolution_transpose_backward_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, weight );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_transpose_backward_input"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, weight ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::miopen_convolution_transpose_backward_input(ks & c10::after_autograd_keyset, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "miopen_convolution_transpose_backward_input");
  return result;
}
Tensor miopen_depthwise_convolution_backward_weight(c10::DispatchKeySet ks, IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_depthwise_convolution_backward_weight"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::miopen_depthwise_convolution_backward_weight(ks & c10::after_autograd_keyset, weight_size, grad_output_, self_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "miopen_depthwise_convolution_backward_weight");
  return result;
}
Tensor mkldnn_adaptive_avg_pool2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_adaptive_avg_pool2d_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::mkldnn_adaptive_avg_pool2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "mkldnn_adaptive_avg_pool2d_backward");
  return result;
}
std::tuple<Tensor,Tensor,Tensor> mkldnn_convolution_backward(c10::DispatchKeySet ks, const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( self, grad_output, weight );
  (void)_any_requires_grad;
  std::shared_ptr<MkldnnConvolutionBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MkldnnConvolutionBackwardBackward>(new MkldnnConvolutionBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::mkldnn_convolution_backward(ks & c10::after_autograd_keyset, self_, grad_output_, weight_, padding, stride, dilation, groups, output_mask);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "mkldnn_convolution_backward");
  throw_error_for_complex_autograd(result1, "mkldnn_convolution_backward");
  throw_error_for_complex_autograd(result2, "mkldnn_convolution_backward");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor mkldnn_reorder_conv3d_weight(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_reorder_conv3d_weight"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::mkldnn_reorder_conv3d_weight(ks & c10::after_autograd_keyset, self_, padding, stride, dilation, groups);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "mkldnn_reorder_conv3d_weight");
  return result;
}
std::tuple<Tensor,Tensor> mode(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ModeBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ModeBackward>(new ModeBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor values;
  Tensor indices;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::mode(ks & c10::after_autograd_keyset, self_, dim, keepdim);
  })();
  std::tie(values, indices) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  throw_error_for_complex_autograd(values, "mode");
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor & multi_margin_loss_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const Scalar & p, const Scalar & margin, const c10::optional<Tensor> & weight, int64_t reduction, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto& out_ = unpack(out, "out", 6);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, weight )) {
    throw_error_out_requires_grad("multi_margin_loss");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("multi_margin_loss");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::multi_margin_loss_outf(ks & c10::after_autograd_keyset, self_, target_, p, margin, weight, reduction, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
std::tuple<Tensor,Tensor> multilabel_margin_loss_forward(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MultilabelMarginLossBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MultilabelMarginLossBackward>(new MultilabelMarginLossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  Tensor output;
  Tensor is_target;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::multilabel_margin_loss_forward(ks & c10::after_autograd_keyset, self_, target_, reduction);
  })();
  std::tie(output, is_target) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->is_target_ = SavedVariable(is_target, true);
  }
  throw_error_for_complex_autograd(output, "multilabel_margin_loss_forward");
  return std::make_tuple(std::move(output), std::move(is_target));
}
Tensor & mv_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & vec, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& vec_ = unpack(vec, "vec", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, vec );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, vec )) {
    throw_error_out_requires_grad("mv");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("mv");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> vec__storage_saved =
    vec_.has_storage() ? c10::optional<Storage>(vec_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec__impl_saved;
  if (vec_.defined()) vec__impl_saved = vec_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::mv_outf(ks & c10::after_autograd_keyset, self_, vec_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec__storage_saved.has_value())
    AT_ASSERT(vec__storage_saved.value().is_alias_of(vec_.storage()));
  if (vec__impl_saved) AT_ASSERT(vec__impl_saved == vec_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
std::tuple<Tensor,Tensor,Tensor> native_batch_norm_backward(c10::DispatchKeySet ks, const Tensor & grad_out, const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, const c10::optional<Tensor> & save_mean, const c10::optional<Tensor> & save_invstd, bool train, double eps, std::array<bool,3> output_mask) {
  auto& grad_out_ = unpack(grad_out, "grad_out", 0);
  auto& input_ = unpack(input, "input", 1);
  auto _any_requires_grad = compute_requires_grad( grad_out, input, weight, save_mean, save_invstd );
  (void)_any_requires_grad;
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<NativeBatchNormBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NativeBatchNormBackwardBackward>(new NativeBatchNormBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_out, input, weight, save_mean, save_invstd ));
    grad_fn->grad_out_ = SavedVariable(grad_out, false);
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->save_mean_ = SavedVariable(save_mean, false);
    grad_fn->save_invstd_ = SavedVariable(save_invstd, false);
    grad_fn->train = train;
    grad_fn->eps = eps;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #ifndef NDEBUG
  c10::optional<Storage> grad_out__storage_saved =
    grad_out_.has_storage() ? c10::optional<Storage>(grad_out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_out__impl_saved;
  if (grad_out_.defined()) grad_out__impl_saved = grad_out_.getIntrusivePtr();
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::native_batch_norm_backward(ks & c10::after_autograd_keyset, grad_out_, input_, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_out__storage_saved.has_value())
    AT_ASSERT(grad_out__storage_saved.value().is_alias_of(grad_out_.storage()));
  if (grad_out__impl_saved) AT_ASSERT(grad_out__impl_saved == grad_out_.getIntrusivePtr());
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "native_batch_norm_backward");
  throw_error_for_complex_autograd(result1, "native_batch_norm_backward");
  throw_error_for_complex_autograd(result2, "native_batch_norm_backward");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor native_norm(c10::DispatchKeySet ks, const Tensor & self, const Scalar & p) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("native_norm"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::native_norm(ks & c10::after_autograd_keyset, self_, p);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "native_norm");
  return result;
}
Tensor native_norm_ScalarOpt_dim_dtype(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("native_norm"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::native_norm(ks & c10::after_autograd_keyset, self_, p, dim, keepdim, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "native_norm");
  return result;
}
Tensor ne_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::ne(ks & c10::after_autograd_keyset, self_, other);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor ne_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::ne(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  return result;
}
Tensor & ne__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NeBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NeBackward0>(new NeBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::ne_(ks & c10::after_autograd_keyset, self_, other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & ne__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NeBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NeBackward1>(new NeBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::ne_(ks & c10::after_autograd_keyset, self_, other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & nextafter_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("nextafter");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("nextafter");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::nextafter_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor nll_loss2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& total_weight_ = unpack(total_weight, "total_weight", 6);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  check_no_requires_grad(weight, "weight");
  check_no_requires_grad(total_weight, "total_weight");
  std::shared_ptr<NllLoss2DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NllLoss2DBackwardBackward>(new NllLoss2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
    grad_fn->ignore_index = ignore_index;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> total_weight__storage_saved =
    total_weight_.has_storage() ? c10::optional<Storage>(total_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> total_weight__impl_saved;
  if (total_weight_.defined()) total_weight__impl_saved = total_weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::nll_loss2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, target_, weight, reduction, ignore_index, total_weight_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (total_weight__storage_saved.has_value())
    AT_ASSERT(total_weight__storage_saved.value().is_alias_of(total_weight_.storage()));
  if (total_weight__impl_saved) AT_ASSERT(total_weight__impl_saved == total_weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "nll_loss2d_backward");
  return result;
}
std::tuple<Tensor &,Tensor &> nll_loss2d_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto& output_ = unpack(output, "output", 5);
  auto& total_weight_ = unpack(total_weight, "total_weight", 6);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, weight )) {
    throw_error_out_requires_grad("nll_loss2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("nll_loss2d_forward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> total_weight__storage_saved =
    total_weight_.has_storage() ? c10::optional<Storage>(total_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> total_weight__impl_saved;
  if (total_weight_.defined()) total_weight__impl_saved = total_weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::nll_loss2d_forward_outf(ks & c10::after_autograd_keyset, self_, target_, weight, reduction, ignore_index, output_, total_weight_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (total_weight__storage_saved.has_value())
    AT_ASSERT(total_weight__storage_saved.value().is_alias_of(total_weight_.storage()));
  if (total_weight__impl_saved) AT_ASSERT(total_weight__impl_saved == total_weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  return std::forward_as_tuple(output, total_weight);
}
Tensor nll_loss_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& total_weight_ = unpack(total_weight, "total_weight", 6);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  check_no_requires_grad(weight, "weight");
  check_no_requires_grad(total_weight, "total_weight");
  std::shared_ptr<NllLossBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NllLossBackwardBackward>(new NllLossBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
    grad_fn->ignore_index = ignore_index;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> total_weight__storage_saved =
    total_weight_.has_storage() ? c10::optional<Storage>(total_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> total_weight__impl_saved;
  if (total_weight_.defined()) total_weight__impl_saved = total_weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::nll_loss_backward(ks & c10::after_autograd_keyset, grad_output_, self_, target_, weight, reduction, ignore_index, total_weight_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (total_weight__storage_saved.has_value())
    AT_ASSERT(total_weight__storage_saved.value().is_alias_of(total_weight_.storage()));
  if (total_weight__impl_saved) AT_ASSERT(total_weight__impl_saved == total_weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "nll_loss_backward");
  return result;
}
std::tuple<Tensor &,Tensor &> nll_loss_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto& output_ = unpack(output, "output", 5);
  auto& total_weight_ = unpack(total_weight, "total_weight", 6);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, weight )) {
    throw_error_out_requires_grad("nll_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("nll_loss_forward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> total_weight__storage_saved =
    total_weight_.has_storage() ? c10::optional<Storage>(total_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> total_weight__impl_saved;
  if (total_weight_.defined()) total_weight__impl_saved = total_weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::nll_loss_forward_outf(ks & c10::after_autograd_keyset, self_, target_, weight, reduction, ignore_index, output_, total_weight_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (total_weight__storage_saved.has_value())
    AT_ASSERT(total_weight__storage_saved.value().is_alias_of(total_weight_.storage()));
  if (total_weight__impl_saved) AT_ASSERT(total_weight__impl_saved == total_weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  return std::forward_as_tuple(output, total_weight);
}
Tensor ormqr(c10::DispatchKeySet ks, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  auto& input3_ = unpack(input3, "input3", 2);
  auto _any_requires_grad = compute_requires_grad( self, input2, input3 );
  (void)_any_requires_grad;
  std::shared_ptr<OrmqrBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<OrmqrBackward>(new OrmqrBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, input2, input3 ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> input2__storage_saved =
    input2_.has_storage() ? c10::optional<Storage>(input2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input2__impl_saved;
  if (input2_.defined()) input2__impl_saved = input2_.getIntrusivePtr();
  c10::optional<Storage> input3__storage_saved =
    input3_.has_storage() ? c10::optional<Storage>(input3_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input3__impl_saved;
  if (input3_.defined()) input3__impl_saved = input3_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::ormqr(ks & c10::after_autograd_keyset, self_, input2_, input3_, left, transpose);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (input2__storage_saved.has_value())
    AT_ASSERT(input2__storage_saved.value().is_alias_of(input2_.storage()));
  if (input2__impl_saved) AT_ASSERT(input2__impl_saved == input2_.getIntrusivePtr());
  if (input3__storage_saved.has_value())
    AT_ASSERT(input3__storage_saved.value().is_alias_of(input3_.storage()));
  if (input3__impl_saved) AT_ASSERT(input3__impl_saved == input3_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "ormqr");
  return result;
}
Tensor & polar_out_out(c10::DispatchKeySet ks, const Tensor & abs, const Tensor & angle, Tensor & out) {
  auto& abs_ = unpack(abs, "abs", 0);
  auto& angle_ = unpack(angle, "angle", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( abs, angle );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( abs, angle )) {
    throw_error_out_requires_grad("polar");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("polar");
  }
  #ifndef NDEBUG
  c10::optional<Storage> abs__storage_saved =
    abs_.has_storage() ? c10::optional<Storage>(abs_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> abs__impl_saved;
  if (abs_.defined()) abs__impl_saved = abs_.getIntrusivePtr();
  c10::optional<Storage> angle__storage_saved =
    angle_.has_storage() ? c10::optional<Storage>(angle_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> angle__impl_saved;
  if (angle_.defined()) angle__impl_saved = angle_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::polar_outf(ks & c10::after_autograd_keyset, abs_, angle_, out_);
  }
  #ifndef NDEBUG
  if (abs__storage_saved.has_value())
    AT_ASSERT(abs__storage_saved.value().is_alias_of(abs_.storage()));
  if (abs__impl_saved) AT_ASSERT(abs__impl_saved == abs_.getIntrusivePtr());
  if (angle__storage_saved.has_value())
    AT_ASSERT(angle__storage_saved.value().is_alias_of(angle_.storage()));
  if (angle__impl_saved) AT_ASSERT(angle__impl_saved == angle_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & polygamma_out_out(c10::DispatchKeySet ks, int64_t n, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("polygamma");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("polygamma");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::polygamma_outf(ks & c10::after_autograd_keyset, n, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & pow_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & exponent, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& exponent_ = unpack(exponent, "exponent", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, exponent );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, exponent )) {
    throw_error_out_requires_grad("pow");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("pow");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> exponent__storage_saved =
    exponent_.has_storage() ? c10::optional<Storage>(exponent_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> exponent__impl_saved;
  if (exponent_.defined()) exponent__impl_saved = exponent_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::pow_outf(ks & c10::after_autograd_keyset, self_, exponent_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (exponent__storage_saved.has_value())
    AT_ASSERT(exponent__storage_saved.value().is_alias_of(exponent_.storage()));
  if (exponent__impl_saved) AT_ASSERT(exponent__impl_saved == exponent_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & pow_out_Scalar_out(c10::DispatchKeySet ks, const Scalar & self, const Tensor & exponent, Tensor & out) {
  auto& exponent_ = unpack(exponent, "exponent", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( exponent );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( exponent )) {
    throw_error_out_requires_grad("pow");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("pow");
  }
  #ifndef NDEBUG
  c10::optional<Storage> exponent__storage_saved =
    exponent_.has_storage() ? c10::optional<Storage>(exponent_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> exponent__impl_saved;
  if (exponent_.defined()) exponent__impl_saved = exponent_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::pow_outf(ks & c10::after_autograd_keyset, self, exponent_, out_);
  }
  #ifndef NDEBUG
  if (exponent__storage_saved.has_value())
    AT_ASSERT(exponent__storage_saved.value().is_alias_of(exponent_.storage()));
  if (exponent__impl_saved) AT_ASSERT(exponent__impl_saved == exponent_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & pow_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & exponent, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("pow");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("pow");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::pow_outf(ks & c10::after_autograd_keyset, self_, exponent, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & prod_out_int_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("prod");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("prod");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::prod_outf(ks & c10::after_autograd_keyset, self_, dim, keepdim, dtype, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
int64_t q_per_channel_axis(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::q_per_channel_axis(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = _tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor q_per_channel_zero_points(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("q_per_channel_zero_points"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::q_per_channel_zero_points(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "q_per_channel_zero_points");
  return result;
}
Tensor & random__from(c10::DispatchKeySet ks, Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<RandomBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RandomBackward0>(new RandomBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::random_(ks & c10::after_autograd_keyset, self_, from, to, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & random__to(c10::DispatchKeySet ks, Tensor & self, int64_t to, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<RandomBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RandomBackward1>(new RandomBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::random_(ks & c10::after_autograd_keyset, self_, to, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & random_(c10::DispatchKeySet ks, Tensor & self, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<RandomBackward2> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RandomBackward2>(new RandomBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::random_(ks & c10::after_autograd_keyset, self_, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & randperm_out_generator_out(c10::DispatchKeySet ks, int64_t n, c10::optional<Generator> generator, Tensor & out) {
  auto& out_ = unpack(out, "out", 2);
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::randperm_outf(ks & c10::after_autograd_keyset, n, generator, out_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  return out;
}
void record_stream(c10::DispatchKeySet ks, Tensor & self, Stream s) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::record_stream(ks & c10::after_autograd_keyset, self_, s);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
}
Tensor reflection_pad1d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<ReflectionPad1DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ReflectionPad1DBackwardBackward>(new ReflectionPad1DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding.vec();
    grad_fn->self_info = self;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::reflection_pad1d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, padding);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor reflection_pad2d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ReflectionPad2DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ReflectionPad2DBackward>(new ReflectionPad2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::reflection_pad2d(ks & c10::after_autograd_keyset, self_, padding);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & reflection_pad2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("reflection_pad2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("reflection_pad2d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::reflection_pad2d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, padding, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  return grad_input;
}
Tensor remainder_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<RemainderBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RemainderBackward0>(new RemainderBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::remainder(ks & c10::after_autograd_keyset, self_, other);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "remainder");
  return result;
}
Tensor remainder_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_no_requires_grad(other, "other");
  std::shared_ptr<RemainderBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RemainderBackward1>(new RemainderBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::remainder(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "remainder");
  return result;
}
Tensor & remainder__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<RemainderBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RemainderBackward0>(new RemainderBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::remainder_(ks & c10::after_autograd_keyset, self_, other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & remainder__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  check_no_requires_grad(other, "other");
  std::shared_ptr<RemainderBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RemainderBackward1>(new RemainderBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::remainder_(ks & c10::after_autograd_keyset, self_, other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor repeat(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef repeats) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<RepeatBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RepeatBackward>(new RepeatBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->repeats = repeats.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::repeat(ks & c10::after_autograd_keyset, self_, repeats);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor replication_pad1d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ReplicationPad1DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ReplicationPad1DBackward>(new ReplicationPad1DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::replication_pad1d(ks & c10::after_autograd_keyset, self_, padding);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & replication_pad1d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("replication_pad1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("replication_pad1d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::replication_pad1d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, padding, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  return grad_input;
}
Tensor & replication_pad2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("replication_pad2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("replication_pad2d");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::replication_pad2d_outf(ks & c10::after_autograd_keyset, self_, padding, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor roll(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef shifts, IntArrayRef dims) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<RollBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RollBackward>(new RollBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->shifts = shifts.vec();
    grad_fn->dims = dims.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::roll(ks & c10::after_autograd_keyset, self_, shifts, dims);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor rot90(c10::DispatchKeySet ks, const Tensor & self, int64_t k, IntArrayRef dims) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Rot90Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Rot90Backward>(new Rot90Backward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->k = k;
    grad_fn->dims = dims.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::rot90(ks & c10::after_autograd_keyset, self_, k, dims);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & round_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("round");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("round");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::round_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor rrelu_with_noise_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & noise, const Scalar & lower, const Scalar & upper, bool training, bool self_is_result) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& noise_ = unpack(noise, "noise", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  check_no_requires_grad(noise, "noise");
  std::shared_ptr<RreluWithNoiseBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RreluWithNoiseBackwardBackward>(new RreluWithNoiseBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->noise_ = SavedVariable(noise, false);
    grad_fn->lower = lower;
    grad_fn->upper = upper;
    grad_fn->training = training;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> noise__storage_saved =
    noise_.has_storage() ? c10::optional<Storage>(noise_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> noise__impl_saved;
  if (noise_.defined()) noise__impl_saved = noise_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::rrelu_with_noise_backward(ks & c10::after_autograd_keyset, grad_output_, self_, noise_, lower, upper, training, self_is_result);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (noise__storage_saved.has_value())
    AT_ASSERT(noise__storage_saved.value().is_alias_of(noise_.storage()));
  if (noise__impl_saved) AT_ASSERT(noise__impl_saved == noise_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "rrelu_with_noise_backward");
  return result;
}
Tensor & rsqrt_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("rsqrt");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("rsqrt");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::rsqrt_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & searchsorted_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right, Tensor & out) {
  auto& sorted_sequence_ = unpack(sorted_sequence, "sorted_sequence", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& out_ = unpack(out, "out", 4);
  #ifndef NDEBUG
  c10::optional<Storage> sorted_sequence__storage_saved =
    sorted_sequence_.has_storage() ? c10::optional<Storage>(sorted_sequence_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> sorted_sequence__impl_saved;
  if (sorted_sequence_.defined()) sorted_sequence__impl_saved = sorted_sequence_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::searchsorted_outf(ks & c10::after_autograd_keyset, sorted_sequence_, self_, out_int32, right, out_);
  }
  #ifndef NDEBUG
  if (sorted_sequence__storage_saved.has_value())
    AT_ASSERT(sorted_sequence__storage_saved.value().is_alias_of(sorted_sequence_.storage()));
  if (sorted_sequence__impl_saved) AT_ASSERT(sorted_sequence__impl_saved == sorted_sequence_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  return out;
}
Tensor & signbit_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("signbit");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("signbit");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::signbit_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor silu_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("silu_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::silu_backward(ks & c10::after_autograd_keyset, grad_output_, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "silu_backward");
  return result;
}
Tensor & slow_conv_transpose2d_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& out_ = unpack(out, "out", 8);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("slow_conv_transpose2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("slow_conv_transpose2d");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::slow_conv_transpose2d_outf(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding, output_padding, dilation, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & smooth_l1_loss_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction, double beta, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self, target );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("smooth_l1_loss");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("smooth_l1_loss");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::smooth_l1_loss_outf(ks & c10::after_autograd_keyset, self_, target_, reduction, beta, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor soft_margin_loss(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_no_requires_grad(target, "target");
  std::shared_ptr<SoftMarginLossBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SoftMarginLossBackward>(new SoftMarginLossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::soft_margin_loss(ks & c10::after_autograd_keyset, self_, target_, reduction);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "soft_margin_loss");
  return result;
}
Tensor & soft_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& grad_input_ = unpack(grad_input, "grad_input", 4);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("soft_margin_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("soft_margin_loss_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::soft_margin_loss_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, target_, reduction, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  return grad_input;
}
Tensor softplus(c10::DispatchKeySet ks, const Tensor & self, const Scalar & beta, const Scalar & threshold) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SoftplusBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SoftplusBackward>(new SoftplusBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->beta = beta;
    grad_fn->threshold = threshold;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::softplus(ks & c10::after_autograd_keyset, self_, beta, threshold);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "softplus");
  return result;
}
Tensor & softplus_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Scalar & beta, const Scalar & threshold, const Tensor & output, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& output_ = unpack(output, "output", 4);
  auto& grad_input_ = unpack(grad_input, "grad_input", 5);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, output )) {
    throw_error_out_requires_grad("softplus_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("softplus_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::softplus_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, beta, threshold, output_, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  return grad_input;
}
std::tuple<Tensor,Tensor> sort(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool descending) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SortBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SortBackward0>(new SortBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
  }
  Tensor values;
  Tensor indices;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::sort(ks & c10::after_autograd_keyset, self_, dim, descending);
  })();
  std::tie(values, indices) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  throw_error_for_complex_autograd(values, "sort");
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> sort_stable(c10::DispatchKeySet ks, const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SortBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SortBackward1>(new SortBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
  }
  Tensor values;
  Tensor indices;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::sort(ks & c10::after_autograd_keyset, self_, stable, dim, descending);
  })();
  std::tie(values, indices) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  throw_error_for_complex_autograd(values, "sort");
  return std::make_tuple(std::move(values), std::move(indices));
}
std::vector<Tensor> split_Tensor(c10::DispatchKeySet ks, const Tensor & self, int64_t split_size, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SplitBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SplitBackward>(new SplitBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_options = self.options();
    grad_fn->split_size = split_size;
    grad_fn->dim = dim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoNonVariableTypeMode guard;
    return at::redispatch::split(ks & c10::after_autograd_keyset, self_, split_size, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & square_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("square");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("square");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::square_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & sspaddmm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  auto& out_ = unpack(out, "out", 5);
  auto _any_requires_grad = compute_requires_grad( self, mat1, mat2 );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    throw_error_out_requires_grad("sspaddmm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sspaddmm");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat1__storage_saved =
    mat1_.has_storage() ? c10::optional<Storage>(mat1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat1__impl_saved;
  if (mat1_.defined()) mat1__impl_saved = mat1_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::sspaddmm_outf(ks & c10::after_autograd_keyset, self_, mat1_, mat2_, beta, alpha, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat1__storage_saved.has_value())
    AT_ASSERT(mat1__storage_saved.value().is_alias_of(mat1_.storage()));
  if (mat1__impl_saved) AT_ASSERT(mat1__impl_saved == mat1_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor std(c10::DispatchKeySet ks, const Tensor & self, bool unbiased) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<StdBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<StdBackward0>(new StdBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->unbiased = unbiased;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::std(ks & c10::after_autograd_keyset, self_, unbiased);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "std");
  return result;
}
Tensor std_dim(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<StdBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<StdBackward1>(new StdBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim.vec();
    grad_fn->unbiased = unbiased;
    grad_fn->keepdim = keepdim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::std(ks & c10::after_autograd_keyset, self_, dim, unbiased, keepdim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "std");
  return result;
}
Tensor sum(c10::DispatchKeySet ks, const Tensor & self, c10::optional<ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SumBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SumBackward0>(new SumBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::sum(ks & c10::after_autograd_keyset, self_, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor sum_dim_IntList(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SumBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SumBackward1>(new SumBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim.vec();
    grad_fn->keepdim = keepdim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::sum(ks & c10::after_autograd_keyset, self_, dim, keepdim, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor t(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<TBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TBackward>(new TBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoNonVariableTypeMode guard;
    return at::redispatch::t(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & t_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<TBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TBackward>(new TBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::t_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor take(c10::DispatchKeySet ks, const Tensor & self, const Tensor & index) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<TakeBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TakeBackward>(new TakeBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->index_ = SavedVariable(index, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::take(ks & c10::after_autograd_keyset, self_, index_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & tanh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("tanh");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("tanh");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::tanh_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor & grad_input, Tensor & grad_weight) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& grad_input_ = unpack(grad_input, "grad_input", 7);
  auto& grad_weight_ = unpack(grad_weight, "grad_weight", 8);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_weight__storage_saved =
    grad_weight_.has_storage() ? c10::optional<Storage>(grad_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_weight__impl_saved;
  if (grad_weight_.defined()) grad_weight__impl_saved = grad_weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::thnn_conv_depthwise2d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, dilation, grad_input_, grad_weight_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_weight__storage_saved.has_value())
    AT_ASSERT(grad_weight__storage_saved.value().is_alias_of(grad_weight_.storage()));
  if (grad_weight__impl_saved) AT_ASSERT(grad_weight__impl_saved == grad_weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input, grad_weight ), grad_fn);
  }
  return std::forward_as_tuple(grad_input, grad_weight);
}
Tensor to_mkldnn(c10::DispatchKeySet ks, const Tensor & self, c10::optional<ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ToMkldnnBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ToMkldnnBackward>(new ToMkldnnBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::to_mkldnn(ks & c10::after_autograd_keyset, self_, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "to_mkldnn");
  return result;
}
Tensor to_sparse_sparse_dim(c10::DispatchKeySet ks, const Tensor & self, int64_t sparse_dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("to_sparse"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::to_sparse(ks & c10::after_autograd_keyset, self_, sparse_dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "to_sparse");
  return result;
}
Tensor to_sparse(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ToSparseBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ToSparseBackward>(new ToSparseBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::to_sparse(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "to_sparse");
  return result;
}
std::tuple<Tensor,Tensor> topk(c10::DispatchKeySet ks, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<TopkBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TopkBackward>(new TopkBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
  }
  Tensor values;
  Tensor indices;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::topk(ks & c10::after_autograd_keyset, self_, k, dim, largest, sorted);
  })();
  std::tie(values, indices) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  throw_error_for_complex_autograd(values, "topk");
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor transpose_int(c10::DispatchKeySet ks, const Tensor & self, int64_t dim0, int64_t dim1) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<TransposeBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TransposeBackward0>(new TransposeBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim0 = dim0;
    grad_fn->dim1 = dim1;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoNonVariableTypeMode guard;
    return at::redispatch::transpose(ks & c10::after_autograd_keyset, self_, dim0, dim1);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & transpose_(c10::DispatchKeySet ks, Tensor & self, int64_t dim0, int64_t dim1) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<TransposeBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TransposeBackward1>(new TransposeBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim0 = dim0;
    grad_fn->dim1 = dim1;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::transpose_(ks & c10::after_autograd_keyset, self_, dim0, dim1);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & triu_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t diagonal, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("triu");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("triu");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::triu_outf(ks & c10::after_autograd_keyset, self_, diagonal, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor trunc(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<TruncBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TruncBackward>(new TruncBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::trunc(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "trunc");
  return result;
}
Tensor & trunc_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<TruncBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TruncBackward>(new TruncBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::trunc_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & upsample_bicubic2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 5);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_bicubic2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_bicubic2d");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::upsample_bicubic2d_outf(ks & c10::after_autograd_keyset, self_, output_size, align_corners, scales_h, scales_w, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor upsample_linear1d_backward_vec(c10::DispatchKeySet ks, const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleLinear1DBackwardBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleLinear1DBackwardBackward1>(new UpsampleLinear1DBackwardBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size;
    grad_fn->align_corners = align_corners;
    grad_fn->scale_factors = scale_factors;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::upsample_linear1d_backward(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, align_corners, scale_factors);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_linear1d_backward");
  return result;
}
Tensor upsample_linear1d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleLinear1DBackwardBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleLinear1DBackwardBackward0>(new UpsampleLinear1DBackwardBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
    grad_fn->scales = scales;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::upsample_linear1d_backward(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, align_corners, scales);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_linear1d_backward");
  return result;
}
Tensor upsample_nearest2d_backward_vec(c10::DispatchKeySet ks, const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, c10::optional<ArrayRef<double>> scale_factors) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleNearest2DBackwardBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleNearest2DBackwardBackward1>(new UpsampleNearest2DBackwardBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size;
    grad_fn->scale_factors = scale_factors;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::upsample_nearest2d_backward(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, scale_factors);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_nearest2d_backward");
  return result;
}
Tensor upsample_nearest2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleNearest2DBackwardBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleNearest2DBackwardBackward0>(new UpsampleNearest2DBackwardBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
    grad_fn->scales_h = scales_h;
    grad_fn->scales_w = scales_w;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::upsample_nearest2d_backward(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, scales_h, scales_w);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_nearest2d_backward");
  return result;
}
Tensor upsample_nearest3d_vec(c10::DispatchKeySet ks, const Tensor & input, c10::optional<IntArrayRef> output_size, c10::optional<ArrayRef<double>> scale_factors) {
  auto& input_ = unpack(input, "input", 0);
  auto _any_requires_grad = compute_requires_grad( input );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleNearest3DBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleNearest3DBackward1>(new UpsampleNearest3DBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input ));
    grad_fn->input_sizes = input.sizes().vec();
    grad_fn->output_size = output_size;
    grad_fn->scale_factors = scale_factors;
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::upsample_nearest3d(ks & c10::after_autograd_keyset, input_, output_size, scale_factors);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_nearest3d");
  return result;
}
Tensor upsample_nearest3d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleNearest3DBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleNearest3DBackward0>(new UpsampleNearest3DBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->scales_d = scales_d;
    grad_fn->scales_h = scales_h;
    grad_fn->scales_w = scales_w;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::upsample_nearest3d(ks & c10::after_autograd_keyset, self_, output_size, scales_d, scales_h, scales_w);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_nearest3d");
  return result;
}
Tensor & upsample_nearest3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& grad_input_ = unpack(grad_input, "grad_input", 6);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_nearest3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_nearest3d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::upsample_nearest3d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, scales_d, scales_h, scales_w, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  return grad_input;
}
Tensor & upsample_trilinear3d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 6);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_trilinear3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_trilinear3d");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::upsample_trilinear3d_outf(ks & c10::after_autograd_keyset, self_, output_size, align_corners, scales_d, scales_h, scales_w, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor values(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ValuesBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ValuesBackward>(new ValuesBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoNonVariableTypeMode guard;
    return at::redispatch::values(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "values");
  return result;
}
std::tuple<Tensor,Tensor> var_mean(c10::DispatchKeySet ks, const Tensor & self, bool unbiased) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<VarMeanBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<VarMeanBackward1>(new VarMeanBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->unbiased = unbiased;
  }
  Tensor result0;
  Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::var_mean(ks & c10::after_autograd_keyset, self_, unbiased);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  throw_error_for_complex_autograd(result0, "var_mean");
  throw_error_for_complex_autograd(result1, "var_mean");
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> var_mean_dim(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<VarMeanBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<VarMeanBackward0>(new VarMeanBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim.vec();
    grad_fn->unbiased = unbiased;
    grad_fn->keepdim = keepdim;
  }
  Tensor result0;
  Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::var_mean(ks & c10::after_autograd_keyset, self_, dim, unbiased, keepdim);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  throw_error_for_complex_autograd(result0, "var_mean");
  throw_error_for_complex_autograd(result1, "var_mean");
  return std::make_tuple(std::move(result0), std::move(result1));
}
}
}

namespace {

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  m.impl("__irshift__.Scalar",
         TORCH_FN(VariableType::__irshift___Scalar)
  );
  m.impl("__irshift__.Tensor",
         TORCH_FN(VariableType::__irshift___Tensor)
  );
  m.impl("__rshift__.Scalar",
         TORCH_FN(VariableType::__rshift___Scalar)
  );
  m.impl("__rshift__.Tensor",
         TORCH_FN(VariableType::__rshift___Tensor)
  );
  m.impl("_adaptive_avg_pool2d",
         TORCH_FN(VariableType::_adaptive_avg_pool2d)
  );
  m.impl("_add_relu.out",
         TORCH_FN(VariableType::_add_relu_out_out)
  );
  m.impl("_amp_update_scale",
         TORCH_FN(VariableType::_amp_update_scale)
  );
  m.impl("_bmm",
         TORCH_FN(VariableType::_bmm)
  );
  m.impl("_cat.out",
         TORCH_FN(VariableType::_cat_out_out)
  );
  m.impl("_cdist_backward",
         TORCH_FN(VariableType::_cdist_backward)
  );
  m.impl("_cholesky_helper",
         TORCH_FN(VariableType::_cholesky_helper)
  );
  m.impl("_coalesced_",
         TORCH_FN(VariableType::_coalesced_)
  );
  m.impl("_compute_linear_combination",
         TORCH_FN(VariableType::_compute_linear_combination)
  );
  m.impl("_conj",
         TORCH_FN(VariableType::_conj)
  );
  m.impl("_cudnn_rnn_backward",
         TORCH_FN(VariableType::_cudnn_rnn_backward)
  );
  m.impl("_cummax_helper",
         TORCH_FN(VariableType::_cummax_helper)
  );
  m.impl("_cumprod.out",
         TORCH_FN(VariableType::_cumprod_out_out)
  );
  m.impl("_cumsum",
         TORCH_FN(VariableType::_cumsum)
  );
  m.impl("_dirichlet_grad",
         TORCH_FN(VariableType::_dirichlet_grad)
  );
  m.impl("_foreach_ceil",
         TORCH_FN(VariableType::_foreach_ceil)
  );
  m.impl("_foreach_ceil_",
         TORCH_FN(VariableType::_foreach_ceil_)
  );
  m.impl("_foreach_div.Scalar",
         TORCH_FN(VariableType::_foreach_div_Scalar)
  );
  m.impl("_foreach_div.List",
         TORCH_FN(VariableType::_foreach_div_List)
  );
  m.impl("_foreach_div.ScalarList",
         TORCH_FN(VariableType::_foreach_div_ScalarList)
  );
  m.impl("_foreach_div_.Scalar",
         TORCH_FN(VariableType::_foreach_div__Scalar)
  );
  m.impl("_foreach_div_.List",
         TORCH_FN(VariableType::_foreach_div__List)
  );
  m.impl("_foreach_div_.ScalarList",
         TORCH_FN(VariableType::_foreach_div__ScalarList)
  );
  m.impl("_foreach_exp",
         TORCH_FN(VariableType::_foreach_exp)
  );
  m.impl("_foreach_exp_",
         TORCH_FN(VariableType::_foreach_exp_)
  );
  m.impl("_foreach_lgamma",
         TORCH_FN(VariableType::_foreach_lgamma)
  );
  m.impl("_foreach_lgamma_",
         TORCH_FN(VariableType::_foreach_lgamma_)
  );
  m.impl("_foreach_log1p",
         TORCH_FN(VariableType::_foreach_log1p)
  );
  m.impl("_foreach_log1p_",
         TORCH_FN(VariableType::_foreach_log1p_)
  );
  m.impl("_foreach_sigmoid",
         TORCH_FN(VariableType::_foreach_sigmoid)
  );
  m.impl("_foreach_sigmoid_",
         TORCH_FN(VariableType::_foreach_sigmoid_)
  );
  m.impl("_foreach_sqrt",
         TORCH_FN(VariableType::_foreach_sqrt)
  );
  m.impl("_foreach_sqrt_",
         TORCH_FN(VariableType::_foreach_sqrt_)
  );
  m.impl("_foreach_tan",
         TORCH_FN(VariableType::_foreach_tan)
  );
  m.impl("_foreach_tan_",
         TORCH_FN(VariableType::_foreach_tan_)
  );
  m.impl("_foreach_zero_",
         TORCH_FN(VariableType::_foreach_zero_)
  );
  m.impl("_fused_dropout",
         TORCH_FN(VariableType::_fused_dropout)
  );
  m.impl("_logcumsumexp",
         TORCH_FN(VariableType::_logcumsumexp)
  );
  m.impl("_make_per_tensor_quantized_tensor",
         TORCH_FN(VariableType::_make_per_tensor_quantized_tensor)
  );
  m.impl("_mode",
         TORCH_FN(VariableType::_mode)
  );
  m.impl("_nnpack_spatial_convolution",
         TORCH_FN(VariableType::_nnpack_spatial_convolution)
  );
  m.impl("_softmax_backward_data",
         TORCH_FN(VariableType::_softmax_backward_data)
  );
  m.impl("_sparse_log_softmax",
         TORCH_FN(VariableType::_sparse_log_softmax)
  );
  m.impl("_sparse_sum_backward",
         TORCH_FN(VariableType::_sparse_sum_backward)
  );
  m.impl("_std",
         TORCH_FN(VariableType::_std)
  );
  m.impl("_test_optional_floatlist",
         TORCH_FN(VariableType::_test_optional_floatlist)
  );
  m.impl("_thnn_fused_gru_cell_backward",
         TORCH_FN(VariableType::_thnn_fused_gru_cell_backward)
  );
  m.impl("_values",
         TORCH_FN(VariableType::_values)
  );
  m.impl("_weight_norm_cuda_interface_backward",
         TORCH_FN(VariableType::_weight_norm_cuda_interface_backward)
  );
  m.impl("acos.out",
         TORCH_FN(VariableType::acos_out_out)
  );
  m.impl("acosh",
         TORCH_FN(VariableType::acosh)
  );
  m.impl("acosh_",
         TORCH_FN(VariableType::acosh_)
  );
  m.impl("adaptive_avg_pool3d.out",
         TORCH_FN(VariableType::adaptive_avg_pool3d_out_out)
  );
  m.impl("adaptive_max_pool3d_backward",
         TORCH_FN(VariableType::adaptive_max_pool3d_backward)
  );
  m.impl("add.out",
         TORCH_FN(VariableType::add_out_out)
  );
  m.impl("addr",
         TORCH_FN(VariableType::addr)
  );
  m.impl("addr_",
         TORCH_FN(VariableType::addr_)
  );
  m.impl("affine_grid_generator",
         TORCH_FN(VariableType::affine_grid_generator)
  );
  m.impl("amin",
         TORCH_FN(VariableType::amin)
  );
  m.impl("arange.start_out",
         TORCH_FN(VariableType::arange_out_start_out)
  );
  m.impl("asin.out",
         TORCH_FN(VariableType::asin_out_out)
  );
  m.impl("asinh",
         TORCH_FN(VariableType::asinh)
  );
  m.impl("asinh_",
         TORCH_FN(VariableType::asinh_)
  );
  m.impl("avg_pool2d_backward",
         TORCH_FN(VariableType::avg_pool2d_backward)
  );
  m.impl("avg_pool3d",
         TORCH_FN(VariableType::avg_pool3d)
  );
  m.impl("avg_pool3d_backward.grad_input",
         TORCH_FN(VariableType::avg_pool3d_backward_out_grad_input)
  );
  m.impl("baddbmm",
         TORCH_FN(VariableType::baddbmm)
  );
  m.impl("baddbmm_",
         TORCH_FN(VariableType::baddbmm_)
  );
  m.impl("batch_norm_backward_reduce",
         TORCH_FN(VariableType::batch_norm_backward_reduce)
  );
  m.impl("bernoulli.out",
         TORCH_FN(VariableType::bernoulli_out_out)
  );
  m.impl("binary_cross_entropy_with_logits",
         TORCH_FN(VariableType::binary_cross_entropy_with_logits)
  );
  m.impl("bincount",
         TORCH_FN(VariableType::bincount)
  );
  m.impl("bmm",
         TORCH_FN(VariableType::bmm)
  );
  m.impl("bucketize.Tensor",
         TORCH_FN(VariableType::bucketize_Tensor)
  );
  m.impl("bucketize.Scalar",
         TORCH_FN(VariableType::bucketize_Scalar)
  );
  m.impl("cat.out",
         TORCH_FN(VariableType::cat_out_out)
  );
  m.impl("cholesky",
         TORCH_FN(VariableType::cholesky)
  );
  m.impl("clamp_max",
         TORCH_FN(VariableType::clamp_max)
  );
  m.impl("clamp_max_",
         TORCH_FN(VariableType::clamp_max_)
  );
  m.impl("coalesce",
         TORCH_FN(VariableType::coalesce)
  );
  m.impl("col2im.out",
         TORCH_FN(VariableType::col2im_out_out)
  );
  m.impl("conv_tbc",
         TORCH_FN(VariableType::conv_tbc)
  );
  m.impl("copysign.Tensor",
         TORCH_FN(VariableType::copysign_Tensor)
  );
  m.impl("copysign.Scalar",
         TORCH_FN(VariableType::copysign_Scalar)
  );
  m.impl("copysign_.Tensor",
         TORCH_FN(VariableType::copysign__Tensor)
  );
  m.impl("copysign_.Scalar",
         TORCH_FN(VariableType::copysign__Scalar)
  );
  m.impl("cudnn_convolution_backward_weight",
         TORCH_FN(VariableType::cudnn_convolution_backward_weight)
  );
  m.impl("cummax",
         TORCH_FN(VariableType::cummax)
  );
  m.impl("cumprod.out",
         TORCH_FN(VariableType::cumprod_out_out)
  );
  m.impl("cumsum",
         TORCH_FN(VariableType::cumsum)
  );
  m.impl("cumsum_",
         TORCH_FN(VariableType::cumsum_)
  );
  m.impl("dense_dim",
         TORCH_FN(VariableType::dense_dim)
  );
  m.impl("diagonal",
         TORCH_FN(VariableType::diagonal)
  );
  m.impl("dist",
         TORCH_FN(VariableType::dist)
  );
  m.impl("dot.out",
         TORCH_FN(VariableType::dot_out_out)
  );
  m.impl("elu",
         TORCH_FN(VariableType::elu)
  );
  m.impl("elu_",
         TORCH_FN(VariableType::elu_)
  );
  m.impl("embedding_renorm_",
         TORCH_FN(VariableType::embedding_renorm_)
  );
  m.impl("equal",
         TORCH_FN(VariableType::equal)
  );
  m.impl("erf.out",
         TORCH_FN(VariableType::erf_out_out)
  );
  m.impl("erfc",
         TORCH_FN(VariableType::erfc)
  );
  m.impl("erfc_",
         TORCH_FN(VariableType::erfc_)
  );
  m.impl("expm1",
         TORCH_FN(VariableType::expm1)
  );
  m.impl("expm1_",
         TORCH_FN(VariableType::expm1_)
  );
  m.impl("exponential_",
         TORCH_FN(VariableType::exponential_)
  );
  m.impl("floor",
         TORCH_FN(VariableType::floor)
  );
  m.impl("floor_",
         TORCH_FN(VariableType::floor_)
  );
  m.impl("fmin",
         TORCH_FN(VariableType::fmin)
  );
  m.impl("fmod.Scalar_out",
         TORCH_FN(VariableType::fmod_out_Scalar_out)
  );
  m.impl("fmod.Tensor_out",
         TORCH_FN(VariableType::fmod_out_Tensor_out)
  );
  m.impl("frac.out",
         TORCH_FN(VariableType::frac_out_out)
  );
  m.impl("fractional_max_pool2d.output",
         TORCH_FN(VariableType::fractional_max_pool2d_out_output)
  );
  m.impl("gcd.out",
         TORCH_FN(VariableType::gcd_out_out)
  );
  m.impl("hardshrink_backward",
         TORCH_FN(VariableType::hardshrink_backward)
  );
  m.impl("hardtanh.out",
         TORCH_FN(VariableType::hardtanh_out_out)
  );
  m.impl("im2col.out",
         TORCH_FN(VariableType::im2col_out_out)
  );
  m.impl("index.Tensor",
         TORCH_FN(VariableType::index_Tensor)
  );
  m.impl("index_put_",
         TORCH_FN(VariableType::index_put_)
  );
  m.impl("index_select",
         TORCH_FN(VariableType::index_select)
  );
  m.impl("is_coalesced",
         TORCH_FN(VariableType::is_coalesced)
  );
  m.impl("l1_loss",
         TORCH_FN(VariableType::l1_loss)
  );
  m.impl("l1_loss_backward.grad_input",
         TORCH_FN(VariableType::l1_loss_backward_out_grad_input)
  );
  m.impl("linalg_cholesky.out",
         TORCH_FN(VariableType::linalg_cholesky_out_out)
  );
  m.impl("linalg_eigvalsh.out",
         TORCH_FN(VariableType::linalg_eigvalsh_out_out)
  );
  m.impl("linalg_householder_product",
         TORCH_FN(VariableType::linalg_householder_product)
  );
  m.impl("linspace.out",
         TORCH_FN(VariableType::linspace_out_out)
  );
  m.impl("log2.out",
         TORCH_FN(VariableType::log2_out_out)
  );
  m.impl("log_normal_",
         TORCH_FN(VariableType::log_normal_)
  );
  m.impl("log.out",
         TORCH_FN(VariableType::log_out_out)
  );
  m.impl("log_sigmoid_backward",
         TORCH_FN(VariableType::log_sigmoid_backward)
  );
  m.impl("log_sigmoid_forward.output",
         TORCH_FN(VariableType::log_sigmoid_forward_out_output)
  );
  m.impl("logaddexp2.out",
         TORCH_FN(VariableType::logaddexp2_out_out)
  );
  m.impl("logaddexp.out",
         TORCH_FN(VariableType::logaddexp_out_out)
  );
  m.impl("logcumsumexp",
         TORCH_FN(VariableType::logcumsumexp)
  );
  m.impl("logsumexp",
         TORCH_FN(VariableType::logsumexp)
  );
  m.impl("lstsq.X",
         TORCH_FN(VariableType::lstsq_out_X)
  );
  m.impl("max.dim",
         TORCH_FN(VariableType::max_dim)
  );
  m.impl("max",
         TORCH_FN(VariableType::max)
  );
  m.impl("max_pool2d_with_indices.out",
         TORCH_FN(VariableType::max_pool2d_with_indices_out_out)
  );
  m.impl("max_unpool2d_backward",
         TORCH_FN(VariableType::max_unpool2d_backward)
  );
  m.impl("max_unpool3d",
         TORCH_FN(VariableType::max_unpool3d)
  );
  m.impl("max_unpool3d_backward.grad_input",
         TORCH_FN(VariableType::max_unpool3d_backward_out_grad_input)
  );
  m.impl("maximum",
         TORCH_FN(VariableType::maximum)
  );
  m.impl("mean.out",
         TORCH_FN(VariableType::mean_out_out)
  );
  m.impl("median.dim_values",
         TORCH_FN(VariableType::median_out_dim_values)
  );
  m.impl("miopen_batch_norm",
         TORCH_FN(VariableType::miopen_batch_norm)
  );
  m.impl("miopen_convolution_transpose_backward",
         TORCH_FN(VariableType::miopen_convolution_transpose_backward)
  );
  m.impl("miopen_convolution_transpose_backward_input",
         TORCH_FN(VariableType::miopen_convolution_transpose_backward_input)
  );
  m.impl("miopen_depthwise_convolution_backward_weight",
         TORCH_FN(VariableType::miopen_depthwise_convolution_backward_weight)
  );
  m.impl("mkldnn_adaptive_avg_pool2d_backward",
         TORCH_FN(VariableType::mkldnn_adaptive_avg_pool2d_backward)
  );
  m.impl("mkldnn_convolution_backward",
         TORCH_FN(VariableType::mkldnn_convolution_backward)
  );
  m.impl("mkldnn_reorder_conv3d_weight",
         TORCH_FN(VariableType::mkldnn_reorder_conv3d_weight)
  );
  m.impl("mode",
         TORCH_FN(VariableType::mode)
  );
  m.impl("multi_margin_loss.out",
         TORCH_FN(VariableType::multi_margin_loss_out_out)
  );
  m.impl("multilabel_margin_loss_forward",
         TORCH_FN(VariableType::multilabel_margin_loss_forward)
  );
  m.impl("mv.out",
         TORCH_FN(VariableType::mv_out_out)
  );
  m.impl("native_batch_norm_backward",
         TORCH_FN(VariableType::native_batch_norm_backward)
  );
  m.impl("native_norm",
         TORCH_FN(VariableType::native_norm)
  );
  m.impl("native_norm.ScalarOpt_dim_dtype",
         TORCH_FN(VariableType::native_norm_ScalarOpt_dim_dtype)
  );
  m.impl("ne.Scalar",
         TORCH_FN(VariableType::ne_Scalar)
  );
  m.impl("ne.Tensor",
         TORCH_FN(VariableType::ne_Tensor)
  );
  m.impl("ne_.Scalar",
         TORCH_FN(VariableType::ne__Scalar)
  );
  m.impl("ne_.Tensor",
         TORCH_FN(VariableType::ne__Tensor)
  );
  m.impl("nextafter.out",
         TORCH_FN(VariableType::nextafter_out_out)
  );
  m.impl("nll_loss2d_backward",
         TORCH_FN(VariableType::nll_loss2d_backward)
  );
  m.impl("nll_loss2d_forward.output",
         TORCH_FN(VariableType::nll_loss2d_forward_out_output)
  );
  m.impl("nll_loss_backward",
         TORCH_FN(VariableType::nll_loss_backward)
  );
  m.impl("nll_loss_forward.output",
         TORCH_FN(VariableType::nll_loss_forward_out_output)
  );
  m.impl("ormqr",
         TORCH_FN(VariableType::ormqr)
  );
  m.impl("polar.out",
         TORCH_FN(VariableType::polar_out_out)
  );
  m.impl("polygamma.out",
         TORCH_FN(VariableType::polygamma_out_out)
  );
  m.impl("pow.Tensor_Tensor_out",
         TORCH_FN(VariableType::pow_out_Tensor_Tensor_out)
  );
  m.impl("pow.Scalar_out",
         TORCH_FN(VariableType::pow_out_Scalar_out)
  );
  m.impl("pow.Tensor_Scalar_out",
         TORCH_FN(VariableType::pow_out_Tensor_Scalar_out)
  );
  m.impl("prod.int_out",
         TORCH_FN(VariableType::prod_out_int_out)
  );
  m.impl("q_per_channel_axis",
         TORCH_FN(VariableType::q_per_channel_axis)
  );
  m.impl("q_per_channel_zero_points",
         TORCH_FN(VariableType::q_per_channel_zero_points)
  );
  m.impl("random_.from",
         TORCH_FN(VariableType::random__from)
  );
  m.impl("random_.to",
         TORCH_FN(VariableType::random__to)
  );
  m.impl("random_",
         TORCH_FN(VariableType::random_)
  );
  m.impl("randperm.generator_out",
         TORCH_FN(VariableType::randperm_out_generator_out)
  );
  m.impl("record_stream",
         TORCH_FN(VariableType::record_stream)
  );
  m.impl("reflection_pad1d_backward",
         TORCH_FN(VariableType::reflection_pad1d_backward)
  );
  m.impl("reflection_pad2d",
         TORCH_FN(VariableType::reflection_pad2d)
  );
  m.impl("reflection_pad2d_backward.grad_input",
         TORCH_FN(VariableType::reflection_pad2d_backward_out_grad_input)
  );
  m.impl("remainder.Scalar",
         TORCH_FN(VariableType::remainder_Scalar)
  );
  m.impl("remainder.Tensor",
         TORCH_FN(VariableType::remainder_Tensor)
  );
  m.impl("remainder_.Scalar",
         TORCH_FN(VariableType::remainder__Scalar)
  );
  m.impl("remainder_.Tensor",
         TORCH_FN(VariableType::remainder__Tensor)
  );
  m.impl("repeat",
         TORCH_FN(VariableType::repeat)
  );
  m.impl("replication_pad1d",
         TORCH_FN(VariableType::replication_pad1d)
  );
  m.impl("replication_pad1d_backward.grad_input",
         TORCH_FN(VariableType::replication_pad1d_backward_out_grad_input)
  );
  m.impl("replication_pad2d.out",
         TORCH_FN(VariableType::replication_pad2d_out_out)
  );
  m.impl("roll",
         TORCH_FN(VariableType::roll)
  );
  m.impl("rot90",
         TORCH_FN(VariableType::rot90)
  );
  m.impl("round.out",
         TORCH_FN(VariableType::round_out_out)
  );
  m.impl("rrelu_with_noise_backward",
         TORCH_FN(VariableType::rrelu_with_noise_backward)
  );
  m.impl("rsqrt.out",
         TORCH_FN(VariableType::rsqrt_out_out)
  );
  m.impl("searchsorted.Tensor_out",
         TORCH_FN(VariableType::searchsorted_out_Tensor_out)
  );
  m.impl("signbit.out",
         TORCH_FN(VariableType::signbit_out_out)
  );
  m.impl("silu_backward",
         TORCH_FN(VariableType::silu_backward)
  );
  m.impl("slow_conv_transpose2d.out",
         TORCH_FN(VariableType::slow_conv_transpose2d_out_out)
  );
  m.impl("smooth_l1_loss.out",
         TORCH_FN(VariableType::smooth_l1_loss_out_out)
  );
  m.impl("soft_margin_loss",
         TORCH_FN(VariableType::soft_margin_loss)
  );
  m.impl("soft_margin_loss_backward.grad_input",
         TORCH_FN(VariableType::soft_margin_loss_backward_out_grad_input)
  );
  m.impl("softplus",
         TORCH_FN(VariableType::softplus)
  );
  m.impl("softplus_backward.grad_input",
         TORCH_FN(VariableType::softplus_backward_out_grad_input)
  );
  m.impl("sort",
         TORCH_FN(VariableType::sort)
  );
  m.impl("sort.stable",
         TORCH_FN(VariableType::sort_stable)
  );
  m.impl("split.Tensor",
         TORCH_FN(VariableType::split_Tensor)
  );
  m.impl("square.out",
         TORCH_FN(VariableType::square_out_out)
  );
  m.impl("sspaddmm.out",
         TORCH_FN(VariableType::sspaddmm_out_out)
  );
  m.impl("std",
         TORCH_FN(VariableType::std)
  );
  m.impl("std.dim",
         TORCH_FN(VariableType::std_dim)
  );
  m.impl("sum",
         TORCH_FN(VariableType::sum)
  );
  m.impl("sum.dim_IntList",
         TORCH_FN(VariableType::sum_dim_IntList)
  );
  m.impl("t",
         TORCH_FN(VariableType::t)
  );
  m.impl("t_",
         TORCH_FN(VariableType::t_)
  );
  m.impl("take",
         TORCH_FN(VariableType::take)
  );
  m.impl("tanh.out",
         TORCH_FN(VariableType::tanh_out_out)
  );
  m.impl("thnn_conv_depthwise2d_backward.grad_input",
         TORCH_FN(VariableType::thnn_conv_depthwise2d_backward_out_grad_input)
  );
  m.impl("to_mkldnn",
         TORCH_FN(VariableType::to_mkldnn)
  );
  m.impl("to_sparse.sparse_dim",
         TORCH_FN(VariableType::to_sparse_sparse_dim)
  );
  m.impl("to_sparse",
         TORCH_FN(VariableType::to_sparse)
  );
  m.impl("topk",
         TORCH_FN(VariableType::topk)
  );
  m.impl("transpose.int",
         TORCH_FN(VariableType::transpose_int)
  );
  m.impl("transpose_",
         TORCH_FN(VariableType::transpose_)
  );
  m.impl("triu.out",
         TORCH_FN(VariableType::triu_out_out)
  );
  m.impl("trunc",
         TORCH_FN(VariableType::trunc)
  );
  m.impl("trunc_",
         TORCH_FN(VariableType::trunc_)
  );
  m.impl("upsample_bicubic2d.out",
         TORCH_FN(VariableType::upsample_bicubic2d_out_out)
  );
  m.impl("upsample_linear1d_backward.vec",
         TORCH_FN(VariableType::upsample_linear1d_backward_vec)
  );
  m.impl("upsample_linear1d_backward",
         TORCH_FN(VariableType::upsample_linear1d_backward)
  );
  m.impl("upsample_nearest2d_backward.vec",
         TORCH_FN(VariableType::upsample_nearest2d_backward_vec)
  );
  m.impl("upsample_nearest2d_backward",
         TORCH_FN(VariableType::upsample_nearest2d_backward)
  );
  m.impl("upsample_nearest3d.vec",
         TORCH_FN(VariableType::upsample_nearest3d_vec)
  );
  m.impl("upsample_nearest3d",
         TORCH_FN(VariableType::upsample_nearest3d)
  );
  m.impl("upsample_nearest3d_backward.grad_input",
         TORCH_FN(VariableType::upsample_nearest3d_backward_out_grad_input)
  );
  m.impl("upsample_trilinear3d.out",
         TORCH_FN(VariableType::upsample_trilinear3d_out_out)
  );
  m.impl("values",
         TORCH_FN(VariableType::values)
  );
  m.impl("var_mean",
         TORCH_FN(VariableType::var_mean)
  );
  m.impl("var_mean.dim",
         TORCH_FN(VariableType::var_mean_dim)
  );
}

}

}} // namespace torch::autograd
