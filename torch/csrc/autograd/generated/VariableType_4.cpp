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
Tensor _cholesky_solve_helper(c10::DispatchKeySet ks, const Tensor & self, const Tensor & A, bool upper) {
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  auto _any_requires_grad = compute_requires_grad( self, A );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cholesky_solve_helper"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_cholesky_solve_helper(ks & c10::after_autograd_keyset, self_, A_, upper);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_cholesky_solve_helper");
  return result;
}
Tensor _copy_from(c10::DispatchKeySet ks, const Tensor & self, const Tensor & dst, bool non_blocking) {
  auto& self_ = unpack(self, "self", 0);
  auto& dst_ = unpack(dst, "dst", 1);
  auto _any_requires_grad = compute_requires_grad( self, dst );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_copy_from"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, dst ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> dst__storage_saved =
    dst_.has_storage() ? c10::optional<Storage>(dst_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> dst__impl_saved;
  if (dst_.defined()) dst__impl_saved = dst_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_copy_from(ks & c10::after_autograd_keyset, self_, dst_, non_blocking);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (dst__storage_saved.has_value())
    AT_ASSERT(dst__storage_saved.value().is_alias_of(dst_.storage()));
  if (dst__impl_saved) AT_ASSERT(dst__impl_saved == dst_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_copy_from");
  return result;
}
Tensor _ctc_loss_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, const Tensor & neg_log_likelihood, const Tensor & log_alpha, int64_t blank, bool zero_infinity) {
  auto& grad_ = unpack(grad, "grad", 0);
  auto& log_probs_ = unpack(log_probs, "log_probs", 1);
  auto& targets_ = unpack(targets, "targets", 2);
  auto& neg_log_likelihood_ = unpack(neg_log_likelihood, "neg_log_likelihood", 5);
  auto& log_alpha_ = unpack(log_alpha, "log_alpha", 6);
  auto _any_requires_grad = compute_requires_grad( grad, log_probs, targets, neg_log_likelihood, log_alpha );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_ctc_loss_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad, log_probs, targets, neg_log_likelihood, log_alpha ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  c10::optional<Storage> log_probs__storage_saved =
    log_probs_.has_storage() ? c10::optional<Storage>(log_probs_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> log_probs__impl_saved;
  if (log_probs_.defined()) log_probs__impl_saved = log_probs_.getIntrusivePtr();
  c10::optional<Storage> targets__storage_saved =
    targets_.has_storage() ? c10::optional<Storage>(targets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> targets__impl_saved;
  if (targets_.defined()) targets__impl_saved = targets_.getIntrusivePtr();
  c10::optional<Storage> neg_log_likelihood__storage_saved =
    neg_log_likelihood_.has_storage() ? c10::optional<Storage>(neg_log_likelihood_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> neg_log_likelihood__impl_saved;
  if (neg_log_likelihood_.defined()) neg_log_likelihood__impl_saved = neg_log_likelihood_.getIntrusivePtr();
  c10::optional<Storage> log_alpha__storage_saved =
    log_alpha_.has_storage() ? c10::optional<Storage>(log_alpha_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> log_alpha__impl_saved;
  if (log_alpha_.defined()) log_alpha__impl_saved = log_alpha_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_ctc_loss_backward(ks & c10::after_autograd_keyset, grad_, log_probs_, targets_, input_lengths, target_lengths, neg_log_likelihood_, log_alpha_, blank, zero_infinity);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (log_probs__storage_saved.has_value())
    AT_ASSERT(log_probs__storage_saved.value().is_alias_of(log_probs_.storage()));
  if (log_probs__impl_saved) AT_ASSERT(log_probs__impl_saved == log_probs_.getIntrusivePtr());
  if (targets__storage_saved.has_value())
    AT_ASSERT(targets__storage_saved.value().is_alias_of(targets_.storage()));
  if (targets__impl_saved) AT_ASSERT(targets__impl_saved == targets_.getIntrusivePtr());
  if (neg_log_likelihood__storage_saved.has_value())
    AT_ASSERT(neg_log_likelihood__storage_saved.value().is_alias_of(neg_log_likelihood_.storage()));
  if (neg_log_likelihood__impl_saved) AT_ASSERT(neg_log_likelihood__impl_saved == neg_log_likelihood_.getIntrusivePtr());
  if (log_alpha__storage_saved.has_value())
    AT_ASSERT(log_alpha__storage_saved.value().is_alias_of(log_alpha_.storage()));
  if (log_alpha__impl_saved) AT_ASSERT(log_alpha__impl_saved == log_alpha_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_ctc_loss_backward");
  return result;
}
std::tuple<Tensor,Tensor> _cudnn_ctc_loss(c10::DispatchKeySet ks, const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) {
  auto& log_probs_ = unpack(log_probs, "log_probs", 0);
  auto& targets_ = unpack(targets, "targets", 1);
  auto _any_requires_grad = compute_requires_grad( log_probs );
  (void)_any_requires_grad;
  check_no_requires_grad(targets, "targets");
  std::shared_ptr<CudnnCtcLossBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CudnnCtcLossBackward>(new CudnnCtcLossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( log_probs ));
    grad_fn->zero_infinity = zero_infinity;
  }
  Tensor result0;
  Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> log_probs__storage_saved =
    log_probs_.has_storage() ? c10::optional<Storage>(log_probs_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> log_probs__impl_saved;
  if (log_probs_.defined()) log_probs__impl_saved = log_probs_.getIntrusivePtr();
  c10::optional<Storage> targets__storage_saved =
    targets_.has_storage() ? c10::optional<Storage>(targets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> targets__impl_saved;
  if (targets_.defined()) targets__impl_saved = targets_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_cudnn_ctc_loss(ks & c10::after_autograd_keyset, log_probs_, targets_, input_lengths, target_lengths, blank, deterministic, zero_infinity);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (log_probs__storage_saved.has_value())
    AT_ASSERT(log_probs__storage_saved.value().is_alias_of(log_probs_.storage()));
  if (log_probs__impl_saved) AT_ASSERT(log_probs__impl_saved == log_probs_.getIntrusivePtr());
  if (targets__storage_saved.has_value())
    AT_ASSERT(targets__storage_saved.value().is_alias_of(targets_.storage()));
  if (targets__impl_saved) AT_ASSERT(targets__impl_saved == targets_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  throw_error_for_complex_autograd(result0, "_cudnn_ctc_loss");
  return std::make_tuple(std::move(result0), std::move(result1));
}
void _cummin_helper(c10::DispatchKeySet ks, const Tensor & self, Tensor & values, Tensor & indices, int64_t dim) {
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
    at::redispatch::_cummin_helper(ks & c10::after_autograd_keyset, self_, values_, indices_, dim);
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
Tensor _euclidean_dist(c10::DispatchKeySet ks, const Tensor & x1, const Tensor & x2) {
  auto& x1_ = unpack(x1, "x1", 0);
  auto& x2_ = unpack(x2, "x2", 1);
  auto _any_requires_grad = compute_requires_grad( x1, x2 );
  (void)_any_requires_grad;
  std::shared_ptr<EuclideanDistBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<EuclideanDistBackward>(new EuclideanDistBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( x1, x2 ));
    grad_fn->x1_ = SavedVariable(x1, false);
    grad_fn->x2_ = SavedVariable(x2, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> x1__storage_saved =
    x1_.has_storage() ? c10::optional<Storage>(x1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> x1__impl_saved;
  if (x1_.defined()) x1__impl_saved = x1_.getIntrusivePtr();
  c10::optional<Storage> x2__storage_saved =
    x2_.has_storage() ? c10::optional<Storage>(x2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> x2__impl_saved;
  if (x2_.defined()) x2__impl_saved = x2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_euclidean_dist(ks & c10::after_autograd_keyset, x1_, x2_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (x1__storage_saved.has_value())
    AT_ASSERT(x1__storage_saved.value().is_alias_of(x1_.storage()));
  if (x1__impl_saved) AT_ASSERT(x1__impl_saved == x1_.getIntrusivePtr());
  if (x2__storage_saved.has_value())
    AT_ASSERT(x2__storage_saved.value().is_alias_of(x2_.storage()));
  if (x2__impl_saved) AT_ASSERT(x2__impl_saved == x2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  throw_error_for_complex_autograd(result, "_euclidean_dist");
  return result;
}
std::vector<Tensor> _foreach_erfc(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_erfc"), deleteNode);
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
    return at::redispatch::_foreach_erfc(ks & c10::after_autograd_keyset, tensors_);
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
void _foreach_erfc_(c10::DispatchKeySet ks, TensorList self) {
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
    at::redispatch::_foreach_erfc_(ks & c10::after_autograd_keyset, self_);
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
std::vector<Tensor> _foreach_expm1(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_expm1"), deleteNode);
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
    return at::redispatch::_foreach_expm1(ks & c10::after_autograd_keyset, tensors_);
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
void _foreach_expm1_(c10::DispatchKeySet ks, TensorList self) {
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
    at::redispatch::_foreach_expm1_(ks & c10::after_autograd_keyset, self_);
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
std::vector<Tensor> _foreach_floor(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_floor"), deleteNode);
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
    return at::redispatch::_foreach_floor(ks & c10::after_autograd_keyset, tensors_);
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
void _foreach_floor_(c10::DispatchKeySet ks, TensorList self) {
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
    at::redispatch::_foreach_floor_(ks & c10::after_autograd_keyset, self_);
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
std::vector<Tensor> _foreach_maximum_List(c10::DispatchKeySet ks, TensorList tensors1, TensorList tensors2) {
  auto tensors1_ = unpack(tensors1, "tensors1", 0);
  auto tensors2_ = unpack(tensors2, "tensors2", 1);
  auto _any_requires_grad = compute_requires_grad( tensors1, tensors2 );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_maximum"), deleteNode);
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
    return at::redispatch::_foreach_maximum(ks & c10::after_autograd_keyset, tensors1_, tensors2_);
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
std::vector<Tensor> _foreach_trunc(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_trunc"), deleteNode);
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
    return at::redispatch::_foreach_trunc(ks & c10::after_autograd_keyset, tensors_);
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
void _foreach_trunc_(c10::DispatchKeySet ks, TensorList self) {
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
    at::redispatch::_foreach_trunc_(ks & c10::after_autograd_keyset, self_);
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
Tensor & _index_copy_(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  auto _any_requires_grad = compute_requires_grad( self, index, source );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_index_copy_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, index, source ));
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
  c10::optional<Storage> source__storage_saved =
    source_.has_storage() ? c10::optional<Storage>(source_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> source__impl_saved;
  if (source_.defined()) source__impl_saved = source_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_index_copy_(ks & c10::after_autograd_keyset, self_, dim, index_, source_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (source__storage_saved.has_value())
    AT_ASSERT(source__storage_saved.value().is_alias_of(source_.storage()));
  if (source__impl_saved) AT_ASSERT(source__impl_saved == source_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor _inverse_helper(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_inverse_helper"), deleteNode);
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
    return at::redispatch::_inverse_helper(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "_inverse_helper");
  return result;
}
Tensor _masked_scale(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mask, double scale) {
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto _any_requires_grad = compute_requires_grad( self, mask );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_masked_scale"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mask ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_masked_scale(ks & c10::after_autograd_keyset, self_, mask_, scale);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_masked_scale");
  return result;
}
Tensor _pdist_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & self, double p, const Tensor & pdist) {
  auto& grad_ = unpack(grad, "grad", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& pdist_ = unpack(pdist, "pdist", 3);
  auto _any_requires_grad = compute_requires_grad( grad, self, pdist );
  (void)_any_requires_grad;
  std::shared_ptr<PdistBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PdistBackwardBackward>(new PdistBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad, self, pdist ));
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
  c10::optional<Storage> pdist__storage_saved =
    pdist_.has_storage() ? c10::optional<Storage>(pdist_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> pdist__impl_saved;
  if (pdist_.defined()) pdist__impl_saved = pdist_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_pdist_backward(ks & c10::after_autograd_keyset, grad_, self_, p, pdist_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (pdist__storage_saved.has_value())
    AT_ASSERT(pdist__storage_saved.value().is_alias_of(pdist_.storage()));
  if (pdist__impl_saved) AT_ASSERT(pdist__impl_saved == pdist_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_pdist_backward");
  return result;
}
Tensor _s_where(c10::DispatchKeySet ks, const Tensor & condition, const Tensor & self, const Tensor & other) {
  auto& condition_ = unpack(condition, "condition", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<SWhereBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SWhereBackward>(new SWhereBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->condition_ = SavedVariable(condition, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> condition__storage_saved =
    condition_.has_storage() ? c10::optional<Storage>(condition_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> condition__impl_saved;
  if (condition_.defined()) condition__impl_saved = condition_.getIntrusivePtr();
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
    return at::redispatch::_s_where(ks & c10::after_autograd_keyset, condition_, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (condition__storage_saved.has_value())
    AT_ASSERT(condition__storage_saved.value().is_alias_of(condition_.storage()));
  if (condition__impl_saved) AT_ASSERT(condition__impl_saved == condition_.getIntrusivePtr());
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
  return result;
}
Tensor _sample_dirichlet(c10::DispatchKeySet ks, const Tensor & self, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_sample_dirichlet"), deleteNode);
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
    return at::redispatch::_sample_dirichlet(ks & c10::after_autograd_keyset, self_, generator);
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
  throw_error_for_complex_autograd(result, "_sample_dirichlet");
  return result;
}
Tensor _sparse_addmm(c10::DispatchKeySet ks, const Tensor & self, const Tensor & sparse, const Tensor & dense, const Scalar & beta, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& sparse_ = unpack(sparse, "sparse", 1);
  auto& dense_ = unpack(dense, "dense", 2);
  auto _any_requires_grad = compute_requires_grad( self, sparse, dense );
  (void)_any_requires_grad;
  std::shared_ptr<SparseAddmmBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SparseAddmmBackward>(new SparseAddmmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, sparse, dense ));
    grad_fn->sparse_ = SavedVariable(sparse, false);
    grad_fn->dense_sizes = dense.sizes().vec();
    grad_fn->dense_strides = strides_or_error(dense, "dense").vec();
    grad_fn->alpha = alpha;
    grad_fn->beta = beta;
    if (grad_fn->should_compute_output(1)) {
      grad_fn->dense_ = SavedVariable(dense, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> sparse__storage_saved =
    sparse_.has_storage() ? c10::optional<Storage>(sparse_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> sparse__impl_saved;
  if (sparse_.defined()) sparse__impl_saved = sparse_.getIntrusivePtr();
  c10::optional<Storage> dense__storage_saved =
    dense_.has_storage() ? c10::optional<Storage>(dense_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> dense__impl_saved;
  if (dense_.defined()) dense__impl_saved = dense_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_sparse_addmm(ks & c10::after_autograd_keyset, self_, sparse_, dense_, beta, alpha);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (sparse__storage_saved.has_value())
    AT_ASSERT(sparse__storage_saved.value().is_alias_of(sparse_.storage()));
  if (sparse__impl_saved) AT_ASSERT(sparse__impl_saved == sparse_.getIntrusivePtr());
  if (dense__storage_saved.has_value())
    AT_ASSERT(dense__storage_saved.value().is_alias_of(dense_.storage()));
  if (dense__impl_saved) AT_ASSERT(dense__impl_saved == dense_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_sparse_addmm");
  return result;
}
Tensor _sparse_coo_tensor_with_dims_and_tensors(c10::DispatchKeySet ks, int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  auto& indices_ = unpack(indices, "indices", 3);
  auto& values_ = unpack(values, "values", 4);
  auto _any_requires_grad = compute_requires_grad( values );
  (void)_any_requires_grad;
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<SparseCooTensorWithDimsAndTensorsBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SparseCooTensorWithDimsAndTensorsBackward>(new SparseCooTensorWithDimsAndTensorsBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( values ));
    grad_fn->indices_ = SavedVariable(indices, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_sparse_coo_tensor_with_dims_and_tensors(ks & c10::after_autograd_keyset, sparse_dim, dense_dim, size, indices_, values_, dtype, layout, device, pin_memory);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_sparse_coo_tensor_with_dims_and_tensors");
  return result;
}
Tensor _sparse_softmax(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool half_to_float) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SparseSoftmaxBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SparseSoftmaxBackward>(new SparseSoftmaxBackward(), deleteNode);
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
    return at::redispatch::_sparse_softmax(ks & c10::after_autograd_keyset, self_, dim, half_to_float);
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
  throw_error_for_complex_autograd(result, "_sparse_softmax");
  return result;
}
Tensor _sparse_sparse_matmul(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<SparseSparseMatmulBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SparseSparseMatmulBackward>(new SparseSparseMatmulBackward(), deleteNode);
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
    return at::redispatch::_sparse_sparse_matmul(ks & c10::after_autograd_keyset, self_, other_);
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
  throw_error_for_complex_autograd(result, "_sparse_sparse_matmul");
  return result;
}
Tensor _stack(c10::DispatchKeySet ks, TensorList tensors, int64_t dim) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_stack"), deleteNode);
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
    return at::redispatch::_stack(ks & c10::after_autograd_keyset, tensors_, dim);
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
  throw_error_for_complex_autograd(result, "_stack");
  return result;
}
Tensor _standard_gamma(c10::DispatchKeySet ks, const Tensor & self, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<StandardGammaBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<StandardGammaBackward>(new StandardGammaBackward(), deleteNode);
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
    return at::redispatch::_standard_gamma(ks & c10::after_autograd_keyset, self_, generator);
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
  throw_error_for_complex_autograd(result, "_standard_gamma");
  return result;
}
std::tuple<Tensor,Tensor> _symeig_helper(c10::DispatchKeySet ks, const Tensor & self, bool eigenvectors, bool upper) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_symeig_helper"), deleteNode);
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
    return at::redispatch::_symeig_helper(ks & c10::after_autograd_keyset, self_, eigenvectors, upper);
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
  throw_error_for_complex_autograd(result0, "_symeig_helper");
  throw_error_for_complex_autograd(result1, "_symeig_helper");
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor,Tensor> _thnn_fused_lstm_cell(c10::DispatchKeySet ks, const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const c10::optional<Tensor> & input_bias, const c10::optional<Tensor> & hidden_bias) {
  auto& input_gates_ = unpack(input_gates, "input_gates", 0);
  auto& hidden_gates_ = unpack(hidden_gates, "hidden_gates", 1);
  auto& cx_ = unpack(cx, "cx", 2);
  auto _any_requires_grad = compute_requires_grad( input_gates, hidden_gates, cx, input_bias, hidden_bias );
  (void)_any_requires_grad;
  std::shared_ptr<ThnnFusedLstmCellBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ThnnFusedLstmCellBackward>(new ThnnFusedLstmCellBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input_gates, hidden_gates, cx, input_bias, hidden_bias ));
    grad_fn->input_gates_ = SavedVariable(input_gates, false);
    grad_fn->hidden_gates_ = SavedVariable(hidden_gates, false);
    grad_fn->cx_ = SavedVariable(cx, false);
    grad_fn->input_bias_ = SavedVariable(input_bias, false);
    grad_fn->hidden_bias_ = SavedVariable(hidden_bias, false);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #ifndef NDEBUG
  c10::optional<Storage> input_gates__storage_saved =
    input_gates_.has_storage() ? c10::optional<Storage>(input_gates_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input_gates__impl_saved;
  if (input_gates_.defined()) input_gates__impl_saved = input_gates_.getIntrusivePtr();
  c10::optional<Storage> hidden_gates__storage_saved =
    hidden_gates_.has_storage() ? c10::optional<Storage>(hidden_gates_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> hidden_gates__impl_saved;
  if (hidden_gates_.defined()) hidden_gates__impl_saved = hidden_gates_.getIntrusivePtr();
  c10::optional<Storage> cx__storage_saved =
    cx_.has_storage() ? c10::optional<Storage>(cx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> cx__impl_saved;
  if (cx_.defined()) cx__impl_saved = cx_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_thnn_fused_lstm_cell(ks & c10::after_autograd_keyset, input_gates_, hidden_gates_, cx_, input_bias, hidden_bias);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
  #ifndef NDEBUG
  if (input_gates__storage_saved.has_value())
    AT_ASSERT(input_gates__storage_saved.value().is_alias_of(input_gates_.storage()));
  if (input_gates__impl_saved) AT_ASSERT(input_gates__impl_saved == input_gates_.getIntrusivePtr());
  if (hidden_gates__storage_saved.has_value())
    AT_ASSERT(hidden_gates__storage_saved.value().is_alias_of(hidden_gates_.storage()));
  if (hidden_gates__impl_saved) AT_ASSERT(hidden_gates__impl_saved == hidden_gates_.getIntrusivePtr());
  if (cx__storage_saved.has_value())
    AT_ASSERT(cx__storage_saved.value().is_alias_of(cx_.storage()));
  if (cx__impl_saved) AT_ASSERT(cx__impl_saved == cx_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  throw_error_for_complex_autograd(result0, "_thnn_fused_lstm_cell");
  throw_error_for_complex_autograd(result1, "_thnn_fused_lstm_cell");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor _var(c10::DispatchKeySet ks, const Tensor & self, bool unbiased) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_var"), deleteNode);
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
    return at::redispatch::_var(ks & c10::after_autograd_keyset, self_, unbiased);
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
  throw_error_for_complex_autograd(result, "_var");
  return result;
}
Tensor & abs_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("abs");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("abs");
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
    at::redispatch::abs_outf(ks & c10::after_autograd_keyset, self_, out_);
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
std::tuple<Tensor,Tensor> adaptive_max_pool2d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AdaptiveMaxPool2DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AdaptiveMaxPool2DBackward>(new AdaptiveMaxPool2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
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
    return at::redispatch::adaptive_max_pool2d(ks & c10::after_autograd_keyset, self_, output_size);
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
  throw_error_for_complex_autograd(result0, "adaptive_max_pool2d");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & adaptive_max_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & indices, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  auto& grad_input_ = unpack(grad_input, "grad_input", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("adaptive_max_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_max_pool2d_backward");
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
    at::redispatch::adaptive_max_pool2d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, indices_, grad_input_);
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
std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, Tensor & out, Tensor & indices) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_max_pool3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("adaptive_max_pool3d");
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
    at::redispatch::adaptive_max_pool3d_outf(ks & c10::after_autograd_keyset, self_, output_size, out_, indices_);
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
Tensor addcdiv(c10::DispatchKeySet ks, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value) {
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  auto _any_requires_grad = compute_requires_grad( self, tensor1, tensor2 );
  (void)_any_requires_grad;
  std::shared_ptr<AddcdivBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddcdivBackward>(new AddcdivBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->tensor1_scalar_type = tensor1.scalar_type();
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->tensor1_ = SavedVariable(tensor1, false);
    }
    grad_fn->tensor2_scalar_type = tensor2.scalar_type();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor1__storage_saved =
    tensor1_.has_storage() ? c10::optional<Storage>(tensor1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor1__impl_saved;
  if (tensor1_.defined()) tensor1__impl_saved = tensor1_.getIntrusivePtr();
  c10::optional<Storage> tensor2__storage_saved =
    tensor2_.has_storage() ? c10::optional<Storage>(tensor2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor2__impl_saved;
  if (tensor2_.defined()) tensor2__impl_saved = tensor2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::addcdiv(ks & c10::after_autograd_keyset, self_, tensor1_, tensor2_, value);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor1__storage_saved.has_value())
    AT_ASSERT(tensor1__storage_saved.value().is_alias_of(tensor1_.storage()));
  if (tensor1__impl_saved) AT_ASSERT(tensor1__impl_saved == tensor1_.getIntrusivePtr());
  if (tensor2__storage_saved.has_value())
    AT_ASSERT(tensor2__storage_saved.value().is_alias_of(tensor2_.storage()));
  if (tensor2__impl_saved) AT_ASSERT(tensor2__impl_saved == tensor2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & addcdiv_(c10::DispatchKeySet ks, Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value) {
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  auto _any_requires_grad = compute_requires_grad( self, tensor1, tensor2 );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<AddcdivBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddcdivBackward>(new AddcdivBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->tensor1_scalar_type = tensor1.scalar_type();
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->tensor1_ = SavedVariable(tensor1, false);
    }
    grad_fn->tensor2_scalar_type = tensor2.scalar_type();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor1__storage_saved =
    tensor1_.has_storage() ? c10::optional<Storage>(tensor1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor1__impl_saved;
  if (tensor1_.defined()) tensor1__impl_saved = tensor1_.getIntrusivePtr();
  c10::optional<Storage> tensor2__storage_saved =
    tensor2_.has_storage() ? c10::optional<Storage>(tensor2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor2__impl_saved;
  if (tensor2_.defined()) tensor2__impl_saved = tensor2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::addcdiv_(ks & c10::after_autograd_keyset, self_, tensor1_, tensor2_, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor1__storage_saved.has_value())
    AT_ASSERT(tensor1__storage_saved.value().is_alias_of(tensor1_.storage()));
  if (tensor1__impl_saved) AT_ASSERT(tensor1__impl_saved == tensor1_.getIntrusivePtr());
  if (tensor2__storage_saved.has_value())
    AT_ASSERT(tensor2__storage_saved.value().is_alias_of(tensor2_.storage()));
  if (tensor2__impl_saved) AT_ASSERT(tensor2__impl_saved == tensor2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & addcmul_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self, tensor1, tensor2 );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    throw_error_out_requires_grad("addcmul");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addcmul");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor1__storage_saved =
    tensor1_.has_storage() ? c10::optional<Storage>(tensor1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor1__impl_saved;
  if (tensor1_.defined()) tensor1__impl_saved = tensor1_.getIntrusivePtr();
  c10::optional<Storage> tensor2__storage_saved =
    tensor2_.has_storage() ? c10::optional<Storage>(tensor2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor2__impl_saved;
  if (tensor2_.defined()) tensor2__impl_saved = tensor2_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::addcmul_outf(ks & c10::after_autograd_keyset, self_, tensor1_, tensor2_, value, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor1__storage_saved.has_value())
    AT_ASSERT(tensor1__storage_saved.value().is_alias_of(tensor1_.storage()));
  if (tensor1__impl_saved) AT_ASSERT(tensor1__impl_saved == tensor1_.getIntrusivePtr());
  if (tensor2__storage_saved.has_value())
    AT_ASSERT(tensor2__storage_saved.value().is_alias_of(tensor2_.storage()));
  if (tensor2__impl_saved) AT_ASSERT(tensor2__impl_saved == tensor2_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & addmm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  auto& out_ = unpack(out, "out", 5);
  auto _any_requires_grad = compute_requires_grad( self, mat1, mat2 );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    throw_error_out_requires_grad("addmm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addmm");
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
    at::redispatch::addmm_outf(ks & c10::after_autograd_keyset, self_, mat1_, mat2_, beta, alpha, out_);
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
Tensor addmv(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat, const Tensor & vec, const Scalar & beta, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& mat_ = unpack(mat, "mat", 1);
  auto& vec_ = unpack(vec, "vec", 2);
  auto _any_requires_grad = compute_requires_grad( self, mat, vec );
  (void)_any_requires_grad;
  std::shared_ptr<AddmvBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddmvBackward>(new AddmvBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat, vec ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->vec_ = SavedVariable(vec, false);
    }
    grad_fn->alpha = alpha;
    grad_fn->beta = beta;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->mat_ = SavedVariable(mat, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat__storage_saved =
    mat_.has_storage() ? c10::optional<Storage>(mat_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat__impl_saved;
  if (mat_.defined()) mat__impl_saved = mat_.getIntrusivePtr();
  c10::optional<Storage> vec__storage_saved =
    vec_.has_storage() ? c10::optional<Storage>(vec_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec__impl_saved;
  if (vec_.defined()) vec__impl_saved = vec_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::addmv(ks & c10::after_autograd_keyset, self_, mat_, vec_, beta, alpha);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat__storage_saved.has_value())
    AT_ASSERT(mat__storage_saved.value().is_alias_of(mat_.storage()));
  if (mat__impl_saved) AT_ASSERT(mat__impl_saved == mat_.getIntrusivePtr());
  if (vec__storage_saved.has_value())
    AT_ASSERT(vec__storage_saved.value().is_alias_of(vec_.storage()));
  if (vec__impl_saved) AT_ASSERT(vec__impl_saved == vec_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor & addmv_(c10::DispatchKeySet ks, Tensor & self, const Tensor & mat, const Tensor & vec, const Scalar & beta, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& mat_ = unpack(mat, "mat", 1);
  auto& vec_ = unpack(vec, "vec", 2);
  auto _any_requires_grad = compute_requires_grad( self, mat, vec );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<AddmvBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddmvBackward>(new AddmvBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat, vec ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->vec_ = SavedVariable(vec, false);
    }
    grad_fn->alpha = alpha;
    grad_fn->beta = beta;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->mat_ = SavedVariable(mat, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat__storage_saved =
    mat_.has_storage() ? c10::optional<Storage>(mat_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat__impl_saved;
  if (mat_.defined()) mat__impl_saved = mat_.getIntrusivePtr();
  c10::optional<Storage> vec__storage_saved =
    vec_.has_storage() ? c10::optional<Storage>(vec_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec__impl_saved;
  if (vec_.defined()) vec__impl_saved = vec_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::addmv_(ks & c10::after_autograd_keyset, self_, mat_, vec_, beta, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat__storage_saved.has_value())
    AT_ASSERT(mat__storage_saved.value().is_alias_of(mat_.storage()));
  if (mat__impl_saved) AT_ASSERT(mat__impl_saved == mat_.getIntrusivePtr());
  if (vec__storage_saved.has_value())
    AT_ASSERT(vec__storage_saved.value().is_alias_of(vec_.storage()));
  if (vec__impl_saved) AT_ASSERT(vec__impl_saved == vec_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor angle(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AngleBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AngleBackward>(new AngleBackward(), deleteNode);
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
    return at::redispatch::angle(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "angle");
  return result;
}
Tensor & argmax_out_out(c10::DispatchKeySet ks, const Tensor & self, c10::optional<int64_t> dim, bool keepdim, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 3);
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
    at::redispatch::argmax_outf(ks & c10::after_autograd_keyset, self_, dim, keepdim, out_);
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
Tensor & atan2_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("atan2");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("atan2");
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
    at::redispatch::atan2_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
Tensor & atan_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("atan");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("atan");
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
    at::redispatch::atan_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor atanh(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AtanhBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AtanhBackward0>(new AtanhBackward0(), deleteNode);
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
    return at::redispatch::atanh(ks & c10::after_autograd_keyset, self_);
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
Tensor & atanh_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<AtanhBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AtanhBackward1>(new AtanhBackward1(), deleteNode);
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
    at::redispatch::atanh_(ks & c10::after_autograd_keyset, self_);
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
Tensor & avg_pool2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 7);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("avg_pool2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("avg_pool2d");
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
    at::redispatch::avg_pool2d_outf(ks & c10::after_autograd_keyset, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out_);
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
Tensor batch_norm_backward_elemt(c10::DispatchKeySet ks, const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & weight, const Tensor & mean_dy, const Tensor & mean_dy_xmu, const Tensor & count) {
  auto& grad_out_ = unpack(grad_out, "grad_out", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& mean_ = unpack(mean, "mean", 2);
  auto& invstd_ = unpack(invstd, "invstd", 3);
  auto& mean_dy_ = unpack(mean_dy, "mean_dy", 5);
  auto& mean_dy_xmu_ = unpack(mean_dy_xmu, "mean_dy_xmu", 6);
  auto& count_ = unpack(count, "count", 7);
  auto _any_requires_grad = compute_requires_grad( grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_backward_elemt"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count ));
  }
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
  c10::optional<Storage> mean_dy__storage_saved =
    mean_dy_.has_storage() ? c10::optional<Storage>(mean_dy_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean_dy__impl_saved;
  if (mean_dy_.defined()) mean_dy__impl_saved = mean_dy_.getIntrusivePtr();
  c10::optional<Storage> mean_dy_xmu__storage_saved =
    mean_dy_xmu_.has_storage() ? c10::optional<Storage>(mean_dy_xmu_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean_dy_xmu__impl_saved;
  if (mean_dy_xmu_.defined()) mean_dy_xmu__impl_saved = mean_dy_xmu_.getIntrusivePtr();
  c10::optional<Storage> count__storage_saved =
    count_.has_storage() ? c10::optional<Storage>(count_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> count__impl_saved;
  if (count_.defined()) count__impl_saved = count_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::batch_norm_backward_elemt(ks & c10::after_autograd_keyset, grad_out_, input_, mean_, invstd_, weight, mean_dy_, mean_dy_xmu_, count_);
  })();
  auto result = std::move(_tmp);
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
  if (mean_dy__storage_saved.has_value())
    AT_ASSERT(mean_dy__storage_saved.value().is_alias_of(mean_dy_.storage()));
  if (mean_dy__impl_saved) AT_ASSERT(mean_dy__impl_saved == mean_dy_.getIntrusivePtr());
  if (mean_dy_xmu__storage_saved.has_value())
    AT_ASSERT(mean_dy_xmu__storage_saved.value().is_alias_of(mean_dy_xmu_.storage()));
  if (mean_dy_xmu__impl_saved) AT_ASSERT(mean_dy_xmu__impl_saved == mean_dy_xmu_.getIntrusivePtr());
  if (count__storage_saved.has_value())
    AT_ASSERT(count__storage_saved.value().is_alias_of(count_.storage()));
  if (count__impl_saved) AT_ASSERT(count__impl_saved == count_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "batch_norm_backward_elemt");
  return result;
}
std::tuple<Tensor,Tensor> batch_norm_gather_stats_with_counts(c10::DispatchKeySet ks, const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, double momentum, double eps, const Tensor & counts) {
  auto& input_ = unpack(input, "input", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  auto& invstd_ = unpack(invstd, "invstd", 2);
  auto& counts_ = unpack(counts, "counts", 7);
  auto _any_requires_grad = compute_requires_grad( input, mean, invstd, running_mean, running_var, counts );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_gather_stats_with_counts"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, mean, invstd, running_mean, running_var, counts ));
  }
  Tensor result0;
  Tensor result1;
  #ifndef NDEBUG
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
  c10::optional<Storage> counts__storage_saved =
    counts_.has_storage() ? c10::optional<Storage>(counts_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> counts__impl_saved;
  if (counts_.defined()) counts__impl_saved = counts_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::batch_norm_gather_stats_with_counts(ks & c10::after_autograd_keyset, input_, mean_, invstd_, running_mean, running_var, momentum, eps, counts_);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (invstd__storage_saved.has_value())
    AT_ASSERT(invstd__storage_saved.value().is_alias_of(invstd_.storage()));
  if (invstd__impl_saved) AT_ASSERT(invstd__impl_saved == invstd_.getIntrusivePtr());
  if (counts__storage_saved.has_value())
    AT_ASSERT(counts__storage_saved.value().is_alias_of(counts_.storage()));
  if (counts__impl_saved) AT_ASSERT(counts__impl_saved == counts_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "batch_norm_gather_stats_with_counts");
  throw_error_for_complex_autograd(result1, "batch_norm_gather_stats_with_counts");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor binary_cross_entropy(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_no_requires_grad(target, "target");
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<BinaryCrossEntropyBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<BinaryCrossEntropyBackward>(new BinaryCrossEntropyBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
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
    return at::redispatch::binary_cross_entropy(ks & c10::after_autograd_keyset, self_, target_, weight, reduction);
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
  throw_error_for_complex_autograd(result, "binary_cross_entropy");
  return result;
}
Tensor & binary_cross_entropy_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& grad_input_ = unpack(grad_input, "grad_input", 5);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, target, weight )) {
    throw_error_out_requires_grad("binary_cross_entropy_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("binary_cross_entropy_backward");
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
    at::redispatch::binary_cross_entropy_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, target_, weight, reduction, grad_input_);
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
Tensor & cholesky_inverse_out_out(c10::DispatchKeySet ks, const Tensor & self, bool upper, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cholesky_inverse");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cholesky_inverse");
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
    at::redispatch::cholesky_inverse_outf(ks & c10::after_autograd_keyset, self_, upper, out_);
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
Tensor cholesky_solve(c10::DispatchKeySet ks, const Tensor & self, const Tensor & input2, bool upper) {
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  auto _any_requires_grad = compute_requires_grad( self, input2 );
  (void)_any_requires_grad;
  std::shared_ptr<CholeskySolveBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CholeskySolveBackward>(new CholeskySolveBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, input2 ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->input2_ = SavedVariable(input2, false);
    grad_fn->upper = upper;
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::cholesky_solve(ks & c10::after_autograd_keyset, self_, input2_, upper);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (input2__storage_saved.has_value())
    AT_ASSERT(input2__storage_saved.value().is_alias_of(input2_.storage()));
  if (input2__impl_saved) AT_ASSERT(input2__impl_saved == input2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor clamp_min(c10::DispatchKeySet ks, const Tensor & self, const Scalar & min) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ClampMinBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ClampMinBackward>(new ClampMinBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min = min;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::clamp_min(ks & c10::after_autograd_keyset, self_, min);
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
  throw_error_for_complex_autograd(result, "clamp_min");
  return result;
}
Tensor & clamp_min_(c10::DispatchKeySet ks, Tensor & self, const Scalar & min) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<ClampMinBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ClampMinBackward>(new ClampMinBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->min = min;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::clamp_min_(ks & c10::after_autograd_keyset, self_, min);
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
Tensor & clamp_out_out(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & min, const c10::optional<Scalar> & max, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("clamp");
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
    at::redispatch::clamp_outf(ks & c10::after_autograd_keyset, self_, min, max, out_);
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
Tensor clone(c10::DispatchKeySet ks, const Tensor & self, c10::optional<MemoryFormat> memory_format) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<CloneBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CloneBackward>(new CloneBackward(), deleteNode);
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
    return at::redispatch::clone(ks & c10::after_autograd_keyset, self_, memory_format);
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
Tensor & complex_out_out(c10::DispatchKeySet ks, const Tensor & real, const Tensor & imag, Tensor & out) {
  auto& real_ = unpack(real, "real", 0);
  auto& imag_ = unpack(imag, "imag", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( real, imag );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( real, imag )) {
    throw_error_out_requires_grad("complex");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("complex");
  }
  #ifndef NDEBUG
  c10::optional<Storage> real__storage_saved =
    real_.has_storage() ? c10::optional<Storage>(real_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> real__impl_saved;
  if (real_.defined()) real__impl_saved = real_.getIntrusivePtr();
  c10::optional<Storage> imag__storage_saved =
    imag_.has_storage() ? c10::optional<Storage>(imag_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> imag__impl_saved;
  if (imag_.defined()) imag__impl_saved = imag_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::complex_outf(ks & c10::after_autograd_keyset, real_, imag_, out_);
  }
  #ifndef NDEBUG
  if (real__storage_saved.has_value())
    AT_ASSERT(real__storage_saved.value().is_alias_of(real_.storage()));
  if (real__impl_saved) AT_ASSERT(real__impl_saved == real_.getIntrusivePtr());
  if (imag__storage_saved.has_value())
    AT_ASSERT(imag__storage_saved.value().is_alias_of(imag_.storage()));
  if (imag__impl_saved) AT_ASSERT(imag__impl_saved == imag_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor convolution_overrideable(c10::DispatchKeySet ks, const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( input, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<ConvolutionOverrideableBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ConvolutionOverrideableBackward>(new ConvolutionOverrideableBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->transposed = transposed;
    grad_fn->output_padding = output_padding.vec();
    grad_fn->groups = groups;
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
    return at::redispatch::convolution_overrideable(ks & c10::after_autograd_keyset, input_, weight_, bias, stride, padding, dilation, transposed, output_padding, groups);
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
  throw_error_for_complex_autograd(result, "convolution_overrideable");
  return result;
}
Tensor & cos_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cos");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cos");
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
    at::redispatch::cos_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor cosh(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<CoshBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CoshBackward>(new CoshBackward(), deleteNode);
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
    return at::redispatch::cosh(ks & c10::after_autograd_keyset, self_);
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
Tensor & cosh_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<CoshBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CoshBackward>(new CoshBackward(), deleteNode);
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
    at::redispatch::cosh_(ks & c10::after_autograd_keyset, self_);
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
Tensor cross(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<CrossBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CrossBackward>(new CrossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    grad_fn->dim = dim;
    if (grad_fn->should_compute_output(0)) {
      grad_fn->other_ = SavedVariable(other, false);
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
    return at::redispatch::cross(ks & c10::after_autograd_keyset, self_, other_, dim);
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
  throw_error_for_complex_autograd(result, "cross");
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> cudnn_batch_norm(c10::DispatchKeySet ks, const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( input, weight, bias );
  (void)_any_requires_grad;
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<CudnnBatchNormBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CudnnBatchNormBackward>(new CudnnBatchNormBackward(), deleteNode);
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
  Tensor result3;
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
    return at::redispatch::cudnn_batch_norm(ks & c10::after_autograd_keyset, input_, weight_, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  })();
  std::tie(result0, result1, result2, result3) = std::move(_tmp);
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
    grad_fn->result3_ = SavedVariable(result3, true);
  }
  throw_error_for_complex_autograd(result0, "cudnn_batch_norm");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
Tensor cudnn_convolution_add_relu(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, const Tensor & z, const c10::optional<Scalar> & alpha, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& z_ = unpack(z, "z", 2);
  auto _any_requires_grad = compute_requires_grad( self, weight, z, bias );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_add_relu"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, z, bias ));
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
  c10::optional<Storage> z__storage_saved =
    z_.has_storage() ? c10::optional<Storage>(z_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> z__impl_saved;
  if (z_.defined()) z__impl_saved = z_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::cudnn_convolution_add_relu(ks & c10::after_autograd_keyset, self_, weight_, z_, alpha, bias, stride, padding, dilation, groups);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (z__storage_saved.has_value())
    AT_ASSERT(z__storage_saved.value().is_alias_of(z_.storage()));
  if (z__impl_saved) AT_ASSERT(z__impl_saved == z_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "cudnn_convolution_add_relu");
  return result;
}
std::tuple<Tensor,Tensor> cudnn_convolution_transpose_backward(c10::DispatchKeySet ks, const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask) {
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( self, grad_output, weight );
  (void)_any_requires_grad;
  std::shared_ptr<CudnnConvolutionTransposeBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CudnnConvolutionTransposeBackwardBackward>(new CudnnConvolutionTransposeBackwardBackward(), deleteNode);
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
    grad_fn->allow_tf32 = allow_tf32;
  }
  Tensor result0;
  Tensor result1;
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
    return at::redispatch::cudnn_convolution_transpose_backward(ks & c10::after_autograd_keyset, self_, grad_output_, weight_, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
  })();
  std::tie(result0, result1) = std::move(_tmp);
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
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "cudnn_convolution_transpose_backward");
  throw_error_for_complex_autograd(result1, "cudnn_convolution_transpose_backward");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor cudnn_convolution_transpose_backward_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, weight );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_transpose_backward_input"), deleteNode);
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
    return at::redispatch::cudnn_convolution_transpose_backward_input(ks & c10::after_autograd_keyset, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
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
  throw_error_for_complex_autograd(result, "cudnn_convolution_transpose_backward_input");
  return result;
}
Tensor cudnn_grid_sampler(c10::DispatchKeySet ks, const Tensor & self, const Tensor & grid) {
  auto& self_ = unpack(self, "self", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  auto _any_requires_grad = compute_requires_grad( self, grid );
  (void)_any_requires_grad;
  std::shared_ptr<CudnnGridSamplerBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CudnnGridSamplerBackward>(new CudnnGridSamplerBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, grid ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grid_ = SavedVariable(grid, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grid__storage_saved =
    grid_.has_storage() ? c10::optional<Storage>(grid_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grid__impl_saved;
  if (grid_.defined()) grid__impl_saved = grid_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::cudnn_grid_sampler(ks & c10::after_autograd_keyset, self_, grid_);
  })();
  auto output = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grid__storage_saved.has_value())
    AT_ASSERT(grid__storage_saved.value().is_alias_of(grid_.storage()));
  if (grid__impl_saved) AT_ASSERT(grid__impl_saved == grid_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  throw_error_for_complex_autograd(output, "cudnn_grid_sampler");
  return output;
}
std::tuple<Tensor,Tensor> cummin(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<CumminBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CumminBackward>(new CumminBackward(), deleteNode);
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
    return at::redispatch::cummin(ks & c10::after_autograd_keyset, self_, dim);
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
  throw_error_for_complex_autograd(values, "cummin");
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor & deg2rad_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("deg2rad");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("deg2rad");
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
    at::redispatch::deg2rad_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor & diag_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t diagonal, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("diag");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("diag");
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
    at::redispatch::diag_outf(ks & c10::after_autograd_keyset, self_, diagonal, out_);
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
Tensor & digamma_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("digamma");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("digamma");
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
    at::redispatch::digamma_outf(ks & c10::after_autograd_keyset, self_, out_);
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
std::tuple<Tensor,Tensor> eig(c10::DispatchKeySet ks, const Tensor & self, bool eigenvectors) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<EigBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<EigBackward>(new EigBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->eigenvectors = eigenvectors;
  }
  Tensor eigenvalues;
  Tensor eigenvectors_return;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::eig(ks & c10::after_autograd_keyset, self_, eigenvectors);
  })();
  std::tie(eigenvalues, eigenvectors_return) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( eigenvalues, eigenvectors_return ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->eigenvalues_ = SavedVariable(eigenvalues, true);
    grad_fn->eigenvectors_return_ = SavedVariable(eigenvectors_return, true);
  }
  return std::make_tuple(std::move(eigenvalues), std::move(eigenvectors_return));
}
Tensor empty_memory_format(c10::DispatchKeySet ks, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::empty(ks & c10::after_autograd_keyset, size, dtype, layout, device, pin_memory, memory_format);
  })();
  auto result = std::move(_tmp);
  return result;
}
Tensor eq_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::eq(ks & c10::after_autograd_keyset, self_, other);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor eq_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
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
    return at::redispatch::eq(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor & eq__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<EqBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<EqBackward0>(new EqBackward0(), deleteNode);
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
    at::redispatch::eq_(ks & c10::after_autograd_keyset, self_, other);
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
Tensor & eq__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<EqBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<EqBackward1>(new EqBackward1(), deleteNode);
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
    at::redispatch::eq_(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor & erfinv_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("erfinv");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("erfinv");
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
    at::redispatch::erfinv_outf(ks & c10::after_autograd_keyset, self_, out_);
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
std::tuple<Tensor,Tensor> fake_quantize_per_channel_affine_cachemask(c10::DispatchKeySet ks, const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  auto& self_ = unpack(self, "self", 0);
  auto& scale_ = unpack(scale, "scale", 1);
  auto& zero_point_ = unpack(zero_point, "zero_point", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_no_requires_grad(scale, "scale");
  check_no_requires_grad(zero_point, "zero_point");
  std::shared_ptr<FakeQuantizePerChannelAffineCachemaskBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FakeQuantizePerChannelAffineCachemaskBackward>(new FakeQuantizePerChannelAffineCachemaskBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor output;
  Tensor mask;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> scale__storage_saved =
    scale_.has_storage() ? c10::optional<Storage>(scale_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> scale__impl_saved;
  if (scale_.defined()) scale__impl_saved = scale_.getIntrusivePtr();
  c10::optional<Storage> zero_point__storage_saved =
    zero_point_.has_storage() ? c10::optional<Storage>(zero_point_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> zero_point__impl_saved;
  if (zero_point_.defined()) zero_point__impl_saved = zero_point_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::fake_quantize_per_channel_affine_cachemask(ks & c10::after_autograd_keyset, self_, scale_, zero_point_, axis, quant_min, quant_max);
  })();
  std::tie(output, mask) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (scale__storage_saved.has_value())
    AT_ASSERT(scale__storage_saved.value().is_alias_of(scale_.storage()));
  if (scale__impl_saved) AT_ASSERT(scale__impl_saved == scale_.getIntrusivePtr());
  if (zero_point__storage_saved.has_value())
    AT_ASSERT(zero_point__storage_saved.value().is_alias_of(zero_point_.storage()));
  if (zero_point__impl_saved) AT_ASSERT(zero_point__impl_saved == zero_point_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->mask_ = SavedVariable(mask, true);
  }
  throw_error_for_complex_autograd(output, "fake_quantize_per_channel_affine_cachemask");
  return std::make_tuple(std::move(output), std::move(mask));
}
Tensor & floor_divide_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("floor_divide");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("floor_divide");
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
    at::redispatch::floor_divide_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
Tensor fractional_max_pool3d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 4);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<FractionalMaxPool3DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FractionalMaxPool3DBackwardBackward>(new FractionalMaxPool3DBackwardBackward(), deleteNode);
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
    return at::redispatch::fractional_max_pool3d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, kernel_size, output_size, indices_);
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
  throw_error_for_complex_autograd(result, "fractional_max_pool3d_backward");
  return result;
}
std::tuple<Tensor,Tensor> frexp_Tensor(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<FrexpBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FrexpBackward>(new FrexpBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor mantissa;
  Tensor exponent;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::frexp(ks & c10::after_autograd_keyset, self_);
  })();
  std::tie(mantissa, exponent) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( mantissa ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->exponent_ = SavedVariable(exponent, true);
  }
  throw_error_for_complex_autograd(mantissa, "frexp");
  return std::make_tuple(std::move(mantissa), std::move(exponent));
}
Tensor & gather_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("gather");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("gather");
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
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::gather_outf(ks & c10::after_autograd_keyset, self_, dim, index_, sparse_grad, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor ge_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::ge(ks & c10::after_autograd_keyset, self_, other);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor ge_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
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
    return at::redispatch::ge(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor & ge__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<GeBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GeBackward0>(new GeBackward0(), deleteNode);
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
    at::redispatch::ge_(ks & c10::after_autograd_keyset, self_, other);
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
Tensor & ge__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<GeBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GeBackward1>(new GeBackward1(), deleteNode);
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
    at::redispatch::ge_(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor gelu(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<GeluBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GeluBackward>(new GeluBackward(), deleteNode);
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
    return at::redispatch::gelu(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "gelu");
  return result;
}
Tensor & geometric_(c10::DispatchKeySet ks, Tensor & self, double p, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<GeometricBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GeometricBackward>(new GeometricBackward(), deleteNode);
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
    at::redispatch::geometric_(ks & c10::after_autograd_keyset, self_, p, generator);
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
Tensor glu_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, int64_t dim) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<GluBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GluBackwardBackward>(new GluBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::glu_backward(ks & c10::after_autograd_keyset, grad_output_, self_, dim);
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
  throw_error_for_complex_autograd(result, "glu_backward");
  return result;
}
std::tuple<Tensor,Tensor> grid_sampler_2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& grid_ = unpack(grid, "grid", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, input, grid );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("grid_sampler_2d_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, input, grid ));
  }
  Tensor result0;
  Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> grid__storage_saved =
    grid_.has_storage() ? c10::optional<Storage>(grid_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grid__impl_saved;
  if (grid_.defined()) grid__impl_saved = grid_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::grid_sampler_2d_backward(ks & c10::after_autograd_keyset, grad_output_, input_, grid_, interpolation_mode, padding_mode, align_corners);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (grid__storage_saved.has_value())
    AT_ASSERT(grid__storage_saved.value().is_alias_of(grid_.storage()));
  if (grid__impl_saved) AT_ASSERT(grid__impl_saved == grid_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "grid_sampler_2d_backward");
  throw_error_for_complex_autograd(result1, "grid_sampler_2d_backward");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor grid_sampler_3d(c10::DispatchKeySet ks, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  auto& input_ = unpack(input, "input", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  auto _any_requires_grad = compute_requires_grad( input, grid );
  (void)_any_requires_grad;
  std::shared_ptr<GridSampler3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GridSampler3DBackward>(new GridSampler3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, grid ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->grid_ = SavedVariable(grid, false);
    grad_fn->interpolation_mode = interpolation_mode;
    grad_fn->padding_mode = padding_mode;
    grad_fn->align_corners = align_corners;
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> grid__storage_saved =
    grid_.has_storage() ? c10::optional<Storage>(grid_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grid__impl_saved;
  if (grid_.defined()) grid__impl_saved = grid_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::grid_sampler_3d(ks & c10::after_autograd_keyset, input_, grid_, interpolation_mode, padding_mode, align_corners);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (grid__storage_saved.has_value())
    AT_ASSERT(grid__storage_saved.value().is_alias_of(grid_.storage()));
  if (grid__impl_saved) AT_ASSERT(grid__impl_saved == grid_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "grid_sampler_3d");
  return result;
}
Tensor gt_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::gt(ks & c10::after_autograd_keyset, self_, other);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor gt_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
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
    return at::redispatch::gt(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor & gt__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<GtBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GtBackward0>(new GtBackward0(), deleteNode);
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
    at::redispatch::gt_(ks & c10::after_autograd_keyset, self_, other);
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
Tensor & gt__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<GtBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GtBackward1>(new GtBackward1(), deleteNode);
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
    at::redispatch::gt_(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor hardsigmoid_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("hardsigmoid_backward"), deleteNode);
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
    return at::redispatch::hardsigmoid_backward(ks & c10::after_autograd_keyset, grad_output_, self_);
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
  throw_error_for_complex_autograd(result, "hardsigmoid_backward");
  return result;
}
Tensor hardswish_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("hardswish_backward"), deleteNode);
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
    return at::redispatch::hardswish_backward(ks & c10::after_autograd_keyset, grad_output_, self_);
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
  throw_error_for_complex_autograd(result, "hardswish_backward");
  return result;
}
Tensor histc(c10::DispatchKeySet ks, const Tensor & self, int64_t bins, const Scalar & min, const Scalar & max) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<HistcBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HistcBackward>(new HistcBackward(), deleteNode);
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
    return at::redispatch::histc(ks & c10::after_autograd_keyset, self_, bins, min, max);
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
  throw_error_for_complex_autograd(result, "histc");
  return result;
}
Tensor hspmm(c10::DispatchKeySet ks, const Tensor & mat1, const Tensor & mat2) {
  auto& mat1_ = unpack(mat1, "mat1", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  auto _any_requires_grad = compute_requires_grad( mat1, mat2 );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("hspmm"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( mat1, mat2 ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> mat1__storage_saved =
    mat1_.has_storage() ? c10::optional<Storage>(mat1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat1__impl_saved;
  if (mat1_.defined()) mat1__impl_saved = mat1_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::hspmm(ks & c10::after_autograd_keyset, mat1_, mat2_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (mat1__storage_saved.has_value())
    AT_ASSERT(mat1__storage_saved.value().is_alias_of(mat1_.storage()));
  if (mat1__impl_saved) AT_ASSERT(mat1__impl_saved == mat1_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "hspmm");
  return result;
}
Tensor huber_loss_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double delta) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, target );
  (void)_any_requires_grad;
  std::shared_ptr<HuberLossBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HuberLossBackwardBackward>(new HuberLossBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
    grad_fn->delta = delta;
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::huber_loss_backward(ks & c10::after_autograd_keyset, grad_output_, self_, target_, reduction, delta);
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
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "huber_loss_backward");
  return result;
}
Tensor hypot(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<HypotBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HypotBackward>(new HypotBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
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
    return at::redispatch::hypot(ks & c10::after_autograd_keyset, self_, other_);
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
  throw_error_for_complex_autograd(result, "hypot");
  return result;
}
Tensor & hypot_(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<HypotBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HypotBackward>(new HypotBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
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
    at::redispatch::hypot_(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor & igamma_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("igamma");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("igamma");
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
    at::redispatch::igamma_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
Tensor igammac(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<IgammacBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<IgammacBackward>(new IgammacBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    if (grad_fn->should_compute_output(1)) {
      grad_fn->other_ = SavedVariable(other, false);
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
    return at::redispatch::igammac(ks & c10::after_autograd_keyset, self_, other_);
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
  throw_error_for_complex_autograd(result, "igammac");
  return result;
}
Tensor & igammac_(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<IgammacBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<IgammacBackward>(new IgammacBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self.clone(), false);
    }
    if (grad_fn->should_compute_output(1)) {
      grad_fn->other_ = SavedVariable(other, false);
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
    at::redispatch::igammac_(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor & index_copy_(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  auto _any_requires_grad = compute_requires_grad( self, source );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<IndexCopyBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<IndexCopyBackward>(new IndexCopyBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
    grad_fn->source_dim = source.dim();
    if (grad_fn->should_compute_output(1)) {
      grad_fn->source_ = SavedVariable(source, false);
    }
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
  c10::optional<Storage> source__storage_saved =
    source_.has_storage() ? c10::optional<Storage>(source_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> source__impl_saved;
  if (source_.defined()) source__impl_saved = source_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::index_copy_(ks & c10::after_autograd_keyset, self_, dim, index_, source_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (source__storage_saved.has_value())
    AT_ASSERT(source__storage_saved.value().is_alias_of(source_.storage()));
  if (source__impl_saved) AT_ASSERT(source__impl_saved == source_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & index_fill__int_Scalar(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Scalar & value) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<IndexFillBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<IndexFillBackward0>(new IndexFillBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
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
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::index_fill_(ks & c10::after_autograd_keyset, self_, dim, index_, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & index_fill__int_Tensor(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& value_ = unpack(value, "value", 3);
  auto _any_requires_grad = compute_requires_grad( self, value );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<IndexFillBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<IndexFillBackward1>(new IndexFillBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
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
  c10::optional<Storage> value__storage_saved =
    value_.has_storage() ? c10::optional<Storage>(value_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::index_fill_(ks & c10::after_autograd_keyset, self_, dim, index_, value_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (value__storage_saved.has_value())
    AT_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved) AT_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor inverse(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<InverseBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<InverseBackward>(new InverseBackward(), deleteNode);
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
    return at::redispatch::inverse(ks & c10::after_autograd_keyset, self_);
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
bool is_set_to(c10::DispatchKeySet ks, const Tensor & self, const Tensor & tensor) {
  auto& self_ = unpack(self, "self", 0);
  auto& tensor_ = unpack(tensor, "tensor", 1);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor__storage_saved =
    tensor_.has_storage() ? c10::optional<Storage>(tensor_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor__impl_saved;
  if (tensor_.defined()) tensor__impl_saved = tensor_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::is_set_to(ks & c10::after_autograd_keyset, self_, tensor_);
  })();
  auto result = _tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor__storage_saved.has_value())
    AT_ASSERT(tensor__storage_saved.value().is_alias_of(tensor_.storage()));
  if (tensor__impl_saved) AT_ASSERT(tensor__impl_saved == tensor_.getIntrusivePtr());
  #endif
  return result;
}
Tensor & isposinf_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
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
    at::redispatch::isposinf_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor kl_div_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, target );
  (void)_any_requires_grad;
  std::shared_ptr<KlDivBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<KlDivBackwardBackward>(new KlDivBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
    grad_fn->log_target = log_target;
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::kl_div_backward(ks & c10::after_autograd_keyset, grad_output_, self_, target_, reduction, log_target);
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
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "kl_div_backward");
  return result;
}
Tensor le_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::le(ks & c10::after_autograd_keyset, self_, other);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor le_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
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
    return at::redispatch::le(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor & le__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<LeBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LeBackward0>(new LeBackward0(), deleteNode);
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
    at::redispatch::le_(ks & c10::after_autograd_keyset, self_, other);
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
Tensor & le__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<LeBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LeBackward1>(new LeBackward1(), deleteNode);
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
    at::redispatch::le_(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor leaky_relu(c10::DispatchKeySet ks, const Tensor & self, const Scalar & negative_slope) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LeakyReluBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LeakyReluBackward0>(new LeakyReluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->negative_slope = negative_slope;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::leaky_relu(ks & c10::after_autograd_keyset, self_, negative_slope);
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
  throw_error_for_complex_autograd(result, "leaky_relu");
  return result;
}
Tensor & leaky_relu_(c10::DispatchKeySet ks, Tensor & self, const Scalar & negative_slope) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<LeakyReluBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LeakyReluBackward1>(new LeakyReluBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->negative_slope = negative_slope;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::leaky_relu_(ks & c10::after_autograd_keyset, self_, negative_slope);
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
Tensor & lerp_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & end, const Scalar & weight, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self, end );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, end )) {
    throw_error_out_requires_grad("lerp");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("lerp");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> end__storage_saved =
    end_.has_storage() ? c10::optional<Storage>(end_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> end__impl_saved;
  if (end_.defined()) end__impl_saved = end_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::lerp_outf(ks & c10::after_autograd_keyset, self_, end_, weight, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (end__storage_saved.has_value())
    AT_ASSERT(end__storage_saved.value().is_alias_of(end_.storage()));
  if (end__impl_saved) AT_ASSERT(end__impl_saved == end_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & lerp_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & end, const Tensor & weight, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self, end, weight );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, end, weight )) {
    throw_error_out_requires_grad("lerp");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("lerp");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> end__storage_saved =
    end_.has_storage() ? c10::optional<Storage>(end_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> end__impl_saved;
  if (end_.defined()) end__impl_saved = end_.getIntrusivePtr();
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
    at::redispatch::lerp_outf(ks & c10::after_autograd_keyset, self_, end_, weight_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (end__storage_saved.has_value())
    AT_ASSERT(end__storage_saved.value().is_alias_of(end_.storage()));
  if (end__impl_saved) AT_ASSERT(end__impl_saved == end_.getIntrusivePtr());
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
std::tuple<Tensor,Tensor> linalg_eigh(c10::DispatchKeySet ks, const Tensor & self, std::string UPLO) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LinalgEighBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LinalgEighBackward>(new LinalgEighBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor eigenvalues;
  Tensor eigenvectors;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::linalg_eigh(ks & c10::after_autograd_keyset, self_, UPLO);
  })();
  std::tie(eigenvalues, eigenvectors) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( eigenvalues, eigenvectors ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->eigenvalues_ = SavedVariable(eigenvalues, true);
    grad_fn->eigenvectors_ = SavedVariable(eigenvectors, true);
  }
  return std::make_tuple(std::move(eigenvalues), std::move(eigenvectors));
}
Tensor linalg_inv(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LinalgInvBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LinalgInvBackward>(new LinalgInvBackward(), deleteNode);
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
    return at::redispatch::linalg_inv(ks & c10::after_autograd_keyset, self_);
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
std::tuple<Tensor &,Tensor &> linalg_slogdet_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & sign, Tensor & logabsdet) {
  auto& self_ = unpack(self, "self", 0);
  auto& sign_ = unpack(sign, "sign", 1);
  auto& logabsdet_ = unpack(logabsdet, "logabsdet", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("linalg_slogdet");
  }
  if (compute_requires_grad( logabsdet )) {
    throw_error_out_requires_grad("linalg_slogdet");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> sign__storage_saved =
    sign_.has_storage() ? c10::optional<Storage>(sign_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> sign__impl_saved;
  if (sign_.defined()) sign__impl_saved = sign_.getIntrusivePtr();
  c10::optional<Storage> logabsdet__storage_saved =
    logabsdet_.has_storage() ? c10::optional<Storage>(logabsdet_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> logabsdet__impl_saved;
  if (logabsdet_.defined()) logabsdet__impl_saved = logabsdet_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::linalg_slogdet_outf(ks & c10::after_autograd_keyset, self_, sign_, logabsdet_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (sign__storage_saved.has_value())
    AT_ASSERT(sign__storage_saved.value().is_alias_of(sign_.storage()));
  if (sign__impl_saved) AT_ASSERT(sign__impl_saved == sign_.getIntrusivePtr());
  if (logabsdet__storage_saved.has_value())
    AT_ASSERT(logabsdet__storage_saved.value().is_alias_of(logabsdet_.storage()));
  if (logabsdet__impl_saved) AT_ASSERT(logabsdet__impl_saved == logabsdet_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( logabsdet ), grad_fn);
  }
  return std::forward_as_tuple(sign, logabsdet);
}
Tensor linalg_solve(c10::DispatchKeySet ks, const Tensor & input, const Tensor & other) {
  auto& input_ = unpack(input, "input", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( input, other );
  (void)_any_requires_grad;
  std::shared_ptr<LinalgSolveBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LinalgSolveBackward>(new LinalgSolveBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, other ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::linalg_solve(ks & c10::after_autograd_keyset, input_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
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
  return result;
}
Tensor linalg_vector_norm(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LinalgVectorNormBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LinalgVectorNormBackward>(new LinalgVectorNormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->ord = ord;
    grad_fn->dim = dim;
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
    return at::redispatch::linalg_vector_norm(ks & c10::after_autograd_keyset, self_, ord, dim, keepdim, dtype);
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
Tensor log10(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Log10Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Log10Backward>(new Log10Backward(), deleteNode);
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
    return at::redispatch::log10(ks & c10::after_autograd_keyset, self_);
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
Tensor & log10_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<Log10Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Log10Backward>(new Log10Backward(), deleteNode);
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
    at::redispatch::log10_(ks & c10::after_autograd_keyset, self_);
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
Tensor logdet(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LogdetBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogdetBackward>(new LogdetBackward(), deleteNode);
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
    return at::redispatch::logdet(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "logdet");
  return result;
}
Tensor logit_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, c10::optional<double> eps) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("logit_backward"), deleteNode);
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
    return at::redispatch::logit_backward(ks & c10::after_autograd_keyset, grad_output_, self_, eps);
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
  throw_error_for_complex_autograd(result, "logit_backward");
  return result;
}
Tensor lt_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::lt(ks & c10::after_autograd_keyset, self_, other);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor lt_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
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
    return at::redispatch::lt(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor & lt__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<LtBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LtBackward0>(new LtBackward0(), deleteNode);
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
    at::redispatch::lt_(ks & c10::after_autograd_keyset, self_, other);
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
Tensor & lt__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<LtBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LtBackward1>(new LtBackward1(), deleteNode);
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
    at::redispatch::lt_(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor masked_select(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mask) {
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MaskedSelectBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaskedSelectBackward>(new MaskedSelectBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::masked_select(ks & c10::after_autograd_keyset, self_, mask_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor matrix_exp(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MatrixExpBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MatrixExpBackward>(new MatrixExpBackward(), deleteNode);
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
    return at::redispatch::matrix_exp(ks & c10::after_autograd_keyset, self_);
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
Tensor max_pool3d_with_indices_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<MaxPool3DWithIndicesBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaxPool3DWithIndicesBackwardBackward>(new MaxPool3DWithIndicesBackwardBackward(), deleteNode);
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
    return at::redispatch::max_pool3d_with_indices_backward(ks & c10::after_autograd_keyset, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
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
  throw_error_for_complex_autograd(result, "max_pool3d_with_indices_backward");
  return result;
}
Tensor & max_unpool2d_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & indices, IntArrayRef output_size, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_unpool2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("max_unpool2d");
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
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::max_unpool2d_outf(ks & c10::after_autograd_keyset, self_, indices_, output_size, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
std::tuple<Tensor,Tensor> min_dim(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MinBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MinBackward0>(new MinBackward0(), deleteNode);
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
    return at::redispatch::min(ks & c10::after_autograd_keyset, self_, dim, keepdim);
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
  throw_error_for_complex_autograd(values, "min");
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor min(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MinBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MinBackward1>(new MinBackward1(), deleteNode);
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
    return at::redispatch::min(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "min");
  return result;
}
Tensor minimum(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<MinimumBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MinimumBackward>(new MinimumBackward(), deleteNode);
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
    return at::redispatch::minimum(ks & c10::after_autograd_keyset, self_, other_);
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
  throw_error_for_complex_autograd(result, "minimum");
  return result;
}
Tensor miopen_convolution(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<MiopenConvolutionBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MiopenConvolutionBackward>(new MiopenConvolutionBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::miopen_convolution(ks & c10::after_autograd_keyset, self_, weight_, bias, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "miopen_convolution");
  return result;
}
Tensor miopen_convolution_transpose_backward_weight(c10::DispatchKeySet ks, IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_transpose_backward_weight"), deleteNode);
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
    return at::redispatch::miopen_convolution_transpose_backward_weight(ks & c10::after_autograd_keyset, weight_size, grad_output_, self_, padding, stride, dilation, groups, benchmark, deterministic);
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
  throw_error_for_complex_autograd(result, "miopen_convolution_transpose_backward_weight");
  return result;
}
Tensor mkldnn_linear(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<MkldnnLinearBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MkldnnLinearBackward>(new MkldnnLinearBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::mkldnn_linear(ks & c10::after_autograd_keyset, self_, weight_, bias);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "mkldnn_linear");
  return result;
}
Tensor mkldnn_max_pool3d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & output, const Tensor & input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  auto& input_ = unpack(input, "input", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, output, input );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_max_pool3d_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output, input ));
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
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::mkldnn_max_pool3d_backward(ks & c10::after_autograd_keyset, grad_output_, output_, input_, kernel_size, stride, padding, dilation, ceil_mode);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "mkldnn_max_pool3d_backward");
  return result;
}
Tensor mse_loss(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self, target );
  (void)_any_requires_grad;
  std::shared_ptr<MseLossBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MseLossBackward>(new MseLossBackward(), deleteNode);
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
    return at::redispatch::mse_loss(ks & c10::after_autograd_keyset, self_, target_, reduction);
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
  throw_error_for_complex_autograd(result, "mse_loss");
  return result;
}
Tensor & mse_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& grad_input_ = unpack(grad_input, "grad_input", 4);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, target );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("mse_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("mse_loss_backward");
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
    at::redispatch::mse_loss_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, target_, reduction, grad_input_);
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
Tensor mul_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<MulBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    grad_fn->other_scalar_type = other.scalar_type();
    grad_fn->self_scalar_type = self.scalar_type();
    if (grad_fn->should_compute_output(0)) {
      grad_fn->other_ = SavedVariable(other, false);
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
    return at::redispatch::mul(ks & c10::after_autograd_keyset, self_, other_);
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
  return result;
}
Tensor mul_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MulBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MulBackward1>(new MulBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->other = other;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::mul(ks & c10::after_autograd_keyset, self_, other);
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
Tensor & mul__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<MulBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self.clone(), false);
    }
    grad_fn->other_scalar_type = other.scalar_type();
    grad_fn->self_scalar_type = self.scalar_type();
    if (grad_fn->should_compute_output(0)) {
      grad_fn->other_ = SavedVariable(other, false);
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
    at::redispatch::mul_(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor & mul__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<MulBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MulBackward1>(new MulBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->other = other;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::mul_(ks & c10::after_autograd_keyset, self_, other);
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
Tensor & multilabel_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& is_target_ = unpack(is_target, "is_target", 4);
  auto& grad_input_ = unpack(grad_input, "grad_input", 5);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, target, is_target );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, target, is_target )) {
    throw_error_out_requires_grad("multilabel_margin_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("multilabel_margin_loss_backward");
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
  c10::optional<Storage> is_target__storage_saved =
    is_target_.has_storage() ? c10::optional<Storage>(is_target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> is_target__impl_saved;
  if (is_target_.defined()) is_target__impl_saved = is_target_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::multilabel_margin_loss_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, target_, reduction, is_target_, grad_input_);
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
  if (is_target__storage_saved.has_value())
    AT_ASSERT(is_target__storage_saved.value().is_alias_of(is_target_.storage()));
  if (is_target__impl_saved) AT_ASSERT(is_target__impl_saved == is_target_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  return grad_input;
}
Tensor & multinomial_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t num_samples, bool replacement, c10::optional<Generator> generator, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
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
    at::redispatch::multinomial_outf(ks & c10::after_autograd_keyset, self_, num_samples, replacement, generator, out_);
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
Tensor & nan_to_num_out_out(c10::DispatchKeySet ks, const Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("nan_to_num");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("nan_to_num");
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
    at::redispatch::nan_to_num_outf(ks & c10::after_autograd_keyset, self_, nan, posinf, neginf, out_);
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
Tensor nanmedian(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NanmedianBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NanmedianBackward0>(new NanmedianBackward0(), deleteNode);
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
    return at::redispatch::nanmedian(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "nanmedian");
  return result;
}
std::tuple<Tensor,Tensor> nanmedian_dim(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NanmedianBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NanmedianBackward1>(new NanmedianBackward1(), deleteNode);
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
    return at::redispatch::nanmedian(ks & c10::after_autograd_keyset, self_, dim, keepdim);
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
  throw_error_for_complex_autograd(values, "nanmedian");
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor &,Tensor &,Tensor &> native_batch_norm_out_out(c10::DispatchKeySet ks, const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double momentum, double eps, Tensor & out, Tensor & save_mean, Tensor & save_invstd) {
  auto& input_ = unpack(input, "input", 0);
  auto& out_ = unpack(out, "out", 8);
  auto& save_mean_ = unpack(save_mean, "save_mean", 9);
  auto& save_invstd_ = unpack(save_invstd, "save_invstd", 10);
  auto _any_requires_grad = compute_requires_grad( input, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( input, weight, bias, running_mean, running_var )) {
    throw_error_out_requires_grad("native_batch_norm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("native_batch_norm");
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> save_mean__storage_saved =
    save_mean_.has_storage() ? c10::optional<Storage>(save_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> save_mean__impl_saved;
  if (save_mean_.defined()) save_mean__impl_saved = save_mean_.getIntrusivePtr();
  c10::optional<Storage> save_invstd__storage_saved =
    save_invstd_.has_storage() ? c10::optional<Storage>(save_invstd_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> save_invstd__impl_saved;
  if (save_invstd_.defined()) save_invstd__impl_saved = save_invstd_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::native_batch_norm_outf(ks & c10::after_autograd_keyset, input_, weight, bias, running_mean, running_var, training, momentum, eps, out_, save_mean_, save_invstd_);
  }
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (save_mean__storage_saved.has_value())
    AT_ASSERT(save_mean__storage_saved.value().is_alias_of(save_mean_.storage()));
  if (save_mean__impl_saved) AT_ASSERT(save_mean__impl_saved == save_mean_.getIntrusivePtr());
  if (save_invstd__storage_saved.has_value())
    AT_ASSERT(save_invstd__storage_saved.value().is_alias_of(save_invstd_.storage()));
  if (save_invstd__impl_saved) AT_ASSERT(save_invstd__impl_saved == save_invstd_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return std::forward_as_tuple(out, save_mean, save_invstd);
}
std::tuple<Tensor,Tensor,Tensor> native_group_norm_backward(c10::DispatchKeySet ks, const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & rstd, const c10::optional<Tensor> & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, std::array<bool,3> output_mask) {
  auto& grad_out_ = unpack(grad_out, "grad_out", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& mean_ = unpack(mean, "mean", 2);
  auto& rstd_ = unpack(rstd, "rstd", 3);
  auto _any_requires_grad = compute_requires_grad( grad_out, input, mean, rstd, weight );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("native_group_norm_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_out, input, mean, rstd, weight ));
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
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> rstd__storage_saved =
    rstd_.has_storage() ? c10::optional<Storage>(rstd_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> rstd__impl_saved;
  if (rstd_.defined()) rstd__impl_saved = rstd_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::native_group_norm_backward(ks & c10::after_autograd_keyset, grad_out_, input_, mean_, rstd_, weight, N, C, HxW, group, output_mask);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
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
  if (rstd__storage_saved.has_value())
    AT_ASSERT(rstd__storage_saved.value().is_alias_of(rstd_.storage()));
  if (rstd__impl_saved) AT_ASSERT(rstd__impl_saved == rstd_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "native_group_norm_backward");
  throw_error_for_complex_autograd(result1, "native_group_norm_backward");
  throw_error_for_complex_autograd(result2, "native_group_norm_backward");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor neg(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NegBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NegBackward>(new NegBackward(), deleteNode);
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
    return at::redispatch::neg(ks & c10::after_autograd_keyset, self_);
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
Tensor & neg_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NegBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NegBackward>(new NegBackward(), deleteNode);
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
    at::redispatch::neg_(ks & c10::after_autograd_keyset, self_);
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
Tensor nonzero(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::nonzero(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor norm_ScalarOpt_dtype(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & p, ScalarType dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NormBackward2> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NormBackward2>(new NormBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::norm(ks & c10::after_autograd_keyset, self_, p, dtype);
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
  throw_error_for_complex_autograd(result, "norm");
  return result;
}
Tensor norm_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & p) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NormBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NormBackward0>(new NormBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::norm(ks & c10::after_autograd_keyset, self_, p);
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
  throw_error_for_complex_autograd(result, "norm");
  return result;
}
Tensor norm_ScalarOpt_dim_dtype(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NormBackward3> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NormBackward3>(new NormBackward3(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
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
    return at::redispatch::norm(ks & c10::after_autograd_keyset, self_, p, dim, keepdim, dtype);
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
  throw_error_for_complex_autograd(result, "norm");
  return result;
}
Tensor norm_ScalarOpt_dim(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NormBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NormBackward1>(new NormBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
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
    return at::redispatch::norm(ks & c10::after_autograd_keyset, self_, p, dim, keepdim);
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
  throw_error_for_complex_autograd(result, "norm");
  return result;
}
Tensor normal_Tensor_float(c10::DispatchKeySet ks, const Tensor & mean, double std, c10::optional<Generator> generator) {
  auto& mean_ = unpack(mean, "mean", 0);
  auto _any_requires_grad = compute_requires_grad( mean );
  (void)_any_requires_grad;
  std::shared_ptr<NormalBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NormalBackward1>(new NormalBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( mean ));
    grad_fn->mean_sizes = mean.sizes().vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::normal(ks & c10::after_autograd_keyset, mean_, std, generator);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "normal");
  return result;
}
Tensor normal_float_Tensor(c10::DispatchKeySet ks, double mean, const Tensor & std, c10::optional<Generator> generator) {
  auto& std_ = unpack(std, "std", 1);
  auto _any_requires_grad = compute_requires_grad( std );
  (void)_any_requires_grad;
  std::shared_ptr<NormalBackward2> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NormalBackward2>(new NormalBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( std ));
    grad_fn->std_sizes = std.sizes().vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> std__storage_saved =
    std_.has_storage() ? c10::optional<Storage>(std_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> std__impl_saved;
  if (std_.defined()) std__impl_saved = std_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::normal(ks & c10::after_autograd_keyset, mean, std_, generator);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (std__storage_saved.has_value())
    AT_ASSERT(std__storage_saved.value().is_alias_of(std_.storage()));
  if (std__impl_saved) AT_ASSERT(std__impl_saved == std_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "normal");
  return result;
}
Tensor normal_Tensor_Tensor(c10::DispatchKeySet ks, const Tensor & mean, const Tensor & std, c10::optional<Generator> generator) {
  auto& mean_ = unpack(mean, "mean", 0);
  auto& std_ = unpack(std, "std", 1);
  auto _any_requires_grad = compute_requires_grad( mean, std );
  (void)_any_requires_grad;
  std::shared_ptr<NormalBackward3> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NormalBackward3>(new NormalBackward3(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( mean, std ));
    grad_fn->mean_sizes = mean.sizes().vec();
    grad_fn->std_sizes = std.sizes().vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> std__storage_saved =
    std_.has_storage() ? c10::optional<Storage>(std_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> std__impl_saved;
  if (std_.defined()) std__impl_saved = std_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::normal(ks & c10::after_autograd_keyset, mean_, std_, generator);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (std__storage_saved.has_value())
    AT_ASSERT(std__storage_saved.value().is_alias_of(std_.storage()));
  if (std__impl_saved) AT_ASSERT(std__impl_saved == std_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "normal");
  return result;
}
Tensor & normal_(c10::DispatchKeySet ks, Tensor & self, double mean, double std, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NormalBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NormalBackward0>(new NormalBackward0(), deleteNode);
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
    at::redispatch::normal_(ks & c10::after_autograd_keyset, self_, mean, std, generator);
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
Tensor poisson(c10::DispatchKeySet ks, const Tensor & self, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<PoissonBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PoissonBackward>(new PoissonBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::poisson(ks & c10::after_autograd_keyset, self_, generator);
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
  throw_error_for_complex_autograd(result, "poisson");
  return result;
}
Tensor q_per_channel_scales(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("q_per_channel_scales"), deleteNode);
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
    return at::redispatch::q_per_channel_scales(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "q_per_channel_scales");
  return result;
}
Tensor quantized_batch_norm(c10::DispatchKeySet ks, const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const Tensor & mean, const Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
  auto& input_ = unpack(input, "input", 0);
  auto& mean_ = unpack(mean, "mean", 3);
  auto& var_ = unpack(var, "var", 4);
  auto _any_requires_grad = compute_requires_grad( input, weight, bias, mean, var );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("quantized_batch_norm"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias, mean, var ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> var__storage_saved =
    var_.has_storage() ? c10::optional<Storage>(var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> var__impl_saved;
  if (var_.defined()) var__impl_saved = var_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::quantized_batch_norm(ks & c10::after_autograd_keyset, input_, weight, bias, mean_, var_, eps, output_scale, output_zero_point);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (var__storage_saved.has_value())
    AT_ASSERT(var__storage_saved.value().is_alias_of(var_.storage()));
  if (var__impl_saved) AT_ASSERT(var__impl_saved == var_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "quantized_batch_norm");
  return result;
}
Tensor quantized_max_pool2d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("quantized_max_pool2d"), deleteNode);
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
    return at::redispatch::quantized_max_pool2d(ks & c10::after_autograd_keyset, self_, kernel_size, stride, padding, dilation, ceil_mode);
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
  throw_error_for_complex_autograd(result, "quantized_max_pool2d");
  return result;
}
Tensor & rad2deg_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("rad2deg");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("rad2deg");
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
    at::redispatch::rad2deg_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor & range_out_out(c10::DispatchKeySet ks, const Scalar & start, const Scalar & end, const Scalar & step, Tensor & out) {
  auto& out_ = unpack(out, "out", 3);
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::range_outf(ks & c10::after_autograd_keyset, start, end, step, out_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  return out;
}
Tensor & reciprocal_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("reciprocal");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("reciprocal");
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
    at::redispatch::reciprocal_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor & reflection_pad1d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("reflection_pad1d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("reflection_pad1d");
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
    at::redispatch::reflection_pad1d_outf(ks & c10::after_autograd_keyset, self_, padding, out_);
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
Tensor renorm(c10::DispatchKeySet ks, const Tensor & self, const Scalar & p, int64_t dim, const Scalar & maxnorm) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<RenormBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RenormBackward>(new RenormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
    grad_fn->dim = dim;
    grad_fn->maxnorm = maxnorm;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::renorm(ks & c10::after_autograd_keyset, self_, p, dim, maxnorm);
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
  throw_error_for_complex_autograd(result, "renorm");
  return result;
}
Tensor & renorm_(c10::DispatchKeySet ks, Tensor & self, const Scalar & p, int64_t dim, const Scalar & maxnorm) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<RenormBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RenormBackward>(new RenormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->p = p;
    grad_fn->dim = dim;
    grad_fn->maxnorm = maxnorm;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::renorm_(ks & c10::after_autograd_keyset, self_, p, dim, maxnorm);
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
Tensor replication_pad3d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<ReplicationPad3DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ReplicationPad3DBackwardBackward>(new ReplicationPad3DBackwardBackward(), deleteNode);
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
    return at::redispatch::replication_pad3d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, padding);
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
Tensor & rrelu_with_noise_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & noise, const Scalar & lower, const Scalar & upper, bool training, c10::optional<Generator> generator, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& noise_ = unpack(noise, "noise", 1);
  auto& out_ = unpack(out, "out", 6);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, noise )) {
    throw_error_out_requires_grad("rrelu_with_noise");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("rrelu_with_noise");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> noise__storage_saved =
    noise_.has_storage() ? c10::optional<Storage>(noise_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> noise__impl_saved;
  if (noise_.defined()) noise__impl_saved = noise_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::rrelu_with_noise_outf(ks & c10::after_autograd_keyset, self_, noise_, lower, upper, training, generator, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (noise__storage_saved.has_value())
    AT_ASSERT(noise__storage_saved.value().is_alias_of(noise_.storage()));
  if (noise__impl_saved) AT_ASSERT(noise__impl_saved == noise_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor rsub_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<RsubBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RsubBackward0>(new RsubBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->other_scalar_type = other.scalar_type();
    grad_fn->alpha = alpha;
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
    return at::redispatch::rsub(ks & c10::after_autograd_keyset, self_, other_, alpha);
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
  return result;
}
Tensor rsub_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<RsubBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RsubBackward1>(new RsubBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->alpha = alpha;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::rsub(ks & c10::after_autograd_keyset, self_, other, alpha);
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
Tensor sigmoid_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & output) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, output );
  (void)_any_requires_grad;
  std::shared_ptr<SigmoidBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SigmoidBackwardBackward>(new SigmoidBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
    grad_fn->output_ = SavedVariable(output, false);
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::sigmoid_backward(ks & c10::after_autograd_keyset, grad_output_, output_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "sigmoid_backward");
  return result;
}
Tensor & silu_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("silu");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("silu");
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
    at::redispatch::silu_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor & sin_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sin");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sin");
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
    at::redispatch::sin_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor sinc(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SincBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SincBackward>(new SincBackward(), deleteNode);
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
    return at::redispatch::sinc(ks & c10::after_autograd_keyset, self_);
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
Tensor & sinc_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<SincBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SincBackward>(new SincBackward(), deleteNode);
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
    at::redispatch::sinc_(ks & c10::after_autograd_keyset, self_);
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
Tensor sinh(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SinhBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SinhBackward>(new SinhBackward(), deleteNode);
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
    return at::redispatch::sinh(ks & c10::after_autograd_keyset, self_);
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
Tensor & sinh_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<SinhBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SinhBackward>(new SinhBackward(), deleteNode);
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
    at::redispatch::sinh_(ks & c10::after_autograd_keyset, self_);
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
std::tuple<Tensor,Tensor> slogdet(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SlogdetBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SlogdetBackward>(new SlogdetBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor sign;
  Tensor logabsdet;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::slogdet(ks & c10::after_autograd_keyset, self_);
  })();
  std::tie(sign, logabsdet) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( logabsdet ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->sign_ = SavedVariable(sign, true);
    grad_fn->logabsdet_ = SavedVariable(logabsdet, true);
  }
  throw_error_for_complex_autograd(logabsdet, "slogdet");
  return std::make_tuple(std::move(sign), std::move(logabsdet));
}
std::tuple<Tensor,Tensor,Tensor> slow_conv3d_forward(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<SlowConv3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SlowConv3DBackward>(new SlowConv3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
  }
  Tensor output;
  Tensor finput;
  Tensor fgrad_input;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::slow_conv3d_forward(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding);
  })();
  std::tie(output, finput, fgrad_input) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->finput_ = SavedVariable(finput, true);
    grad_fn->fgrad_input_ = SavedVariable(fgrad_input, true);
  }
  throw_error_for_complex_autograd(output, "slow_conv3d_forward");
  return std::make_tuple(std::move(output), std::move(finput), std::move(fgrad_input));
}
std::tuple<Tensor,Tensor,Tensor> slow_conv_dilated2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight );
  (void)_any_requires_grad;
  std::shared_ptr<SlowConvDilated2DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SlowConvDilated2DBackwardBackward>(new SlowConvDilated2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::slow_conv_dilated2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, dilation, output_mask);
  })();
  std::tie(grad_input, grad_weight, grad_bias) = std::move(_tmp);
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
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  throw_error_for_complex_autograd(grad_input, "slow_conv_dilated2d_backward");
  throw_error_for_complex_autograd(grad_weight, "slow_conv_dilated2d_backward");
  throw_error_for_complex_autograd(grad_bias, "slow_conv_dilated2d_backward");
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor slow_conv_dilated3d(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<SlowConvDilated3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SlowConvDilated3DBackward>(new SlowConvDilated3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::slow_conv_dilated3d(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding, dilation);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "slow_conv_dilated3d");
  return result;
}
std::tuple<Tensor,Tensor,Tensor> slow_conv_transpose3d_backward_output_mask(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 8);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 9);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight );
  (void)_any_requires_grad;
  check_no_requires_grad(finput, "finput");
  check_no_requires_grad(fgrad_input, "fgrad_input");
  std::shared_ptr<SlowConvTranspose3DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SlowConvTranspose3DBackwardBackward>(new SlowConvTranspose3DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
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
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::slow_conv_transpose3d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, finput_, fgrad_input_, output_mask);
  })();
  std::tie(grad_input, grad_weight, grad_bias) = std::move(_tmp);
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
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  throw_error_for_complex_autograd(grad_input, "slow_conv_transpose3d_backward");
  throw_error_for_complex_autograd(grad_weight, "slow_conv_transpose3d_backward");
  throw_error_for_complex_autograd(grad_bias, "slow_conv_transpose3d_backward");
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor softshrink(c10::DispatchKeySet ks, const Tensor & self, const Scalar & lambd) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SoftshrinkBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SoftshrinkBackward>(new SoftshrinkBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->lambd = lambd;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::softshrink(ks & c10::after_autograd_keyset, self_, lambd);
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
  throw_error_for_complex_autograd(result, "softshrink");
  return result;
}
Tensor & softshrink_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Scalar & lambd, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("softshrink_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("softshrink_backward");
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
    at::redispatch::softshrink_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, lambd, grad_input_);
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
Tensor stack(c10::DispatchKeySet ks, TensorList tensors, int64_t dim) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<StackBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<StackBackward>(new StackBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
    grad_fn->tensors_ = make_saved_variable_list(tensors);
    grad_fn->dim = dim;
    grad_fn->tensors_size_ = tensors.size();
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
    return at::redispatch::stack(ks & c10::after_autograd_keyset, tensors_, dim);
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
Tensor & sub_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("sub");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sub");
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
    at::redispatch::sub_outf(ks & c10::after_autograd_keyset, self_, other_, alpha, out_);
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
std::tuple<Tensor,Tensor> symeig(c10::DispatchKeySet ks, const Tensor & self, bool eigenvectors, bool upper) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SymeigBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SymeigBackward>(new SymeigBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->eigenvectors = eigenvectors;
    grad_fn->upper = upper;
  }
  Tensor eigenvalues;
  Tensor eigenvectors_return;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::symeig(ks & c10::after_autograd_keyset, self_, eigenvectors, upper);
  })();
  std::tie(eigenvalues, eigenvectors_return) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( eigenvalues, eigenvectors_return ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->eigenvalues_ = SavedVariable(eigenvalues, true);
    grad_fn->eigenvectors_return_ = SavedVariable(eigenvectors_return, true);
  }
  throw_error_for_complex_autograd(eigenvalues, "symeig");
  throw_error_for_complex_autograd(eigenvectors_return, "symeig");
  return std::make_tuple(std::move(eigenvalues), std::move(eigenvectors_return));
}
std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_backward_output_mask(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 6);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 7);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight );
  (void)_any_requires_grad;
  check_no_requires_grad(finput, "finput");
  check_no_requires_grad(fgrad_input, "fgrad_input");
  std::shared_ptr<ThnnConv2DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ThnnConv2DBackwardBackward>(new ThnnConv2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
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
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::thnn_conv2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_, output_mask);
  })();
  std::tie(grad_input, grad_weight, grad_bias) = std::move(_tmp);
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
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  throw_error_for_complex_autograd(grad_input, "thnn_conv2d_backward");
  throw_error_for_complex_autograd(grad_weight, "thnn_conv2d_backward");
  throw_error_for_complex_autograd(grad_bias, "thnn_conv2d_backward");
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, Tensor & output, Tensor & finput, Tensor & fgrad_input) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& output_ = unpack(output, "output", 6);
  auto& finput_ = unpack(finput, "finput", 7);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 8);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv2d_forward");
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
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::thnn_conv2d_forward_outf(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding, output_, finput_, fgrad_input_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  return std::forward_as_tuple(output, finput, fgrad_input);
}
Tensor threshold_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Scalar & threshold) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<ThresholdBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ThresholdBackwardBackward>(new ThresholdBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->threshold = threshold;
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
    return at::redispatch::threshold_backward(ks & c10::after_autograd_keyset, grad_output_, self_, threshold);
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
  throw_error_for_complex_autograd(result, "threshold_backward");
  return result;
}
Tensor to_dense(c10::DispatchKeySet ks, const Tensor & self, c10::optional<ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ToDenseBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ToDenseBackward>(new ToDenseBackward(), deleteNode);
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
    return at::redispatch::to_dense(ks & c10::after_autograd_keyset, self_, dtype);
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
  throw_error_for_complex_autograd(result, "to_dense");
  return result;
}
std::tuple<Tensor,Tensor> triangular_solve(c10::DispatchKeySet ks, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  auto _any_requires_grad = compute_requires_grad( self, A );
  (void)_any_requires_grad;
  std::shared_ptr<TriangularSolveBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TriangularSolveBackward>(new TriangularSolveBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->A_ = SavedVariable(A, false);
    grad_fn->upper = upper;
    grad_fn->transpose = transpose;
    grad_fn->unitriangular = unitriangular;
  }
  Tensor solution;
  Tensor cloned_coefficient;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::triangular_solve(ks & c10::after_autograd_keyset, self_, A_, upper, transpose, unitriangular);
  })();
  std::tie(solution, cloned_coefficient) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( solution, cloned_coefficient ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->solution_ = SavedVariable(solution, true);
  }
  return std::make_tuple(std::move(solution), std::move(cloned_coefficient));
}
Tensor unfold_backward(c10::DispatchKeySet ks, const Tensor & grad_in, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
  auto& grad_in_ = unpack(grad_in, "grad_in", 0);
  auto _any_requires_grad = compute_requires_grad( grad_in );
  (void)_any_requires_grad;
  std::shared_ptr<UnfoldBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UnfoldBackwardBackward>(new UnfoldBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_in ));
    grad_fn->dim = dim;
    grad_fn->size = size;
    grad_fn->step = step;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_in__storage_saved =
    grad_in_.has_storage() ? c10::optional<Storage>(grad_in_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_in__impl_saved;
  if (grad_in_.defined()) grad_in__impl_saved = grad_in_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::unfold_backward(ks & c10::after_autograd_keyset, grad_in_, input_sizes, dim, size, step);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_in__storage_saved.has_value())
    AT_ASSERT(grad_in__storage_saved.value().is_alias_of(grad_in_.storage()));
  if (grad_in__impl_saved) AT_ASSERT(grad_in__impl_saved == grad_in_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "unfold_backward");
  return result;
}
std::tuple<Tensor,Tensor,Tensor> unique_dim_consecutive(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool return_inverse, bool return_counts) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<UniqueDimConsecutiveBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UniqueDimConsecutiveBackward>(new UniqueDimConsecutiveBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::unique_dim_consecutive(ks & c10::after_autograd_keyset, self_, dim, return_inverse, return_counts);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "unique_dim_consecutive");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor upsample_bilinear2d_vec(c10::DispatchKeySet ks, const Tensor & input, c10::optional<IntArrayRef> output_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
  auto& input_ = unpack(input, "input", 0);
  auto _any_requires_grad = compute_requires_grad( input );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleBilinear2DBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleBilinear2DBackward1>(new UpsampleBilinear2DBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input ));
    grad_fn->input_sizes = input.sizes().vec();
    grad_fn->output_size = output_size;
    grad_fn->align_corners = align_corners;
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
    return at::redispatch::upsample_bilinear2d(ks & c10::after_autograd_keyset, input_, output_size, align_corners, scale_factors);
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
  throw_error_for_complex_autograd(result, "upsample_bilinear2d");
  return result;
}
Tensor upsample_bilinear2d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleBilinear2DBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleBilinear2DBackward0>(new UpsampleBilinear2DBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
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
    return at::redispatch::upsample_bilinear2d(ks & c10::after_autograd_keyset, self_, output_size, align_corners, scales_h, scales_w);
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
  throw_error_for_complex_autograd(result, "upsample_bilinear2d");
  return result;
}
Tensor & upsample_bilinear2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& grad_input_ = unpack(grad_input, "grad_input", 6);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_bilinear2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_bilinear2d_backward");
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
    at::redispatch::upsample_bilinear2d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, align_corners, scales_h, scales_w, grad_input_);
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
Tensor & upsample_linear1d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_linear1d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_linear1d");
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
    at::redispatch::upsample_linear1d_outf(ks & c10::after_autograd_keyset, self_, output_size, align_corners, scales, out_);
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
Tensor upsample_nearest1d_vec(c10::DispatchKeySet ks, const Tensor & input, c10::optional<IntArrayRef> output_size, c10::optional<ArrayRef<double>> scale_factors) {
  auto& input_ = unpack(input, "input", 0);
  auto _any_requires_grad = compute_requires_grad( input );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleNearest1DBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleNearest1DBackward1>(new UpsampleNearest1DBackward1(), deleteNode);
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
    return at::redispatch::upsample_nearest1d(ks & c10::after_autograd_keyset, input_, output_size, scale_factors);
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
  throw_error_for_complex_autograd(result, "upsample_nearest1d");
  return result;
}
Tensor upsample_nearest1d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleNearest1DBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleNearest1DBackward0>(new UpsampleNearest1DBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->scales = scales;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::upsample_nearest1d(ks & c10::after_autograd_keyset, self_, output_size, scales);
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
  throw_error_for_complex_autograd(result, "upsample_nearest1d");
  return result;
}
Tensor & upsample_nearest1d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& grad_input_ = unpack(grad_input, "grad_input", 4);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_nearest1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_nearest1d_backward");
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
    at::redispatch::upsample_nearest1d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, scales, grad_input_);
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
Tensor & upsample_nearest2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_nearest2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_nearest2d");
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
    at::redispatch::upsample_nearest2d_outf(ks & c10::after_autograd_keyset, self_, output_size, scales_h, scales_w, out_);
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
Tensor var(c10::DispatchKeySet ks, const Tensor & self, bool unbiased) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<VarBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<VarBackward0>(new VarBackward0(), deleteNode);
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
    return at::redispatch::var(ks & c10::after_autograd_keyset, self_, unbiased);
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
  throw_error_for_complex_autograd(result, "var");
  return result;
}
Tensor var_dim(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<VarBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<VarBackward1>(new VarBackward1(), deleteNode);
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
    return at::redispatch::var(ks & c10::after_autograd_keyset, self_, dim, unbiased, keepdim);
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
  throw_error_for_complex_autograd(result, "var");
  return result;
}
Tensor & vdot_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("vdot");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("vdot");
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
    at::redispatch::vdot_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
}
}

namespace {

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  m.impl("_cholesky_solve_helper",
         TORCH_FN(VariableType::_cholesky_solve_helper)
  );
  m.impl("_copy_from",
         TORCH_FN(VariableType::_copy_from)
  );
  m.impl("_ctc_loss_backward",
         TORCH_FN(VariableType::_ctc_loss_backward)
  );
  m.impl("_cudnn_ctc_loss",
         TORCH_FN(VariableType::_cudnn_ctc_loss)
  );
  m.impl("_cummin_helper",
         TORCH_FN(VariableType::_cummin_helper)
  );
  m.impl("_euclidean_dist",
         TORCH_FN(VariableType::_euclidean_dist)
  );
  m.impl("_foreach_erfc",
         TORCH_FN(VariableType::_foreach_erfc)
  );
  m.impl("_foreach_erfc_",
         TORCH_FN(VariableType::_foreach_erfc_)
  );
  m.impl("_foreach_expm1",
         TORCH_FN(VariableType::_foreach_expm1)
  );
  m.impl("_foreach_expm1_",
         TORCH_FN(VariableType::_foreach_expm1_)
  );
  m.impl("_foreach_floor",
         TORCH_FN(VariableType::_foreach_floor)
  );
  m.impl("_foreach_floor_",
         TORCH_FN(VariableType::_foreach_floor_)
  );
  m.impl("_foreach_maximum.List",
         TORCH_FN(VariableType::_foreach_maximum_List)
  );
  m.impl("_foreach_trunc",
         TORCH_FN(VariableType::_foreach_trunc)
  );
  m.impl("_foreach_trunc_",
         TORCH_FN(VariableType::_foreach_trunc_)
  );
  m.impl("_index_copy_",
         TORCH_FN(VariableType::_index_copy_)
  );
  m.impl("_inverse_helper",
         TORCH_FN(VariableType::_inverse_helper)
  );
  m.impl("_masked_scale",
         TORCH_FN(VariableType::_masked_scale)
  );
  m.impl("_pdist_backward",
         TORCH_FN(VariableType::_pdist_backward)
  );
  m.impl("_s_where",
         TORCH_FN(VariableType::_s_where)
  );
  m.impl("_sample_dirichlet",
         TORCH_FN(VariableType::_sample_dirichlet)
  );
  m.impl("_sparse_addmm",
         TORCH_FN(VariableType::_sparse_addmm)
  );
  m.impl("_sparse_coo_tensor_with_dims_and_tensors",
         TORCH_FN(VariableType::_sparse_coo_tensor_with_dims_and_tensors)
  );
  m.impl("_sparse_softmax",
         TORCH_FN(VariableType::_sparse_softmax)
  );
  m.impl("_sparse_sparse_matmul",
         TORCH_FN(VariableType::_sparse_sparse_matmul)
  );
  m.impl("_stack",
         TORCH_FN(VariableType::_stack)
  );
  m.impl("_standard_gamma",
         TORCH_FN(VariableType::_standard_gamma)
  );
  m.impl("_symeig_helper",
         TORCH_FN(VariableType::_symeig_helper)
  );
  m.impl("_thnn_fused_lstm_cell",
         TORCH_FN(VariableType::_thnn_fused_lstm_cell)
  );
  m.impl("_var",
         TORCH_FN(VariableType::_var)
  );
  m.impl("abs.out",
         TORCH_FN(VariableType::abs_out_out)
  );
  m.impl("adaptive_max_pool2d",
         TORCH_FN(VariableType::adaptive_max_pool2d)
  );
  m.impl("adaptive_max_pool2d_backward.grad_input",
         TORCH_FN(VariableType::adaptive_max_pool2d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool3d.out",
         TORCH_FN(VariableType::adaptive_max_pool3d_out_out)
  );
  m.impl("addcdiv",
         TORCH_FN(VariableType::addcdiv)
  );
  m.impl("addcdiv_",
         TORCH_FN(VariableType::addcdiv_)
  );
  m.impl("addcmul.out",
         TORCH_FN(VariableType::addcmul_out_out)
  );
  m.impl("addmm.out",
         TORCH_FN(VariableType::addmm_out_out)
  );
  m.impl("addmv",
         TORCH_FN(VariableType::addmv)
  );
  m.impl("addmv_",
         TORCH_FN(VariableType::addmv_)
  );
  m.impl("angle",
         TORCH_FN(VariableType::angle)
  );
  m.impl("argmax.out",
         TORCH_FN(VariableType::argmax_out_out)
  );
  m.impl("atan2.out",
         TORCH_FN(VariableType::atan2_out_out)
  );
  m.impl("atan.out",
         TORCH_FN(VariableType::atan_out_out)
  );
  m.impl("atanh",
         TORCH_FN(VariableType::atanh)
  );
  m.impl("atanh_",
         TORCH_FN(VariableType::atanh_)
  );
  m.impl("avg_pool2d.out",
         TORCH_FN(VariableType::avg_pool2d_out_out)
  );
  m.impl("batch_norm_backward_elemt",
         TORCH_FN(VariableType::batch_norm_backward_elemt)
  );
  m.impl("batch_norm_gather_stats_with_counts",
         TORCH_FN(VariableType::batch_norm_gather_stats_with_counts)
  );
  m.impl("binary_cross_entropy",
         TORCH_FN(VariableType::binary_cross_entropy)
  );
  m.impl("binary_cross_entropy_backward.grad_input",
         TORCH_FN(VariableType::binary_cross_entropy_backward_out_grad_input)
  );
  m.impl("cholesky_inverse.out",
         TORCH_FN(VariableType::cholesky_inverse_out_out)
  );
  m.impl("cholesky_solve",
         TORCH_FN(VariableType::cholesky_solve)
  );
  m.impl("clamp_min",
         TORCH_FN(VariableType::clamp_min)
  );
  m.impl("clamp_min_",
         TORCH_FN(VariableType::clamp_min_)
  );
  m.impl("clamp.out",
         TORCH_FN(VariableType::clamp_out_out)
  );
  m.impl("clone",
         TORCH_FN(VariableType::clone)
  );
  m.impl("complex.out",
         TORCH_FN(VariableType::complex_out_out)
  );
  m.impl("convolution_overrideable",
         TORCH_FN(VariableType::convolution_overrideable)
  );
  m.impl("cos.out",
         TORCH_FN(VariableType::cos_out_out)
  );
  m.impl("cosh",
         TORCH_FN(VariableType::cosh)
  );
  m.impl("cosh_",
         TORCH_FN(VariableType::cosh_)
  );
  m.impl("cross",
         TORCH_FN(VariableType::cross)
  );
  m.impl("cudnn_batch_norm",
         TORCH_FN(VariableType::cudnn_batch_norm)
  );
  m.impl("cudnn_convolution_add_relu",
         TORCH_FN(VariableType::cudnn_convolution_add_relu)
  );
  m.impl("cudnn_convolution_transpose_backward",
         TORCH_FN(VariableType::cudnn_convolution_transpose_backward)
  );
  m.impl("cudnn_convolution_transpose_backward_input",
         TORCH_FN(VariableType::cudnn_convolution_transpose_backward_input)
  );
  m.impl("cudnn_grid_sampler",
         TORCH_FN(VariableType::cudnn_grid_sampler)
  );
  m.impl("cummin",
         TORCH_FN(VariableType::cummin)
  );
  m.impl("deg2rad.out",
         TORCH_FN(VariableType::deg2rad_out_out)
  );
  m.impl("diag.out",
         TORCH_FN(VariableType::diag_out_out)
  );
  m.impl("digamma.out",
         TORCH_FN(VariableType::digamma_out_out)
  );
  m.impl("eig",
         TORCH_FN(VariableType::eig)
  );
  m.impl("empty.memory_format",
         TORCH_FN(VariableType::empty_memory_format)
  );
  m.impl("eq.Scalar",
         TORCH_FN(VariableType::eq_Scalar)
  );
  m.impl("eq.Tensor",
         TORCH_FN(VariableType::eq_Tensor)
  );
  m.impl("eq_.Scalar",
         TORCH_FN(VariableType::eq__Scalar)
  );
  m.impl("eq_.Tensor",
         TORCH_FN(VariableType::eq__Tensor)
  );
  m.impl("erfinv.out",
         TORCH_FN(VariableType::erfinv_out_out)
  );
  m.impl("fake_quantize_per_channel_affine_cachemask",
         TORCH_FN(VariableType::fake_quantize_per_channel_affine_cachemask)
  );
  m.impl("floor_divide.out",
         TORCH_FN(VariableType::floor_divide_out_out)
  );
  m.impl("fractional_max_pool3d_backward",
         TORCH_FN(VariableType::fractional_max_pool3d_backward)
  );
  m.impl("frexp.Tensor",
         TORCH_FN(VariableType::frexp_Tensor)
  );
  m.impl("gather.out",
         TORCH_FN(VariableType::gather_out_out)
  );
  m.impl("ge.Scalar",
         TORCH_FN(VariableType::ge_Scalar)
  );
  m.impl("ge.Tensor",
         TORCH_FN(VariableType::ge_Tensor)
  );
  m.impl("ge_.Scalar",
         TORCH_FN(VariableType::ge__Scalar)
  );
  m.impl("ge_.Tensor",
         TORCH_FN(VariableType::ge__Tensor)
  );
  m.impl("gelu",
         TORCH_FN(VariableType::gelu)
  );
  m.impl("geometric_",
         TORCH_FN(VariableType::geometric_)
  );
  m.impl("glu_backward",
         TORCH_FN(VariableType::glu_backward)
  );
  m.impl("grid_sampler_2d_backward",
         TORCH_FN(VariableType::grid_sampler_2d_backward)
  );
  m.impl("grid_sampler_3d",
         TORCH_FN(VariableType::grid_sampler_3d)
  );
  m.impl("gt.Scalar",
         TORCH_FN(VariableType::gt_Scalar)
  );
  m.impl("gt.Tensor",
         TORCH_FN(VariableType::gt_Tensor)
  );
  m.impl("gt_.Scalar",
         TORCH_FN(VariableType::gt__Scalar)
  );
  m.impl("gt_.Tensor",
         TORCH_FN(VariableType::gt__Tensor)
  );
  m.impl("hardsigmoid_backward",
         TORCH_FN(VariableType::hardsigmoid_backward)
  );
  m.impl("hardswish_backward",
         TORCH_FN(VariableType::hardswish_backward)
  );
  m.impl("histc",
         TORCH_FN(VariableType::histc)
  );
  m.impl("hspmm",
         TORCH_FN(VariableType::hspmm)
  );
  m.impl("huber_loss_backward",
         TORCH_FN(VariableType::huber_loss_backward)
  );
  m.impl("hypot",
         TORCH_FN(VariableType::hypot)
  );
  m.impl("hypot_",
         TORCH_FN(VariableType::hypot_)
  );
  m.impl("igamma.out",
         TORCH_FN(VariableType::igamma_out_out)
  );
  m.impl("igammac",
         TORCH_FN(VariableType::igammac)
  );
  m.impl("igammac_",
         TORCH_FN(VariableType::igammac_)
  );
  m.impl("index_copy_",
         TORCH_FN(VariableType::index_copy_)
  );
  m.impl("index_fill_.int_Scalar",
         TORCH_FN(VariableType::index_fill__int_Scalar)
  );
  m.impl("index_fill_.int_Tensor",
         TORCH_FN(VariableType::index_fill__int_Tensor)
  );
  m.impl("inverse",
         TORCH_FN(VariableType::inverse)
  );
  m.impl("is_set_to",
         TORCH_FN(VariableType::is_set_to)
  );
  m.impl("isposinf.out",
         TORCH_FN(VariableType::isposinf_out_out)
  );
  m.impl("kl_div_backward",
         TORCH_FN(VariableType::kl_div_backward)
  );
  m.impl("le.Scalar",
         TORCH_FN(VariableType::le_Scalar)
  );
  m.impl("le.Tensor",
         TORCH_FN(VariableType::le_Tensor)
  );
  m.impl("le_.Scalar",
         TORCH_FN(VariableType::le__Scalar)
  );
  m.impl("le_.Tensor",
         TORCH_FN(VariableType::le__Tensor)
  );
  m.impl("leaky_relu",
         TORCH_FN(VariableType::leaky_relu)
  );
  m.impl("leaky_relu_",
         TORCH_FN(VariableType::leaky_relu_)
  );
  m.impl("lerp.Scalar_out",
         TORCH_FN(VariableType::lerp_out_Scalar_out)
  );
  m.impl("lerp.Tensor_out",
         TORCH_FN(VariableType::lerp_out_Tensor_out)
  );
  m.impl("linalg_eigh",
         TORCH_FN(VariableType::linalg_eigh)
  );
  m.impl("linalg_inv",
         TORCH_FN(VariableType::linalg_inv)
  );
  m.impl("linalg_slogdet.out",
         TORCH_FN(VariableType::linalg_slogdet_out_out)
  );
  m.impl("linalg_solve",
         TORCH_FN(VariableType::linalg_solve)
  );
  m.impl("linalg_vector_norm",
         TORCH_FN(VariableType::linalg_vector_norm)
  );
  m.impl("log10",
         TORCH_FN(VariableType::log10)
  );
  m.impl("log10_",
         TORCH_FN(VariableType::log10_)
  );
  m.impl("logdet",
         TORCH_FN(VariableType::logdet)
  );
  m.impl("logit_backward",
         TORCH_FN(VariableType::logit_backward)
  );
  m.impl("lt.Scalar",
         TORCH_FN(VariableType::lt_Scalar)
  );
  m.impl("lt.Tensor",
         TORCH_FN(VariableType::lt_Tensor)
  );
  m.impl("lt_.Scalar",
         TORCH_FN(VariableType::lt__Scalar)
  );
  m.impl("lt_.Tensor",
         TORCH_FN(VariableType::lt__Tensor)
  );
  m.impl("masked_select",
         TORCH_FN(VariableType::masked_select)
  );
  m.impl("matrix_exp",
         TORCH_FN(VariableType::matrix_exp)
  );
  m.impl("max_pool3d_with_indices_backward",
         TORCH_FN(VariableType::max_pool3d_with_indices_backward)
  );
  m.impl("max_unpool2d.out",
         TORCH_FN(VariableType::max_unpool2d_out_out)
  );
  m.impl("min.dim",
         TORCH_FN(VariableType::min_dim)
  );
  m.impl("min",
         TORCH_FN(VariableType::min)
  );
  m.impl("minimum",
         TORCH_FN(VariableType::minimum)
  );
  m.impl("miopen_convolution",
         TORCH_FN(VariableType::miopen_convolution)
  );
  m.impl("miopen_convolution_transpose_backward_weight",
         TORCH_FN(VariableType::miopen_convolution_transpose_backward_weight)
  );
  m.impl("mkldnn_linear",
         TORCH_FN(VariableType::mkldnn_linear)
  );
  m.impl("mkldnn_max_pool3d_backward",
         TORCH_FN(VariableType::mkldnn_max_pool3d_backward)
  );
  m.impl("mse_loss",
         TORCH_FN(VariableType::mse_loss)
  );
  m.impl("mse_loss_backward.grad_input",
         TORCH_FN(VariableType::mse_loss_backward_out_grad_input)
  );
  m.impl("mul.Tensor",
         TORCH_FN(VariableType::mul_Tensor)
  );
  m.impl("mul.Scalar",
         TORCH_FN(VariableType::mul_Scalar)
  );
  m.impl("mul_.Tensor",
         TORCH_FN(VariableType::mul__Tensor)
  );
  m.impl("mul_.Scalar",
         TORCH_FN(VariableType::mul__Scalar)
  );
  m.impl("multilabel_margin_loss_backward.grad_input",
         TORCH_FN(VariableType::multilabel_margin_loss_backward_out_grad_input)
  );
  m.impl("multinomial.out",
         TORCH_FN(VariableType::multinomial_out_out)
  );
  m.impl("nan_to_num.out",
         TORCH_FN(VariableType::nan_to_num_out_out)
  );
  m.impl("nanmedian",
         TORCH_FN(VariableType::nanmedian)
  );
  m.impl("nanmedian.dim",
         TORCH_FN(VariableType::nanmedian_dim)
  );
  m.impl("native_batch_norm.out",
         TORCH_FN(VariableType::native_batch_norm_out_out)
  );
  m.impl("native_group_norm_backward",
         TORCH_FN(VariableType::native_group_norm_backward)
  );
  m.impl("neg",
         TORCH_FN(VariableType::neg)
  );
  m.impl("neg_",
         TORCH_FN(VariableType::neg_)
  );
  m.impl("nonzero",
         TORCH_FN(VariableType::nonzero)
  );
  m.impl("norm.ScalarOpt_dtype",
         TORCH_FN(VariableType::norm_ScalarOpt_dtype)
  );
  m.impl("norm.Scalar",
         TORCH_FN(VariableType::norm_Scalar)
  );
  m.impl("norm.ScalarOpt_dim_dtype",
         TORCH_FN(VariableType::norm_ScalarOpt_dim_dtype)
  );
  m.impl("norm.ScalarOpt_dim",
         TORCH_FN(VariableType::norm_ScalarOpt_dim)
  );
  m.impl("normal.Tensor_float",
         TORCH_FN(VariableType::normal_Tensor_float)
  );
  m.impl("normal.float_Tensor",
         TORCH_FN(VariableType::normal_float_Tensor)
  );
  m.impl("normal.Tensor_Tensor",
         TORCH_FN(VariableType::normal_Tensor_Tensor)
  );
  m.impl("normal_",
         TORCH_FN(VariableType::normal_)
  );
  m.impl("poisson",
         TORCH_FN(VariableType::poisson)
  );
  m.impl("q_per_channel_scales",
         TORCH_FN(VariableType::q_per_channel_scales)
  );
  m.impl("quantized_batch_norm",
         TORCH_FN(VariableType::quantized_batch_norm)
  );
  m.impl("quantized_max_pool2d",
         TORCH_FN(VariableType::quantized_max_pool2d)
  );
  m.impl("rad2deg.out",
         TORCH_FN(VariableType::rad2deg_out_out)
  );
  m.impl("range.out",
         TORCH_FN(VariableType::range_out_out)
  );
  m.impl("reciprocal.out",
         TORCH_FN(VariableType::reciprocal_out_out)
  );
  m.impl("reflection_pad1d.out",
         TORCH_FN(VariableType::reflection_pad1d_out_out)
  );
  m.impl("renorm",
         TORCH_FN(VariableType::renorm)
  );
  m.impl("renorm_",
         TORCH_FN(VariableType::renorm_)
  );
  m.impl("replication_pad3d_backward",
         TORCH_FN(VariableType::replication_pad3d_backward)
  );
  m.impl("rrelu_with_noise.out",
         TORCH_FN(VariableType::rrelu_with_noise_out_out)
  );
  m.impl("rsub.Tensor",
         TORCH_FN(VariableType::rsub_Tensor)
  );
  m.impl("rsub.Scalar",
         TORCH_FN(VariableType::rsub_Scalar)
  );
  m.impl("sigmoid_backward",
         TORCH_FN(VariableType::sigmoid_backward)
  );
  m.impl("silu.out",
         TORCH_FN(VariableType::silu_out_out)
  );
  m.impl("sin.out",
         TORCH_FN(VariableType::sin_out_out)
  );
  m.impl("sinc",
         TORCH_FN(VariableType::sinc)
  );
  m.impl("sinc_",
         TORCH_FN(VariableType::sinc_)
  );
  m.impl("sinh",
         TORCH_FN(VariableType::sinh)
  );
  m.impl("sinh_",
         TORCH_FN(VariableType::sinh_)
  );
  m.impl("slogdet",
         TORCH_FN(VariableType::slogdet)
  );
  m.impl("slow_conv3d_forward",
         TORCH_FN(VariableType::slow_conv3d_forward)
  );
  m.impl("slow_conv_dilated2d_backward",
         TORCH_FN(VariableType::slow_conv_dilated2d_backward)
  );
  m.impl("slow_conv_dilated3d",
         TORCH_FN(VariableType::slow_conv_dilated3d)
  );
  m.impl("slow_conv_transpose3d_backward.output_mask",
         TORCH_FN(VariableType::slow_conv_transpose3d_backward_output_mask)
  );
  m.impl("softshrink",
         TORCH_FN(VariableType::softshrink)
  );
  m.impl("softshrink_backward.grad_input",
         TORCH_FN(VariableType::softshrink_backward_out_grad_input)
  );
  m.impl("stack",
         TORCH_FN(VariableType::stack)
  );
  m.impl("sub.out",
         TORCH_FN(VariableType::sub_out_out)
  );
  m.impl("symeig",
         TORCH_FN(VariableType::symeig)
  );
  m.impl("thnn_conv2d_backward.output_mask",
         TORCH_FN(VariableType::thnn_conv2d_backward_output_mask)
  );
  m.impl("thnn_conv2d_forward.output",
         TORCH_FN(VariableType::thnn_conv2d_forward_out_output)
  );
  m.impl("threshold_backward",
         TORCH_FN(VariableType::threshold_backward)
  );
  m.impl("to_dense",
         TORCH_FN(VariableType::to_dense)
  );
  m.impl("triangular_solve",
         TORCH_FN(VariableType::triangular_solve)
  );
  m.impl("unfold_backward",
         TORCH_FN(VariableType::unfold_backward)
  );
  m.impl("unique_dim_consecutive",
         TORCH_FN(VariableType::unique_dim_consecutive)
  );
  m.impl("upsample_bilinear2d.vec",
         TORCH_FN(VariableType::upsample_bilinear2d_vec)
  );
  m.impl("upsample_bilinear2d",
         TORCH_FN(VariableType::upsample_bilinear2d)
  );
  m.impl("upsample_bilinear2d_backward.grad_input",
         TORCH_FN(VariableType::upsample_bilinear2d_backward_out_grad_input)
  );
  m.impl("upsample_linear1d.out",
         TORCH_FN(VariableType::upsample_linear1d_out_out)
  );
  m.impl("upsample_nearest1d.vec",
         TORCH_FN(VariableType::upsample_nearest1d_vec)
  );
  m.impl("upsample_nearest1d",
         TORCH_FN(VariableType::upsample_nearest1d)
  );
  m.impl("upsample_nearest1d_backward.grad_input",
         TORCH_FN(VariableType::upsample_nearest1d_backward_out_grad_input)
  );
  m.impl("upsample_nearest2d.out",
         TORCH_FN(VariableType::upsample_nearest2d_out_out)
  );
  m.impl("var",
         TORCH_FN(VariableType::var)
  );
  m.impl("var.dim",
         TORCH_FN(VariableType::var_dim)
  );
  m.impl("vdot.out",
         TORCH_FN(VariableType::vdot_out_out)
  );
}

}

}} // namespace torch::autograd
