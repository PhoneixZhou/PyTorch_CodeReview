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
Tensor & _addmv_impl_(c10::DispatchKeySet ks, Tensor & self, const Tensor & self2, const Tensor & mat, const Tensor & vec, const Scalar & beta, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& self2_ = unpack(self2, "self2", 1);
  auto& mat_ = unpack(mat, "mat", 2);
  auto& vec_ = unpack(vec, "vec", 3);
  auto _any_requires_grad = compute_requires_grad( self, self2, mat, vec );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_addmv_impl_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, self2, mat, vec ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> self2__storage_saved =
    self2_.has_storage() ? c10::optional<Storage>(self2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self2__impl_saved;
  if (self2_.defined()) self2__impl_saved = self2_.getIntrusivePtr();
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
    at::redispatch::_addmv_impl_(ks & c10::after_autograd_keyset, self_, self2_, mat_, vec_, beta, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (self2__storage_saved.has_value())
    AT_ASSERT(self2__storage_saved.value().is_alias_of(self2_.storage()));
  if (self2__impl_saved) AT_ASSERT(self2__impl_saved == self2_.getIntrusivePtr());
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
std::tuple<Tensor,Tensor> _ctc_loss(c10::DispatchKeySet ks, const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity) {
  auto& log_probs_ = unpack(log_probs, "log_probs", 0);
  auto& targets_ = unpack(targets, "targets", 1);
  auto _any_requires_grad = compute_requires_grad( log_probs );
  (void)_any_requires_grad;
  check_no_requires_grad(targets, "targets");
  std::shared_ptr<CtcLossBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CtcLossBackward>(new CtcLossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( log_probs ));
    grad_fn->log_probs_ = SavedVariable(log_probs, false);
    grad_fn->targets_ = SavedVariable(targets, false);
    grad_fn->input_lengths = input_lengths.vec();
    grad_fn->target_lengths = target_lengths.vec();
    grad_fn->blank = blank;
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
    return at::redispatch::_ctc_loss(ks & c10::after_autograd_keyset, log_probs_, targets_, input_lengths, target_lengths, blank, zero_infinity);
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
  throw_error_for_complex_autograd(result0, "_ctc_loss");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor _cudnn_init_dropout_state(c10::DispatchKeySet ks, double dropout, bool train, int64_t dropout_seed, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_cudnn_init_dropout_state(ks & c10::after_autograd_keyset, dropout, train, dropout_seed, dtype, layout, device, pin_memory);
  })();
  auto result = std::move(_tmp);
  return result;
}
Tensor _cudnn_rnn_flatten_weight(c10::DispatchKeySet ks, TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, bool bidirectional) {
  auto weight_arr_ = unpack(weight_arr, "weight_arr", 0);
  auto _any_requires_grad = compute_requires_grad( weight_arr );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cudnn_rnn_flatten_weight"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( weight_arr ));
  }
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_cudnn_rnn_flatten_weight(ks & c10::after_autograd_keyset, weight_arr_, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional);
  })();
  auto result = std::move(_tmp);
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_cudnn_rnn_flatten_weight");
  return result;
}
Tensor _embedding_bag_dense_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<Tensor> & per_sample_weights) {
  auto& grad_ = unpack(grad, "grad", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& offsets_ = unpack(offsets, "offsets", 2);
  auto& offset2bag_ = unpack(offset2bag, "offset2bag", 3);
  auto& bag_size_ = unpack(bag_size, "bag_size", 4);
  auto& maximum_indices_ = unpack(maximum_indices, "maximum_indices", 5);
  auto _any_requires_grad = compute_requires_grad( grad, per_sample_weights );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_embedding_bag_dense_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad, per_sample_weights ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> offsets__storage_saved =
    offsets_.has_storage() ? c10::optional<Storage>(offsets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> offsets__impl_saved;
  if (offsets_.defined()) offsets__impl_saved = offsets_.getIntrusivePtr();
  c10::optional<Storage> offset2bag__storage_saved =
    offset2bag_.has_storage() ? c10::optional<Storage>(offset2bag_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> offset2bag__impl_saved;
  if (offset2bag_.defined()) offset2bag__impl_saved = offset2bag_.getIntrusivePtr();
  c10::optional<Storage> bag_size__storage_saved =
    bag_size_.has_storage() ? c10::optional<Storage>(bag_size_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bag_size__impl_saved;
  if (bag_size_.defined()) bag_size__impl_saved = bag_size_.getIntrusivePtr();
  c10::optional<Storage> maximum_indices__storage_saved =
    maximum_indices_.has_storage() ? c10::optional<Storage>(maximum_indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> maximum_indices__impl_saved;
  if (maximum_indices_.defined()) maximum_indices__impl_saved = maximum_indices_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_embedding_bag_dense_backward(ks & c10::after_autograd_keyset, grad_, indices_, offsets_, offset2bag_, bag_size_, maximum_indices_, num_weights, scale_grad_by_freq, mode, per_sample_weights);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (offsets__storage_saved.has_value())
    AT_ASSERT(offsets__storage_saved.value().is_alias_of(offsets_.storage()));
  if (offsets__impl_saved) AT_ASSERT(offsets__impl_saved == offsets_.getIntrusivePtr());
  if (offset2bag__storage_saved.has_value())
    AT_ASSERT(offset2bag__storage_saved.value().is_alias_of(offset2bag_.storage()));
  if (offset2bag__impl_saved) AT_ASSERT(offset2bag__impl_saved == offset2bag_.getIntrusivePtr());
  if (bag_size__storage_saved.has_value())
    AT_ASSERT(bag_size__storage_saved.value().is_alias_of(bag_size_.storage()));
  if (bag_size__impl_saved) AT_ASSERT(bag_size__impl_saved == bag_size_.getIntrusivePtr());
  if (maximum_indices__storage_saved.has_value())
    AT_ASSERT(maximum_indices__storage_saved.value().is_alias_of(maximum_indices_.storage()));
  if (maximum_indices__impl_saved) AT_ASSERT(maximum_indices__impl_saved == maximum_indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_embedding_bag_dense_backward");
  return result;
}
Tensor _embedding_bag_per_sample_weights_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & weight, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, int64_t mode) {
  auto& grad_ = unpack(grad, "grad", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  auto& offsets_ = unpack(offsets, "offsets", 3);
  auto& offset2bag_ = unpack(offset2bag, "offset2bag", 4);
  auto _any_requires_grad = compute_requires_grad( grad, weight, indices, offsets, offset2bag );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_embedding_bag_per_sample_weights_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad, weight, indices, offsets, offset2bag ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> offsets__storage_saved =
    offsets_.has_storage() ? c10::optional<Storage>(offsets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> offsets__impl_saved;
  if (offsets_.defined()) offsets__impl_saved = offsets_.getIntrusivePtr();
  c10::optional<Storage> offset2bag__storage_saved =
    offset2bag_.has_storage() ? c10::optional<Storage>(offset2bag_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> offset2bag__impl_saved;
  if (offset2bag_.defined()) offset2bag__impl_saved = offset2bag_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_embedding_bag_per_sample_weights_backward(ks & c10::after_autograd_keyset, grad_, weight_, indices_, offsets_, offset2bag_, mode);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (offsets__storage_saved.has_value())
    AT_ASSERT(offsets__storage_saved.value().is_alias_of(offsets_.storage()));
  if (offsets__impl_saved) AT_ASSERT(offsets__impl_saved == offsets_.getIntrusivePtr());
  if (offset2bag__storage_saved.has_value())
    AT_ASSERT(offset2bag__storage_saved.value().is_alias_of(offset2bag_.storage()));
  if (offset2bag__impl_saved) AT_ASSERT(offset2bag__impl_saved == offset2bag_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_embedding_bag_per_sample_weights_backward");
  return result;
}
Tensor _empty_per_channel_affine_quantized(c10::DispatchKeySet ks, IntArrayRef size, const Tensor & scales, const Tensor & zero_points, int64_t axis, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
  auto& scales_ = unpack(scales, "scales", 1);
  auto& zero_points_ = unpack(zero_points, "zero_points", 2);
  auto _any_requires_grad = compute_requires_grad( scales, zero_points );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_empty_per_channel_affine_quantized"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( scales, zero_points ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> scales__storage_saved =
    scales_.has_storage() ? c10::optional<Storage>(scales_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> scales__impl_saved;
  if (scales_.defined()) scales__impl_saved = scales_.getIntrusivePtr();
  c10::optional<Storage> zero_points__storage_saved =
    zero_points_.has_storage() ? c10::optional<Storage>(zero_points_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> zero_points__impl_saved;
  if (zero_points_.defined()) zero_points__impl_saved = zero_points_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_empty_per_channel_affine_quantized(ks & c10::after_autograd_keyset, size, scales_, zero_points_, axis, dtype, layout, device, pin_memory, memory_format);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (scales__storage_saved.has_value())
    AT_ASSERT(scales__storage_saved.value().is_alias_of(scales_.storage()));
  if (scales__impl_saved) AT_ASSERT(scales__impl_saved == scales_.getIntrusivePtr());
  if (zero_points__storage_saved.has_value())
    AT_ASSERT(zero_points__storage_saved.value().is_alias_of(zero_points_.storage()));
  if (zero_points__impl_saved) AT_ASSERT(zero_points__impl_saved == zero_points_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_empty_per_channel_affine_quantized");
  return result;
}
Tensor _fake_quantize_learnable_per_tensor_affine(c10::DispatchKeySet ks, const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) {
  auto& self_ = unpack(self, "self", 0);
  auto& scale_ = unpack(scale, "scale", 1);
  auto& zero_point_ = unpack(zero_point, "zero_point", 2);
  auto _any_requires_grad = compute_requires_grad( self, scale, zero_point );
  (void)_any_requires_grad;
  std::shared_ptr<FakeQuantizeLearnablePerTensorAffineBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FakeQuantizeLearnablePerTensorAffineBackward>(new FakeQuantizeLearnablePerTensorAffineBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, scale, zero_point ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->scale_ = SavedVariable(scale, false);
    grad_fn->zero_point_ = SavedVariable(zero_point, false);
    grad_fn->quant_min = quant_min;
    grad_fn->quant_max = quant_max;
    grad_fn->grad_factor = grad_factor;
  }
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
    return at::redispatch::_fake_quantize_learnable_per_tensor_affine(ks & c10::after_autograd_keyset, self_, scale_, zero_point_, quant_min, quant_max, grad_factor);
  })();
  auto result = std::move(_tmp);
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
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_fake_quantize_learnable_per_tensor_affine");
  return result;
}
Tensor _fft_c2c(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, int64_t normalization, bool forward) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<FftC2CBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FftC2CBackward>(new FftC2CBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim.vec();
    grad_fn->normalization = normalization;
    grad_fn->forward = forward;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_fft_c2c(ks & c10::after_autograd_keyset, self_, dim, normalization, forward);
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
Tensor _fft_c2r(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<FftC2RBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FftC2RBackward>(new FftC2RBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim.vec();
    grad_fn->normalization = normalization;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_fft_c2r(ks & c10::after_autograd_keyset, self_, dim, normalization, last_dim_size);
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
  throw_error_for_complex_autograd(result, "_fft_c2r");
  return result;
}
Tensor _fft_r2c(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, int64_t normalization, bool onesided) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<FftR2CBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FftR2CBackward>(new FftR2CBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim.vec();
    grad_fn->normalization = normalization;
    grad_fn->onesided = onesided;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_fft_r2c(ks & c10::after_autograd_keyset, self_, dim, normalization, onesided);
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
std::vector<Tensor> _foreach_abs(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_abs"), deleteNode);
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
    return at::redispatch::_foreach_abs(ks & c10::after_autograd_keyset, tensors_);
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
void _foreach_abs_(c10::DispatchKeySet ks, TensorList self) {
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
    at::redispatch::_foreach_abs_(ks & c10::after_autograd_keyset, self_);
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
std::vector<Tensor> _foreach_addcmul_Scalar(c10::DispatchKeySet ks, TensorList input, TensorList tensor1, TensorList tensor2, const Scalar & value) {
  auto input_ = unpack(input, "input", 0);
  auto tensor1_ = unpack(tensor1, "tensor1", 1);
  auto tensor2_ = unpack(tensor2, "tensor2", 2);
  auto _any_requires_grad = compute_requires_grad( input, tensor1, tensor2 );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_addcmul"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, tensor1, tensor2 ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> input__storage_saved(input_.size());
  for (const Tensor& tensor : input_)
    input__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> input__impl_saved(input_.size());
  for (size_t i=0; i<input_.size(); i++)
    if (input_[i].defined()) input__impl_saved[i] = input_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor1__storage_saved(tensor1_.size());
  for (const Tensor& tensor : tensor1_)
    tensor1__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor1__impl_saved(tensor1_.size());
  for (size_t i=0; i<tensor1_.size(); i++)
    if (tensor1_[i].defined()) tensor1__impl_saved[i] = tensor1_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor2__storage_saved(tensor2_.size());
  for (const Tensor& tensor : tensor2_)
    tensor2__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor2__impl_saved(tensor2_.size());
  for (size_t i=0; i<tensor2_.size(); i++)
    if (tensor2_[i].defined()) tensor2__impl_saved[i] = tensor2_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_addcmul(ks & c10::after_autograd_keyset, input_, tensor1_, tensor2_, value);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<input_.size(); i++) {
    if (input__storage_saved[i].has_value())
      AT_ASSERT(input__storage_saved[i].value().is_alias_of(input_[i].storage()));
  }
  for (size_t i=0; i<input_.size(); i++) {
    if (input__impl_saved[i])
      AT_ASSERT(input__impl_saved[i] == input_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__storage_saved[i].has_value())
      AT_ASSERT(tensor1__storage_saved[i].value().is_alias_of(tensor1_[i].storage()));
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__impl_saved[i])
      AT_ASSERT(tensor1__impl_saved[i] == tensor1_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__storage_saved[i].has_value())
      AT_ASSERT(tensor2__storage_saved[i].value().is_alias_of(tensor2_[i].storage()));
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__impl_saved[i])
      AT_ASSERT(tensor2__impl_saved[i] == tensor2_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
std::vector<Tensor> _foreach_addcmul_ScalarList(c10::DispatchKeySet ks, TensorList input, TensorList tensor1, TensorList tensor2, ArrayRef<Scalar> scalars) {
  auto input_ = unpack(input, "input", 0);
  auto tensor1_ = unpack(tensor1, "tensor1", 1);
  auto tensor2_ = unpack(tensor2, "tensor2", 2);
  auto _any_requires_grad = compute_requires_grad( input, tensor1, tensor2 );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_addcmul"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, tensor1, tensor2 ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> input__storage_saved(input_.size());
  for (const Tensor& tensor : input_)
    input__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> input__impl_saved(input_.size());
  for (size_t i=0; i<input_.size(); i++)
    if (input_[i].defined()) input__impl_saved[i] = input_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor1__storage_saved(tensor1_.size());
  for (const Tensor& tensor : tensor1_)
    tensor1__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor1__impl_saved(tensor1_.size());
  for (size_t i=0; i<tensor1_.size(); i++)
    if (tensor1_[i].defined()) tensor1__impl_saved[i] = tensor1_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor2__storage_saved(tensor2_.size());
  for (const Tensor& tensor : tensor2_)
    tensor2__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor2__impl_saved(tensor2_.size());
  for (size_t i=0; i<tensor2_.size(); i++)
    if (tensor2_[i].defined()) tensor2__impl_saved[i] = tensor2_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_foreach_addcmul(ks & c10::after_autograd_keyset, input_, tensor1_, tensor2_, scalars);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<input_.size(); i++) {
    if (input__storage_saved[i].has_value())
      AT_ASSERT(input__storage_saved[i].value().is_alias_of(input_[i].storage()));
  }
  for (size_t i=0; i<input_.size(); i++) {
    if (input__impl_saved[i])
      AT_ASSERT(input__impl_saved[i] == input_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__storage_saved[i].has_value())
      AT_ASSERT(tensor1__storage_saved[i].value().is_alias_of(tensor1_[i].storage()));
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__impl_saved[i])
      AT_ASSERT(tensor1__impl_saved[i] == tensor1_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__storage_saved[i].has_value())
      AT_ASSERT(tensor2__storage_saved[i].value().is_alias_of(tensor2_[i].storage()));
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__impl_saved[i])
      AT_ASSERT(tensor2__impl_saved[i] == tensor2_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
void _foreach_addcmul__Scalar(c10::DispatchKeySet ks, TensorList self, TensorList tensor1, TensorList tensor2, const Scalar & value) {
  auto self_ = unpack(self, "self", 0);
  auto tensor1_ = unpack(tensor1, "tensor1", 1);
  auto tensor2_ = unpack(tensor2, "tensor2", 2);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor1__storage_saved(tensor1_.size());
  for (const Tensor& tensor : tensor1_)
    tensor1__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor1__impl_saved(tensor1_.size());
  for (size_t i=0; i<tensor1_.size(); i++)
    if (tensor1_[i].defined()) tensor1__impl_saved[i] = tensor1_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor2__storage_saved(tensor2_.size());
  for (const Tensor& tensor : tensor2_)
    tensor2__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor2__impl_saved(tensor2_.size());
  for (size_t i=0; i<tensor2_.size(); i++)
    if (tensor2_[i].defined()) tensor2__impl_saved[i] = tensor2_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_addcmul_(ks & c10::after_autograd_keyset, self_, tensor1_, tensor2_, value);
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
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__storage_saved[i].has_value())
      AT_ASSERT(tensor1__storage_saved[i].value().is_alias_of(tensor1_[i].storage()));
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__impl_saved[i])
      AT_ASSERT(tensor1__impl_saved[i] == tensor1_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__storage_saved[i].has_value())
      AT_ASSERT(tensor2__storage_saved[i].value().is_alias_of(tensor2_[i].storage()));
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__impl_saved[i])
      AT_ASSERT(tensor2__impl_saved[i] == tensor2_[i].getIntrusivePtr());
  }
  #endif
}
void _foreach_addcmul__ScalarList(c10::DispatchKeySet ks, TensorList self, TensorList tensor1, TensorList tensor2, ArrayRef<Scalar> scalars) {
  auto self_ = unpack(self, "self", 0);
  auto tensor1_ = unpack(tensor1, "tensor1", 1);
  auto tensor2_ = unpack(tensor2, "tensor2", 2);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor1__storage_saved(tensor1_.size());
  for (const Tensor& tensor : tensor1_)
    tensor1__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor1__impl_saved(tensor1_.size());
  for (size_t i=0; i<tensor1_.size(); i++)
    if (tensor1_[i].defined()) tensor1__impl_saved[i] = tensor1_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor2__storage_saved(tensor2_.size());
  for (const Tensor& tensor : tensor2_)
    tensor2__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor2__impl_saved(tensor2_.size());
  for (size_t i=0; i<tensor2_.size(); i++)
    if (tensor2_[i].defined()) tensor2__impl_saved[i] = tensor2_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_foreach_addcmul_(ks & c10::after_autograd_keyset, self_, tensor1_, tensor2_, scalars);
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
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__storage_saved[i].has_value())
      AT_ASSERT(tensor1__storage_saved[i].value().is_alias_of(tensor1_[i].storage()));
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__impl_saved[i])
      AT_ASSERT(tensor1__impl_saved[i] == tensor1_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__storage_saved[i].has_value())
      AT_ASSERT(tensor2__storage_saved[i].value().is_alias_of(tensor2_[i].storage()));
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__impl_saved[i])
      AT_ASSERT(tensor2__impl_saved[i] == tensor2_[i].getIntrusivePtr());
  }
  #endif
}
std::vector<Tensor> _foreach_atan(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_atan"), deleteNode);
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
    return at::redispatch::_foreach_atan(ks & c10::after_autograd_keyset, tensors_);
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
void _foreach_atan_(c10::DispatchKeySet ks, TensorList self) {
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
    at::redispatch::_foreach_atan_(ks & c10::after_autograd_keyset, self_);
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
std::vector<Tensor> _foreach_cos(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_cos"), deleteNode);
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
    return at::redispatch::_foreach_cos(ks & c10::after_autograd_keyset, tensors_);
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
void _foreach_cos_(c10::DispatchKeySet ks, TensorList self) {
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
    at::redispatch::_foreach_cos_(ks & c10::after_autograd_keyset, self_);
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
std::vector<Tensor> _foreach_reciprocal(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_reciprocal"), deleteNode);
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
    return at::redispatch::_foreach_reciprocal(ks & c10::after_autograd_keyset, tensors_);
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
void _foreach_reciprocal_(c10::DispatchKeySet ks, TensorList self) {
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
    at::redispatch::_foreach_reciprocal_(ks & c10::after_autograd_keyset, self_);
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
std::vector<Tensor> _foreach_sin(c10::DispatchKeySet ks, TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_sin"), deleteNode);
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
    return at::redispatch::_foreach_sin(ks & c10::after_autograd_keyset, tensors_);
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
void _foreach_sin_(c10::DispatchKeySet ks, TensorList self) {
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
    at::redispatch::_foreach_sin_(ks & c10::after_autograd_keyset, self_);
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
std::vector<Tensor> _foreach_sub_Scalar(c10::DispatchKeySet ks, TensorList tensors, const Scalar & scalar) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_sub"), deleteNode);
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
    return at::redispatch::_foreach_sub(ks & c10::after_autograd_keyset, tensors_, scalar);
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
std::vector<Tensor> _foreach_sub_List(c10::DispatchKeySet ks, TensorList tensors1, TensorList tensors2, const Scalar & alpha) {
  auto tensors1_ = unpack(tensors1, "tensors1", 0);
  auto tensors2_ = unpack(tensors2, "tensors2", 1);
  auto _any_requires_grad = compute_requires_grad( tensors1, tensors2 );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_sub"), deleteNode);
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
    return at::redispatch::_foreach_sub(ks & c10::after_autograd_keyset, tensors1_, tensors2_, alpha);
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
std::vector<Tensor> _foreach_sub_ScalarList(c10::DispatchKeySet ks, TensorList tensors, ArrayRef<Scalar> scalars) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_sub"), deleteNode);
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
    return at::redispatch::_foreach_sub(ks & c10::after_autograd_keyset, tensors_, scalars);
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
void _foreach_sub__Scalar(c10::DispatchKeySet ks, TensorList self, const Scalar & scalar) {
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
    at::redispatch::_foreach_sub_(ks & c10::after_autograd_keyset, self_, scalar);
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
void _foreach_sub__List(c10::DispatchKeySet ks, TensorList self, TensorList other, const Scalar & alpha) {
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
    at::redispatch::_foreach_sub_(ks & c10::after_autograd_keyset, self_, other_, alpha);
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
void _foreach_sub__ScalarList(c10::DispatchKeySet ks, TensorList self, ArrayRef<Scalar> scalars) {
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
    at::redispatch::_foreach_sub_(ks & c10::after_autograd_keyset, self_, scalars);
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
Tensor & _linalg_inv_out_helper_(c10::DispatchKeySet ks, Tensor & self, Tensor & infos_lu, Tensor & infos_getri) {
  auto& self_ = unpack(self, "self", 0);
  auto& infos_lu_ = unpack(infos_lu, "infos_lu", 1);
  auto& infos_getri_ = unpack(infos_getri, "infos_getri", 2);
  auto _any_requires_grad = compute_requires_grad( self, infos_lu, infos_getri );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_linalg_inv_out_helper_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, infos_lu, infos_getri ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> infos_lu__storage_saved =
    infos_lu_.has_storage() ? c10::optional<Storage>(infos_lu_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> infos_lu__impl_saved;
  if (infos_lu_.defined()) infos_lu__impl_saved = infos_lu_.getIntrusivePtr();
  c10::optional<Storage> infos_getri__storage_saved =
    infos_getri_.has_storage() ? c10::optional<Storage>(infos_getri_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> infos_getri__impl_saved;
  if (infos_getri_.defined()) infos_getri__impl_saved = infos_getri_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_linalg_inv_out_helper_(ks & c10::after_autograd_keyset, self_, infos_lu_, infos_getri_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (infos_lu__storage_saved.has_value())
    AT_ASSERT(infos_lu__storage_saved.value().is_alias_of(infos_lu_.storage()));
  if (infos_lu__impl_saved) AT_ASSERT(infos_lu__impl_saved == infos_lu_.getIntrusivePtr());
  if (infos_getri__storage_saved.has_value())
    AT_ASSERT(infos_getri__storage_saved.value().is_alias_of(infos_getri_.storage()));
  if (infos_getri__impl_saved) AT_ASSERT(infos_getri__impl_saved == infos_getri_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
std::tuple<Tensor,Tensor> _linalg_qr_helper(c10::DispatchKeySet ks, const Tensor & self, std::string mode) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_linalg_qr_helper"), deleteNode);
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
    return at::redispatch::_linalg_qr_helper(ks & c10::after_autograd_keyset, self_, mode);
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
  throw_error_for_complex_autograd(result0, "_linalg_qr_helper");
  throw_error_for_complex_autograd(result1, "_linalg_qr_helper");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & _linalg_solve_out_helper_(c10::DispatchKeySet ks, Tensor & self, Tensor & other, Tensor & infos) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& infos_ = unpack(infos, "infos", 2);
  auto _any_requires_grad = compute_requires_grad( self, other, infos );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_linalg_solve_out_helper_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other, infos ));
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
  c10::optional<Storage> infos__storage_saved =
    infos_.has_storage() ? c10::optional<Storage>(infos_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> infos__impl_saved;
  if (infos_.defined()) infos__impl_saved = infos_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::_linalg_solve_out_helper_(ks & c10::after_autograd_keyset, self_, other_, infos_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (infos__storage_saved.has_value())
    AT_ASSERT(infos__storage_saved.value().is_alias_of(infos_.storage()));
  if (infos__impl_saved) AT_ASSERT(infos__impl_saved == infos_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor _log_softmax_backward_data(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  auto& self_ = unpack(self, "self", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<LogSoftmaxBackwardDataBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogSoftmaxBackwardDataBackward>(new LogSoftmaxBackwardDataBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->dim = dim;
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
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
    return at::redispatch::_log_softmax_backward_data(ks & c10::after_autograd_keyset, grad_output_, output_, dim, self_);
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
  throw_error_for_complex_autograd(result, "_log_softmax_backward_data");
  return result;
}
Tensor _lu_solve_helper(c10::DispatchKeySet ks, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  auto& self_ = unpack(self, "self", 0);
  auto& LU_data_ = unpack(LU_data, "LU_data", 1);
  auto& LU_pivots_ = unpack(LU_pivots, "LU_pivots", 2);
  auto _any_requires_grad = compute_requires_grad( self, LU_data, LU_pivots );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_lu_solve_helper"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, LU_data, LU_pivots ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> LU_data__storage_saved =
    LU_data_.has_storage() ? c10::optional<Storage>(LU_data_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_data__impl_saved;
  if (LU_data_.defined()) LU_data__impl_saved = LU_data_.getIntrusivePtr();
  c10::optional<Storage> LU_pivots__storage_saved =
    LU_pivots_.has_storage() ? c10::optional<Storage>(LU_pivots_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_pivots__impl_saved;
  if (LU_pivots_.defined()) LU_pivots__impl_saved = LU_pivots_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_lu_solve_helper(ks & c10::after_autograd_keyset, self_, LU_data_, LU_pivots_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (LU_data__storage_saved.has_value())
    AT_ASSERT(LU_data__storage_saved.value().is_alias_of(LU_data_.storage()));
  if (LU_data__impl_saved) AT_ASSERT(LU_data__impl_saved == LU_data_.getIntrusivePtr());
  if (LU_pivots__storage_saved.has_value())
    AT_ASSERT(LU_pivots__storage_saved.value().is_alias_of(LU_pivots_.storage()));
  if (LU_pivots__impl_saved) AT_ASSERT(LU_pivots__impl_saved == LU_pivots_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_lu_solve_helper");
  return result;
}
Tensor _make_per_channel_quantized_tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis) {
  auto& self_ = unpack(self, "self", 0);
  auto& scale_ = unpack(scale, "scale", 1);
  auto& zero_point_ = unpack(zero_point, "zero_point", 2);
  auto _any_requires_grad = compute_requires_grad( self, scale, zero_point );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_make_per_channel_quantized_tensor"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, scale, zero_point ));
  }
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
    return at::redispatch::_make_per_channel_quantized_tensor(ks & c10::after_autograd_keyset, self_, scale_, zero_point_, axis);
  })();
  auto result = std::move(_tmp);
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
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_make_per_channel_quantized_tensor");
  return result;
}
Tensor _mkldnn_reshape(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef shape) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MkldnnReshapeBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MkldnnReshapeBackward>(new MkldnnReshapeBackward(), deleteNode);
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
    return at::redispatch::_mkldnn_reshape(ks & c10::after_autograd_keyset, self_, shape);
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
  throw_error_for_complex_autograd(result, "_mkldnn_reshape");
  return result;
}
std::tuple<Tensor,Tensor> _solve_helper(c10::DispatchKeySet ks, const Tensor & self, const Tensor & A) {
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  auto _any_requires_grad = compute_requires_grad( self, A );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_solve_helper"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  Tensor result0;
  Tensor result1;
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
    return at::redispatch::_solve_helper(ks & c10::after_autograd_keyset, self_, A_);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_solve_helper");
  throw_error_for_complex_autograd(result1, "_solve_helper");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & _stack_out_out(c10::DispatchKeySet ks, TensorList tensors, int64_t dim, Tensor & out) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( tensors )) {
    throw_error_out_requires_grad("_stack");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_stack");
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
    at::redispatch::_stack_outf(ks & c10::after_autograd_keyset, tensors_, dim, out_);
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
Tensor _standard_gamma_grad(c10::DispatchKeySet ks, const Tensor & self, const Tensor & output) {
  auto& self_ = unpack(self, "self", 0);
  auto& output_ = unpack(output, "output", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<StandardGammaGradBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<StandardGammaGradBackward>(new StandardGammaGradBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_standard_gamma_grad(ks & c10::after_autograd_keyset, self_, output_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_standard_gamma_grad");
  return result;
}
std::tuple<Tensor,Tensor,Tensor> _svd_helper(c10::DispatchKeySet ks, const Tensor & self, bool some, bool compute_uv) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SvdHelperBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SvdHelperBackward>(new SvdHelperBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->some = some;
    grad_fn->compute_uv = compute_uv;
  }
  Tensor U;
  Tensor S;
  Tensor V;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::_svd_helper(ks & c10::after_autograd_keyset, self_, some, compute_uv);
  })();
  std::tie(U, S, V) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( U, S, V ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->U_ = SavedVariable(U, true);
    grad_fn->S_ = SavedVariable(S, true);
    grad_fn->V_ = SavedVariable(V, true);
  }
  return std::make_tuple(std::move(U), std::move(S), std::move(V));
}
Tensor _test_optional_intlist(c10::DispatchKeySet ks, const Tensor & values, c10::optional<IntArrayRef> addends) {
  auto& values_ = unpack(values, "values", 0);
  auto _any_requires_grad = compute_requires_grad( values );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_test_optional_intlist"), deleteNode);
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
    return at::redispatch::_test_optional_intlist(ks & c10::after_autograd_keyset, values_, addends);
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
  throw_error_for_complex_autograd(result, "_test_optional_intlist");
  return result;
}
std::tuple<Tensor,Tensor> _unique(c10::DispatchKeySet ks, const Tensor & self, bool sorted, bool return_inverse) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<UniqueBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UniqueBackward>(new UniqueBackward(), deleteNode);
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
    return at::redispatch::_unique(ks & c10::after_autograd_keyset, self_, sorted, return_inverse);
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
  throw_error_for_complex_autograd(result0, "_unique");
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor,Tensor> _unique2(c10::DispatchKeySet ks, const Tensor & self, bool sorted, bool return_inverse, bool return_counts) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Unique2Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Unique2Backward>(new Unique2Backward(), deleteNode);
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
    return at::redispatch::_unique2(ks & c10::after_autograd_keyset, self_, sorted, return_inverse, return_counts);
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
  throw_error_for_complex_autograd(result0, "_unique2");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor adaptive_avg_pool3d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<AdaptiveAvgPool3DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AdaptiveAvgPool3DBackwardBackward>(new AdaptiveAvgPool3DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
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
    return at::redispatch::adaptive_avg_pool3d_backward(ks & c10::after_autograd_keyset, grad_output_, self_);
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
  throw_error_for_complex_autograd(result, "adaptive_avg_pool3d_backward");
  return result;
}
std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, Tensor & out, Tensor & indices) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_max_pool2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("adaptive_max_pool2d");
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
    at::redispatch::adaptive_max_pool2d_outf(ks & c10::after_autograd_keyset, self_, output_size, out_, indices_);
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
Tensor addbmm(c10::DispatchKeySet ks, const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  auto _any_requires_grad = compute_requires_grad( self, batch1, batch2 );
  (void)_any_requires_grad;
  std::shared_ptr<AddbmmBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddbmmBackward>(new AddbmmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    grad_fn->batch1_argsize_0 = batch1.size(0);
    grad_fn->batch1_argsize_1 = batch1.size(1);
    grad_fn->batch2_argsize_2 = batch2.size(2);
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
    return at::redispatch::addbmm(ks & c10::after_autograd_keyset, self_, batch1_, batch2_, beta, alpha);
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
Tensor & addbmm_(c10::DispatchKeySet ks, Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  auto _any_requires_grad = compute_requires_grad( self, batch1, batch2 );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<AddbmmBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddbmmBackward>(new AddbmmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    grad_fn->batch1_argsize_0 = batch1.size(0);
    grad_fn->batch1_argsize_1 = batch1.size(1);
    grad_fn->batch2_argsize_2 = batch2.size(2);
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
    at::redispatch::addbmm_(ks & c10::after_autograd_keyset, self_, batch1_, batch2_, beta, alpha);
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
Tensor & addcdiv_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self, tensor1, tensor2 );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    throw_error_out_requires_grad("addcdiv");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addcdiv");
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
    at::redispatch::addcdiv_outf(ks & c10::after_autograd_keyset, self_, tensor1_, tensor2_, value, out_);
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
Tensor & addmv_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat, const Tensor & vec, const Scalar & beta, const Scalar & alpha, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& mat_ = unpack(mat, "mat", 1);
  auto& vec_ = unpack(vec, "vec", 2);
  auto& out_ = unpack(out, "out", 5);
  auto _any_requires_grad = compute_requires_grad( self, mat, vec );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    throw_error_out_requires_grad("addmv");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addmv");
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
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::addmv_outf(ks & c10::after_autograd_keyset, self_, mat_, vec_, beta, alpha, out_);
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
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor all_dim(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AllBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AllBackward1>(new AllBackward1(), deleteNode);
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
    return at::redispatch::all(ks & c10::after_autograd_keyset, self_, dim, keepdim);
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
  throw_error_for_complex_autograd(result, "all");
  return result;
}
Tensor all(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AllBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AllBackward0>(new AllBackward0(), deleteNode);
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
    return at::redispatch::all(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "all");
  return result;
}
Tensor amax(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AmaxBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AmaxBackward>(new AmaxBackward(), deleteNode);
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
    return at::redispatch::amax(ks & c10::after_autograd_keyset, self_, dim, keepdim);
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
  throw_error_for_complex_autograd(result, "amax");
  return result;
}
Tensor & angle_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("angle");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("angle");
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
    at::redispatch::angle_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor any_dim(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AnyBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AnyBackward1>(new AnyBackward1(), deleteNode);
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
    return at::redispatch::any(ks & c10::after_autograd_keyset, self_, dim, keepdim);
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
  throw_error_for_complex_autograd(result, "any");
  return result;
}
Tensor any(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AnyBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AnyBackward0>(new AnyBackward0(), deleteNode);
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
    return at::redispatch::any(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "any");
  return result;
}
Tensor argmin(c10::DispatchKeySet ks, const Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::argmin(ks & c10::after_autograd_keyset, self_, dim, keepdim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor as_strided(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AsStridedBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AsStridedBackward>(new AsStridedBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_geometry = TensorGeometry(self);
    grad_fn->size = size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->storage_offset = storage_offset;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoNonVariableTypeMode guard;
    return at::redispatch::as_strided(ks & c10::after_autograd_keyset, self_, size, stride, storage_offset);
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
Tensor & as_strided_(c10::DispatchKeySet ks, Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<AsStridedBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AsStridedBackward>(new AsStridedBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_geometry = TensorGeometry(self);
    grad_fn->size = size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->storage_offset = storage_offset;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::as_strided_(ks & c10::after_autograd_keyset, self_, size, stride, storage_offset);
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
Tensor & atanh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("atanh");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("atanh");
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
    at::redispatch::atanh_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor batch_norm_elemt(c10::DispatchKeySet ks, const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const Tensor & mean, const Tensor & invstd, double eps) {
  auto& input_ = unpack(input, "input", 0);
  auto& mean_ = unpack(mean, "mean", 3);
  auto& invstd_ = unpack(invstd, "invstd", 4);
  auto _any_requires_grad = compute_requires_grad( input, weight, bias, mean, invstd );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_elemt"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias, mean, invstd ));
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
  c10::optional<Storage> invstd__storage_saved =
    invstd_.has_storage() ? c10::optional<Storage>(invstd_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> invstd__impl_saved;
  if (invstd_.defined()) invstd__impl_saved = invstd_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::batch_norm_elemt(ks & c10::after_autograd_keyset, input_, weight, bias, mean_, invstd_, eps);
  })();
  auto result = std::move(_tmp);
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
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "batch_norm_elemt");
  return result;
}
Tensor & binary_cross_entropy_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, target, weight )) {
    throw_error_out_requires_grad("binary_cross_entropy");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("binary_cross_entropy");
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
    at::redispatch::binary_cross_entropy_outf(ks & c10::after_autograd_keyset, self_, target_, weight, reduction, out_);
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
Tensor binomial(c10::DispatchKeySet ks, const Tensor & count, const Tensor & prob, c10::optional<Generator> generator) {
  auto& count_ = unpack(count, "count", 0);
  auto& prob_ = unpack(prob, "prob", 1);
  auto _any_requires_grad = compute_requires_grad( count, prob );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("binomial"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( count, prob ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> count__storage_saved =
    count_.has_storage() ? c10::optional<Storage>(count_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> count__impl_saved;
  if (count_.defined()) count__impl_saved = count_.getIntrusivePtr();
  c10::optional<Storage> prob__storage_saved =
    prob_.has_storage() ? c10::optional<Storage>(prob_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> prob__impl_saved;
  if (prob_.defined()) prob__impl_saved = prob_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::binomial(ks & c10::after_autograd_keyset, count_, prob_, generator);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (count__storage_saved.has_value())
    AT_ASSERT(count__storage_saved.value().is_alias_of(count_.storage()));
  if (count__impl_saved) AT_ASSERT(count__impl_saved == count_.getIntrusivePtr());
  if (prob__storage_saved.has_value())
    AT_ASSERT(prob__storage_saved.value().is_alias_of(prob_.storage()));
  if (prob__impl_saved) AT_ASSERT(prob__impl_saved == prob_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "binomial");
  return result;
}
Tensor & bitwise_or_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("bitwise_or");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_or");
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
    at::redispatch::bitwise_or_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
Tensor & bitwise_or_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("bitwise_or");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_or");
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
    at::redispatch::bitwise_or_outf(ks & c10::after_autograd_keyset, self_, other, out_);
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
Tensor & bitwise_xor_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("bitwise_xor");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_xor");
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
    at::redispatch::bitwise_xor_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
Tensor & bitwise_xor_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("bitwise_xor");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_xor");
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
    at::redispatch::bitwise_xor_outf(ks & c10::after_autograd_keyset, self_, other, out_);
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
Tensor ceil(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<CeilBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CeilBackward>(new CeilBackward(), deleteNode);
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
    return at::redispatch::ceil(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "ceil");
  return result;
}
Tensor & ceil_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<CeilBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CeilBackward>(new CeilBackward(), deleteNode);
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
    at::redispatch::ceil_(ks & c10::after_autograd_keyset, self_);
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
Tensor channel_shuffle(c10::DispatchKeySet ks, const Tensor & self, int64_t groups) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("channel_shuffle"), deleteNode);
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
    return at::redispatch::channel_shuffle(ks & c10::after_autograd_keyset, self_, groups);
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
  throw_error_for_complex_autograd(result, "channel_shuffle");
  return result;
}
Tensor & cholesky_solve_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & input2, bool upper, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self, input2 );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    throw_error_out_requires_grad("cholesky_solve");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cholesky_solve");
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
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::cholesky_solve_outf(ks & c10::after_autograd_keyset, self_, input2_, upper, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (input2__storage_saved.has_value())
    AT_ASSERT(input2__storage_saved.value().is_alias_of(input2_.storage()));
  if (input2__impl_saved) AT_ASSERT(input2__impl_saved == input2_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & clamp_min_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & min, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp_min");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("clamp_min");
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
    at::redispatch::clamp_min_outf(ks & c10::after_autograd_keyset, self_, min, out_);
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
Tensor col2im_backward(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<Col2ImBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Col2ImBackwardBackward>(new Col2ImBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->grad_output_argsize_2 = grad_output.size(2);
    grad_fn->grad_output_argsize_3 = grad_output.size(3);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::col2im_backward(ks & c10::after_autograd_keyset, grad_output_, kernel_size, dilation, padding, stride);
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
  throw_error_for_complex_autograd(result, "col2im_backward");
  return result;
}
Tensor constant_pad_nd(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef pad, const Scalar & value) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ConstantPadNdBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ConstantPadNdBackward>(new ConstantPadNdBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->pad = pad.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::constant_pad_nd(ks & c10::after_autograd_keyset, self_, pad, value);
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
std::tuple<Tensor,Tensor,Tensor> conv_depthwise3d_backward_output_mask(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight );
  (void)_any_requires_grad;
  std::shared_ptr<ConvDepthwise3DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ConvDepthwise3DBackwardBackward>(new ConvDepthwise3DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_argsize_1 = self.size(1);
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
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::conv_depthwise3d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, dilation, output_mask);
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
  throw_error_for_complex_autograd(grad_input, "conv_depthwise3d_backward");
  throw_error_for_complex_autograd(grad_weight, "conv_depthwise3d_backward");
  throw_error_for_complex_autograd(grad_bias, "conv_depthwise3d_backward");
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor & copy_sparse_to_sparse_(c10::DispatchKeySet ks, Tensor & self, const Tensor & src, bool non_blocking) {
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  auto _any_requires_grad = compute_requires_grad( self, src );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("copy_sparse_to_sparse_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> src__storage_saved =
    src_.has_storage() ? c10::optional<Storage>(src_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> src__impl_saved;
  if (src_.defined()) src__impl_saved = src_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::copy_sparse_to_sparse_(ks & c10::after_autograd_keyset, self_, src_, non_blocking);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (src__storage_saved.has_value())
    AT_ASSERT(src__storage_saved.value().is_alias_of(src_.storage()));
  if (src__impl_saved) AT_ASSERT(src__impl_saved == src_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & cosh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cosh");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cosh");
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
    at::redispatch::cosh_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor & cross_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, c10::optional<int64_t> dim, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("cross");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cross");
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
    at::redispatch::cross_outf(ks & c10::after_autograd_keyset, self_, other_, dim, out_);
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
Tensor cudnn_affine_grid_generator_backward(c10::DispatchKeySet ks, const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) {
  auto& grad_ = unpack(grad, "grad", 0);
  auto _any_requires_grad = compute_requires_grad( grad );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_affine_grid_generator_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::cudnn_affine_grid_generator_backward(ks & c10::after_autograd_keyset, grad_, N, C, H, W);
  })();
  auto grad_theta = std::move(_tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_theta ), grad_fn);
  }
  throw_error_for_complex_autograd(grad_theta, "cudnn_affine_grid_generator_backward");
  return grad_theta;
}
std::tuple<Tensor,Tensor> cudnn_convolution_backward(c10::DispatchKeySet ks, const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask) {
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( self, grad_output, weight );
  (void)_any_requires_grad;
  std::shared_ptr<CudnnConvolutionBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CudnnConvolutionBackwardBackward>(new CudnnConvolutionBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
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
    return at::redispatch::cudnn_convolution_backward(ks & c10::after_autograd_keyset, self_, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
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
  throw_error_for_complex_autograd(result0, "cudnn_convolution_backward");
  throw_error_for_complex_autograd(result1, "cudnn_convolution_backward");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor cudnn_convolution_backward_input(c10::DispatchKeySet ks, IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, weight );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_backward_input"), deleteNode);
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
    return at::redispatch::cudnn_convolution_backward_input(ks & c10::after_autograd_keyset, self_size, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
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
  throw_error_for_complex_autograd(result, "cudnn_convolution_backward_input");
  return result;
}
Tensor cudnn_convolution_transpose_deprecated(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_transpose"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
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
    return at::redispatch::cudnn_convolution_transpose(ks & c10::after_autograd_keyset, self_, weight_, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
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
  throw_error_for_complex_autograd(result, "cudnn_convolution_transpose");
  return result;
}
Tensor cudnn_convolution_transpose_deprecated2(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_transpose"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight ));
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
    return at::redispatch::cudnn_convolution_transpose(ks & c10::after_autograd_keyset, self_, weight_, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
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
  throw_error_for_complex_autograd(result, "cudnn_convolution_transpose");
  return result;
}
Tensor cudnn_convolution_transpose(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight );
  (void)_any_requires_grad;
  std::shared_ptr<CudnnConvolutionTransposeBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CudnnConvolutionTransposeBackward>(new CudnnConvolutionTransposeBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight ));
    grad_fn->self_ = SavedVariable(self, false);
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
    return at::redispatch::cudnn_convolution_transpose(ks & c10::after_autograd_keyset, self_, weight_, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
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
  throw_error_for_complex_autograd(result, "cudnn_convolution_transpose");
  return result;
}
std::tuple<Tensor &,Tensor &> cummin_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) {
  auto& self_ = unpack(self, "self", 0);
  auto& values_ = unpack(values, "values", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cummin");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("cummin");
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
    at::redispatch::cummin_outf(ks & c10::after_autograd_keyset, self_, dim, values_, indices_);
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
Tensor div_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<DivBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DivBackward0>(new DivBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    grad_fn->other_ = SavedVariable(other, false);
    grad_fn->self_scalar_type = self.scalar_type();
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
    return at::redispatch::div(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor div_Tensor_mode(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, std::string rounding_mode) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<DivBackward2> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DivBackward2>(new DivBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    grad_fn->other_ = SavedVariable(other, false);
    grad_fn->rounding_mode = rounding_mode;
    grad_fn->self_scalar_type = self.scalar_type();
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
    return at::redispatch::div(ks & c10::after_autograd_keyset, self_, other_, rounding_mode);
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
Tensor div_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<DivBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DivBackward1>(new DivBackward1(), deleteNode);
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
    return at::redispatch::div(ks & c10::after_autograd_keyset, self_, other);
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
Tensor div_Scalar_mode(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, std::string rounding_mode) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<DivBackward3> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DivBackward3>(new DivBackward3(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->other = other;
    grad_fn->rounding_mode = rounding_mode;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::div(ks & c10::after_autograd_keyset, self_, other, rounding_mode);
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
Tensor & div__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<DivBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DivBackward0>(new DivBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self.clone(), false);
    }
    grad_fn->other_ = SavedVariable(other, false);
    grad_fn->self_scalar_type = self.scalar_type();
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
    at::redispatch::div_(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor & div__Tensor_mode(c10::DispatchKeySet ks, Tensor & self, const Tensor & other, std::string rounding_mode) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<DivBackward2> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DivBackward2>(new DivBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self.clone(), false);
    }
    grad_fn->other_ = SavedVariable(other, false);
    grad_fn->rounding_mode = rounding_mode;
    grad_fn->self_scalar_type = self.scalar_type();
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
    at::redispatch::div_(ks & c10::after_autograd_keyset, self_, other_, rounding_mode);
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
Tensor & div__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<DivBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DivBackward1>(new DivBackward1(), deleteNode);
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
    at::redispatch::div_(ks & c10::after_autograd_keyset, self_, other);
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
Tensor & div__Scalar_mode(c10::DispatchKeySet ks, Tensor & self, const Scalar & other, std::string rounding_mode) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<DivBackward3> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DivBackward3>(new DivBackward3(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->other = other;
    grad_fn->rounding_mode = rounding_mode;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::div_(ks & c10::after_autograd_keyset, self_, other, rounding_mode);
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
std::tuple<Tensor &,Tensor &> eig_out_e(c10::DispatchKeySet ks, const Tensor & self, bool eigenvectors, Tensor & e, Tensor & v) {
  auto& self_ = unpack(self, "self", 0);
  auto& e_ = unpack(e, "e", 2);
  auto& v_ = unpack(v, "v", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("eig");
  }
  if (compute_requires_grad( e, v )) {
    throw_error_out_requires_grad("eig");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> e__storage_saved =
    e_.has_storage() ? c10::optional<Storage>(e_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> e__impl_saved;
  if (e_.defined()) e__impl_saved = e_.getIntrusivePtr();
  c10::optional<Storage> v__storage_saved =
    v_.has_storage() ? c10::optional<Storage>(v_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> v__impl_saved;
  if (v_.defined()) v__impl_saved = v_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::eig_outf(ks & c10::after_autograd_keyset, self_, eigenvectors, e_, v_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (e__storage_saved.has_value())
    AT_ASSERT(e__storage_saved.value().is_alias_of(e_.storage()));
  if (e__impl_saved) AT_ASSERT(e__impl_saved == e_.getIntrusivePtr());
  if (v__storage_saved.has_value())
    AT_ASSERT(v__storage_saved.value().is_alias_of(v_.storage()));
  if (v__impl_saved) AT_ASSERT(v__impl_saved == v_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( e, v ), grad_fn);
  }
  return std::forward_as_tuple(e, v);
}
Tensor & eq_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
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
    at::redispatch::eq_outf(ks & c10::after_autograd_keyset, self_, other, out_);
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
Tensor & eq_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
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
    at::redispatch::eq_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
  return out;
}
Tensor exp(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ExpBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ExpBackward>(new ExpBackward(), deleteNode);
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
    return at::redispatch::exp(ks & c10::after_autograd_keyset, self_);
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
Tensor exp2(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Exp2Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Exp2Backward>(new Exp2Backward(), deleteNode);
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
    return at::redispatch::exp2(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "exp2");
  return result;
}
Tensor & exp2_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<Exp2Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Exp2Backward>(new Exp2Backward(), deleteNode);
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
    at::redispatch::exp2_(ks & c10::after_autograd_keyset, self_);
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
Tensor & exp_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<ExpBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ExpBackward>(new ExpBackward(), deleteNode);
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
    at::redispatch::exp_(ks & c10::after_autograd_keyset, self_);
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
Tensor & fill__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & value) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<FillBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FillBackward0>(new FillBackward0(), deleteNode);
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
    at::redispatch::fill_(ks & c10::after_autograd_keyset, self_, value);
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
Tensor & fill__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & value) {
  auto& self_ = unpack(self, "self", 0);
  auto& value_ = unpack(value, "value", 1);
  auto _any_requires_grad = compute_requires_grad( self, value );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<FillBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FillBackward1>(new FillBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> value__storage_saved =
    value_.has_storage() ? c10::optional<Storage>(value_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::fill_(ks & c10::after_autograd_keyset, self_, value_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (value__storage_saved.has_value())
    AT_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved) AT_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor fmax(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<FmaxBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FmaxBackward>(new FmaxBackward(), deleteNode);
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
    return at::redispatch::fmax(ks & c10::after_autograd_keyset, self_, other_);
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
  throw_error_for_complex_autograd(result, "fmax");
  return result;
}
Tensor fractional_max_pool2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 4);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<FractionalMaxPool2DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FractionalMaxPool2DBackwardBackward>(new FractionalMaxPool2DBackwardBackward(), deleteNode);
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
    return at::redispatch::fractional_max_pool2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, kernel_size, output_size, indices_);
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
  throw_error_for_complex_autograd(result, "fractional_max_pool2d_backward");
  return result;
}
std::tuple<Tensor,Tensor> fractional_max_pool3d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
  auto& self_ = unpack(self, "self", 0);
  auto& random_samples_ = unpack(random_samples, "random_samples", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_no_requires_grad(random_samples, "random_samples");
  std::shared_ptr<FractionalMaxPool3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FractionalMaxPool3DBackward>(new FractionalMaxPool3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->output_size = output_size.vec();
  }
  Tensor result0;
  Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> random_samples__storage_saved =
    random_samples_.has_storage() ? c10::optional<Storage>(random_samples_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> random_samples__impl_saved;
  if (random_samples_.defined()) random_samples__impl_saved = random_samples_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::fractional_max_pool3d(ks & c10::after_autograd_keyset, self_, kernel_size, output_size, random_samples_);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (random_samples__storage_saved.has_value())
    AT_ASSERT(random_samples__storage_saved.value().is_alias_of(random_samples_.storage()));
  if (random_samples__impl_saved) AT_ASSERT(random_samples__impl_saved == random_samples_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  throw_error_for_complex_autograd(result0, "fractional_max_pool3d");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & fractional_max_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 4);
  auto& grad_input_ = unpack(grad_input, "grad_input", 5);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("fractional_max_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("fractional_max_pool3d_backward");
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
    at::redispatch::fractional_max_pool3d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, kernel_size, output_size, indices_, grad_input_);
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
std::tuple<Tensor &,Tensor &> frexp_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & mantissa, Tensor & exponent) {
  auto& self_ = unpack(self, "self", 0);
  auto& mantissa_ = unpack(mantissa, "mantissa", 1);
  auto& exponent_ = unpack(exponent, "exponent", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("frexp");
  }
  if (compute_requires_grad( mantissa )) {
    throw_error_out_requires_grad("frexp");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mantissa__storage_saved =
    mantissa_.has_storage() ? c10::optional<Storage>(mantissa_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mantissa__impl_saved;
  if (mantissa_.defined()) mantissa__impl_saved = mantissa_.getIntrusivePtr();
  c10::optional<Storage> exponent__storage_saved =
    exponent_.has_storage() ? c10::optional<Storage>(exponent_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> exponent__impl_saved;
  if (exponent_.defined()) exponent__impl_saved = exponent_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::frexp_outf(ks & c10::after_autograd_keyset, self_, mantissa_, exponent_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mantissa__storage_saved.has_value())
    AT_ASSERT(mantissa__storage_saved.value().is_alias_of(mantissa_.storage()));
  if (mantissa__impl_saved) AT_ASSERT(mantissa__impl_saved == mantissa_.getIntrusivePtr());
  if (exponent__storage_saved.has_value())
    AT_ASSERT(exponent__storage_saved.value().is_alias_of(exponent_.storage()));
  if (exponent__impl_saved) AT_ASSERT(exponent__impl_saved == exponent_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( mantissa ), grad_fn);
  }
  return std::forward_as_tuple(mantissa, exponent);
}
Tensor & ge_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
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
    at::redispatch::ge_outf(ks & c10::after_autograd_keyset, self_, other, out_);
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
Tensor & ge_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
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
    at::redispatch::ge_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
  return out;
}
std::tuple<Tensor,Tensor> geqrf(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<GeqrfBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GeqrfBackward>(new GeqrfBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor a;
  Tensor tau;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::geqrf(ks & c10::after_autograd_keyset, self_);
  })();
  std::tie(a, tau) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( a, tau ), grad_fn);
  }
  throw_error_for_complex_autograd(a, "geqrf");
  throw_error_for_complex_autograd(tau, "geqrf");
  return std::make_tuple(std::move(a), std::move(tau));
}
Tensor ger(c10::DispatchKeySet ks, const Tensor & self, const Tensor & vec2) {
  auto& self_ = unpack(self, "self", 0);
  auto& vec2_ = unpack(vec2, "vec2", 1);
  auto _any_requires_grad = compute_requires_grad( self, vec2 );
  (void)_any_requires_grad;
  std::shared_ptr<GerBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GerBackward>(new GerBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, vec2 ));
    if (grad_fn->should_compute_output(0)) {
      grad_fn->vec2_ = SavedVariable(vec2, false);
    }
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> vec2__storage_saved =
    vec2_.has_storage() ? c10::optional<Storage>(vec2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec2__impl_saved;
  if (vec2_.defined()) vec2__impl_saved = vec2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::ger(ks & c10::after_autograd_keyset, self_, vec2_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec2__storage_saved.has_value())
    AT_ASSERT(vec2__storage_saved.value().is_alias_of(vec2_.storage()));
  if (vec2__impl_saved) AT_ASSERT(vec2__impl_saved == vec2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
Tensor glu(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<GluBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GluBackward>(new GluBackward(), deleteNode);
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
    return at::redispatch::glu(ks & c10::after_autograd_keyset, self_, dim);
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
  throw_error_for_complex_autograd(result, "glu");
  return result;
}
Tensor & glu_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, int64_t dim, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("glu_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("glu_backward");
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
    at::redispatch::glu_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, dim, grad_input_);
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
Tensor grid_sampler_2d(c10::DispatchKeySet ks, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  auto& input_ = unpack(input, "input", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  auto _any_requires_grad = compute_requires_grad( input, grid );
  (void)_any_requires_grad;
  std::shared_ptr<GridSampler2DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GridSampler2DBackward>(new GridSampler2DBackward(), deleteNode);
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
    return at::redispatch::grid_sampler_2d(ks & c10::after_autograd_keyset, input_, grid_, interpolation_mode, padding_mode, align_corners);
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
  throw_error_for_complex_autograd(result, "grid_sampler_2d");
  return result;
}
Tensor & gt_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
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
    at::redispatch::gt_outf(ks & c10::after_autograd_keyset, self_, other, out_);
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
Tensor & gt_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
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
    at::redispatch::gt_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
  return out;
}
Tensor hardsigmoid(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<HardsigmoidBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HardsigmoidBackward>(new HardsigmoidBackward(), deleteNode);
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
    return at::redispatch::hardsigmoid(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "hardsigmoid");
  return result;
}
Tensor & hardsigmoid_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<HardsigmoidBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HardsigmoidBackward>(new HardsigmoidBackward(), deleteNode);
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
    at::redispatch::hardsigmoid_(ks & c10::after_autograd_keyset, self_);
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
Tensor hardswish(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<HardswishBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HardswishBackward>(new HardswishBackward(), deleteNode);
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
    return at::redispatch::hardswish(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "hardswish");
  return result;
}
Tensor & hardswish_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<HardswishBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HardswishBackward>(new HardswishBackward(), deleteNode);
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
    at::redispatch::hardswish_(ks & c10::after_autograd_keyset, self_);
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
Tensor hardtanh_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Scalar & min_val, const Scalar & max_val) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<HardtanhBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HardtanhBackwardBackward>(new HardtanhBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
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
    return at::redispatch::hardtanh_backward(ks & c10::after_autograd_keyset, grad_output_, self_, min_val, max_val);
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
  throw_error_for_complex_autograd(result, "hardtanh_backward");
  return result;
}
Tensor & histc_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t bins, const Scalar & min, const Scalar & max, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("histc");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("histc");
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
    at::redispatch::histc_outf(ks & c10::after_autograd_keyset, self_, bins, min, max, out_);
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
Tensor & hspmm_out_out(c10::DispatchKeySet ks, const Tensor & mat1, const Tensor & mat2, Tensor & out) {
  auto& mat1_ = unpack(mat1, "mat1", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( mat1, mat2 );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( mat1, mat2 )) {
    throw_error_out_requires_grad("hspmm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("hspmm");
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
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::hspmm_outf(ks & c10::after_autograd_keyset, mat1_, mat2_, out_);
  }
  #ifndef NDEBUG
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
Tensor huber_loss(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction, double delta) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self, target );
  (void)_any_requires_grad;
  std::shared_ptr<HuberLossBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HuberLossBackward>(new HuberLossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
    grad_fn->delta = delta;
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
    return at::redispatch::huber_loss(ks & c10::after_autograd_keyset, self_, target_, reduction, delta);
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
  throw_error_for_complex_autograd(result, "huber_loss");
  return result;
}
Tensor & huber_loss_backward_out_out(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double delta, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& grad_input_ = unpack(grad_input, "grad_input", 5);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, target );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("huber_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("huber_loss_backward");
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
    at::redispatch::huber_loss_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, target_, reduction, delta, grad_input_);
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
Tensor & hypot_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("hypot");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("hypot");
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
    at::redispatch::hypot_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
Tensor i0(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<I0Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<I0Backward>(new I0Backward(), deleteNode);
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
    return at::redispatch::i0(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "i0");
  return result;
}
Tensor & i0_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<I0Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<I0Backward>(new I0Backward(), deleteNode);
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
    at::redispatch::i0_(ks & c10::after_autograd_keyset, self_);
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
Tensor & igammac_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("igammac");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("igammac");
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
    at::redispatch::igammac_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
Tensor im2col_backward(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<Im2ColBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Im2ColBackwardBackward>(new Im2ColBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::im2col_backward(ks & c10::after_autograd_keyset, grad_output_, input_size, kernel_size, dilation, padding, stride);
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
  throw_error_for_complex_autograd(result, "im2col_backward");
  return result;
}
Tensor & index_add_(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  auto _any_requires_grad = compute_requires_grad( self, source );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<IndexAddBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<IndexAddBackward>(new IndexAddBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->dim = dim;
    if (grad_fn->should_compute_output(1)) {
      grad_fn->index_ = SavedVariable(index, false);
    }
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
    at::redispatch::index_add_(ks & c10::after_autograd_keyset, self_, dim, index_, source_);
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
Tensor & inverse_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("inverse");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("inverse");
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
    at::redispatch::inverse_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor kl_div(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self, target );
  (void)_any_requires_grad;
  std::shared_ptr<KlDivBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<KlDivBackward>(new KlDivBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
    grad_fn->log_target = log_target;
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
    return at::redispatch::kl_div(ks & c10::after_autograd_keyset, self_, target_, reduction, log_target);
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
  throw_error_for_complex_autograd(result, "kl_div");
  return result;
}
std::tuple<Tensor,Tensor> kthvalue(c10::DispatchKeySet ks, const Tensor & self, int64_t k, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<KthvalueBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<KthvalueBackward>(new KthvalueBackward(), deleteNode);
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
    return at::redispatch::kthvalue(ks & c10::after_autograd_keyset, self_, k, dim, keepdim);
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
  throw_error_for_complex_autograd(values, "kthvalue");
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor & le_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
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
    at::redispatch::le_outf(ks & c10::after_autograd_keyset, self_, other, out_);
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
Tensor & le_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
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
    at::redispatch::le_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
  return out;
}
Tensor & leaky_relu_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & negative_slope, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("leaky_relu");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("leaky_relu");
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
    at::redispatch::leaky_relu_outf(ks & c10::after_autograd_keyset, self_, negative_slope, out_);
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
Tensor lgamma(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LgammaBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LgammaBackward>(new LgammaBackward(), deleteNode);
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
    return at::redispatch::lgamma(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "lgamma");
  return result;
}
Tensor & lgamma_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<LgammaBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LgammaBackward>(new LgammaBackward(), deleteNode);
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
    at::redispatch::lgamma_(ks & c10::after_autograd_keyset, self_);
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
std::tuple<Tensor &,Tensor &> linalg_eigh_out_eigvals(c10::DispatchKeySet ks, const Tensor & self, std::string UPLO, Tensor & eigvals, Tensor & eigvecs) {
  auto& self_ = unpack(self, "self", 0);
  auto& eigvals_ = unpack(eigvals, "eigvals", 2);
  auto& eigvecs_ = unpack(eigvecs, "eigvecs", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("linalg_eigh");
  }
  if (compute_requires_grad( eigvals, eigvecs )) {
    throw_error_out_requires_grad("linalg_eigh");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> eigvals__storage_saved =
    eigvals_.has_storage() ? c10::optional<Storage>(eigvals_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> eigvals__impl_saved;
  if (eigvals_.defined()) eigvals__impl_saved = eigvals_.getIntrusivePtr();
  c10::optional<Storage> eigvecs__storage_saved =
    eigvecs_.has_storage() ? c10::optional<Storage>(eigvecs_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> eigvecs__impl_saved;
  if (eigvecs_.defined()) eigvecs__impl_saved = eigvecs_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::linalg_eigh_outf(ks & c10::after_autograd_keyset, self_, UPLO, eigvals_, eigvecs_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (eigvals__storage_saved.has_value())
    AT_ASSERT(eigvals__storage_saved.value().is_alias_of(eigvals_.storage()));
  if (eigvals__impl_saved) AT_ASSERT(eigvals__impl_saved == eigvals_.getIntrusivePtr());
  if (eigvecs__storage_saved.has_value())
    AT_ASSERT(eigvecs__storage_saved.value().is_alias_of(eigvecs_.storage()));
  if (eigvecs__impl_saved) AT_ASSERT(eigvecs__impl_saved == eigvecs_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( eigvals, eigvecs ), grad_fn);
  }
  return std::forward_as_tuple(eigvals, eigvecs);
}
Tensor & linalg_inv_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("linalg_inv");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("linalg_inv");
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
    at::redispatch::linalg_inv_outf(ks & c10::after_autograd_keyset, self_, out_);
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
std::tuple<Tensor,Tensor,Tensor,Tensor> linalg_lstsq(c10::DispatchKeySet ks, const Tensor & self, const Tensor & b, c10::optional<double> cond, c10::optional<std::string> driver) {
  auto& self_ = unpack(self, "self", 0);
  auto& b_ = unpack(b, "b", 1);
  auto _any_requires_grad = compute_requires_grad( self, b );
  (void)_any_requires_grad;
  std::shared_ptr<LinalgLstsqBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LinalgLstsqBackward>(new LinalgLstsqBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, b ));
  }
  Tensor solution;
  Tensor residuals;
  Tensor rank;
  Tensor singular_values;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> b__storage_saved =
    b_.has_storage() ? c10::optional<Storage>(b_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> b__impl_saved;
  if (b_.defined()) b__impl_saved = b_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::linalg_lstsq(ks & c10::after_autograd_keyset, self_, b_, cond, driver);
  })();
  std::tie(solution, residuals, rank, singular_values) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (b__storage_saved.has_value())
    AT_ASSERT(b__storage_saved.value().is_alias_of(b_.storage()));
  if (b__impl_saved) AT_ASSERT(b__impl_saved == b_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( solution, residuals ), grad_fn);
  }
  throw_error_for_complex_autograd(solution, "linalg_lstsq");
  throw_error_for_complex_autograd(residuals, "linalg_lstsq");
  return std::make_tuple(std::move(solution), std::move(residuals), std::move(rank), std::move(singular_values));
}
std::tuple<Tensor,Tensor> linalg_qr(c10::DispatchKeySet ks, const Tensor & self, std::string mode) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LinalgQrBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LinalgQrBackward>(new LinalgQrBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->mode = mode;
  }
  Tensor Q;
  Tensor R;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::linalg_qr(ks & c10::after_autograd_keyset, self_, mode);
  })();
  std::tie(Q, R) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( Q, R ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->Q_ = SavedVariable(Q, true);
    grad_fn->R_ = SavedVariable(R, true);
  }
  return std::make_tuple(std::move(Q), std::move(R));
}
Tensor & linalg_solve_out_out(c10::DispatchKeySet ks, const Tensor & input, const Tensor & other, Tensor & out) {
  auto& input_ = unpack(input, "input", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( input, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( input, other )) {
    throw_error_out_requires_grad("linalg_solve");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("linalg_solve");
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
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::linalg_solve_outf(ks & c10::after_autograd_keyset, input_, other_, out_);
  }
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
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
Tensor & linalg_vector_norm_out_out(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 5);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("linalg_vector_norm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("linalg_vector_norm");
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
    at::redispatch::linalg_vector_norm_outf(ks & c10::after_autograd_keyset, self_, ord, dim, keepdim, dtype, out_);
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
Tensor & log10_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log10");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("log10");
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
    at::redispatch::log10_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor log1p(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Log1PBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Log1PBackward>(new Log1PBackward(), deleteNode);
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
    return at::redispatch::log1p(ks & c10::after_autograd_keyset, self_);
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
Tensor & log1p_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<Log1PBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Log1PBackward>(new Log1PBackward(), deleteNode);
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
    at::redispatch::log1p_(ks & c10::after_autograd_keyset, self_);
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
Tensor logit(c10::DispatchKeySet ks, const Tensor & self, c10::optional<double> eps) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LogitBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogitBackward>(new LogitBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->eps = eps;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::logit(ks & c10::after_autograd_keyset, self_, eps);
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
  throw_error_for_complex_autograd(result, "logit");
  return result;
}
Tensor & logit_(c10::DispatchKeySet ks, Tensor & self, c10::optional<double> eps) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<LogitBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogitBackward>(new LogitBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->eps = eps;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::logit_(ks & c10::after_autograd_keyset, self_, eps);
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
Tensor & logit_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, c10::optional<double> eps, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("logit_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("logit_backward");
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
    at::redispatch::logit_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, eps, grad_input_);
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
Tensor & lt_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
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
    at::redispatch::lt_outf(ks & c10::after_autograd_keyset, self_, other, out_);
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
Tensor & lt_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
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
    at::redispatch::lt_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
  return out;
}
Tensor lu_solve(c10::DispatchKeySet ks, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  auto& self_ = unpack(self, "self", 0);
  auto& LU_data_ = unpack(LU_data, "LU_data", 1);
  auto& LU_pivots_ = unpack(LU_pivots, "LU_pivots", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_no_requires_grad(LU_data, "LU_data");
  check_no_requires_grad(LU_pivots, "LU_pivots");
  std::shared_ptr<LuSolveBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LuSolveBackward>(new LuSolveBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> LU_data__storage_saved =
    LU_data_.has_storage() ? c10::optional<Storage>(LU_data_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_data__impl_saved;
  if (LU_data_.defined()) LU_data__impl_saved = LU_data_.getIntrusivePtr();
  c10::optional<Storage> LU_pivots__storage_saved =
    LU_pivots_.has_storage() ? c10::optional<Storage>(LU_pivots_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_pivots__impl_saved;
  if (LU_pivots_.defined()) LU_pivots__impl_saved = LU_pivots_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::lu_solve(ks & c10::after_autograd_keyset, self_, LU_data_, LU_pivots_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (LU_data__storage_saved.has_value())
    AT_ASSERT(LU_data__storage_saved.value().is_alias_of(LU_data_.storage()));
  if (LU_data__impl_saved) AT_ASSERT(LU_data__impl_saved == LU_data_.getIntrusivePtr());
  if (LU_pivots__storage_saved.has_value())
    AT_ASSERT(LU_pivots__storage_saved.value().is_alias_of(LU_pivots_.storage()));
  if (LU_pivots__impl_saved) AT_ASSERT(LU_pivots__impl_saved == LU_pivots_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "lu_solve");
  return result;
}
Tensor & masked_select_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mask, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("masked_select");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("masked_select");
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
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::masked_select_outf(ks & c10::after_autograd_keyset, self_, mask_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor max_pool2d_with_indices_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<MaxPool2DWithIndicesBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaxPool2DWithIndicesBackwardBackward>(new MaxPool2DWithIndicesBackwardBackward(), deleteNode);
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
    return at::redispatch::max_pool2d_with_indices_backward(ks & c10::after_autograd_keyset, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
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
  throw_error_for_complex_autograd(result, "max_pool2d_with_indices_backward");
  return result;
}
std::tuple<Tensor,Tensor> max_pool3d_with_indices(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MaxPool3DWithIndicesBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaxPool3DWithIndicesBackward>(new MaxPool3DWithIndicesBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->ceil_mode = ceil_mode;
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
    return at::redispatch::max_pool3d_with_indices(ks & c10::after_autograd_keyset, self_, kernel_size, stride, padding, dilation, ceil_mode);
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
  throw_error_for_complex_autograd(result0, "max_pool3d_with_indices");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & max_pool3d_with_indices_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  auto& grad_input_ = unpack(grad_input, "grad_input", 8);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("max_pool3d_with_indices_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_pool3d_with_indices_backward");
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
    at::redispatch::max_pool3d_with_indices_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_, grad_input_);
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
std::tuple<Tensor &,Tensor &> min_out_dim_min(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & min, Tensor & min_indices) {
  auto& self_ = unpack(self, "self", 0);
  auto& min_ = unpack(min, "min", 3);
  auto& min_indices_ = unpack(min_indices, "min_indices", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("min");
  }
  if (compute_requires_grad( min )) {
    throw_error_out_requires_grad("min");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> min__storage_saved =
    min_.has_storage() ? c10::optional<Storage>(min_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> min__impl_saved;
  if (min_.defined()) min__impl_saved = min_.getIntrusivePtr();
  c10::optional<Storage> min_indices__storage_saved =
    min_indices_.has_storage() ? c10::optional<Storage>(min_indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> min_indices__impl_saved;
  if (min_indices_.defined()) min_indices__impl_saved = min_indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::min_outf(ks & c10::after_autograd_keyset, self_, dim, keepdim, min_, min_indices_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (min__storage_saved.has_value())
    AT_ASSERT(min__storage_saved.value().is_alias_of(min_.storage()));
  if (min__impl_saved) AT_ASSERT(min__impl_saved == min_.getIntrusivePtr());
  if (min_indices__storage_saved.has_value())
    AT_ASSERT(min_indices__storage_saved.value().is_alias_of(min_indices_.storage()));
  if (min_indices__impl_saved) AT_ASSERT(min_indices__impl_saved == min_indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( min ), grad_fn);
  }
  return std::forward_as_tuple(min, min_indices);
}
Tensor & minimum_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("minimum");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("minimum");
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
    at::redispatch::minimum_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
Tensor miopen_convolution_backward_weight(c10::DispatchKeySet ks, IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_backward_weight"), deleteNode);
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
    return at::redispatch::miopen_convolution_backward_weight(ks & c10::after_autograd_keyset, weight_size, grad_output_, self_, padding, stride, dilation, groups, benchmark, deterministic);
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
  throw_error_for_complex_autograd(result, "miopen_convolution_backward_weight");
  return result;
}
std::tuple<Tensor,Tensor,Tensor> miopen_depthwise_convolution_backward(c10::DispatchKeySet ks, const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( self, grad_output, weight );
  (void)_any_requires_grad;
  std::shared_ptr<MiopenDepthwiseConvolutionBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MiopenDepthwiseConvolutionBackwardBackward>(new MiopenDepthwiseConvolutionBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
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
    return at::redispatch::miopen_depthwise_convolution_backward(ks & c10::after_autograd_keyset, self_, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
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
  throw_error_for_complex_autograd(result0, "miopen_depthwise_convolution_backward");
  throw_error_for_complex_autograd(result1, "miopen_depthwise_convolution_backward");
  throw_error_for_complex_autograd(result2, "miopen_depthwise_convolution_backward");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor miopen_depthwise_convolution_backward_input(c10::DispatchKeySet ks, IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, weight );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_depthwise_convolution_backward_input"), deleteNode);
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
    return at::redispatch::miopen_depthwise_convolution_backward_input(ks & c10::after_autograd_keyset, self_size, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic);
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
  throw_error_for_complex_autograd(result, "miopen_depthwise_convolution_backward_input");
  return result;
}
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> miopen_rnn_backward(c10::DispatchKeySet ks, const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const c10::optional<Tensor> & cx, const Tensor & output, const c10::optional<Tensor> & grad_output, const c10::optional<Tensor> & grad_hy, const c10::optional<Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const c10::optional<Tensor> & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) {
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack(weight, "weight", 1);
  auto& weight_buf_ = unpack(weight_buf, "weight_buf", 3);
  auto& hx_ = unpack(hx, "hx", 4);
  auto& output_ = unpack(output, "output", 6);
  auto& reserve_ = unpack(reserve, "reserve", 19);
  auto _any_requires_grad = compute_requires_grad( input, weight, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, reserve );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_rnn_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, reserve ));
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
    return at::redispatch::miopen_rnn_backward(ks & c10::after_autograd_keyset, input_, weight_, weight_stride0, weight_buf_, hx_, cx, output_, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve_, output_mask);
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
  throw_error_for_complex_autograd(result0, "miopen_rnn_backward");
  throw_error_for_complex_autograd(result1, "miopen_rnn_backward");
  throw_error_for_complex_autograd(result2, "miopen_rnn_backward");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
std::tuple<Tensor,Tensor> mkldnn_linear_backward_weights(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & input, const Tensor & weight, bool bias_defined) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, input, weight );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_linear_backward_weights"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, input, weight ));
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
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::mkldnn_linear_backward_weights(ks & c10::after_autograd_keyset, grad_output_, input_, weight_, bias_defined);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "mkldnn_linear_backward_weights");
  throw_error_for_complex_autograd(result1, "mkldnn_linear_backward_weights");
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor mkldnn_max_pool2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & output, const Tensor & input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  auto& input_ = unpack(input, "input", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, output, input );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_max_pool2d_backward"), deleteNode);
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
    return at::redispatch::mkldnn_max_pool2d_backward(ks & c10::after_autograd_keyset, grad_output_, output_, input_, kernel_size, stride, padding, dilation, ceil_mode);
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
  throw_error_for_complex_autograd(result, "mkldnn_max_pool2d_backward");
  return result;
}
Tensor mkldnn_max_pool3d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MkldnnMaxPool3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MkldnnMaxPool3DBackward>(new MkldnnMaxPool3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->ceil_mode = ceil_mode;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::mkldnn_max_pool3d(ks & c10::after_autograd_keyset, self_, kernel_size, stride, padding, dilation, ceil_mode);
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
  throw_error_for_complex_autograd(result, "mkldnn_max_pool3d");
  return result;
}
Tensor mm(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat2) {
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  auto _any_requires_grad = compute_requires_grad( self, mat2 );
  (void)_any_requires_grad;
  std::shared_ptr<MmBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MmBackward>(new MmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    grad_fn->mat2_sizes = mat2.sizes().vec();
    grad_fn->mat2_strides = strides_or_error(mat2, "mat2").vec();
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_strides = strides_or_error(self, "self").vec();
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
    return at::redispatch::mm(ks & c10::after_autograd_keyset, self_, mat2_);
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
Tensor & mse_loss_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self, target );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("mse_loss");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("mse_loss");
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
    at::redispatch::mse_loss_outf(ks & c10::after_autograd_keyset, self_, target_, reduction, out_);
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
Tensor & mul_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("mul");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("mul");
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
    at::redispatch::mul_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
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
Tensor multi_margin_loss_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Scalar & p, const Scalar & margin, const c10::optional<Tensor> & weight, int64_t reduction) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, target, weight );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("multi_margin_loss_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target, weight ));
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
    return at::redispatch::multi_margin_loss_backward(ks & c10::after_autograd_keyset, grad_output_, self_, target_, p, margin, weight, reduction);
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
  throw_error_for_complex_autograd(result, "multi_margin_loss_backward");
  return result;
}
std::tuple<Tensor &,Tensor &> nanmedian_out_dim_values(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
  auto& self_ = unpack(self, "self", 0);
  auto& values_ = unpack(values, "values", 3);
  auto& indices_ = unpack(indices, "indices", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("nanmedian");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("nanmedian");
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
    at::redispatch::nanmedian_outf(ks & c10::after_autograd_keyset, self_, dim, keepdim, values_, indices_);
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
Tensor nansum(c10::DispatchKeySet ks, const Tensor & self, c10::optional<ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NansumBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NansumBackward0>(new NansumBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_scalar_type = self.scalar_type();
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
    return at::redispatch::nansum(ks & c10::after_autograd_keyset, self_, dtype);
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
  throw_error_for_complex_autograd(result, "nansum");
  return result;
}
Tensor nansum_dim_IntList(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NansumBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NansumBackward1>(new NansumBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
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
    return at::redispatch::nansum(ks & c10::after_autograd_keyset, self_, dim, keepdim, dtype);
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
  throw_error_for_complex_autograd(result, "nansum");
  return result;
}
Tensor narrow_copy(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, int64_t start, int64_t length) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("narrow_copy"), deleteNode);
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
    return at::redispatch::narrow_copy(ks & c10::after_autograd_keyset, self_, dim, start, length);
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
  throw_error_for_complex_autograd(result, "narrow_copy");
  return result;
}
std::tuple<Tensor,Tensor,Tensor> native_group_norm(c10::DispatchKeySet ks, const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {
  auto& input_ = unpack(input, "input", 0);
  auto _any_requires_grad = compute_requires_grad( input, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<NativeGroupNormBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NativeGroupNormBackward>(new NativeGroupNormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->N = N;
    grad_fn->C = C;
    grad_fn->HxW = HxW;
    grad_fn->group = group;
    grad_fn->eps = eps;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::native_group_norm(ks & c10::after_autograd_keyset, input_, weight, bias, N, C, HxW, group, eps);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  throw_error_for_complex_autograd(result0, "native_group_norm");
  throw_error_for_complex_autograd(result1, "native_group_norm");
  throw_error_for_complex_autograd(result2, "native_group_norm");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor,Tensor> native_layer_norm_backward(c10::DispatchKeySet ks, const Tensor & grad_out, const Tensor & input, IntArrayRef normalized_shape, const Tensor & mean, const Tensor & rstd, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, std::array<bool,3> output_mask) {
  auto& grad_out_ = unpack(grad_out, "grad_out", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& mean_ = unpack(mean, "mean", 3);
  auto& rstd_ = unpack(rstd, "rstd", 4);
  auto _any_requires_grad = compute_requires_grad( grad_out, input, mean, rstd, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("native_layer_norm_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_out, input, mean, rstd, weight, bias ));
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
    return at::redispatch::native_layer_norm_backward(ks & c10::after_autograd_keyset, grad_out_, input_, normalized_shape, mean_, rstd_, weight, bias, output_mask);
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
  throw_error_for_complex_autograd(result0, "native_layer_norm_backward");
  throw_error_for_complex_autograd(result1, "native_layer_norm_backward");
  throw_error_for_complex_autograd(result2, "native_layer_norm_backward");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor & neg_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("neg");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("neg");
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
    at::redispatch::neg_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor & nonzero_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
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
    at::redispatch::nonzero_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor & norm_out_dtype_out(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, ScalarType dtype, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 5);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("norm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("norm");
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
    at::redispatch::norm_outf(ks & c10::after_autograd_keyset, self_, p, dim, keepdim, dtype, out_);
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
Tensor & norm_out_out(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("norm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("norm");
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
    at::redispatch::norm_outf(ks & c10::after_autograd_keyset, self_, p, dim, keepdim, out_);
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
Tensor & normal_out_Tensor_float_out(c10::DispatchKeySet ks, const Tensor & mean, double std, c10::optional<Generator> generator, Tensor & out) {
  auto& mean_ = unpack(mean, "mean", 0);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( mean );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( mean )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("normal");
  }
  #ifndef NDEBUG
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::normal_outf(ks & c10::after_autograd_keyset, mean_, std, generator, out_);
  }
  #ifndef NDEBUG
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & normal_out_float_Tensor_out(c10::DispatchKeySet ks, double mean, const Tensor & std, c10::optional<Generator> generator, Tensor & out) {
  auto& std_ = unpack(std, "std", 1);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( std );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( std )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("normal");
  }
  #ifndef NDEBUG
  c10::optional<Storage> std__storage_saved =
    std_.has_storage() ? c10::optional<Storage>(std_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> std__impl_saved;
  if (std_.defined()) std__impl_saved = std_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::normal_outf(ks & c10::after_autograd_keyset, mean, std_, generator, out_);
  }
  #ifndef NDEBUG
  if (std__storage_saved.has_value())
    AT_ASSERT(std__storage_saved.value().is_alias_of(std_.storage()));
  if (std__impl_saved) AT_ASSERT(std__impl_saved == std_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
Tensor & normal_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const Tensor & mean, const Tensor & std, c10::optional<Generator> generator, Tensor & out) {
  auto& mean_ = unpack(mean, "mean", 0);
  auto& std_ = unpack(std, "std", 1);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( mean, std );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( mean, std )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("normal");
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
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::normal_outf(ks & c10::after_autograd_keyset, mean_, std_, generator, out_);
  }
  #ifndef NDEBUG
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (std__storage_saved.has_value())
    AT_ASSERT(std__storage_saved.value().is_alias_of(std_.storage()));
  if (std__impl_saved) AT_ASSERT(std__impl_saved == std_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  return out;
}
std::tuple<Tensor,Tensor> prelu_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight );
  (void)_any_requires_grad;
  std::shared_ptr<PreluBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PreluBackwardBackward>(new PreluBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  Tensor result0;
  Tensor result1;
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
    return at::redispatch::prelu_backward(ks & c10::after_autograd_keyset, grad_output_, self_, weight_);
  })();
  std::tie(result0, result1) = std::move(_tmp);
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
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "prelu_backward");
  throw_error_for_complex_autograd(result1, "prelu_backward");
  return std::make_tuple(std::move(result0), std::move(result1));
}
double q_scale(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::q_scale(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = _tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor quantized_max_pool1d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("quantized_max_pool1d"), deleteNode);
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
    return at::redispatch::quantized_max_pool1d(ks & c10::after_autograd_keyset, self_, kernel_size, stride, padding, dilation, ceil_mode);
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
  throw_error_for_complex_autograd(result, "quantized_max_pool1d");
  return result;
}
Tensor & renorm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & p, int64_t dim, const Scalar & maxnorm, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("renorm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("renorm");
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
    at::redispatch::renorm_outf(ks & c10::after_autograd_keyset, self_, p, dim, maxnorm, out_);
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
Tensor replication_pad2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<ReplicationPad2DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ReplicationPad2DBackwardBackward>(new ReplicationPad2DBackwardBackward(), deleteNode);
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
    return at::redispatch::replication_pad2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, padding);
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
Tensor replication_pad3d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ReplicationPad3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ReplicationPad3DBackward>(new ReplicationPad3DBackward(), deleteNode);
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
    return at::redispatch::replication_pad3d(ks & c10::after_autograd_keyset, self_, padding);
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
Tensor & replication_pad3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("replication_pad3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("replication_pad3d_backward");
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
    at::redispatch::replication_pad3d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, padding, grad_input_);
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
Tensor & scatter__src(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  auto _any_requires_grad = compute_requires_grad( self, src );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<ScatterBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ScatterBackward0>(new ScatterBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
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
  c10::optional<Storage> src__storage_saved =
    src_.has_storage() ? c10::optional<Storage>(src_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> src__impl_saved;
  if (src_.defined()) src__impl_saved = src_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::scatter_(ks & c10::after_autograd_keyset, self_, dim, index_, src_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (src__storage_saved.has_value())
    AT_ASSERT(src__storage_saved.value().is_alias_of(src_.storage()));
  if (src__impl_saved) AT_ASSERT(src__impl_saved == src_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & scatter__value(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Scalar & value) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<ScatterBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ScatterBackward1>(new ScatterBackward1(), deleteNode);
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
    at::redispatch::scatter_(ks & c10::after_autograd_keyset, self_, dim, index_, value);
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
Tensor & scatter__reduce(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, std::string reduce) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  auto _any_requires_grad = compute_requires_grad( self, index, src );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("scatter_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, index, src ));
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
  c10::optional<Storage> src__storage_saved =
    src_.has_storage() ? c10::optional<Storage>(src_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> src__impl_saved;
  if (src_.defined()) src__impl_saved = src_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::scatter_(ks & c10::after_autograd_keyset, self_, dim, index_, src_, reduce);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (src__storage_saved.has_value())
    AT_ASSERT(src__storage_saved.value().is_alias_of(src_.storage()));
  if (src__impl_saved) AT_ASSERT(src__impl_saved == src_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & scatter__value_reduce(c10::DispatchKeySet ks, Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, std::string reduce) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto _any_requires_grad = compute_requires_grad( self, index );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("scatter_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, index ));
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
    at::redispatch::scatter_(ks & c10::after_autograd_keyset, self_, dim, index_, value, reduce);
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
Tensor sgn(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SgnBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SgnBackward>(new SgnBackward(), deleteNode);
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
    return at::redispatch::sgn(ks & c10::after_autograd_keyset, self_);
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
Tensor & sgn_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<SgnBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SgnBackward>(new SgnBackward(), deleteNode);
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
    at::redispatch::sgn_(ks & c10::after_autograd_keyset, self_);
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
Tensor sigmoid(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SigmoidBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SigmoidBackward>(new SigmoidBackward(), deleteNode);
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
    return at::redispatch::sigmoid(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "sigmoid");
  return result;
}
Tensor & sigmoid_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<SigmoidBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SigmoidBackward>(new SigmoidBackward(), deleteNode);
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
    at::redispatch::sigmoid_(ks & c10::after_autograd_keyset, self_);
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
Tensor & sigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & output, Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, output );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("sigmoid_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("sigmoid_backward");
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
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::sigmoid_backward_outf(ks & c10::after_autograd_keyset, grad_output_, output_, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
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
Tensor sign(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SignBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SignBackward>(new SignBackward(), deleteNode);
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
    return at::redispatch::sign(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "sign");
  return result;
}
Tensor & sign_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<SignBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SignBackward>(new SignBackward(), deleteNode);
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
    at::redispatch::sign_(ks & c10::after_autograd_keyset, self_);
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
Tensor & sinc_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sinc");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sinc");
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
    at::redispatch::sinc_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor & sinh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sinh");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sinh");
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
    at::redispatch::sinh_outf(ks & c10::after_autograd_keyset, self_, out_);
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
Tensor slice_Tensor(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SliceBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SliceBackward>(new SliceBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->start = start;
    grad_fn->end = end;
    grad_fn->step = step;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoNonVariableTypeMode guard;
    return at::redispatch::slice(ks & c10::after_autograd_keyset, self_, dim, start, end, step);
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
std::tuple<Tensor,Tensor,Tensor> slow_conv3d_backward_output_mask(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 6);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 7);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight );
  (void)_any_requires_grad;
  check_no_requires_grad(finput, "finput");
  check_no_requires_grad(fgrad_input, "fgrad_input");
  std::shared_ptr<SlowConv3DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SlowConv3DBackwardBackward>(new SlowConv3DBackwardBackward(), deleteNode);
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
    return at::redispatch::slow_conv3d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_, output_mask);
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
  throw_error_for_complex_autograd(grad_input, "slow_conv3d_backward");
  throw_error_for_complex_autograd(grad_weight, "slow_conv3d_backward");
  throw_error_for_complex_autograd(grad_bias, "slow_conv3d_backward");
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv3d_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, Tensor & output, Tensor & finput, Tensor & fgrad_input) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& output_ = unpack(output, "output", 6);
  auto& finput_ = unpack(finput, "finput", 7);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 8);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("slow_conv3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("slow_conv3d_forward");
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
    at::redispatch::slow_conv3d_forward_outf(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding, output_, finput_, fgrad_input_);
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
Tensor slow_conv_dilated2d(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<SlowConvDilated2DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SlowConvDilated2DBackward>(new SlowConvDilated2DBackward(), deleteNode);
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
    return at::redispatch::slow_conv_dilated2d(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding, dilation);
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
  throw_error_for_complex_autograd(result, "slow_conv_dilated2d");
  return result;
}
std::tuple<Tensor,Tensor,Tensor> slow_conv_transpose2d_backward_output_mask(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& columns_ = unpack(columns, "columns", 8);
  auto& ones_ = unpack(ones, "ones", 9);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight );
  (void)_any_requires_grad;
  check_no_requires_grad(columns, "columns");
  check_no_requires_grad(ones, "ones");
  std::shared_ptr<SlowConvTranspose2DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SlowConvTranspose2DBackwardBackward>(new SlowConvTranspose2DBackwardBackward(), deleteNode);
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
  c10::optional<Storage> columns__storage_saved =
    columns_.has_storage() ? c10::optional<Storage>(columns_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> columns__impl_saved;
  if (columns_.defined()) columns__impl_saved = columns_.getIntrusivePtr();
  c10::optional<Storage> ones__storage_saved =
    ones_.has_storage() ? c10::optional<Storage>(ones_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> ones__impl_saved;
  if (ones_.defined()) ones__impl_saved = ones_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::slow_conv_transpose2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, columns_, ones_, output_mask);
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
  if (columns__storage_saved.has_value())
    AT_ASSERT(columns__storage_saved.value().is_alias_of(columns_.storage()));
  if (columns__impl_saved) AT_ASSERT(columns__impl_saved == columns_.getIntrusivePtr());
  if (ones__storage_saved.has_value())
    AT_ASSERT(ones__storage_saved.value().is_alias_of(ones_.storage()));
  if (ones__impl_saved) AT_ASSERT(ones__impl_saved == ones_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  throw_error_for_complex_autograd(grad_input, "slow_conv_transpose2d_backward");
  throw_error_for_complex_autograd(grad_weight, "slow_conv_transpose2d_backward");
  throw_error_for_complex_autograd(grad_bias, "slow_conv_transpose2d_backward");
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor slow_conv_transpose3d(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<SlowConvTranspose3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SlowConvTranspose3DBackward>(new SlowConvTranspose3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
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
    return at::redispatch::slow_conv_transpose3d(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding, output_padding, dilation);
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
  throw_error_for_complex_autograd(result, "slow_conv_transpose3d");
  return result;
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv_transpose3d_backward_out_grad_output(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input, Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 8);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 9);
  auto& grad_input_ = unpack(grad_input, "grad_input", 10);
  auto& grad_weight_ = unpack(grad_weight, "grad_weight", 11);
  auto& grad_bias_ = unpack(grad_bias, "grad_bias", 12);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight, finput, fgrad_input );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, finput, fgrad_input )) {
    throw_error_out_requires_grad("slow_conv_transpose3d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("slow_conv_transpose3d_backward");
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
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_weight__storage_saved =
    grad_weight_.has_storage() ? c10::optional<Storage>(grad_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_weight__impl_saved;
  if (grad_weight_.defined()) grad_weight__impl_saved = grad_weight_.getIntrusivePtr();
  c10::optional<Storage> grad_bias__storage_saved =
    grad_bias_.has_storage() ? c10::optional<Storage>(grad_bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_bias__impl_saved;
  if (grad_bias_.defined()) grad_bias__impl_saved = grad_bias_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::slow_conv_transpose3d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, finput_, fgrad_input_, grad_input_, grad_weight_, grad_bias_);
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
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_weight__storage_saved.has_value())
    AT_ASSERT(grad_weight__storage_saved.value().is_alias_of(grad_weight_.storage()));
  if (grad_weight__impl_saved) AT_ASSERT(grad_weight__impl_saved == grad_weight_.getIntrusivePtr());
  if (grad_bias__storage_saved.has_value())
    AT_ASSERT(grad_bias__storage_saved.value().is_alias_of(grad_bias_.storage()));
  if (grad_bias__impl_saved) AT_ASSERT(grad_bias__impl_saved == grad_bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
Tensor smooth_l1_loss_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double beta) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, target );
  (void)_any_requires_grad;
  std::shared_ptr<SmoothL1LossBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SmoothL1LossBackwardBackward>(new SmoothL1LossBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
    grad_fn->beta = beta;
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
    return at::redispatch::smooth_l1_loss_backward(ks & c10::after_autograd_keyset, grad_output_, self_, target_, reduction, beta);
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
  throw_error_for_complex_autograd(result, "smooth_l1_loss_backward");
  return result;
}
Tensor & softshrink_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & lambd, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("softshrink");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("softshrink");
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
    at::redispatch::softshrink_outf(ks & c10::after_autograd_keyset, self_, lambd, out_);
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
std::tuple<Tensor,Tensor> solve(c10::DispatchKeySet ks, const Tensor & self, const Tensor & A) {
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  auto _any_requires_grad = compute_requires_grad( self, A );
  (void)_any_requires_grad;
  std::shared_ptr<SolveBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SolveBackward>(new SolveBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->A_ = SavedVariable(A, false);
  }
  Tensor solution;
  Tensor LU;
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
    return at::redispatch::solve(ks & c10::after_autograd_keyset, self_, A_);
  })();
  std::tie(solution, LU) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( solution ), grad_fn);
  }
  if (grad_fn) {
    grad_fn->solution_ = SavedVariable(solution, true);
  }
  return std::make_tuple(std::move(solution), std::move(LU));
}
int64_t sparse_dim(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::sparse_dim(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = _tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor & sparse_resize_and_clear_(c10::DispatchKeySet ks, Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("sparse_resize_and_clear_"), deleteNode);
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
    at::redispatch::sparse_resize_and_clear_(ks & c10::after_autograd_keyset, self_, size, sparse_dim, dense_dim);
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
Tensor special_entr(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SpecialEntrBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SpecialEntrBackward>(new SpecialEntrBackward(), deleteNode);
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
    return at::redispatch::special_entr(ks & c10::after_autograd_keyset, self_);
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
  throw_error_for_complex_autograd(result, "special_entr");
  return result;
}
std::vector<Tensor> split_with_sizes(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef split_sizes, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SplitWithSizesBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SplitWithSizesBackward>(new SplitWithSizesBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_options = self.options();
    grad_fn->split_sizes = split_sizes.vec();
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
    return at::redispatch::split_with_sizes(ks & c10::after_autograd_keyset, self_, split_sizes, dim);
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
Tensor sqrt(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SqrtBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SqrtBackward>(new SqrtBackward(), deleteNode);
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
    return at::redispatch::sqrt(ks & c10::after_autograd_keyset, self_);
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
Tensor & sqrt_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<SqrtBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SqrtBackward>(new SqrtBackward(), deleteNode);
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
    at::redispatch::sqrt_(ks & c10::after_autograd_keyset, self_);
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
Tensor & stack_out_out(c10::DispatchKeySet ks, TensorList tensors, int64_t dim, Tensor & out) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( tensors )) {
    throw_error_out_requires_grad("stack");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("stack");
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
    at::redispatch::stack_outf(ks & c10::after_autograd_keyset, tensors_, dim, out_);
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
std::tuple<Tensor,Tensor> std_mean(c10::DispatchKeySet ks, const Tensor & self, bool unbiased) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<StdMeanBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<StdMeanBackward1>(new StdMeanBackward1(), deleteNode);
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
    return at::redispatch::std_mean(ks & c10::after_autograd_keyset, self_, unbiased);
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
  throw_error_for_complex_autograd(result0, "std_mean");
  throw_error_for_complex_autograd(result1, "std_mean");
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> std_mean_dim(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<StdMeanBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<StdMeanBackward0>(new StdMeanBackward0(), deleteNode);
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
    return at::redispatch::std_mean(ks & c10::after_autograd_keyset, self_, dim, unbiased, keepdim);
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
  throw_error_for_complex_autograd(result0, "std_mean");
  throw_error_for_complex_autograd(result1, "std_mean");
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor &,Tensor &> symeig_out_e(c10::DispatchKeySet ks, const Tensor & self, bool eigenvectors, bool upper, Tensor & e, Tensor & V) {
  auto& self_ = unpack(self, "self", 0);
  auto& e_ = unpack(e, "e", 3);
  auto& V_ = unpack(V, "V", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("symeig");
  }
  if (compute_requires_grad( e, V )) {
    throw_error_out_requires_grad("symeig");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> e__storage_saved =
    e_.has_storage() ? c10::optional<Storage>(e_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> e__impl_saved;
  if (e_.defined()) e__impl_saved = e_.getIntrusivePtr();
  c10::optional<Storage> V__storage_saved =
    V_.has_storage() ? c10::optional<Storage>(V_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> V__impl_saved;
  if (V_.defined()) V__impl_saved = V_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::symeig_outf(ks & c10::after_autograd_keyset, self_, eigenvectors, upper, e_, V_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (e__storage_saved.has_value())
    AT_ASSERT(e__storage_saved.value().is_alias_of(e_.storage()));
  if (e__impl_saved) AT_ASSERT(e__impl_saved == e_.getIntrusivePtr());
  if (V__storage_saved.has_value())
    AT_ASSERT(V__storage_saved.value().is_alias_of(V_.storage()));
  if (V__impl_saved) AT_ASSERT(V__impl_saved == V_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( e, V ), grad_fn);
  }
  return std::forward_as_tuple(e, V);
}
Tensor tan(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<TanBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TanBackward>(new TanBackward(), deleteNode);
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
    return at::redispatch::tan(ks & c10::after_autograd_keyset, self_);
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
Tensor & tan_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<TanBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TanBackward>(new TanBackward(), deleteNode);
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
    at::redispatch::tan_(ks & c10::after_autograd_keyset, self_);
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
Tensor tanh_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & output) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, output );
  (void)_any_requires_grad;
  std::shared_ptr<TanhBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TanhBackwardBackward>(new TanhBackwardBackward(), deleteNode);
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
    return at::redispatch::tanh_backward(ks & c10::after_autograd_keyset, grad_output_, output_);
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
  return result;
}
Tensor & tensordot_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("tensordot");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("tensordot");
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
    at::redispatch::tensordot_outf(ks & c10::after_autograd_keyset, self_, other_, dims_self, dims_other, out_);
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
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 6);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 7);
  auto& grad_input_ = unpack(grad_input, "grad_input", 8);
  auto& grad_weight_ = unpack(grad_weight, "grad_weight", 9);
  auto& grad_bias_ = unpack(grad_bias, "grad_bias", 10);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight, finput, fgrad_input );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, finput, fgrad_input )) {
    throw_error_out_requires_grad("thnn_conv2d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("thnn_conv2d_backward");
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
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_weight__storage_saved =
    grad_weight_.has_storage() ? c10::optional<Storage>(grad_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_weight__impl_saved;
  if (grad_weight_.defined()) grad_weight__impl_saved = grad_weight_.getIntrusivePtr();
  c10::optional<Storage> grad_bias__storage_saved =
    grad_bias_.has_storage() ? c10::optional<Storage>(grad_bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_bias__impl_saved;
  if (grad_bias_.defined()) grad_bias__impl_saved = grad_bias_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::thnn_conv2d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_, grad_input_, grad_weight_, grad_bias_);
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
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_weight__storage_saved.has_value())
    AT_ASSERT(grad_weight__storage_saved.value().is_alias_of(grad_weight_.storage()));
  if (grad_weight__impl_saved) AT_ASSERT(grad_weight__impl_saved == grad_weight_.getIntrusivePtr());
  if (grad_bias__storage_saved.has_value())
    AT_ASSERT(grad_bias__storage_saved.value().is_alias_of(grad_bias_.storage()));
  if (grad_bias__impl_saved) AT_ASSERT(grad_bias__impl_saved == grad_bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
Tensor thnn_conv_depthwise2d_forward(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<ThnnConvDepthwise2DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ThnnConvDepthwise2DBackward>(new ThnnConvDepthwise2DBackward(), deleteNode);
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
    return at::redispatch::thnn_conv_depthwise2d_forward(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding, dilation);
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
  throw_error_for_complex_autograd(result, "thnn_conv_depthwise2d_forward");
  return result;
}
Tensor threshold(c10::DispatchKeySet ks, const Tensor & self, const Scalar & threshold, const Scalar & value) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ThresholdBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ThresholdBackward0>(new ThresholdBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
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
    return at::redispatch::threshold(ks & c10::after_autograd_keyset, self_, threshold, value);
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
  throw_error_for_complex_autograd(result, "threshold");
  return result;
}
Tensor & threshold_(c10::DispatchKeySet ks, Tensor & self, const Scalar & threshold, const Scalar & value) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<ThresholdBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ThresholdBackward1>(new ThresholdBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->threshold = threshold;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::threshold_(ks & c10::after_autograd_keyset, self_, threshold, value);
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
std::tuple<Tensor &,Tensor &> triangular_solve_out_X(c10::DispatchKeySet ks, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular, Tensor & X, Tensor & M) {
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  auto& X_ = unpack(X, "X", 5);
  auto& M_ = unpack(M, "M", 6);
  auto _any_requires_grad = compute_requires_grad( self, A );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, A )) {
    throw_error_out_requires_grad("triangular_solve");
  }
  if (compute_requires_grad( X, M )) {
    throw_error_out_requires_grad("triangular_solve");
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
  c10::optional<Storage> M__storage_saved =
    M_.has_storage() ? c10::optional<Storage>(M_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> M__impl_saved;
  if (M_.defined()) M__impl_saved = M_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::triangular_solve_outf(ks & c10::after_autograd_keyset, self_, A_, upper, transpose, unitriangular, X_, M_);
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
  if (M__storage_saved.has_value())
    AT_ASSERT(M__storage_saved.value().is_alias_of(M_.storage()));
  if (M__impl_saved) AT_ASSERT(M__impl_saved == M_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( X, M ), grad_fn);
  }
  return std::forward_as_tuple(X, M);
}
Tensor tril(c10::DispatchKeySet ks, const Tensor & self, int64_t diagonal) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<TrilBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TrilBackward>(new TrilBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::tril(ks & c10::after_autograd_keyset, self_, diagonal);
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
Tensor & tril_(c10::DispatchKeySet ks, Tensor & self, int64_t diagonal) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<TrilBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TrilBackward>(new TrilBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::tril_(ks & c10::after_autograd_keyset, self_, diagonal);
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
Tensor tril_indices(c10::DispatchKeySet ks, int64_t row, int64_t col, int64_t offset, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::tril_indices(ks & c10::after_autograd_keyset, row, col, offset, dtype, layout, device, pin_memory);
  })();
  auto result = std::move(_tmp);
  return result;
}
Tensor unfold(c10::DispatchKeySet ks, const Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<UnfoldBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UnfoldBackward>(new UnfoldBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dimension = dimension;
    grad_fn->size = size;
    grad_fn->step = step;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoNonVariableTypeMode guard;
    return at::redispatch::unfold(ks & c10::after_autograd_keyset, self_, dimension, size, step);
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
  throw_error_for_complex_autograd(result, "unfold");
  return result;
}
Tensor & uniform_(c10::DispatchKeySet ks, Tensor & self, double from, double to, c10::optional<Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<UniformBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UniformBackward>(new UniformBackward(), deleteNode);
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
    at::redispatch::uniform_(ks & c10::after_autograd_keyset, self_, from, to, generator);
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
std::vector<Tensor> unsafe_split_Tensor(c10::DispatchKeySet ks, const Tensor & self, int64_t split_size, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<UnsafeSplitBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UnsafeSplitBackward>(new UnsafeSplitBackward(), deleteNode);
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
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::unsafe_split(ks & c10::after_autograd_keyset, self_, split_size, dim);
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
Tensor upsample_bicubic2d_backward_vec(c10::DispatchKeySet ks, const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleBicubic2DBackwardBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleBicubic2DBackwardBackward1>(new UpsampleBicubic2DBackwardBackward1(), deleteNode);
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
    return at::redispatch::upsample_bicubic2d_backward(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, align_corners, scale_factors);
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
  throw_error_for_complex_autograd(result, "upsample_bicubic2d_backward");
  return result;
}
Tensor upsample_bicubic2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleBicubic2DBackwardBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleBicubic2DBackwardBackward0>(new UpsampleBicubic2DBackwardBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
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
    return at::redispatch::upsample_bicubic2d_backward(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, align_corners, scales_h, scales_w);
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
  throw_error_for_complex_autograd(result, "upsample_bicubic2d_backward");
  return result;
}
Tensor & upsample_bilinear2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 5);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_bilinear2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_bilinear2d");
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
    at::redispatch::upsample_bilinear2d_outf(ks & c10::after_autograd_keyset, self_, output_size, align_corners, scales_h, scales_w, out_);
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
Tensor & upsample_nearest1d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_nearest1d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_nearest1d");
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
    at::redispatch::upsample_nearest1d_outf(ks & c10::after_autograd_keyset, self_, output_size, scales, out_);
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
Tensor upsample_trilinear3d_backward_vec(c10::DispatchKeySet ks, const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleTrilinear3DBackwardBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleTrilinear3DBackwardBackward1>(new UpsampleTrilinear3DBackwardBackward1(), deleteNode);
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
    return at::redispatch::upsample_trilinear3d_backward(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, align_corners, scale_factors);
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
  throw_error_for_complex_autograd(result, "upsample_trilinear3d_backward");
  return result;
}
Tensor upsample_trilinear3d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleTrilinear3DBackwardBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleTrilinear3DBackwardBackward0>(new UpsampleTrilinear3DBackwardBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
    grad_fn->scales_d = scales_d;
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
    return at::redispatch::upsample_trilinear3d_backward(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
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
  throw_error_for_complex_autograd(result, "upsample_trilinear3d_backward");
  return result;
}
Tensor & var_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim, Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("var");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("var");
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
    at::redispatch::var_outf(ks & c10::after_autograd_keyset, self_, dim, unbiased, keepdim, out_);
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
Tensor view(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef size) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ViewBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ViewBackward>(new ViewBackward(), deleteNode);
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
    at::AutoNonVariableTypeMode guard;
    return at::redispatch::view(ks & c10::after_autograd_keyset, self_, size);
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
Tensor view_dtype(c10::DispatchKeySet ks, const Tensor & self, ScalarType dtype) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoNonVariableTypeMode guard;
    return at::redispatch::view(ks & c10::after_autograd_keyset, self_, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor xlogy_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<XlogyBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<XlogyBackward0>(new XlogyBackward0(), deleteNode);
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
    return at::redispatch::xlogy(ks & c10::after_autograd_keyset, self_, other_);
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
  throw_error_for_complex_autograd(result, "xlogy");
  return result;
}
Tensor xlogy_Scalar_Self(c10::DispatchKeySet ks, const Scalar & self, const Tensor & other) {
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( other );
  (void)_any_requires_grad;
  std::shared_ptr<XlogyBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<XlogyBackward1>(new XlogyBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( other ));
    grad_fn->self = self;
    grad_fn->other_ = SavedVariable(other, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::xlogy(ks & c10::after_autograd_keyset, self, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "xlogy");
  return result;
}
Tensor xlogy_Scalar_Other(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<XlogyBackward2> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<XlogyBackward2>(new XlogyBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
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
    return at::redispatch::xlogy(ks & c10::after_autograd_keyset, self_, other);
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
  throw_error_for_complex_autograd(result, "xlogy");
  return result;
}
Tensor & xlogy__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<XlogyBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<XlogyBackward0>(new XlogyBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
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
  {
    at::AutoNonVariableTypeMode guard;
    at::redispatch::xlogy_(ks & c10::after_autograd_keyset, self_, other_);
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
Tensor & xlogy__Scalar_Other(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<XlogyBackward2> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<XlogyBackward2>(new XlogyBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
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
    at::redispatch::xlogy_(ks & c10::after_autograd_keyset, self_, other);
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
Tensor & zero_(c10::DispatchKeySet ks, Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  std::shared_ptr<ZeroBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ZeroBackward>(new ZeroBackward(), deleteNode);
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
    at::redispatch::zero_(ks & c10::after_autograd_keyset, self_);
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
}
}

namespace {

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  m.impl("_addmv_impl_",
         TORCH_FN(VariableType::_addmv_impl_)
  );
  m.impl("_ctc_loss",
         TORCH_FN(VariableType::_ctc_loss)
  );
  m.impl("_cudnn_init_dropout_state",
         TORCH_FN(VariableType::_cudnn_init_dropout_state)
  );
  m.impl("_cudnn_rnn_flatten_weight",
         TORCH_FN(VariableType::_cudnn_rnn_flatten_weight)
  );
  m.impl("_embedding_bag_dense_backward",
         TORCH_FN(VariableType::_embedding_bag_dense_backward)
  );
  m.impl("_embedding_bag_per_sample_weights_backward",
         TORCH_FN(VariableType::_embedding_bag_per_sample_weights_backward)
  );
  m.impl("_empty_per_channel_affine_quantized",
         TORCH_FN(VariableType::_empty_per_channel_affine_quantized)
  );
  m.impl("_fake_quantize_learnable_per_tensor_affine",
         TORCH_FN(VariableType::_fake_quantize_learnable_per_tensor_affine)
  );
  m.impl("_fft_c2c",
         TORCH_FN(VariableType::_fft_c2c)
  );
  m.impl("_fft_c2r",
         TORCH_FN(VariableType::_fft_c2r)
  );
  m.impl("_fft_r2c",
         TORCH_FN(VariableType::_fft_r2c)
  );
  m.impl("_foreach_abs",
         TORCH_FN(VariableType::_foreach_abs)
  );
  m.impl("_foreach_abs_",
         TORCH_FN(VariableType::_foreach_abs_)
  );
  m.impl("_foreach_addcmul.Scalar",
         TORCH_FN(VariableType::_foreach_addcmul_Scalar)
  );
  m.impl("_foreach_addcmul.ScalarList",
         TORCH_FN(VariableType::_foreach_addcmul_ScalarList)
  );
  m.impl("_foreach_addcmul_.Scalar",
         TORCH_FN(VariableType::_foreach_addcmul__Scalar)
  );
  m.impl("_foreach_addcmul_.ScalarList",
         TORCH_FN(VariableType::_foreach_addcmul__ScalarList)
  );
  m.impl("_foreach_atan",
         TORCH_FN(VariableType::_foreach_atan)
  );
  m.impl("_foreach_atan_",
         TORCH_FN(VariableType::_foreach_atan_)
  );
  m.impl("_foreach_cos",
         TORCH_FN(VariableType::_foreach_cos)
  );
  m.impl("_foreach_cos_",
         TORCH_FN(VariableType::_foreach_cos_)
  );
  m.impl("_foreach_reciprocal",
         TORCH_FN(VariableType::_foreach_reciprocal)
  );
  m.impl("_foreach_reciprocal_",
         TORCH_FN(VariableType::_foreach_reciprocal_)
  );
  m.impl("_foreach_sin",
         TORCH_FN(VariableType::_foreach_sin)
  );
  m.impl("_foreach_sin_",
         TORCH_FN(VariableType::_foreach_sin_)
  );
  m.impl("_foreach_sub.Scalar",
         TORCH_FN(VariableType::_foreach_sub_Scalar)
  );
  m.impl("_foreach_sub.List",
         TORCH_FN(VariableType::_foreach_sub_List)
  );
  m.impl("_foreach_sub.ScalarList",
         TORCH_FN(VariableType::_foreach_sub_ScalarList)
  );
  m.impl("_foreach_sub_.Scalar",
         TORCH_FN(VariableType::_foreach_sub__Scalar)
  );
  m.impl("_foreach_sub_.List",
         TORCH_FN(VariableType::_foreach_sub__List)
  );
  m.impl("_foreach_sub_.ScalarList",
         TORCH_FN(VariableType::_foreach_sub__ScalarList)
  );
  m.impl("_linalg_inv_out_helper_",
         TORCH_FN(VariableType::_linalg_inv_out_helper_)
  );
  m.impl("_linalg_qr_helper",
         TORCH_FN(VariableType::_linalg_qr_helper)
  );
  m.impl("_linalg_solve_out_helper_",
         TORCH_FN(VariableType::_linalg_solve_out_helper_)
  );
  m.impl("_log_softmax_backward_data",
         TORCH_FN(VariableType::_log_softmax_backward_data)
  );
  m.impl("_lu_solve_helper",
         TORCH_FN(VariableType::_lu_solve_helper)
  );
  m.impl("_make_per_channel_quantized_tensor",
         TORCH_FN(VariableType::_make_per_channel_quantized_tensor)
  );
  m.impl("_mkldnn_reshape",
         TORCH_FN(VariableType::_mkldnn_reshape)
  );
  m.impl("_solve_helper",
         TORCH_FN(VariableType::_solve_helper)
  );
  m.impl("_stack.out",
         TORCH_FN(VariableType::_stack_out_out)
  );
  m.impl("_standard_gamma_grad",
         TORCH_FN(VariableType::_standard_gamma_grad)
  );
  m.impl("_svd_helper",
         TORCH_FN(VariableType::_svd_helper)
  );
  m.impl("_test_optional_intlist",
         TORCH_FN(VariableType::_test_optional_intlist)
  );
  m.impl("_unique",
         TORCH_FN(VariableType::_unique)
  );
  m.impl("_unique2",
         TORCH_FN(VariableType::_unique2)
  );
  m.impl("adaptive_avg_pool3d_backward",
         TORCH_FN(VariableType::adaptive_avg_pool3d_backward)
  );
  m.impl("adaptive_max_pool2d.out",
         TORCH_FN(VariableType::adaptive_max_pool2d_out_out)
  );
  m.impl("addbmm",
         TORCH_FN(VariableType::addbmm)
  );
  m.impl("addbmm_",
         TORCH_FN(VariableType::addbmm_)
  );
  m.impl("addcdiv.out",
         TORCH_FN(VariableType::addcdiv_out_out)
  );
  m.impl("addmv.out",
         TORCH_FN(VariableType::addmv_out_out)
  );
  m.impl("all.dim",
         TORCH_FN(VariableType::all_dim)
  );
  m.impl("all",
         TORCH_FN(VariableType::all)
  );
  m.impl("amax",
         TORCH_FN(VariableType::amax)
  );
  m.impl("angle.out",
         TORCH_FN(VariableType::angle_out_out)
  );
  m.impl("any.dim",
         TORCH_FN(VariableType::any_dim)
  );
  m.impl("any",
         TORCH_FN(VariableType::any)
  );
  m.impl("argmin",
         TORCH_FN(VariableType::argmin)
  );
  m.impl("as_strided",
         TORCH_FN(VariableType::as_strided)
  );
  m.impl("as_strided_",
         TORCH_FN(VariableType::as_strided_)
  );
  m.impl("atanh.out",
         TORCH_FN(VariableType::atanh_out_out)
  );
  m.impl("batch_norm_elemt",
         TORCH_FN(VariableType::batch_norm_elemt)
  );
  m.impl("binary_cross_entropy.out",
         TORCH_FN(VariableType::binary_cross_entropy_out_out)
  );
  m.impl("binomial",
         TORCH_FN(VariableType::binomial)
  );
  m.impl("bitwise_or.Tensor_out",
         TORCH_FN(VariableType::bitwise_or_out_Tensor_out)
  );
  m.impl("bitwise_or.Scalar_out",
         TORCH_FN(VariableType::bitwise_or_out_Scalar_out)
  );
  m.impl("bitwise_xor.Tensor_out",
         TORCH_FN(VariableType::bitwise_xor_out_Tensor_out)
  );
  m.impl("bitwise_xor.Scalar_out",
         TORCH_FN(VariableType::bitwise_xor_out_Scalar_out)
  );
  m.impl("ceil",
         TORCH_FN(VariableType::ceil)
  );
  m.impl("ceil_",
         TORCH_FN(VariableType::ceil_)
  );
  m.impl("channel_shuffle",
         TORCH_FN(VariableType::channel_shuffle)
  );
  m.impl("cholesky_solve.out",
         TORCH_FN(VariableType::cholesky_solve_out_out)
  );
  m.impl("clamp_min.out",
         TORCH_FN(VariableType::clamp_min_out_out)
  );
  m.impl("col2im_backward",
         TORCH_FN(VariableType::col2im_backward)
  );
  m.impl("constant_pad_nd",
         TORCH_FN(VariableType::constant_pad_nd)
  );
  m.impl("conv_depthwise3d_backward.output_mask",
         TORCH_FN(VariableType::conv_depthwise3d_backward_output_mask)
  );
  m.impl("copy_sparse_to_sparse_",
         TORCH_FN(VariableType::copy_sparse_to_sparse_)
  );
  m.impl("cosh.out",
         TORCH_FN(VariableType::cosh_out_out)
  );
  m.impl("cross.out",
         TORCH_FN(VariableType::cross_out_out)
  );
  m.impl("cudnn_affine_grid_generator_backward",
         TORCH_FN(VariableType::cudnn_affine_grid_generator_backward)
  );
  m.impl("cudnn_convolution_backward",
         TORCH_FN(VariableType::cudnn_convolution_backward)
  );
  m.impl("cudnn_convolution_backward_input",
         TORCH_FN(VariableType::cudnn_convolution_backward_input)
  );
  m.impl("cudnn_convolution_transpose.deprecated",
         TORCH_FN(VariableType::cudnn_convolution_transpose_deprecated)
  );
  m.impl("cudnn_convolution_transpose.deprecated2",
         TORCH_FN(VariableType::cudnn_convolution_transpose_deprecated2)
  );
  m.impl("cudnn_convolution_transpose",
         TORCH_FN(VariableType::cudnn_convolution_transpose)
  );
  m.impl("cummin.out",
         TORCH_FN(VariableType::cummin_out_out)
  );
  m.impl("div.Tensor",
         TORCH_FN(VariableType::div_Tensor)
  );
  m.impl("div.Tensor_mode",
         TORCH_FN(VariableType::div_Tensor_mode)
  );
  m.impl("div.Scalar",
         TORCH_FN(VariableType::div_Scalar)
  );
  m.impl("div.Scalar_mode",
         TORCH_FN(VariableType::div_Scalar_mode)
  );
  m.impl("div_.Tensor",
         TORCH_FN(VariableType::div__Tensor)
  );
  m.impl("div_.Tensor_mode",
         TORCH_FN(VariableType::div__Tensor_mode)
  );
  m.impl("div_.Scalar",
         TORCH_FN(VariableType::div__Scalar)
  );
  m.impl("div_.Scalar_mode",
         TORCH_FN(VariableType::div__Scalar_mode)
  );
  m.impl("eig.e",
         TORCH_FN(VariableType::eig_out_e)
  );
  m.impl("eq.Scalar_out",
         TORCH_FN(VariableType::eq_out_Scalar_out)
  );
  m.impl("eq.Tensor_out",
         TORCH_FN(VariableType::eq_out_Tensor_out)
  );
  m.impl("exp",
         TORCH_FN(VariableType::exp)
  );
  m.impl("exp2",
         TORCH_FN(VariableType::exp2)
  );
  m.impl("exp2_",
         TORCH_FN(VariableType::exp2_)
  );
  m.impl("exp_",
         TORCH_FN(VariableType::exp_)
  );
  m.impl("fill_.Scalar",
         TORCH_FN(VariableType::fill__Scalar)
  );
  m.impl("fill_.Tensor",
         TORCH_FN(VariableType::fill__Tensor)
  );
  m.impl("fmax",
         TORCH_FN(VariableType::fmax)
  );
  m.impl("fractional_max_pool2d_backward",
         TORCH_FN(VariableType::fractional_max_pool2d_backward)
  );
  m.impl("fractional_max_pool3d",
         TORCH_FN(VariableType::fractional_max_pool3d)
  );
  m.impl("fractional_max_pool3d_backward.grad_input",
         TORCH_FN(VariableType::fractional_max_pool3d_backward_out_grad_input)
  );
  m.impl("frexp.Tensor_out",
         TORCH_FN(VariableType::frexp_out_Tensor_out)
  );
  m.impl("ge.Scalar_out",
         TORCH_FN(VariableType::ge_out_Scalar_out)
  );
  m.impl("ge.Tensor_out",
         TORCH_FN(VariableType::ge_out_Tensor_out)
  );
  m.impl("geqrf",
         TORCH_FN(VariableType::geqrf)
  );
  m.impl("ger",
         TORCH_FN(VariableType::ger)
  );
  m.impl("glu",
         TORCH_FN(VariableType::glu)
  );
  m.impl("glu_backward.grad_input",
         TORCH_FN(VariableType::glu_backward_out_grad_input)
  );
  m.impl("grid_sampler_2d",
         TORCH_FN(VariableType::grid_sampler_2d)
  );
  m.impl("gt.Scalar_out",
         TORCH_FN(VariableType::gt_out_Scalar_out)
  );
  m.impl("gt.Tensor_out",
         TORCH_FN(VariableType::gt_out_Tensor_out)
  );
  m.impl("hardsigmoid",
         TORCH_FN(VariableType::hardsigmoid)
  );
  m.impl("hardsigmoid_",
         TORCH_FN(VariableType::hardsigmoid_)
  );
  m.impl("hardswish",
         TORCH_FN(VariableType::hardswish)
  );
  m.impl("hardswish_",
         TORCH_FN(VariableType::hardswish_)
  );
  m.impl("hardtanh_backward",
         TORCH_FN(VariableType::hardtanh_backward)
  );
  m.impl("histc.out",
         TORCH_FN(VariableType::histc_out_out)
  );
  m.impl("hspmm.out",
         TORCH_FN(VariableType::hspmm_out_out)
  );
  m.impl("huber_loss",
         TORCH_FN(VariableType::huber_loss)
  );
  m.impl("huber_loss_backward.out",
         TORCH_FN(VariableType::huber_loss_backward_out_out)
  );
  m.impl("hypot.out",
         TORCH_FN(VariableType::hypot_out_out)
  );
  m.impl("i0",
         TORCH_FN(VariableType::i0)
  );
  m.impl("i0_",
         TORCH_FN(VariableType::i0_)
  );
  m.impl("igammac.out",
         TORCH_FN(VariableType::igammac_out_out)
  );
  m.impl("im2col_backward",
         TORCH_FN(VariableType::im2col_backward)
  );
  m.impl("index_add_",
         TORCH_FN(VariableType::index_add_)
  );
  m.impl("inverse.out",
         TORCH_FN(VariableType::inverse_out_out)
  );
  m.impl("kl_div",
         TORCH_FN(VariableType::kl_div)
  );
  m.impl("kthvalue",
         TORCH_FN(VariableType::kthvalue)
  );
  m.impl("le.Scalar_out",
         TORCH_FN(VariableType::le_out_Scalar_out)
  );
  m.impl("le.Tensor_out",
         TORCH_FN(VariableType::le_out_Tensor_out)
  );
  m.impl("leaky_relu.out",
         TORCH_FN(VariableType::leaky_relu_out_out)
  );
  m.impl("lgamma",
         TORCH_FN(VariableType::lgamma)
  );
  m.impl("lgamma_",
         TORCH_FN(VariableType::lgamma_)
  );
  m.impl("linalg_eigh.eigvals",
         TORCH_FN(VariableType::linalg_eigh_out_eigvals)
  );
  m.impl("linalg_inv.out",
         TORCH_FN(VariableType::linalg_inv_out_out)
  );
  m.impl("linalg_lstsq",
         TORCH_FN(VariableType::linalg_lstsq)
  );
  m.impl("linalg_qr",
         TORCH_FN(VariableType::linalg_qr)
  );
  m.impl("linalg_solve.out",
         TORCH_FN(VariableType::linalg_solve_out_out)
  );
  m.impl("linalg_vector_norm.out",
         TORCH_FN(VariableType::linalg_vector_norm_out_out)
  );
  m.impl("log10.out",
         TORCH_FN(VariableType::log10_out_out)
  );
  m.impl("log1p",
         TORCH_FN(VariableType::log1p)
  );
  m.impl("log1p_",
         TORCH_FN(VariableType::log1p_)
  );
  m.impl("logit",
         TORCH_FN(VariableType::logit)
  );
  m.impl("logit_",
         TORCH_FN(VariableType::logit_)
  );
  m.impl("logit_backward.grad_input",
         TORCH_FN(VariableType::logit_backward_out_grad_input)
  );
  m.impl("lt.Scalar_out",
         TORCH_FN(VariableType::lt_out_Scalar_out)
  );
  m.impl("lt.Tensor_out",
         TORCH_FN(VariableType::lt_out_Tensor_out)
  );
  m.impl("lu_solve",
         TORCH_FN(VariableType::lu_solve)
  );
  m.impl("masked_select.out",
         TORCH_FN(VariableType::masked_select_out_out)
  );
  m.impl("max_pool2d_with_indices_backward",
         TORCH_FN(VariableType::max_pool2d_with_indices_backward)
  );
  m.impl("max_pool3d_with_indices",
         TORCH_FN(VariableType::max_pool3d_with_indices)
  );
  m.impl("max_pool3d_with_indices_backward.grad_input",
         TORCH_FN(VariableType::max_pool3d_with_indices_backward_out_grad_input)
  );
  m.impl("min.dim_min",
         TORCH_FN(VariableType::min_out_dim_min)
  );
  m.impl("minimum.out",
         TORCH_FN(VariableType::minimum_out_out)
  );
  m.impl("miopen_convolution_backward_weight",
         TORCH_FN(VariableType::miopen_convolution_backward_weight)
  );
  m.impl("miopen_depthwise_convolution_backward",
         TORCH_FN(VariableType::miopen_depthwise_convolution_backward)
  );
  m.impl("miopen_depthwise_convolution_backward_input",
         TORCH_FN(VariableType::miopen_depthwise_convolution_backward_input)
  );
  m.impl("miopen_rnn_backward",
         TORCH_FN(VariableType::miopen_rnn_backward)
  );
  m.impl("mkldnn_linear_backward_weights",
         TORCH_FN(VariableType::mkldnn_linear_backward_weights)
  );
  m.impl("mkldnn_max_pool2d_backward",
         TORCH_FN(VariableType::mkldnn_max_pool2d_backward)
  );
  m.impl("mkldnn_max_pool3d",
         TORCH_FN(VariableType::mkldnn_max_pool3d)
  );
  m.impl("mm",
         TORCH_FN(VariableType::mm)
  );
  m.impl("mse_loss.out",
         TORCH_FN(VariableType::mse_loss_out_out)
  );
  m.impl("mul.out",
         TORCH_FN(VariableType::mul_out_out)
  );
  m.impl("multi_margin_loss_backward",
         TORCH_FN(VariableType::multi_margin_loss_backward)
  );
  m.impl("nanmedian.dim_values",
         TORCH_FN(VariableType::nanmedian_out_dim_values)
  );
  m.impl("nansum",
         TORCH_FN(VariableType::nansum)
  );
  m.impl("nansum.dim_IntList",
         TORCH_FN(VariableType::nansum_dim_IntList)
  );
  m.impl("narrow_copy",
         TORCH_FN(VariableType::narrow_copy)
  );
  m.impl("native_group_norm",
         TORCH_FN(VariableType::native_group_norm)
  );
  m.impl("native_layer_norm_backward",
         TORCH_FN(VariableType::native_layer_norm_backward)
  );
  m.impl("neg.out",
         TORCH_FN(VariableType::neg_out_out)
  );
  m.impl("nonzero.out",
         TORCH_FN(VariableType::nonzero_out_out)
  );
  m.impl("norm.dtype_out",
         TORCH_FN(VariableType::norm_out_dtype_out)
  );
  m.impl("norm.out",
         TORCH_FN(VariableType::norm_out_out)
  );
  m.impl("normal.Tensor_float_out",
         TORCH_FN(VariableType::normal_out_Tensor_float_out)
  );
  m.impl("normal.float_Tensor_out",
         TORCH_FN(VariableType::normal_out_float_Tensor_out)
  );
  m.impl("normal.Tensor_Tensor_out",
         TORCH_FN(VariableType::normal_out_Tensor_Tensor_out)
  );
  m.impl("prelu_backward",
         TORCH_FN(VariableType::prelu_backward)
  );
  m.impl("q_scale",
         TORCH_FN(VariableType::q_scale)
  );
  m.impl("quantized_max_pool1d",
         TORCH_FN(VariableType::quantized_max_pool1d)
  );
  m.impl("renorm.out",
         TORCH_FN(VariableType::renorm_out_out)
  );
  m.impl("replication_pad2d_backward",
         TORCH_FN(VariableType::replication_pad2d_backward)
  );
  m.impl("replication_pad3d",
         TORCH_FN(VariableType::replication_pad3d)
  );
  m.impl("replication_pad3d_backward.grad_input",
         TORCH_FN(VariableType::replication_pad3d_backward_out_grad_input)
  );
  m.impl("scatter_.src",
         TORCH_FN(VariableType::scatter__src)
  );
  m.impl("scatter_.value",
         TORCH_FN(VariableType::scatter__value)
  );
  m.impl("scatter_.reduce",
         TORCH_FN(VariableType::scatter__reduce)
  );
  m.impl("scatter_.value_reduce",
         TORCH_FN(VariableType::scatter__value_reduce)
  );
  m.impl("sgn",
         TORCH_FN(VariableType::sgn)
  );
  m.impl("sgn_",
         TORCH_FN(VariableType::sgn_)
  );
  m.impl("sigmoid",
         TORCH_FN(VariableType::sigmoid)
  );
  m.impl("sigmoid_",
         TORCH_FN(VariableType::sigmoid_)
  );
  m.impl("sigmoid_backward.grad_input",
         TORCH_FN(VariableType::sigmoid_backward_out_grad_input)
  );
  m.impl("sign",
         TORCH_FN(VariableType::sign)
  );
  m.impl("sign_",
         TORCH_FN(VariableType::sign_)
  );
  m.impl("sinc.out",
         TORCH_FN(VariableType::sinc_out_out)
  );
  m.impl("sinh.out",
         TORCH_FN(VariableType::sinh_out_out)
  );
  m.impl("slice.Tensor",
         TORCH_FN(VariableType::slice_Tensor)
  );
  m.impl("slow_conv3d_backward.output_mask",
         TORCH_FN(VariableType::slow_conv3d_backward_output_mask)
  );
  m.impl("slow_conv3d_forward.output",
         TORCH_FN(VariableType::slow_conv3d_forward_out_output)
  );
  m.impl("slow_conv_dilated2d",
         TORCH_FN(VariableType::slow_conv_dilated2d)
  );
  m.impl("slow_conv_transpose2d_backward.output_mask",
         TORCH_FN(VariableType::slow_conv_transpose2d_backward_output_mask)
  );
  m.impl("slow_conv_transpose3d",
         TORCH_FN(VariableType::slow_conv_transpose3d)
  );
  m.impl("slow_conv_transpose3d_backward.grad_output",
         TORCH_FN(VariableType::slow_conv_transpose3d_backward_out_grad_output)
  );
  m.impl("smooth_l1_loss_backward",
         TORCH_FN(VariableType::smooth_l1_loss_backward)
  );
  m.impl("softshrink.out",
         TORCH_FN(VariableType::softshrink_out_out)
  );
  m.impl("solve",
         TORCH_FN(VariableType::solve)
  );
  m.impl("sparse_dim",
         TORCH_FN(VariableType::sparse_dim)
  );
  m.impl("sparse_resize_and_clear_",
         TORCH_FN(VariableType::sparse_resize_and_clear_)
  );
  m.impl("special_entr",
         TORCH_FN(VariableType::special_entr)
  );
  m.impl("split_with_sizes",
         TORCH_FN(VariableType::split_with_sizes)
  );
  m.impl("sqrt",
         TORCH_FN(VariableType::sqrt)
  );
  m.impl("sqrt_",
         TORCH_FN(VariableType::sqrt_)
  );
  m.impl("stack.out",
         TORCH_FN(VariableType::stack_out_out)
  );
  m.impl("std_mean",
         TORCH_FN(VariableType::std_mean)
  );
  m.impl("std_mean.dim",
         TORCH_FN(VariableType::std_mean_dim)
  );
  m.impl("symeig.e",
         TORCH_FN(VariableType::symeig_out_e)
  );
  m.impl("tan",
         TORCH_FN(VariableType::tan)
  );
  m.impl("tan_",
         TORCH_FN(VariableType::tan_)
  );
  m.impl("tanh_backward",
         TORCH_FN(VariableType::tanh_backward)
  );
  m.impl("tensordot.out",
         TORCH_FN(VariableType::tensordot_out_out)
  );
  m.impl("thnn_conv2d_backward.grad_input",
         TORCH_FN(VariableType::thnn_conv2d_backward_out_grad_input)
  );
  m.impl("thnn_conv_depthwise2d_forward",
         TORCH_FN(VariableType::thnn_conv_depthwise2d_forward)
  );
  m.impl("threshold",
         TORCH_FN(VariableType::threshold)
  );
  m.impl("threshold_",
         TORCH_FN(VariableType::threshold_)
  );
  m.impl("triangular_solve.X",
         TORCH_FN(VariableType::triangular_solve_out_X)
  );
  m.impl("tril",
         TORCH_FN(VariableType::tril)
  );
  m.impl("tril_",
         TORCH_FN(VariableType::tril_)
  );
  m.impl("tril_indices",
         TORCH_FN(VariableType::tril_indices)
  );
  m.impl("unfold",
         TORCH_FN(VariableType::unfold)
  );
  m.impl("uniform_",
         TORCH_FN(VariableType::uniform_)
  );
  m.impl("unsafe_split.Tensor",
         TORCH_FN(VariableType::unsafe_split_Tensor)
  );
  m.impl("upsample_bicubic2d_backward.vec",
         TORCH_FN(VariableType::upsample_bicubic2d_backward_vec)
  );
  m.impl("upsample_bicubic2d_backward",
         TORCH_FN(VariableType::upsample_bicubic2d_backward)
  );
  m.impl("upsample_bilinear2d.out",
         TORCH_FN(VariableType::upsample_bilinear2d_out_out)
  );
  m.impl("upsample_nearest1d.out",
         TORCH_FN(VariableType::upsample_nearest1d_out_out)
  );
  m.impl("upsample_trilinear3d_backward.vec",
         TORCH_FN(VariableType::upsample_trilinear3d_backward_vec)
  );
  m.impl("upsample_trilinear3d_backward",
         TORCH_FN(VariableType::upsample_trilinear3d_backward)
  );
  m.impl("var.out",
         TORCH_FN(VariableType::var_out_out)
  );
  m.impl("view",
         TORCH_FN(VariableType::view)
  );
  m.impl("view.dtype",
         TORCH_FN(VariableType::view_dtype)
  );
  m.impl("xlogy.Tensor",
         TORCH_FN(VariableType::xlogy_Tensor)
  );
  m.impl("xlogy.Scalar_Self",
         TORCH_FN(VariableType::xlogy_Scalar_Self)
  );
  m.impl("xlogy.Scalar_Other",
         TORCH_FN(VariableType::xlogy_Scalar_Other)
  );
  m.impl("xlogy_.Tensor",
         TORCH_FN(VariableType::xlogy__Tensor)
  );
  m.impl("xlogy_.Scalar_Other",
         TORCH_FN(VariableType::xlogy__Scalar_Other)
  );
  m.impl("zero_",
         TORCH_FN(VariableType::zero_)
  );
}

}

}} // namespace torch::autograd
