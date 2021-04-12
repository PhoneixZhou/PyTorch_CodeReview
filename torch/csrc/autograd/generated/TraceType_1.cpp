#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <torch/library.h>

#include "torch/csrc/autograd/function.h"

#include <ATen/RedispatchFunctions.h>
#include "ATen/quantized/Quantizer.h"

// @generated from tools/autograd/templates/TraceType.cpp

// See the `Tracer` section in `torch/csrc/jit/OVERVIEW.md`.
// NOTE See [Sharded File] comment in VariableType

using namespace at;

namespace torch {

namespace TraceType {

namespace {
Tensor & __irshift___Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__irshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::__irshift__(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & __irshift___Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__irshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::__irshift__(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor __rshift___Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__rshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::__rshift__(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor __rshift___Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__rshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::__rshift__(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _adaptive_avg_pool2d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_adaptive_avg_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_adaptive_avg_pool2d(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & _add_relu_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_add_relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_add_relu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::_add_relu_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor _amp_update_scale(c10::DispatchKeySet ks, Tensor & growth_tracker, const Tensor & current_scale, const Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_amp_update_scale");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "growth_tracker", growth_tracker);
    jit::tracer::addInputs(node, "current_scale", current_scale);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    jit::tracer::addInputs(node, "scale_growth_factor", scale_growth_factor);
    jit::tracer::addInputs(node, "scale_backoff_factor", scale_backoff_factor);
    jit::tracer::addInputs(node, "growth_interval", growth_interval);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_amp_update_scale(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), growth_tracker, current_scale, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _bmm(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat2, bool deterministic) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_bmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_bmm(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat2, deterministic);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _cast_Byte(c10::DispatchKeySet ks, const Tensor & self, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Byte");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_cast_Byte(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _cast_Half(c10::DispatchKeySet ks, const Tensor & self, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Half");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_cast_Half(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _cast_Int(c10::DispatchKeySet ks, const Tensor & self, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Int");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_cast_Int(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & _cat_out_out(c10::DispatchKeySet ks, TensorList tensors, int64_t dim, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cat_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::_cat_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor _cdist_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & x1, const Tensor & x2, double p, const Tensor & cdist) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cdist_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "cdist", cdist);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_cdist_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, x1, x2, p, cdist);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _cholesky_helper(c10::DispatchKeySet ks, const Tensor & self, bool upper) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cholesky_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_cholesky_helper(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, upper);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & _coalesced_(c10::DispatchKeySet ks, Tensor & self, bool coalesced) {
  at::redispatch::_coalesced_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, coalesced);
  return self;
}
Tensor _compute_linear_combination(c10::DispatchKeySet ks, const Tensor & input, const Tensor & coefficients) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_compute_linear_combination");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "coefficients", coefficients);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_compute_linear_combination(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, coefficients);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _conj(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_conj");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_conj(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _convolution(c10::DispatchKeySet ks, const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    jit::tracer::addInputs(node, "allow_tf32", allow_tf32);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_convolution(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _convolution_deprecated(c10::DispatchKeySet ks, const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_convolution(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> _cudnn_rnn_backward(c10::DispatchKeySet ks, const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const c10::optional<Tensor> & cx, const Tensor & output, const c10::optional<Tensor> & grad_output, const c10::optional<Tensor> & grad_hy, const c10::optional<Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const c10::optional<Tensor> & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_rnn_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "weight_buf", weight_buf);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "grad_cy", grad_cy);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "proj_size", proj_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "dropout_state", dropout_state);
    jit::tracer::addInputs(node, "reserve", reserve);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  std::vector<Tensor> result3;
  std::tie(result0, result1, result2, result3) =at::redispatch::_cudnn_rnn_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
int64_t _cufft_get_plan_cache_size(c10::DispatchKeySet ks, int64_t device_index) {
  auto result =at::redispatch::_cufft_get_plan_cache_size(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), device_index);
  return result;
}
void _cummax_helper(c10::DispatchKeySet ks, const Tensor & self, Tensor & values, Tensor & indices, int64_t dim) {
  at::redispatch::_cummax_helper(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, values, indices, dim);
}
Tensor & _cumprod_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cumprod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cumprod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::_cumprod_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor _cumsum(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_cumsum(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _dim_arange(c10::DispatchKeySet ks, const Tensor & like, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_dim_arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "like", like);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_dim_arange(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), like, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _dirichlet_grad(c10::DispatchKeySet ks, const Tensor & x, const Tensor & alpha, const Tensor & total) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_dirichlet_grad");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "total", total);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_dirichlet_grad(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, alpha, total);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _embedding_bag_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<Tensor> & per_sample_weights) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_embedding_bag_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "bag_size", bag_size);
    jit::tracer::addInputs(node, "maximum_indices", maximum_indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_embedding_bag_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> _fake_quantize_learnable_per_channel_affine_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_fake_quantize_learnable_per_channel_affine_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    jit::tracer::addInputs(node, "grad_factor", grad_factor);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  std::tie(result0, result1, result2) =at::redispatch::_fake_quantize_learnable_per_channel_affine_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::vector<Tensor> _foreach_ceil(c10::DispatchKeySet ks, TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_foreach_ceil");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_foreach_ceil(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_ceil_(c10::DispatchKeySet ks, TensorList self) {
  at::redispatch::_foreach_ceil_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
std::vector<Tensor> _foreach_div_Scalar(c10::DispatchKeySet ks, TensorList tensors, const Scalar & scalar) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_foreach_div");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "scalar", scalar);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_foreach_div(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, scalar);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::vector<Tensor> _foreach_div_List(c10::DispatchKeySet ks, TensorList tensors1, TensorList tensors2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_foreach_div");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors1", tensors1);
    jit::tracer::addInputs(node, "tensors2", tensors2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_foreach_div(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors1, tensors2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::vector<Tensor> _foreach_div_ScalarList(c10::DispatchKeySet ks, TensorList tensors, ArrayRef<Scalar> scalars) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_foreach_div");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "scalars", scalars);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_foreach_div(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, scalars);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_div__Scalar(c10::DispatchKeySet ks, TensorList self, const Scalar & scalar) {
  at::redispatch::_foreach_div_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar);
}
void _foreach_div__List(c10::DispatchKeySet ks, TensorList self, TensorList other) {
  at::redispatch::_foreach_div_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
}
void _foreach_div__ScalarList(c10::DispatchKeySet ks, TensorList self, ArrayRef<Scalar> scalars) {
  at::redispatch::_foreach_div_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars);
}
std::vector<Tensor> _foreach_exp(c10::DispatchKeySet ks, TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_foreach_exp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_foreach_exp(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_exp_(c10::DispatchKeySet ks, TensorList self) {
  at::redispatch::_foreach_exp_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
std::vector<Tensor> _foreach_lgamma(c10::DispatchKeySet ks, TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_foreach_lgamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_foreach_lgamma(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_lgamma_(c10::DispatchKeySet ks, TensorList self) {
  at::redispatch::_foreach_lgamma_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
std::vector<Tensor> _foreach_log1p(c10::DispatchKeySet ks, TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_foreach_log1p");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_foreach_log1p(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_log1p_(c10::DispatchKeySet ks, TensorList self) {
  at::redispatch::_foreach_log1p_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
std::vector<Tensor> _foreach_sigmoid(c10::DispatchKeySet ks, TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_foreach_sigmoid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_foreach_sigmoid(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_sigmoid_(c10::DispatchKeySet ks, TensorList self) {
  at::redispatch::_foreach_sigmoid_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
std::vector<Tensor> _foreach_sqrt(c10::DispatchKeySet ks, TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_foreach_sqrt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_foreach_sqrt(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_sqrt_(c10::DispatchKeySet ks, TensorList self) {
  at::redispatch::_foreach_sqrt_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
std::vector<Tensor> _foreach_tan(c10::DispatchKeySet ks, TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_foreach_tan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_foreach_tan(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_tan_(c10::DispatchKeySet ks, TensorList self) {
  at::redispatch::_foreach_tan_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
void _foreach_zero_(c10::DispatchKeySet ks, TensorList self) {
  at::redispatch::_foreach_zero_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
std::tuple<Tensor,Tensor> _fused_dropout(c10::DispatchKeySet ks, const Tensor & self, double p, c10::optional<Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_fused_dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  std::tie(result0, result1) =at::redispatch::_fused_dropout(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor _logcumsumexp(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_logcumsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_logcumsumexp(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _make_dual(c10::DispatchKeySet ks, const Tensor & primal, const Tensor & tangent, int64_t level) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_make_dual");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "primal", primal);
    jit::tracer::addInputs(node, "tangent", tangent);
    jit::tracer::addInputs(node, "level", level);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_make_dual(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), primal, tangent, level);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _make_per_tensor_quantized_tensor(c10::DispatchKeySet ks, const Tensor & self, double scale, int64_t zero_point) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_make_per_tensor_quantized_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_make_per_tensor_quantized_tensor(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> _mode(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  std::tie(result0, result1) =at::redispatch::_mode(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor _nnpack_spatial_convolution(c10::DispatchKeySet ks, const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef stride) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_nnpack_spatial_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_nnpack_spatial_convolution(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, padding, stride);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _softmax_backward_data(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_softmax_backward_data");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_softmax_backward_data(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output, dim, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _sparse_log_softmax_int(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_log_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_sparse_log_softmax(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _sparse_log_softmax_Dimname(c10::DispatchKeySet ks, const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_log_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_sparse_log_softmax(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _sparse_log_softmax(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool half_to_float) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_log_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "half_to_float", half_to_float);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_sparse_log_softmax(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, half_to_float);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _sparse_sum_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & self, IntArrayRef dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_sparse_sum_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _std(c10::DispatchKeySet ks, const Tensor & self, bool unbiased) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_std(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, unbiased);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _test_ambiguous_defaults_a(c10::DispatchKeySet ks, const Tensor & dummy, int64_t a, int64_t b) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_test_ambiguous_defaults");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "dummy", dummy);
    jit::tracer::addInputs(node, "a", a);
    jit::tracer::addInputs(node, "b", b);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_test_ambiguous_defaults(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), dummy, a, b);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _test_ambiguous_defaults_b(c10::DispatchKeySet ks, const Tensor & dummy, int64_t a, std::string b) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_test_ambiguous_defaults");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "dummy", dummy);
    jit::tracer::addInputs(node, "a", a);
    jit::tracer::addInputs(node, "b", b);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_test_ambiguous_defaults(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), dummy, a, b);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor _test_optional_floatlist(c10::DispatchKeySet ks, const Tensor & values, c10::optional<ArrayRef<double>> addends) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_test_optional_floatlist");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "addends", addends);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_test_optional_floatlist(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), values, addends);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_differentiable_gru_cell_backward(c10::DispatchKeySet ks, const Tensor & grad_hy, const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const c10::optional<Tensor> & input_bias, const c10::optional<Tensor> & hidden_bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_differentiable_gru_cell_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  Tensor result4;
  std::tie(result0, result1, result2, result3, result4) =at::redispatch::_thnn_differentiable_gru_cell_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_gru_cell_backward(c10::DispatchKeySet ks, const Tensor & grad_hy, const Tensor & workspace, bool has_bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fused_gru_cell_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "workspace", workspace);
    jit::tracer::addInputs(node, "has_bias", has_bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  Tensor result4;
  std::tie(result0, result1, result2, result3, result4) =at::redispatch::_thnn_fused_gru_cell_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_hy, workspace, has_bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
bool _use_cudnn_rnn_flatten_weight(c10::DispatchKeySet ks) {
  auto result =at::redispatch::_use_cudnn_rnn_flatten_weight(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer));
  return result;
}
Tensor _values(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_values");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::_values(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface_backward(c10::DispatchKeySet ks, const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_weight_norm_cuda_interface_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_w", grad_w);
    jit::tracer::addInputs(node, "saved_v", saved_v);
    jit::tracer::addInputs(node, "saved_g", saved_g);
    jit::tracer::addInputs(node, "saved_norms", saved_norms);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  std::tie(result0, result1) =at::redispatch::_weight_norm_cuda_interface_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_w, saved_v, saved_g, saved_norms, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & acos_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::acos");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("acos_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::acos_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor acosh(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::acosh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::acosh(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & acosh_(c10::DispatchKeySet ks, Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::acosh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::acosh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("acosh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::acosh_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor adaptive_avg_pool2d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::adaptive_avg_pool2d(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & adaptive_avg_pool3d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_avg_pool3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::adaptive_avg_pool3d_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor adaptive_max_pool3d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::adaptive_max_pool3d_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & add_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("add_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::add_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor addr(c10::DispatchKeySet ks, const Tensor & self, const Tensor & vec1, const Tensor & vec2, const Scalar & beta, const Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::addr(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec1, vec2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & addr_(c10::DispatchKeySet ks, Tensor & self, const Tensor & vec1, const Tensor & vec2, const Scalar & beta, const Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addr");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addr_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addr_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::addr_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec1, vec2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor affine_grid_generator(c10::DispatchKeySet ks, const Tensor & theta, IntArrayRef size, bool align_corners) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::affine_grid_generator");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "theta", theta);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::affine_grid_generator(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), theta, size, align_corners);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor amin(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::amin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::amin(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & arange_out_out(c10::DispatchKeySet ks, const Scalar & end, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "end", end);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arange_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::arange_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), end, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & arange_out_start_out(c10::DispatchKeySet ks, const Scalar & start, const Scalar & end, const Scalar & step, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arange_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::arange_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, step, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & arctanh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::arctanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arctanh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::arctanh_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & asin_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::asin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("asin_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::asin_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor asinh(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::asinh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::asinh(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & asinh_(c10::DispatchKeySet ks, Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::asinh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::asinh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("asinh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::asinh_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor atleast_3d(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::atleast_3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::atleast_3d(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::vector<Tensor> atleast_3d_Sequence(c10::DispatchKeySet ks, TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::atleast_3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::atleast_3d(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor avg_pool2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::avg_pool2d_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor avg_pool3d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::avg_pool3d(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & avg_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::avg_pool3d_backward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor baddbmm(c10::DispatchKeySet ks, const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::baddbmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::baddbmm(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, batch1, batch2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & baddbmm_(c10::DispatchKeySet ks, Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::baddbmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::baddbmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("baddbmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::baddbmm_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, batch1, batch2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> batch_norm_backward_reduce(c10::DispatchKeySet ks, const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & weight, bool input_g, bool weight_g, bool bias_g) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_backward_reduce");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "input_g", input_g);
    jit::tracer::addInputs(node, "weight_g", weight_g);
    jit::tracer::addInputs(node, "bias_g", bias_g);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  std::tie(result0, result1, result2, result3) =at::redispatch::batch_norm_backward_reduce(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
Tensor & bernoulli_out_out(c10::DispatchKeySet ks, const Tensor & self, c10::optional<Generator> generator, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bernoulli");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bernoulli_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::bernoulli_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor binary_cross_entropy_with_logits(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & pos_weight, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy_with_logits");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "pos_weight", pos_weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::binary_cross_entropy_with_logits(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, pos_weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor bincount(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Tensor> & weights, int64_t minlength) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bincount");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weights", weights);
    jit::tracer::addInputs(node, "minlength", minlength);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::bincount(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weights, minlength);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor bitwise_and_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::bitwise_and(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor bitwise_and_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::bitwise_and(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & bitwise_and__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_and");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_and_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::bitwise_and_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & bitwise_and__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_and");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_and_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::bitwise_and_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor bitwise_not(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_not");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::bitwise_not(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & bitwise_not_(c10::DispatchKeySet ks, Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_not");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_not_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_not_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::bitwise_not_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor bmm(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::bmm(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor bucketize_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & boundaries, bool out_int32, bool right) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::bucketize(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, boundaries, out_int32, right);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor bucketize_Scalar(c10::DispatchKeySet ks, const Scalar & self, const Tensor & boundaries, bool out_int32, bool right) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::bucketize(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, boundaries, out_int32, right);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor cartesian_prod(c10::DispatchKeySet ks, TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cartesian_prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::cartesian_prod(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & cat_out_out(c10::DispatchKeySet ks, TensorList tensors, int64_t dim, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cat_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::cat_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & cat_out_names_out(c10::DispatchKeySet ks, TensorList tensors, Dimname dim, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cat_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::cat_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor chain_matmul(c10::DispatchKeySet ks, TensorList matrices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::chain_matmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "matrices", matrices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::chain_matmul(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), matrices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor cholesky(c10::DispatchKeySet ks, const Tensor & self, bool upper) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cholesky");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::cholesky(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, upper);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor clamp_max(c10::DispatchKeySet ks, const Tensor & self, const Scalar & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clamp_max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::clamp_max(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & clamp_max_(c10::DispatchKeySet ks, Tensor & self, const Scalar & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::clamp_max");
    } else {
      op_name = jit::Symbol::fromQualString("aten::clamp_max_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_max_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::clamp_max_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor coalesce(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::coalesce");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::coalesce(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & col2im_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::col2im");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("col2im_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::col2im_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, kernel_size, dilation, padding, stride, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor combinations(c10::DispatchKeySet ks, const Tensor & self, int64_t r, bool with_replacement) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::combinations");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "r", r);
    jit::tracer::addInputs(node, "with_replacement", with_replacement);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::combinations(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, r, with_replacement);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor conj(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::conj");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::conj(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor conv_tbc(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::conv_tbc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "pad", pad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::conv_tbc(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, pad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor convolution(c10::DispatchKeySet ks, const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
  auto result =at::redispatch::convolution(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  return result;
}
Tensor copysign_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::copysign");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::copysign(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor copysign_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::copysign");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::copysign(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & copysign__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::copysign");
    } else {
      op_name = jit::Symbol::fromQualString("aten::copysign_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("copysign_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::copysign_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & copysign__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::copysign");
    } else {
      op_name = jit::Symbol::fromQualString("aten::copysign_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("copysign_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::copysign_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor cosine_similarity(c10::DispatchKeySet ks, const Tensor & x1, const Tensor & x2, int64_t dim, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cosine_similarity");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::cosine_similarity(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x1, x2, dim, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor cudnn_convolution_backward_weight(c10::DispatchKeySet ks, IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_backward_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_size", weight_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "allow_tf32", allow_tf32);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::cudnn_convolution_backward_weight(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> cummax(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor values;
  Tensor indices;
  std::tie(values, indices) =at::redispatch::cummax(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> cummax_dimname(c10::DispatchKeySet ks, const Tensor & self, Dimname dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor values;
  Tensor indices;
  std::tie(values, indices) =at::redispatch::cummax(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor cummaxmin_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & input, const Tensor & indices, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummaxmin_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::cummaxmin_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, input, indices, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & cumprod_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumprod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumprod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::cumprod_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & cumprod_out_dimname_out(c10::DispatchKeySet ks, const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumprod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumprod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::cumprod_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor cumsum(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::cumsum(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor cumsum_dimname(c10::DispatchKeySet ks, const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::cumsum(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & cumsum_(c10::DispatchKeySet ks, Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::cumsum");
    } else {
      op_name = jit::Symbol::fromQualString("aten::cumsum_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::cumsum_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & cumsum__dimname(c10::DispatchKeySet ks, Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::cumsum");
    } else {
      op_name = jit::Symbol::fromQualString("aten::cumsum_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::cumsum_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
int64_t dense_dim(c10::DispatchKeySet ks, const Tensor & self) {
  auto result =at::redispatch::dense_dim(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
Tensor diag_backward(c10::DispatchKeySet ks, const Tensor & grad, IntArrayRef input_sizes, int64_t diagonal) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diag_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input_sizes", input_sizes);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::diag_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, input_sizes, diagonal);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor diagonal(c10::DispatchKeySet ks, const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diagonal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::diagonal(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, offset, dim1, dim2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor diagonal_Dimname(c10::DispatchKeySet ks, const Tensor & self, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diagonal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "outdim", outdim);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);
    jit::tracer::addInputs(node, "offset", offset);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::diagonal(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, outdim, dim1, dim2, offset);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor dist(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::dist");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::dist(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & dot_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & tensor, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::dot");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor", tensor);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("dot_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::dot_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor dropout(c10::DispatchKeySet ks, const Tensor & input, double p, bool train) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::dropout(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & dropout_(c10::DispatchKeySet ks, Tensor & self, double p, bool train) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::dropout");
    } else {
      op_name = jit::Symbol::fromQualString("aten::dropout_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::dropout_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor elu(c10::DispatchKeySet ks, const Tensor & self, const Scalar & alpha, const Scalar & scale, const Scalar & input_scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::elu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::elu(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, alpha, scale, input_scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & elu_(c10::DispatchKeySet ks, Tensor & self, const Scalar & alpha, const Scalar & scale, const Scalar & input_scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::elu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::elu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("elu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::elu_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, alpha, scale, input_scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & embedding_renorm_(c10::DispatchKeySet ks, Tensor & self, const Tensor & indices, double max_norm, double norm_type) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::embedding_renorm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::embedding_renorm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "max_norm", max_norm);
    jit::tracer::addInputs(node, "norm_type", norm_type);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("embedding_renorm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::embedding_renorm_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, max_norm, norm_type);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
bool equal(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  auto result =at::redispatch::equal(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  return result;
}
Tensor & erf_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::erf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erf_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::erf_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor erfc(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::erfc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::erfc(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & erfc_(c10::DispatchKeySet ks, Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::erfc");
    } else {
      op_name = jit::Symbol::fromQualString("aten::erfc_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfc_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::erfc_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor expm1(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::expm1");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::expm1(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & expm1_(c10::DispatchKeySet ks, Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::expm1");
    } else {
      op_name = jit::Symbol::fromQualString("aten::expm1_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("expm1_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::expm1_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & exponential_(c10::DispatchKeySet ks, Tensor & self, double lambd, c10::optional<Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::exponential");
    } else {
      op_name = jit::Symbol::fromQualString("aten::exponential_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("exponential_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::exponential_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, lambd, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor fake_quantize_per_channel_affine(c10::DispatchKeySet ks, const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fake_quantize_per_channel_affine");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::fake_quantize_per_channel_affine(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, axis, quant_min, quant_max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor fbgemm_linear_fp16_weight_fp32_activation(c10::DispatchKeySet ks, const Tensor & input, const Tensor & packed_weight, const Tensor & bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fbgemm_linear_fp16_weight_fp32_activation");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "packed_weight", packed_weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::fbgemm_linear_fp16_weight_fp32_activation(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, packed_weight, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor fbgemm_linear_int8_weight_fp32_activation(c10::DispatchKeySet ks, const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, const Scalar & weight_scale, const Scalar & weight_zero_point, const Tensor & bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fbgemm_linear_int8_weight_fp32_activation");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "packed", packed);
    jit::tracer::addInputs(node, "col_offsets", col_offsets);
    jit::tracer::addInputs(node, "weight_scale", weight_scale);
    jit::tracer::addInputs(node, "weight_zero_point", weight_zero_point);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::fbgemm_linear_int8_weight_fp32_activation(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,double,int64_t> fbgemm_linear_quantize_weight(c10::DispatchKeySet ks, const Tensor & input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fbgemm_linear_quantize_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  double result2;
  int64_t result3;
  std::tie(result0, result1, result2, result3) =at::redispatch::fbgemm_linear_quantize_weight(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
Tensor & fix_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fix");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fix_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::fix_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor float_power_Tensor_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::float_power");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::float_power(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor float_power_Scalar(c10::DispatchKeySet ks, const Scalar & self, const Tensor & exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::float_power");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::float_power(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor float_power_Tensor_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::float_power");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::float_power(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & float_power__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::float_power");
    } else {
      op_name = jit::Symbol::fromQualString("aten::float_power_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("float_power_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::float_power_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & float_power__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::float_power");
    } else {
      op_name = jit::Symbol::fromQualString("aten::float_power_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("float_power_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::float_power_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor floor(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::floor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::floor(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & floor_(c10::DispatchKeySet ks, Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::floor");
    } else {
      op_name = jit::Symbol::fromQualString("aten::floor_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::floor_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor fmin(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fmin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::fmin(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & fmod_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fmod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::fmod_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & fmod_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fmod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::fmod_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & frac_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::frac");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("frac_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::frac_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out_output(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples, Tensor & output, Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "random_samples", random_samples);
    jit::tracer::addInputs(node, "output", output);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fractional_max_pool2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::fractional_max_pool2d_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, output_size, random_samples, output, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(output, indices);
}
Tensor & frobenius_norm_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::frobenius_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("frobenius_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::frobenius_norm_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor full_like(c10::DispatchKeySet ks, const Tensor & self, const Scalar & fill_value, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::full_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::full_like(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, fill_value, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor gather_backward(c10::DispatchKeySet ks, const Tensor & grad, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gather_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "sparse_grad", sparse_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::gather_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, self, dim, index, sparse_grad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & gcd_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gcd");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gcd_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::gcd_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor greater_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::greater");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::greater(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor greater_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::greater");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::greater(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & greater__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::greater");
    } else {
      op_name = jit::Symbol::fromQualString("aten::greater_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("greater_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::greater_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & greater__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::greater");
    } else {
      op_name = jit::Symbol::fromQualString("aten::greater_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("greater_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::greater_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & greater_equal_out_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::greater_equal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("greater_equal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::greater_equal_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & greater_equal_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::greater_equal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("greater_equal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::greater_equal_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor group_norm(c10::DispatchKeySet ks, const Tensor & input, int64_t num_groups, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, double eps, bool cudnn_enabled) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::group_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "num_groups", num_groups);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::group_norm(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, num_groups, weight, bias, eps, cudnn_enabled);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor hamming_window(c10::DispatchKeySet ks, int64_t window_length, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::hamming_window(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor hamming_window_periodic(c10::DispatchKeySet ks, int64_t window_length, bool periodic, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::hamming_window(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor hamming_window_periodic_alpha(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::hamming_window(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, alpha, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor hamming_window_periodic_alpha_beta(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, double beta, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::hamming_window(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, alpha, beta, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor hardshrink_backward(c10::DispatchKeySet ks, const Tensor & grad_out, const Tensor & self, const Scalar & lambd) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardshrink_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::hardshrink_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, self, lambd);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & hardtanh_out_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & min_val, const Scalar & max_val, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardtanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardtanh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::hardtanh_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min_val, max_val, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor heaviside(c10::DispatchKeySet ks, const Tensor & self, const Tensor & values) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::heaviside");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "values", values);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::heaviside(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, values);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & heaviside_(c10::DispatchKeySet ks, Tensor & self, const Tensor & values) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::heaviside");
    } else {
      op_name = jit::Symbol::fromQualString("aten::heaviside_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "values", values);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("heaviside_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::heaviside_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, values);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & im2col_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::im2col");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("im2col_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::im2col_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, dilation, padding, stride, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor index_Tensor(c10::DispatchKeySet ks, const Tensor & self, const c10::List<c10::optional<Tensor>> & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::index(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor index_put(c10::DispatchKeySet ks, const Tensor & self, const c10::List<c10::optional<Tensor>> & indices, const Tensor & values, bool accumulate) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_put");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::index_put(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, values, accumulate);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & index_put_(c10::DispatchKeySet ks, Tensor & self, const c10::List<c10::optional<Tensor>> & indices, const Tensor & values, bool accumulate) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_put");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_put_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_put_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::index_put_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, values, accumulate);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor index_select(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, const Tensor & index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::index_select(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor index_select_dimname(c10::DispatchKeySet ks, const Tensor & self, Dimname dim, const Tensor & index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::index_select(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool is_coalesced(c10::DispatchKeySet ks, const Tensor & self) {
  auto result =at::redispatch::is_coalesced(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
bool is_floating_point(c10::DispatchKeySet ks, const Tensor & self) {
  auto result =at::redispatch::__dispatch_is_floating_point(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
bool is_vulkan_available(c10::DispatchKeySet ks) {
  auto result =at::redispatch::is_vulkan_available(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer));
  return result;
}
Tensor isneginf(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::isneginf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::isneginf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Scalar item(c10::DispatchKeySet ks, const Tensor & self) {
  auto result =at::redispatch::item(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
Tensor & kron_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::kron");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("kron_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::kron_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor l1_loss(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::l1_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::l1_loss(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & l1_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::l1_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("l1_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::l1_loss_backward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, reduction, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor lcm(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lcm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::lcm(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & lcm_(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lcm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lcm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lcm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::lcm_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor ldexp_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ldexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::ldexp(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & ldexp_(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::ldexp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::ldexp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ldexp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::ldexp_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & linalg_cholesky_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linalg_cholesky");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_cholesky_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::linalg_cholesky_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor linalg_cond(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linalg_cond");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::linalg_cond(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor linalg_cond_p_str(c10::DispatchKeySet ks, const Tensor & self, std::string p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linalg_cond");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::linalg_cond(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & linalg_eigvalsh_out_out(c10::DispatchKeySet ks, const Tensor & self, std::string UPLO, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linalg_eigvalsh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "UPLO", UPLO);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_eigvalsh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::linalg_eigvalsh_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, UPLO, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor linalg_householder_product(c10::DispatchKeySet ks, const Tensor & input, const Tensor & tau) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linalg_householder_product");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "tau", tau);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::linalg_householder_product(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, tau);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor linalg_pinv(c10::DispatchKeySet ks, const Tensor & self, double rcond, bool hermitian) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linalg_pinv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "rcond", rcond);
    jit::tracer::addInputs(node, "hermitian", hermitian);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::linalg_pinv(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, rcond, hermitian);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor linalg_pinv_rcond_tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & rcond, bool hermitian) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linalg_pinv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "rcond", rcond);
    jit::tracer::addInputs(node, "hermitian", hermitian);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::linalg_pinv(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, rcond, hermitian);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor linalg_tensorinv(c10::DispatchKeySet ks, const Tensor & self, int64_t ind) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linalg_tensorinv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "ind", ind);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::linalg_tensorinv(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, ind);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor linalg_tensorsolve(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, c10::optional<IntArrayRef> dims) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linalg_tensorsolve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::linalg_tensorsolve(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, dims);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & linspace_out_out(c10::DispatchKeySet ks, const Scalar & start, const Scalar & end, c10::optional<int64_t> steps, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linspace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::linspace_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & log2_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::log2_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & log_normal_(c10::DispatchKeySet ks, Tensor & self, double mean, double std, c10::optional<Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::log_normal");
    } else {
      op_name = jit::Symbol::fromQualString("aten::log_normal_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_normal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::log_normal_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & log_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::log_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor log_sigmoid_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "buffer", buffer);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::log_sigmoid_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, buffer);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, Tensor & output, Tensor & buffer) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output", output);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_sigmoid_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::log_sigmoid_forward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output, buffer);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, buffer);
  }
  return std::forward_as_tuple(output, buffer);
}
Tensor & logaddexp2_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logaddexp2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logaddexp2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::logaddexp2_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & logaddexp_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logaddexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logaddexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::logaddexp_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor logcumsumexp(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logcumsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::logcumsumexp(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor logcumsumexp_dimname(c10::DispatchKeySet ks, const Tensor & self, Dimname dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logcumsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::logcumsumexp(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor logical_or(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logical_or");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::logical_or(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & logical_or_(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::logical_or");
    } else {
      op_name = jit::Symbol::fromQualString("aten::logical_or_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_or_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::logical_or_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor logical_xor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logical_xor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::logical_xor(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & logical_xor_(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::logical_xor");
    } else {
      op_name = jit::Symbol::fromQualString("aten::logical_xor_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_xor_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::logical_xor_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor logspace(c10::DispatchKeySet ks, const Scalar & start, const Scalar & end, c10::optional<int64_t> steps, double base, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logspace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::logspace(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, base, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor logsumexp(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::logsumexp(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor logsumexp_names(c10::DispatchKeySet ks, const Tensor & self, DimnameList dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::logsumexp(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> lstsq_out_X(c10::DispatchKeySet ks, const Tensor & self, const Tensor & A, Tensor & X, Tensor & qr) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lstsq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "X", X);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "X", X);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lstsq_out", X);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::lstsq_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, A, X, qr);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, X);
    jit::tracer::addOutput(node, qr);
  }
  return std::forward_as_tuple(X, qr);
}
Tensor matmul(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::matmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::matmul(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> max_dim(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor values;
  Tensor indices;
  std::tie(values, indices) =at::redispatch::max(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> max_names_dim(c10::DispatchKeySet ks, const Tensor & self, Dimname dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor values;
  Tensor indices;
  std::tie(values, indices) =at::redispatch::max(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor max(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::max(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor max_other(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::max(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> max_pool1d_with_indices(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool1d_with_indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  std::tie(result0, result1) =at::redispatch::max_pool1d_with_indices(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, Tensor & out, Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool2d_with_indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "out", out);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool2d_with_indices_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::max_pool2d_with_indices_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(out, indices);
}
Tensor max_unpool2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::max_unpool2d_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, indices, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor max_unpool3d(c10::DispatchKeySet ks, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::max_unpool3d(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, output_size, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & max_unpool3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding, Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::max_unpool3d_backward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, indices, output_size, stride, padding, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor maximum(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::maximum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::maximum(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & mean_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mean_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::mean_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & mean_out_names_out(c10::DispatchKeySet ks, const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mean_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::mean_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::tuple<Tensor &,Tensor &> median_out_dim_values(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::median");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "values", values);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("median_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::median_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> median_out_names_dim_values(c10::DispatchKeySet ks, const Tensor & self, Dimname dim, bool keepdim, Tensor & values, Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::median");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "values", values);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("median_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::median_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
std::vector<Tensor> meshgrid(c10::DispatchKeySet ks, TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::meshgrid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::meshgrid(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm(c10::DispatchKeySet ks, const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "exponential_average_factor", exponential_average_factor);
    jit::tracer::addInputs(node, "epsilon", epsilon);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  std::tie(result0, result1, result2) =at::redispatch::miopen_batch_norm(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor,Tensor> miopen_convolution_transpose_backward(c10::DispatchKeySet ks, const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_transpose_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  std::tie(result0, result1, result2) =at::redispatch::miopen_convolution_transpose_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor miopen_convolution_transpose_backward_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_transpose_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::miopen_convolution_transpose_backward_input(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor miopen_depthwise_convolution_backward_weight(c10::DispatchKeySet ks, IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_depthwise_convolution_backward_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_size", weight_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::miopen_depthwise_convolution_backward_weight(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor mkldnn_adaptive_avg_pool2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_adaptive_avg_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::mkldnn_adaptive_avg_pool2d_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> mkldnn_convolution_backward(c10::DispatchKeySet ks, const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_convolution_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  std::tie(result0, result1, result2) =at::redispatch::mkldnn_convolution_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grad_output, weight, padding, stride, dilation, groups, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor mkldnn_convolution_backward_input(c10::DispatchKeySet ks, IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_convolution_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self_size", self_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "bias_defined", bias_defined);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::mkldnn_convolution_backward_input(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor mkldnn_reorder_conv3d_weight(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_reorder_conv3d_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::mkldnn_reorder_conv3d_weight(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, stride, dilation, groups);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> mode(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor values;
  Tensor indices;
  std::tie(values, indices) =at::redispatch::mode(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> mode_dimname(c10::DispatchKeySet ks, const Tensor & self, Dimname dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor values;
  Tensor indices;
  std::tie(values, indices) =at::redispatch::mode(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor moveaxis_intlist(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef source, IntArrayRef destination) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::moveaxis");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "destination", destination);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::moveaxis(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, destination);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor moveaxis_int(c10::DispatchKeySet ks, const Tensor & self, int64_t source, int64_t destination) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::moveaxis");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "destination", destination);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::moveaxis(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, destination);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & multi_margin_loss_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const Scalar & p, const Scalar & margin, const c10::optional<Tensor> & weight, int64_t reduction, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multi_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multi_margin_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::multi_margin_loss_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, p, margin, weight, reduction, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::tuple<Tensor,Tensor> multilabel_margin_loss_forward(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor output;
  Tensor is_target;
  std::tie(output, is_target) =at::redispatch::multilabel_margin_loss_forward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, is_target);
  }
  return std::make_tuple(std::move(output), std::move(is_target));
}
Tensor multiply_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multiply");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::multiply(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor multiply_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multiply");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::multiply(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & multiply__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::multiply");
    } else {
      op_name = jit::Symbol::fromQualString("aten::multiply_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multiply_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::multiply_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & multiply__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::multiply");
    } else {
      op_name = jit::Symbol::fromQualString("aten::multiply_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multiply_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::multiply_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & mv_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & vec, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec", vec);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::mv_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::tuple<Tensor,Tensor,Tensor> native_batch_norm_backward(c10::DispatchKeySet ks, const Tensor & grad_out, const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, const c10::optional<Tensor> & save_mean, const c10::optional<Tensor> & save_invstd, bool train, double eps, std::array<bool,3> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_batch_norm_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_invstd", save_invstd);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  std::tie(result0, result1, result2) =at::redispatch::native_batch_norm_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor native_norm(c10::DispatchKeySet ks, const Tensor & self, const Scalar & p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::native_norm(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor native_norm_ScalarOpt_dim_dtype(c10::DispatchKeySet ks, const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::native_norm(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor ne_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ne");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::ne(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor ne_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ne");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::ne(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & ne__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::ne");
    } else {
      op_name = jit::Symbol::fromQualString("aten::ne_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::ne_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & ne__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::ne");
    } else {
      op_name = jit::Symbol::fromQualString("aten::ne_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::ne_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor negative(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::negative");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::negative(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & negative_(c10::DispatchKeySet ks, Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::negative");
    } else {
      op_name = jit::Symbol::fromQualString("aten::negative_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("negative_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::negative_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & nextafter_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nextafter");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nextafter_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::nextafter_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor nll_loss2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::nll_loss2d_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, weight, reduction, ignore_index, total_weight);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> nll_loss2d_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "output", output);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::nll_loss2d_forward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, reduction, ignore_index, output, total_weight);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  return std::forward_as_tuple(output, total_weight);
}
Tensor nll_loss_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::nll_loss_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, weight, reduction, ignore_index, total_weight);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> nll_loss_forward_out_output(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "output", output);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::nll_loss_forward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, reduction, ignore_index, output, total_weight);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  return std::forward_as_tuple(output, total_weight);
}
Tensor & ones_out_out(c10::DispatchKeySet ks, IntArrayRef size, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ones_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::ones_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor ormqr(c10::DispatchKeySet ks, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ormqr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "input3", input3);
    jit::tracer::addInputs(node, "left", left);
    jit::tracer::addInputs(node, "transpose", transpose);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::ormqr(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, input2, input3, left, transpose);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor pairwise_distance(c10::DispatchKeySet ks, const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pairwise_distance");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::pairwise_distance(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x1, x2, p, eps, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor pinverse(c10::DispatchKeySet ks, const Tensor & self, double rcond) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pinverse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "rcond", rcond);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::pinverse(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, rcond);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & polar_out_out(c10::DispatchKeySet ks, const Tensor & abs, const Tensor & angle, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::polar");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "abs", abs);
    jit::tracer::addInputs(node, "angle", angle);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("polar_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::polar_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), abs, angle, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & polygamma_out_out(c10::DispatchKeySet ks, int64_t n, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::polygamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("polygamma_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::polygamma_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & pow_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & exponent, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pow_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::pow_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & pow_out_Scalar_out(c10::DispatchKeySet ks, const Scalar & self, const Tensor & exponent, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pow_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::pow_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & pow_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const Tensor & self, const Scalar & exponent, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pow_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::pow_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & prod_out_int_out(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("prod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::prod_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & prod_out_Dimname_out(c10::DispatchKeySet ks, const Tensor & self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("prod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::prod_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
int64_t q_per_channel_axis(c10::DispatchKeySet ks, const Tensor & self) {
  auto result =at::redispatch::q_per_channel_axis(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
Tensor q_per_channel_zero_points(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::q_per_channel_zero_points");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::q_per_channel_zero_points(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> qr_out_Q(c10::DispatchKeySet ks, const Tensor & self, bool some, Tensor & Q, Tensor & R) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::qr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    jit::tracer::addInputs(node, "Q", Q);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "Q", Q);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("qr_out", Q);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::qr_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, some, Q, R);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, Q);
    jit::tracer::addOutput(node, R);
  }
  return std::forward_as_tuple(Q, R);
}
Tensor & quantile_out_scalar_out(c10::DispatchKeySet ks, const Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantile");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("quantile_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::quantile_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & quantile_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & q, c10::optional<int64_t> dim, bool keepdim, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantile");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("quantile_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::quantile_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor quantized_gru_cell(c10::DispatchKeySet ks, const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, const Scalar & scale_ih, const Scalar & scale_hh, const Scalar & zero_point_ih, const Scalar & zero_point_hh) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_gru_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "w_ih", w_ih);
    jit::tracer::addInputs(node, "w_hh", w_hh);
    jit::tracer::addInputs(node, "b_ih", b_ih);
    jit::tracer::addInputs(node, "b_hh", b_hh);
    jit::tracer::addInputs(node, "packed_ih", packed_ih);
    jit::tracer::addInputs(node, "packed_hh", packed_hh);
    jit::tracer::addInputs(node, "col_offsets_ih", col_offsets_ih);
    jit::tracer::addInputs(node, "col_offsets_hh", col_offsets_hh);
    jit::tracer::addInputs(node, "scale_ih", scale_ih);
    jit::tracer::addInputs(node, "scale_hh", scale_hh);
    jit::tracer::addInputs(node, "zero_point_ih", zero_point_ih);
    jit::tracer::addInputs(node, "zero_point_hh", zero_point_hh);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::quantized_gru_cell(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor quantized_rnn_relu_cell(c10::DispatchKeySet ks, const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, const Scalar & scale_ih, const Scalar & scale_hh, const Scalar & zero_point_ih, const Scalar & zero_point_hh) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_rnn_relu_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "w_ih", w_ih);
    jit::tracer::addInputs(node, "w_hh", w_hh);
    jit::tracer::addInputs(node, "b_ih", b_ih);
    jit::tracer::addInputs(node, "b_hh", b_hh);
    jit::tracer::addInputs(node, "packed_ih", packed_ih);
    jit::tracer::addInputs(node, "packed_hh", packed_hh);
    jit::tracer::addInputs(node, "col_offsets_ih", col_offsets_ih);
    jit::tracer::addInputs(node, "col_offsets_hh", col_offsets_hh);
    jit::tracer::addInputs(node, "scale_ih", scale_ih);
    jit::tracer::addInputs(node, "scale_hh", scale_hh);
    jit::tracer::addInputs(node, "zero_point_ih", zero_point_ih);
    jit::tracer::addInputs(node, "zero_point_hh", zero_point_hh);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::quantized_rnn_relu_cell(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor rand_names(c10::DispatchKeySet ks, IntArrayRef size, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::rand(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, names, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor rand_generator_with_names(c10::DispatchKeySet ks, IntArrayRef size, c10::optional<Generator> generator, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::rand(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, names, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor rand(c10::DispatchKeySet ks, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::rand(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor rand_generator(c10::DispatchKeySet ks, IntArrayRef size, c10::optional<Generator> generator, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::rand(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & randint_out_out(c10::DispatchKeySet ks, int64_t high, IntArrayRef size, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::randint_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), high, size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & randint_out_generator_out(c10::DispatchKeySet ks, int64_t high, IntArrayRef size, c10::optional<Generator> generator, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::randint_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), high, size, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & randint_out_low_out(c10::DispatchKeySet ks, int64_t low, int64_t high, IntArrayRef size, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::randint_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), low, high, size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & randint_out_low_generator_out(c10::DispatchKeySet ks, int64_t low, int64_t high, IntArrayRef size, c10::optional<Generator> generator, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::randint_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), low, high, size, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor randn(c10::DispatchKeySet ks, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::randn(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor randn_generator(c10::DispatchKeySet ks, IntArrayRef size, c10::optional<Generator> generator, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::randn(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor randn_names(c10::DispatchKeySet ks, IntArrayRef size, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::randn(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, names, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor randn_generator_with_names(c10::DispatchKeySet ks, IntArrayRef size, c10::optional<Generator> generator, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::randn(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, names, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & random__from(c10::DispatchKeySet ks, Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::random");
    } else {
      op_name = jit::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::random_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, from, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & random__to(c10::DispatchKeySet ks, Tensor & self, int64_t to, c10::optional<Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::random");
    } else {
      op_name = jit::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::random_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & random_(c10::DispatchKeySet ks, Tensor & self, c10::optional<Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::random");
    } else {
      op_name = jit::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::random_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & randperm_out_out(c10::DispatchKeySet ks, int64_t n, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randperm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randperm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::randperm_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & randperm_out_generator_out(c10::DispatchKeySet ks, int64_t n, c10::optional<Generator> generator, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randperm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randperm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::randperm_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void record_stream(c10::DispatchKeySet ks, Tensor & self, Stream s) {
  at::redispatch::record_stream(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s);
}
Tensor reflection_pad1d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::reflection_pad1d_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor reflection_pad2d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::reflection_pad2d(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & reflection_pad2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::reflection_pad2d_backward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, padding, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor remainder_Scalar(c10::DispatchKeySet ks, const Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::remainder");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::remainder(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor remainder_Tensor(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::remainder");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::remainder(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & remainder__Scalar(c10::DispatchKeySet ks, Tensor & self, const Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::remainder");
    } else {
      op_name = jit::Symbol::fromQualString("aten::remainder_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::remainder_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & remainder__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::remainder");
    } else {
      op_name = jit::Symbol::fromQualString("aten::remainder_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::remainder_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor repeat(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef repeats) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::repeat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "repeats", repeats);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::repeat(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, repeats);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor replication_pad1d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::replication_pad1d(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & replication_pad1d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::replication_pad1d_backward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, padding, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & replication_pad2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef padding, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::replication_pad2d_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor reshape_as(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reshape_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::reshape_as(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
ScalarType result_type_Tensor(c10::DispatchKeySet ks, const Tensor & tensor, const Tensor & other) {
  auto result =at::redispatch::result_type(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensor, other);
  return result;
}
ScalarType result_type_Scalar(c10::DispatchKeySet ks, const Tensor & tensor, const Scalar & other) {
  auto result =at::redispatch::result_type(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensor, other);
  return result;
}
ScalarType result_type_Scalar_Tensor(c10::DispatchKeySet ks, const Scalar & scalar, const Tensor & tensor) {
  auto result =at::redispatch::result_type(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), scalar, tensor);
  return result;
}
ScalarType result_type_Scalar_Scalar(c10::DispatchKeySet ks, const Scalar & scalar1, const Scalar & scalar2) {
  auto result =at::redispatch::result_type(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), scalar1, scalar2);
  return result;
}
std::tuple<Tensor,Tensor> rnn_tanh_input(c10::DispatchKeySet ks, const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rnn_tanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  std::tie(result0, result1) =at::redispatch::rnn_tanh(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> rnn_tanh_data(c10::DispatchKeySet ks, const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rnn_tanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  std::tie(result0, result1) =at::redispatch::rnn_tanh(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor roll(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef shifts, IntArrayRef dims) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::roll");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "shifts", shifts);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::roll(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, shifts, dims);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor rot90(c10::DispatchKeySet ks, const Tensor & self, int64_t k, IntArrayRef dims) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rot90");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::rot90(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dims);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & round_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::round");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::round_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor rrelu_with_noise_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & noise, const Scalar & lower, const Scalar & upper, bool training, bool self_is_result) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "self_is_result", self_is_result);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::rrelu_with_noise_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, noise, lower, upper, training, self_is_result);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & rsqrt_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rsqrt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rsqrt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::rsqrt_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & searchsorted_out_Tensor_out(c10::DispatchKeySet ks, const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::searchsorted");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sorted_sequence", sorted_sequence);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("searchsorted_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::searchsorted_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), sorted_sequence, self, out_int32, right, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor select_backward(c10::DispatchKeySet ks, const Tensor & grad, IntArrayRef input_sizes, int64_t dim, int64_t index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::select_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input_sizes", input_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::select_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, input_sizes, dim, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor selu(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::selu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::selu(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & selu_(c10::DispatchKeySet ks, Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::selu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::selu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("selu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::selu_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & signbit_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::signbit");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("signbit_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::signbit_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor silu_backward(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::silu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::silu_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & slow_conv3d_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("slow_conv3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::slow_conv3d_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & slow_conv_transpose2d_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv_transpose2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("slow_conv_transpose2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::slow_conv_transpose2d_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & smooth_l1_loss_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction, double beta, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::smooth_l1_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "beta", beta);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("smooth_l1_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::smooth_l1_loss_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction, beta, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor soft_margin_loss(c10::DispatchKeySet ks, const Tensor & self, const Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::soft_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::soft_margin_loss(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & soft_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::soft_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("soft_margin_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::soft_margin_loss_backward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, reduction, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor softplus(c10::DispatchKeySet ks, const Tensor & self, const Scalar & beta, const Scalar & threshold) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softplus");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::softplus(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, beta, threshold);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & softplus_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Scalar & beta, const Scalar & threshold, const Tensor & output, Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softplus_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    jit::tracer::addInputs(node, "output", output);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softplus_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::softplus_backward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, beta, threshold, output, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor,Tensor> sort(c10::DispatchKeySet ks, const Tensor & self, int64_t dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor values;
  Tensor indices;
  std::tie(values, indices) =at::redispatch::sort(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> sort_stable(c10::DispatchKeySet ks, const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stable", stable);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor values;
  Tensor indices;
  std::tie(values, indices) =at::redispatch::sort(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, stable, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> sort_dimname(c10::DispatchKeySet ks, const Tensor & self, Dimname dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor values;
  Tensor indices;
  std::tie(values, indices) =at::redispatch::sort(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> sort_dimname_stable(c10::DispatchKeySet ks, const Tensor & self, c10::optional<bool> stable, Dimname dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stable", stable);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor values;
  Tensor indices;
  std::tie(values, indices) =at::redispatch::sort(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, stable, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor & special_erfinv_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::special_erfinv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_erfinv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::special_erfinv_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::vector<Tensor> split_Tensor(c10::DispatchKeySet ks, const Tensor & self, int64_t split_size, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::split");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "split_size", split_size);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::split(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, split_size, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & square_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::square");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("square_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::square_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & sspaddmm_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sspaddmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sspaddmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::sspaddmm_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat1, mat2, beta, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor std(c10::DispatchKeySet ks, const Tensor & self, bool unbiased) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::std(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, unbiased);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor std_dim(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::std(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, unbiased, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor std_names_dim(c10::DispatchKeySet ks, const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::std(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, unbiased, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t stride_int(c10::DispatchKeySet ks, const Tensor & self, int64_t dim) {
  auto result =at::redispatch::__dispatch_stride(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  return result;
}
int64_t stride_Dimname(c10::DispatchKeySet ks, const Tensor & self, Dimname dim) {
  auto result =at::redispatch::stride(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  return result;
}
Tensor & subtract_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::subtract");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("subtract_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::subtract_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor sum(c10::DispatchKeySet ks, const Tensor & self, c10::optional<ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::sum(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor sum_dim_IntList(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::sum(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor sum_dim_DimnameList(c10::DispatchKeySet ks, const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::sum(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor sum_to_size(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum_to_size");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::sum_to_size(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor swapaxes(c10::DispatchKeySet ks, const Tensor & self, int64_t axis0, int64_t axis1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::swapaxes");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "axis0", axis0);
    jit::tracer::addInputs(node, "axis1", axis1);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::swapaxes(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, axis0, axis1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & swapaxes_(c10::DispatchKeySet ks, Tensor & self, int64_t axis0, int64_t axis1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::swapaxes");
    } else {
      op_name = jit::Symbol::fromQualString("aten::swapaxes_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "axis0", axis0);
    jit::tracer::addInputs(node, "axis1", axis1);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("swapaxes_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::swapaxes_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, axis0, axis1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor t(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::t");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::t(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & t_(c10::DispatchKeySet ks, Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::t");
    } else {
      op_name = jit::Symbol::fromQualString("aten::t_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("t_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::t_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor take(c10::DispatchKeySet ks, const Tensor & self, const Tensor & index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::take");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::take(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & tanh_out_out(c10::DispatchKeySet ks, const Tensor & self, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tanh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::tanh_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor thnn_conv_depthwise2d(c10::DispatchKeySet ks, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_depthwise2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::thnn_conv_depthwise2d(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor & grad_input, Tensor & grad_weight) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_depthwise2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "grad_input", grad_input);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_depthwise2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::thnn_conv_depthwise2d_backward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, weight, kernel_size, stride, padding, dilation, grad_input, grad_weight);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
  }
  return std::forward_as_tuple(grad_input, grad_weight);
}
Tensor to_mkldnn(c10::DispatchKeySet ks, const Tensor & self, c10::optional<ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_mkldnn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::to_mkldnn(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor to_sparse_sparse_dim(c10::DispatchKeySet ks, const Tensor & self, int64_t sparse_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_sparse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::to_sparse(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, sparse_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor to_sparse(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_sparse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::to_sparse(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> topk(c10::DispatchKeySet ks, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::topk");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "largest", largest);
    jit::tracer::addInputs(node, "sorted", sorted);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor values;
  Tensor indices;
  std::tie(values, indices) =at::redispatch::topk(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dim, largest, sorted);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor transpose_int(c10::DispatchKeySet ks, const Tensor & self, int64_t dim0, int64_t dim1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::transpose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::transpose(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim0, dim1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor transpose_Dimname(c10::DispatchKeySet ks, const Tensor & self, Dimname dim0, Dimname dim1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::transpose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::transpose(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim0, dim1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & transpose_(c10::DispatchKeySet ks, Tensor & self, int64_t dim0, int64_t dim1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::transpose");
    } else {
      op_name = jit::Symbol::fromQualString("aten::transpose_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("transpose_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::transpose_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim0, dim1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor trapz_x(c10::DispatchKeySet ks, const Tensor & y, const Tensor & x, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::trapz");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "y", y);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::trapz(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), y, x, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor trapz_dx(c10::DispatchKeySet ks, const Tensor & y, double dx, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::trapz");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "y", y);
    jit::tracer::addInputs(node, "dx", dx);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::trapz(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), y, dx, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & triu_out_out(c10::DispatchKeySet ks, const Tensor & self, int64_t diagonal, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::triu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("triu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::triu_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, diagonal, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & true_divide_out_out(c10::DispatchKeySet ks, const Tensor & self, const Tensor & other, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::true_divide");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("true_divide_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::true_divide_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor trunc(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::trunc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::trunc(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & trunc_(c10::DispatchKeySet ks, Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::trunc");
    } else {
      op_name = jit::Symbol::fromQualString("aten::trunc_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("trunc_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::trunc_(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & upsample_bicubic2d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bicubic2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_bicubic2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::upsample_bicubic2d_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, align_corners, scales_h, scales_w, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor upsample_linear1d_backward_vec(c10::DispatchKeySet ks, const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_linear1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scale_factors", scale_factors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::upsample_linear1d_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, align_corners, scale_factors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor upsample_linear1d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_linear1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::upsample_linear1d_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, align_corners, scales);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor upsample_nearest2d_backward_vec(c10::DispatchKeySet ks, const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, c10::optional<ArrayRef<double>> scale_factors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scale_factors", scale_factors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::upsample_nearest2d_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, scale_factors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor upsample_nearest2d_backward(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::upsample_nearest2d_backward(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor upsample_nearest3d_vec(c10::DispatchKeySet ks, const Tensor & input, c10::optional<IntArrayRef> output_size, c10::optional<ArrayRef<double>> scale_factors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scale_factors", scale_factors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::upsample_nearest3d(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, output_size, scale_factors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor upsample_nearest3d(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales_d", scales_d);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::upsample_nearest3d(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, scales_d, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & upsample_nearest3d_backward_out_grad_input(c10::DispatchKeySet ks, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales_d", scales_d);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::upsample_nearest3d_backward_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & upsample_trilinear3d_out_out(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_trilinear3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_d", scales_d);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_trilinear3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::upsample_trilinear3d_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, align_corners, scales_d, scales_h, scales_w, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor values(c10::DispatchKeySet ks, const Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::values");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::redispatch::values(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> var_mean(c10::DispatchKeySet ks, const Tensor & self, bool unbiased) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  std::tie(result0, result1) =at::redispatch::var_mean(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, unbiased);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> var_mean_dim(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  std::tie(result0, result1) =at::redispatch::var_mean(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, unbiased, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> var_mean_names_dim(c10::DispatchKeySet ks, const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  Tensor result0;
  Tensor result1;
  std::tie(result0, result1) =at::redispatch::var_mean(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, unbiased, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & vstack_out_out(c10::DispatchKeySet ks, TensorList tensors, Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::vstack");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("vstack_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::redispatch::vstack_outf(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
}  // namespace
}  // namespace TraceType

namespace {

TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  m.impl("__irshift__.Scalar",
         TORCH_FN(TraceType::__irshift___Scalar)
  );
  m.impl("__irshift__.Tensor",
         TORCH_FN(TraceType::__irshift___Tensor)
  );
  m.impl("__rshift__.Scalar",
         TORCH_FN(TraceType::__rshift___Scalar)
  );
  m.impl("__rshift__.Tensor",
         TORCH_FN(TraceType::__rshift___Tensor)
  );
  m.impl("_adaptive_avg_pool2d",
         TORCH_FN(TraceType::_adaptive_avg_pool2d)
  );
  m.impl("_add_relu.out",
         TORCH_FN(TraceType::_add_relu_out_out)
  );
  m.impl("_amp_update_scale",
         TORCH_FN(TraceType::_amp_update_scale)
  );
  m.impl("_bmm",
         TORCH_FN(TraceType::_bmm)
  );
  m.impl("_cast_Byte",
         TORCH_FN(TraceType::_cast_Byte)
  );
  m.impl("_cast_Half",
         TORCH_FN(TraceType::_cast_Half)
  );
  m.impl("_cast_Int",
         TORCH_FN(TraceType::_cast_Int)
  );
  m.impl("_cat.out",
         TORCH_FN(TraceType::_cat_out_out)
  );
  m.impl("_cdist_backward",
         TORCH_FN(TraceType::_cdist_backward)
  );
  m.impl("_cholesky_helper",
         TORCH_FN(TraceType::_cholesky_helper)
  );
  m.impl("_coalesced_",
         TORCH_FN(TraceType::_coalesced_)
  );
  m.impl("_compute_linear_combination",
         TORCH_FN(TraceType::_compute_linear_combination)
  );
  m.impl("_conj",
         TORCH_FN(TraceType::_conj)
  );
  m.impl("_convolution",
         TORCH_FN(TraceType::_convolution)
  );
  m.impl("_convolution.deprecated",
         TORCH_FN(TraceType::_convolution_deprecated)
  );
  m.impl("_cudnn_rnn_backward",
         TORCH_FN(TraceType::_cudnn_rnn_backward)
  );
  m.impl("_cufft_get_plan_cache_size",
         TORCH_FN(TraceType::_cufft_get_plan_cache_size)
  );
  m.impl("_cummax_helper",
         TORCH_FN(TraceType::_cummax_helper)
  );
  m.impl("_cumprod.out",
         TORCH_FN(TraceType::_cumprod_out_out)
  );
  m.impl("_cumsum",
         TORCH_FN(TraceType::_cumsum)
  );
  m.impl("_dim_arange",
         TORCH_FN(TraceType::_dim_arange)
  );
  m.impl("_dirichlet_grad",
         TORCH_FN(TraceType::_dirichlet_grad)
  );
  m.impl("_embedding_bag_backward",
         TORCH_FN(TraceType::_embedding_bag_backward)
  );
  m.impl("_fake_quantize_learnable_per_channel_affine_backward",
         TORCH_FN(TraceType::_fake_quantize_learnable_per_channel_affine_backward)
  );
  m.impl("_foreach_ceil",
         TORCH_FN(TraceType::_foreach_ceil)
  );
  m.impl("_foreach_ceil_",
         TORCH_FN(TraceType::_foreach_ceil_)
  );
  m.impl("_foreach_div.Scalar",
         TORCH_FN(TraceType::_foreach_div_Scalar)
  );
  m.impl("_foreach_div.List",
         TORCH_FN(TraceType::_foreach_div_List)
  );
  m.impl("_foreach_div.ScalarList",
         TORCH_FN(TraceType::_foreach_div_ScalarList)
  );
  m.impl("_foreach_div_.Scalar",
         TORCH_FN(TraceType::_foreach_div__Scalar)
  );
  m.impl("_foreach_div_.List",
         TORCH_FN(TraceType::_foreach_div__List)
  );
  m.impl("_foreach_div_.ScalarList",
         TORCH_FN(TraceType::_foreach_div__ScalarList)
  );
  m.impl("_foreach_exp",
         TORCH_FN(TraceType::_foreach_exp)
  );
  m.impl("_foreach_exp_",
         TORCH_FN(TraceType::_foreach_exp_)
  );
  m.impl("_foreach_lgamma",
         TORCH_FN(TraceType::_foreach_lgamma)
  );
  m.impl("_foreach_lgamma_",
         TORCH_FN(TraceType::_foreach_lgamma_)
  );
  m.impl("_foreach_log1p",
         TORCH_FN(TraceType::_foreach_log1p)
  );
  m.impl("_foreach_log1p_",
         TORCH_FN(TraceType::_foreach_log1p_)
  );
  m.impl("_foreach_sigmoid",
         TORCH_FN(TraceType::_foreach_sigmoid)
  );
  m.impl("_foreach_sigmoid_",
         TORCH_FN(TraceType::_foreach_sigmoid_)
  );
  m.impl("_foreach_sqrt",
         TORCH_FN(TraceType::_foreach_sqrt)
  );
  m.impl("_foreach_sqrt_",
         TORCH_FN(TraceType::_foreach_sqrt_)
  );
  m.impl("_foreach_tan",
         TORCH_FN(TraceType::_foreach_tan)
  );
  m.impl("_foreach_tan_",
         TORCH_FN(TraceType::_foreach_tan_)
  );
  m.impl("_foreach_zero_",
         TORCH_FN(TraceType::_foreach_zero_)
  );
  m.impl("_fused_dropout",
         TORCH_FN(TraceType::_fused_dropout)
  );
  m.impl("_logcumsumexp",
         TORCH_FN(TraceType::_logcumsumexp)
  );
  m.impl("_make_dual",
         TORCH_FN(TraceType::_make_dual)
  );
  m.impl("_make_per_tensor_quantized_tensor",
         TORCH_FN(TraceType::_make_per_tensor_quantized_tensor)
  );
  m.impl("_mode",
         TORCH_FN(TraceType::_mode)
  );
  m.impl("_nnpack_spatial_convolution",
         TORCH_FN(TraceType::_nnpack_spatial_convolution)
  );
  m.impl("_softmax_backward_data",
         TORCH_FN(TraceType::_softmax_backward_data)
  );
  m.impl("_sparse_log_softmax.int",
         TORCH_FN(TraceType::_sparse_log_softmax_int)
  );
  m.impl("_sparse_log_softmax.Dimname",
         TORCH_FN(TraceType::_sparse_log_softmax_Dimname)
  );
  m.impl("_sparse_log_softmax",
         TORCH_FN(TraceType::_sparse_log_softmax)
  );
  m.impl("_sparse_sum_backward",
         TORCH_FN(TraceType::_sparse_sum_backward)
  );
  m.impl("_std",
         TORCH_FN(TraceType::_std)
  );
  m.impl("_test_ambiguous_defaults.a",
         TORCH_FN(TraceType::_test_ambiguous_defaults_a)
  );
  m.impl("_test_ambiguous_defaults.b",
         TORCH_FN(TraceType::_test_ambiguous_defaults_b)
  );
  m.impl("_test_optional_floatlist",
         TORCH_FN(TraceType::_test_optional_floatlist)
  );
  m.impl("_thnn_differentiable_gru_cell_backward",
         TORCH_FN(TraceType::_thnn_differentiable_gru_cell_backward)
  );
  m.impl("_thnn_fused_gru_cell_backward",
         TORCH_FN(TraceType::_thnn_fused_gru_cell_backward)
  );
  m.impl("_use_cudnn_rnn_flatten_weight",
         TORCH_FN(TraceType::_use_cudnn_rnn_flatten_weight)
  );
  m.impl("_values",
         TORCH_FN(TraceType::_values)
  );
  m.impl("_weight_norm_cuda_interface_backward",
         TORCH_FN(TraceType::_weight_norm_cuda_interface_backward)
  );
  m.impl("acos.out",
         TORCH_FN(TraceType::acos_out_out)
  );
  m.impl("acosh",
         TORCH_FN(TraceType::acosh)
  );
  m.impl("acosh_",
         TORCH_FN(TraceType::acosh_)
  );
  m.impl("adaptive_avg_pool2d",
         TORCH_FN(TraceType::adaptive_avg_pool2d)
  );
  m.impl("adaptive_avg_pool3d.out",
         TORCH_FN(TraceType::adaptive_avg_pool3d_out_out)
  );
  m.impl("adaptive_max_pool3d_backward",
         TORCH_FN(TraceType::adaptive_max_pool3d_backward)
  );
  m.impl("add.out",
         TORCH_FN(TraceType::add_out_out)
  );
  m.impl("addr",
         TORCH_FN(TraceType::addr)
  );
  m.impl("addr_",
         TORCH_FN(TraceType::addr_)
  );
  m.impl("affine_grid_generator",
         TORCH_FN(TraceType::affine_grid_generator)
  );
  m.impl("amin",
         TORCH_FN(TraceType::amin)
  );
  m.impl("arange.out",
         TORCH_FN(TraceType::arange_out_out)
  );
  m.impl("arange.start_out",
         TORCH_FN(TraceType::arange_out_start_out)
  );
  m.impl("arctanh.out",
         TORCH_FN(TraceType::arctanh_out_out)
  );
  m.impl("asin.out",
         TORCH_FN(TraceType::asin_out_out)
  );
  m.impl("asinh",
         TORCH_FN(TraceType::asinh)
  );
  m.impl("asinh_",
         TORCH_FN(TraceType::asinh_)
  );
  m.impl("atleast_3d",
         TORCH_FN(TraceType::atleast_3d)
  );
  m.impl("atleast_3d.Sequence",
         TORCH_FN(TraceType::atleast_3d_Sequence)
  );
  m.impl("avg_pool2d_backward",
         TORCH_FN(TraceType::avg_pool2d_backward)
  );
  m.impl("avg_pool3d",
         TORCH_FN(TraceType::avg_pool3d)
  );
  m.impl("avg_pool3d_backward.grad_input",
         TORCH_FN(TraceType::avg_pool3d_backward_out_grad_input)
  );
  m.impl("baddbmm",
         TORCH_FN(TraceType::baddbmm)
  );
  m.impl("baddbmm_",
         TORCH_FN(TraceType::baddbmm_)
  );
  m.impl("batch_norm_backward_reduce",
         TORCH_FN(TraceType::batch_norm_backward_reduce)
  );
  m.impl("bernoulli.out",
         TORCH_FN(TraceType::bernoulli_out_out)
  );
  m.impl("binary_cross_entropy_with_logits",
         TORCH_FN(TraceType::binary_cross_entropy_with_logits)
  );
  m.impl("bincount",
         TORCH_FN(TraceType::bincount)
  );
  m.impl("bitwise_and.Scalar",
         TORCH_FN(TraceType::bitwise_and_Scalar)
  );
  m.impl("bitwise_and.Tensor",
         TORCH_FN(TraceType::bitwise_and_Tensor)
  );
  m.impl("bitwise_and_.Scalar",
         TORCH_FN(TraceType::bitwise_and__Scalar)
  );
  m.impl("bitwise_and_.Tensor",
         TORCH_FN(TraceType::bitwise_and__Tensor)
  );
  m.impl("bitwise_not",
         TORCH_FN(TraceType::bitwise_not)
  );
  m.impl("bitwise_not_",
         TORCH_FN(TraceType::bitwise_not_)
  );
  m.impl("bmm",
         TORCH_FN(TraceType::bmm)
  );
  m.impl("bucketize.Tensor",
         TORCH_FN(TraceType::bucketize_Tensor)
  );
  m.impl("bucketize.Scalar",
         TORCH_FN(TraceType::bucketize_Scalar)
  );
  m.impl("cartesian_prod",
         TORCH_FN(TraceType::cartesian_prod)
  );
  m.impl("cat.out",
         TORCH_FN(TraceType::cat_out_out)
  );
  m.impl("cat.names_out",
         TORCH_FN(TraceType::cat_out_names_out)
  );
  m.impl("chain_matmul",
         TORCH_FN(TraceType::chain_matmul)
  );
  m.impl("cholesky",
         TORCH_FN(TraceType::cholesky)
  );
  m.impl("clamp_max",
         TORCH_FN(TraceType::clamp_max)
  );
  m.impl("clamp_max_",
         TORCH_FN(TraceType::clamp_max_)
  );
  m.impl("coalesce",
         TORCH_FN(TraceType::coalesce)
  );
  m.impl("col2im.out",
         TORCH_FN(TraceType::col2im_out_out)
  );
  m.impl("combinations",
         TORCH_FN(TraceType::combinations)
  );
  m.impl("conj",
         TORCH_FN(TraceType::conj)
  );
  m.impl("conv_tbc",
         TORCH_FN(TraceType::conv_tbc)
  );
  m.impl("convolution",
         TORCH_FN(TraceType::convolution)
  );
  m.impl("copysign.Tensor",
         TORCH_FN(TraceType::copysign_Tensor)
  );
  m.impl("copysign.Scalar",
         TORCH_FN(TraceType::copysign_Scalar)
  );
  m.impl("copysign_.Tensor",
         TORCH_FN(TraceType::copysign__Tensor)
  );
  m.impl("copysign_.Scalar",
         TORCH_FN(TraceType::copysign__Scalar)
  );
  m.impl("cosine_similarity",
         TORCH_FN(TraceType::cosine_similarity)
  );
  m.impl("cudnn_convolution_backward_weight",
         TORCH_FN(TraceType::cudnn_convolution_backward_weight)
  );
  m.impl("cummax",
         TORCH_FN(TraceType::cummax)
  );
  m.impl("cummax.dimname",
         TORCH_FN(TraceType::cummax_dimname)
  );
  m.impl("cummaxmin_backward",
         TORCH_FN(TraceType::cummaxmin_backward)
  );
  m.impl("cumprod.out",
         TORCH_FN(TraceType::cumprod_out_out)
  );
  m.impl("cumprod.dimname_out",
         TORCH_FN(TraceType::cumprod_out_dimname_out)
  );
  m.impl("cumsum",
         TORCH_FN(TraceType::cumsum)
  );
  m.impl("cumsum.dimname",
         TORCH_FN(TraceType::cumsum_dimname)
  );
  m.impl("cumsum_",
         TORCH_FN(TraceType::cumsum_)
  );
  m.impl("cumsum_.dimname",
         TORCH_FN(TraceType::cumsum__dimname)
  );
  m.impl("dense_dim",
         TORCH_FN(TraceType::dense_dim)
  );
  m.impl("diag_backward",
         TORCH_FN(TraceType::diag_backward)
  );
  m.impl("diagonal",
         TORCH_FN(TraceType::diagonal)
  );
  m.impl("diagonal.Dimname",
         TORCH_FN(TraceType::diagonal_Dimname)
  );
  m.impl("dist",
         TORCH_FN(TraceType::dist)
  );
  m.impl("dot.out",
         TORCH_FN(TraceType::dot_out_out)
  );
  m.impl("dropout",
         TORCH_FN(TraceType::dropout)
  );
  m.impl("dropout_",
         TORCH_FN(TraceType::dropout_)
  );
  m.impl("elu",
         TORCH_FN(TraceType::elu)
  );
  m.impl("elu_",
         TORCH_FN(TraceType::elu_)
  );
  m.impl("embedding_renorm_",
         TORCH_FN(TraceType::embedding_renorm_)
  );
  m.impl("equal",
         TORCH_FN(TraceType::equal)
  );
  m.impl("erf.out",
         TORCH_FN(TraceType::erf_out_out)
  );
  m.impl("erfc",
         TORCH_FN(TraceType::erfc)
  );
  m.impl("erfc_",
         TORCH_FN(TraceType::erfc_)
  );
  m.impl("expm1",
         TORCH_FN(TraceType::expm1)
  );
  m.impl("expm1_",
         TORCH_FN(TraceType::expm1_)
  );
  m.impl("exponential_",
         TORCH_FN(TraceType::exponential_)
  );
  m.impl("fake_quantize_per_channel_affine",
         TORCH_FN(TraceType::fake_quantize_per_channel_affine)
  );
  m.impl("fbgemm_linear_fp16_weight_fp32_activation",
         TORCH_FN(TraceType::fbgemm_linear_fp16_weight_fp32_activation)
  );
  m.impl("fbgemm_linear_int8_weight_fp32_activation",
         TORCH_FN(TraceType::fbgemm_linear_int8_weight_fp32_activation)
  );
  m.impl("fbgemm_linear_quantize_weight",
         TORCH_FN(TraceType::fbgemm_linear_quantize_weight)
  );
  m.impl("fix.out",
         TORCH_FN(TraceType::fix_out_out)
  );
  m.impl("float_power.Tensor_Tensor",
         TORCH_FN(TraceType::float_power_Tensor_Tensor)
  );
  m.impl("float_power.Scalar",
         TORCH_FN(TraceType::float_power_Scalar)
  );
  m.impl("float_power.Tensor_Scalar",
         TORCH_FN(TraceType::float_power_Tensor_Scalar)
  );
  m.impl("float_power_.Scalar",
         TORCH_FN(TraceType::float_power__Scalar)
  );
  m.impl("float_power_.Tensor",
         TORCH_FN(TraceType::float_power__Tensor)
  );
  m.impl("floor",
         TORCH_FN(TraceType::floor)
  );
  m.impl("floor_",
         TORCH_FN(TraceType::floor_)
  );
  m.impl("fmin",
         TORCH_FN(TraceType::fmin)
  );
  m.impl("fmod.Scalar_out",
         TORCH_FN(TraceType::fmod_out_Scalar_out)
  );
  m.impl("fmod.Tensor_out",
         TORCH_FN(TraceType::fmod_out_Tensor_out)
  );
  m.impl("frac.out",
         TORCH_FN(TraceType::frac_out_out)
  );
  m.impl("fractional_max_pool2d.output",
         TORCH_FN(TraceType::fractional_max_pool2d_out_output)
  );
  m.impl("frobenius_norm.out",
         TORCH_FN(TraceType::frobenius_norm_out_out)
  );
  m.impl("full_like",
         TORCH_FN(TraceType::full_like)
  );
  m.impl("gather_backward",
         TORCH_FN(TraceType::gather_backward)
  );
  m.impl("gcd.out",
         TORCH_FN(TraceType::gcd_out_out)
  );
  m.impl("greater.Scalar",
         TORCH_FN(TraceType::greater_Scalar)
  );
  m.impl("greater.Tensor",
         TORCH_FN(TraceType::greater_Tensor)
  );
  m.impl("greater_.Scalar",
         TORCH_FN(TraceType::greater__Scalar)
  );
  m.impl("greater_.Tensor",
         TORCH_FN(TraceType::greater__Tensor)
  );
  m.impl("greater_equal.Scalar_out",
         TORCH_FN(TraceType::greater_equal_out_Scalar_out)
  );
  m.impl("greater_equal.Tensor_out",
         TORCH_FN(TraceType::greater_equal_out_Tensor_out)
  );
  m.impl("group_norm",
         TORCH_FN(TraceType::group_norm)
  );
  m.impl("hamming_window",
         TORCH_FN(TraceType::hamming_window)
  );
  m.impl("hamming_window.periodic",
         TORCH_FN(TraceType::hamming_window_periodic)
  );
  m.impl("hamming_window.periodic_alpha",
         TORCH_FN(TraceType::hamming_window_periodic_alpha)
  );
  m.impl("hamming_window.periodic_alpha_beta",
         TORCH_FN(TraceType::hamming_window_periodic_alpha_beta)
  );
  m.impl("hardshrink_backward",
         TORCH_FN(TraceType::hardshrink_backward)
  );
  m.impl("hardtanh.out",
         TORCH_FN(TraceType::hardtanh_out_out)
  );
  m.impl("heaviside",
         TORCH_FN(TraceType::heaviside)
  );
  m.impl("heaviside_",
         TORCH_FN(TraceType::heaviside_)
  );
  m.impl("im2col.out",
         TORCH_FN(TraceType::im2col_out_out)
  );
  m.impl("index.Tensor",
         TORCH_FN(TraceType::index_Tensor)
  );
  m.impl("index_put",
         TORCH_FN(TraceType::index_put)
  );
  m.impl("index_put_",
         TORCH_FN(TraceType::index_put_)
  );
  m.impl("index_select",
         TORCH_FN(TraceType::index_select)
  );
  m.impl("index_select.dimname",
         TORCH_FN(TraceType::index_select_dimname)
  );
  m.impl("is_coalesced",
         TORCH_FN(TraceType::is_coalesced)
  );
  m.impl("is_floating_point",
         TORCH_FN(TraceType::is_floating_point)
  );
  m.impl("is_vulkan_available",
         TORCH_FN(TraceType::is_vulkan_available)
  );
  m.impl("isneginf",
         TORCH_FN(TraceType::isneginf)
  );
  m.impl("item",
         TORCH_FN(TraceType::item)
  );
  m.impl("kron.out",
         TORCH_FN(TraceType::kron_out_out)
  );
  m.impl("l1_loss",
         TORCH_FN(TraceType::l1_loss)
  );
  m.impl("l1_loss_backward.grad_input",
         TORCH_FN(TraceType::l1_loss_backward_out_grad_input)
  );
  m.impl("lcm",
         TORCH_FN(TraceType::lcm)
  );
  m.impl("lcm_",
         TORCH_FN(TraceType::lcm_)
  );
  m.impl("ldexp.Tensor",
         TORCH_FN(TraceType::ldexp_Tensor)
  );
  m.impl("ldexp_",
         TORCH_FN(TraceType::ldexp_)
  );
  m.impl("linalg_cholesky.out",
         TORCH_FN(TraceType::linalg_cholesky_out_out)
  );
  m.impl("linalg_cond",
         TORCH_FN(TraceType::linalg_cond)
  );
  m.impl("linalg_cond.p_str",
         TORCH_FN(TraceType::linalg_cond_p_str)
  );
  m.impl("linalg_eigvalsh.out",
         TORCH_FN(TraceType::linalg_eigvalsh_out_out)
  );
  m.impl("linalg_householder_product",
         TORCH_FN(TraceType::linalg_householder_product)
  );
  m.impl("linalg_pinv",
         TORCH_FN(TraceType::linalg_pinv)
  );
  m.impl("linalg_pinv.rcond_tensor",
         TORCH_FN(TraceType::linalg_pinv_rcond_tensor)
  );
  m.impl("linalg_tensorinv",
         TORCH_FN(TraceType::linalg_tensorinv)
  );
  m.impl("linalg_tensorsolve",
         TORCH_FN(TraceType::linalg_tensorsolve)
  );
  m.impl("linspace.out",
         TORCH_FN(TraceType::linspace_out_out)
  );
  m.impl("log2.out",
         TORCH_FN(TraceType::log2_out_out)
  );
  m.impl("log_normal_",
         TORCH_FN(TraceType::log_normal_)
  );
  m.impl("log.out",
         TORCH_FN(TraceType::log_out_out)
  );
  m.impl("log_sigmoid_backward",
         TORCH_FN(TraceType::log_sigmoid_backward)
  );
  m.impl("log_sigmoid_forward.output",
         TORCH_FN(TraceType::log_sigmoid_forward_out_output)
  );
  m.impl("logaddexp2.out",
         TORCH_FN(TraceType::logaddexp2_out_out)
  );
  m.impl("logaddexp.out",
         TORCH_FN(TraceType::logaddexp_out_out)
  );
  m.impl("logcumsumexp",
         TORCH_FN(TraceType::logcumsumexp)
  );
  m.impl("logcumsumexp.dimname",
         TORCH_FN(TraceType::logcumsumexp_dimname)
  );
  m.impl("logical_or",
         TORCH_FN(TraceType::logical_or)
  );
  m.impl("logical_or_",
         TORCH_FN(TraceType::logical_or_)
  );
  m.impl("logical_xor",
         TORCH_FN(TraceType::logical_xor)
  );
  m.impl("logical_xor_",
         TORCH_FN(TraceType::logical_xor_)
  );
  m.impl("logspace",
         TORCH_FN(TraceType::logspace)
  );
  m.impl("logsumexp",
         TORCH_FN(TraceType::logsumexp)
  );
  m.impl("logsumexp.names",
         TORCH_FN(TraceType::logsumexp_names)
  );
  m.impl("lstsq.X",
         TORCH_FN(TraceType::lstsq_out_X)
  );
  m.impl("matmul",
         TORCH_FN(TraceType::matmul)
  );
  m.impl("max.dim",
         TORCH_FN(TraceType::max_dim)
  );
  m.impl("max.names_dim",
         TORCH_FN(TraceType::max_names_dim)
  );
  m.impl("max",
         TORCH_FN(TraceType::max)
  );
  m.impl("max.other",
         TORCH_FN(TraceType::max_other)
  );
  m.impl("max_pool1d_with_indices",
         TORCH_FN(TraceType::max_pool1d_with_indices)
  );
  m.impl("max_pool2d_with_indices.out",
         TORCH_FN(TraceType::max_pool2d_with_indices_out_out)
  );
  m.impl("max_unpool2d_backward",
         TORCH_FN(TraceType::max_unpool2d_backward)
  );
  m.impl("max_unpool3d",
         TORCH_FN(TraceType::max_unpool3d)
  );
  m.impl("max_unpool3d_backward.grad_input",
         TORCH_FN(TraceType::max_unpool3d_backward_out_grad_input)
  );
  m.impl("maximum",
         TORCH_FN(TraceType::maximum)
  );
  m.impl("mean.out",
         TORCH_FN(TraceType::mean_out_out)
  );
  m.impl("mean.names_out",
         TORCH_FN(TraceType::mean_out_names_out)
  );
  m.impl("median.dim_values",
         TORCH_FN(TraceType::median_out_dim_values)
  );
  m.impl("median.names_dim_values",
         TORCH_FN(TraceType::median_out_names_dim_values)
  );
  m.impl("meshgrid",
         TORCH_FN(TraceType::meshgrid)
  );
  m.impl("miopen_batch_norm",
         TORCH_FN(TraceType::miopen_batch_norm)
  );
  m.impl("miopen_convolution_transpose_backward",
         TORCH_FN(TraceType::miopen_convolution_transpose_backward)
  );
  m.impl("miopen_convolution_transpose_backward_input",
         TORCH_FN(TraceType::miopen_convolution_transpose_backward_input)
  );
  m.impl("miopen_depthwise_convolution_backward_weight",
         TORCH_FN(TraceType::miopen_depthwise_convolution_backward_weight)
  );
  m.impl("mkldnn_adaptive_avg_pool2d_backward",
         TORCH_FN(TraceType::mkldnn_adaptive_avg_pool2d_backward)
  );
  m.impl("mkldnn_convolution_backward",
         TORCH_FN(TraceType::mkldnn_convolution_backward)
  );
  m.impl("mkldnn_convolution_backward_input",
         TORCH_FN(TraceType::mkldnn_convolution_backward_input)
  );
  m.impl("mkldnn_reorder_conv3d_weight",
         TORCH_FN(TraceType::mkldnn_reorder_conv3d_weight)
  );
  m.impl("mode",
         TORCH_FN(TraceType::mode)
  );
  m.impl("mode.dimname",
         TORCH_FN(TraceType::mode_dimname)
  );
  m.impl("moveaxis.intlist",
         TORCH_FN(TraceType::moveaxis_intlist)
  );
  m.impl("moveaxis.int",
         TORCH_FN(TraceType::moveaxis_int)
  );
  m.impl("multi_margin_loss.out",
         TORCH_FN(TraceType::multi_margin_loss_out_out)
  );
  m.impl("multilabel_margin_loss_forward",
         TORCH_FN(TraceType::multilabel_margin_loss_forward)
  );
  m.impl("multiply.Tensor",
         TORCH_FN(TraceType::multiply_Tensor)
  );
  m.impl("multiply.Scalar",
         TORCH_FN(TraceType::multiply_Scalar)
  );
  m.impl("multiply_.Tensor",
         TORCH_FN(TraceType::multiply__Tensor)
  );
  m.impl("multiply_.Scalar",
         TORCH_FN(TraceType::multiply__Scalar)
  );
  m.impl("mv.out",
         TORCH_FN(TraceType::mv_out_out)
  );
  m.impl("native_batch_norm_backward",
         TORCH_FN(TraceType::native_batch_norm_backward)
  );
  m.impl("native_norm",
         TORCH_FN(TraceType::native_norm)
  );
  m.impl("native_norm.ScalarOpt_dim_dtype",
         TORCH_FN(TraceType::native_norm_ScalarOpt_dim_dtype)
  );
  m.impl("ne.Scalar",
         TORCH_FN(TraceType::ne_Scalar)
  );
  m.impl("ne.Tensor",
         TORCH_FN(TraceType::ne_Tensor)
  );
  m.impl("ne_.Scalar",
         TORCH_FN(TraceType::ne__Scalar)
  );
  m.impl("ne_.Tensor",
         TORCH_FN(TraceType::ne__Tensor)
  );
  m.impl("negative",
         TORCH_FN(TraceType::negative)
  );
  m.impl("negative_",
         TORCH_FN(TraceType::negative_)
  );
  m.impl("nextafter.out",
         TORCH_FN(TraceType::nextafter_out_out)
  );
  m.impl("nll_loss2d_backward",
         TORCH_FN(TraceType::nll_loss2d_backward)
  );
  m.impl("nll_loss2d_forward.output",
         TORCH_FN(TraceType::nll_loss2d_forward_out_output)
  );
  m.impl("nll_loss_backward",
         TORCH_FN(TraceType::nll_loss_backward)
  );
  m.impl("nll_loss_forward.output",
         TORCH_FN(TraceType::nll_loss_forward_out_output)
  );
  m.impl("ones.out",
         TORCH_FN(TraceType::ones_out_out)
  );
  m.impl("ormqr",
         TORCH_FN(TraceType::ormqr)
  );
  m.impl("pairwise_distance",
         TORCH_FN(TraceType::pairwise_distance)
  );
  m.impl("pinverse",
         TORCH_FN(TraceType::pinverse)
  );
  m.impl("polar.out",
         TORCH_FN(TraceType::polar_out_out)
  );
  m.impl("polygamma.out",
         TORCH_FN(TraceType::polygamma_out_out)
  );
  m.impl("pow.Tensor_Tensor_out",
         TORCH_FN(TraceType::pow_out_Tensor_Tensor_out)
  );
  m.impl("pow.Scalar_out",
         TORCH_FN(TraceType::pow_out_Scalar_out)
  );
  m.impl("pow.Tensor_Scalar_out",
         TORCH_FN(TraceType::pow_out_Tensor_Scalar_out)
  );
  m.impl("prod.int_out",
         TORCH_FN(TraceType::prod_out_int_out)
  );
  m.impl("prod.Dimname_out",
         TORCH_FN(TraceType::prod_out_Dimname_out)
  );
  m.impl("q_per_channel_axis",
         TORCH_FN(TraceType::q_per_channel_axis)
  );
  m.impl("q_per_channel_zero_points",
         TORCH_FN(TraceType::q_per_channel_zero_points)
  );
  m.impl("qr.Q",
         TORCH_FN(TraceType::qr_out_Q)
  );
  m.impl("quantile.scalar_out",
         TORCH_FN(TraceType::quantile_out_scalar_out)
  );
  m.impl("quantile.out",
         TORCH_FN(TraceType::quantile_out_out)
  );
  m.impl("quantized_gru_cell",
         TORCH_FN(TraceType::quantized_gru_cell)
  );
  m.impl("quantized_rnn_relu_cell",
         TORCH_FN(TraceType::quantized_rnn_relu_cell)
  );
  m.impl("rand.names",
         TORCH_FN(TraceType::rand_names)
  );
  m.impl("rand.generator_with_names",
         TORCH_FN(TraceType::rand_generator_with_names)
  );
  m.impl("rand",
         TORCH_FN(TraceType::rand)
  );
  m.impl("rand.generator",
         TORCH_FN(TraceType::rand_generator)
  );
  m.impl("randint.out",
         TORCH_FN(TraceType::randint_out_out)
  );
  m.impl("randint.generator_out",
         TORCH_FN(TraceType::randint_out_generator_out)
  );
  m.impl("randint.low_out",
         TORCH_FN(TraceType::randint_out_low_out)
  );
  m.impl("randint.low_generator_out",
         TORCH_FN(TraceType::randint_out_low_generator_out)
  );
  m.impl("randn",
         TORCH_FN(TraceType::randn)
  );
  m.impl("randn.generator",
         TORCH_FN(TraceType::randn_generator)
  );
  m.impl("randn.names",
         TORCH_FN(TraceType::randn_names)
  );
  m.impl("randn.generator_with_names",
         TORCH_FN(TraceType::randn_generator_with_names)
  );
  m.impl("random_.from",
         TORCH_FN(TraceType::random__from)
  );
  m.impl("random_.to",
         TORCH_FN(TraceType::random__to)
  );
  m.impl("random_",
         TORCH_FN(TraceType::random_)
  );
  m.impl("randperm.out",
         TORCH_FN(TraceType::randperm_out_out)
  );
  m.impl("randperm.generator_out",
         TORCH_FN(TraceType::randperm_out_generator_out)
  );
  m.impl("record_stream",
         TORCH_FN(TraceType::record_stream)
  );
  m.impl("reflection_pad1d_backward",
         TORCH_FN(TraceType::reflection_pad1d_backward)
  );
  m.impl("reflection_pad2d",
         TORCH_FN(TraceType::reflection_pad2d)
  );
  m.impl("reflection_pad2d_backward.grad_input",
         TORCH_FN(TraceType::reflection_pad2d_backward_out_grad_input)
  );
  m.impl("remainder.Scalar",
         TORCH_FN(TraceType::remainder_Scalar)
  );
  m.impl("remainder.Tensor",
         TORCH_FN(TraceType::remainder_Tensor)
  );
  m.impl("remainder_.Scalar",
         TORCH_FN(TraceType::remainder__Scalar)
  );
  m.impl("remainder_.Tensor",
         TORCH_FN(TraceType::remainder__Tensor)
  );
  m.impl("repeat",
         TORCH_FN(TraceType::repeat)
  );
  m.impl("replication_pad1d",
         TORCH_FN(TraceType::replication_pad1d)
  );
  m.impl("replication_pad1d_backward.grad_input",
         TORCH_FN(TraceType::replication_pad1d_backward_out_grad_input)
  );
  m.impl("replication_pad2d.out",
         TORCH_FN(TraceType::replication_pad2d_out_out)
  );
  m.impl("reshape_as",
         TORCH_FN(TraceType::reshape_as)
  );
  m.impl("result_type.Tensor",
         TORCH_FN(TraceType::result_type_Tensor)
  );
  m.impl("result_type.Scalar",
         TORCH_FN(TraceType::result_type_Scalar)
  );
  m.impl("result_type.Scalar_Tensor",
         TORCH_FN(TraceType::result_type_Scalar_Tensor)
  );
  m.impl("result_type.Scalar_Scalar",
         TORCH_FN(TraceType::result_type_Scalar_Scalar)
  );
  m.impl("rnn_tanh.input",
         TORCH_FN(TraceType::rnn_tanh_input)
  );
  m.impl("rnn_tanh.data",
         TORCH_FN(TraceType::rnn_tanh_data)
  );
  m.impl("roll",
         TORCH_FN(TraceType::roll)
  );
  m.impl("rot90",
         TORCH_FN(TraceType::rot90)
  );
  m.impl("round.out",
         TORCH_FN(TraceType::round_out_out)
  );
  m.impl("rrelu_with_noise_backward",
         TORCH_FN(TraceType::rrelu_with_noise_backward)
  );
  m.impl("rsqrt.out",
         TORCH_FN(TraceType::rsqrt_out_out)
  );
  m.impl("searchsorted.Tensor_out",
         TORCH_FN(TraceType::searchsorted_out_Tensor_out)
  );
  m.impl("select_backward",
         TORCH_FN(TraceType::select_backward)
  );
  m.impl("selu",
         TORCH_FN(TraceType::selu)
  );
  m.impl("selu_",
         TORCH_FN(TraceType::selu_)
  );
  m.impl("signbit.out",
         TORCH_FN(TraceType::signbit_out_out)
  );
  m.impl("silu_backward",
         TORCH_FN(TraceType::silu_backward)
  );
  m.impl("slow_conv3d.out",
         TORCH_FN(TraceType::slow_conv3d_out_out)
  );
  m.impl("slow_conv_transpose2d.out",
         TORCH_FN(TraceType::slow_conv_transpose2d_out_out)
  );
  m.impl("smooth_l1_loss.out",
         TORCH_FN(TraceType::smooth_l1_loss_out_out)
  );
  m.impl("soft_margin_loss",
         TORCH_FN(TraceType::soft_margin_loss)
  );
  m.impl("soft_margin_loss_backward.grad_input",
         TORCH_FN(TraceType::soft_margin_loss_backward_out_grad_input)
  );
  m.impl("softplus",
         TORCH_FN(TraceType::softplus)
  );
  m.impl("softplus_backward.grad_input",
         TORCH_FN(TraceType::softplus_backward_out_grad_input)
  );
  m.impl("sort",
         TORCH_FN(TraceType::sort)
  );
  m.impl("sort.stable",
         TORCH_FN(TraceType::sort_stable)
  );
  m.impl("sort.dimname",
         TORCH_FN(TraceType::sort_dimname)
  );
  m.impl("sort.dimname_stable",
         TORCH_FN(TraceType::sort_dimname_stable)
  );
  m.impl("special_erfinv.out",
         TORCH_FN(TraceType::special_erfinv_out_out)
  );
  m.impl("split.Tensor",
         TORCH_FN(TraceType::split_Tensor)
  );
  m.impl("square.out",
         TORCH_FN(TraceType::square_out_out)
  );
  m.impl("sspaddmm.out",
         TORCH_FN(TraceType::sspaddmm_out_out)
  );
  m.impl("std",
         TORCH_FN(TraceType::std)
  );
  m.impl("std.dim",
         TORCH_FN(TraceType::std_dim)
  );
  m.impl("std.names_dim",
         TORCH_FN(TraceType::std_names_dim)
  );
  m.impl("stride.int",
         TORCH_FN(TraceType::stride_int)
  );
  m.impl("stride.Dimname",
         TORCH_FN(TraceType::stride_Dimname)
  );
  m.impl("subtract.out",
         TORCH_FN(TraceType::subtract_out_out)
  );
  m.impl("sum",
         TORCH_FN(TraceType::sum)
  );
  m.impl("sum.dim_IntList",
         TORCH_FN(TraceType::sum_dim_IntList)
  );
  m.impl("sum.dim_DimnameList",
         TORCH_FN(TraceType::sum_dim_DimnameList)
  );
  m.impl("sum_to_size",
         TORCH_FN(TraceType::sum_to_size)
  );
  m.impl("swapaxes",
         TORCH_FN(TraceType::swapaxes)
  );
  m.impl("swapaxes_",
         TORCH_FN(TraceType::swapaxes_)
  );
  m.impl("t",
         TORCH_FN(TraceType::t)
  );
  m.impl("t_",
         TORCH_FN(TraceType::t_)
  );
  m.impl("take",
         TORCH_FN(TraceType::take)
  );
  m.impl("tanh.out",
         TORCH_FN(TraceType::tanh_out_out)
  );
  m.impl("thnn_conv_depthwise2d",
         TORCH_FN(TraceType::thnn_conv_depthwise2d)
  );
  m.impl("thnn_conv_depthwise2d_backward.grad_input",
         TORCH_FN(TraceType::thnn_conv_depthwise2d_backward_out_grad_input)
  );
  m.impl("to_mkldnn",
         TORCH_FN(TraceType::to_mkldnn)
  );
  m.impl("to_sparse.sparse_dim",
         TORCH_FN(TraceType::to_sparse_sparse_dim)
  );
  m.impl("to_sparse",
         TORCH_FN(TraceType::to_sparse)
  );
  m.impl("topk",
         TORCH_FN(TraceType::topk)
  );
  m.impl("transpose.int",
         TORCH_FN(TraceType::transpose_int)
  );
  m.impl("transpose.Dimname",
         TORCH_FN(TraceType::transpose_Dimname)
  );
  m.impl("transpose_",
         TORCH_FN(TraceType::transpose_)
  );
  m.impl("trapz.x",
         TORCH_FN(TraceType::trapz_x)
  );
  m.impl("trapz.dx",
         TORCH_FN(TraceType::trapz_dx)
  );
  m.impl("triu.out",
         TORCH_FN(TraceType::triu_out_out)
  );
  m.impl("true_divide.out",
         TORCH_FN(TraceType::true_divide_out_out)
  );
  m.impl("trunc",
         TORCH_FN(TraceType::trunc)
  );
  m.impl("trunc_",
         TORCH_FN(TraceType::trunc_)
  );
  m.impl("upsample_bicubic2d.out",
         TORCH_FN(TraceType::upsample_bicubic2d_out_out)
  );
  m.impl("upsample_linear1d_backward.vec",
         TORCH_FN(TraceType::upsample_linear1d_backward_vec)
  );
  m.impl("upsample_linear1d_backward",
         TORCH_FN(TraceType::upsample_linear1d_backward)
  );
  m.impl("upsample_nearest2d_backward.vec",
         TORCH_FN(TraceType::upsample_nearest2d_backward_vec)
  );
  m.impl("upsample_nearest2d_backward",
         TORCH_FN(TraceType::upsample_nearest2d_backward)
  );
  m.impl("upsample_nearest3d.vec",
         TORCH_FN(TraceType::upsample_nearest3d_vec)
  );
  m.impl("upsample_nearest3d",
         TORCH_FN(TraceType::upsample_nearest3d)
  );
  m.impl("upsample_nearest3d_backward.grad_input",
         TORCH_FN(TraceType::upsample_nearest3d_backward_out_grad_input)
  );
  m.impl("upsample_trilinear3d.out",
         TORCH_FN(TraceType::upsample_trilinear3d_out_out)
  );
  m.impl("values",
         TORCH_FN(TraceType::values)
  );
  m.impl("var_mean",
         TORCH_FN(TraceType::var_mean)
  );
  m.impl("var_mean.dim",
         TORCH_FN(TraceType::var_mean_dim)
  );
  m.impl("var_mean.names_dim",
         TORCH_FN(TraceType::var_mean_names_dim)
  );
  m.impl("vstack.out",
         TORCH_FN(TraceType::vstack_out_out)
  );;
}

}  // namespace

} // namespace torch
