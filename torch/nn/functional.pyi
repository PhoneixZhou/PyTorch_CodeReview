from torch import Tensor
from torch.types import _size
from typing import Any, Optional, Tuple, Dict, List, Callable, Sequence, Union
from .common_types import _ratio_any_t, _size_any_t, _size_1_t, _size_2_t, _size_3_t, _size_2_opt_t, _size_3_opt_t

# 'TypedDict' is a new accepted type that represents a dictionary with a fixed set of allowed keys.
# It is standards-track but not in `typing` yet. We leave this hear to be uncommented once the feature
# is wide-spread.

# from mypy_extensions import TypedDict

# GRID_SAMPLE_INTERPOLATION_MODES = TypedDict('GRID_SAMPLE_INTERPOLATION_MODES', {'bilinear': int, 'nearest': int})
# GRID_SAMPLE_PADDING_MODES = TypedDict('GRID_SAMPLE_PADDING_MODES', {'zeros': int, 'border': int, 'reflection': int})

GRID_SAMPLE_INTERPOLATION_MODES = Dict[str, int]
GRID_SAMPLE_PADDING_MODES = Dict[str, int]


# These stubs were generated by running stubgen (`stubgen --parse-only functional.py`), followed by manual cleaning.
#
# The 'BroadcastingList{1,2,3}' types were replaced by `_size` or _output_ratio, as appropriate.
# This was necessary since the JIT uses BroadcastingList* types but static checking with mypy etc requires a `Sequence`
# type. There is no way to express the expected lengths of these lists in the current Python typing system.
#
# Functions created via `_add_docstr` in `functional.py` where merely typed as `Any` by `stubgen`, so those were
# deleted from the stub and replaced by generated declarations. See `gen_pyi` for the implementation of the code
# generation logic for those functions. In the future, it might be worth looking into using the mypy plugin system
# to encode the type semantics of `_add_docstr`, should that system ever become widespread.
def fractional_max_pool2d_with_indices(input: Tensor, kernel_size: _size, output_size: Optional[_size] = ...,
                                       output_ratio: Optional[_ratio_any_t] = ..., return_indices: bool = ...,
                                       _random_samples: Optional[Tensor] = ...) -> Tuple[Tensor, Tensor]: ...


def fractional_max_pool3d_with_indices(input: Tensor, kernel_size: _size, output_size: Optional[_size] = ...,
                                       output_ratio: Optional[_ratio_any_t] = ..., return_indices: bool = ...,
                                       _random_samples: Optional[Tensor] = ...) -> Tuple[Tensor, Tensor]: ...


def max_pool1d_with_indices(input: Tensor, kernel_size: _size, stride: Optional[_size] = ..., padding: _size = ...,
                            dilation: _size = ..., ceil_mode: bool = ..., return_indices: bool = ...) -> Tuple[
    Tensor, Tensor]: ...


def max_pool2d_with_indices(input: Tensor, kernel_size: _size, stride: Optional[_size] = ..., padding: _size = ...,
                            dilation: _size = ..., ceil_mode: bool = ..., return_indices: bool = ...) -> Tuple[
    Tensor, Tensor]: ...


def max_pool3d_with_indices(input: Tensor, kernel_size: _size, stride: Optional[_size] = ..., padding: _size = ...,
                            dilation: _size = ..., ceil_mode: bool = ..., return_indices: bool = ...) -> Tuple[
    Tensor, Tensor]: ...


def max_unpool1d(input: Tensor, indices: Tensor, kernel_size: _size, stride: Optional[_size] = ...,
                 padding: _size = ..., output_size: Optional[_size] = ...) -> Tensor: ...


def max_unpool2d(input: Tensor, indices: Tensor, kernel_size: _size, stride: Optional[_size] = ...,
                 padding: _size = ..., output_size: Optional[_size] = ...) -> Tensor: ...


def max_unpool3d(input: Tensor, indices: Tensor, kernel_size: _size, stride: Optional[_size] = ...,
                 padding: _size = ..., output_size: Optional[_size] = ...) -> Tensor: ...


def lp_pool1d(input: Tensor, norm_type: float, kernel_size: _size_1_t, stride: Union[Optional[_size], Optional[int]] = ...,
              ceil_mode: bool = ...) -> Tensor: ...


def lp_pool2d(input: Tensor, norm_type: float, kernel_size: _size_2_t, stride: Union[Optional[_size], Optional[int]] = ...,
              ceil_mode: bool = ...) -> Tensor: ...


def adaptive_max_pool1d_with_indices(input: Tensor, output_size: _size, return_indices: bool = ...) -> Tuple[
    Tensor, Tensor]: ...


def adaptive_max_pool2d_with_indices(input: Tensor, output_size: _size_2_opt_t, return_indices: bool = ...) -> Tuple[
    Tensor, Tensor]: ...


def adaptive_max_pool3d_with_indices(input: Tensor, output_size: _size_3_opt_t, return_indices: bool = ...) -> Tuple[
    Tensor, Tensor]: ...


def adaptive_avg_pool1d(input: Tensor, output_size: _size_1_t) -> Tensor: ...


def adaptive_avg_pool2d(input: Tensor, output_size: _size_2_opt_t) -> Tensor: ...


def adaptive_avg_pool3d(input: Tensor, output_size: _size_3_opt_t) -> Tensor: ...


def dropout(input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...) -> Tensor: ...


def alpha_dropout(input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...) -> Tensor: ...


def dropout2d(input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...) -> Tensor: ...


def dropout3d(input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...) -> Tensor: ...


def feature_alpha_dropout(input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...) -> Tensor: ...


def threshold(input: Tensor, threshold: float, value: float, inplace: bool = ...) -> Tensor: ...


def relu(input: Tensor, inplace: bool = ...) -> Tensor: ...


def glu(input: Tensor, dim: int = ...) -> Tensor: ...


def hardtanh(input: Tensor, min_val: float = ..., max_val: float = ..., inplace: bool = ...) -> Tensor: ...


def relu6(input: Tensor, inplace: bool = ...) -> Tensor: ...


def elu(input: Tensor, alpha: float = ..., inplace: bool = ...) -> Tensor: ...


def selu(input: Tensor, inplace: bool = ...) -> Tensor: ...


def celu(input: Tensor, alpha: float = ..., inplace: bool = ...) -> Tensor: ...


def leaky_relu(input: Tensor, negative_slope: float = ..., inplace: bool = ...) -> Tensor: ...


def prelu(input: Tensor, weight: Tensor) -> Tensor: ...


def rrelu(input: Tensor, lower: float = ..., upper: float = ..., training: bool = ...,
          inplace: bool = ...) -> Tensor: ...


def gelu(input: Any): ...


def hardshrink(input: Tensor, lambd: float = ...) -> Tensor: ...


def tanhshrink(input: Any): ...


def softsign(input: Any): ...


def softmin(input: Tensor, dim: Optional[int] = ..., _stacklevel: int = ..., dtype: Optional[int] = ...) -> Tensor: ...


def softmax(input: Tensor, dim: Optional[int] = ..., _stacklevel: int = ..., dtype: Optional[int] = ...) -> Tensor: ...


def gumbel_softmax(logits: Tensor, tau: float = ..., hard: bool = ..., eps: float = ..., dim: int = ...) -> Tensor: ...


def log_softmax(input: Tensor, dim: Optional[int] = ..., _stacklevel: int = ...,
                dtype: Optional[int] = ...) -> Tensor: ...


def tanh(input: Any): ...


def sigmoid(input: Any) -> Tensor: ...

def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor: ...


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = ...) -> Tensor: ...


def bilinear(input1: Tensor, input2: Tensor, weight: Tensor, bias: Optional[Tensor] = ...) -> Tensor: ...


def silu(input: Tensor, inplace: bool = False) -> Tensor: ...


def hardswish(input: Tensor, inplace: bool = False) -> Tensor: ...


def embedding(input: Tensor, weight: Tensor, padding_idx: Optional[int] = ..., max_norm: Optional[float] = ...,
              norm_type: float = ..., scale_grad_by_freq: bool = ..., sparse: bool = ...) -> Tensor: ...


def embedding_bag(input: Tensor, weight: Tensor, offsets: Optional[Tensor] = ..., max_norm: Optional[float] = ...,
                  norm_type: float = ..., scale_grad_by_freq: bool = ..., mode: str = ...,
                  sparse: bool = ..., per_sample_weights: Optional[Tensor] = ...,
                  include_last_offset: bool = ...) -> Tensor: ...

def batch_norm(input: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor],
               weight: Optional[Tensor] = ..., bias: Optional[Tensor] = ..., training: bool = ...,
               momentum: float = ..., eps: float = ...) -> Tensor: ...


def instance_norm(input: Tensor, running_mean: Optional[Tensor] = ..., running_var: Optional[Tensor] = ...,
                  weight: Optional[Tensor] = ..., bias: Optional[Tensor] = ..., use_input_stats: bool = ...,
                  momentum: float = ..., eps: float = ...) -> Tensor: ...


def layer_norm(input: Tensor, normalized_shape: Sequence[int], weight: Optional[Tensor] = ..., bias: Optional[Tensor] = ...,
               eps: float = ...) -> Tensor: ...


def group_norm(input: Tensor, num_groups: int, weight: Optional[Tensor] = ..., bias: Optional[Tensor] = ...,
               eps: float = ...) -> Tensor: ...


def local_response_norm(input: Tensor, size: int, alpha: float = ..., beta: float = ..., k: float = ...) -> Tensor: ...


def ctc_loss(log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank: int = ...,
             reduction: str = ..., zero_infinity: bool = ...) -> Tensor: ...


def nll_loss(input: Tensor, target: Tensor, weight: Optional[Tensor] = ..., size_average: Optional[bool] = ...,
             ignore_index: int = ..., reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


def poisson_nll_loss(input: Tensor, target: Tensor, log_input: bool = ..., full: bool = ...,
                     size_average: Optional[bool] = ..., eps: float = ..., reduce: Optional[bool] = ...,
                     reduction: str = ...) -> Tensor: ...


def gaussian_nll_loss(input: Tensor, target: Tensor, var: Tensor, full: Optional[bool] = ...,
                      eps: Optional[float] = ..., reduction: Optional[str] = ...) -> Tensor: ...


def kl_div(input: Tensor, target: Tensor, size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
           reduction: str = ..., log_target: bool = ...) -> Tensor: ...


def cross_entropy(input: Tensor, target: Tensor, weight: Optional[Tensor] = ..., size_average: Optional[bool] = ...,
                  ignore_index: int = ..., reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


def binary_cross_entropy(input: Tensor, target: Tensor, weight: Optional[Tensor] = ...,
                         size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
                         reduction: str = ...) -> Tensor: ...


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor, weight: Optional[Tensor] = ...,
                                     size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
                                     reduction: str = ..., pos_weight: Optional[Tensor] = ...) -> Tensor: ...


def smooth_l1_loss(input: Tensor, target: Tensor, size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
                   reduction: str = ..., beta: float = ...) -> Tensor: ...


def huber_loss(input: Tensor, target: Tensor, reduction: str = ..., delta: float = ...) -> Tensor: ...


def l1_loss(input: Tensor, target: Tensor, size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
            reduction: str = ...) -> Tensor: ...


def mse_loss(input: Tensor, target: Tensor, size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
             reduction: str = ...) -> Tensor: ...


def margin_ranking_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = ...,
                        size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
                        reduction: str = ...) -> Tensor: ...


def hinge_embedding_loss(input: Tensor, target: Tensor, margin: float = ..., size_average: Optional[bool] = ...,
                         reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


def multilabel_margin_loss(input: Tensor, target: Tensor, size_average: Optional[bool] = ...,
                           reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


def soft_margin_loss(input: Tensor, target: Tensor, size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
                     reduction: str = ...) -> Tensor: ...


def multilabel_soft_margin_loss(input: Tensor, target: Tensor, weight: Optional[Tensor] = ...,
                                size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
                                reduction: str = ...) -> Tensor: ...


def cosine_embedding_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = ...,
                          size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
                          reduction: str = ...) -> Tensor: ...


def multi_margin_loss(input: Tensor, target: Tensor, p: int = ..., margin: float = ..., weight: Optional[Tensor] = ...,
                      size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
                      reduction: str = ...) -> Tensor: ...


def upsample(input: Any, size: Optional[Any] = ..., scale_factor: Optional[Any] = ..., mode: str = ...,
             align_corners: Optional[Any] = ...): ...


def interpolate(input: Any, size: Optional[Any] = ..., scale_factor: Optional[Any] = ..., mode: str = ...,
                align_corners: Optional[Any] = ...): ...


def upsample_nearest(input: Any, size: Optional[Any] = ..., scale_factor: Optional[Any] = ...): ...


def upsample_bilinear(input: Any, size: Optional[Any] = ..., scale_factor: Optional[Any] = ...): ...


def grid_sample(input: Tensor, grid: Tensor, mode: str = ..., padding_mode: str = ...,
                align_corners: Optional[Any] = ...) -> Tensor: ...


def affine_grid(theta: Tensor, size: List[int], align_corners: Optional[Any] = ...) -> Tensor: ...


def pad(input: Tensor, pad: Sequence[int], mode: str = ..., value: float = ...) -> Tensor: ...


def pairwise_distance(x1: Tensor, x2: Tensor, p: float = ..., eps: float = ..., keepdim: bool = ...) -> Tensor: ...


def triplet_margin_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = ..., p: float = ...,
                        eps: float = ..., swap: bool = ..., size_average: Optional[bool] = ...,
                        reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


def triplet_margin_with_distance_loss(anchor: Tensor, positive: Tensor, negative: Tensor, *,
                                      distance_function: Optional[Callable[[Tensor, Tensor], Tensor]]=...,
                                      margin: float=..., swap: bool=..., reduction: str=...) -> Tensor: ...


def normalize(input: Tensor, p: float = ..., dim: int = ..., eps: float = ...,
              out: Optional[Tensor] = ...) -> Tensor: ...


def assert_int_or_pair(arg: Any, arg_name: Any, message: Any) -> None: ...


def unfold(input: Tensor, kernel_size: _size_any_t, dilation: _size_any_t = ..., padding: _size_any_t = ...,
           stride: _size_any_t = ...) -> Tensor: ...


def fold(input: Tensor, output_size: _size_any_t, kernel_size: _size_any_t, dilation: _size_any_t = ..., padding: _size_any_t = ...,
         stride: _size_any_t = ...) -> Tensor: ...


def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None
                                 ) -> Tuple[Tensor, Optional[Tensor]]: ...


from .. import conv1d as conv1d
from .. import conv2d as conv2d
from .. import conv3d as conv3d
from .. import conv_transpose1d as conv_transpose1d
from .. import conv_transpose2d as conv_transpose2d
from .. import conv_transpose3d as conv_transpose3d
from .. import conv_tbc as conv_tbc
from .. import avg_pool1d as avg_pool1d
from .. import relu_ as relu_
from .. import selu_ as selu_
from .. import celu_ as celu_
from .. import rrelu_ as rrelu_
from .. import pixel_shuffle as pixel_shuffle
from .. import pixel_unshuffle as pixel_unshuffle
from .. import channel_shuffle as channel_shuffle
from .. import pdist as pdist
from .. import cosine_similarity as cosine_similarity

fractional_max_pool2d: Callable
fractional_max_pool3d: Callable
max_pool1d: Callable
max_pool2d: Callable
max_pool3d: Callable
adaptive_max_pool1d: Callable
adaptive_max_pool2d: Callable
adaptive_max_pool3d: Callable
avg_pool2d: Callable
avg_pool3d: Callable
hardtanh_: Callable
elu_: Callable
leaky_relu_: Callable
logsigmoid: Callable
softplus: Callable
softshrink: Callable
one_hot: Callable
