// @generated from tools/autograd/templates/python_linalg_functions.cpp

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_linalg_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;

using namespace torch::autograd::utils;

namespace torch { namespace autograd {

// generated forward declarations start here

static PyObject * THPVariable_linalg_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_cond(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_det(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_eigh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_eigvalsh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_householder_product(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_inv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_lstsq(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_matrix_rank(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_pinv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_qr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_svd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_tensorinv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_tensorsolve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_vector_norm(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef linalg_functions[] = {
  {"linalg_cholesky", castPyCFunctionWithKeywords(THPVariable_linalg_cholesky), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_cond", castPyCFunctionWithKeywords(THPVariable_linalg_cond), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_det", castPyCFunctionWithKeywords(THPVariable_linalg_det), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_eigh", castPyCFunctionWithKeywords(THPVariable_linalg_eigh), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_eigvalsh", castPyCFunctionWithKeywords(THPVariable_linalg_eigvalsh), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_householder_product", castPyCFunctionWithKeywords(THPVariable_linalg_householder_product), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_inv", castPyCFunctionWithKeywords(THPVariable_linalg_inv), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_lstsq", castPyCFunctionWithKeywords(THPVariable_linalg_lstsq), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_matrix_power", castPyCFunctionWithKeywords(THPVariable_linalg_matrix_power), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_matrix_rank", castPyCFunctionWithKeywords(THPVariable_linalg_matrix_rank), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_norm", castPyCFunctionWithKeywords(THPVariable_linalg_norm), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_pinv", castPyCFunctionWithKeywords(THPVariable_linalg_pinv), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_qr", castPyCFunctionWithKeywords(THPVariable_linalg_qr), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_slogdet", castPyCFunctionWithKeywords(THPVariable_linalg_slogdet), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_solve", castPyCFunctionWithKeywords(THPVariable_linalg_solve), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_svd", castPyCFunctionWithKeywords(THPVariable_linalg_svd), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_tensorinv", castPyCFunctionWithKeywords(THPVariable_linalg_tensorinv), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_tensorsolve", castPyCFunctionWithKeywords(THPVariable_linalg_tensorsolve), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_vector_norm", castPyCFunctionWithKeywords(THPVariable_linalg_vector_norm), METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL}
};

static PyObject* THPLinalgVariableFunctionsModule = NULL;

void initLinalgFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._linalg",
     NULL,
     -1,
     linalg_functions
  };
  PyObject* linalg = PyModule_Create(&def);
  THPLinalgVariableFunctionsModule = linalg;
  if (!linalg) {
    throw python_error();
  }
  // steals a reference to linalg
  if (PyModule_AddObject(module, "_linalg", linalg) != 0) {
    throw python_error();
  }
}

// generated methods start here

// linalg_cholesky
static PyObject * THPVariable_linalg_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_cholesky(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_cholesky(Tensor self) -> Tensor
    
    auto dispatch_linalg_cholesky = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cholesky(self);
    };
    return wrap(dispatch_linalg_cholesky(_r.tensor(0)));
  } else {
    // aten::linalg_cholesky.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_cholesky_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cholesky_out(out, self);
    };
    return wrap(dispatch_linalg_cholesky_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_cond
static PyObject * THPVariable_linalg_cond(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_cond(Tensor input, Scalar? p=None, *, Tensor out=None)",
    "linalg_cond(Tensor input, std::string p, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::linalg_cond(Tensor self, Scalar? p=None) -> Tensor
        
        auto dispatch_linalg_cond = [](const Tensor & self, const c10::optional<Scalar> & p) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond(self, p);
        };
        return wrap(dispatch_linalg_cond(_r.tensor(0), _r.scalarOptional(1)));
      } else {
        // aten::linalg_cond.out(Tensor self, Scalar? p=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_cond_out = [](Tensor out, const Tensor & self, const c10::optional<Scalar> & p) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond_out(out, self, p);
        };
        return wrap(dispatch_linalg_cond_out(_r.tensor(2), _r.tensor(0), _r.scalarOptional(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::linalg_cond.p_str(Tensor self, str p) -> Tensor
        
        auto dispatch_linalg_cond = [](const Tensor & self, std::string p) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond(self, p);
        };
        return wrap(dispatch_linalg_cond(_r.tensor(0), _r.string(1)));
      } else {
        // aten::linalg_cond.p_str_out(Tensor self, str p, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_cond_out = [](Tensor out, const Tensor & self, std::string p) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond_out(out, self, p);
        };
        return wrap(dispatch_linalg_cond_out(_r.tensor(2), _r.tensor(0), _r.string(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_det
static PyObject * THPVariable_linalg_det(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_det(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  // aten::linalg_det(Tensor self) -> Tensor
  
  auto dispatch_linalg_det = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::linalg_det(self);
  };
  return wrap(dispatch_linalg_det(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_eigh
static PyObject * THPVariable_linalg_eigh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"eigenvalues", ""}, {"eigenvectors", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_eigh", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_eigh_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_eigh(Tensor input, std::string UPLO=\"L\", *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_eigh(Tensor self, str UPLO="L") -> (Tensor eigenvalues, Tensor eigenvectors)
    
    auto dispatch_linalg_eigh = [](const Tensor & self, std::string UPLO) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigh(self, UPLO);
    };
    return wrap(&NamedTuple, dispatch_linalg_eigh(_r.tensor(0), _r.string(1)));
  } else {
    // aten::linalg_eigh.eigvals(Tensor self, str UPLO="L", *, Tensor(a!) eigvals, Tensor(b!) eigvecs) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_linalg_eigh_out = [](Tensor & eigvals, Tensor & eigvecs, const Tensor & self, std::string UPLO) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigh_out(eigvals, eigvecs, self, UPLO);
    };
    return wrap(&NamedTuple1, dispatch_linalg_eigh_out(out[0], out[1], _r.tensor(0), _r.string(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_eigvalsh
static PyObject * THPVariable_linalg_eigvalsh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_eigvalsh(Tensor input, std::string UPLO=\"L\", *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_eigvalsh(Tensor self, str UPLO="L") -> Tensor
    
    auto dispatch_linalg_eigvalsh = [](const Tensor & self, std::string UPLO) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigvalsh(self, UPLO);
    };
    return wrap(dispatch_linalg_eigvalsh(_r.tensor(0), _r.string(1)));
  } else {
    // aten::linalg_eigvalsh.out(Tensor self, str UPLO='L', *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_eigvalsh_out = [](Tensor out, const Tensor & self, std::string UPLO) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigvalsh_out(out, self, UPLO);
    };
    return wrap(dispatch_linalg_eigvalsh_out(_r.tensor(2), _r.tensor(0), _r.string(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_householder_product
static PyObject * THPVariable_linalg_householder_product(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_householder_product(Tensor input, Tensor tau, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_householder_product(Tensor input, Tensor tau) -> Tensor
    
    auto dispatch_linalg_householder_product = [](const Tensor & input, const Tensor & tau) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_householder_product(input, tau);
    };
    return wrap(dispatch_linalg_householder_product(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::linalg_householder_product.out(Tensor input, Tensor tau, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_householder_product_out = [](Tensor out, const Tensor & input, const Tensor & tau) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_householder_product_out(out, input, tau);
    };
    return wrap(dispatch_linalg_householder_product_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_inv
static PyObject * THPVariable_linalg_inv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_inv(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_inv(Tensor self) -> Tensor
    
    auto dispatch_linalg_inv = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_inv(self);
    };
    return wrap(dispatch_linalg_inv(_r.tensor(0)));
  } else {
    // aten::linalg_inv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_inv_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_inv_out(out, self);
    };
    return wrap(dispatch_linalg_inv_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_lstsq
static PyObject * THPVariable_linalg_lstsq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"solution", ""}, {"residuals", ""}, {"rank", ""}, {"singular_values", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_lstsq", nullptr, NamedTuple_fields, 4 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_lstsq(Tensor input, Tensor b, double? cond=None, *, std::string? driver=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  // aten::linalg_lstsq(Tensor self, Tensor b, float? cond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
  
  auto dispatch_linalg_lstsq = [](const Tensor & self, const Tensor & b, c10::optional<double> cond, c10::optional<std::string> driver) -> std::tuple<Tensor,Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::linalg_lstsq(self, b, cond, driver);
  };
  return wrap(&NamedTuple, dispatch_linalg_lstsq(_r.tensor(0), _r.tensor(1), _r.toDoubleOptional(2), _r.stringOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_matrix_power
static PyObject * THPVariable_linalg_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_matrix_power(Tensor input, int64_t n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_matrix_power(Tensor self, int n) -> Tensor
    
    auto dispatch_linalg_matrix_power = [](const Tensor & self, int64_t n) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_matrix_power(self, n);
    };
    return wrap(dispatch_linalg_matrix_power(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::linalg_matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_matrix_power_out = [](Tensor out, const Tensor & self, int64_t n) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_matrix_power_out(out, self, n);
    };
    return wrap(dispatch_linalg_matrix_power_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_matrix_rank
static PyObject * THPVariable_linalg_matrix_rank(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_matrix_rank(Tensor input, double? tol=None, bool hermitian=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_matrix_rank(Tensor self, float? tol=None, bool hermitian=False) -> Tensor
    
    auto dispatch_linalg_matrix_rank = [](const Tensor & self, c10::optional<double> tol, bool hermitian) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_matrix_rank(self, tol, hermitian);
    };
    return wrap(dispatch_linalg_matrix_rank(_r.tensor(0), _r.toDoubleOptional(1), _r.toBool(2)));
  } else {
    // aten::linalg_matrix_rank.out(Tensor self, float? tol=None, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_matrix_rank_out = [](Tensor out, const Tensor & self, c10::optional<double> tol, bool hermitian) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_matrix_rank_out(out, self, tol, hermitian);
    };
    return wrap(dispatch_linalg_matrix_rank_out(_r.tensor(3), _r.tensor(0), _r.toDoubleOptional(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_norm
static PyObject * THPVariable_linalg_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_norm(Tensor input, Scalar? ord=None, IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
    "linalg_norm(Tensor input, std::string ord, IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_linalg_norm = [](const Tensor & self, const c10::optional<Scalar> & ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm(self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm(_r.tensor(0), _r.scalarOptional(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      } else {
        // aten::linalg_norm.out(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_norm_out = [](Tensor out, const Tensor & self, const c10::optional<Scalar> & ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm_out(out, self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm_out(_r.tensor(5), _r.tensor(0), _r.scalarOptional(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      }
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_linalg_norm = [](const Tensor & self, std::string ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm(self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm(_r.tensor(0), _r.string(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      } else {
        // aten::linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_norm_out = [](Tensor out, const Tensor & self, std::string ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm_out(out, self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm_out(_r.tensor(5), _r.tensor(0), _r.string(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_pinv
static PyObject * THPVariable_linalg_pinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_pinv(Tensor input, Tensor rcond, bool hermitian=False, *, Tensor out=None)",
    "linalg_pinv(Tensor input, double rcond=1e-15, bool hermitian=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::linalg_pinv.rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_pinv = [](const Tensor & self, const Tensor & rcond, bool hermitian) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv(self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
      } else {
        // aten::linalg_pinv.out_rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_pinv_out = [](Tensor out, const Tensor & self, const Tensor & rcond, bool hermitian) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv_out(out, self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toBool(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::linalg_pinv(Tensor self, float rcond=1e-15, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_pinv = [](const Tensor & self, double rcond, bool hermitian) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv(self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
      } else {
        // aten::linalg_pinv.out(Tensor self, float rcond=1e-15, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_pinv_out = [](Tensor out, const Tensor & self, double rcond, bool hermitian) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv_out(out, self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv_out(_r.tensor(3), _r.tensor(0), _r.toDouble(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_qr
static PyObject * THPVariable_linalg_qr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"Q", ""}, {"R", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_qr", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_qr_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_qr(Tensor input, std::string mode=\"reduced\", *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_qr(Tensor self, str mode='reduced') -> (Tensor Q, Tensor R)
    
    auto dispatch_linalg_qr = [](const Tensor & self, std::string mode) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_qr(self, mode);
    };
    return wrap(&NamedTuple, dispatch_linalg_qr(_r.tensor(0), _r.string(1)));
  } else {
    // aten::linalg_qr.out(Tensor self, str mode='reduced', *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_linalg_qr_out = [](Tensor & Q, Tensor & R, const Tensor & self, std::string mode) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_qr_out(Q, R, self, mode);
    };
    return wrap(&NamedTuple1, dispatch_linalg_qr_out(out[0], out[1], _r.tensor(0), _r.string(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_slogdet
static PyObject * THPVariable_linalg_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"sign", ""}, {"logabsdet", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_slogdet", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_slogdet_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_slogdet(Tensor input, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
    
    auto dispatch_linalg_slogdet = [](const Tensor & self) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_slogdet(self);
    };
    return wrap(&NamedTuple, dispatch_linalg_slogdet(_r.tensor(0)));
  } else {
    // aten::linalg_slogdet.out(Tensor self, *, Tensor(a!) sign, Tensor(b!) logabsdet) -> (Tensor(a!) sign, Tensor(b!) logabsdet)
    auto out = _r.tensorlist_n<2>(1);
    auto dispatch_linalg_slogdet_out = [](Tensor & sign, Tensor & logabsdet, const Tensor & self) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_slogdet_out(sign, logabsdet, self);
    };
    return wrap(&NamedTuple1, dispatch_linalg_slogdet_out(out[0], out[1], _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_solve
static PyObject * THPVariable_linalg_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_solve(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_solve(Tensor input, Tensor other) -> Tensor
    
    auto dispatch_linalg_solve = [](const Tensor & input, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_solve(input, other);
    };
    return wrap(dispatch_linalg_solve(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::linalg_solve.out(Tensor input, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_solve_out = [](Tensor out, const Tensor & input, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_solve_out(out, input, other);
    };
    return wrap(dispatch_linalg_solve_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_svd
static PyObject * THPVariable_linalg_svd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"U", ""}, {"S", ""}, {"V", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_svd_out", nullptr, NamedTuple_fields, 3 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_svd", nullptr, NamedTuple_fields, 3 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_svd(Tensor input, bool full_matrices=True, bool compute_uv=True, *, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_svd(Tensor self, bool full_matrices=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
    
    auto dispatch_linalg_svd = [](const Tensor & self, bool full_matrices, bool compute_uv) -> std::tuple<Tensor,Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_svd(self, full_matrices, compute_uv);
    };
    return wrap(&NamedTuple1, dispatch_linalg_svd(_r.tensor(0), _r.toBool(1), _r.toBool(2)));
  } else {
    // aten::linalg_svd.U(Tensor self, bool full_matrices=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)
    auto out = _r.tensorlist_n<3>(3);
    auto dispatch_linalg_svd_out = [](Tensor & U, Tensor & S, Tensor & V, const Tensor & self, bool full_matrices, bool compute_uv) -> std::tuple<Tensor,Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_svd_out(U, S, V, self, full_matrices, compute_uv);
    };
    return wrap(&NamedTuple, dispatch_linalg_svd_out(out[0], out[1], out[2], _r.tensor(0), _r.toBool(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_tensorinv
static PyObject * THPVariable_linalg_tensorinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_tensorinv(Tensor input, int64_t ind=2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_tensorinv(Tensor self, int ind=2) -> Tensor
    
    auto dispatch_linalg_tensorinv = [](const Tensor & self, int64_t ind) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorinv(self, ind);
    };
    return wrap(dispatch_linalg_tensorinv(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::linalg_tensorinv.out(Tensor self, int ind=2, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_tensorinv_out = [](Tensor out, const Tensor & self, int64_t ind) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorinv_out(out, self, ind);
    };
    return wrap(dispatch_linalg_tensorinv_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_tensorsolve
static PyObject * THPVariable_linalg_tensorsolve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_tensorsolve(Tensor input, Tensor other, IntArrayRef? dims=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_tensorsolve(Tensor self, Tensor other, int[]? dims=None) -> Tensor
    
    auto dispatch_linalg_tensorsolve = [](const Tensor & self, const Tensor & other, c10::optional<IntArrayRef> dims) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorsolve(self, other, dims);
    };
    return wrap(dispatch_linalg_tensorsolve(_r.tensor(0), _r.tensor(1), _r.intlistOptional(2)));
  } else {
    // aten::linalg_tensorsolve.out(Tensor self, Tensor other, int[]? dims=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_tensorsolve_out = [](Tensor out, const Tensor & self, const Tensor & other, c10::optional<IntArrayRef> dims) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorsolve_out(out, self, other, dims);
    };
    return wrap(dispatch_linalg_tensorsolve_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.intlistOptional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_vector_norm
static PyObject * THPVariable_linalg_vector_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_vector_norm(Tensor input, Scalar? ord=None, IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(5)) {
    // aten::linalg_vector_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    
    auto dispatch_linalg_vector_norm = [](const Tensor & self, const c10::optional<Scalar> & ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_vector_norm(self, ord, dim, keepdim, dtype);
    };
    return wrap(dispatch_linalg_vector_norm(_r.tensor(0), _r.scalarOptional(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
  } else {
    // aten::linalg_vector_norm.out(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_vector_norm_out = [](Tensor out, const Tensor & self, const c10::optional<Scalar> & ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_vector_norm_out(out, self, ord, dim, keepdim, dtype);
    };
    return wrap(dispatch_linalg_vector_norm_out(_r.tensor(5), _r.tensor(0), _r.scalarOptional(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

}} // namespace torch::autograd
