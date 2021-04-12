// @generated from tools/autograd/templates/python_special_functions.cpp

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_special_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/out_types.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

#include <ATen/ATen.h>

using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;

using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

// generated forward declarations start here

static PyObject * THPVariable_special_entr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_erf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_erfc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_erfinv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_gammaln(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef special_functions[] = {
  {"special_entr", castPyCFunctionWithKeywords(THPVariable_special_entr), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_erf", castPyCFunctionWithKeywords(THPVariable_special_erf), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_erfc", castPyCFunctionWithKeywords(THPVariable_special_erfc), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_erfinv", castPyCFunctionWithKeywords(THPVariable_special_erfinv), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_gammaln", castPyCFunctionWithKeywords(THPVariable_special_gammaln), METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL}
};

static PyObject* THPSpecialVariableFunctionsModule = NULL;

void initSpecialFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._special",
     NULL,
     -1,
     special_functions
  };
  PyObject* special = PyModule_Create(&def);
  THPSpecialVariableFunctionsModule = special;
  if (!special) {
    throw python_error();
  }
  // steals a reference to special
  if (PyModule_AddObject(module, "_special", special) != 0) {
    throw python_error();
  }
}

// generated methods start here

// special_entr
static PyObject * THPVariable_special_entr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_entr(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_entr(Tensor self) -> Tensor
    
    auto dispatch_special_entr = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_entr(self);
    };
    return wrap(dispatch_special_entr(_r.tensor(0)));
  } else {
    // aten::special_entr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_entr_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_entr_out(out, self);
    };
    return wrap(dispatch_special_entr_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_erf
static PyObject * THPVariable_special_erf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_erf(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_erf(Tensor self) -> Tensor
    
    auto dispatch_special_erf = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erf(self);
    };
    return wrap(dispatch_special_erf(_r.tensor(0)));
  } else {
    // aten::special_erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_erf_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erf_out(out, self);
    };
    return wrap(dispatch_special_erf_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_erfc
static PyObject * THPVariable_special_erfc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_erfc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_erfc(Tensor self) -> Tensor
    
    auto dispatch_special_erfc = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfc(self);
    };
    return wrap(dispatch_special_erfc(_r.tensor(0)));
  } else {
    // aten::special_erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_erfc_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfc_out(out, self);
    };
    return wrap(dispatch_special_erfc_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_erfinv
static PyObject * THPVariable_special_erfinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_erfinv(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_erfinv(Tensor self) -> Tensor
    
    auto dispatch_special_erfinv = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfinv(self);
    };
    return wrap(dispatch_special_erfinv(_r.tensor(0)));
  } else {
    // aten::special_erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_erfinv_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfinv_out(out, self);
    };
    return wrap(dispatch_special_erfinv_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_gammaln
static PyObject * THPVariable_special_gammaln(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_gammaln(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_gammaln(Tensor self) -> Tensor
    
    auto dispatch_special_gammaln = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_gammaln(self);
    };
    return wrap(dispatch_special_gammaln(_r.tensor(0)));
  } else {
    // aten::special_gammaln.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_gammaln_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_gammaln_out(out, self);
    };
    return wrap(dispatch_special_gammaln_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

}} // namespace torch::autograd
