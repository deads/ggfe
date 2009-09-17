#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <sstream>
#include <cvd/image.h>
#include <cvd/image_io.h>
#include <cvd/vector_image_ref.h>
#include <cvd/convolution.h>
#include <cvd/integral_image.h>
#include "viola_jones.h"
#include <PyCPP/PyCVD.hpp>

using namespace std;
using namespace CVD;
using namespace PyCPP;

extern "C" {

  PyObject *compute_integral_image_wrap(PyObject *self, PyObject *args) {
    PyArrayObject *_in, *_out;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &_in)) {
      return 0;
    }
    try {
      int height = rows(_in);
      int width = cols(_in);
      BasicImage<float> in(0, ImageRef(width, height));
      convert(_in, in);
      BiAllocator<BasicImage<float>, PyArrayObject*, float, std::pair<int, int> > bout(std::pair<int, int>(width, height));
      _out = bout.getSecond();
      BasicImage<float> out(bout.getFirst());
      integral_image(in, out);
    }
    catch(string err) {
      PyErr_SetString(PyExc_RuntimeError, err.c_str());
      return 0;
    }
    return (PyObject*)_out;
  }

  PyObject *apply_kernel_to_image_wrap(PyObject *self, PyObject *args) {
    PyArrayObject *_in, *_out;
    const char *spec;

    if (!PyArg_ParseTuple(args, "O!s", &PyArray_Type, &_in, &spec)) {
      return 0;
    }

    try {
      int height = rows(_in);
      int width = cols(_in);

      BasicImage<float> in(0, ImageRef(width, height));
      convert(_in, in);

      BiAllocator<BasicImage<float>, PyArrayObject*, float, std::pair<int, int> > bout(std::pair<int, int>(width, height));
      _out = bout.getSecond();
      BasicImage<float> out(bout.getFirst());

      istringstream iss(spec);
      Kernel kernel;
      iss >> kernel;
      out.fill(0.0);

      apply_kernel(in, kernel, out);
    }
    catch(string err) {
      PyErr_SetString(PyExc_RuntimeError, err.c_str());
      return 0;
    }
    return (PyObject*)_out;
  }

  static PyMethodDef _ggfe_image_wrap_methods[] = {
    {"compute_integral_image_wrap", compute_integral_image_wrap, METH_VARARGS},
    {"apply_kernel_to_image_wrap", apply_kernel_to_image_wrap, METH_VARARGS},
    {NULL, NULL}
  };

  void init_ggfe_image_wrap()  {
    (void) Py_InitModule("_ggfe_image_wrap", _ggfe_image_wrap_methods);
    import_array();  // Must be present for NumPy.  Called first after above line.
  }

}
