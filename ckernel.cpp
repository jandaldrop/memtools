#include <iostream>

#include "Python.h"
#include "numpy/arrayobject.h"

double w(int j,int i)
{
  if (j==0 || j==i)
    return 0.5;
  else
    return 1.;
}

static PyObject*
ckernel_core (PyObject *dummy, PyObject *args)
{
    // kernel is the output argument. Prepare all arrays in the right size!
    PyObject *v_acf_arg=NULL, *va_cf_arg=NULL, *f_acf_arg=NULL, *au_cf_arg=NULL, *out=NULL;
    PyArrayObject *v_acf_ar=NULL, *va_cf_ar=NULL, *f_acf_ar=NULL, *au_cf_ar=NULL, *kernel_ar=NULL;

    double dt, k0;

    if (!PyArg_ParseTuple(args, "OOOOddO!", &v_acf_arg, &va_cf_arg, &f_acf_arg, &au_cf_arg, &dt, &k0,
        &PyArray_Type, &out)) return NULL;



    v_acf_ar = (PyArrayObject*)PyArray_FROM_OTF(v_acf_arg, NPY_DOUBLE, NPY_IN_ARRAY);
    if (v_acf_ar== NULL) return NULL;
    va_cf_ar = (PyArrayObject*)PyArray_FROM_OTF(va_cf_arg, NPY_DOUBLE, NPY_IN_ARRAY);
    f_acf_ar = (PyArrayObject*)PyArray_FROM_OTF(f_acf_arg, NPY_DOUBLE, NPY_IN_ARRAY);
    au_cf_ar = (PyArrayObject*)PyArray_FROM_OTF(au_cf_arg, NPY_DOUBLE, NPY_IN_ARRAY);
    kernel_ar = (PyArrayObject*)PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_INOUT_ARRAY);

    if (kernel_ar == NULL || f_acf_ar==NULL || va_cf_ar==NULL)
    {
      Py_XDECREF(v_acf_ar);
      Py_XDECREF(va_cf_ar);
      Py_XDECREF(f_acf_ar);
      Py_XDECREF(au_cf_ar);
      PyArray_XDECREF_ERR(kernel_ar);
      return NULL;
    }

    // do some assertions.
    if (!((kernel_ar->nd == 1) &&
    (v_acf_ar->nd == 1) &&
    (va_cf_ar->nd == 1) &&
    (f_acf_ar->nd == 1) &&
    (au_cf_ar->nd == 1) &&

    (kernel_ar->nd == 1) &&
    (va_cf_ar->dimensions[0] == v_acf_ar->dimensions[0]) &&
    (f_acf_ar->dimensions[0] == v_acf_ar->dimensions[0]) &&
    (au_cf_ar->dimensions[0] == v_acf_ar->dimensions[0]) &&
    (kernel_ar->dimensions[0] <= v_acf_ar->dimensions[0]) ))
    {
      std::cerr << "Size mismatch." << std::endl;
      Py_XDECREF(v_acf_ar);
      Py_XDECREF(va_cf_ar);
      Py_XDECREF(f_acf_ar);
      Py_XDECREF(au_cf_ar);
      PyArray_XDECREF_ERR(kernel_ar);
      return NULL;
    }

    // get c-style arrays for calculation;
    double *kernel = (double*)PyArray_GETPTR2(kernel_ar, 0, 0);
    double *v_acf = (double*)PyArray_GETPTR2(v_acf_ar, 0, 0);
    double *va_cf = (double*)PyArray_GETPTR2(va_cf_ar, 0, 0);
    double *f_acf = (double*)PyArray_GETPTR2(f_acf_ar, 0, 0); // this is really m<aa>
    double *au_cf = (double*)PyArray_GETPTR2(au_cf_ar, 0, 0);

    // do the actual calculation
    double prefac=1./(v_acf[0]+va_cf[0]*dt*w(0,0));
    double num;

    if (k0 == 0)
    {
        kernel[0]=(f_acf[0]+au_cf[0])/v_acf[0];
    }
    else
    {
        kernel[0]=k0;
    }

    for  (int i=1; i < kernel_ar->dimensions[0]; i++)
    {
        num=f_acf[i]+au_cf[i];
        for (int j=0; j < i; j++)
        {
          num-=dt*w(j,i)*va_cf[i-j]*kernel[j];
        }
      kernel[i]=prefac*num;
    }

    Py_DECREF(v_acf_ar);
    Py_DECREF(va_cf_ar);
    Py_DECREF(f_acf_ar);
    Py_DECREF(au_cf_ar);
    Py_DECREF(kernel_ar);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
ckernel_first_order_core (PyObject *dummy, PyObject *args)
{
    // kernel is the output argument. Prepare all arrays in the right size!
    PyObject *v_acf_arg=NULL, *vf_cf_arg=NULL, *vu_cf_arg=NULL, *f_acf_arg=NULL, *au_cf_arg=NULL, *out=NULL;
    PyArrayObject *v_acf_ar=NULL, *vf_cf_ar=NULL, *vu_cf_ar=NULL, *f_acf_ar=NULL, *au_cf_ar=NULL, *kernel_ar=NULL;

    double dt, k0;

    if (!PyArg_ParseTuple(args, "OOOOOddO!", &v_acf_arg, &vf_cf_arg, &f_acf_arg, &vu_cf_arg, &au_cf_arg, &dt, &k0,
        &PyArray_Type, &out)) return NULL;



    v_acf_ar = (PyArrayObject*)PyArray_FROM_OTF(v_acf_arg, NPY_DOUBLE, NPY_IN_ARRAY);
    if (v_acf_ar== NULL) return NULL;
    vf_cf_ar = (PyArrayObject*)PyArray_FROM_OTF(vf_cf_arg, NPY_DOUBLE, NPY_IN_ARRAY);
    f_acf_ar = (PyArrayObject*)PyArray_FROM_OTF(f_acf_arg, NPY_DOUBLE, NPY_IN_ARRAY);
    vu_cf_ar = (PyArrayObject*)PyArray_FROM_OTF(vu_cf_arg, NPY_DOUBLE, NPY_IN_ARRAY);
    au_cf_ar = (PyArrayObject*)PyArray_FROM_OTF(au_cf_arg, NPY_DOUBLE, NPY_IN_ARRAY);
    kernel_ar = (PyArrayObject*)PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_INOUT_ARRAY);

    if (kernel_ar == NULL || f_acf_ar==NULL || vf_cf_ar==NULL || au_cf_ar==NULL)
    {
      Py_XDECREF(v_acf_ar);
      Py_XDECREF(vf_cf_ar);
      Py_XDECREF(f_acf_ar);
      Py_XDECREF(vu_cf_ar);
      Py_XDECREF(au_cf_ar);
      PyArray_XDECREF_ERR(kernel_ar);
      return NULL;
    }

    // do some assertions.
    if (!((kernel_ar->nd == 1) &&
    (v_acf_ar->nd == 1) &&
    (vf_cf_ar->nd == 1) &&
    (vu_cf_ar->nd == 1) &&
    (au_cf_ar->nd == 1) &&

    (kernel_ar->nd == 1) &&
    (vf_cf_ar->dimensions[0] == v_acf_ar->dimensions[0]) &&
    (f_acf_ar->dimensions[0] == v_acf_ar->dimensions[0]) &&
    (vu_cf_ar->dimensions[0] == v_acf_ar->dimensions[0]) &&
    (au_cf_ar->dimensions[0] == v_acf_ar->dimensions[0]) &&
    (kernel_ar->dimensions[0] <= v_acf_ar->dimensions[0]) ))
    {
      std::cerr << "Size mismatch." << std::endl;
      Py_XDECREF(v_acf_ar);
      Py_XDECREF(vf_cf_ar);
      Py_XDECREF(f_acf_ar);
      Py_XDECREF(vu_cf_ar);
      Py_XDECREF(au_cf_ar);
      PyArray_XDECREF_ERR(kernel_ar);
      return NULL;
    }

    // get c-style arrays for calculation;
    double *kernel = (double*)PyArray_GETPTR2(kernel_ar, 0, 0);
    double *v_acf = (double*)PyArray_GETPTR2(v_acf_ar, 0, 0);
    double *vf_cf = (double*)PyArray_GETPTR2(vf_cf_ar, 0, 0);
    double *f_acf = (double*)PyArray_GETPTR2(f_acf_ar, 0, 0); // this is really m<aa>
    double *vu_cf = (double*)PyArray_GETPTR2(vu_cf_ar, 0, 0);
    double *au_cf = (double*)PyArray_GETPTR2(au_cf_ar, 0, 0);

    // do the actual calculation
    double prefac=1./(v_acf[0]*dt*w(0,0));
    double num;

    if (k0 == 0)
    {
        kernel[0]=(f_acf[0]+au_cf[0])/v_acf[0];
    }
    else
    {
        kernel[0]=k0;
    }

    for  (int i=1; i < kernel_ar->dimensions[0]; i++)
    {
        num=-vf_cf[i]-vu_cf[i];
        for (int j=0; j < i; j++)
        {
          num-=dt*w(j,i)*v_acf[i-j]*kernel[j];
        }
      kernel[i]=prefac*num;
    }

    Py_DECREF(v_acf_ar);
    Py_DECREF(vf_cf_ar);
    Py_DECREF(f_acf_ar);
    Py_DECREF(vu_cf_ar);
    Py_DECREF(au_cf_ar);
    Py_DECREF(kernel_ar);
    Py_INCREF(Py_None);
    return Py_None;
}

static struct PyMethodDef methods[] = {
    {"ckernel_core", ckernel_core, METH_VARARGS, "Calculates the memory kernel iteratively (Volterra equation of the second kind)."},
    {"ckernel_first_order_core", ckernel_first_order_core, METH_VARARGS, "Calculates the memory kernel iteratively (Volterra equation of the first kind)."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initckernel (void)
{
    (void)Py_InitModule("ckernel", methods);
    import_array();
}
