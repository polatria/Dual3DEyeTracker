#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "tracker.h"
#include <numpy/arrayobject.h>
#include "converter.h"

#ifdef PYTHONLIB

static void py_tracker_free(PyObject *obj) {
	Tracker* const d = (Tracker*)PyCapsule_GetPointer(obj, "_Tracker");
	delete d;
}

static PyObject* py_tracker_alloc(PyObject *self, PyObject *args) {
	Tracker* const h = new Tracker(2);
	return PyCapsule_New(h, "_Tracker", py_tracker_free);
}

static PyObject* py_detect(PyObject* self, PyObject *args) {
	PyObject *py_obj, *image;
	int cam_num, key;

	if (!PyArg_ParseTuple(args, "OOii", &py_obj, &image, &cam_num, &key))
		Py_RETURN_NONE;
	//std::cout << "CAM_NUM:" << cam_num << std::endl;
	//std::cout << "Key:" << key << std::endl;

	// Get Tracker class pointer
	auto trk = (Tracker*)PyCapsule_GetPointer(py_obj, "_Tracker");
	if (trk == NULL) Py_RETURN_NONE;

	// Restore image array
	cv::Mat img = pbcvt::fromNDArrayToMat(image);
	
	//std::ostringstream ss;
	//ss << "Test" << cam_num;
	//string winname = ss.str();
	//cv::imshow(winname, img);

	trk->detect(img, cam_num, key);
	Py_RETURN_NONE;
}

static PyObject* py_example(PyObject *self, PyObject *args) {
	PyObject *py_obj, *py_list;
	int w, h;

	if (!PyArg_ParseTuple(args, "OOii", &py_obj, &py_list, &w, &h))
		Py_RETURN_NONE;

	auto trk = (Tracker*)PyCapsule_GetPointer(py_obj, "_Tracker");
	if (trk == NULL) Py_RETURN_NONE;
	auto carr = (int*)malloc((size_t)sizeof(int) * w * h);
	for (auto i = 0; i < w * h; i++) {
		auto item = PyList_GetItem(py_list, i);
		auto value = PyLong_AsLong(item);
		carr[i] = (int)value;
	}
	auto succ = trk->dummy(w, h, carr);
	free(carr);
	return Py_BuildValue("i", succ);
}


static PyMethodDef methods[] = {
	// The first property is the name exposed to Python, fast_tanh, the second is the C++
	// function name that contains the implementation.
	{"alloc", py_tracker_alloc, METH_VARARGS, nullptr},
	{"list_ex", py_example, METH_VARARGS, nullptr},
	{"detect", py_detect, METH_VARARGS, nullptr},

	// Terminate the array with an object containing nulls.
	{nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef detectormodule = {
	PyModuleDef_HEAD_INIT,
	"EyeDetector",			// Module name to use with Python import statements
	NULL,					// Module description, may be NULL
	-1,						/* size of per-interpreter state of the module,
							or -1 if the module keeps state in global variables. */
	methods					// Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_EyeDetector(void) {
	import_array();
	return PyModule_Create(&detectormodule);
}

#endif