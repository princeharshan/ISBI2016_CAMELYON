#include <iostream>
#include "numpy_boost_python.hpp"
#include "numpy_boost.hpp"

/*
  This demonstrates how to use numpy_boost with boost.python.

 */

using namespace boost::python;

numpy_boost<uint64_t, 2> compute(const numpy_boost<uint64_t, 2>& array)
{
 // // Write out the array contents using [] syntax
 // std::cout << "C++" << std::endl;
 // for (size_t i = 0; i < array.shape()[0]; ++i) {
 //   std::cout << "[";
 //   for (size_t j = 0; j < array.shape()[1]; ++j) {
 //     std::cout << "[";
 //     for (size_t k = 0; k < array.shape()[2]; ++k) {
 //       std::cout << array[i][j][k] << " ";
 //     }
 //     std::cout << "]";
 //   }
 //   std::cout << "]";
 // }
 // std::cout << std::endl;

 // // Create a new numpy_boost array (with a new Numpy array to back
 // // its memory), it fill it with some contents.
 // int dims[] = { 32, 64 };
 // numpy_boost<int, 2> array2(dims);

 // for (size_t i = 0; i < array2.shape()[0]; ++i) {
 //   for (size_t j = 0; j < array2.shape()[1]; ++j) {
 //     array2[i][j] = i * j;
 //   }
 // }

 // return array2;
  int rows = array.shape()[0];
  int cols = array.shape()[1];
  int dims[] = {rows-256,cols-256};
  numpy_boost<uint64_t, 2> array2(dims);
  for (size_t i = 256; i < rows; ++i)
  {
    for (size_t j = 256; j < cols; ++j)
    {
      array2[i-256][j-256] = array[i][j]+array[i-256][j-256]-array[i][j-256]-array[i-256][j];
    }
  }

  return array2;
}


BOOST_PYTHON_MODULE(numpy_boost_camelyon)
{
  /* Initialize the Numpy support */
  import_array();

  /* You must call this function inside of the BOOST_PYTHON_MODULE
     init function to set up the type conversions.  It must be called
     for every type and dimensionality of array you intend to return.

     If you don't, the code will compile, but you will get error
     messages at runtime.
   */
  numpy_boost_python_register_type<uint64_t, 2>();
  //numpy_boost_python_register_type<double, 3>();

  /* Declare a function that takes and returns an array */
  def("compute", compute);
}
