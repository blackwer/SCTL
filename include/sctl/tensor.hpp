#ifndef _SCTL_TENSOR_HPP_
#define _SCTL_TENSOR_HPP_

#include <ostream>                // for ostream
#include <initializer_list>       // for initializer_list

#include "sctl/common.hpp"        // for Long, Integer, sctl
#include "sctl/static-array.hpp"  // for StaticArray

namespace sctl {

  template <Long k, Long d0, Long... dd> static constexpr Long TensorArgExtract();
  template <class ValueType, bool own_data, Long... dd> struct TensorRotateLeftType;
  template <class ValueType, bool own_data, Long... dd> struct TensorRotateRightType;

  /**
   * A template class representing a multidimensional tensor.
   * The data is stored in row-major order.
   *
   * @tparam ValueType The type of the elements stored in the tensor.
   * @tparam own_data A boolean indicating whether the tensor owns its data.
   * @tparam Args The dimensions of the tensor.
   */
  template <class ValueType, bool own_data, Long... Args> class Tensor {
    public:
      /**
       * Get the order of the tensor.
       *
       * @return The order of the tensor.
       */
      static constexpr Long Order();

      /**
       * Get the total number of elements in the tensor.
       *
       * @return The total number of elements in the tensor.
       */
      static constexpr Long Size();

      /**
       * Get the size of a specific dimension of the tensor.
       *
       * @tparam k The index of the dimension.
       * @return The size of the specified dimension.
       */
      template <Long k> static constexpr Long Dim();

      /**
       * A static function to test the functionality of the Tensor class.
       */
      static void test();

      // Constructor and Destructor

      /**
       * Default constructor.
       *
       * Constructs an empty tensor.
       */
      Tensor();

      /**
       * Constructor initializing tensor from an iterator.
       *
       * @param src_iter Iterator pointing to the beginning of data to initialize the tensor.
       */
      explicit Tensor(Iterator<ValueType> src_iter);

      /**
       * Constructor initializing tensor from a const iterator.
       *
       * @param src_iter Const iterator pointing to the beginning of data to initialize the tensor.
       */
      explicit Tensor(ConstIterator<ValueType> src_iter);

      /**
       * Constructor with initializer list.
       *
       * @param arr An initializer list to initialize the tensor.
       */
      explicit Tensor(std::initializer_list<ValueType> arr);

      /**
       * Copy constructor.
       *
       * @param M Another tensor to copy from.
       */
      Tensor(const Tensor &M);

      /**
       * Constructor initializing all elements of the tensor with a specific value.
       *
       * @param v The value to initialize all elements of the tensor with.
       */
      explicit Tensor(const ValueType& v);

      /**
       * Constructor converting tensor of another data type.
       *
       * @tparam own_data_ Boolean indicating whether the new tensor owns its data.
       * @param M Another tensor to copy from.
       */
      template <bool own_data_> Tensor(const Tensor<ValueType, own_data_, Args...> &M);

      /**
       * Destructor.
       */
      ~Tensor() = default;

      // Assignment Operator

      /**
       * Copy assignment operator.
       *
       * @param M Another tensor to copy from.
       * @return Reference to this tensor.
       */
      Tensor &operator=(const Tensor &M);

      /**
       * Assignment operator setting all elements of the tensor to a specific value.
       *
       * @param v The value to set all elements of the tensor to.
       * @return Reference to this tensor.
       */
      Tensor &operator=(const ValueType& v);

      /**
       * Copy assignment operator from another type.
       *
       * @param M Another tensor to copy from.
       * @return Reference to this tensor.
       */
      template <bool own_data_> Tensor &operator=(const Tensor<ValueType, own_data_, Args...> &M);

      // Member Functions

      /**
       * Get an iterator to the beginning of the tensor.
       *
       * @return An iterator to the beginning of the tensor.
       */
      Iterator<ValueType> begin();

      /**
       * Get a const iterator to the beginning of the tensor.
       *
       * @return A const iterator to the beginning of the tensor.
       */
      ConstIterator<ValueType> begin() const;

      /**
       * Get an iterator to the end of the tensor.
       *
       * @return An iterator to the end of the tensor.
       */
      Iterator<ValueType> end();

      /**
       * Get a const iterator to the end of the tensor.
       *
       * @return A const iterator to the end of the tensor.
       */
      ConstIterator<ValueType> end() const;

      /**
       * Access a specific element of the tensor.
       *
       * @tparam PackedLong Variadic template parameter for the indices of the element.
       * @param ii Indices of the element.
       * @return Reference to the element.
       */
      template <class ...PackedLong> ValueType& operator()(PackedLong... ii);

      /**
       * Access a specific element of the tensor.
       *
       * @tparam PackedLong Variadic template parameter for the indices of the element.
       * @param ii Indices of the element.
       * @return Copy of the element.
       */
      template <class ...PackedLong> ValueType operator()(PackedLong... ii) const;

      /**
       * Returns a new tensor obtained by rotating the dimensions of the current tensor to the left.
       * This operation shifts the dimensions of the tensor. If the original tensor has dimensions n1 x n2 x n3,
       * then the left-rotated tensor will be of dimensions n2 x n3 x n1 and the data will be rearranged in a similar way.
       * This operation is a generalization of the transpose operation for matrices.
       *
       * @return A new tensor with dimensions rotated to the left.
       */
      typename TensorRotateLeftType<ValueType, true, Args...>::Value RotateLeft() const;

      /**
       * Returns a new tensor obtained by rotating the dimensions of the current tensor to the right.
       * This operation shifts the dimensions of the tensor. If the original tensor has dimensions n1 x n2 x n3,
       * then the right-rotated tensor will be of dimensions n3 x n1 x n2 and the data will be rearranged accordingly.
       * This operation is a generalization of the transpose operation for matrices.
       *
       * @return A new tensor with dimensions rotated to the right.
       */
      typename TensorRotateRightType<ValueType, true, Args...>::Value RotateRight() const;

      /**
       * Unary positive operator.
       *
       * @return The tensor itself.
       */
      Tensor<ValueType, true, Args...> operator+() const;

      /**
       * Unary negative operator.
       *
       * @return Negated tensor.
       */
      Tensor<ValueType, true, Args...> operator-() const;

      /**
       * Addition operator.
       *
       * @param s Scalar value to add to the tensor.
       * @return Result of addition.
       */
      Tensor<ValueType, true, Args...> operator+(const ValueType &s) const;

      /**
       * Subtraction operator.
       *
       * @param s Scalar value to subtract from the tensor.
       * @return Result of subtraction.
       */
      Tensor<ValueType, true, Args...> operator-(const ValueType &s) const;

      /**
       * Multiplication operator.
       *
       * @param s Scalar value to multiply the tensor by.
       * @return Result of multiplication.
       */
      Tensor<ValueType, true, Args...> operator*(const ValueType &s) const;

      /**
       * Division operator.
       *
       * @param s Scalar value to divide the tensor by.
       * @return Result of division.
       */
      Tensor<ValueType, true, Args...> operator/(const ValueType &s) const;

      /**
       * Addition operator.
       *
       * @tparam own_data_ Boolean indicating whether the resulting tensor owns its data.
       * @param M2 Another tensor to add.
       * @return Result of addition.
       */
      template <bool own_data_> Tensor<ValueType, true, Args...> operator+(const Tensor<ValueType, own_data_, Args...> &M2) const;

      /**
       * Subtraction operator.
       *
       * @tparam own_data_ Boolean indicating whether the resulting tensor owns its data.
       * @param M2 Another tensor to subtract.
       * @return Result of subtraction.
       */
      template <bool own_data_> Tensor<ValueType, true, Args...> operator-(const Tensor<ValueType, own_data_, Args...> &M2) const;

      /**
       * Multiplication operator.
       *
       * This is matrix multiplication. The second tensor should have dimensions compatible with the first.
       *
       * @tparam own_data_ Boolean indicating whether the resulting tensor owns its data.
       * @tparam N1 The size of the second dimension of the first tensor.
       * @tparam N2 The size of the second dimension of the second tensor.
       * @param M2 Another tensor to multiply.
       * @return Result of multiplication.
       */
      template <bool own_data_, Long N1, Long N2> Tensor<ValueType, true, TensorArgExtract<0, Args...>(), N2> operator*(const Tensor<ValueType, own_data_, N1, N2> &M2) const;

    private:

      template <Integer k> static Long offset();

      template <Integer k, class ...PackedLong> static Long offset(Long i, PackedLong... ii);

      void Init(Iterator<ValueType> src_iter);

      StaticArray<ValueType, own_data ? Size() : 0> buff;
      StaticArray<Iterator<ValueType>, own_data ? 0 : 1> iter_;
  };

  template <class ValueType, bool own_data, Long N1, Long N2> std::ostream& operator<<(std::ostream &output, const Tensor<ValueType, own_data, N1, N2> &M);

}  // end namespace

#endif // _SCTL_TENSOR_HPP_
