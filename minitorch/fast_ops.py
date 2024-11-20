from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Check if the output and input tensors are stride-aligned and have identical shapes
        if (out_strides == in_strides).all() and (out_shape == in_shape).all() and len(out_shape) == len(in_shape):
            for ord in prange(len(out)):
                out[ord] = fn(in_storage[ord])
            return
       
        # General case: Handle broadcasting and indexing for misaligned or differently shaped tensors.
        for out_ord in prange(len(out)):
            # Convert the flat index in the output tensor to a multi-dimensional index
            out_index = out_shape.copy()
            to_index(out_ord, out_shape, out_index)

            # Broadcast the output index to the corresponding input index
            in_index = in_shape.copy()
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Convert the input multi-dimensional index back to a flat index
            in_ord = index_to_position(in_index, in_strides)

            # Apply the function to the input value and store the result in the output tensor.
            out[out_ord] = fn(in_storage[in_ord])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Check if output, a, and b tensors are aligned in strides and have identical shapes.
        if (
            len(out_shape) == len(a_shape) and len(out_shape) == len(b_shape) and
            (out_strides == a_strides).all() and (out_strides == b_strides).all() and
            (out_shape == a_shape).all() and (out_shape == b_shape).all()
        ):
            for ord in prange(len(out)):
                out[ord] = fn(a_storage[ord], b_storage[ord])
            return
        
        # General case: Handle broadcasting and indexing for misaligned or differently shaped tensors
        for out_ord in prange(len(out)):
            # Convert the flat index in the output tensor to a multi-dimensional index
            out_index = out_shape.copy()
            to_index(out_ord, out_shape, out_index)

            # Broadcast the output index to corresponding indices in input tensors a and b
            a_index = a_shape.copy()
            b_index = b_shape.copy()
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Convert the multi-dimensional indices for a and b back to flat indices
            a_ord = index_to_position(a_index, a_strides)
            b_ord = index_to_position(b_index, b_strides)

            # Apply the function to elements from a and b, and store the result in the output tensor
            out[out_ord] = fn(a_storage[a_ord], b_storage[b_ord])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # Iterate over all elements of the output tensor in parallel
        for ord in prange(len(out)):
            # Create an index for the current element in the output tensor
            out_index: Index = np.zeros_like(out_shape, dtype=np.int32)
            to_index(ord, out_shape, out_index)

            # Map the output index to the corresponding position in the input tensor
            a_ord = index_to_position(out_index, a_strides)

            # Initialize the reduction value with the current value in the output storage
            reduce_value = out[ord]

            # Perform the reduction along the specified dimension
            for i in prange(a_shape[reduce_dim]):
                reduce_value = fn(reduce_value, a_storage[a_ord + i * a_strides[reduce_dim]])

            # Store the final reduction result in the output tensor
            out[ord] = reduce_value
        

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Outer loops over the batch dimension, rows of `a`, and columns of `b`
    for i in prange(out_shape[0]):  # Iterate over the batch size
        for j in prange(out_shape[1]):  # Iterate over the rows of the output
            for k in prange(out_shape[2]):  # Iterate over the columns of the output
                # Calculate the initial strides for the current batch and position
                a_inner_stride = i * a_batch_stride + j * a_strides[1]
                b_inner_stride = j * b_batch_stride + k * b_strides[2]

                # Initialize the output value for the current position
                val = 0.0

                # Perform the dot product between the row of `a` and the column of `b`
                for l in prange(a_shape[2]):  # Iterate over the shared dimension
                    val += a_storage[a_inner_stride] * b_storage[b_inner_stride]
                    # Update the inner strides for the next element
                    a_inner_stride += a_strides[2]
                    b_inner_stride += b_strides[1]

                # Write the result to the output storage at the correct position
                out[i * out_strides[0] + j * out_strides[1] + k * out_strides[2]] = val

tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
