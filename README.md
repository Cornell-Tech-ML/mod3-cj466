# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Task 3.1 & 3.2Parallel Check Diagnostics Output
```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/ivanjie/workspace/mod3-cj466/minitorch/fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/ivanjie/workspace/mod3-cj466/minitorch/fast_ops.py (163)
----------------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                             |
        out: Storage,                                                                                     |
        out_shape: Shape,                                                                                 |
        out_strides: Strides,                                                                             |
        in_storage: Storage,                                                                              |
        in_shape: Shape,                                                                                  |
        in_strides: Strides,                                                                              |
    ) -> None:                                                                                            |
        # Check if the output and input tensors are stride-aligned and have identical shapes              |
        if (                                                                                              |
            (out_strides == in_strides).all()-------------------------------------------------------------| #0
            and (out_shape == in_shape).all()-------------------------------------------------------------| #1
            and len(out_shape) == len(in_shape)                                                           |
        ):                                                                                                |
            for ord in prange(len(out)):------------------------------------------------------------------| #2
                out[ord] = fn(in_storage[ord])                                                            |
            return                                                                                        |
                                                                                                          |
        # General case: Handle broadcasting and indexing for misaligned or differently shaped tensors.    |
        for out_ord in prange(len(out)):------------------------------------------------------------------| #3
            # Convert the flat index in the output tensor to a multi-dimensional index                    |
            out_index = out_shape.copy()                                                                  |
            to_index(out_ord, out_shape, out_index)                                                       |
                                                                                                          |
            # Broadcast the output index to the corresponding input index                                 |
            in_index = in_shape.copy()                                                                    |
            broadcast_index(out_index, out_shape, in_shape, in_index)                                     |
                                                                                                          |
            # Convert the input multi-dimensional index back to a flat index                              |
            in_ord = index_to_position(in_index, in_strides)                                              |
                                                                                                          |
            # Apply the function to the input value and store the result in the output tensor.            |
            out[out_ord] = fn(in_storage[in_ord])                                                         |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/ivanjie/workspace/mod3-cj466/minitorch/fast_ops.py (223)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/ivanjie/workspace/mod3-cj466/minitorch/fast_ops.py (223)
---------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                            |
        out: Storage,                                                                                    |
        out_shape: Shape,                                                                                |
        out_strides: Strides,                                                                            |
        a_storage: Storage,                                                                              |
        a_shape: Shape,                                                                                  |
        a_strides: Strides,                                                                              |
        b_storage: Storage,                                                                              |
        b_shape: Shape,                                                                                  |
        b_strides: Strides,                                                                              |
    ) -> None:                                                                                           |
        # Check if output, a, and b tensors are aligned in strides and have identical shapes.            |
        if (                                                                                             |
            len(out_shape) == len(a_shape)                                                               |
            and len(out_shape) == len(b_shape)                                                           |
            and (out_strides == a_strides).all()---------------------------------------------------------| #4
            and (out_strides == b_strides).all()---------------------------------------------------------| #5
            and (out_shape == a_shape).all()-------------------------------------------------------------| #6
            and (out_shape == b_shape).all()-------------------------------------------------------------| #7
        ):                                                                                               |
            for ord in prange(len(out)):-----------------------------------------------------------------| #8
                out[ord] = fn(a_storage[ord], b_storage[ord])                                            |
            return                                                                                       |
                                                                                                         |
        # General case: Handle broadcasting and indexing for misaligned or differently shaped tensors    |
        for out_ord in prange(len(out)):-----------------------------------------------------------------| #9
            # Convert the flat index in the output tensor to a multi-dimensional index                   |
            out_index = out_shape.copy()                                                                 |
            to_index(out_ord, out_shape, out_index)                                                      |
                                                                                                         |
            # Broadcast the output index to corresponding indices in input tensors a and b               |
            a_index = a_shape.copy()                                                                     |
            b_index = b_shape.copy()                                                                     |
            broadcast_index(out_index, out_shape, a_shape, a_index)                                      |
            broadcast_index(out_index, out_shape, b_shape, b_index)                                      |
                                                                                                         |
            # Convert the multi-dimensional indices for a and b back to flat indices                     |
            a_ord = index_to_position(a_index, a_strides)                                                |
            b_ord = index_to_position(b_index, b_strides)                                                |
                                                                                                         |
            # Apply the function to elements from a and b, and store the result in the output tensor     |
            out[out_ord] = fn(a_storage[a_ord], b_storage[b_ord])                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/ivanjie/workspace/mod3-cj466/minitorch/fast_ops.py (290)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/ivanjie/workspace/mod3-cj466/minitorch/fast_ops.py (290)
---------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                             |
        out: Storage,                                                                        |
        out_shape: Shape,                                                                    |
        out_strides: Strides,                                                                |
        a_storage: Storage,                                                                  |
        a_shape: Shape,                                                                      |
        a_strides: Strides,                                                                  |
        reduce_dim: int,                                                                     |
    ) -> None:                                                                               |
        # Iterate over all elements of the output tensor in parallel                         |
        for ord in prange(len(out)):---------------------------------------------------------| #11
            # Create an index for the current element in the output tensor                   |
            out_index: Index = np.zeros_like(out_shape, dtype=np.int32)                      |
            to_index(ord, out_shape, out_index)                                              |
                                                                                             |
            # Map the output index to the corresponding position in the input tensor         |
            a_ord = index_to_position(out_index, a_strides)                                  |
                                                                                             |
            # Initialize the reduction value with the current value in the output storage    |
            reduce_value = out[ord]                                                          |
                                                                                             |
            # Perform the reduction along the specified dimension                            |
            for i in prange(a_shape[reduce_dim]):--------------------------------------------| #10
                reduce_value = fn(                                                           |
                    float(a_storage[a_ord + i * a_strides[reduce_dim]]), reduce_value        |
                )                                                                            |
                                                                                             |
            # Store the final reduction result in the output tensor                          |
            out[ord] = reduce_value                                                          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--11 is a parallel loop
   +--10 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--10 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--10 (serial)



Parallel region 0 (loop #11) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#11).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/ivanjie/workspace/mod3-cj466/minitorch/fast_ops.py (323)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/ivanjie/workspace/mod3-cj466/minitorch/fast_ops.py (323)
----------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                  |
    out: Storage,                                                                             |
    out_shape: Shape,                                                                         |
    out_strides: Strides,                                                                     |
    a_storage: Storage,                                                                       |
    a_shape: Shape,                                                                           |
    a_strides: Strides,                                                                       |
    b_storage: Storage,                                                                       |
    b_shape: Shape,                                                                           |
    b_strides: Strides,                                                                       |
) -> None:                                                                                    |
    """NUMBA tensor matrix multiply function.                                                 |
                                                                                              |
    Should work for any tensor shapes that broadcast as long as                               |
                                                                                              |
    ```                                                                                       |
    assert a_shape[-1] == b_shape[-2]                                                         |
    ```                                                                                       |
                                                                                              |
    Optimizations:                                                                            |
                                                                                              |
    * Outer loop in parallel                                                                  |
    * No index buffers or function calls                                                      |
    * Inner loop should have no global writes, 1 multiply.                                    |
                                                                                              |
                                                                                              |
    Args:                                                                                     |
    ----                                                                                      |
        out (Storage): storage for `out` tensor                                               |
        out_shape (Shape): shape for `out` tensor                                             |
        out_strides (Strides): strides for `out` tensor                                       |
        a_storage (Storage): storage for `a` tensor                                           |
        a_shape (Shape): shape for `a` tensor                                                 |
        a_strides (Strides): strides for `a` tensor                                           |
        b_storage (Storage): storage for `b` tensor                                           |
        b_shape (Shape): shape for `b` tensor                                                 |
        b_strides (Strides): strides for `b` tensor                                           |
                                                                                              |
    Returns:                                                                                  |
    -------                                                                                   |
        None : Fills in `out`                                                                 |
                                                                                              |
    """                                                                                       |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                    |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                    |
                                                                                              |
    N, I, J, K = out_shape[0], out_shape[1], out_shape[2], a_shape[-1]                        |
    for n in prange(N):  # Batch dimension----------------------------------------------------| #15
        for i in prange(I):  # Row of the result matrix---------------------------------------| #14
            for j in prange(J):  # Column of the result matrix--------------------------------| #13
                for k in prange(K):  # Inner dimension of the matrix multiplication-----------| #12
                    # Calculate the flat index for the output tensor                          |
                    out_ord = (                                                               |
                        n * out_strides[0] + i * out_strides[1] + j * out_strides[2]          |
                    )                                                                         |
                                                                                              |
                    # Calculate the flat index for the corresponding element in tensor `a`    |
                    a_ord = n * a_batch_stride + i * a_strides[1] + k * a_strides[2]          |
                                                                                              |
                    # Calculate the flat index for the corresponding element in tensor `b`    |
                    b_ord = n * b_batch_stride + k * b_strides[1] + j * b_strides[2]          |
                                                                                              |
                    # Update the output tensor by accumulating the product of `a` and `b`     |
                    out[out_ord] += a_storage[a_ord] * b_storage[b_ord]                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #15, #14).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--15 is a parallel loop
   +--14 --> rewritten as a serial loop
      +--13 --> rewritten as a serial loop
         +--12 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--15 (parallel)
   +--14 (parallel)
      +--13 (parallel)
         +--12 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--15 (parallel)
   +--14 (serial)
      +--13 (serial)
         +--12 (serial)



Parallel region 0 (loop #15) had 0 loop(s) fused and 3 loop(s) serialized as
part of the larger parallel loop (#15).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
## Task 3.3 Colab Tests Passed
![image](images/3_3_tests.png)

## Task 3.4 Colab Tests Passed
![image](images/3_4_tests.png)

## Task 3.4 Comparison
```
Timing summary
Size: 64
    fast: 0.00497
    gpu: 0.00735
Size: 128
    fast: 0.01643
    gpu: 0.01549
Size: 256
    fast: 0.10202
    gpu: 0.05411
Size: 512
    fast: 1.24839
    gpu: 0.29812
Size: 1024
    fast: 7.84488
    gpu: 0.99288
```
![image](images/comparison.png)

## Task 3.5 Training
### Small Model - Hidden Size 100
#### Simple Dataset
CPU
```
Epoch  0  loss  4.554181077579846 correct 45
Epoch  10  loss  1.5090083188339602 correct 48
Epoch  20  loss  1.9742535010939828 correct 49
Epoch  30  loss  0.5127729899778487 correct 49
Epoch  40  loss  0.7186091043609798 correct 50
Epoch  50  loss  0.5987435265080489 correct 50
Epoch  60  loss  1.1837308011220888 correct 50
Epoch  70  loss  0.12218397628019075 correct 50
Epoch  80  loss  0.4049360237360724 correct 50
Epoch  90  loss  0.9687995164628633 correct 50
Epoch  100  loss  0.5223630576862013 correct 50
Epoch  110  loss  0.7083021254303025 correct 50
Epoch  120  loss  0.19787301121021836 correct 50
Epoch  130  loss  0.6667049297034455 correct 50
Epoch  140  loss  0.19842760811190222 correct 50
Epoch  150  loss  0.38918391054576196 correct 50
Epoch  160  loss  0.1638523723383155 correct 50
Epoch  170  loss  0.3283445785996876 correct 50
Epoch  180  loss  0.4120566673699339 correct 50
Epoch  190  loss  0.49854273853912356 correct 50
Epoch  200  loss  0.1648567531594662 correct 50
Epoch  210  loss  0.025299397670491417 correct 50
Epoch  220  loss  0.0039457823774615855 correct 50
Epoch  230  loss  0.11841094077029239 correct 50
Epoch  240  loss  0.02239508944601168 correct 50
Epoch  250  loss  0.012288002765423906 correct 50
Epoch  260  loss  0.04214571894622858 correct 50
Epoch  270  loss  0.11599225765072799 correct 50
Epoch  280  loss  0.018826103276133512 correct 50
Epoch  290  loss  0.015755191173576776 correct 50
Epoch  300  loss  0.06412986989326609 correct 50
Epoch  310  loss  0.06752114055950108 correct 50
Epoch  320  loss  0.23977389457905604 correct 50
Epoch  330  loss  0.1827051125275794 correct 50
Epoch  340  loss  0.3006096702439851 correct 50
Epoch  350  loss  0.20595987256054138 correct 50
Epoch  360  loss  0.23747391999194126 correct 50
Epoch  370  loss  0.18273693461353038 correct 50
Epoch  380  loss  0.06030065001331317 correct 50
Epoch  390  loss  0.1651024926684946 correct 50
Epoch  400  loss  0.006620110897598633 correct 50
Epoch  410  loss  0.00927123563109152 correct 50
Epoch  420  loss  0.0357229021953348 correct 50
Epoch  430  loss  0.04499339944358401 correct 50
Epoch  440  loss  0.029190106598736883 correct 50
Epoch  450  loss  0.006594987989185569 correct 50
Epoch  460  loss  0.18112341890802877 correct 50
Epoch  470  loss  0.0002466958317571157 correct 50
Epoch  480  loss  0.011005845978470331 correct 50
Epoch  490  loss  0.07121396899652256 correct 50
Average time per epoch: 0.20498333740234376
```
GPU
```
Epoch  0  loss  5.087150443717258 correct 42
Epoch  10  loss  1.966906786504059 correct 45
Epoch  20  loss  1.1364743437708582 correct 49
Epoch  30  loss  2.1181814511333092 correct 48
Epoch  40  loss  1.332343358712413 correct 47
Epoch  50  loss  1.4008940028707453 correct 49
Epoch  60  loss  0.20078376367976855 correct 50
Epoch  70  loss  0.3473257810424449 correct 48
Epoch  80  loss  0.15870909629264263 correct 49
Epoch  90  loss  1.8717360206679188 correct 49
Epoch  100  loss  0.10218601400940498 correct 49
Epoch  110  loss  0.08365030201306813 correct 50
Epoch  120  loss  0.13983916250170933 correct 49
Epoch  130  loss  0.6323544769908471 correct 50
Epoch  140  loss  0.7122422160482524 correct 50
Epoch  150  loss  1.7797539796550472 correct 49
Epoch  160  loss  1.1118090693034386 correct 49
Epoch  170  loss  0.19621690541169173 correct 49
Epoch  180  loss  0.34080106507914754 correct 49
Epoch  190  loss  0.7104929504841849 correct 50
Epoch  200  loss  0.03619222076842221 correct 49
Epoch  210  loss  0.3213803581965569 correct 50
Epoch  220  loss  2.0624573477639716 correct 49
Epoch  230  loss  0.4871569178581991 correct 49
Epoch  240  loss  2.8234387548557622 correct 46
Epoch  250  loss  1.5205173448648663 correct 50
Epoch  260  loss  0.44973552115250665 correct 50
Epoch  270  loss  0.25714426407163443 correct 49
Epoch  280  loss  0.10500017187499686 correct 49
Epoch  290  loss  0.06821218925802372 correct 50
Epoch  300  loss  0.47522271174938846 correct 49
Epoch  310  loss  1.0029859110505082 correct 50
Epoch  320  loss  0.6721852981520278 correct 49
Epoch  330  loss  0.6087331113637315 correct 49
Epoch  340  loss  0.48058067039758817 correct 49
Epoch  350  loss  0.013371945486432278 correct 49
Epoch  360  loss  0.027679556320570474 correct 48
Epoch  370  loss  0.08124422050275065 correct 49
Epoch  380  loss  1.1936078895582931 correct 49
Epoch  390  loss  0.00027666166314333786 correct 50
Epoch  400  loss  1.1857089421184352 correct 50
Epoch  410  loss  1.74384890968933 correct 48
Epoch  420  loss  0.5921747689236045 correct 49
Epoch  430  loss  0.0017928080844643996 correct 49
Epoch  440  loss  0.2726169363341542 correct 49
Epoch  450  loss  0.8189420845972865 correct 49
Epoch  460  loss  0.051852841909777056 correct 49
Epoch  470  loss  0.01784417021726695 correct 49
Epoch  480  loss  0.41709047381651393 correct 49
Epoch  490  loss  0.4029780503133864 correct 50
Average time per epoch: 1.9572782425880433
```
#### XOR Dataset
CPU
```
Epoch  0  loss  6.645685143063538 correct 40
Epoch  10  loss  2.8109682558078797 correct 46
Epoch  20  loss  2.5742125209212334 correct 47
Epoch  30  loss  1.7929209277781903 correct 47
Epoch  40  loss  2.0493449864511297 correct 46
Epoch  50  loss  2.0347663398003766 correct 46
Epoch  60  loss  3.2013906857499217 correct 48
Epoch  70  loss  2.017167731130316 correct 46
Epoch  80  loss  1.9586760627311026 correct 48
Epoch  90  loss  0.8040167327107473 correct 47
Epoch  100  loss  1.9840896915572144 correct 47
Epoch  110  loss  1.6286725921164789 correct 48
Epoch  120  loss  0.39710947318647466 correct 48
Epoch  130  loss  0.3959405624720464 correct 49
Epoch  140  loss  3.086622215675925 correct 47
Epoch  150  loss  1.7666772975816654 correct 49
Epoch  160  loss  1.6236665719415297 correct 49
Epoch  170  loss  0.5083637436577244 correct 49
Epoch  180  loss  0.5694684315763937 correct 49
Epoch  190  loss  0.18777158539270009 correct 48
Epoch  200  loss  0.30069278484381995 correct 48
Epoch  210  loss  0.6649991611167915 correct 49
Epoch  220  loss  1.4311089491063345 correct 49
Epoch  230  loss  1.695904565884088 correct 49
Epoch  240  loss  1.3047108284762197 correct 49
Epoch  250  loss  0.17083427552550126 correct 50
Epoch  260  loss  0.15901294768231422 correct 49
Epoch  270  loss  0.07333525701537909 correct 49
Epoch  280  loss  0.7248132214952854 correct 49
Epoch  290  loss  0.12389549448469948 correct 49
Epoch  300  loss  0.0594566039242437 correct 48
Epoch  310  loss  2.1186100839786786 correct 48
Epoch  320  loss  0.7421268443197685 correct 49
Epoch  330  loss  0.14819177926087704 correct 49
Epoch  340  loss  0.4558232267890152 correct 49
Epoch  350  loss  0.14240829028629762 correct 49
Epoch  360  loss  0.466661194941104 correct 50
Epoch  370  loss  0.13805553115803243 correct 49
Epoch  380  loss  0.034709160559625474 correct 49
Epoch  390  loss  0.18730772679577343 correct 49
Epoch  400  loss  1.4050654758390828 correct 50
Epoch  410  loss  0.29585498354904927 correct 49
Epoch  420  loss  1.193273872388212 correct 49
Epoch  430  loss  1.1156644562564595 correct 49
Epoch  440  loss  0.531375677736569 correct 49
Epoch  450  loss  0.060133972807823705 correct 49
Epoch  460  loss  0.04557440402028535 correct 50
Epoch  470  loss  0.026096340200740756 correct 50
Epoch  480  loss  0.8721060544161219 correct 49
Epoch  490  loss  0.9481485517660045 correct 49
Average time per epoch: 0.22334553861618042
```
GPU
```
Epoch  0  loss  8.21074579047815 correct 38
Epoch  10  loss  3.64283082630679 correct 39
Epoch  20  loss  3.483977365802019 correct 47
Epoch  30  loss  1.931204651143148 correct 49
Epoch  40  loss  1.920254962424416 correct 49
Epoch  50  loss  1.3136393758198945 correct 50
Epoch  60  loss  0.9354821561527278 correct 49
Epoch  70  loss  1.044158925835296 correct 49
Epoch  80  loss  0.7384080897161137 correct 50
Epoch  90  loss  0.5578472807147097 correct 50
Epoch  100  loss  0.4254701316676963 correct 50
Epoch  110  loss  0.5853649037481734 correct 50
Epoch  120  loss  0.5053541645690303 correct 50
Epoch  130  loss  0.21289616367326794 correct 50
Epoch  140  loss  0.5440333152768534 correct 50
Epoch  150  loss  0.3789088556606609 correct 50
Epoch  160  loss  0.621086299353663 correct 50
Epoch  170  loss  0.4803114433868402 correct 50
Epoch  180  loss  0.35468288218576366 correct 50
Epoch  190  loss  0.43174802449215066 correct 50
Epoch  200  loss  0.5626014651929235 correct 50
Epoch  210  loss  0.24081519306633617 correct 50
Epoch  220  loss  0.1572796970930473 correct 50
Epoch  230  loss  0.43298363087784886 correct 50
Epoch  240  loss  0.27725906262410865 correct 50
Epoch  250  loss  0.2694897960459083 correct 50
Epoch  260  loss  0.34302262431233116 correct 50
Epoch  270  loss  0.33251793173965505 correct 50
Epoch  280  loss  0.16850935757480182 correct 50
Epoch  290  loss  0.08400140099751226 correct 50
Epoch  300  loss  0.2223089901811453 correct 50
Epoch  310  loss  0.15850654295351232 correct 50
Epoch  320  loss  0.10903848569006266 correct 50
Epoch  330  loss  0.21789271459331577 correct 50
Epoch  340  loss  0.15176460334308225 correct 50
Epoch  350  loss  0.13255864592018274 correct 50
Epoch  360  loss  0.0765466383162719 correct 50
Epoch  370  loss  0.08796232705774495 correct 50
Epoch  380  loss  0.08323501936922134 correct 50
Epoch  390  loss  0.16901236436826483 correct 50
Epoch  400  loss  0.22097427511351259 correct 50
Epoch  410  loss  0.16344073879053175 correct 50
Epoch  420  loss  0.0343820997922726 correct 50
Epoch  430  loss  0.09272716240242791 correct 50
Epoch  440  loss  0.059890532053839166 correct 50
Epoch  450  loss  0.03535733366996672 correct 50
Epoch  460  loss  0.12106379122001974 correct 50
Epoch  470  loss  0.14612502385581252 correct 50
Epoch  480  loss  0.09731806180278908 correct 50
Epoch  490  loss  0.14921681737564998 correct 50
Average time per epoch: 1.9450337109565734
```
#### Split Dataset
CPU
```
Epoch  0  loss  7.936803844172463 correct 29
Epoch  10  loss  6.707426540419143 correct 29
Epoch  20  loss  6.646014212923118 correct 36
Epoch  30  loss  5.476212623916372 correct 42
Epoch  40  loss  4.243760902905562 correct 48
Epoch  50  loss  5.90950367431508 correct 43
Epoch  60  loss  3.589654775768451 correct 49
Epoch  70  loss  2.8075098762094086 correct 48
Epoch  80  loss  1.7073363867316396 correct 48
Epoch  90  loss  3.9974518831272148 correct 41
Epoch  100  loss  1.2702581232163601 correct 47
Epoch  110  loss  2.0297650008372 correct 49
Epoch  120  loss  2.124421330510582 correct 49
Epoch  130  loss  1.421565390863915 correct 49
Epoch  140  loss  1.378759255504011 correct 49
Epoch  150  loss  1.9695158987136203 correct 49
Epoch  160  loss  1.6284900723568763 correct 46
Epoch  170  loss  2.3891948758172257 correct 48
Epoch  180  loss  1.5396432158726119 correct 48
Epoch  190  loss  0.47806084278165445 correct 50
Epoch  200  loss  2.3347356567747286 correct 47
Epoch  210  loss  0.4911372572257482 correct 49
Epoch  220  loss  0.6357363679950139 correct 50
Epoch  230  loss  1.029482634632986 correct 49
Epoch  240  loss  0.649984281085427 correct 49
Epoch  250  loss  1.6942078036979829 correct 49
Epoch  260  loss  0.3082011426557499 correct 48
Epoch  270  loss  0.5494344059602743 correct 49
Epoch  280  loss  0.4323721307374382 correct 49
Epoch  290  loss  1.9617121658937655 correct 48
Epoch  300  loss  0.08218344532464678 correct 49
Epoch  310  loss  0.4170097874356226 correct 48
Epoch  320  loss  1.5039695843347138 correct 49
Epoch  330  loss  1.4803551404282644 correct 50
Epoch  340  loss  0.7382179388658527 correct 49
Epoch  350  loss  0.3800571632319718 correct 50
Epoch  360  loss  0.8750381791691868 correct 49
Epoch  370  loss  0.8152978036483889 correct 49
Epoch  380  loss  0.5261104161954708 correct 49
Epoch  390  loss  0.5168017374279115 correct 49
Epoch  400  loss  1.1743036083640446 correct 49
Epoch  410  loss  0.6276962462912236 correct 49
Epoch  420  loss  0.188459964199818 correct 49
Epoch  430  loss  0.05590778613996791 correct 49
Epoch  440  loss  1.7305735649547218 correct 49
Epoch  450  loss  0.38919759777493 correct 49
Epoch  460  loss  1.7442025464745643 correct 49
Epoch  470  loss  0.25096164823860717 correct 49
Epoch  480  loss  0.43483103966548 correct 49
Epoch  490  loss  0.08745604862037439 correct 49
Average time per epoch: 0.2261737103462219
```
GPU
```
Epoch  0  loss  7.902353777004769 correct 29
Epoch  10  loss  5.9356737414937575 correct 39
Epoch  20  loss  4.939809563006199 correct 47
Epoch  30  loss  3.8342534673499853 correct 43
Epoch  40  loss  3.8488752778700697 correct 39
Epoch  50  loss  2.300040122921313 correct 49
Epoch  60  loss  2.0524821273679024 correct 48
Epoch  70  loss  2.9677973960329935 correct 49
Epoch  80  loss  1.1341943533161534 correct 49
Epoch  90  loss  3.1443671146903793 correct 48
Epoch  100  loss  2.0641663859226718 correct 47
Epoch  110  loss  2.160089000479159 correct 47
Epoch  120  loss  1.2181715867580654 correct 47
Epoch  130  loss  1.6992761459970196 correct 48
Epoch  140  loss  2.46906282259048 correct 47
Epoch  150  loss  2.5506754910837035 correct 48
Epoch  160  loss  3.0777446821561654 correct 48
Epoch  170  loss  0.7511194940454833 correct 49
Epoch  180  loss  1.311252230293073 correct 47
Epoch  190  loss  0.8052085426646274 correct 48
Epoch  200  loss  1.1673063312782372 correct 49
Epoch  210  loss  1.2762304167419405 correct 49
Epoch  220  loss  0.7775514110684861 correct 49
Epoch  230  loss  1.589133034953208 correct 50
Epoch  240  loss  0.7245686302423384 correct 48
Epoch  250  loss  1.86445785061826 correct 47
Epoch  260  loss  1.3311232267910855 correct 50
Epoch  270  loss  0.6922599160988185 correct 48
Epoch  280  loss  0.28146492231361087 correct 49
Epoch  290  loss  2.026876641155987 correct 49
Epoch  300  loss  0.40931742688537237 correct 47
Epoch  310  loss  1.2131869988657193 correct 50
Epoch  320  loss  0.7246396091191739 correct 49
Epoch  330  loss  1.6656936875315305 correct 49
Epoch  340  loss  0.9448539826849306 correct 49
Epoch  350  loss  1.352454471450082 correct 49
Epoch  360  loss  0.39031136919332077 correct 49
Epoch  370  loss  0.527973836740328 correct 48
Epoch  380  loss  0.37586472077191746 correct 48
Epoch  390  loss  1.291114788847838 correct 49
Epoch  400  loss  0.1628216665128324 correct 50
Epoch  410  loss  2.629121509457415 correct 48
Epoch  420  loss  2.6236721445035776 correct 48
Epoch  430  loss  0.7987222641478291 correct 49
Epoch  440  loss  3.5848482040130616 correct 47
Epoch  450  loss  0.2102807539853159 correct 48
Epoch  460  loss  0.5400089585601642 correct 49
Epoch  470  loss  1.3061140344300788 correct 48
Epoch  480  loss  0.058762670592670534 correct 49
Epoch  490  loss  0.15416252923687382 correct 49
Average time per epoch: 1.9389712119102478
```
### Bigger Model - Hidden Size 200
#### Simple Dataset
CPU
```
Epoch  0  loss  7.515965236493035 correct 42
Epoch  10  loss  1.6498734691540138 correct 49
Epoch  20  loss  0.03710139858676592 correct 50
Epoch  30  loss  0.15556503284801326 correct 50
Epoch  40  loss  1.5100820532174766 correct 50
Epoch  50  loss  0.5210324441822126 correct 50
Epoch  60  loss  0.15979509070615405 correct 50
Epoch  70  loss  0.14318961513061237 correct 49
Epoch  80  loss  0.5232751877729962 correct 49
Epoch  90  loss  0.008697244823328943 correct 50
Epoch  100  loss  0.7772612463367575 correct 50
Epoch  110  loss  0.24246384452444333 correct 50
Epoch  120  loss  0.06101479444097101 correct 50
Epoch  130  loss  0.294514707766897 correct 50
Epoch  140  loss  0.8036252289426211 correct 50
Epoch  150  loss  0.41906847731563407 correct 49
Epoch  160  loss  0.09269086689623919 correct 50
Epoch  170  loss  0.21214597169407717 correct 50
Epoch  180  loss  0.1111985662236698 correct 50
Epoch  190  loss  0.01831056431928793 correct 50
Epoch  200  loss  0.09281842590767442 correct 50
Epoch  210  loss  0.009700981173774758 correct 50
Epoch  220  loss  0.024399384268247302 correct 50
Epoch  230  loss  0.6328881215548753 correct 50
Epoch  240  loss  0.464853099328698 correct 50
Epoch  250  loss  0.05477422789418777 correct 50
Epoch  260  loss  0.12871044024578807 correct 50
Epoch  270  loss  0.012402175251427313 correct 50
Epoch  280  loss  0.10358684143395258 correct 50
Epoch  290  loss  0.010553653237372842 correct 50
Epoch  300  loss  0.015691851908954446 correct 50
Epoch  310  loss  0.5052054631112919 correct 50
Epoch  320  loss  0.08103300224752824 correct 50
Epoch  330  loss  0.5411752911933436 correct 50
Epoch  340  loss  0.09529348269291406 correct 50
Epoch  350  loss  0.36961275305176766 correct 50
Epoch  360  loss  0.39949422124340905 correct 50
Epoch  370  loss  0.09616970666438947 correct 50
Epoch  380  loss  0.09342820284234729 correct 50
Epoch  390  loss  0.42818155768893784 correct 50
Epoch  400  loss  -8.595242055954016e-06 correct 50
Epoch  410  loss  0.024178241027478824 correct 50
Epoch  420  loss  0.007034609226402175 correct 50
Epoch  430  loss  0.027592775947564742 correct 50
Epoch  440  loss  0.4737379728125964 correct 50
Epoch  450  loss  0.0098327626028109 correct 50
Epoch  460  loss  0.45236853343722955 correct 50
Epoch  470  loss  0.3556922597668809 correct 50
Epoch  480  loss  0.42669817206073246 correct 50
Epoch  490  loss  0.4112582628651945 correct 50
Average time per epoch: 0.35724978971481325
```
GPU
```
Epoch  0  loss  16.11839144865031 correct 45
Epoch  10  loss  1.1963614453377738 correct 46
Epoch  20  loss  2.422509888885071 correct 47
Epoch  30  loss  1.0039626510257837 correct 50
Epoch  40  loss  1.4943389470939472 correct 49
Epoch  50  loss  1.008403639348147 correct 49
Epoch  60  loss  1.0696993215166157 correct 50
Epoch  70  loss  0.5084443884255988 correct 49
Epoch  80  loss  0.5219503131484252 correct 49
Epoch  90  loss  0.030336276859628905 correct 49
Epoch  100  loss  0.004592086925838261 correct 49
Epoch  110  loss  0.5594159971234676 correct 49
Epoch  120  loss  0.3037938843213494 correct 50
Epoch  130  loss  0.005645765380129609 correct 49
Epoch  140  loss  0.16717991193757217 correct 49
Epoch  150  loss  0.6360331915714833 correct 49
Epoch  160  loss  0.8765003323407458 correct 50
Epoch  170  loss  0.003068097722540937 correct 49
Epoch  180  loss  1.2606193923416593 correct 49
Epoch  190  loss  0.4221992465222363 correct 50
Epoch  200  loss  0.21085022496278355 correct 50
Epoch  210  loss  1.0252414916581525 correct 50
Epoch  220  loss  0.19007745539238394 correct 50
Epoch  230  loss  0.9045320176123116 correct 49
Epoch  240  loss  0.20314530615486828 correct 49
Epoch  250  loss  0.13741544778557457 correct 50
Epoch  260  loss  0.9548277003956668 correct 50
Epoch  270  loss  0.010919175473733701 correct 50
Epoch  280  loss  0.016139093049320093 correct 49
Epoch  290  loss  0.020986511076753197 correct 50
Epoch  300  loss  0.0018383442237920402 correct 49
Epoch  310  loss  0.006118770754795372 correct 50
Epoch  320  loss  0.0033316038606935024 correct 50
Epoch  330  loss  0.9839156150130981 correct 50
Epoch  340  loss  1.1344204960968491 correct 49
Epoch  350  loss  0.05015892612113559 correct 49
Epoch  360  loss  0.013807249412038874 correct 50
Epoch  370  loss  0.7576685712578306 correct 49
Epoch  380  loss  0.06176470490516899 correct 49
Epoch  390  loss  0.010879855853662308 correct 50
Epoch  400  loss  0.06752395459550632 correct 50
Epoch  410  loss  0.7925666467225042 correct 50
Epoch  420  loss  0.2829833005786914 correct 50
Epoch  430  loss  0.004249140503155512 correct 49
Epoch  440  loss  0.015950276050868167 correct 50
Epoch  450  loss  0.17752991388044548 correct 50
Epoch  460  loss  0.1687146167664151 correct 50
Epoch  470  loss  0.5609786658054848 correct 50
Epoch  480  loss  0.0010461893294267332 correct 49
Epoch  490  loss  0.6092409200737559 correct 49
Average time per epoch: 2.027365529537201
```