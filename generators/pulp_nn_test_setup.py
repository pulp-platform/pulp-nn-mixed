#
# pulp_nn_test_setup.py
# Nazareno Bruschi <nazareno.bruschi@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# If is 1 only the selected-below kernel is created (SINGLE KERNEL SOLUTION). Otherwise, all kernels are created (ALL KERNELS SOLUTION)

SINGLE_KERNEL		= 1

# Select from the supported ones:

# -> XpulpV2, XpulpNN, XpulpNN-mixed

ISA 				= 'XpulpNN'

# Select from the supported ones:

# -> 4x2, 4x4
MATMUL_FMT			= '4x2'

# Select from the supported ones:

# -> convolution, pointwise, depthwise, linear_no_quant, linear_quant, maxpool, avgpool, add

TYPE_OF_KERNEL 		= 'convolution'

# If SINGLE_KERNEL = 0 these will be ignored. Otherwise, select the possibilities from the supported ones
#
# -> input activations precision:
#       - 8, 4, 2
# -> input signedness:
#       - True or False for all except QuantAdd
#       - [{True, False}, {True, False}] for QuantAdd (describing
#         input1/input2's signedness respectively)
# -> output activations precision:
#       - 8, 4, 2
# -> output signedness:
#       - True or False
# -> weights precision:
#       - 8, 4, 2
# -> quantization method:
#       - shift_clip

in_precision 		= 2
wt_precision 		= 2
in_signed = True
out_precision 		= 2 # if is add layer, out_precision is the second input
# precision
out_signed = False
quantization_type 	= 'shift_clip' # TODO / WARNING: THRESHOLD QUANTIZATION NOT
# SUPPORTED FOR SIGNED OUTPUT KERNELS

# Select layer dimensions from supported ones:

# -> input channels:
#       - all values if less precision between ifmaps and weights is INT8
#       - multiples of 2 if less precision between ifmaps and weights is INT4
#       - multiples of 4 if less precision between ifmaps and weights in INT2
# -> output channels:
#       - all values if less precision is INT8
#       - multiples of 2 for uint4 output activations precision
#       - multiples of 4 for uint2 output activations precision
# -> input/output activations:
#       - all values for dim_x
#       - all values for dim_y

# if depthwise CH_IM_IN must be equal to CH_IM_OUT
DIM_IM_IN_X 		= 8
DIM_IM_IN_Y 		= 8
CH_IM_IN 			= 4
# if pooling, output dimensions must be relative to kernel and stride sizes
DIM_IM_OUT_X 		= 8
DIM_IM_OUT_Y 		= 8
CH_IM_OUT 			= 4
# if is not linear
DIM_KERNEL_X 		= 3 # 1 if is pointwise, free otherwise
DIM_KERNEL_Y 		= 3 # 1 if is pointwise, free otherwise
PADDING_Y_TOP 		= 1 # 0 if is pointwise, free otherwise
PADDING_Y_BOTTOM 	= 1 # 0 if is pointwise, free otherwise
PADDING_X_LEFT 		= 1 # 0 if is pointwise, free otherwise
PADDING_X_RIGHT 	= 1 # 0 if is pointwise, free otherwise
STRIDE_X 			= 1
STRIDE_Y 			= 1
# Other parameters
BIAS 				= False
BN 					= True # must be false for linear_no_quant
RELU 				= True # must be false for linear_no_quant
# If is pooling
POOL_KERNEL 		= 2
POOL_STRIDE 		= 2


