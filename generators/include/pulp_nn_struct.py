#
# pulp_nn_struct.py
# Nazareno Bruschi <nazareno.bruschi@unibo.it>
# Alessandro Nadalini <alessandro.nadalini3@unibo.it>
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

import errno
import os
import shutil
import pathlib

# PULP-NN Mixed structure

# Root
PULPNNInstallPath = os.getcwd() + "/../"

# Generation scripts path
PULPNNSrcDirs = PULPNNInstallPath + "generators/"

# Extentions dependent: Software -> XpulpV2
PULPNNInstallSWPath    = PULPNNInstallPath + "XpulpV2/"
PULPNNInstallHWPath    = PULPNNInstallPath + "XpulpNN/"
PULPNNInstallExtHWPath = PULPNNInstallPath + "XpulpNN-mixed/"

# Model dependent
PULPNNInstallPathSW32bit    = PULPNNInstallSWPath + "32bit/" 
PULPNNInstallPathSW64bit    = PULPNNInstallSWPath + "64bit/"
PULPNNInstallPathHW32bit    = PULPNNInstallHWPath + "32bit/" 
PULPNNInstallPathHW64bit    = PULPNNInstallHWPath + "64bit/"
PULPNNInstallPathExtHW32bit = PULPNNInstallExtHWPath + "32bit/"
PULPNNInstallPathExtHW64bit = PULPNNInstallExtHWPath + "64bit/"

# Testing path
PULPNNTestFolderSW32bit    = PULPNNInstallPathSW32bit + "test/"
PULPNNTestFolderSW64bit    = PULPNNInstallPathSW64bit + "test/"
PULPNNTestFolderHW32bit    = PULPNNInstallPathHW32bit + "test/"
PULPNNTestFolderHW64bit    = PULPNNInstallPathHW64bit + "test/"
PULPNNTestFolderExtHW32bit = PULPNNInstallPathExtHW32bit + "test/"
PULPNNTestFolderExtHW64bit = PULPNNInstallPathExtHW64bit + "test/"

# Sotware sources path
PULPNNSrcDirsSW32bit = {'src':                      PULPNNInstallPathSW32bit + "src/",
                        'inc':                      PULPNNInstallPathSW32bit + "include/",
                        'convolution':              PULPNNInstallPathSW32bit + "src/Convolution/",
                        'pointwise':                PULPNNInstallPathSW32bit + "src/Pointwise/",
                        'matmul':                   PULPNNInstallPathSW32bit + "src/MatrixMultiplication/",
                        'depthwise':                PULPNNInstallPathSW32bit + "src/Depthwise/",
                        'linear_nq':                PULPNNInstallPathSW32bit + "src/LinearNoQuant/",
                        'linear_q':                 PULPNNInstallPathSW32bit + "src/LinearQuant/",
                        'pooling':                  PULPNNInstallPathSW32bit + "src/Pooling/",
                        'maxpool':                  PULPNNInstallPathSW32bit + "src/Pooling/MaxPool/",
                        'avgpool':                  PULPNNInstallPathSW32bit + "src/Pooling/AvgPool/",
                        'add':                      PULPNNInstallPathSW32bit + "src/Add/",
                        'pulp_nn_include':          PULPNNTestFolderSW32bit + "include/",
                        'pulp_nn_src':              PULPNNTestFolderSW32bit + "src/",
                        'pulp_nn_convolution':      PULPNNTestFolderSW32bit + "src/Convolution/",
                        'pulp_nn_pointwise':        PULPNNTestFolderSW32bit + "src/Pointwise/",
                        'pulp_nn_matmul':           PULPNNTestFolderSW32bit + "src/MatrixMultiplication/",
                        'pulp_nn_depthwise':        PULPNNTestFolderSW32bit + "src/Depthwise/",
                        'pulp_nn_linear_nq':        PULPNNTestFolderSW32bit + "src/LinearNoQuant/",
                        'pulp_nn_linear_q':         PULPNNTestFolderSW32bit + "src/LinearQuant/",
                        'pulp_nn_maxpool':          PULPNNTestFolderSW32bit + "src/Pooling/MaxPool/",
                        'pulp_nn_avgpool':          PULPNNTestFolderSW32bit + "src/Pooling/AvgPool/",
                        'pulp_nn_add':              PULPNNTestFolderSW32bit + "src/Add/",
                        'data_allocation_matm':     PULPNNTestFolderSW32bit + "include/DataAllocationMatMul/",
                        'data_allocation_conv':     PULPNNTestFolderSW32bit + "include/DataAllocationConvolution/",
                        'data_allocation_pw':       PULPNNTestFolderSW32bit + "include/DataAllocationPointwise/",
                        'data_allocation_dw':       PULPNNTestFolderSW32bit + "include/DataAllocationDepthwise/",
                        'data_allocation_ln_nq':    PULPNNTestFolderSW32bit + "include/DataAllocationLinearNoQuant/",
                        'data_allocation_ln_q':     PULPNNTestFolderSW32bit + "include/DataAllocationLinearQuant/",
                        'data_allocation_maxp':     PULPNNTestFolderSW32bit + "include/DataAllocationMaxPool/",
                        'data_allocation_avgp':     PULPNNTestFolderSW32bit + "include/DataAllocationAvgPool/",
                        'data_allocation_add':      PULPNNTestFolderSW32bit + "include/DataAllocationAdd/",
                        'golden_model_matm':        PULPNNTestFolderSW32bit + "include/GoldenModelMatMul/",
                        'golden_model_conv':        PULPNNTestFolderSW32bit + "include/GoldenModelConvolution/",
                        'golden_model_pw':          PULPNNTestFolderSW32bit + "include/GoldenModelPointwise/",
                        'golden_model_dw':          PULPNNTestFolderSW32bit + "include/GoldenModelDepthwise/",
                        'golden_model_ln_nq':       PULPNNTestFolderSW32bit + "include/GoldenModelLinearNoQuant/",
                        'golden_model_ln_q':        PULPNNTestFolderSW32bit + "include/GoldenModelLinearQuant/",
                        'golden_model_maxp':        PULPNNTestFolderSW32bit + "include/GoldenModelMaxPool/",
                        'golden_model_avgp':        PULPNNTestFolderSW32bit + "include/GoldenModelAvgPool/",
                        'golden_model_add':         PULPNNTestFolderSW32bit + "include/GoldenModelAdd/",
                        'test':                     PULPNNTestFolderSW32bit}

PULPNNSrcDirsSW64bit = {'src':                      PULPNNInstallPathSW64bit + "src/",
                        'inc':                      PULPNNInstallPathSW64bit + "include/",
                        'convolution':              PULPNNInstallPathSW64bit + "src/Convolution/",
                        'pointwise':                PULPNNInstallPathSW64bit + "src/Pointwise/",
                        'matmul':                   PULPNNInstallPathSW64bit + "src/MatrixMultiplication/",
                        'depthwise':                PULPNNInstallPathSW64bit + "src/Depthwise/",
                        'linear_nq':                PULPNNInstallPathSW64bit + "src/LinearNoQuant/",
                        'linear_q':                 PULPNNInstallPathSW64bit + "src/LinearQuant/",
                        'pooling':                  PULPNNInstallPathSW64bit + "src/Pooling/",
                        'maxpool':                  PULPNNInstallPathSW64bit + "src/Pooling/MaxPool/",
                        'avgpool':                  PULPNNInstallPathSW64bit + "src/Pooling/AvgPool/",
                        'add':                      PULPNNInstallPathSW64bit + "src/Add/",
                        'pulp_nn_include':          PULPNNTestFolderSW64bit + "include/",
                        'pulp_nn_src':              PULPNNTestFolderSW64bit + "src/",
                        'pulp_nn_convolution':      PULPNNTestFolderSW64bit + "src/Convolution/",
                        'pulp_nn_pointwise':        PULPNNTestFolderSW64bit + "src/Pointwise/",
                        'pulp_nn_matmul':           PULPNNTestFolderSW64bit + "src/MatrixMultiplication/",
                        'pulp_nn_depthwise':        PULPNNTestFolderSW64bit + "src/Depthwise/",
                        'pulp_nn_linear_nq':        PULPNNTestFolderSW64bit + "src/LinearNoQuant/",
                        'pulp_nn_linear_q':         PULPNNTestFolderSW64bit + "src/LinearQuant/",
                        'pulp_nn_maxpool':          PULPNNTestFolderSW64bit + "src/Pooling/MaxPool/",
                        'pulp_nn_avgpool':          PULPNNTestFolderSW64bit + "src/Pooling/AvgPool/",
                        'pulp_nn_add':              PULPNNTestFolderSW64bit + "src/Add/",
                        'data_allocation_matm':     PULPNNTestFolderSW64bit + "include/DataAllocationMatMul/",
                        'data_allocation_conv':     PULPNNTestFolderSW64bit + "include/DataAllocationConvolution/",
                        'data_allocation_pw':       PULPNNTestFolderSW64bit + "include/DataAllocationPointwise/",
                        'data_allocation_dw':       PULPNNTestFolderSW64bit + "include/DataAllocationDepthwise/",
                        'data_allocation_ln_nq':    PULPNNTestFolderSW64bit + "include/DataAllocationLinearNoQuant/",
                        'data_allocation_ln_q':     PULPNNTestFolderSW64bit + "include/DataAllocationLinearQuant/",
                        'data_allocation_maxp':     PULPNNTestFolderSW64bit + "include/DataAllocationMaxPool/",
                        'data_allocation_avgp':     PULPNNTestFolderSW64bit + "include/DataAllocationAvgPool/",
                        'data_allocation_add':      PULPNNTestFolderSW64bit + "include/DataAllocationAdd/",
                        'golden_model_matm':        PULPNNTestFolderSW64bit + "include/GoldenModelMatMul/",
                        'golden_model_conv':        PULPNNTestFolderSW64bit + "include/GoldenModelConvolution/",
                        'golden_model_pw':          PULPNNTestFolderSW64bit + "include/GoldenModelPointwise/",
                        'golden_model_dw':          PULPNNTestFolderSW64bit + "include/GoldenModelDepthwise/",
                        'golden_model_ln_nq':       PULPNNTestFolderSW64bit + "include/GoldenModelLinearNoQuant/",
                        'golden_model_ln_q':        PULPNNTestFolderSW64bit + "include/GoldenModelLinearQuant/",
                        'golden_model_maxp':        PULPNNTestFolderSW64bit + "include/GoldenModelMaxPool/",
                        'golden_model_avgp':        PULPNNTestFolderSW64bit + "include/GoldenModelAvgPool/",
                        'golden_model_add':         PULPNNTestFolderSW64bit + "include/GoldenModelAdd/",
                        'test':                     PULPNNTestFolderSW64bit}

# XpulpNN sources path
PULPNNSrcDirsHW32bit = {'src':                      PULPNNInstallPathHW32bit + "src/",
                        'inc':                      PULPNNInstallPathHW32bit + "include/",
                        'convolution':              PULPNNInstallPathHW32bit + "src/Convolution/",
                        'conv1d':                   PULPNNInstallPathHW32bit + "src/Convolution/",
                        'pointwise':                PULPNNInstallPathHW32bit + "src/Pointwise/",
                        'matmul':                   PULPNNInstallPathHW32bit + "src/MatrixMultiplication/",
                        'depthwise':                PULPNNInstallPathHW32bit + "src/Depthwise/",
                        'linear_nq':                PULPNNInstallPathHW32bit + "src/LinearNoQuant/",
                        'linear_q':                 PULPNNInstallPathHW32bit + "src/LinearQuant/",
                        'pooling':                  PULPNNInstallPathHW32bit + "src/Pooling/",
                        'maxpool':                  PULPNNInstallPathHW32bit + "src/Pooling/MaxPool/",
                        'avgpool':                  PULPNNInstallPathHW32bit + "src/Pooling/AvgPool/",
                        'add':                      PULPNNInstallPathHW32bit + "src/Add/",
                        'pulp_nn_include':          PULPNNTestFolderHW32bit + "include/",
                        'pulp_nn_src':              PULPNNTestFolderHW32bit + "src/",
                        'pulp_nn_convolution':      PULPNNTestFolderHW32bit + "src/Convolution/",
                        'pulp_nn_pointwise':        PULPNNTestFolderHW32bit + "src/Pointwise/",
                        'pulp_nn_matmul':           PULPNNTestFolderHW32bit + "src/MatrixMultiplication/",
                        'pulp_nn_depthwise':        PULPNNTestFolderHW32bit + "src/Depthwise/",
                        'pulp_nn_linear_nq':        PULPNNTestFolderHW32bit + "src/LinearNoQuant/",
                        'pulp_nn_linear_q':         PULPNNTestFolderHW32bit + "src/LinearQuant/",
                        'pulp_nn_maxpool':          PULPNNTestFolderHW32bit + "src/Pooling/MaxPool/",
                        'pulp_nn_avgpool':          PULPNNTestFolderHW32bit + "src/Pooling/AvgPool/",
                        'pulp_nn_add':              PULPNNTestFolderHW32bit + "src/Add/",
                        'data_allocation_matm':     PULPNNTestFolderHW32bit + "include/DataAllocationMatMul/",
                        'data_allocation_conv':     PULPNNTestFolderHW32bit + "include/DataAllocationConvolution/",
                        'data_allocation_pw':       PULPNNTestFolderHW32bit + "include/DataAllocationPointwise/",
                        'data_allocation_dw':       PULPNNTestFolderHW32bit + "include/DataAllocationDepthwise/",
                        'data_allocation_ln_nq':    PULPNNTestFolderHW32bit + "include/DataAllocationLinearNoQuant/",
                        'data_allocation_ln_q':     PULPNNTestFolderHW32bit + "include/DataAllocationLinearQuant/",
                        'data_allocation_maxp':     PULPNNTestFolderHW32bit + "include/DataAllocationMaxPool/",
                        'data_allocation_avgp':     PULPNNTestFolderHW32bit + "include/DataAllocationAvgPool/",
                        'data_allocation_add':      PULPNNTestFolderHW32bit + "include/DataAllocationAdd/",
                        'golden_model_matm':        PULPNNTestFolderHW32bit + "include/GoldenModelMatMul/",
                        'golden_model_conv':        PULPNNTestFolderHW32bit + "include/GoldenModelConvolution/",
                        'golden_model_pw':          PULPNNTestFolderHW32bit + "include/GoldenModelPointwise/",
                        'golden_model_dw':          PULPNNTestFolderHW32bit + "include/GoldenModelDepthwise/",
                        'golden_model_ln_nq':       PULPNNTestFolderHW32bit + "include/GoldenModelLinearNoQuant/",
                        'golden_model_ln_q':        PULPNNTestFolderHW32bit + "include/GoldenModelLinearQuant/",
                        'golden_model_maxp':        PULPNNTestFolderHW32bit + "include/GoldenModelMaxPool/",
                        'golden_model_avgp':        PULPNNTestFolderHW32bit + "include/GoldenModelAvgPool/",
                        'golden_model_add':         PULPNNTestFolderHW32bit + "include/GoldenModelAdd/",
                        'test':                     PULPNNTestFolderHW32bit}

PULPNNSrcDirsHW64bit = {'src':                      PULPNNInstallPathHW64bit + "src/",
                        'inc':                      PULPNNInstallPathHW64bit + "include/",
                        'convolution':              PULPNNInstallPathHW64bit + "src/Convolution/",
                        'conv1d':                   PULPNNInstallPathHW64bit + "src/Convolution/",
                        'pointwise':                PULPNNInstallPathHW64bit + "src/Pointwise/",
                        'matmul':                   PULPNNInstallPathHW64bit + "src/MatrixMultiplication/",
                        'depthwise':                PULPNNInstallPathHW64bit + "src/Depthwise/",
                        'linear_nq':                PULPNNInstallPathHW64bit + "src/LinearNoQuant/",
                        'linear_q':                 PULPNNInstallPathHW64bit + "src/LinearQuant/",
                        'pooling':                  PULPNNInstallPathHW64bit + "src/Pooling/",
                        'maxpool':                  PULPNNInstallPathHW64bit + "src/Pooling/MaxPool/",
                        'avgpool':                  PULPNNInstallPathHW64bit + "src/Pooling/AvgPool/",
                        'add':                      PULPNNInstallPathHW64bit + "src/Add/",
                        'pulp_nn_include':          PULPNNTestFolderHW64bit + "include/",
                        'pulp_nn_src':              PULPNNTestFolderHW64bit + "src/",
                        'pulp_nn_convolution':      PULPNNTestFolderHW64bit + "src/Convolution/",
                        'pulp_nn_pointwise':        PULPNNTestFolderHW64bit + "src/Pointwise/",
                        'pulp_nn_matmul':           PULPNNTestFolderHW64bit + "src/MatrixMultiplication/",
                        'pulp_nn_depthwise':        PULPNNTestFolderHW64bit + "src/Depthwise/",
                        'pulp_nn_linear_nq':        PULPNNTestFolderHW64bit + "src/LinearNoQuant/",
                        'pulp_nn_linear_q':         PULPNNTestFolderHW64bit + "src/LinearQuant/",
                        'pulp_nn_maxpool':          PULPNNTestFolderHW64bit + "src/Pooling/MaxPool/",
                        'pulp_nn_avgpool':          PULPNNTestFolderHW64bit + "src/Pooling/AvgPool/",
                        'pulp_nn_add':              PULPNNTestFolderHW64bit + "src/Add/",
                        'data_allocation_matm':     PULPNNTestFolderHW64bit + "include/DataAllocationMatMul/",
                        'data_allocation_conv':     PULPNNTestFolderHW64bit + "include/DataAllocationConvolution/",
                        'data_allocation_pw':       PULPNNTestFolderHW64bit + "include/DataAllocationPointwise/",
                        'data_allocation_dw':       PULPNNTestFolderHW64bit + "include/DataAllocationDepthwise/",
                        'data_allocation_ln_nq':    PULPNNTestFolderHW64bit + "include/DataAllocationLinearNoQuant/",
                        'data_allocation_ln_q':     PULPNNTestFolderHW64bit + "include/DataAllocationLinearQuant/",
                        'data_allocation_maxp':     PULPNNTestFolderHW64bit + "include/DataAllocationMaxPool/",
                        'data_allocation_avgp':     PULPNNTestFolderHW64bit + "include/DataAllocationAvgPool/",
                        'data_allocation_add':      PULPNNTestFolderHW64bit + "include/DataAllocationAdd/",
                        'golden_model_matm':        PULPNNTestFolderHW64bit + "include/GoldenModelMatMul/",
                        'golden_model_conv':        PULPNNTestFolderHW64bit + "include/GoldenModelConvolution/",
                        'golden_model_pw':          PULPNNTestFolderHW64bit + "include/GoldenModelPointwise/",
                        'golden_model_dw':          PULPNNTestFolderHW64bit + "include/GoldenModelDepthwise/",
                        'golden_model_ln_nq':       PULPNNTestFolderHW64bit + "include/GoldenModelLinearNoQuant/",
                        'golden_model_ln_q':        PULPNNTestFolderHW64bit + "include/GoldenModelLinearQuant/",
                        'golden_model_maxp':        PULPNNTestFolderHW64bit + "include/GoldenModelMaxPool/",
                        'golden_model_avgp':        PULPNNTestFolderHW64bit + "include/GoldenModelAvgPool/",
                        'golden_model_add':         PULPNNTestFolderHW64bit + "include/GoldenModelAdd/",
                        'test':                     PULPNNTestFolderHW64bit}

# XpulpNN-mixed sources path
PULPNNSrcDirsExtHW32bit = {'src':                   PULPNNInstallPathExtHW32bit + "src/",
                           'inc':                   PULPNNInstallPathExtHW32bit + "include/",
                           'convolution':           PULPNNInstallPathExtHW32bit + "src/Convolution/",
                           'pointwise':             PULPNNInstallPathExtHW32bit + "src/Pointwise/",
                           'matmul':                PULPNNInstallPathExtHW32bit + "src/MatrixMultiplication/",
                           'depthwise':             PULPNNInstallPathExtHW32bit + "src/Depthwise/",
                           'linear_nq':             PULPNNInstallPathExtHW32bit + "src/LinearNoQuant/",
                           'linear_q':              PULPNNInstallPathExtHW32bit + "src/LinearQuant/",
                           'pooling':               PULPNNInstallPathExtHW32bit + "src/Pooling/",
                           'maxpool':               PULPNNInstallPathExtHW32bit + "src/Pooling/MaxPool/",
                           'avgpool':               PULPNNInstallPathExtHW32bit + "src/Pooling/AvgPool/",
                           'add':                   PULPNNInstallPathExtHW32bit + "src/Add/",
                           'pulp_nn_include':       PULPNNTestFolderExtHW32bit + "include/",
                           'pulp_nn_src':           PULPNNTestFolderExtHW32bit + "src/",
                           'pulp_nn_convolution':   PULPNNTestFolderExtHW32bit + "src/Convolution/",
                           'pulp_nn_pointwise':     PULPNNTestFolderExtHW32bit + "src/Pointwise/",
                           'pulp_nn_matmul':        PULPNNTestFolderExtHW32bit + "src/MatrixMultiplication/",
                           'pulp_nn_depthwise':     PULPNNTestFolderExtHW32bit + "src/Depthwise/",
                           'pulp_nn_linear_nq':     PULPNNTestFolderExtHW32bit + "src/LinearNoQuant/",
                           'pulp_nn_linear_q':      PULPNNTestFolderExtHW32bit + "src/LinearQuant/",
                           'pulp_nn_maxpool':       PULPNNTestFolderExtHW32bit + "src/Pooling/MaxPool/",
                           'pulp_nn_avgpool':       PULPNNTestFolderExtHW32bit + "src/Pooling/AvgPool/",
                           'pulp_nn_add':           PULPNNTestFolderExtHW32bit + "src/Add/",
                           'data_allocation_matm':  PULPNNTestFolderExtHW32bit + "include/DataAllocationMatMul/",
                           'data_allocation_conv':  PULPNNTestFolderExtHW32bit + "include/DataAllocationConvolution/",
                           'data_allocation_pw':    PULPNNTestFolderExtHW32bit + "include/DataAllocationPointwise/",
                           'data_allocation_dw':    PULPNNTestFolderExtHW32bit + "include/DataAllocationDepthwise/",
                           'data_allocation_ln_nq': PULPNNTestFolderExtHW32bit + "include/DataAllocationLinearNoQuant/",
                           'data_allocation_ln_q':  PULPNNTestFolderExtHW32bit + "include/DataAllocationLinearQuant/",
                           'data_allocation_maxp':  PULPNNTestFolderExtHW32bit + "include/DataAllocationMaxPool/",
                           'data_allocation_avgp':  PULPNNTestFolderExtHW32bit + "include/DataAllocationAvgPool/",
                           'data_allocation_add':   PULPNNTestFolderExtHW32bit + "include/DataAllocationAdd/",
                           'golden_model_matm':     PULPNNTestFolderExtHW32bit + "include/GoldenModelMatMul/",
                           'golden_model_conv':     PULPNNTestFolderExtHW32bit + "include/GoldenModelConvolution/",
                           'golden_model_pw':       PULPNNTestFolderExtHW32bit + "include/GoldenModelPointwise/",
                           'golden_model_dw':       PULPNNTestFolderExtHW32bit + "include/GoldenModelDepthwise/",
                           'golden_model_ln_nq':    PULPNNTestFolderExtHW32bit + "include/GoldenModelLinearNoQuant/",
                           'golden_model_ln_q':     PULPNNTestFolderExtHW32bit + "include/GoldenModelLinearQuant/",
                           'golden_model_maxp':     PULPNNTestFolderExtHW32bit + "include/GoldenModelMaxPool/",
                           'golden_model_avgp':     PULPNNTestFolderExtHW32bit + "include/GoldenModelAvgPool/",
                           'golden_model_add':      PULPNNTestFolderExtHW32bit + "include/GoldenModelAdd/",
                           'test':                  PULPNNTestFolderExtHW32bit}

PULPNNSrcDirsExtHW64bit = {'src':                   PULPNNInstallPathExtHW64bit + "src/",
                           'inc':                   PULPNNInstallPathExtHW64bit + "include/",
                           'convolution':           PULPNNInstallPathExtHW64bit + "src/Convolution/",
                           'pointwise':             PULPNNInstallPathExtHW64bit + "src/Pointwise/",
                           'matmul':                PULPNNInstallPathExtHW64bit + "src/MatrixMultiplication/",
                           'depthwise':             PULPNNInstallPathExtHW64bit + "src/Depthwise/",
                           'linear_nq':             PULPNNInstallPathExtHW64bit + "src/LinearNoQuant/",
                           'linear_q':              PULPNNInstallPathExtHW64bit + "src/LinearQuant/",
                           'pooling':               PULPNNInstallPathExtHW64bit + "src/Pooling/",
                           'maxpool':               PULPNNInstallPathExtHW64bit + "src/Pooling/MaxPool/",
                           'avgpool':               PULPNNInstallPathExtHW64bit + "src/Pooling/AvgPool/",
                           'add':                   PULPNNInstallPathExtHW64bit + "src/Add/",
                           'pulp_nn_include':       PULPNNTestFolderExtHW64bit + "include/",
                           'pulp_nn_src':           PULPNNTestFolderExtHW64bit + "src/",
                           'pulp_nn_convolution':   PULPNNTestFolderExtHW64bit + "src/Convolution/",
                           'pulp_nn_pointwise':     PULPNNTestFolderExtHW64bit + "src/Pointwise/",
                           'pulp_nn_matmul':        PULPNNTestFolderExtHW64bit + "src/MatrixMultiplication/",
                           'pulp_nn_depthwise':     PULPNNTestFolderExtHW64bit + "src/Depthwise/",
                           'pulp_nn_linear_nq':     PULPNNTestFolderExtHW64bit + "src/LinearNoQuant/",
                           'pulp_nn_linear_q':      PULPNNTestFolderExtHW64bit + "src/LinearQuant/",
                           'pulp_nn_maxpool':       PULPNNTestFolderExtHW64bit + "src/Pooling/MaxPool/",
                           'pulp_nn_avgpool':       PULPNNTestFolderExtHW64bit + "src/Pooling/AvgPool/",
                           'pulp_nn_add':           PULPNNTestFolderExtHW64bit + "src/Add/",
                           'data_allocation_matm':  PULPNNTestFolderExtHW64bit + "include/DataAllocationMatMul/",
                           'data_allocation_conv':  PULPNNTestFolderExtHW64bit + "include/DataAllocationConvolution/",
                           'data_allocation_pw':    PULPNNTestFolderExtHW64bit + "include/DataAllocationPointwise/",
                           'data_allocation_dw':    PULPNNTestFolderExtHW64bit + "include/DataAllocationDepthwise/",
                           'data_allocation_ln_nq': PULPNNTestFolderExtHW64bit + "include/DataAllocationLinearNoQuant/",
                           'data_allocation_ln_q':  PULPNNTestFolderExtHW64bit + "include/DataAllocationLinearQuant/",
                           'data_allocation_maxp':  PULPNNTestFolderExtHW64bit + "include/DataAllocationMaxPool/",
                           'data_allocation_avgp':  PULPNNTestFolderExtHW64bit + "include/DataAllocationAvgPool/",
                           'data_allocation_add':   PULPNNTestFolderExtHW64bit + "include/DataAllocationAdd/",
                           'golden_model_matm':     PULPNNTestFolderExtHW64bit + "include/GoldenModelMatMul/",
                           'golden_model_conv':     PULPNNTestFolderExtHW64bit + "include/GoldenModelConvolution/",
                           'golden_model_pw':       PULPNNTestFolderExtHW64bit + "include/GoldenModelPointwise/",
                           'golden_model_dw':       PULPNNTestFolderExtHW64bit + "include/GoldenModelDepthwise/",
                           'golden_model_ln_nq':    PULPNNTestFolderExtHW64bit + "include/GoldenModelLinearNoQuant/",
                           'golden_model_ln_q':     PULPNNTestFolderExtHW64bit + "include/GoldenModelLinearQuant/",
                           'golden_model_maxp':     PULPNNTestFolderExtHW64bit + "include/GoldenModelMaxPool/",
                           'golden_model_avgp':     PULPNNTestFolderExtHW64bit + "include/GoldenModelAvgPool/",
                           'golden_model_add':      PULPNNTestFolderExtHW64bit + "include/GoldenModelAdd/",
                           'test':                  PULPNNTestFolderExtHW64bit}

# Folder generation
def mkdir_p(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

# Sources folder generation
def mkdir_src(act_prec, ext):
        if ext == 'XpulpV2':
            if act_prec == '32bit':
                mkdir_p(PULPNNSrcDirsSW32bit['src'])
                mkdir_p(PULPNNSrcDirsSW32bit['inc'])
                mkdir_p(PULPNNSrcDirsSW32bit['convolution'])
                mkdir_p(PULPNNSrcDirsSW32bit['pointwise'])
                mkdir_p(PULPNNSrcDirsSW32bit['matmul'])
                mkdir_p(PULPNNSrcDirsSW32bit['depthwise'])
                mkdir_p(PULPNNSrcDirsSW32bit['linear_nq'])
                mkdir_p(PULPNNSrcDirsSW32bit['linear_q'])
                mkdir_p(PULPNNSrcDirsSW32bit['pooling'])
                mkdir_p(PULPNNSrcDirsSW32bit['maxpool'])
                mkdir_p(PULPNNSrcDirsSW32bit['avgpool'])
                mkdir_p(PULPNNSrcDirsSW32bit['add'])
            elif act_prec == '64bit':
                mkdir_p(PULPNNSrcDirsSW64bit['src'])
                mkdir_p(PULPNNSrcDirsSW64bit['inc'])
                mkdir_p(PULPNNSrcDirsSW64bit['convolution'])
                mkdir_p(PULPNNSrcDirsSW64bit['pointwise'])
                mkdir_p(PULPNNSrcDirsSW64bit['matmul'])
                mkdir_p(PULPNNSrcDirsSW64bit['depthwise'])
                mkdir_p(PULPNNSrcDirsSW64bit['linear_nq'])
                mkdir_p(PULPNNSrcDirsSW64bit['linear_q'])
                mkdir_p(PULPNNSrcDirsSW64bit['pooling'])
                mkdir_p(PULPNNSrcDirsSW64bit['maxpool'])
                mkdir_p(PULPNNSrcDirsSW64bit['avgpool'])
                mkdir_p(PULPNNSrcDirsSW64bit['add'])

        elif ext == 'XpulpNN':       
            if act_prec == '32bit':
                mkdir_p(PULPNNSrcDirsHW32bit['src'])
                mkdir_p(PULPNNSrcDirsHW32bit['inc'])
                mkdir_p(PULPNNSrcDirsHW32bit['convolution'])
                mkdir_p(PULPNNSrcDirsHW32bit['pointwise'])
                mkdir_p(PULPNNSrcDirsHW32bit['matmul'])
                mkdir_p(PULPNNSrcDirsHW32bit['depthwise'])
                mkdir_p(PULPNNSrcDirsHW32bit['linear_nq'])
                mkdir_p(PULPNNSrcDirsHW32bit['linear_q'])
                mkdir_p(PULPNNSrcDirsHW32bit['pooling'])
                mkdir_p(PULPNNSrcDirsHW32bit['maxpool'])
                mkdir_p(PULPNNSrcDirsHW32bit['avgpool'])
                mkdir_p(PULPNNSrcDirsHW32bit['add'])
            elif act_prec == '64bit':
                mkdir_p(PULPNNSrcDirsHW64bit['src'])
                mkdir_p(PULPNNSrcDirsHW64bit['inc'])
                mkdir_p(PULPNNSrcDirsHW64bit['convolution'])
                mkdir_p(PULPNNSrcDirsHW64bit['pointwise'])
                mkdir_p(PULPNNSrcDirsHW64bit['matmul'])
                mkdir_p(PULPNNSrcDirsHW64bit['depthwise'])
                mkdir_p(PULPNNSrcDirsHW64bit['linear_nq'])
                mkdir_p(PULPNNSrcDirsHW64bit['linear_q'])
                mkdir_p(PULPNNSrcDirsHW64bit['pooling'])
                mkdir_p(PULPNNSrcDirsHW64bit['maxpool'])
                mkdir_p(PULPNNSrcDirsHW64bit['avgpool'])
                mkdir_p(PULPNNSrcDirsHW64bit['add'])

        elif ext == 'XpulpNN-mixed':
            if act_prec == '32bit':
                mkdir_p(PULPNNSrcDirsExtHW32bit['src'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['inc'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['convolution'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['pointwise'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['matmul'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['depthwise'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['linear_nq'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['linear_q'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['pooling'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['maxpool'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['avgpool'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['add'])
            elif act_prec == '64bit':
                mkdir_p(PULPNNSrcDirsExtHW64bit['src'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['inc'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['convolution'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['pointwise'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['matmul'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['depthwise'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['linear_nq'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['linear_q'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['pooling'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['maxpool'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['avgpool'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['add'])

# Testing folder generation
def mkdir_test(kernel, act_prec, ext):
    if ext == 'XpulpNN':
        if act_prec == '32bit':
            try:
                os.remove(PULPNNSrcDirsHW32bit['test'])
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    pass
            mkdir_p(PULPNNTestFolderHW32bit)
            mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_src'])
            mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_include'])
            if kernel=='matmul':
                mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsHW32bit['data_allocation_matm'])
                mkdir_p(PULPNNSrcDirsHW32bit['golden_model_matm'])            
            elif kernel=='pointwise':
                mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_pointwise'])
                mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsHW32bit['data_allocation_pw'])
                mkdir_p(PULPNNSrcDirsHW32bit['golden_model_pw'])
            elif kernel=='convolution':
                mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_convolution'])
                mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsHW32bit['data_allocation_conv'])
                mkdir_p(PULPNNSrcDirsHW32bit['golden_model_conv'])
            elif kernel=='depthwise':
                mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_depthwise'])
                mkdir_p(PULPNNSrcDirsHW32bit['data_allocation_dw'])
                mkdir_p(PULPNNSrcDirsHW32bit['golden_model_dw'])
            elif kernel=='linear_no_quant':
                mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_linear_nq'])
                mkdir_p(PULPNNSrcDirsHW32bit['data_allocation_ln_nq'])
                mkdir_p(PULPNNSrcDirsHW32bit['golden_model_ln_nq'])
            elif kernel=='linear_quant':
                mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_linear_q'])
                mkdir_p(PULPNNSrcDirsHW32bit['data_allocation_ln_q'])
                mkdir_p(PULPNNSrcDirsHW32bit['golden_model_ln_q'])
            elif kernel=='maxpool':
                mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_maxpool'])
                mkdir_p(PULPNNSrcDirsHW32bit['data_allocation_maxp'])
                mkdir_p(PULPNNSrcDirsHW32bit['golden_model_maxp'])
            elif kernel=='avgpool':
                mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_avgpool'])
                mkdir_p(PULPNNSrcDirsHW32bit['data_allocation_avgp'])
                mkdir_p(PULPNNSrcDirsHW32bit['golden_model_avgp'])
            elif kernel=='add':
                mkdir_p(PULPNNSrcDirsHW32bit['pulp_nn_add'])
                mkdir_p(PULPNNSrcDirsHW32bit['data_allocation_add'])
                mkdir_p(PULPNNSrcDirsHW32bit['golden_model_add'])

        elif act_prec == '64bit':
            try:
                os.remove(PULPNNSrcDirsHW64bit['test'])
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    pass
            mkdir_p(PULPNNTestFolderHW64bit)
            mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_src'])
            mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_include'])
            if kernel=='matmul':
                mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsHW64bit['data_allocation_matm'])
                mkdir_p(PULPNNSrcDirsHW64bit['golden_model_matm'])            
            elif kernel=='pointwise':
                mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_pointwise'])
                mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsHW64bit['data_allocation_pw'])
                mkdir_p(PULPNNSrcDirsHW64bit['golden_model_pw'])
            elif kernel=='convolution':
                mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_convolution'])
                mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsHW64bit['data_allocation_conv'])
                mkdir_p(PULPNNSrcDirsHW64bit['golden_model_conv'])
            elif kernel=='depthwise':
                mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_depthwise'])
                mkdir_p(PULPNNSrcDirsHW64bit['data_allocation_dw'])
                mkdir_p(PULPNNSrcDirsHW64bit['golden_model_dw'])
            elif kernel=='linear_no_quant':
                mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_linear_nq'])
                mkdir_p(PULPNNSrcDirsHW64bit['data_allocation_ln_nq'])
                mkdir_p(PULPNNSrcDirsHW64bit['golden_model_ln_nq'])
            elif kernel=='linear_quant':
                mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_linear_q'])
                mkdir_p(PULPNNSrcDirsHW64bit['data_allocation_ln_q'])
                mkdir_p(PULPNNSrcDirsHW64bit['golden_model_ln_q'])
            elif kernel=='maxpool':
                mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_maxpool'])
                mkdir_p(PULPNNSrcDirsHW64bit['data_allocation_maxp'])
                mkdir_p(PULPNNSrcDirsHW64bit['golden_model_maxp'])
            elif kernel=='avgpool':
                mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_avgpool'])
                mkdir_p(PULPNNSrcDirsHW64bit['data_allocation_avgp'])
                mkdir_p(PULPNNSrcDirsHW64bit['golden_model_avgp'])
            elif kernel=='add':
                mkdir_p(PULPNNSrcDirsHW64bit['pulp_nn_add'])
                mkdir_p(PULPNNSrcDirsHW64bit['data_allocation_add'])
                mkdir_p(PULPNNSrcDirsHW64bit['golden_model_add'])

    elif ext == 'XpulpNN-mixed':
        if act_prec == '32bit':
            try:
                os.remove(PULPNNSrcDirsExtHW32bit['test'])
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    pass
            mkdir_p(PULPNNTestFolderExtHW32bit)
            mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_src'])
            mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_include'])
            if kernel=='matmul':
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['data_allocation_matm'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['golden_model_matm'])
            elif kernel=='pointwise':
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_pointwise'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['data_allocation_pw'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['golden_model_pw'])
            elif kernel=='convolution':
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_convolution'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['data_allocation_conv'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['golden_model_conv'])
            elif kernel=='depthwise':
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_depthwise'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['data_allocation_dw'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['golden_model_dw'])
            elif kernel=='linear_no_quant':
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_linear_nq'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['data_allocation_ln_nq'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['golden_model_ln_nq'])
            elif kernel=='linear_quant':
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_linear_q'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['data_allocation_ln_q'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['golden_model_ln_q'])
            elif kernel=='maxpool':
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_maxpool'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['data_allocation_maxp'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['golden_model_maxp'])
            elif kernel=='avgpool':
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_avgpool'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['data_allocation_avgp'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['golden_model_avgp'])
            elif kernel=='add':
                mkdir_p(PULPNNSrcDirsExtHW32bit['pulp_nn_add'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['data_allocation_add'])
                mkdir_p(PULPNNSrcDirsExtHW32bit['golden_model_add'])

        elif act_prec == '64bit':
            try:
                os.remove(PULPNNSrcDirsExtHW64bit['test'])
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    pass
            mkdir_p(PULPNNTestFolderExtHW64bit)
            mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_src'])
            mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_include'])
            if kernel=='matmul':
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['data_allocation_matm'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['golden_model_matm'])
            elif kernel=='pointwise':
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_pointwise'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['data_allocation_pw'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['golden_model_pw'])
            elif kernel=='convolution':
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_convolution'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['data_allocation_conv'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['golden_model_conv'])
            elif kernel=='depthwise':
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_depthwise'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['data_allocation_dw'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['golden_model_dw'])
            elif kernel=='linear_no_quant':
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_linear_nq'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['data_allocation_ln_nq'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['golden_model_ln_nq'])
            elif kernel=='linear_quant':
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_linear_q'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['data_allocation_ln_q'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['golden_model_ln_q'])
            elif kernel=='maxpool':
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_maxpool'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['data_allocation_maxp'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['golden_model_maxp'])
            elif kernel=='avgpool':
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_avgpool'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['data_allocation_avgp'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['golden_model_avgp'])
            elif kernel=='add':
                mkdir_p(PULPNNSrcDirsExtHW64bit['pulp_nn_add'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['data_allocation_add'])
                mkdir_p(PULPNNSrcDirsExtHW64bit['golden_model_add'])

    elif ext == 'XpulpV2':
        if act_prec == '32bit':
            try:
                os.remove(PULPNNSrcDirsSW32bit['test'])
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    pass
            mkdir_p(PULPNNTestFolderSW32bit)
            mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_src'])
            mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_include'])
            if kernel=='matmul':
                mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsSW32bit['data_allocation_matm'])
                mkdir_p(PULPNNSrcDirsSW32bit['golden_model_matm'])            
            elif kernel=='pointwise':
                mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_pointwise'])
                mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsSW32bit['data_allocation_pw'])
                mkdir_p(PULPNNSrcDirsSW32bit['golden_model_pw'])
            elif kernel=='convolution':
                mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_convolution'])
                mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsSW32bit['data_allocation_conv'])
                mkdir_p(PULPNNSrcDirsSW32bit['golden_model_conv'])
            elif kernel=='depthwise':
                mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_depthwise'])
                mkdir_p(PULPNNSrcDirsSW32bit['data_allocation_dw'])
                mkdir_p(PULPNNSrcDirsSW32bit['golden_model_dw'])
            elif kernel=='linear_no_quant':
                mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_linear_nq'])
                mkdir_p(PULPNNSrcDirsSW32bit['data_allocation_ln_nq'])
                mkdir_p(PULPNNSrcDirsSW32bit['golden_model_ln_nq'])
            elif kernel=='linear_quant':
                mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_linear_q'])
                mkdir_p(PULPNNSrcDirsSW32bit['data_allocation_ln_q'])
                mkdir_p(PULPNNSrcDirsSW32bit['golden_model_ln_q'])
            elif kernel=='maxpool':
                mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_maxpool'])
                mkdir_p(PULPNNSrcDirsSW32bit['data_allocation_maxp'])
                mkdir_p(PULPNNSrcDirsSW32bit['golden_model_maxp'])
            elif kernel=='avgpool':
                mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_avgpool'])
                mkdir_p(PULPNNSrcDirsSW32bit['data_allocation_avgp'])
                mkdir_p(PULPNNSrcDirsSW32bit['golden_model_avgp'])
            elif kernel=='add':
                mkdir_p(PULPNNSrcDirsSW32bit['pulp_nn_add'])
                mkdir_p(PULPNNSrcDirsSW32bit['data_allocation_add'])
                mkdir_p(PULPNNSrcDirsSW32bit['golden_model_add'])

        elif act_prec == '64bit':
            try:
                os.remove(PULPNNSrcDirsSW64bit['test'])
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    pass
            mkdir_p(PULPNNTestFolderSW64bit)
            mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_src'])
            mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_include'])
            if kernel=='matmul':
                mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsSW64bit['data_allocation_matm'])
                mkdir_p(PULPNNSrcDirsSW64bit['golden_model_matm'])            
            elif kernel=='pointwise':
                mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_pointwise'])
                mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsSW64bit['data_allocation_pw'])
                mkdir_p(PULPNNSrcDirsSW64bit['golden_model_pw'])
            elif kernel=='convolution':
                mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_convolution'])
                mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_matmul'])
                mkdir_p(PULPNNSrcDirsSW64bit['data_allocation_conv'])
                mkdir_p(PULPNNSrcDirsSW64bit['golden_model_conv'])
            elif kernel=='depthwise':
                mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_depthwise'])
                mkdir_p(PULPNNSrcDirsSW64bit['data_allocation_dw'])
                mkdir_p(PULPNNSrcDirsSW64bit['golden_model_dw'])
            elif kernel=='linear_no_quant':
                mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_linear_nq'])
                mkdir_p(PULPNNSrcDirsSW64bit['data_allocation_ln_nq'])
                mkdir_p(PULPNNSrcDirsSW64bit['golden_model_ln_nq'])
            elif kernel=='linear_quant':
                mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_linear_q'])
                mkdir_p(PULPNNSrcDirsSW64bit['data_allocation_ln_q'])
                mkdir_p(PULPNNSrcDirsSW64bit['golden_model_ln_q'])
            elif kernel=='maxpool':
                mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_maxpool'])
                mkdir_p(PULPNNSrcDirsSW64bit['data_allocation_maxp'])
                mkdir_p(PULPNNSrcDirsSW64bit['golden_model_maxp'])
            elif kernel=='avgpool':
                mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_avgpool'])
                mkdir_p(PULPNNSrcDirsSW64bit['data_allocation_avgp'])
                mkdir_p(PULPNNSrcDirsSW64bit['golden_model_avgp'])
            elif kernel=='add':
                mkdir_p(PULPNNSrcDirsSW64bit['pulp_nn_add'])
                mkdir_p(PULPNNSrcDirsSW64bit['data_allocation_add'])
                mkdir_p(PULPNNSrcDirsSW64bit['golden_model_add'])
