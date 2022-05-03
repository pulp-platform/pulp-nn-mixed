<%! import math %>
<%
def iu(sgn):
    return 'i' if sgn else 'u'
in_w = config.kernel.in_data_t
out_w = config.kernel.out_data_t
%>\
%if config.api == 'PULPNNConvolve':
#if (KERNEL == ${in_w}${out_w}${config.kernel.wt_data_t}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
${config.fn_name}(IN_INT8_L1,
                    IM2COL_L1,
                    BIAS_L1,
                    OUT_L1,
                    WEIGHT_INT8_L1,
                    KAPPA_L1,
                    LAMBDA_L1,
                    OUT_MULT,
                    OUT_SHIFT,
                    DIM_IM_IN_X,
                    DIM_IM_IN_Y,
                    CH_IM_IN,
                    DIM_IM_OUT_X,
                    DIM_IM_OUT_Y,
                    CH_IM_OUT,
                    DIM_KERNEL_X,
                    DIM_KERNEL_Y,
                    PADDING_Y_TOP,
                    PADDING_Y_BOTTOM,
                    PADDING_X_LEFT,
                    PADDING_X_RIGHT,
                    STRIDE_X,
                    STRIDE_Y,
%if config.layer.bn == True:
                    1,
%else:
                    0,
%endif
%if config.layer.relu == True:
                    1
%else:
                    0
%endif
                        );
#endif
%elif config.api == 'PULPNNConvolvePointwise':
#if (KERNEL == ${in_w}${out_w}${config.kernel.wt_data_t}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
${config.fn_name}(IN_INT8_L1,
                    IM2COL_L1,
                    BIAS_L1,
                    OUT_L1,
                    WEIGHT_INT8_L1,
                    KAPPA_L1,
                    LAMBDA_L1,
                    OUT_MULT,
                    OUT_SHIFT,
                    DIM_IM_IN_X,
                    DIM_IM_IN_Y,
                    CH_IM_IN,
                    DIM_IM_OUT_X,
                    DIM_IM_OUT_Y,
                    CH_IM_OUT,
                    DIM_KERNEL_X,
                    DIM_KERNEL_Y,
                    PADDING_Y_TOP,
                    PADDING_Y_BOTTOM,
                    PADDING_X_LEFT,
                    PADDING_X_RIGHT,
                    STRIDE_X,
                    STRIDE_Y,
%if config.layer.bn == True:
                    1,
%else:
                    0,
%endif
%if config.layer.relu == True:
                    1
%else:
                    0
%endif
                        );
#endif
%elif config.api == 'PULPNNConvolveDepthwise':
#if (KERNEL == ${in_w}${out_w}${config.kernel.wt_data_t}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
${config.fn_name}(IN_INT8_L1_CHW,
                    IM2COL_L1,
                    BIAS_L1,
                    OUT_L1,
                    WEIGHT_INT8_L1_CHW,
                    WTBUFF_L1,
                    KAPPA_L1,
                    LAMBDA_L1,
                    OUT_MULT,
                    OUT_SHIFT,
                    DIM_IM_IN_X,
                    DIM_IM_IN_Y,
                    CH_IM_IN,
                    DIM_IM_OUT_X,
                    DIM_IM_OUT_Y,
                    CH_IM_OUT,
                    DIM_KERNEL_X,
                    DIM_KERNEL_Y,
                    PADDING_Y_TOP,
                    PADDING_Y_BOTTOM,
                    PADDING_X_LEFT,
                    PADDING_X_RIGHT,
                    STRIDE_X,
                    STRIDE_Y,
%if config.layer.bn == True:
                    1,
%else:
                    0,
%endif
%if config.layer.relu == True:
                    1
%else:
                    0
%endif
                        );
#endif
%elif config.api=="PULPNNMatMul":
#if (KERNEL == ${in_w}${out_w}${config.kernel.wt_data_t}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
OUT_L1 = ${config.fn_name}(IN_INT8_L1,
                    BIAS_L1,
                    OUT_L1,
                    OUT_L1 + CH_IM_OUT >> ${int(math.log2((int(8/config.kernel.out_data_t))))},
                    WEIGHT_INT8_L1,
                    KAPPA_L1,
                    LAMBDA_L1,
                    OUT_MULT,
                    OUT_SHIFT,
                    (CH_IM_OUT * DIM_KERNEL_X * DIM_KERNEL_Y) << 1,
                    CH_IM_OUT,
%if config.layer.bn == True:
                    1,
%else:
                    0,
%endif
%if config.layer.relu == True:
                    1
%else:
                    0
%endif
                        );
#endif
%elif config.api == 'PULPNNLinearNoQuant':
#if (KERNEL == ${in_w}${config.kernel.wt_data_t}) && (SIGNED == ${f"{int(config.kernel.in_signed)}"})
${config.fn_name}(IN_INT8_L1,
                    BIAS_L1,
                    OUT_L1,
                    WEIGHT_INT8_L1,
                    DIM_IM_IN_X*DIM_IM_IN_Y*CH_IM_IN,
                    CH_IM_OUT);
#endif
%elif config.api == 'PULPNNLinearQuant':
#if (KERNEL == ${in_w}${out_w}${config.kernel.wt_data_t}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
${config.fn_name}(IN_INT8_L1,
                    BIAS_L1,
                    OUT_L1,
                    WEIGHT_INT8_L1,
                    KAPPA_L1,
                    LAMBDA_L1,
                    OUT_MULT,
                    OUT_SHIFT,
                    DIM_IM_IN_X*DIM_IM_IN_Y*CH_IM_IN,
                    CH_IM_OUT,
%if config.layer.bn == True:
                    1,
%else:
                    0,
%endif
%if config.layer.relu == True:
                    1
%else:
                    0
%endif
                        );
#endif
%elif config.api == 'PULPNNMaxPool':
#if (KERNEL == ${in_w}) && (SIGNED == ${f"{int(config.kernel.in_signed)}"})
${config.fn_name}(IN_INT8_L1,
                    OUT_L1,
                    DIM_IM_IN_X,
                    DIM_IM_IN_Y,
                    CH_IM_IN,
                    DIM_IM_OUT_X,
                    DIM_IM_OUT_Y,
                    POOL_KERNEL,
                    POOL_KERNEL,
                    PADDING_Y_TOP,
                    PADDING_Y_BOTTOM,
                    PADDING_X_LEFT,
                    PADDING_X_RIGHT,
                    POOL_STRIDE,
                    POOL_STRIDE);
#endif
%elif config.api == 'PULPNNAvgPoolNew':
#if (KERNEL == ${in_w}${out_w}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
${config.fn_name}(IN_INT8_L1,
                    OUT_L1,
                    ${"OUT_MULT" if config.layer.bn else "1"},
                    ${"OUT_SHIFT" if config.layer.bn else "0"},
                    ${"OUT_ADD" if config.layer.bn else "0"},
                    DIM_IM_IN_X,
                    DIM_IM_IN_Y,
                    CH_IM_IN,
                    DIM_IM_OUT_X,
                    DIM_IM_OUT_Y,
                    POOL_KERNEL,
                    POOL_KERNEL,
                    PADDING_Y_TOP,
                    PADDING_Y_BOTTOM,
                    PADDING_X_LEFT,
                    PADDING_X_RIGHT,
                    POOL_STRIDE,
                    POOL_STRIDE,
                    ${int(config.layer.bn)});
#endif
%elif config.api == 'PULPNNAdd':
#if (KERNEL == ${config.in1_data_t}${config.in2_data_t})
${config.fn_name}(IN1_INT8_L1,
                    IN2_INT8_L1,
                    OUT_L1,
                    OUT_MULT1,
                    OUT_MULT2,
                    OUT_SHIFT,
                    DIM_IM_IN_X,
                    DIM_IM_IN_Y,
                    CH_IM_IN);
#endif
%endif
