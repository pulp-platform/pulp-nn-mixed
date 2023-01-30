<%
act_prec = int(config.kernel.act_prec[0:2])
act_t = f"int{act_prec}_t"

def u_(sgn):
    return '' if sgn else 'u'
pt_in = f"{u_(config.kernel.in_signed)}int8_t"
pt_out = f"{u_(config.kernel.out_signed)}int8_t"
%>\
%if config.api == "PULPNNConvolve":
void ${config.fn_name}(
                        ${pt_in} *pIn,
                        ${pt_in} *pIm2ColBuffer,
                        int8_t *pBias,
                        ${pt_out} *pOut,
                        int8_t *pWeight,
                        ${act_t} *pKappa,
                        ${act_t} *pLambda,
                        uint16_t out_mul,
                        uint16_t out_shift,
                        uint16_t dim_in_x,
                        uint16_t dim_in_y,
                        uint16_t ch_in,
                        uint16_t dim_out_x,
                        uint16_t dim_out_y,
                        uint16_t ch_out,
                        uint16_t dim_kernel_x,
                        uint16_t dim_kernel_y,
                        uint16_t padding_y_top,
                        uint16_t padding_y_bottom,
                        uint16_t padding_x_left,
                        uint16_t padding_x_right,
                        uint16_t stride_x,
                        uint16_t stride_y,
                        uint8_t flag_relu,
                        uint8_t flag_batchnorm);
%elif config.api == "PULPNNConvolve1D":
void ${config.fn_name}(
                        ${pt_in} *pIn,
                        ${pt_in} *pIm2ColBuffer,
                        int8_t *pBias,
                        ${pt_out} *pOut,
                        int8_t *pWeight,
                        ${act_t} *pKappa,
                        ${act_t} *pLambda,
                        uint16_t out_mul,
                        uint16_t out_shift,
                        uint16_t dim_in_x,
                        uint16_t ch_in,
                        uint16_t dim_out_x,
                        uint16_t ch_out,
                        uint16_t dim_kernel_x,
                        uint16_t padding_x_left,
                        uint16_t padding_x_right,
                        uint16_t stride_x,
                        uint16_t dilation_x,
                        uint8_t flag_relu,
                        uint8_t flag_batchnorm);
%elif config.api == "PULPNNConvolvePointwise":
void ${config.fn_name}(
                        ${pt_in} *pIn,
                        ${pt_in} *pIm2ColBuffer,
                        int8_t *pBias,
                        ${pt_out} *pOut,
                        int8_t *pWeight,
                        ${act_t} *pKappa,
                        ${act_t} *pLambda,
                        uint16_t out_mul,
                        uint16_t out_shift,
                        uint16_t dim_in_x,
                        uint16_t dim_in_y,
                        uint16_t ch_in,
                        uint16_t dim_out_x,
                        uint16_t dim_out_y,
                        uint16_t ch_out,
                        uint16_t dim_kernel_x,
                        uint16_t dim_kernel_y,
                        uint16_t padding_y_top,
                        uint16_t padding_y_bottom,
                        uint16_t padding_x_left,
                        uint16_t padding_x_right,
                        uint16_t stride_x,
                        uint16_t stride_y,
                        uint8_t flag_relu,
                        uint8_t flag_batchnorm);
% elif config.api=="PULPNNMatMul":
uint8_t *${config.fn_name}(
                        ${pt_in} *pIn,
                        int8_t *pBias,
                        ${pt_out} *pOut,
                        ${pt_out} *pOut2,
%if config.kernel.matmul_fmt == '4x4':
                        ${pt_out} *pOut3,
                        ${pt_out} *pOut4,
%endif
                        int8_t *pWeight,
                        ${act_t} *pKappa,
                        ${act_t} *pLambda,
                        uint16_t out_mul,
                        uint16_t out_shift,
                        uint16_t num_col_im2col,
                        uint16_t ch_out,
                        uint8_t flag_relu,
                        uint8_t flag_batchnorm);
% elif config.api=="PULPNNConvolveDepthwise":
void ${config.fn_name}(
                        ${pt_in} *pIn,
                        ${pt_in} *pIm2ColBuffer,
                        int8_t *pBias,
                        ${pt_out} *pOut,
                        int8_t *pWeight,
                        int8_t *pWtBuffer,
                        ${act_t} *pKappa,
                        ${act_t} *pLambda,
                        uint16_t out_mul,
                        uint16_t out_shift,
                        uint16_t dim_in_x,
                        uint16_t dim_in_y,
                        uint16_t ch_in,
                        uint16_t dim_out_x,
                        uint16_t dim_out_y,
                        uint16_t ch_out,
                        uint16_t dim_kernel_x,
                        uint16_t dim_kernel_y,
                        uint16_t padding_y_top,
                        uint16_t padding_y_bottom,
                        uint16_t padding_x_left,
                        uint16_t padding_x_right,
                        uint16_t stride_x,
                        uint16_t stride_y,
                        uint8_t flag_relu,
                        uint8_t flag_batchnorm);
%elif config.api=="PULPNNLinearNoQuant":
void ${config.fn_name}(
                        ${pt_in} *pIn,
                        int8_t *pBias,
                        ${pt_out} *pOut,
                        int8_t *pWeight,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons);
%elif config.api=="PULPNNLinearQuant":
void ${config.fn_name}(
                        ${pt_in} *pIn,
                        int8_t *pBias,
                        ${pt_out} *pOut,
                        int8_t *pWeight,
                        ${act_t} *pKappa,
                        ${act_t} *pLambda,
                        uint16_t out_mul,
                        uint16_t out_shift,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons,
                        uint8_t flag_relu,
                        uint8_t flag_batchnorm);
%elif config.api=="PULPNNMaxPool":
void ${config.fn_name}(
                        ${pt_in} * pIn,
                        ${pt_in} * pOut,
                        uint16_t  dim_im_in_x,
                        uint16_t  dim_im_in_y,
                        uint16_t  ch_im_in,
                        uint16_t  dim_im_out_x,
                        uint16_t  dim_im_out_y,
                        uint16_t  dim_kernel_x,
                        uint16_t  dim_kernel_y,
                        uint16_t  padding_t,
                        uint16_t  padding_b,
                        uint16_t  padding_l,
                        uint16_t  padding_r,
                        uint16_t  stride_x,
                        uint16_t  stride_y);
%elif config.api=="PULPNNAdd":
void ${config.fn_name}(
                        uint8_t * pIn1,
                        uint8_t * pIn2,
                        uint8_t * pOut,
                        uint16_t out_mult1,
                        uint16_t out_mult2,
                        uint16_t out_shift,
                        uint16_t dim_im_in_x,
                        uint16_t dim_im_in_y,
                        uint16_t ch_im_in);
%elif config.api=="PULPNNQuantAdd":
void ${config.fn_name}(
                       uint8_t * pIn1,
                       uint8_t * pIn2,
                       uint8_t * pOut,
                       ${act_t} in_mult1,
                       ${act_t} in_add1,
                       uint16_t in_shift1,
                       ${act_t} in_mult2,
                       ${act_t} in_add2,
                       uint16_t in_shift2,
                       ${act_t} out_mult,
                       ${act_t} out_add,
                       uint16_t out_shift,
                       uint16_t dim_im_in_x,
                       uint16_t dim_im_in_y,
                       uint16_t ch_im_in,
                       int      out_requant_flag);
%elif config.api=="PULPNNAvgPoolNew":
void ${config.fn_name}(
                       ${pt_in} * pIn,
                       ${pt_out} * pOut,
                       ${act_t} lambda,
                       uint16_t out_shift,
                       ${act_t} out_add,
                       uint16_t dim_im_in_x,
                       uint16_t dim_im_in_y,
                       uint16_t ch_im_in,
                       uint16_t dim_im_out_x,
                       uint16_t dim_im_out_y,
                       uint16_t dim_kernel_x,
                       uint16_t dim_kernel_y,
                       uint16_t padding_t,
                       uint16_t padding_b,
                       uint16_t padding_l,
                       uint16_t padding_r,
                       uint16_t stride_x,
                       uint16_t stride_y,
                       int flag_requant
);
%endif
