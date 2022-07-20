/*
 * ${config.filename}
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "pulp_nn_kernels.h"
<%
act_prec = int(config.kernel.act_prec[0:2])
act_t = f"int{act_prec}_t"
def su(sgn):
    return 's' if sgn else 'u'
def u_(sgn):
    return '' if sgn else 'u'
def s_(sgn):
    return 's' if sgn else ''

pt_in = f"{u_(config.kernel.in_signed)}int8_t"
vt_in = f"v4{su(config.kernel.in_signed)}"
int_t_in = f"{u_(config.kernel.in_signed)}int32_t"
pt_out = f"{u_(config.kernel.out_signed)}int8_t"
macload_fn = f"MacLoad{s_(config.kernel.in_signed)}4"
sumdotp_fn = f"SumDotp{s_(config.kernel.in_signed)}4"
out_clip_fn = f"clip{s_(config.kernel.out_signed)}{config.kernel.out_data_t}"
bex = f"bitext{u_(config.kernel.in_signed)}"
%>

void ${config.fn_name}(
                        ${pt_in} *pIn,
                        ${pt_in} *pIm2ColBuffer,
                        int8_t *pBias,
                        ${pt_out} *pOut,
                        int8_t *pWeight,
                        ${act_t} *pKappa,
                        ${act_t} *pLambda,
                        uint16_t out_mult,
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
                        uint8_t flag_batch_norm)
{
  uint16_t ch_in_r = PACK_INT${config.kernel.in_data_t}_SIZE(ch_in);
  uint16_t ch_out_r = PACK_INT${config.kernel.out_data_t}_SIZE(ch_out);

  int core_id = pi_core_id();
  int i_out_y, i_out_x, i_ker_y, i_ker_x;
  int Log2Core;

  uint8_t extra_chunk = ((dim_out_y & (NUM_CORES-1)) != 0);
  uint8_t extra_chunk_r;
  uint16_t dim_out_x_r;
  uint8_t section;
  int core_id_r;

  if(extra_chunk && dim_out_x > 1)
  {
    Log2Core = log2(NUM_CORES >> 1);
    core_id_r = (core_id >> 1);
    dim_out_x_r = (dim_out_x >> 1);
    section = (core_id & 0x1);
    extra_chunk_r = ((dim_out_y & ((NUM_CORES >> 1) - 1)) != 0);
  }
  else
  {
    Log2Core = log2(NUM_CORES);
    core_id_r = core_id;
    dim_out_x_r = dim_out_x;
    section = 0;
    extra_chunk_r = extra_chunk;
    extra_chunk = 0;
  }

  uint8_t flag_dim_out_x_odd = dim_out_x & 0x01;

  int chunk = (dim_out_y >> Log2Core) + extra_chunk_r;

  int start_pixel = min((chunk * core_id_r), dim_out_y);
  int stop_pixel = min(start_pixel + chunk, dim_out_y);

  ${pt_out} *pOutBuffer = pOut + (start_pixel * ch_out_r * dim_out_x) + (section * ch_out_r * dim_out_x_r);
%if config.kernel.in_data_t < config.kernel.wt_data_t:
%if config.kernel.matmul_fmt == '4x2':  
  ${pt_in} *pIm2Col = pIm2ColBuffer + (2 * core_id * PACK_INT${config.kernel.wt_data_t}_SIZE(ch_in));
%elif config.kernel.matmul_fmt == '4x4':
  ${pt_in} *pIm2Col = pIm2ColBuffer + (4 * core_id * PACK_INT${config.kernel.wt_data_t}_SIZE(ch_in));
%endif
%endif

  for (i_out_y = start_pixel; i_out_y < stop_pixel; i_out_y++)
  {
    i_out_x= (section * dim_out_x_r);

%if config.kernel.matmul_fmt == '4x2':
    for(int n = 0; n<((dim_out_x_r + (section * flag_dim_out_x_odd)) >> 1); n++)
%elif config.kernel.matmul_fmt == '4x4':
    for(int n = 0; n<((dim_out_x_r + (section * flag_dim_out_x_odd)) >> 2); n++)
%endif
    {
%if config.kernel.in_data_t < config.kernel.wt_data_t:
%if config.kernel.matmul_fmt == '4x2':
      ${config.im2col_fn}(pIn + (i_out_x * ch_in_r) + (i_out_y * dim_in_x * ch_in_r), pIm2Col, ch_in<<1);
%elif config.kernel.matmul_fmt == '4x4':
      ${config.im2col_fn}(pIn + (i_out_x * ch_in_r) + (i_out_y * dim_in_x * ch_in_r), pIm2Col, ch_in<<2);
%endif
%else:
      ${pt_in} *pIm2Col = (pIn + (i_out_x * ch_in_r) + (i_out_y * dim_in_x * ch_in_r));
%endif
      pOutBuffer = ${config.mat_mul_fn}(
          pIm2Col,
          pBias,
          pOutBuffer,
          pOutBuffer + ch_out_r,
%if config.kernel.matmul_fmt == '4x4':
          pOutBuffer + (ch_out_r << 1),
          pOutBuffer + (ch_out_r << 1) + ch_out_r,
%endif
          pWeight,
          pKappa,
          pLambda,
          out_mult,
          out_shift,
          (ch_in * dim_kernel_x * dim_kernel_y),
          ch_out,
          flag_relu,
          flag_batch_norm
          );
%if config.kernel.matmul_fmt == '4x2':
      i_out_x+=2;
%elif config.kernel.matmul_fmt == '4x4':
      i_out_x+=4;
%endif
    }

    if(((dim_out_x_r + (section * flag_dim_out_x_odd)) & 0x0001))
    {
  %if config.kernel.out_data_t == 2:
      int8_t mask2 = 0x0c;
      int8_t n_mask2 = ~ mask2;
      int8_t mask4 = 0x30;
      int8_t n_mask4 = ~ mask4;
      int8_t mask6 = 0xc0;
      int8_t n_mask6 = ~ mask6;
      int8_t off2 = 2;
      int8_t off4 = 4;
      int8_t off6 = 6;
  %elif config.kernel.out_data_t == 4:
      int8_t mask = 0xf0;
      int8_t n_mask = ~ mask;
      int8_t off = 0x04;
  %endif
      const int8_t *pA = pWeight;
      int i;
      ${act_t} * k1 = pKappa;
      ${act_t} * lambda1 = pLambda;

%if config.kernel.wt_data_t == 2:
      v4s inA[4];
%elif config.kernel.wt_data_t == 4:
      v4s inA[2];
%else:
      v4s inA;
%endif
%if config.kernel.in_data_t == 2:
      ${vt_in} inB[4];
%elif config.kernel.in_data_t == 4:
      ${vt_in} inB[2];
%else:
      ${vt_in} inB;
%endif
  %if config.kernel.out_data_t == 4:
      ${pt_out} out[2];
  %elif config.kernel.out_data_t == 2:
      ${pt_out} out[4];
  %endif
      for(i = 0; i < ch_out; i++)
      {
        int sum = 0;
        if (pBias != NULL)
        {
          sum = ((int) (*pBias++));
        }

        ${pt_in} *pB = (pIn + (i_out_x * ch_in) + (i_out_y * dim_in_x * ch_in));

        uint16_t col_cnt_im2col = ch_in * dim_kernel_x * dim_kernel_y;
<%! import math %>
        for(int j=0; j < (col_cnt_im2col >> ${int(math.log2(int(32/config.kernel.wt_data_t)))}); j++)
        {
%if config.kernel.wt_data_t == 2:
%if config.kernel.in_data_t == 8:
          inB = *((${vt_in}*) pB);

          pB+=4;

          pA = ${config.unpack_wt_fn}(pA,inA);

          sum = ${sumdotp_fn}(inB, inA[0], sum);

          inB = *((${vt_in}*) pB);

          pB+=4;

          sum = ${sumdotp_fn}(inB, inA[1], sum);

          inB = *((${vt_in}*) pB);

          pB+=4;

          sum = ${sumdotp_fn}(inB, inA[2], sum);

          inB = *((${vt_in}*) pB);

          pB+=4;

          sum = ${sumdotp_fn}(inB, inA[3], sum);
%elif config.kernel.in_data_t == 4:
          pB = ${config.unpack_in_fn}(pB,inB);

          pA = ${config.unpack_wt_fn}(pA,inA);

          sum = ${sumdotp_fn}(inB[0], inA[0], sum);

          sum = ${sumdotp_fn}(inB[1], inA[1], sum);

          pB = ${config.unpack_in_fn}(pB,inB);

          sum = ${sumdotp_fn}(inB[0], inA[2], sum);

          sum = ${sumdotp_fn}(inB[1], inA[3], sum);
%elif config.kernel.in_data_t == 2:
          pB = ${config.unpack_in_fn}(pB,inB);

          pA = ${config.unpack_wt_fn}(pA,inA);

          sum = ${sumdotp_fn}(inB[0], inA[0], sum);

          sum = ${sumdotp_fn}(inB[1], inA[1], sum);

          sum = ${sumdotp_fn}(inB[2], inA[2], sum);

          sum = ${sumdotp_fn}(inB[3], inA[3], sum);
%endif
%elif config.kernel.wt_data_t == 4:
%if config.kernel.in_data_t == 8:
          inB = *((${vt_in}*) pB);

          pB+=4;

          pA = ${config.unpack_wt_fn}(pA,inA);

          sum = ${sumdotp_fn}(inB, inA[0], sum);

          inB = *((${vt_in}*) pB);

          sum = ${sumdotp_fn}(inB, inA[1], sum);

          pB+=4;
%elif config.kernel.in_data_t == 4:
          pB = ${config.unpack_in_fn}(pB,inB);

          pA = ${config.unpack_wt_fn}(pA,inA);

          sum = ${sumdotp_fn}(inB[0], inA[0], sum);

          sum = ${sumdotp_fn}(inB[1], inA[1], sum);
%elif config.kernel.in_data_t == 2:
          pB = ${config.unpack_in_fn}(pB,inB);

          pA = ${config.unpack_wt_fn}(pA,inA);

          sum = ${sumdotp_fn}(inB[0], inA[0], sum);

          sum = ${sumdotp_fn}(inB[1], inA[1], sum);

          pA = ${config.unpack_wt_fn}(pA,inA);

          sum = ${sumdotp_fn}(inB[0], inA[0], sum);

          sum = ${sumdotp_fn}(inB[1], inA[1], sum);
%endif
%else:
%if config.kernel.in_data_t == 8:
          v4s inA = *((v4s*) pA);
          ${vt_in} inB = *((${vt_in}*) pB);

          sum = ${sumdotp_fn}(inB, inA, sum);
          pA+=4;
          pB+=4;
%elif config.kernel.in_data_t == 4:
          inA = *((v4s*) pA);

          pA+=4;

          pB = ${config.unpack_in_fn}(pB,inB);

          sum = ${sumdotp_fn}(inB[0], inA, sum);

          inA = *((v4s*) pA);

          pA+=4;

          sum = ${sumdotp_fn}(inB[1], inA, sum);
%elif config.kernel.in_data_t == 2:
          inA = *((v4s*) pA);

          pA+=4;

          pB = ${config.unpack_in_fn}(pB,inB);

          sum = ${sumdotp_fn}(inB[0], inA, sum);

          inA = *((v4s*) pA);

          pA+=4;

          sum = ${sumdotp_fn}(inB[1], inA, sum);

          inA = *((v4s*) pA);

          pA+=4;

          sum = ${sumdotp_fn}(inB[2], inA, sum);

          inA = *((v4s*) pA);

          pA+=4;

          sum = ${sumdotp_fn}(inB[3], inA, sum);
%endif
%endif
        }
  %if config.kernel.wt_data_t == 2:
        col_cnt_im2col = (ch_in * dim_kernel_y * dim_kernel_x) & 0xf;
  %elif config.kernel.wt_data_t == 4:
        col_cnt_im2col = (ch_in * dim_kernel_y * dim_kernel_x) & 0x7;
  %else:
        col_cnt_im2col = (ch_in * dim_kernel_y * dim_kernel_x) & 0x3;
  %endif
        while (col_cnt_im2col)
        {
%if config.kernel.wt_data_t == 2:
%if config.kernel.in_data_t == 8:
          int8_t inA1 = (int8_t) bitext((int) *pA, 2, 0);
          ${pt_in} inB1 = *pB++;
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 2, 2);
          inB1 = *pB++;
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 2, 4);
          inB1 = *pB++;
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 2, 6);
          inB1 = *pB++;
          sum += inA1 * inB1;

          pA++;
          col_cnt_im2col-=4;
%elif config.kernel.in_data_t == 4:
          int8_t inA1 = (int8_t) bitext((int) *pA, 2, 0);
          ${pt_in} inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 0);
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 2, 2);
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 4);
          sum += inA1 * inB1;
          pB++;
          inA1 = (int8_t) bitext((int) *pA, 2, 4);
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 0);
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 2, 6);
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 4);
          sum += inA1 * inB1;

          pA++;
          pB++;
          col_cnt_im2col-=4;
%elif config.kernel.in_data_t == 2:
          int8_t inA1 = (int8_t) bitext((int) *pA, 2, 0);
          ${pt_in} inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 0);
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 2, 2);
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 2);
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 2, 4);
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 4);
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 2, 6);
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 6);
          sum += inA1 * inB1;

          pA++;
          pB++;
          col_cnt_im2col-=4;
%endif
%elif config.kernel.wt_data_t == 4:
%if config.kernel.in_data_t == 8:
          int8_t inA1 = (int8_t) bitext((int) *pA, 4, 0);
          ${pt_in} inB1 = *pB++;
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 4, 4);
          inB1 = *pB++;
          sum += inA1 * inB1;

          pA++;
          col_cnt_im2col-=2;
%elif config.kernel.in_data_t == 4:
          int8_t inA1 = (int8_t) bitext((int) *pA, 4, 0);
          ${pt_in} inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 0);
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 4, 4);
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 4);
          sum += inA1 * inB1;

          pA++;
          pB++;
          col_cnt_im2col-=2;
%elif config.kernel.in_data_t == 2:
          int8_t inA1 = (int8_t) bitext((int) *pA, 4, 0);
          ${pt_in} inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 0);
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 4, 4);
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 2);
          sum += inA1 * inB1;
          pA++;
          inA1 = (int8_t) bitext((int) *pA, 4, 0);
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 4);
          sum += inA1 * inB1;
          inA1 = (int8_t) bitext((int) *pA, 4, 4);
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 6);
          sum += inA1 * inB1;

          pA++;
          pB++;
          col_cnt_im2col-=4;
%endif
%else:
%if config.kernel.in_data_t == 8:
          int8_t inA1 = *pA++;
          ${pt_in} inB1 = *pB++;
          asm volatile("": : :"memory");
          sum += inA1 * inB1;

          col_cnt_im2col--;
%elif config.kernel.in_data_t == 4:
          int8_t inA1 = *pA++;
          ${pt_in} inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 0);
          sum += inA1 * inB1;
          inA1 = *pA++;
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 4);
          sum += inA1 * inB1;
          pB++;
          col_cnt_im2col-=2;
%elif config.kernel.in_data_t == 2:
          int8_t inA1 = *pA++;
          ${pt_in} inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 0);
          sum += inA1 * inB1;
          inA1 = *pA++;
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 2);
          sum += inA1 * inB1;
          inA1 = *pA++;
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 4);
          sum += inA1 * inB1;
          inA1 = *pA++;
          inB1 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 6);
          sum += inA1 * inB1;
          pB++;
          col_cnt_im2col-=4;
%endif
%endif
        }
  %if config.kernel.out_data_t == 8 or config.kernel.quantization == 'shift_clip':
        if (flag_batch_norm && flag_relu)
        {
  %if config.kernel.out_data_t == 8:
          *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          k1++;
          lambda1++;
          pOutBuffer++;
  %elif config.kernel.out_data_t == 4:
          uint8_t i_o = i & 0x01;
          out[i_o] = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          k1++;
          lambda1++;
          if(i_o == 0x01)
          {
            *pOutBuffer = bitins(out[0], n_mask, out[1], mask, off);
            pOutBuffer++;
          }
  %elif config.kernel.out_data_t == 2:
          uint8_t i_o = i & 0x03;
          out[i_o] = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          k1++;
          lambda1++;
          if(i_o == 0x03)
          {
            out[0] = bitins(out[0], n_mask2, out[1], mask2, off2);
            out[0] = bitins(out[0], n_mask4, out[2], mask4, off4);
            *pOutBuffer = bitins(out[0], n_mask6, out[3], mask6, off6);
            pOutBuffer++;
          }
  %endif
        }
        else
        {
          if(flag_relu == 1)
          {
  %if config.kernel.out_data_t == 8:
            *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
            pOutBuffer++;
  %elif config.kernel.out_data_t == 4:
            uint8_t i_o = i & 0x01;
            out[i_o] = ${config.relu_fn}(sum, out_mult, out_shift);
            if(i_o == 0x01)
            {
              *pOutBuffer = bitins(out[0], n_mask, out[1], mask, off);
              pOutBuffer++;
            }
  %elif config.kernel.out_data_t == 2:
            uint8_t i_o = i & 0x03;
            out[i_o] = ${config.relu_fn}(sum, out_mult, out_shift);
            if(i_o == 0x03)
            {
              out[0] = bitins(out[0], n_mask2, out[1], mask2, off2);
              out[0] = bitins(out[0], n_mask4, out[2], mask4, off4);
              *pOutBuffer = bitins(out[0], n_mask6, out[3], mask6, off6);
              pOutBuffer++;
            }
  %endif
          }
          else
          {
  %if config.kernel.out_data_t == 8:
            *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            pOutBuffer++;
  %elif config.kernel.out_data_t == 4:
            uint8_t i_o = i & 0x01;
            out[i_o] = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            if(i_o == 0x01)
            {
              *pOutBuffer = bitins(out[0], n_mask, out[1], mask, off);
              pOutBuffer++;
            }
  %elif config.kernel.out_data_t == 2:
            uint8_t i_o = i & 0x03;
            out[i_o] = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            if(i_o == 0x03)
            {
              out[0] = bitins(out[0], n_mask2, out[1], mask2, off2);
              out[0] = bitins(out[0], n_mask4, out[2], mask4, off4);
              *pOutBuffer = bitins(out[0], n_mask6, out[3], mask6, off6);
              pOutBuffer++;
            }
  %endif
          }
        }
  %elif config.kernel.out_data_t == 4:
        uint8_t i_o = i & 0x01;
        out[i_o] = pulp_nn_i4_quant(sum, pThr);
        pThr++;
        if(i_o == 0x01)
        {
          *pOutBuffer = bitins(out[0], n_mask, out[1], mask, off);
          pOutBuffer++;
        }
  %elif config.kernel.out_data_t == 2:
        uint8_t i_o = i & 0x03;
        out[i_o] = pulp_nn_i2_quant(sum, pThr);
        pThr++;
        if(i_o == 0x03)
        {
          out[0] = bitins(out[0], n_mask2, out[1], mask2, off2);
          out[0] = bitins(out[0], n_mask4, out[2], mask4, off4);
          *pOutBuffer = bitins(out[0], n_mask6, out[3], mask6, off6);
          pOutBuffer++;
        }
  %endif
      }
    }
    pOutBuffer+=(extra_chunk * ((dim_out_x_r + ((1 - section) * flag_dim_out_x_odd)) * ch_out));
  }
  pi_cl_team_barrier(0);
}
