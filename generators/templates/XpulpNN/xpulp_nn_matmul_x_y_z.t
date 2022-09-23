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

<%
act_prec = int(config.kernel.act_prec[0:2])
act_t = f"int{act_prec}_t"
def su(sgn):
    return 's' if sgn else 'u'
def u_(sgn):
    return '' if sgn else 'u'
def s_(sgn):
    return 's' if sgn else ''


bex = f"bitext{u_(config.kernel.in_signed)}"
pt_in = f"{u_(config.kernel.in_signed)}int8_t"
vt_in = f"v4{su(config.kernel.in_signed)}"
pt_out = f"{u_(config.kernel.out_signed)}int8_t"
mac_fn = f"MacLoad{s_(config.kernel.in_signed)}{32//config.max_precision}"
int_t_in = f"{u_(config.kernel.in_signed)}int32_t"
out_clip_fn = f"clip{s_(config.kernel.out_signed)}{config.kernel.out_data_t}"
%>

uint8_t * __attribute__((noinline)) ${config.fn_name}(
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
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t num_col_im2col,
                        uint16_t ch_out,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
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
%if config.kernel.wt_data_t < config.kernel.in_data_t:
  int32_t vecA[${int(config.max_precision/config.kernel.wt_data_t)}];
  int32_t vecA2[${int(config.max_precision/config.kernel.wt_data_t)}];
  int32_t vecA3[${int(config.max_precision/config.kernel.wt_data_t)}];
  int32_t vecA4[${int(config.max_precision/config.kernel.wt_data_t)}];
%endif

  uint16_t ch_out_r = PACK_INT${config.kernel.out_data_t}_SIZE(ch_out);

  uint16_t num_col_im2col_w = PACK_INT${config.kernel.wt_data_t}_SIZE(num_col_im2col);
  uint16_t num_col_im2col_a = PACK_INT${config.max_precision}_SIZE(num_col_im2col);

  //uint8_t *pOut2 = pOut + ch_out_r;
  int8_t *pA = pWeight;

  uint16_t chan_left = ch_out & 0x3;

  for(int i=0; i < (ch_out >> 2); i++)
  {
    ${pt_in} *pB =  pIn;
    ${pt_in} *pB2 = (pB + num_col_im2col_a);
%if config.kernel.matmul_fmt == '4x4':
    ${pt_in} *pB3 = (pB2 + num_col_im2col_a);
    ${pt_in} *pB4 = (pB3 + num_col_im2col_a);
%endif

    ${int_t_in} *ptrB  = (${int_t_in} *) pB;
    ${int_t_in} *ptrB2 = (${int_t_in} *) pB2;
%if config.kernel.matmul_fmt == '4x4':
    ${int_t_in} *ptrB3 = (${int_t_in} *) pB3;
    ${int_t_in} *ptrB4 = (${int_t_in} *) pB4;
%endif

    int8_t *pA2 = (pA + num_col_im2col_w);
    int8_t *pA3 = (pA2 + num_col_im2col_w);
    int8_t *pA4 = (pA3 + num_col_im2col_w);

%if config.kernel.wt_data_t < config.kernel.in_data_t:
    pA  = ${config.unpack_wt_fn}(pA , vecA);
    pA2 = ${config.unpack_wt_fn}(pA2, vecA2);
    pA3 = ${config.unpack_wt_fn}(pA3, vecA3);
    pA4 = ${config.unpack_wt_fn}(pA4, vecA4);

    int32_t *startA;
    int32_t *startA2;
    int32_t *startA3;
    int32_t *startA4;

    asm volatile("mv %0, %1":"=r"(startA):"r"(vecA));
    asm volatile("mv %0, %1":"=r"(startA2):"r"(vecA2));
    asm volatile("mv %0, %1":"=r"(startA3):"r"(vecA3));
    asm volatile("mv %0, %1":"=r"(startA4):"r"(vecA4));

    int32_t *ptrA  = (int32_t *) vecA ;
    int32_t *ptrA2 = (int32_t *) vecA2;
    int32_t *ptrA3 = (int32_t *) vecA3;
    int32_t *ptrA4 = (int32_t *) vecA4;
%else:
    int32_t *ptrA  = (int32_t *) pA ;
    int32_t *ptrA2 = (int32_t *) pA2;
    int32_t *ptrA3 = (int32_t *) pA3;
    int32_t *ptrA4 = (int32_t *) pA4;
%endif

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
    ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
    ptrA3 = MacLoadInit(1, 0, 2, 0, ptrA3);
    ptrA4 = MacLoadInit(1, 0, 3, 0, ptrA4);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

    int sum = 0;
    int sum2 = 0;
    int sum3 = 0;
    int sum4 = 0;
    int sum5 = 0;
    int sum6 = 0;
    int sum7 = 0;
    int sum8 = 0;

%if config.kernel.matmul_fmt == '4x4':
    int sum9  = 0;
    int sum10 = 0;
    int sum11 = 0;
    int sum12 = 0;
    int sum13 = 0;
    int sum14 = 0;
    int sum15 = 0;
    int sum16 = 0;
%endif

    if (pBias != NULL)
    {
      sum = *((int*)  pBias);
      pBias+= 4;
      sum2 = *((int*)  pBias);
      pBias+= 4;
      sum3 = *((int*)  pBias);
      pBias+= 4;
      sum4 = *((int*)  pBias);
      pBias+= 4;

      sum5 = sum;
      sum6 = sum2;
      sum7 = sum3;
      sum8 = sum4;
    }
<%! import math %>
    for(int j=0; j<(num_col_im2col >> ${int(math.log2(((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t)))))}); j++)
    {
      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = ${mac_fn}(0, 0, 0, 0, ptrA, sum);
      sum2 = ${mac_fn}(0, 0, 1, 0, ptrA2, sum2);
      sum3 = ${mac_fn}(0, 0, 2, 0, ptrA3, sum3);
%if config.kernel.matmul_fmt == '4x2':
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.matmul_fmt == '4x4':
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB3, sum4);
      ptrB3 = MacLoadUpdate(ptrB3);
%endif


%if config.kernel.matmul_fmt == '4x2':
      sum5 = ${mac_fn}(1, 0, 0, 1, ptrA, sum5);
      ptrA = MacLoadUpdate(ptrA);

      sum6 = ${mac_fn}(1, 0, 1, 1, ptrA2, sum6);
      ptrA2 = MacLoadUpdate(ptrA2);

      sum7 = ${mac_fn}(1, 0, 2, 1, ptrA3, sum7);
      ptrA3 = MacLoadUpdate(ptrA3);

      sum8 = ${mac_fn}(1, 0, 3, 1, ptrA4, sum8);
      ptrA4 = MacLoadUpdate(ptrA4);
%elif config.kernel.matmul_fmt == '4x4':
      sum5 = ${mac_fn}(0, 0, 0, 1, ptrA, sum5);
      sum6 = ${mac_fn}(0, 0, 1, 1, ptrA2, sum6);
      sum7 = ${mac_fn}(0, 0, 2, 1, ptrA3, sum7);
      sum8 = ${mac_fn}(0, 1, 3, 1, ptrB4, sum8);
      ptrB4 = MacLoadUpdate(ptrB4);

      sum9  = ${mac_fn}(0, 0, 0, 0, ptrA, sum9);
      sum10 = ${mac_fn}(0, 0, 1, 0, ptrA2, sum10);
      sum11 = ${mac_fn}(0, 0, 2, 0, ptrA3, sum11);
      sum12 = ${mac_fn}(0, 1, 3, 0, ptrB, sum12);
      ptrB = MacLoadUpdate(ptrB);

      sum13 = ${mac_fn}(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);
      sum14 = ${mac_fn}(1, 0, 1, 1, ptrA2, sum14);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum15 = ${mac_fn}(1, 0, 2, 1, ptrA3, sum15);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum16 = ${mac_fn}(1, 0, 3, 1, ptrA4, sum16);
      ptrA4 = MacLoadUpdate(ptrA4);
%endif

%if config.kernel.wt_data_t < config.kernel.in_data_t:
%if (config.kernel.in_data_t/config.kernel.wt_data_t) == 4:
      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = ${mac_fn}(0, 0, 0, 0, ptrA, sum);
      sum2 = ${mac_fn}(0, 0, 1, 0, ptrA2, sum2);
      sum3 = ${mac_fn}(0, 0, 2, 0, ptrA3, sum3);
%if config.kernel.matmul_fmt == '4x2':
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.matmul_fmt == '4x4':
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB3, sum4);
      ptrB3 = MacLoadUpdate(ptrB3);
%endif

%if config.kernel.matmul_fmt == '4x2':
      sum5 = ${mac_fn}(1, 0, 0, 1, ptrA, sum5);
      ptrA = MacLoadUpdate(ptrA);

      sum6 = ${mac_fn}(1, 0, 1, 1, ptrA2, sum6);
      ptrA2 = MacLoadUpdate(ptrA2);

      sum7 = ${mac_fn}(1, 0, 2, 1, ptrA3, sum7);
      ptrA3 = MacLoadUpdate(ptrA3);

      sum8 = ${mac_fn}(1, 0, 3, 1, ptrA4, sum8);
      ptrA4 = MacLoadUpdate(ptrA4);

      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);
%elif config.kernel.matmul_fmt == '4x4':
      sum5 = ${mac_fn}(0, 0, 0, 1, ptrA, sum5);
      sum6 = ${mac_fn}(0, 0, 1, 1, ptrA2, sum6);
      sum7 = ${mac_fn}(0, 0, 2, 1, ptrA3, sum7);
      sum8 = ${mac_fn}(0, 1, 3, 1, ptrB4, sum8);
      ptrB4 = MacLoadUpdate(ptrB4);

      sum9  = ${mac_fn}(0, 0, 0, 0, ptrA, sum9);
      sum10 = ${mac_fn}(0, 0, 1, 0, ptrA2, sum10);
      sum11 = ${mac_fn}(0, 0, 2, 0, ptrA3, sum11);
      sum12 = ${mac_fn}(0, 1, 3, 0, ptrB, sum12);
      ptrB = MacLoadUpdate(ptrB);

      sum13 = ${mac_fn}(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);
      sum14 = ${mac_fn}(1, 0, 1, 1, ptrA2, sum14);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum15 = ${mac_fn}(1, 0, 2, 1, ptrA3, sum15);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum16 = ${mac_fn}(1, 0, 3, 1, ptrA4, sum16);
      ptrA4 = MacLoadUpdate(ptrA4);

      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);
%endif

      sum  = ${mac_fn}(0, 0, 0, 0, ptrA, sum);
      sum2 = ${mac_fn}(0, 0, 1, 0, ptrA2, sum2);
      sum3 = ${mac_fn}(0, 0, 2, 0, ptrA3, sum3);
%if config.kernel.matmul_fmt == '4x2':
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.matmul_fmt == '4x4':
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB3, sum4);
      ptrB3 = MacLoadUpdate(ptrB3);
%endif

%if config.kernel.matmul_fmt == '4x2':
      sum5 = ${mac_fn}(1, 0, 0, 1, ptrA, sum5);
      ptrA = MacLoadUpdate(ptrA);

      sum6 = ${mac_fn}(1, 0, 1, 1, ptrA2, sum6);
      ptrA2 = MacLoadUpdate(ptrA2);

      sum7 = ${mac_fn}(1, 0, 2, 1, ptrA3, sum7);
      ptrA3 = MacLoadUpdate(ptrA3);

      sum8 = ${mac_fn}(1, 0, 3, 1, ptrA4, sum8);
      ptrA4 = MacLoadUpdate(ptrA4);
%elif config.kernel.matmul_fmt == '4x4':
      sum5 = ${mac_fn}(0, 0, 0, 1, ptrA, sum5);
      sum6 = ${mac_fn}(0, 0, 1, 1, ptrA2, sum6);
      sum7 = ${mac_fn}(0, 0, 2, 1, ptrA3, sum7);
      sum8 = ${mac_fn}(0, 1, 3, 1, ptrB4, sum8);
      ptrB4 = MacLoadUpdate(ptrB4);

      sum9  = ${mac_fn}(0, 0, 0, 0, ptrA, sum9);
      sum10 = ${mac_fn}(0, 0, 1, 0, ptrA2, sum10);
      sum11 = ${mac_fn}(0, 0, 2, 0, ptrA3, sum11);
      sum12 = ${mac_fn}(0, 1, 3, 0, ptrB, sum12);
      ptrB = MacLoadUpdate(ptrB);

      sum13 = ${mac_fn}(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);
      sum14 = ${mac_fn}(1, 0, 1, 1, ptrA2, sum14);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum15 = ${mac_fn}(1, 0, 2, 1, ptrA3, sum15);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum16 = ${mac_fn}(1, 0, 3, 1, ptrA4, sum16);
      ptrA4 = MacLoadUpdate(ptrA4);
%endif
%endif
      ptrB2  = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = ${mac_fn}(0, 0, 0, 0, ptrA, sum);
      sum2 = ${mac_fn}(0, 0, 1, 0, ptrA2, sum2);
      sum3 = ${mac_fn}(0, 0, 2, 0, ptrA3, sum3);
%if config.kernel.matmul_fmt == '4x2':
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.matmul_fmt == '4x4':
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB3, sum4);
      ptrB3 = MacLoadUpdate(ptrB3);
%endif

      pA  = ${config.unpack_wt_fn}(pA , vecA);
      pA2 = ${config.unpack_wt_fn}(pA2, vecA2);
      pA3 = ${config.unpack_wt_fn}(pA3, vecA3);
      pA4 = ${config.unpack_wt_fn}(pA4, vecA4);

      ptrA   = MacLoadAssign(vecA);
      ptrA2  = MacLoadAssign(vecA2);
      ptrA3  = MacLoadAssign(vecA3);
      ptrA4  = MacLoadAssign(vecA4);

%if config.kernel.matmul_fmt == '4x2':
      sum5  = ${mac_fn}(1, 0, 0, 1, ptrA, sum5);
      ptrA  = MacLoadUpdate(ptrA);

      sum6  = ${mac_fn}(1, 0, 1, 1, ptrA2, sum6);
      ptrA2 = MacLoadUpdate(ptrA2);

      sum7  = ${mac_fn}(1, 0, 2, 1, ptrA3, sum7);
      ptrA3 = MacLoadUpdate(ptrA3);

      sum8  = ${mac_fn}(1, 0, 3, 1, ptrA4, sum8);
      ptrA4 = MacLoadUpdate(ptrA4);
%elif config.kernel.matmul_fmt == '4x4':
      sum5 = ${mac_fn}(0, 0, 0, 1, ptrA, sum5);
      sum6 = ${mac_fn}(0, 0, 1, 1, ptrA2, sum6);
      sum7 = ${mac_fn}(0, 0, 2, 1, ptrA3, sum7);
      sum8 = ${mac_fn}(0, 1, 3, 1, ptrB4, sum8);
      ptrB4 = MacLoadUpdate(ptrB4);

      sum9  = ${mac_fn}(0, 0, 0, 0, ptrA, sum9);
      sum10 = ${mac_fn}(0, 0, 1, 0, ptrA2, sum10);
      sum11 = ${mac_fn}(0, 0, 2, 0, ptrA3, sum11);
      sum12 = ${mac_fn}(0, 1, 3, 0, ptrB, sum12);
      ptrB = MacLoadUpdate(ptrB);

      sum13 = ${mac_fn}(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);
      sum14 = ${mac_fn}(1, 0, 1, 1, ptrA2, sum14);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum15 = ${mac_fn}(1, 0, 2, 1, ptrA3, sum15);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum16 = ${mac_fn}(1, 0, 3, 1, ptrA4, sum16);
      ptrA4 = MacLoadUpdate(ptrA4);
%endif
%endif
    }
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    pA-=4;
    pA2-=4;
    pA3-=4;
    pA4-=4;
%endif

    int col_cnt_im2col = num_col_im2col & ${hex((((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t))))-1)};

    if(col_cnt_im2col)
    {
%if config.kernel.wt_data_t >= config.kernel.in_data_t:
      uint16_t loop_cnt_im2col_w = (num_col_im2col >> ${int(math.log2(((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t)))))}) << 2;
      pA+=loop_cnt_im2col_w;
      pA2+=loop_cnt_im2col_w;
      pA3+=loop_cnt_im2col_w;
      pA4+=loop_cnt_im2col_w;
%endif

%if config.kernel.wt_data_t < config.kernel.in_data_t:
      uint16_t loop_cnt_im2col_a = (num_col_im2col >> ${int(math.log2(((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t)))))}) << ${int(2+int(math.log2(int(config.kernel.in_data_t/config.kernel.wt_data_t))))};
%else:
      uint16_t loop_cnt_im2col_a = (num_col_im2col >> ${int(math.log2(((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t)))))}) << 2;
%endif
      pB+=loop_cnt_im2col_a;
      pB2+=loop_cnt_im2col_a;

      do
      {
%if config.kernel.in_data_t == 8:
%if config.kernel.wt_data_t == 2:
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA2, 2, 0);
        int8_t inA3 = (int8_t) bitext((int) *pA3, 2, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA4, 2, 0);

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 2);
        inA2 = (int8_t) bitext((int) *pA2, 2, 2);
        inA3 = (int8_t) bitext((int) *pA3, 2, 2);
        inA4 = (int8_t) bitext((int) *pA4, 2, 2);

        inB = *pB++;
        inB2 = *pB2++;

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 4);
        inA2 = (int8_t) bitext((int) *pA2, 2, 4);
        inA3 = (int8_t) bitext((int) *pA3, 2, 4);
        inA4 = (int8_t) bitext((int) *pA4, 2, 4);

        inB = *pB++;
        inB2 = *pB2++;

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 6);
        inA2 = (int8_t) bitext((int) *pA2, 2, 6);
        inA3 = (int8_t) bitext((int) *pA3, 2, 6);
        inA4 = (int8_t) bitext((int) *pA4, 2, 6);

        inB = *pB++;
        inB2 = *pB2++;

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        pA++;
        pA2++;
        pA3++;
        pA4++;

        col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
        int8_t inA = (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA2, 4, 0);
        int8_t inA3 = (int8_t) bitext((int) *pA3, 4, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA4, 4, 0);

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        inA = (int8_t) bitext((int) *pA, 4, 4);
        inA2 = (int8_t) bitext((int) *pA2, 4, 4);
        inA3 = (int8_t) bitext((int) *pA3, 4, 4);
        inA4 = (int8_t) bitext((int) *pA4, 4, 4);

        inB = *pB++;
        inB2 = *pB2++;

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        pA++;
        pA2++;
        pA3++;
        pA4++;

        col_cnt_im2col-=2;
%else:
        int8_t inA = *pA++;
        int8_t inA2 = *pA2++;
        int8_t inA3 = *pA3++;
        int8_t inA4 = *pA4++;

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;
        asm volatile("": : :"memory");
        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        col_cnt_im2col--;
%endif
%elif config.kernel.in_data_t == 4:
%if config.kernel.wt_data_t == 2:
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA2, 2, 0);
        int8_t inA3 = (int8_t) bitext((int) *pA3, 2, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA4, 2, 0);

        ${pt_in} inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_in} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 2);
        inA2 = (int8_t) bitext((int) *pA2, 2, 2);
        inA3 = (int8_t) bitext((int) *pA3, 2, 2);
        inA4 = (int8_t) bitext((int) *pA4, 2, 2);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        pB++;
        pB2++;

        inA = (int8_t) bitext((int) *pA, 2, 4);
        inA2 = (int8_t) bitext((int) *pA2, 2, 4);
        inA3 = (int8_t) bitext((int) *pA3, 2, 4);
        inA4 = (int8_t) bitext((int) *pA4, 2, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 6);
        inA2 = (int8_t) bitext((int) *pA2, 2, 6);
        inA3 = (int8_t) bitext((int) *pA3, 2, 6);
        inA4 = (int8_t) bitext((int) *pA4, 2, 6);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        pA++;
        pA2++;
        pA3++;
        pA4++;

        pB++;
        pB2++;

        col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
        int8_t inA = (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA2, 4, 0);
        int8_t inA3 = (int8_t) bitext((int) *pA3, 4, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA4, 4, 0);

        ${pt_in} inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_in} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        inA = (int8_t) bitext((int) *pA, 4, 4);
        inA2 = (int8_t) bitext((int) *pA2, 4, 4);
        inA3 = (int8_t) bitext((int) *pA3, 4, 4);
        inA4 = (int8_t) bitext((int) *pA4, 4, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        pA++;
        pA2++;
        pA3++;
        pA4++;

        pB++;
        pB2++;

        col_cnt_im2col-=2;
%else:
        int8_t inA = *pA++;
        int8_t inA2 = *pA2++;
        int8_t inA3 = *pA3++;
        int8_t inA4 = *pA4++;

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;
        asm volatile("": : :"memory");
        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        col_cnt_im2col--;
%endif
%elif config.kernel.in_data_t == 2:
%if config.kernel.wt_data_t == 2:
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA2, 2, 0);
        int8_t inA3 = (int8_t) bitext((int) *pA3, 2, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA4, 2, 0);

        ${pt_in} inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 0);
        ${pt_in} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 0);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 2);
        inA2 = (int8_t) bitext((int) *pA2, 2, 2);
        inA3 = (int8_t) bitext((int) *pA3, 2, 2);
        inA4 = (int8_t) bitext((int) *pA4, 2, 2);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 2);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 2);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 4);
        inA2 = (int8_t) bitext((int) *pA2, 2, 4);
        inA3 = (int8_t) bitext((int) *pA3, 2, 4);
        inA4 = (int8_t) bitext((int) *pA4, 2, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 4);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 6);
        inA2 = (int8_t) bitext((int) *pA2, 2, 6);
        inA3 = (int8_t) bitext((int) *pA3, 2, 6);
        inA4 = (int8_t) bitext((int) *pA4, 2, 6);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 6);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 6);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        pA++;
        pA2++;
        pA3++;
        pA4++;

        pB++;
        pB2++;

        col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
        int8_t inA = (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA2, 4, 0);
        int8_t inA3 = (int8_t) bitext((int) *pA3, 4, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA4, 4, 0);

        ${pt_in} inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_in} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        inA = (int8_t) bitext((int) *pA, 4, 4);
        inA2 = (int8_t) bitext((int) *pA2, 4, 4);
        inA3 = (int8_t) bitext((int) *pA3, 4, 4);
        inA4 = (int8_t) bitext((int) *pA4, 4, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        pA++;
        pA2++;
        pA3++;
        pA4++;

        pB++;
        pB2++;

        col_cnt_im2col-=2;
%else:
        int8_t inA = *pA++;
        int8_t inA2 = *pA2++;
        int8_t inA3 = *pA3++;
        int8_t inA4 = *pA4++;

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;
        asm volatile("": : :"memory");
        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        col_cnt_im2col--;
%endif
%endif
      } while(col_cnt_im2col > 0);
%if config.kernel.wt_data_t >= config.kernel.in_data_t:
      pA-=num_col_im2col_w;
%endif
    }
%if config.kernel.out_data_t == 8 or config.kernel.quantization == 'shift_clip':
    if (flag_batch_norm && flag_relu)
    {
%if config.kernel.out_data_t == 8:
      *pOut = ${config.bn_fn}(sum, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = ${config.bn_fn}(sum5, *pKappa, *pLambda, out_shift);
      pOut2++;
%if config.kernel.matmul_fmt == '4x4':
      *pOut3 = ${config.bn_fn}(sum9, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = ${config.bn_fn}(sum13, *pKappa, *pLambda, out_shift);
      pOut4++;
%endif
      pKappa++;
      pLambda++;

      *pOut = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = ${config.bn_fn}(sum6, *pKappa, *pLambda, out_shift);
      pOut2++;
%if config.kernel.matmul_fmt == '4x4':
      *pOut3 = ${config.bn_fn}(sum10, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = ${config.bn_fn}(sum14, *pKappa, *pLambda, out_shift);
      pOut4++;
%endif
      pKappa++;
      pLambda++;

      *pOut = ${config.bn_fn}(sum3, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = ${config.bn_fn}(sum7, *pKappa, *pLambda, out_shift);
      pOut2++;
%if config.kernel.matmul_fmt == '4x4':
      *pOut3 = ${config.bn_fn}(sum11, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = ${config.bn_fn}(sum15, *pKappa, *pLambda, out_shift);
      pOut4++;
%endif
      pKappa++;
      pLambda++;

      *pOut = ${config.bn_fn}(sum4, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = ${config.bn_fn}(sum8, *pKappa, *pLambda, out_shift);
      pOut2++;
%if config.kernel.matmul_fmt == '4x4':
      *pOut3 = ${config.bn_fn}(sum12, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = ${config.bn_fn}(sum16, *pKappa, *pLambda, out_shift);
      pOut4++;
%endif
      pKappa++;
      pLambda++;
%elif config.kernel.out_data_t == 4:
      sum = ${config.bn_fn}(sum, *pKappa, *pLambda, out_shift);
      sum5 = ${config.bn_fn}(sum5, *pKappa, *pLambda, out_shift);
%if config.kernel.matmul_fmt == '4x4':
      sum9 = ${config.bn_fn}(sum9, *pKappa, *pLambda, out_shift);
      sum13 = ${config.bn_fn}(sum13, *pKappa, *pLambda, out_shift);
%endif
      pKappa++;
      pLambda++;
      sum2 = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
      sum6 = ${config.bn_fn}(sum6, *pKappa, *pLambda, out_shift);
%if config.kernel.matmul_fmt == '4x4':
      sum10 = ${config.bn_fn}(sum10, *pKappa, *pLambda, out_shift);
      sum14 = ${config.bn_fn}(sum14, *pKappa, *pLambda, out_shift);
%endif
      *pOut = bitins(sum, n_mask, sum2, mask, off);
      *pOut2 = bitins(sum5, n_mask, sum6, mask, off);
%if config.kernel.matmul_fmt == '4x4':
      *pOut3 = bitins(sum9, n_mask, sum10, mask, off);
      *pOut4 = bitins(sum13, n_mask, sum14, mask, off);
%endif
      pKappa++;
      pLambda++;
      pOut++;
      pOut2++;
%if config.kernel.matmul_fmt == '4x4':
      pOut3++;
      pOut4++;
%endif
      sum3 = ${config.bn_fn}(sum3, *pKappa, *pLambda, out_shift);
      sum7 = ${config.bn_fn}(sum7, *pKappa, *pLambda, out_shift);
%if config.kernel.matmul_fmt == '4x4':
      sum11 = ${config.bn_fn}(sum11, *pKappa, *pLambda, out_shift);
      sum15 = ${config.bn_fn}(sum15, *pKappa, *pLambda, out_shift);
%endif
      pKappa++;
      pLambda++;
      sum4 = ${config.bn_fn}(sum4, *pKappa, *pLambda, out_shift);
      sum8 = ${config.bn_fn}(sum8, *pKappa, *pLambda, out_shift);
%if config.kernel.matmul_fmt == '4x4':
      sum12 = ${config.bn_fn}(sum12, *pKappa, *pLambda, out_shift);
      sum16 = ${config.bn_fn}(sum16, *pKappa, *pLambda, out_shift);
%endif
      pKappa++;
      pLambda++;
      *pOut = bitins(sum3, n_mask, sum4, mask, off);
      *pOut2 = bitins(sum7, n_mask, sum8, mask, off);
%if config.kernel.matmul_fmt == '4x4':
      *pOut3 = bitins(sum11, n_mask, sum12, mask, off);
      *pOut4 = bitins(sum15, n_mask, sum16, mask, off);
%endif
      pOut++;
      pOut2++;
%if config.kernel.matmul_fmt == '4x4':
      pOut3++;
      pOut4++;
%endif
%elif config.kernel.out_data_t == 2:
      sum = ${config.bn_fn}(sum, *pKappa, *pLambda, out_shift);
      sum5 = ${config.bn_fn}(sum5, *pKappa, *pLambda, out_shift);
%if config.kernel.matmul_fmt == '4x4':
      sum9 = ${config.bn_fn}(sum9, *pKappa, *pLambda, out_shift);
      sum13 = ${config.bn_fn}(sum13, *pKappa, *pLambda, out_shift);
%endif
      pKappa++;
      pLambda++;
      sum2 = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
      sum6 = ${config.bn_fn}(sum6, *pKappa, *pLambda, out_shift);
%if config.kernel.matmul_fmt == '4x4':
      sum10 = ${config.bn_fn}(sum10, *pKappa, *pLambda, out_shift);
      sum14 = ${config.bn_fn}(sum14, *pKappa, *pLambda, out_shift);
%endif
      pKappa++;
      pLambda++;
      sum3 = ${config.bn_fn}(sum3, *pKappa, *pLambda, out_shift);
      sum7 = ${config.bn_fn}(sum7, *pKappa, *pLambda, out_shift);
%if config.kernel.matmul_fmt == '4x4':
      sum11 = ${config.bn_fn}(sum11, *pKappa, *pLambda, out_shift);
      sum15 = ${config.bn_fn}(sum15, *pKappa, *pLambda, out_shift);
%endif
      pKappa++;
      pLambda++;
      sum4 = ${config.bn_fn}(sum4, *pKappa, *pLambda, out_shift);
      sum8 = ${config.bn_fn}(sum8, *pKappa, *pLambda, out_shift);
%if config.kernel.matmul_fmt == '4x4':
      sum12 = ${config.bn_fn}(sum12, *pKappa, *pLambda, out_shift);
      sum16 = ${config.bn_fn}(sum16, *pKappa, *pLambda, out_shift);
%endif
      pKappa++;
      pLambda++;
      sum = bitins(sum, n_mask2, sum2, mask2, off2);
      sum = bitins(sum, n_mask4, sum3, mask4, off4);
      *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
      sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
      sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
      *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);
%if config.kernel.matmul_fmt == '4x4':
      sum9 = bitins(sum9, n_mask2, sum10, mask2, off2);
      sum9 = bitins(sum9, n_mask4, sum11, mask4, off4);
      *pOut3 = bitins(sum9, n_mask6, sum12, mask6, off6);
      sum13 = bitins(sum13, n_mask2, sum14, mask2, off2);
      sum13 = bitins(sum13, n_mask4, sum15, mask4, off4);
      *pOut4 = bitins(sum13, n_mask6, sum16, mask6, off6);
      pOut4++;
      pOut3++;
%endif
      pOut2++;
      pOut++;
%endif
    }
    else
    {
      if (flag_relu == 1)
      {
%if config.kernel.out_data_t == 8:
        *pOut = ${config.relu_fn}(sum, out_mult, out_shift);
        pOut++;
        *pOut = ${config.relu_fn}(sum2, out_mult, out_shift);
        pOut++;
        *pOut = ${config.relu_fn}(sum3, out_mult, out_shift);
        pOut++;
        *pOut = ${config.relu_fn}(sum4, out_mult, out_shift);
        pOut++;

        *pOut2 = ${config.relu_fn}(sum5, out_mult, out_shift);
        pOut2++;
        *pOut2 = ${config.relu_fn}(sum6, out_mult, out_shift);
        pOut2++;
        *pOut2 = ${config.relu_fn}(sum7, out_mult, out_shift);
        pOut2++;
        *pOut2 = ${config.relu_fn}(sum8, out_mult, out_shift);
        pOut2++;

%if config.kernel.matmul_fmt == '4x4':
        *pOut3 = ${config.relu_fn}(sum9, out_mult, out_shift);
        pOut3++;
        *pOut3 = ${config.relu_fn}(sum10, out_mult, out_shift);
        pOut3++;
        *pOut3 = ${config.relu_fn}(sum11, out_mult, out_shift);
        pOut3++;
        *pOut3 = ${config.relu_fn}(sum12, out_mult, out_shift);
        pOut3++;

        *pOut4 = ${config.relu_fn}(sum13, out_mult, out_shift);
        pOut4++;
        *pOut4 = ${config.relu_fn}(sum14, out_mult, out_shift);
        pOut4++;
        *pOut4 = ${config.relu_fn}(sum15, out_mult, out_shift);
        pOut4++;
        *pOut4 = ${config.relu_fn}(sum16, out_mult, out_shift);
        pOut4++;
%endif
%elif config.kernel.out_data_t == 4:
        sum = ${config.relu_fn}(sum, out_mult, out_shift);
        sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
        *pOut = bitins(sum, n_mask, sum2, mask, off);
        pOut++;
        sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
        sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
        *pOut = bitins(sum3, n_mask, sum4, mask, off);
        pOut++;

        sum5 = ${config.relu_fn}(sum5, out_mult, out_shift);
        sum6 = ${config.relu_fn}(sum6, out_mult, out_shift);
        *pOut2 = bitins(sum5, n_mask, sum6, mask, off);
        pOut2++;
        sum7 = ${config.relu_fn}(sum7, out_mult, out_shift);
        sum8 = ${config.relu_fn}(sum8, out_mult, out_shift);
        *pOut2 = bitins(sum7, n_mask, sum8, mask, off);
        pOut2++;

%if config.kernel.matmul_fmt == '4x4':
        sum9 = ${config.relu_fn}(sum9, out_mult, out_shift);
        sum10 = ${config.relu_fn}(sum10, out_mult, out_shift);
        *pOut3 = bitins(sum9, n_mask, sum10, mask, off);
        pOut3++;
        sum11 = ${config.relu_fn}(sum11, out_mult, out_shift);
        sum12 = ${config.relu_fn}(sum12, out_mult, out_shift);
        *pOut3 = bitins(sum11, n_mask, sum12, mask, off);
        pOut3++;

        sum13 = ${config.relu_fn}(sum13, out_mult, out_shift);
        sum14 = ${config.relu_fn}(sum14, out_mult, out_shift);
        *pOut4 = bitins(sum13, n_mask, sum14, mask, off);
        pOut4++;
        sum15 = ${config.relu_fn}(sum15, out_mult, out_shift);
        sum16 = ${config.relu_fn}(sum16, out_mult, out_shift);
        *pOut4 = bitins(sum15, n_mask, sum16, mask, off);
        pOut4++;
%endif
%elif config.kernel.out_data_t == 2:
        sum = ${config.relu_fn}(sum, out_mult, out_shift);
        sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
        sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
        sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
        sum = bitins(sum, n_mask2, sum2, mask2, off2);
        sum = bitins(sum, n_mask4, sum3, mask4, off4);
        *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
        pOut++;
        sum5 = ${config.relu_fn}(sum5, out_mult, out_shift);
        sum6 = ${config.relu_fn}(sum6, out_mult, out_shift);
        sum7 = ${config.relu_fn}(sum7, out_mult, out_shift);
        sum8 = ${config.relu_fn}(sum8, out_mult, out_shift);
        sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
        sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
        *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);
        pOut2++;
%if config.kernel.matmul_fmt == '4x4':
        sum9 = ${config.relu_fn}(sum9, out_mult, out_shift);
        sum10 = ${config.relu_fn}(sum10, out_mult, out_shift);
        sum11 = ${config.relu_fn}(sum11, out_mult, out_shift);
        sum12 = ${config.relu_fn}(sum12, out_mult, out_shift);
        sum9 = bitins(sum9, n_mask2, sum10, mask2, off2);
        sum9 = bitins(sum9, n_mask4, sum11, mask4, off4);
        *pOut3 = bitins(sum9, n_mask6, sum12, mask6, off6);
        pOut3++;
        sum13 = ${config.relu_fn}(sum13, out_mult, out_shift);
        sum14 = ${config.relu_fn}(sum14, out_mult, out_shift);
        sum15 = ${config.relu_fn}(sum15, out_mult, out_shift);
        sum16 = ${config.relu_fn}(sum16, out_mult, out_shift);
        sum13 = bitins(sum13, n_mask2, sum14, mask2, off2);
        sum13 = bitins(sum13, n_mask4, sum15, mask4, off4);
        *pOut4 = bitins(sum13, n_mask6, sum16, mask6, off6);
        pOut4++;
%endif
%endif
      }
      else
      {
%if config.kernel.out_data_t == 8:
        *pOut = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
        pOut++;
        *pOut = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
        pOut++;
        *pOut = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
        pOut++;
        *pOut = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
        pOut++;

        *pOut2 = (${pt_out}) ${out_clip_fn}(sum5 >> out_shift);
        pOut2++;
        *pOut2 = (${pt_out}) ${out_clip_fn}(sum6 >> out_shift);
        pOut2++;
        *pOut2 = (${pt_out}) ${out_clip_fn}(sum7 >> out_shift);
        pOut2++;
        *pOut2 = (${pt_out}) ${out_clip_fn}(sum8 >> out_shift);
        pOut2++;

%if config.kernel.matmul_fmt == '4x4':
        *pOut3 = (${pt_out}) ${out_clip_fn}(sum9 >> out_shift);
        pOut3++;
        *pOut3 = (${pt_out}) ${out_clip_fn}(sum10 >> out_shift);
        pOut3++;
        *pOut3 = (${pt_out}) ${out_clip_fn}(sum11 >> out_shift);
        pOut3++;
        *pOut3 = (${pt_out}) ${out_clip_fn}(sum12 >> out_shift);
        pOut3++;

        *pOut4 = (${pt_out}) ${out_clip_fn}(sum13 >> out_shift);
        pOut4++;
        *pOut4 = (${pt_out}) ${out_clip_fn}(sum14 >> out_shift);
        pOut4++;
        *pOut4 = (${pt_out}) ${out_clip_fn}(sum15 >> out_shift);
        pOut4++;
        *pOut4 = (${pt_out}) ${out_clip_fn}(sum16 >> out_shift);
        pOut4++;
%endif
%elif config.kernel.out_data_t == 4:
        sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
        sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
        *pOut = bitins(sum, n_mask, sum2, mask, off);
        pOut++;
        sum3 = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
        sum4 = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
        *pOut = bitins(sum3, n_mask, sum4, mask, off);
        pOut++;

        sum5 = (${pt_out}) ${out_clip_fn}(sum5 >> out_shift);
        sum6 = (${pt_out}) ${out_clip_fn}(sum6 >> out_shift);
        *pOut2 = bitins(sum5, n_mask, sum6, mask, off);
        pOut2++;
        sum7 = (${pt_out}) ${out_clip_fn}(sum7 >> out_shift);
        sum8 = (${pt_out}) ${out_clip_fn}(sum8 >> out_shift);
        *pOut2 = bitins(sum7, n_mask, sum8, mask, off);
        pOut2++;

%if config.kernel.matmul_fmt == '4x4':
        sum9 = (${pt_out}) ${out_clip_fn}(sum9 >> out_shift);
        sum10 = (${pt_out}) ${out_clip_fn}(sum10 >> out_shift);
        *pOut3 = bitins(sum9, n_mask, sum10, mask, off);
        pOut3++;
        sum11 = (${pt_out}) ${out_clip_fn}(sum11 >> out_shift);
        sum12 = (${pt_out}) ${out_clip_fn}(sum12 >> out_shift);
        *pOut3 = bitins(sum11, n_mask, sum12, mask, off);
        pOut3++;

        sum13 = (${pt_out}) ${out_clip_fn}(sum13 >> out_shift);
        sum14 = (${pt_out}) ${out_clip_fn}(sum14 >> out_shift);
        *pOut4 = bitins(sum13, n_mask, sum14, mask, off);
        pOut4++;
        sum15 = (${pt_out}) ${out_clip_fn}(sum15 >> out_shift);
        sum16 = (${pt_out}) ${out_clip_fn}(sum16 >> out_shift);
        pOut4++;
%endif
%elif config.kernel.out_data_t == 2:
        sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
        sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
        sum3 = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
        sum4 = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
        sum = bitins(sum, n_mask2, sum2, mask2, off2);
        sum = bitins(sum, n_mask4, sum3, mask4, off4);
        *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
        pOut++;

        sum5 = (${pt_out}) ${out_clip_fn}(sum5 >> out_shift);
        sum6 = (${pt_out}) ${out_clip_fn}(sum6 >> out_shift);
        sum7 = (${pt_out}) ${out_clip_fn}(sum7 >> out_shift);
        sum8 = (${pt_out}) ${out_clip_fn}(sum8 >> out_shift);
        sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
        sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
        *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);
        pOut2++;

%if config.kernel.matmul_fmt == '4x4':
        sum9 = (${pt_out}) ${out_clip_fn}(sum9 >> out_shift);
        sum10 = (${pt_out}) ${out_clip_fn}(sum10 >> out_shift);
        sum11 = (${pt_out}) ${out_clip_fn}(sum11 >> out_shift);
        sum12 = (${pt_out}) ${out_clip_fn}(sum12 >> out_shift);
        sum9 = bitins(sum9, n_mask2, sum10, mask2, off2);
        sum9 = bitins(sum9, n_mask4, sum11, mask4, off4);
        *pOut3 = bitins(sum9, n_mask6, sum12, mask6, off6);
        pOut3++;

        sum13 = (${pt_out}) ${out_clip_fn}(sum13 >> out_shift);
        sum14 = (${pt_out}) ${out_clip_fn}(sum14 >> out_shift);
        sum15 = (${pt_out}) ${out_clip_fn}(sum15 >> out_shift);
        sum16 = (${pt_out}) ${out_clip_fn}(sum16 >> out_shift);
        sum13 = bitins(sum13, n_mask2, sum14, mask2, off2);
        sum13 = bitins(sum13, n_mask4, sum15, mask4, off4);
        *pOut4 = bitins(sum13, n_mask6, sum16, mask6, off6);
        pOut4++;
%endif
%endif
      }
    }
%elif config.kernel.out_data_t == 4:
    sum = pulp_nn_i4_quant(sum, pThr);
    sum5 = pulp_nn_i4_quant(sum5, pThr);
%if config.kernel.matmul_fmt == '4x4':
    sum9 = pulp_nn_i4_quant(sum9, pThr);
    sum13 = pulp_nn_i4_quant(sum13, pThr);
%endif

    pThr+=16;

    sum2 = pulp_nn_i4_quant(sum2, pThr);
    sum6 = pulp_nn_i4_quant(sum6, pThr);
%if config.kernel.matmul_fmt == '4x4':
    sum10 = pulp_nn_i4_quant(sum10, pThr);
    sum14 = pulp_nn_i4_quant(sum14, pThr);
%endif

    pThr+=16;

    sum3 = pulp_nn_i4_quant(sum3, pThr);
    sum7 = pulp_nn_i4_quant(sum7, pThr);
%if config.kernel.matmul_fmt == '4x4':
    sum11 = pulp_nn_i4_quant(sum11, pThr);
    sum15 = pulp_nn_i4_quant(sum15, pThr);
%endif

    pThr+=16;

    sum4 = pulp_nn_i4_quant(sum4, pThr);
    sum8 = pulp_nn_i4_quant(sum8, pThr);
%if config.kernel.matmul_fmt == '4x4':
    sum12 = pulp_nn_i4_quant(sum12, pThr);
    sum16 = pulp_nn_i4_quant(sum16, pThr);
%endif


    pThr+=16;

    *pOut = bitins(sum, n_mask, sum2, mask, off);

    pOut++;

    *pOut2 = bitins(sum5, n_mask, sum6, mask, off);

    pOut2++;

%if config.kernel.matmul_fmt == '4x4':
    *pOut3 = bitins(sum9, n_mask, sum10, mask, off);

    pOut3++;

    *pOut4 = bitins(sum13, n_mask, sum14, mask, off);

    pOut4++;
%endif

    *pOut = bitins(sum3, n_mask, sum4, mask, off);

    pOut++;

    *pOut2 = bitins(sum7, n_mask, sum8, mask, off);

    pOut2++;

%if config.kernel.matmul_fmt == '4x4':
    *pOut3 = bitins(sum11, n_mask, su12, mask, off);

    pOut3++;

    *pOut4 = bitins(sum15, n_mask, sum16, mask, off);

    pOut4++;
%endif
%elif config.kernel.out_data_t == 2:
    sum = pulp_nn_i2_quant(sum, pThr);
    sum5 = pulp_nn_i2_quant(sum5, pThr);
%if config.kernel.matmul_fmt == '4x4':
    sum9 = pulp_nn_i2_quant(sum9, pThr);
    sum13 = pulp_nn_i2_quant(sum13, pThr);
%endif

    pThr+=4;

    sum2 = pulp_nn_i2_quant(sum2, pThr);
    sum6 = pulp_nn_i2_quant(sum6, pThr);
%if config.kernel.matmul_fmt == '4x4':
    sum10 = pulp_nn_i2_quant(sum10, pThr);
    sum14 = pulp_nn_i2_quant(sum14, pThr);
%endif

    sum = bitins(sum, n_mask2, sum2, mask2, off2);
    sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
%if config.kernel.matmul_fmt == '4x4':
    sum9 = bitins(sum9, n_mask2, sum10, mask2, off2);
    sum13 = bitins(sum13, n_mask2, sum14, mask2, off2);
%endif

    pThr+=4;

    sum3 = pulp_nn_i2_quant(sum3, pThr);
    sum7 = pulp_nn_i2_quant(sum7, pThr);
%if config.kernel.matmul_fmt == '4x4':
    sum11 = pulp_nn_i2_quant(sum11, pThr);
    sum15 = pulp_nn_i2_quant(sum15, pThr);
%endif

    sum = bitins(sum, n_mask4, sum3, mask4, off4);
    sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
%if config.kernel.matmul_fmt == '4x4':
    sum9 = bitins(sum9, n_mask4, sum11, mask4, off4);
    sum13 = bitins(sum13, n_mask4, sum15, mask4, off4);
%endif

    pThr+=4;

    sum4 = pulp_nn_i2_quant(sum4, pThr);
    sum8 = pulp_nn_i2_quant(sum8, pThr);
%if config.kernel.matmul_fmt == '4x4':
    sum12 = pulp_nn_i2_quant(sum12, pThr);
    sum16 = pulp_nn_i2_quant(sum16, pThr);
%endif

    pThr+=4;

    *pOut = bitins(sum, n_mask6, sum4, mask6, off6);

    pOut++;

    *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);

    pOut2++;

%if config.kernel.matmul_fmt == '4x4':
    *pOut3 = bitins(sum9, n_mask6, sum12, mask6, off6);

    pOut3++;

    *pOut4 = bitins(sum13, n_mask6, sum16, mask6, off6);

    pOut4++;
%endif
%endif
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    pA+=(3 * num_col_im2col_w);
%else:
    pA+=(4 * num_col_im2col_w);
%endif
  }
%if config.kernel.out_data_t != 2:
%if config.kernel.out_data_t == 4:
  int i = 0;
%endif
  while(chan_left)
  {
    ${pt_in} *pB = pIn;
    ${pt_in} *pB2 = (pB + num_col_im2col_a);

    ${int_t_in} *ptrB  = (${int_t_in} *) pB;
    ${int_t_in} *ptrB2 = (${int_t_in} *) pB2;

%if config.kernel.wt_data_t < config.kernel.in_data_t:
    pA  = ${config.unpack_wt_fn}(pA , vecA);

    int32_t *startA;

    asm volatile("mv %0, %1":"=r"(startA):"r"(vecA));

    int32_t *ptrA  = (int32_t *) vecA;
%else:
    int32_t *ptrA  = (int32_t *) pA;
%endif

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

    int sum = 0;
    if (pBias != NULL)
    {
      sum = *((int*) pBias++);
    }
    int sum2 = sum;

%if config.kernel.out_data_t == 4:
    ${pt_out} out[2];
    ${pt_out} out2[2];
%endif
    for(int j=0; j < (num_col_im2col >> ${int(math.log2(((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t)))))}); j++)
    {
      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = ${mac_fn}(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

      sum2 = ${mac_fn}(1, 0, 0, 1, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.wt_data_t < config.kernel.in_data_t:
%if (config.kernel.in_data_t/config.kernel.wt_data_t) == 4:
      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = ${mac_fn}(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

      sum2 = ${mac_fn}(1, 0, 0, 1, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);

      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = ${mac_fn}(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

      sum2 = ${mac_fn}(1, 0, 0, 1, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
%endif
      ptrB2  = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = ${mac_fn}(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

      pA  = ${config.unpack_wt_fn}(pA , vecA);

      ptrA   = MacLoadAssign(vecA);

      sum2  = ${mac_fn}(1, 0, 0, 1, ptrA, sum2);
      ptrA  = MacLoadUpdate(ptrA);
%endif
    }
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    pA-=4;
%endif
    int col_cnt_im2col = num_col_im2col & ${hex((((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t))))-1)};

    if(col_cnt_im2col)
    {
%if config.kernel.wt_data_t >= config.kernel.in_data_t:
      uint16_t loop_cnt_im2col_w = (num_col_im2col >> ${int(math.log2(((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t)))))}) << 2;
      pA+=loop_cnt_im2col_w;
%endif

%if config.kernel.wt_data_t < config.kernel.in_data_t:
      uint16_t loop_cnt_im2col_a = (num_col_im2col >> ${int(math.log2(((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t)))))}) << ${int(2+int(math.log2(int(config.kernel.in_data_t/config.kernel.wt_data_t))))};
%else:
      uint16_t loop_cnt_im2col_a = (num_col_im2col >> ${int(math.log2(((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t)))))}) << 2;
%endif
      pB+=loop_cnt_im2col_a;
      pB2+=loop_cnt_im2col_a;

      do
      {
%if config.kernel.in_data_t == 8:
%if config.kernel.wt_data_t == 2:
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);

        ${pt_out} inB = *pB++;
        ${pt_out} inB2 = *pB2++;

        sum += inA * inB;

        sum2 += inA * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 2);

        inB = *pB++;
        inB2 = *pB2++;

        sum += inA * inB;

        sum2 += inA * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 4);

        inB = *pB++;
        inB2 = *pB2++;

        sum += inA * inB;

        sum2 += inA * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 6);

        inB = *pB++;
        inB2 = *pB2++;

        sum += inA * inB;

        sum2 += inA * inB2;

        pA++;

        col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
        int8_t inA = (int8_t) bitext((int) *pA, 4, 0);

        ${pt_out} inB = *pB++;
        ${pt_out} inB2 = *pB2++;

        sum += inA * inB;

        sum2 += inA * inB2;

        inA = (int8_t) bitext((int) *pA, 4, 4);

        inB = *pB++;
        inB2 = *pB2++;

        sum += inA * inB;

        sum2 += inA * inB2;

        pA++;

        col_cnt_im2col-=2;
%else:
        int8_t inA = *pA++;

        ${pt_out} inB = *pB++;
        ${pt_out} inB2 = *pB2++;
        asm volatile("": : :"memory");
        sum += inA * inB;

        sum2 += inA * inB2;

        col_cnt_im2col--;
%endif
%elif config.kernel.in_data_t == 4:
%if config.kernel.wt_data_t == 2:
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);

        ${pt_out} inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_out} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);

        sum += inA * inB;

        sum2 += inA * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 2);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);

        sum += inA * inB;

        sum2 += inA * inB2;

        pB++;
        pB2++;

        inA = (int8_t) bitext((int) *pA, 2, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);

        sum += inA * inB;

        sum2 += inA * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 6);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);

        sum += inA * inB;

        sum2 += inA * inB2;

        pA++;

        pB++;
        pB2++;

        col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
        int8_t inA = (int8_t) bitext((int) *pA, 4, 0);

        ${pt_out} inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_out} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);

        sum += inA * inB;

        sum2 += inA * inB2;

        inA = (int8_t) bitext((int) *pA, 4, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);

        sum += inA * inB;

        sum2 += inA * inB2;

        pA++;

        pB++;
        pB2++;

        col_cnt_im2col-=2;
%else:
        int8_t inA = *pA++;

        ${pt_out} inB = *pB++;
        ${pt_out} inB2 = *pB2++;
        asm volatile("": : :"memory");
        sum += inA * inB;

        sum2 += inA * inB2;

        col_cnt_im2col--;
%endif
%elif config.kernel.in_data_t == 2:
%if config.kernel.wt_data_t == 2:
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);

        ${pt_out} inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 0);
        ${pt_out} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 0);

        sum += inA * inB;

        sum2 += inA * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 2);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 2);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 2);

        sum += inA * inB;

        sum2 += inA * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 4);

        sum += inA * inB;

        sum2 += inA * inB2;

        inA = (int8_t) bitext((int) *pA, 2, 6);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 6);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 6);

        sum += inA * inB;

        sum2 += inA * inB2;

        pA++;

        pB++;
        pB2++;

        col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
        int8_t inA = (int8_t) bitext((int) *pA, 4, 0);

        ${pt_out} inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_out} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);

        sum += inA * inB;

        sum2 += inA * inB2;

        inA = (int8_t) bitext((int) *pA, 4, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);

        sum += inA * inB;

        sum2 += inA * inB2;

        pA++;

        pB++;
        pB2++;

        col_cnt_im2col-=2;
%else:
        int8_t inA = *pA++;

        ${pt_out} inB = *pB++;
        ${pt_out} inB2 = *pB2++;
        asm volatile("": : :"memory");
        sum += inA * inB;

        sum2 += inA * inB2;

        col_cnt_im2col--;
%endif
%endif
      } while(col_cnt_im2col > 0);
%if config.kernel.wt_data_t >= config.kernel.in_data_t:
      pA-=num_col_im2col_w;
%endif
    }
%if config.kernel.out_data_t == 8 or config.kernel.quantization == 'shift_clip':
    if (flag_batch_norm && flag_relu)
    {
%if config.kernel.out_data_t == 8:
      *pOut = ${config.bn_fn}(sum, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
      pOut2++;
      pKappa++;
      pLambda++;
%elif config.kernel.out_data_t == 4:
      uint8_t i_o = i & 0x01;
      out[i_o] = ${config.bn_fn}(sum, *pKappa, *pLambda, out_shift);
      out2[i_o] = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      if(i_o == 0x01)
      {
        *pOut = bitins(out[0], n_mask, out[1], mask, off);
        *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
        pOut++;
        pOut2++;
      }
%endif
    }
    else
    {
      if (flag_relu == 1)
      {
%if config.kernel.out_data_t == 8:
        *pOut = ${config.relu_fn}(sum, out_mult, out_shift);
        pOut++;
        *pOut2 = ${config.relu_fn}(sum2, out_mult, out_shift);
        pOut2++;
%elif config.kernel.out_data_t == 4:
        uint8_t i_o = i & 0x01;
        out[i_o] = ${config.relu_fn}(sum, out_mult, out_shift);
        out2[i_o] = ${config.relu_fn}(sum2, out_mult, out_shift);
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask, out[1], mask, off);
          *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
          pOut++;
          pOut2++;
        }
%endif
      }
      else
      {
%if config.kernel.out_data_t == 8:
        *pOut = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
        pOut++;
        *pOut2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
        pOut2++;
%elif config.kernel.out_data_t == 4:
        uint8_t i_o = i & 0x01;
        out[i_o] = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
        out2[i_o] = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask, out[1], mask, off);
          *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
          pOut++;
          pOut2++;
        }
%endif
      }
    }
%elif config.kernel.out_data_t == 4:
    uint8_t i_o = i & 0x01;
    out[i_o] = pulp_nn_i4_quant(sum, pThr);
    out2[i_o] = pulp_nn_i4_quant(sum2, pThr);
    pThr+=16;
    if(i_o == 0x01)
    {
      *pOut = bitins(out[0], n_mask, out[1], mask, off);
      *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
      pOut++;
      pOut2++;
    }
%endif
%if config.kernel.out_data_t == 4:
    i++;
%endif
%if config.kernel.wt_data_t >= config.kernel.in_data_t:
    pA+=num_col_im2col_w;
%endif
    chan_left--;
  }
%endif
%if config.kernel.matmul_fmt == '4x2':
  pOut+=ch_out_r;
%elif config.kernel.matmul_fmt == '4x4':
  pOut += 3 * ch_out_r;
%endif
  return pOut;
}
