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

pt_in = f"{u_(config.kernel.in_signed)}int8_t"
vt_in = f"v4{su(config.kernel.in_signed)}"
int_t_in = f"{u_(config.kernel.in_signed)}int32_t"
pt_out = f"{u_(config.kernel.out_signed)}int8_t"
sumdotp_fn = f"SumDotp{s_(config.kernel.in_signed)}4"
out_clip_fn = f"clip{s_(config.kernel.out_signed)}{config.kernel.out_data_t}"
bex = f"bitext{u_(config.kernel.in_signed)}"
%>


void ${config.fn_name}(
                        ${pt_in} *pIn,
                        int8_t *pBias,
                        ${pt_out} *pOut,
                        int8_t *pWeight,
                        ${act_t} *pKappa,
                        ${act_t} *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons,
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
%if config.kernel.in_data_t == 8:
    uint16_t dim_vec_in = dim_vec;
%elif config.kernel.in_data_t == 4:
    uint16_t dim_vec_in = dim_vec >> 1;
%elif config.kernel.in_data_t == 2:
    uint16_t dim_vec_in = dim_vec >> 2;
%endif
%if config.kernel.wt_data_t == 8:
    uint16_t dim_vec_wt = dim_vec;
%elif config.kernel.wt_data_t == 4:
    uint16_t dim_vec_wt = dim_vec >> 1;
%elif config.kernel.wt_data_t == 2:
    uint16_t dim_vec_wt = dim_vec >> 2;
%endif

    int core_id = pi_core_id();
    int Log2Core = log2(NUM_CORES);
    int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
%if config.kernel.out_data_t == 8:
    int start = min(chunk * core_id, num_o_neurons);
    int stop = min(start + chunk, num_o_neurons);
%elif config.kernel.out_data_t == 4:
    int start = min((chunk << (chunk == 1)) * core_id, num_o_neurons);
    int stop = min(start + (chunk << (chunk == 1)), num_o_neurons);
%elif config.kernel.out_data_t == 2:
    int neuron_left = 0;
    if (chunk & 0x3)
    {
        neuron_left = (4 - (chunk & 0x7));
    }
    int start = min((chunk + neuron_left) * core_id, num_o_neurons);
    int stop = min(start + chunk + neuron_left, num_o_neurons);
%endif

%if config.less_precision == 8:
    ${vt_in} vecA;
    v4s vecB;
    v4s vecB2;
%if config.kernel.out_data_t == 2:
    v4s vecB3;
    v4s vecB4;
%endif
%elif config.less_precision == 4:
    ${vt_in} vecA[2];
    v4s vecB[2];
    v4s vecB2[2];
%if config.kernel.out_data_t == 2:
    v4s vecB3[2];
    v4s vecB4[2];
%endif
%elif config.less_precision == 2:
    ${vt_in} vecA[4];
    v4s vecB[4];
    v4s vecB2[4];
%if config.kernel.out_data_t == 2:
    v4s vecB3[4];
    v4s vecB4[4];
%endif
%endif

%if config.kernel.out_data_t == 8:
    ${pt_out} *pOutBuffer = (${pt_out} *) pOut + start;
    int lft_neurons = chunk & 0x01;
    int stop_even = stop - lft_neurons;
%elif config.kernel.out_data_t == 4:
    ${pt_out} *pOutBuffer = (${pt_out} *) pOut + (start >> 1);
%elif config.kernel.out_data_t == 2:
    ${pt_out} *pOutBuffer = (${pt_out} *) pOut + (start >> 2);
%endif

    int i;
    ${act_t} *k1 = pKappa + start;
    ${act_t} *lambda1 = pLambda + start;

%if config.kernel.out_data_t == 2:
    for(i=start; i<stop; i+=4)
%elif config.kernel.out_data_t == 4:
    for(i=start; i<stop; i+=2)
%elif config.kernel.out_data_t == 8:
    for(i=start; i<stop_even; i+=2)
%endif
    {
        int sum = 0;
        int sum2 = 0;
        if (pBias != NULL)
        {
          sum = *(int32_t *)(pBias + 4*i);
          sum2 = *(int32_t *)(pBias + 4*i + 4);
        }

%if config.kernel.out_data_t == 2:
        int sum3 = 0;
        int sum4 = 0;
        if (pBias != NULL)
        {
          sum3 = *(int32_t *)(pBias + 4*i + 8);
          sum4 = *(int32_t *)(pBias + 4*i + 12);
        }
%endif

        ${pt_in} *pA = pIn;
        int8_t *pB = pWeight + (i * dim_vec_wt);
        int8_t *pB2 = pB + dim_vec_wt;
%if config.kernel.out_data_t == 2:
        int8_t *pB3 = pB2 + dim_vec_wt;
        int8_t *pB4 = pB3 + dim_vec_wt;
%endif

%if config.less_precision == 8:
        for (int j=0; j<(dim_vec >> 2); j++)
%elif config.less_precision == 4:
        for (int j=0; j<(dim_vec >> 3); j++)
%elif config.less_precision == 2:
        for (int j=0; j<(dim_vec >> 4); j++)
%endif
        {
%if config.less_precision == 8:
          vecA = *((${vt_in}*)pA);
          vecB = *((v4s*)pB);
          vecB2 = *((v4s*)pB2);
%if config.kernel.out_data_t == 2:
          vecB3 = *((v4s*)pB3);
          vecB4 = *((v4s*)pB4);
%endif
          sum = ${sumdotp_fn}(vecA, vecB, sum);
          sum2 = ${sumdotp_fn}(vecA, vecB2, sum2);
%if config.kernel.out_data_t == 2:
          sum3 = ${sumdotp_fn}(vecA, vecB3, sum3);
          sum4 = ${sumdotp_fn}(vecA, vecB4, sum4);
%endif
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 8:
          vecA[0] = *((${vt_in}*)pA);
          pA+=4;
          vecA[1] = *((${vt_in}*)pA);
%else:
          ${config.unpack_in_fn}(pA,vecA);
%endif
%if config.kernel.wt_data_t == 8:
          vecB[0] = *((v4s*)pB);
          vecB2[0] = *((v4s*)pB2);
%if config.kernel.out_data_t == 2:
          vecB3[0] = *((v4s*)pB3);
          vecB4[0] = *((v4s*)pB4);
%endif
          pB+=4;
          pB2+=4;
%if config.kernel.out_data_t == 2:
          pB3+=4;
          pB4+=4;
%endif
          vecB[1] = *((v4s*)pB);
          vecB2[1] = *((v4s*)pB2);
%if config.kernel.out_data_t == 2:
          vecB3[1] = *((v4s*)pB3);
          vecB4[1] = *((v4s*)pB4);
%endif
%else:
          ${config.unpack_wt_fn}(pB,vecB);
          ${config.unpack_wt_fn}(pB2,vecB2);
%if config.kernel.out_data_t == 2:
          ${config.unpack_wt_fn}(pB3,vecB3);
          ${config.unpack_wt_fn}(pB4,vecB4);
%endif
%endif
          sum = ${sumdotp_fn}(vecA[0], vecB[0], sum);
          sum = ${sumdotp_fn}(vecA[1], vecB[1], sum);
          sum2 = ${sumdotp_fn}(vecA[0], vecB2[0], sum2);
          sum2 = ${sumdotp_fn}(vecA[1], vecB2[1], sum2);
%if config.kernel.out_data_t == 2:
          sum3 = ${sumdotp_fn}(vecA[0], vecB3[0], sum3);
          sum3 = ${sumdotp_fn}(vecA[1], vecB3[1], sum3);
          sum4 = ${sumdotp_fn}(vecA[0], vecB4[0], sum4);
          sum4 = ${sumdotp_fn}(vecA[1], vecB4[1], sum4);
%endif
%elif config.less_precision == 2:
%if config.kernel.in_data_t == 8:
          vecA[0] = *((${vt_in}*)pA);
          pA+=4;
          vecA[1] = *((${vt_in}*)pA);
          pA+=4;
          vecA[2] = *((${vt_in}*)pA);
          pA+=4;
          vecA[3] = *((${vt_in}*)pA);
%elif config.kernel.in_data_t == 4:
          ${config.unpack_in_fn}(pA,vecA);
          pA+=4;
          ${config.unpack_in_fn}(pA,vecA + 2);
%elif config.kernel.in_data_t == 2:
          ${config.unpack_in_fn}(pA,vecA);
%endif
%if config.kernel.wt_data_t == 8:
          vecB[0] = *((v4s*)pB);
          vecB2[0] = *((v4s*)pB2);
%if config.kernel.out_data_t == 2:
          vecB3[0] = *((v4s*)pB3);
          vecB4[0] = *((v4s*)pB4);
%endif
          pB+=4;
          pB2+=4;
%if config.kernel.out_data_t == 2:
          pB3+=4;
          pB4+=4;
%endif
          vecB[1] = *((v4s*)pB);
          vecB2[1] = *((v4s*)pB2);
%if config.kernel.out_data_t == 2:
          vecB3[1] = *((v4s*)pB3);
          vecB4[1] = *((v4s*)pB4);
%endif
          pB+=4;
          pB2+=4;
%if config.kernel.out_data_t == 2:
          pB3+=4;
          pB4+=4;
%endif
          vecB[2] = *((v4s*)pB);
          vecB2[2] = *((v4s*)pB2);
%if config.kernel.out_data_t == 2:
          vecB3[2] = *((v4s*)pB3);
          vecB4[2] = *((v4s*)pB4);
%endif
          pB+=4;
          pB2+=4;
%if config.kernel.out_data_t == 2:
          pB3+=4;
          pB4+=4;
%endif
          vecB[3] = *((v4s*)pB);
          vecB2[3] = *((v4s*)pB2);
%if config.kernel.out_data_t == 2:
          vecB3[3] = *((v4s*)pB3);
          vecB4[3] = *((v4s*)pB4);
%endif
%elif config.kernel.wt_data_t == 4:
          ${config.unpack_wt_fn}(pB,vecB);
          ${config.unpack_wt_fn}(pB2,vecB2);
%if config.kernel.out_data_t == 2:
          ${config.unpack_wt_fn}(pB3,vecB3);
          ${config.unpack_wt_fn}(pB4,vecB4);
%endif
          pB+=4;
          pB2+=4;
%if config.kernel.out_data_t == 2:
          pB3+=4;
          pB4+=4;
%endif
          ${config.unpack_wt_fn}(pB,vecB + 2);
          ${config.unpack_wt_fn}(pB2,vecB2 + 2);
%if config.kernel.out_data_t == 2:
          ${config.unpack_wt_fn}(pB,vecB3 + 2);
          ${config.unpack_wt_fn}(pB2,vecB4 + 2);
%endif
%elif config.kernel.wt_data_t == 2:
          ${config.unpack_wt_fn}(pB,vecB);
          ${config.unpack_wt_fn}(pB2,vecB2);
%if config.kernel.out_data_t == 2:
          ${config.unpack_wt_fn}(pB3,vecB3);
          ${config.unpack_wt_fn}(pB4,vecB4);
%endif
%endif
          sum = ${sumdotp_fn}(vecA[0], vecB[0], sum);
          sum = ${sumdotp_fn}(vecA[1], vecB[1], sum);
          sum = ${sumdotp_fn}(vecA[2], vecB[2], sum);
          sum = ${sumdotp_fn}(vecA[3], vecB[3], sum);
          sum2 = ${sumdotp_fn}(vecA[0], vecB2[0], sum2);
          sum2 = ${sumdotp_fn}(vecA[1], vecB2[1], sum2);
          sum2 = ${sumdotp_fn}(vecA[2], vecB2[2], sum2);
          sum2 = ${sumdotp_fn}(vecA[3], vecB2[3], sum2);
%if config.kernel.out_data_t == 2:
          sum3 = ${sumdotp_fn}(vecA[0], vecB3[0], sum3);
          sum3 = ${sumdotp_fn}(vecA[1], vecB3[1], sum3);
          sum3 = ${sumdotp_fn}(vecA[2], vecB3[2], sum3);
          sum3 = ${sumdotp_fn}(vecA[3], vecB3[3], sum3);
          sum4 = ${sumdotp_fn}(vecA[0], vecB4[0], sum4);
          sum4 = ${sumdotp_fn}(vecA[1], vecB4[1], sum4);
          sum4 = ${sumdotp_fn}(vecA[2], vecB4[2], sum4);
          sum4 = ${sumdotp_fn}(vecA[3], vecB4[3], sum4);
%endif
%endif
          pA+=4;
          pB+=4;
          pB2+=4;
%if config.kernel.out_data_t == 2:
          pB3+=4;
          pB4+=4;
%endif
        }
%if config.less_precision == 2:
        uint16_t col_cnt = dim_vec & 0xf;
%elif config.less_precision == 4:
        uint16_t col_cnt = dim_vec & 0x7;
%elif config.less_precision == 8:
        uint16_t col_cnt = dim_vec & 0x3;
%endif
        while (col_cnt)
        {
%if config.less_precision == 2:
%if config.kernel.in_data_t == 2:
          ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 0);
          ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 2);
          ${pt_in} inA3 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 4);
          ${pt_in} inA4 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 6);
          pA++;
%elif config.kernel.in_data_t == 4:
          ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
          ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
          pA++;
          ${pt_in} inA3 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
          ${pt_in} inA4 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
          pA++;
%elif config.kernel.in_data_t == 8:
          ${pt_in} inA = *pA;
          pA++;
          ${pt_in} inA2 = *pA;
          pA++;
          ${pt_in} inA3 = *pA;
          pA++;
          ${pt_in} inA4 = *pA;
          pA++;
%endif
%if config.kernel.wt_data_t == 2:
          int8_t inB = (int8_t) bitext((int) *pB, 2, 0);
          int8_t inB2 = (int8_t) bitext((int) *pB, 2, 2);
          int8_t inB3 = (int8_t) bitext((int) *pB, 2, 4);
          int8_t inB4 = (int8_t) bitext((int) *pB, 2, 6);
          pB++;
          int8_t inB5 = (int8_t) bitext((int) *pB2, 2, 0);
          int8_t inB6 = (int8_t) bitext((int) *pB2, 2, 2);
          int8_t inB7 = (int8_t) bitext((int) *pB2, 2, 4);
          int8_t inB8 = (int8_t) bitext((int) *pB2, 2, 6);
          pB2++;
%if config.kernel.out_data_t == 2:
          int8_t inB9 = (int8_t) bitext((int) *pB3, 2, 0);
          int8_t inB10 = (int8_t) bitext((int) *pB3, 2, 2);
          int8_t inB11 = (int8_t) bitext((int) *pB3, 2, 4);
          int8_t inB12 = (int8_t) bitext((int) *pB3, 2, 6);
          pB3++;
          int8_t inB13 = (int8_t) bitext((int) *pB4, 2, 0);
          int8_t inB14 = (int8_t) bitext((int) *pB4, 2, 2);
          int8_t inB15 = (int8_t) bitext((int) *pB4, 2, 4);
          int8_t inB16 = (int8_t) bitext((int) *pB4, 2, 6);
          pB4++;
%endif
%elif config.kernel.wt_data_t == 4:
          int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
          int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
          pB++;
          int8_t inB3 = (int8_t) bitext((int) *pB, 4, 0);
          int8_t inB4 = (int8_t) bitext((int) *pB, 4, 4);
          pB++;
          int8_t inB5 = (int8_t) bitext((int) *pB2, 4, 0);
          int8_t inB6 = (int8_t) bitext((int) *pB2, 4, 4);
          pB2++;
          int8_t inB7 = (int8_t) bitext((int) *pB2, 4, 0);
          int8_t inB8 = (int8_t) bitext((int) *pB2, 4, 4);
          pB2++;
%if config.kernel.out_data_t == 2:
          int8_t inB9 = (int8_t) bitext((int) *pB3, 4, 0);
          int8_t inB10 = (int8_t) bitext((int) *pB3, 4, 4);
          pB3++;
          int8_t inB11 = (int8_t) bitext((int) *pB3, 4, 0);
          int8_t inB12 = (int8_t) bitext((int) *pB3, 4, 4);
          pB3++;
          int8_t inB13 = (int8_t) bitext((int) *pB4, 4, 0);
          int8_t inB14 = (int8_t) bitext((int) *pB4, 4, 4);
          pB4++;
          int8_t inB15 = (int8_t) bitext((int) *pB4, 4, 0);
          int8_t inB16 = (int8_t) bitext((int) *pB4, 4, 4);
          pB4++;
%endif
%elif config.kernel.wt_data_t == 8:
          int8_t inB = *pB;
          pB++;
          int8_t inB2 = *pB;
          pB++;
          int8_t inB3 = *pB;
          pB++;
          int8_t inB4 = *pB;
          pB++;
          int8_t inB5 = *pB2;
          pB2++;
          int8_t inB6 = *pB2;
          pB2++;
          int8_t inB7 = *pB2;
          pB2++;
          int8_t inB8 = *pB2;
          pB2++;
%if config.kernel.out_data_t == 2:
          int8_t inB9 = *pB3;
          pB3++;
          int8_t inB10 = *pB3;
          pB3++;
          int8_t inB11 = *pB3;
          pB3++;
          int8_t inB12 = *pB3;
          pB3++;
          int8_t inB13 = *pB4;
          pB4++;
          int8_t inB14 = *pB4;
          pB4++;
          int8_t inB15 = *pB4;
          pB4++;
          int8_t inB16 = *pB4;
          pB4++;
%endif
%endif
          sum += inA * inB;
          sum += inA2 * inB2;
          sum += inA3 * inB3;
          sum += inA4 * inB4;
          sum2 += inA * inB5;
          sum2 += inA2 * inB6;
          sum2 += inA3 * inB7;
          sum2 += inA4 * inB8;
%if config.kernel.out_data_t == 2:
          sum3 += inA * inB9;
          sum3 += inA2 * inB10;
          sum3 += inA3 * inB11;
          sum3 += inA4 * inB12;
          sum4 += inA * inB13;
          sum4 += inA2 * inB14;
          sum4 += inA3 * inB15;
          sum4 += inA4 * inB16;
%endif
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 4:
          ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
          ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
          pA++;
%elif config.kernel.in_data_t == 8:
          ${pt_in} inA = *pA;
          pA++;
          ${pt_in} inA2 = *pA;
          pA++;
%endif
%if config.kernel.wt_data_t == 4:
          int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
          int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
          pB++;
          int8_t inB3 = (int8_t) bitext((int) *pB2, 4, 0);
          int8_t inB4 = (int8_t) bitext((int) *pB2, 4, 4);
          pB2++;
%if config.kernel.out_data_t == 2:
          int8_t inB5 = (int8_t) bitext((int) *pB3, 4, 0);
          int8_t inB6 = (int8_t) bitext((int) *pB3, 4, 4);
          pB3++;
          int8_t inB7 = (int8_t) bitext((int) *pB4, 4, 0);
          int8_t inB8 = (int8_t) bitext((int) *pB4, 4, 4);
          pB4++;
%endif
%elif config.kernel.wt_data_t == 8:
          int8_t inB = *pB;
          pB++;
          int8_t inB2 = *pB;
          pB++;
          int8_t inB3 = *pB2;
          pB2++;
          int8_t inB4 = *pB2;
          pB2++;
%if config.kernel.out_data_t == 2:
          int8_t inB5 = *pB3;
          pB3++;
          int8_t inB6 = *pB3;
          pB3++;
          int8_t inB7 = *pB4;
          pB4++;
          int8_t inB8 = *pB4;
          pB4++;
%endif
%endif
          sum += inA * inB;
          sum += inA2 * inB2;
          sum2 += inA * inB3;
          sum2 += inA2 * inB4;
%if config.kernel.out_data_t == 2:
          sum3 += inA * inB5;
          sum3 += inA2 * inB6;
          sum4 += inA * inB7;
          sum4 += inA2 * inB8;
%endif
%elif config.less_precision == 8:
          ${pt_in} inA = *pA;
          pA++;
          int8_t inB = *pB;
          pB++;
          int8_t inB2 = *pB2;
          pB2++;
%if config.kernel.out_data_t == 2:
          int8_t inB3 = *pB3;
          pB3++;
          int8_t inB4 = *pB4;
          pB4++;
%endif
          sum += inA * inB;
          sum2 += inA * inB2;
%if config.kernel.out_data_t == 2:
          sum3 += inA * inB3;
          sum4 += inA * inB4;
%endif
%endif
          col_cnt--;
        }
        if (flag_batch_norm && flag_relu)
        {
%if config.kernel.out_data_t == 8:
          *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          pOutBuffer++;
          *pOutBuffer = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          pOutBuffer++;
          k1+=2;
          lambda1+=2;
%elif config.kernel.out_data_t == 4:
          sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
          pOutBuffer++;
          k1+=2;
          lambda1+=2;
%elif config.kernel.out_data_t == 2:
          sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          sum3 = ${config.bn_fn}(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
          sum4 = ${config.bn_fn}(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
          k1+=4;
          lambda1+=4;
          sum = bitins(sum, n_mask2, sum2, mask2, off2);
          sum = bitins(sum, n_mask4, sum3, mask4, off4);
          *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
          pOutBuffer++;
%endif
        }
        else
        {
          if (flag_relu == 1)
          {
%if config.kernel.out_data_t == 8:
            *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
            pOutBuffer++;
            *pOutBuffer = ${config.relu_fn}(sum2, out_mult, out_shift);
            pOutBuffer++;
%elif config.kernel.out_data_t == 4:
            sum = ${config.relu_fn}(sum, out_mult, out_shift);
            sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
            pOutBuffer++;
%elif config.kernel.out_data_t == 2:
            sum = ${config.relu_fn}(sum, out_mult, out_shift);
            sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
            sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
            sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
            pOutBuffer++;
%endif
          }
          else
          {
%if config.kernel.out_data_t == 8:
            *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            pOutBuffer++;
            *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
            pOutBuffer++;
%elif config.kernel.out_data_t == 4:
            sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
            pOutBuffer++;
%elif config.kernel.out_data_t == 2:
            sum = (${pt_out})  ${out_clip_fn}(sum >> out_shift);
            sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
            sum3 = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
            sum4 = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
            pOutBuffer++;
%endif
          }
        }
    }
%if config.kernel.out_data_t == 8:
    if (lft_neurons && (stop - start) > 0)
    {
        int sum = 0;
        if (pBias != NULL)
        {
          sum = *(int32_t *)(pBias + 4*i);
        }

        ${pt_in} *pA = pIn;
        int8_t *pB = pWeight + (i * dim_vec_wt);

%if config.less_precision == 8:
        for (int j=0; j<(dim_vec >> 2); j++)
%elif config.less_precision == 4:
        for (int j=0; j<(dim_vec >> 3); j++)
%elif config.less_precision == 2:
        for (int j=0; j<(dim_vec >> 4); j++)
%endif
        {
%if config.less_precision == 8:
            vecA = *((${vt_in}*)pA);
            vecB = *((v4s*)pB);
            sum = ${sumdotp_fn}(vecA, vecB, sum);
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 8:
            vecA[0] = *((${vt_in}*)pA);
            pA+=4;
            vecA[1] = *((${vt_in}*)pA);
%else:
            ${config.unpack_in_fn}(pA,vecA);
%endif
%if config.kernel.wt_data_t == 8:
            vecB[0] = *((v4s*)pB);
            pB+=4;
            vecB[1] = *((v4s*)pB);
%else:
            ${config.unpack_wt_fn}(pB,vecB);
%endif
            sum = ${sumdotp_fn}(vecA[0], vecB[0], sum);
            sum = ${sumdotp_fn}(vecA[1], vecB[1], sum);
%elif config.less_precision == 2:
%if config.kernel.in_data_t == 8:
            vecA[0] = *((${vt_in}*)pA);
            pA+=4;
            vecA[1] = *((${vt_in}*)pA);
            pA+=4;
            vecA[2] = *((${vt_in}*)pA);
            pA+=4;
            vecA[3] = *((${vt_in}*)pA);
%elif config.kernel.in_data_t == 4:
            ${config.unpack_in_fn}(pA,vecA);
            pA+=4;
            ${config.unpack_in_fn}(pA,vecA + 2);
%elif config.kernel.in_data_t == 2:
            ${config.unpack_in_fn}(pA,vecA);
%if config.kernel.wt_data_t == 8:
            vecB[0] = *((v4s*)pB);
            pB+=4;
            vecB[1] = *((v4s*)pB);
            pB+=4;
            vecB[2] = *((v4s*)pB);
            pB+=4;
            vecB[3] = *((v4s*)pB);
%elif config.kernel.wt_data_t == 4:
            ${config.unpack_wt_fn}(pB,vecB);
            pB+=4;
            ${config.unpack_wt_fn}(pB,vecB + 2);
%elif config.kernel.wt_data_t == 2:
            ${config.unpack_wt_fn}(pB,vecB);
%endif
            sum = ${sumdotp_fn}(vecA[0], vecB[0], sum);
            sum = ${sumdotp_fn}(vecA[1], vecB[1], sum);
            sum = ${sumdotp_fn}(vecA[2], vecB[2], sum);
            sum = ${sumdotp_fn}(vecA[3], vecB[3], sum);
%endif
%endif
            pA+=4;
            pB+=4;
        }
%if config.less_precision == 2:
        uint16_t col_cnt = dim_vec & 0xf;
%elif config.less_precision == 4:
        uint16_t col_cnt = dim_vec & 0x7;
%elif config.less_precision == 8:
        uint16_t col_cnt = dim_vec & 0x3;
%endif
        while (col_cnt)
        {
%if config.less_precision == 2:
%if config.kernel.in_data_t == 2:
          ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 0);
          ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 2);
          ${pt_in} inA3 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 4);
          ${pt_in} inA4 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 6);
          pA++;
%elif config.kernel.in_data_t == 4:
          ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
          ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
          pA++;
          ${pt_in} inA3 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
          ${pt_in} inA4 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
          pA++;
%elif config.kernel.in_data_t == 8:
          ${pt_in} inA = *pA;
          pA++;
          ${pt_in} inA2 = *pA;
          pA++;
          ${pt_in} inA3 = *pA;
          pA++;
          ${pt_in} inA4 = *pA;
          pA++;
%endif
%if config.kernel.wt_data_t == 2:
          int8_t inB = (int8_t) bitext((int) *pB, 2, 0);
          int8_t inB2 = (int8_t) bitext((int) *pB, 2, 2);
          int8_t inB3 = (int8_t) bitext((int) *pB, 2, 4);
          int8_t inB4 = (int8_t) bitext((int) *pB, 2, 6);
          pB++;
%elif config.kernel.wt_data_t == 4:
          int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
          int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
          pB++;
          int8_t inB3 = (int8_t) bitext((int) *pB, 4, 0);
          int8_t inB4 = (int8_t) bitext((int) *pB, 4, 4);
          pB++;
%elif config.kernel.wt_data_t == 8:
          int8_t inB = *pB;
          pB++;
          int8_t inB2 = *pB;
          pB++;
          int8_t inB3 = *pB;
          pB++;
          int8_t inB4 = *pB;
          pB++;
%endif
          sum += inA * inB;
          sum += inA2 * inB2;
          sum += inA3 * inB3;
          sum += inA4 * inB4;
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 4:
          ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
          ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
          pA++;
%elif config.kernel.in_data_t == 8:
          ${pt_in} inA = *pA;
          pA++;
          ${pt_in} inA2 = *pA;
          pA++;
%endif
%if config.kernel.wt_data_t == 4:
          int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
          int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
          pB++;
%elif config.kernel.wt_data_t == 8:
          int8_t inB = *pB;
          pB++;
          int8_t inB2 = *pB;
          pB++;
%endif
          sum += inA * inB;
          sum += inA2 * inB2;
%elif config.less_precision == 8:
          ${pt_in} inA = *pA;
          pA++;
          int8_t inB = *pB;
          pB++;
          sum += inA * inB;
%endif
          col_cnt--;
        }
        if (flag_batch_norm && flag_relu)
        {
          *pOutBuffer = ${config.bn_fn}(sum, *pKappa, *pLambda, out_shift);
          pOutBuffer++;
          pKappa++;
          pLambda++;
        }
        else
        {
          if (flag_relu == 1)
          {
            *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
            pOutBuffer++;
          }
          else
          {
            *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            pOutBuffer++;
          }
        }
    }
%endif
    pi_cl_team_barrier(0);
}
