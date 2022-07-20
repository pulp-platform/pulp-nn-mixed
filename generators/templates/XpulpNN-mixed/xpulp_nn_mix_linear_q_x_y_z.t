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
macload_fn = f"MacLoad{s_(config.kernel.in_signed)}{int(32/config.max_precision)}"
out_clip_fn = f"clip{s_(config.kernel.out_signed)}{config.kernel.out_data_t}"
bex = f"bitext{u_(config.kernel.in_signed)}"
%>


void __attribute__((noinline)) ${config.fn_name}(
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
  uint16_t dim_vec_in = PACK_INT${config.kernel.in_data_t}_SIZE(dim_vec);
  uint16_t dim_vec_wt = PACK_INT${config.kernel.wt_data_t}_SIZE(dim_vec);

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

  ${vt_in} vecB[${int(8/config.less_precision)}];

%if config.kernel.out_data_t == 8:
  ${pt_out} *pOutBuffer = (${pt_out} *) pOut + start;
%elif config.kernel.out_data_t == 4:
  ${pt_out} *pOutBuffer = (${pt_out} *) pOut + (start >> 1);
%elif config.kernel.out_data_t == 2:
  ${pt_out} *pOutBuffer = (${pt_out} *) pOut + (start >> 2);
%endif

  int i;

  ${act_t} *k1 = pKappa + start;
  ${act_t} *lambda1 = pLambda + start;

  int32_t a_tollback = 4;
%if config.kernel.out_data_t == 8:
  int32_t w_rollback = 4;
%elif config.kernel.out_data_t == 4:
  int32_t w_rollback = 4 - dim_vec_wt;
%elif config.kernel.out_data_t == 2:
  int32_t w_rollback = 4 - (3 * dim_vec_wt);
%endif

  A_STRIDE(0);
%if config.kernel.out_data_t != 8:
  W_STRIDE(dim_vec_wt);
%else:
  W_STRIDE(0);
%endif
  A_ROLLBACK(a_tollback);
  W_ROLLBACK(w_rollback);

%if config.kernel.out_data_t < 8:
  for(i=start; i<stop; i+=${int(8/config.kernel.out_data_t)})
%else:
  for(i=start; i<stop; i++)
%endif
  {
    int sum = 0;
%if config.kernel.out_data_t < 8:
    int sum2 = 0;
%if config.kernel.out_data_t == 2:
    int sum3 = 0;
    int sum4 = 0;
%endif
%endif

    if (pBias != NULL)
    {
      sum = ((int) (pBias[i]));
%if config.kernel.out_data_t < 8:
      sum2 = (pBias[i + 1]);
%if config.kernel.out_data_t == 2:
      sum3 = (pBias[i + 2]);
      sum4 = (pBias[i + 3]);
%endif
%endif
    }

    ${pt_in} *pB = pIn;
    int8_t *pA = pWeight + (i * dim_vec_wt);
%if config.kernel.out_data_t < 8:
    int8_t *pA2 = pA + dim_vec_wt;
%if config.kernel.out_data_t == 2:
    int8_t *pA3 = pA2 + dim_vec_wt;
    int8_t *pA4 = pA3 + dim_vec_wt;
%endif
%endif
    
    int32_t *ptrA  = (int32_t *) pA ;

%if config.kernel.in_data_t < config.kernel.wt_data_t:
    pB  = ${config.unpack_in_fn}(pB , vecB);

    ${int_t_in} *startB;

    asm volatile("mv %0, %1":"=r"(startB):"r"(vecB));

    ${int_t_in} *ptrB  = (${int_t_in} *) vecB;
%else:
    ${int_t_in} *ptrB  = (${int_t_in} *) pB ;
%endif

    W_ADDRESS(ptrA);
    A_ADDRESS(ptrB);

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
%if config.kernel.out_data_t < 8:
    ptrA  = MacLoadInit(1, 0, 1, 0, ptrA);
%if config.kernel.out_data_t == 2:
    ptrA  = MacLoadInit(1, 0, 2, 0, ptrA);
    ptrA  = MacLoadInit(1, 0, 3, 0, ptrA);
%endif
%endif

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

<%! import math %>
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    for(int j=0; j < (dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.in_data_t/config.kernel.wt_data_t))))}); j++)
%elif config.kernel.in_data_t <= config.kernel.wt_data_t:
    for(int j=0; j < (dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}); j++)
%endif
    {
%if config.kernel.in_data_t < config.kernel.wt_data_t:
      sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t < 8:
      sum2 = ${macload_fn}(1, 0, 1, 0, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t == 2:
      sum3 = ${macload_fn}(1, 0, 2, 0, ptrA, sum3);
      ptrA = MacLoadUpdate(ptrA);
      sum4 = ${macload_fn}(1, 0, 3, 0, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);
%endif
%endif

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

%if (int(config.kernel.wt_data_t/config.kernel.in_data_t) == 4):
      sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t < 8:
      sum2 = ${macload_fn}(1, 0, 1, 0, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t == 2:
      sum3 = ${macload_fn}(1, 0, 2, 0, ptrA, sum3);
      ptrA = MacLoadUpdate(ptrA);
      sum4 = ${macload_fn}(1, 0, 3, 0, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);
%endif
%endif

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

      sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t < 8:
      sum2 = ${macload_fn}(1, 0, 1, 0, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t == 2:
      sum3 = ${macload_fn}(1, 0, 2, 0, ptrA, sum3);
      ptrA = MacLoadUpdate(ptrA);
      sum4 = ${macload_fn}(1, 0, 3, 0, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);
%endif
%endif

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
%endif

      pB  = ${config.unpack_in_fn}(pB , vecB);

      ptrB   = MacLoadAssign(startB);
      A_ADDRESS(ptrB);

      sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t < 8:
      sum2 = ${macload_fn}(1, 0, 1, 0, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t == 2:
      sum3 = ${macload_fn}(1, 0, 2, 0, ptrA, sum3);
      ptrA = MacLoadUpdate(ptrA);
      sum4 = ${macload_fn}(1, 0, 3, 0, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);
%endif
%endif

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
%elif config.kernel.wt_data_t < config.kernel.in_data_t:
%if (int(config.kernel.in_data_t/config.kernel.wt_data_t) == 4):
%if config.kernel.out_data_t == 8:
      sum  = ${macload_fn}(0, 1, 0, 0, ptrB, sum);      
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.out_data_t < 8:
      sum  = ${macload_fn}(0, 0, 0, 0, ptrA, sum);
%if config.kernel.out_data_t == 4:
      sum2 = ${macload_fn}(0, 1, 1, 0, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.out_data_t == 2:
      sum2 = ${macload_fn}(0, 0, 1, 0, ptrA, sum2);
      sum3 = ${macload_fn}(0, 0, 2, 0, ptrA, sum3);
      sum4 = ${macload_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);
%endif
%endif

      MemoryFence();

%if config.kernel.out_data_t == 8:
      sum  = ${macload_fn}(0, 1, 0, 0, ptrB, sum);      
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.out_data_t < 8:
      sum  = ${macload_fn}(0, 0, 0, 0, ptrA, sum);
%if config.kernel.out_data_t == 4:
      sum2 = ${macload_fn}(0, 1, 1, 0, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.out_data_t == 2:
      sum2 = ${macload_fn}(0, 0, 1, 0, ptrA, sum2);
      sum3 = ${macload_fn}(0, 0, 2, 0, ptrA, sum3);
      sum4 = ${macload_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);
%endif
%endif
      
      MemoryFence();

%if config.kernel.out_data_t == 8:
      sum  = ${macload_fn}(0, 1, 0, 0, ptrB, sum);      
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.out_data_t < 8:
      sum  = ${macload_fn}(0, 0, 0, 0, ptrA, sum);
%if config.kernel.out_data_t == 4:
      sum2 = ${macload_fn}(0, 1, 1, 0, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.out_data_t == 2:
      sum2 = ${macload_fn}(0, 0, 1, 0, ptrA, sum2);
      sum3 = ${macload_fn}(0, 0, 2, 0, ptrA, sum3);
      sum4 = ${macload_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);
%endif
%endif

      MemoryFence();
%elif (int(config.kernel.in_data_t/config.kernel.wt_data_t) == 2):
%if config.kernel.out_data_t == 8:
      sum  = ${macload_fn}(0, 1, 0, 0, ptrB, sum);      
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.out_data_t < 8:
      sum  = ${macload_fn}(0, 0, 0, 0, ptrA, sum);
%if config.kernel.out_data_t == 4:
      sum2 = ${macload_fn}(0, 1, 1, 0, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);
%elif config.kernel.out_data_t == 2:
      sum2 = ${macload_fn}(0, 0, 1, 0, ptrA, sum2);
      sum3 = ${macload_fn}(0, 0, 2, 0, ptrA, sum3);
      sum4 = ${macload_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);
%endif
%endif

      MemoryFence();
%endif

      sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t < 8:
      sum2 = ${macload_fn}(1, 0, 1, 0, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t == 2:
      sum3 = ${macload_fn}(1, 0, 2, 0, ptrA, sum3);
      ptrA = MacLoadUpdate(ptrA);
      sum4 = ${macload_fn}(1, 0, 3, 0, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);
%endif
%endif
    
      ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
%else:
      sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t < 8:
      sum2 = ${macload_fn}(1, 0, 1, 0, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t == 2:
      sum3 = ${macload_fn}(1, 0, 2, 0, ptrA, sum3);
      ptrA = MacLoadUpdate(ptrA);
      sum4 = ${macload_fn}(1, 0, 3, 0, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);
%endif
%endif
    
      ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
%endif
    }
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    uint16_t col_cnt = dim_vec & ${hex(((int(32/config.max_precision))*(int(config.kernel.in_data_t/config.kernel.wt_data_t)))-1)};
%elif config.kernel.in_data_t <= config.kernel.wt_data_t:
    uint16_t col_cnt = dim_vec & ${hex(((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t)))-1)};
%endif
    if(col_cnt)
    {
%if config.kernel.wt_data_t < config.kernel.in_data_t:
      pA-=4;
%if config.kernel.out_data_t < 8:
      pA2-=4;
%if config.kernel.out_data_t == 2:
      pA3-=4;
      pA4-=4;
%endif
%endif
%else:
      pA=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.wt_data_t/config.kernel.in_data_t))))});
%if config.kernel.out_data_t < 8:
      pA2+=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.wt_data_t/config.kernel.in_data_t))))});
%if config.kernel.out_data_t == 2:
      pA3+=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.wt_data_t/config.kernel.in_data_t))))});
      pA4+=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.wt_data_t/config.kernel.in_data_t))))});
%endif
%endif
%endif
%if config.kernel.in_data_t < config.kernel.wt_data_t:
      pB-=4;
%else:
      pB=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.in_data_t/config.kernel.wt_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.in_data_t/config.kernel.wt_data_t))))});
%endif
      do
      {
%if config.less_precision == 2:
%if config.kernel.in_data_t == 2:
        ${pt_in} inB =  (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 0);
        ${pt_in} inB2 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 2);
        ${pt_in} inB3 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 4);
        ${pt_in} inB4 = (${pt_in}) ${bex}((${int_t_in}) *pB, 2, 6);
        pB++;
%elif config.kernel.in_data_t == 4:
        ${pt_in} inB =  (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_in} inB2 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 4);
        pB++;
        ${pt_in} inB3 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_in} inB4 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 4);
        pB++;
%elif config.kernel.in_data_t == 8:
        ${pt_in} inB = *pB;
        pB++;
        ${pt_in} inB2 = *pB;
        pB++;
        ${pt_in} inB3 = *pB;
        pB++;
        ${pt_in} inB4 = *pB;
        pB++;
%endif
%if config.kernel.wt_data_t == 2:
        int8_t inA =  (int8_t) bitext((int) *pA, 2, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA, 2, 2);
        int8_t inA3 = (int8_t) bitext((int) *pA, 2, 4);
        int8_t inA4 = (int8_t) bitext((int) *pA, 2, 6);
        pA++;
%elif config.kernel.wt_data_t == 4:
        int8_t inA =  (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA, 4, 4);
        pA++;
        int8_t inA3 = (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA, 4, 4);
        pA++;
%elif config.kernel.wt_data_t == 8:
        int8_t inA = *pA;
        pA++;
        int8_t inA2 = *pA;
        pA++;
        int8_t inA3 = *pA;
        pA++;
        int8_t inA4 = *pA;
        pA++;
%endif
        sum += inA * inB;
        sum += inA2 * inB2;
        sum += inA3 * inB3;
        sum += inA4 * inB4;
%if config.kernel.out_data_t < 8:
%if config.kernel.wt_data_t == 2:
        inA =  (int8_t) bitext((int) *pA2, 2, 0);
        inA2 = (int8_t) bitext((int) *pA2, 2, 2);
        inA3 = (int8_t) bitext((int) *pA2, 2, 4);
        inA4 = (int8_t) bitext((int) *pA2, 2, 6);
        pA2++;
%elif config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA2, 4, 0);
        inA2 = (int8_t) bitext((int) *pA2, 4, 4);
        pA2++;
        inA3 = (int8_t) bitext((int) *pA2, 4, 0);
        inA4 = (int8_t) bitext((int) *pA2, 4, 4);
        pA2++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA2;
        pA2++;
        inA2 = *pA2;
        pA2++;
        inA3 = *pA2;
        pA2++;
        inA4 = *pA2;
        pA2++;
%endif
        sum2 += inA * inB;
        sum2 += inA2 * inB2;
        sum2 += inA3 * inB3;
        sum2 += inA4 * inB4;
%if config.kernel.out_data_t == 2:
%if config.kernel.wt_data_t == 2:
        inA =  (int8_t) bitext((int) *pA3, 2, 0);
        inA2 = (int8_t) bitext((int) *pA3, 2, 2);
        inA3 = (int8_t) bitext((int) *pA3, 2, 4);
        inA4 = (int8_t) bitext((int) *pA3, 2, 6);
        pA3++;
%elif config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA3, 4, 0);
        inA2 = (int8_t) bitext((int) *pA3, 4, 4);
        pA3++;
        inA3 = (int8_t) bitext((int) *pA3, 4, 0);
        inA4 = (int8_t) bitext((int) *pA3, 4, 4);
        pA3++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA3;
        pA3++;
        inA2 = *pA3;
        pA3++;
        inA3 = *pA3;
        pA3++;
        inA4 = *pA3;
        pA3++;
%endif
        sum3 += inA * inB;
        sum3 += inA2 * inB2;
        sum3 += inA3 * inB3;
        sum3 += inA4 * inB4;
%if config.kernel.wt_data_t == 2:
        inA =  (int8_t) bitext((int) *pA4, 2, 0);
        inA2 = (int8_t) bitext((int) *pA4, 2, 2);
        inA3 = (int8_t) bitext((int) *pA4, 2, 4);
        inA4 = (int8_t) bitext((int) *pA4, 2, 6);
        pA4++;
%elif config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA4, 4, 0);
        inA2 = (int8_t) bitext((int) *pA4, 4, 4);
        pA4++;
        inA3 = (int8_t) bitext((int) *pA4, 4, 0);
        inA4 = (int8_t) bitext((int) *pA4, 4, 4);
        pA4++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA4;
        pA4++;
        inA2 = *pA4;
        pA4++;
        inA3 = *pA4;
        pA4++;
        inA4 = *pA4;
        pA4++;
%endif
        sum4 += inA * inB;
        sum4 += inA2 * inB2;
        sum4 += inA3 * inB3;
        sum4 += inA4 * inB4;
%endif
%endif
        col_cnt-=4;
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 4:
        ${pt_in} inB  = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_in} inB2 = (${pt_in}) ${bex}((${int_t_in}) *pB, 4, 4);
        pB++;
%elif config.kernel.in_data_t == 8:
        ${pt_in} inB = *pB;
        pB++;
        ${pt_in} inB2 = *pB;
        pB++;
%endif
%if config.kernel.wt_data_t == 4:
        int8_t inA  = (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA, 4, 4);
        pA++;
%elif config.kernel.wt_data_t == 8:
        int8_t inA = *pA;
        pA++;
        int8_t inA2 = *pA;
        pA++;
%endif
        sum += inA * inB;
        sum += inA2 * inB2;
%if config.kernel.out_data_t < 8:
%if config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA2, 4, 0);
        inA2 = (int8_t) bitext((int) *pA2, 4, 4);
        pA2++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA2;
        pA2++;
        inA2 = *pA2;
        pA2++;
%endif
        sum2 += inA * inB;
        sum2 += inA2 * inB2;
%if config.kernel.out_data_t == 2:
%if config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA3, 4, 0);
        inA2 = (int8_t) bitext((int) *pA3, 4, 4);
        pA3++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA3;
        pA3++;
        inA2 = *pA3;
        pA3++;
%endif
        sum3 += inA * inB;
        sum3 += inA2 * inB2;
%if config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA4, 4, 0);
        inA2 = (int8_t) bitext((int) *pA4, 4, 4);
        pA4++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA4;
        pA4++;
        inA2 = *pA4;
        pA4++;
%endif
        sum4 += inA * inB;
        sum4 += inA2 * inB2;
%endif
%endif
        col_cnt-=2;
%elif config.less_precision == 8:
        ${pt_in} inB = *pB;
        pB++;
        int8_t inA = *pA;
        pA++;
        sum += inA * inB;
%if config.kernel.out_data_t < 8:
        inA = *pA2;
        pA2++;

        sum2 += inA * inB;
%if config.kernel.out_data_t == 2:
        inA = *pA3;
        pA3++;

        sum3 += inA * inB;

        inA = *pA4;
        pA4++;

        sum4 += inA * inB;
%endif
%endif
        col_cnt--;
%endif
      }while (col_cnt);
    }
    if (flag_batch_norm && flag_relu)
    {
%if config.kernel.out_data_t == 8:
      *pOutBuffer = ${config.bn_fn}(sum, *k1++, *lambda1++, out_shift);
      pOutBuffer++;
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
%elif config.kernel.out_data_t == 4:
        sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
        sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
        *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
        pOutBuffer++;
%elif config.kernel.out_data_t == 2:
        sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
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
  pi_cl_team_barrier(0);
}
