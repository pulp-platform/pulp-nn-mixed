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
int_t_in = f"{u_(config.kernel.in_signed)}int32_t"
pt_out = f"{u_(config.kernel.out_signed)}int8_t"
macload_fn = f"MacLoad{s_(config.kernel.in_signed)}{int(32/config.max_precision)}"
bex = f"bitext{u_(config.kernel.in_signed)}"
%>

void __attribute__((noinline)) ${config.fn_name}(
                  ${pt_in} *pIn,
                  int8_t *pBias,
                  ${pt_out} *pOut,
                  int8_t *pWeight,
                  uint16_t dim_vec,
                  uint16_t num_o_neurons)
{
  uint16_t dim_vec_in = PACK_INT${config.kernel.in_data_t}_SIZE(dim_vec);
  uint16_t dim_vec_wt = PACK_INT${config.kernel.wt_data_t}_SIZE(dim_vec);

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
  int start = min(chunk * core_id, num_o_neurons);
  int stop = min(start + chunk, num_o_neurons);

  int32_t *pOutBuffer = (int32_t *) pOut + start;

%if config.kernel.wt_data_t < config.kernel.in_data_t:
  int32_t vecA[${int(config.max_precision/config.kernel.wt_data_t)}];
%endif
%if config.kernel.in_data_t <= config.kernel.wt_data_t:
  ${int_t_in} vecB[${int(config.max_precision/config.kernel.in_data_t)}];
%endif

  for(int i=start; i<stop; i++)
  {
    int sum = 0;

    if (pBias != NULL)
    {
      sum = *(int32_t *)(pBias + 4*i);
    }

    int8_t *pA = pWeight + (i * dim_vec_wt);

    ${pt_in} *pB = pIn;

%if config.kernel.in_data_t != config.kernel.wt_data_t:
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    pA  = ${config.unpack_wt_fn}(pA , vecA);

    int32_t *startA;

    asm volatile("mv %0, %1":"=r"(startA):"r"(vecA));

    int32_t *ptrA  = (int32_t *) vecA ;
%else:
    int32_t *ptrA  = (int32_t *) pA ;
%endif
%if config.kernel.in_data_t < config.kernel.wt_data_t:
    pB  = ${config.unpack_in_fn}(pB , vecB);

    ${int_t_in} *startB;

    asm volatile("mv %0, %1":"=r"(startB):"r"(vecB));

    ${int_t_in} *ptrB  = vecB;
%else:
    ${int_t_in} *ptrB  = pB ;
%endif
%else:
    int32_t *ptrA  = (int32_t *) pA ;

    ${int_t_in} *ptrB  = pB ;
%endif

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

<%! import math %>
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    for(int j=0; j < (dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.in_data_t/config.kernel.wt_data_t))))}); j++)
%elif config.kernel.in_data_t <= config.kernel.wt_data_t:
    for(int j=0; j < (dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}); j++)
%endif
    {
      sum = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

%if config.kernel.in_data_t != config.kernel.wt_data_t:
%if (int(config.kernel.in_data_t/config.kernel.wt_data_t) == 4) or (int(config.kernel.wt_data_t/config.kernel.in_data_t) == 4):
      sum = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

      sum = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
%endif
%if config.kernel.wt_data_t < config.kernel.in_data_t:
      pA  = ${config.unpack_wt_fn}(pA , vecA);

      ptrA   = MacLoadAssign(startA);
%endif
%if config.kernel.in_data_t < config.kernel.wt_data_t:
      pB  = ${config.unpack_in_fn}(pB , vecB);

      ptrB   = MacLoadAssign(startB);
%endif
      sum = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
%else:
      //ensure enough instructions in the HW loop - otherwise it will work on GVSOC but not in real hardware!
      asm volatile("nop;");
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
%else:
      pA=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.wt_data_t/config.kernel.in_data_t))))});
%endif
%if config.kernel.in_data_t < config.kernel.wt_data_t:
      pB-=4;
%else:
      pB=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.in_data_t/config.kernel.wt_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.in_data_t/config.kernel.wt_data_t))))});
%endif
      do
      {
%if config.less_precision == 2:
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
        sum += inA * inB;
        sum += inA2 * inB2;
        sum += inA3 * inB3;
        sum += inA4 * inB4;
        col_cnt-=4;
%elif config.less_precision == 4:
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
        sum += inA * inB;
        sum += inA2 * inB2;
        col_cnt-=2;
%elif config.less_precision == 8:
        int8_t inA = *pA;
        pA++;
        ${pt_in} inB = *pB;
        pB++;
        sum += inA * inB;
        col_cnt--;
%endif
      }while (col_cnt);
    }
    *pOutBuffer = sum;
    pOutBuffer++;
  }
  pi_cl_team_barrier(0);
}
