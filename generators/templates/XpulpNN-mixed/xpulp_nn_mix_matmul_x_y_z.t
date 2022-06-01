/*
 * ${config.filename}
 * Nazareno   Bruschi  <nazareno.bruschi@unibo.it>
 * Alessandro Nadalini <alessandro.nadalini3@unibo.it>
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

  uint16_t ch_out_r = PACK_INT${config.kernel.out_data_t}_SIZE(ch_out);

  uint16_t num_col_im2col_w = PACK_INT${config.kernel.wt_data_t}_SIZE(num_col_im2col);
  uint16_t num_col_im2col_a = PACK_INT${config.max_precision}_SIZE(num_col_im2col);

%if config.kernel.matmul_fmt == '4x2': 
  int32_t a_rollback = 4 - num_col_im2col_a;
%elif config.kernel.matmul_fmt == '4x4':
  int32_t a_rollback = 4 - (num_col_im2col_a + (num_col_im2col_a << 1));
%endif
  int32_t w_rollback = 4 - (num_col_im2col_w + (num_col_im2col_w << 1));

  LEGACY_MODE("0");
%if config.kernel.wt_data_t == 8 and config.kernel.wt_data_t == 8:
  IVEC_FMT("2");
%elif config.kernel.wt_data_t == 4 and config.kernel.in_data_t == 4:
  IVEC_FMT("3");
%elif config.kernel.wt_data_t == 2 and config.kernel.in_data_t == 2:
  IVEC_FMT("4");
%elif config.kernel.wt_data_t == 2 and config.kernel.in_data_t == 4:
  IVEC_FMT("5");
%elif config.kernel.wt_data_t == 2 and config.kernel.in_data_t == 8:
  IVEC_FMT("6");
%elif config.kernel.wt_data_t == 4 and config.kernel.in_data_t == 8:
  IVEC_FMT("8");
%elif config.kernel.in_data_t < config.kernel.wt_data_t:
%if config.max_precision == 8:
  IVEC_FMT("2");
%elif config.max_precision == 4:
  IVEC_FMT("3");
%endif
%endif
  A_STRIDE(num_col_im2col_a);
  W_STRIDE(num_col_im2col_w);
  A_ROLLBACK(a_rollback);
  W_ROLLBACK(w_rollback);
%if config.kernel.matmul_fmt == '4x2':
  A_SKIP("1");
  W_SKIP("3");
  MIXED_SKIP("8");
%elif config.kernel.matmul_fmt == '4x4':
  A_SKIP("3");
  W_SKIP("3");
  MIXED_SKIP("16");
%endif

  int8_t *pA = pWeight;

  uint16_t chan_left = ch_out & 0x3;

  for(int i=0; i < (ch_out >> 2); i++)
  {
    ${pt_in} *pB =  pIn;

    ${int_t_in} *ptrB  = (${int_t_in} *) pB;

    int32_t *ptrA  = (int32_t *) pA ;

    A_ADDRESS(ptrB);
    W_ADDRESS(ptrA);

    ptrA = MacLoadInit(1, 0, 0, 0, ptrA);
    ptrA = MacLoadInit(1, 0, 1, 0, ptrA);
    ptrA = MacLoadInit(1, 0, 2, 0, ptrA);
    ptrA = MacLoadInit(1, 0, 3, 0, ptrA);

    ptrB = MacLoadInit(0, 1, 0, 0, ptrB);

    int sum  = 0;
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
      sum = ((int) (*pBias++));
      sum2 = ((int) (*pBias++));
      sum3 = ((int) (*pBias++));
      sum4 = ((int) (*pBias++));

      sum5 = sum;
      sum6 = sum2;
      sum7 = sum3;
      sum8 = sum4;

%if config.kernel.matmul_fmt == '4x4':
      sum9 = sum;
      sum10 = sum2;
      sum11 = sum3;
      sum12 = sum4;

      sum13 = sum;
      sum14 = sum2;
      sum15 = sum3;
      sum16 = sum4;
%endif
    }
<%! import math %>
    for(int j=0; j<(num_col_im2col >> ${int(math.log2(((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t)))))}); j++)
    {
%if config.kernel.in_data_t <= config.kernel.wt_data_t:
      ptrB = MacLoadInit(0, 1, 0, 1, ptrB);

      sum  = ${mac_fn}(0, 0, 0, 0, ptrA, sum);
      sum2 = ${mac_fn}(0, 0, 1, 0, ptrA, sum2);
      sum3 = ${mac_fn}(0, 0, 2, 0, ptrA, sum3);
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x2':
      sum5 = ${mac_fn}(1, 0, 0, 1, ptrA, sum5);
      ptrA = MacLoadUpdate(ptrA);

      sum6 = ${mac_fn}(1, 0, 1, 1, ptrA, sum6);
      ptrA = MacLoadUpdate(ptrA);

      sum7 = ${mac_fn}(1, 0, 2, 1, ptrA, sum7);
      ptrA = MacLoadUpdate(ptrA);

      sum8 = ${mac_fn}(1, 0, 3, 1, ptrA, sum8);
      ptrA = MacLoadUpdate(ptrA);
%elif config.kernel.matmul_fmt == '4x4':
      sum5 = ${mac_fn}(0, 0, 0, 1, ptrA, sum5);
      sum6 = ${mac_fn}(0, 0, 1, 1, ptrA, sum6);
      sum7 = ${mac_fn}(0, 0, 2, 1, ptrA, sum7);
      sum8 = ${mac_fn}(0, 1, 3, 1, ptrB, sum8);
      ptrB = MacLoadUpdate(ptrB);

      sum9  = ${mac_fn}(0, 0, 0, 0, ptrA, sum9);
      sum10 = ${mac_fn}(0, 0, 1, 0, ptrA, sum10);
      sum11 = ${mac_fn}(0, 0, 2, 0, ptrA, sum11);
      sum12 = ${mac_fn}(0, 1, 3, 0, ptrB, sum12);
      ptrB  = MacLoadUpdate(ptrB);

      sum13 = ${mac_fn}(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);

      sum14 = ${mac_fn}(1, 0, 1, 1, ptrA, sum14);
      ptrA  = MacLoadUpdate(ptrA);

      sum15 = ${mac_fn}(1, 0, 2, 1, ptrA, sum15);
      ptrA  = MacLoadUpdate(ptrA);

      sum16 = ${mac_fn}(1, 0, 3, 1, ptrA, sum16);
      ptrA  = MacLoadUpdate(ptrA);
%endif

%elif config.kernel.wt_data_t < config.kernel.in_data_t:
%if (config.kernel.in_data_t/config.kernel.wt_data_t) == 4:
      ptrB = MacLoadInit(0, 1, 0, 1, ptrB);

      sum  = ${mac_fn}(0, 0, 0, 0, ptrA, sum);
      sum2 = ${mac_fn}(0, 0, 1, 0, ptrA, sum2);
      sum3 = ${mac_fn}(0, 0, 2, 0, ptrA, sum3);
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);

      sum5 = ${mac_fn}(0, 0, 0, 1, ptrA, sum5);
      sum6 = ${mac_fn}(0, 0, 1, 1, ptrA, sum6);
      sum7 = ${mac_fn}(0, 0, 2, 1, ptrA, sum7);
      sum8 = ${mac_fn}(0, 1, 3, 1, ptrB, sum8);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x4':
      sum9  = ${mac_fn}(0, 0, 0, 0, ptrA, sum9);
      sum10 = ${mac_fn}(0, 0, 1, 0, ptrA, sum10);
      sum11 = ${mac_fn}(0, 0, 2, 0, ptrA, sum11);
      sum12 = ${mac_fn}(0, 1, 3, 0, ptrB, sum12);
      ptrB  = MacLoadUpdate(ptrB);

      sum13 = ${mac_fn}(0, 0, 0, 1, ptrA, sum13);
      sum14 = ${mac_fn}(0, 0, 1, 1, ptrA, sum14);
      sum15 = ${mac_fn}(0, 0, 2, 1, ptrA, sum15);
      sum16 = ${mac_fn}(0, 1, 3, 1, ptrB, sum16);
      ptrB  = MacLoadUpdate(ptrB);
%endif

      MemoryFence();

      sum  = ${mac_fn}(0, 0, 0, 0, ptrA, sum);
      sum2 = ${mac_fn}(0, 0, 1, 0, ptrA, sum2);
      sum3 = ${mac_fn}(0, 0, 2, 0, ptrA, sum3);
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);

      sum5 = ${mac_fn}(0, 0, 0, 1, ptrA, sum5);
      sum6 = ${mac_fn}(0, 0, 1, 1, ptrA, sum6);
      sum7 = ${mac_fn}(0, 0, 2, 1, ptrA, sum7);
      sum8 = ${mac_fn}(0, 1, 3, 1, ptrB, sum8);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x4':
      sum9  = ${mac_fn}(0, 0, 0, 0, ptrA, sum9);
      sum10 = ${mac_fn}(0, 0, 1, 0, ptrA, sum10);
      sum11 = ${mac_fn}(0, 0, 2, 0, ptrA, sum11);
      sum12 = ${mac_fn}(0, 1, 3, 0, ptrB, sum12);
      ptrB  = MacLoadUpdate(ptrB);

      sum13 = ${mac_fn}(0, 0, 0, 1, ptrA, sum13);
      sum14 = ${mac_fn}(0, 0, 1, 1, ptrA, sum14);
      sum15 = ${mac_fn}(0, 0, 2, 1, ptrA, sum15);
      sum16 = ${mac_fn}(0, 1, 3, 1, ptrB, sum16);
      ptrB  = MacLoadUpdate(ptrB);
%endif

      MemoryFence();

      sum  = ${mac_fn}(0, 0, 0, 0, ptrA, sum);
      sum2 = ${mac_fn}(0, 0, 1, 0, ptrA, sum2);
      sum3 = ${mac_fn}(0, 0, 2, 0, ptrA, sum3);
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);

      sum5 = ${mac_fn}(0, 0, 0, 1, ptrA,  sum5);
      sum6 = ${mac_fn}(0, 0, 1, 1, ptrA, sum6);
      sum7 = ${mac_fn}(0, 0, 2, 1, ptrA, sum7);
      sum8 = ${mac_fn}(0, 1, 3, 1, ptrB, sum8);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x4':
      sum9  = ${mac_fn}(0, 0, 0, 0, ptrA, sum9);
      sum10 = ${mac_fn}(0, 0, 1, 0, ptrA, sum10);
      sum11 = ${mac_fn}(0, 0, 2, 0, ptrA, sum11);
      sum12 = ${mac_fn}(0, 1, 3, 0, ptrB, sum12);
      ptrB  = MacLoadUpdate(ptrB);

      sum13 = ${mac_fn}(0, 0, 0, 1, ptrA, sum13);
      sum14 = ${mac_fn}(0, 0, 1, 1, ptrA, sum14);
      sum15 = ${mac_fn}(0, 0, 2, 1, ptrA, sum15);
      sum16 = ${mac_fn}(0, 1, 3, 1, ptrB, sum16);
      ptrB  = MacLoadUpdate(ptrB);
%endif

      MemoryFence();

      sum  = ${mac_fn}(0, 0, 0, 0, ptrA, sum);
      sum2 = ${mac_fn}(0, 0, 1, 0, ptrA, sum2);
      sum3 = ${mac_fn}(0, 0, 2, 0, ptrA, sum3);
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x2':
      sum5 = ${mac_fn}(1, 0, 0, 1, ptrA, sum5);
      ptrA = MacLoadUpdate(ptrA);

      sum6 = ${mac_fn}(1, 0, 1, 1, ptrA, sum6);
      ptrA = MacLoadUpdate(ptrA);

      sum7 = ${mac_fn}(1, 0, 2, 1, ptrA, sum7);
      ptrA = MacLoadUpdate(ptrA);

      sum8 = ${mac_fn}(1, 0, 3, 1, ptrA, sum8);
      ptrA = MacLoadUpdate(ptrA);
%elif config.kernel.matmul_fmt == '4x4':
      sum5 = ${mac_fn}(0, 0, 0, 1, ptrA, sum5);
      sum6 = ${mac_fn}(0, 0, 1, 1, ptrA, sum6);
      sum7 = ${mac_fn}(0, 0, 2, 1, ptrA, sum7);
      sum8 = ${mac_fn}(0, 1, 3, 1, ptrB, sum8);
      ptrB = MacLoadUpdate(ptrB);

      sum9  = ${mac_fn}(0, 0, 0, 0, ptrA, sum9);
      sum10 = ${mac_fn}(0, 0, 1, 0, ptrA, sum10);
      sum11 = ${mac_fn}(0, 0, 2, 0, ptrA, sum11);
      sum12 = ${mac_fn}(0, 1, 3, 0, ptrB, sum12);
      ptrB  = MacLoadUpdate(ptrB);

      sum13 = ${mac_fn}(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);

      sum14 = ${mac_fn}(1, 0, 1, 1, ptrA, sum14);
      ptrA  = MacLoadUpdate(ptrA);

      sum15 = ${mac_fn}(1, 0, 2, 1, ptrA, sum15);
      ptrA  = MacLoadUpdate(ptrA);

      sum16 = ${mac_fn}(1, 0, 3, 1, ptrA, sum16);
      ptrA  = MacLoadUpdate(ptrA);
%endif

%elif (config.kernel.in_data_t/config.kernel.wt_data_t) == 2:
      ptrB  = MacLoadInit(0, 1, 0, 1, ptrB);

      sum  = ${mac_fn}(0, 0, 0, 0, ptrA, sum);
      sum2 = ${mac_fn}(0, 0, 1, 0, ptrA, sum2);
      sum3 = ${mac_fn}(0, 0, 2, 0, ptrA, sum3);
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);

      sum5 = ${mac_fn}(0, 0, 0, 1, ptrA,  sum5);
      sum6 = ${mac_fn}(0, 0, 1, 1, ptrA, sum6);
      sum7 = ${mac_fn}(0, 0, 2, 1, ptrA, sum7);
      sum8 = ${mac_fn}(0, 1, 3, 1, ptrB, sum8);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x4':
      sum9  = ${mac_fn}(0, 0, 0, 0, ptrA, sum9);
      sum10 = ${mac_fn}(0, 0, 1, 0, ptrA, sum10);
      sum11 = ${mac_fn}(0, 0, 2, 0, ptrA, sum11);
      sum12 = ${mac_fn}(0, 1, 3, 0, ptrB, sum12);
      ptrB  = MacLoadUpdate(ptrB);

      sum13 = ${mac_fn}(0, 0, 0, 1, ptrA, sum13);
      sum14 = ${mac_fn}(0, 0, 1, 1, ptrA, sum14);
      sum15 = ${mac_fn}(0, 0, 2, 1, ptrA, sum15);
      sum16 = ${mac_fn}(0, 1, 3, 1, ptrB, sum16);
      ptrB  = MacLoadUpdate(ptrB);
%endif

      MemoryFence();

      sum  = ${mac_fn}(0, 0, 0, 0, ptrA, sum);
      sum2 = ${mac_fn}(0, 0, 1, 0, ptrA, sum2);
      sum3 = ${mac_fn}(0, 0, 2, 0, ptrA, sum3);
      sum4 = ${mac_fn}(0, 1, 3, 0, ptrB, sum4);      
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x2':
      sum5 = ${mac_fn}(1, 0, 0, 1, ptrA, sum5);
      ptrA = MacLoadUpdate(ptrA);

      sum6 = ${mac_fn}(1, 0, 1, 1, ptrA, sum6);
      ptrA = MacLoadUpdate(ptrA);

      sum7 = ${mac_fn}(1, 0, 2, 1, ptrA, sum7);
      ptrA = MacLoadUpdate(ptrA);

      sum8 = ${mac_fn}(1, 0, 3, 1, ptrA, sum8);
      ptrA = MacLoadUpdate(ptrA);
%elif config.kernel.matmul_fmt == '4x4':
      sum5 = ${mac_fn}(0, 0, 0, 1, ptrA, sum5);
      sum6 = ${mac_fn}(0, 0, 1, 1, ptrA, sum6);
      sum7 = ${mac_fn}(0, 0, 2, 1, ptrA, sum7);
      sum8 = ${mac_fn}(0, 1, 3, 1, ptrB, sum8);
      ptrB = MacLoadUpdate(ptrB);

      sum9  = ${mac_fn}(0, 0, 0, 0, ptrA, sum9);
      sum10 = ${mac_fn}(0, 0, 1, 0, ptrA, sum10);
      sum11 = ${mac_fn}(0, 0, 2, 0, ptrA, sum11);
      sum12 = ${mac_fn}(0, 1, 3, 0, ptrB, sum12);
      ptrB  = MacLoadUpdate(ptrB);

      sum13 = ${mac_fn}(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);

      sum14 = ${mac_fn}(1, 0, 1, 1, ptrA, sum14);
      ptrA  = MacLoadUpdate(ptrA);

      sum15 = ${mac_fn}(1, 0, 2, 1, ptrA, sum15);
      ptrA  = MacLoadUpdate(ptrA);

      sum16 = ${mac_fn}(1, 0, 3, 1, ptrA, sum16);
      ptrA  = MacLoadUpdate(ptrA);
%endif
%endif
%endif
    }

%if config.kernel.wt_data_t < config.kernel.in_data_t:    
    asm volatile ("csrr %0, 0x101" : "=r" (pA));
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
      
      int8_t *pA2 = (pA  + num_col_im2col_w);
      int8_t *pA3 = (pA2 + num_col_im2col_w);
      int8_t *pA4 = (pA3 + num_col_im2col_w);

      pB+=loop_cnt_im2col_a;
      
      ${pt_in} *pB2 = (pB + num_col_im2col_a);
%if config.kernel.matmul_fmt == '4x4':
      ${pt_in} *pB3 = (pB2 + num_col_im2col_a);
      ${pt_in} *pB4 = (pB3 + num_col_im2col_a);
%endif

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
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = *pB3++;
        ${pt_in} inB4 = *pB4++;
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 2);
        inA2 = (int8_t) bitext((int) *pA2, 2, 2);
        inA3 = (int8_t) bitext((int) *pA3, 2, 2);
        inA4 = (int8_t) bitext((int) *pA4, 2, 2);

        inB = *pB++;
        inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        inB3 = *pB3++;
        inB4 = *pB4++;
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 4);
        inA2 = (int8_t) bitext((int) *pA2, 2, 4);
        inA3 = (int8_t) bitext((int) *pA3, 2, 4);
        inA4 = (int8_t) bitext((int) *pA4, 2, 4);

        inB = *pB++;
        inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        inB3 = *pB3++;
        inB4 = *pB4++;
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 6);
        inA2 = (int8_t) bitext((int) *pA2, 2, 6);
        inA3 = (int8_t) bitext((int) *pA3, 2, 6);
        inA4 = (int8_t) bitext((int) *pA4, 2, 6);

        inB = *pB++;
        inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        inB3 = *pB3++;
        inB4 = *pB4++;
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

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
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = *pB3++;
        ${pt_in} inB4 = *pB4++;
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 4, 4);
        inA2 = (int8_t) bitext((int) *pA2, 4, 4);
        inA3 = (int8_t) bitext((int) *pA3, 4, 4);
        inA4 = (int8_t) bitext((int) *pA4, 4, 4);

        inB = *pB++;
        inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        inB3 = *pB3++;
        inB4 = *pB4++;
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

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
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = *pB3++;
        ${pt_in} inB4 = *pB4++;
%endif
        asm volatile("": : :"memory");
        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

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
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 0);
        ${pt_in} inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 0);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 2);
        inA2 = (int8_t) bitext((int) *pA2, 2, 2);
        inA3 = (int8_t) bitext((int) *pA3, 2, 2);
        inA4 = (int8_t) bitext((int) *pA4, 2, 2);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 4);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 4);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        pB++;
        pB2++;
%if config.kernel.matmul_fmt == '4x4':
        pB3++;
        pB4++;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 4);
        inA2 = (int8_t) bitext((int) *pA2, 2, 4);
        inA3 = (int8_t) bitext((int) *pA3, 2, 4);
        inA4 = (int8_t) bitext((int) *pA4, 2, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 0);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 0);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 6);
        inA2 = (int8_t) bitext((int) *pA2, 2, 6);
        inA3 = (int8_t) bitext((int) *pA3, 2, 6);
        inA4 = (int8_t) bitext((int) *pA4, 2, 6);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 4);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 4);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        pA++;
        pA2++;
        pA3++;
        pA4++;

        pB++;
        pB2++;
%if config.kernel.matmul_fmt == '4x4':
        pB3++;
        pB4++;
%endif

        col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
        int8_t inA = (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA2, 4, 0);
        int8_t inA3 = (int8_t) bitext((int) *pA3, 4, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA4, 4, 0);

        ${pt_in} inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_in} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 0);
        ${pt_in} inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 0);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 4, 4);
        inA2 = (int8_t) bitext((int) *pA2, 4, 4);
        inA3 = (int8_t) bitext((int) *pA3, 4, 4);
        inA4 = (int8_t) bitext((int) *pA4, 4, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 4);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 4);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        pA++;
        pA2++;
        pA3++;
        pA4++;

        pB++;
        pB2++;
%if config.kernel.matmul_fmt == '4x4':
        pB3++;
        pB4++;
%endif

        col_cnt_im2col-=2;
%else:
        int8_t inA = *pA++;
        int8_t inA2 = *pA2++;
        int8_t inA3 = *pA3++;
        int8_t inA4 = *pA4++;

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = *pB3++;
        ${pt_in} inB4 = *pB4++;
%endif
        asm volatile("": : :"memory");
        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

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
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 2, 0);
        ${pt_in} inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 2, 0);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 2);
        inA2 = (int8_t) bitext((int) *pA2, 2, 2);
        inA3 = (int8_t) bitext((int) *pA3, 2, 2);
        inA4 = (int8_t) bitext((int) *pA4, 2, 2);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 2);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 2);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 2, 2);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 2, 2);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 4);
        inA2 = (int8_t) bitext((int) *pA2, 2, 4);
        inA3 = (int8_t) bitext((int) *pA3, 2, 4);
        inA4 = (int8_t) bitext((int) *pA4, 2, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 4);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 2, 4);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 2, 4);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 6);
        inA2 = (int8_t) bitext((int) *pA2, 2, 6);
        inA3 = (int8_t) bitext((int) *pA3, 2, 6);
        inA4 = (int8_t) bitext((int) *pA4, 2, 6);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 6);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 6);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 2, 6);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 2, 6);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        pA++;
        pA2++;
        pA3++;
        pA4++;

        pB++;
        pB2++;
%if config.kernel.matmul_fmt == '4x4':
        pB3++;
        pB4++;
%endif

        col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
        int8_t inA = (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA2, 4, 0);
        int8_t inA3 = (int8_t) bitext((int) *pA3, 4, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA4, 4, 0);

        ${pt_in} inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_in} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 0);
        ${pt_in} inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 0);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 4, 4);
        inA2 = (int8_t) bitext((int) *pA2, 4, 4);
        inA3 = (int8_t) bitext((int) *pA3, 4, 4);
        inA4 = (int8_t) bitext((int) *pA4, 4, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 4);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 4);
%endif

        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        pA++;
        pA2++;
        pA3++;
        pA4++;

        pB++;
        pB2++;
%if config.kernel.matmul_fmt == '4x4':
        pB3++;
        pB4++;
%endif

        col_cnt_im2col-=2;
%else:
        int8_t inA = *pA++;
        int8_t inA2 = *pA2++;
        int8_t inA3 = *pA3++;
        int8_t inA4 = *pA4++;

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = *pB3++;
        ${pt_in} inB4 = *pB4++;
%endif
        asm volatile("": : :"memory");
        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;
%endif

        col_cnt_im2col--;
%endif
%endif
      } while(col_cnt_im2col);
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
      sum   = ${config.bn_fn}(sum, *pKappa, *pLambda, out_shift);
      sum5  = ${config.bn_fn}(sum5, *pKappa, *pLambda, out_shift);
%if config.kernel.matmul_fmt == '4x4':
      sum9  = ${config.bn_fn}(sum9, *pKappa, *pLambda, out_shift);
      sum13 = ${config.bn_fn}(sum13, *pKappa, *pLambda, out_shift);
%endif
      pKappa++;
      pLambda++;
      sum2  = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
      sum6  = ${config.bn_fn}(sum6, *pKappa, *pLambda, out_shift);
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
%if config.kernel.matmul_fmt=='4x4':
      pOut3++;
      pOut4++;
%endif
%elif config.kernel.out_data_t == 2:
      sum = ${config.bn_fn}(sum, *pKappa, *pLambda, out_shift);
      sum5 = ${config.bn_fn}(sum5, *pKappa, *pLambda, out_shift);
%if config.kernel.matmul_fmt == '4x4':
      sum9  = ${config.bn_fn}(sum9, *pKappa, *pLambda, out_shift);
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
      *pOut4 = bitins(sum13, n_mask6, sum16,mask6, off6);
%endif
%if config.kernel.matmul_fmt == '4x4':
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
        sum9  = ${config.relu_fn}(sum9, out_mult, out_shift);
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
        sum15 = ${config.relu_fn}(sum15, out_mult, out_shift);
        sum16 = ${config.relu_fn}(sum16, out_mult, out_shift);
        *pOut4 = bitins(sum15, n_mask, sum16, mask, off);
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
        sum9  = ${config.relu_fn}(sum9, out_mult, out_shift);
        sum10 = ${config.relu_fn}(sum10, out_mult, out_shift);
        sum11 = ${config.relu_fn}(sum11, out_mult, out_shift);
        sum12 = ${config.relu_fn}(sum12, out_mult, out_shift);
        sum9  = bitins(sum9, n_mask2, sum10, mask2, off2);
        sum9  = bitins(sum9, n_mask4, sum11, mask4, off4);
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
        sum9  = (${pt_out}) ${out_clip_fn}(sum9 >> out_shift);
        sum10 = (${pt_out}) ${out_clip_fn}(sum10 >> out_shift);
        *pOut3 = bitins(sum9, n_mask, sum10, mask, off);
        pOut3++;
        sum11 = (${pt_out}) ${out_clip_fn}(sum11 >> out_shift);
        sum12 = (${pt_out}) ${out_clip_fn}(sum12 >> out_shift);
        pOut3++;

        sum13 = (${pt_out}) ${out_clip_fn}(sum13 >> out_shift);
        sum14 = (${pt_out}) ${out_clip_fn}(sum14 >> out_shift);
        *pOut4 = bitins(sum13, n_mask, sum14, mask, off);
        pOut4++;
        sum15 = (${pt_out}) ${out_clip_fn}(sum15 >> out_shift);
        sum16 = (${pt_out}) ${out_clip_fn}(sum16 >> out_shift);
        *pOut4 = bitins(sum15, n_mask, sum16, mask, off);
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
        sum9  = (${pt_out}) ${out_clip_fn}(sum9 >> out_shift);
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

    pThr+=16;

    sum2 = pulp_nn_i4_quant(sum2, pThr);
    sum6 = pulp_nn_i4_quant(sum6, pThr);

    pThr+=16;

    sum3 = pulp_nn_i4_quant(sum3, pThr);
    sum7 = pulp_nn_i4_quant(sum7, pThr);

    pThr+=16;

    sum4 = pulp_nn_i4_quant(sum4, pThr);
    sum8 = pulp_nn_i4_quant(sum8, pThr);


    pThr+=16;

%if config.kernel.matmul_fmt == '4x4':
    sum9  = pulp_nn_i4_quant(sum9, pThr);
    sum13 = pulp_nn_i4_quant(sum13, pThr);

    pThr+=16;

    sum10 = pulp_nn_i4_quant(sum10, pThr);
    sum14 = pulp_nn_i4_quant(sum14, pThr);

    pThr+=16;

    sum11 = pulp_nn_i4_quant(sum11, pThr);
    sum15 = pulp_nn_i4_quant(sum15, pThr);

    pThr+=16;

    sum12 = pulp_nn_i4_quant(sum12, pThr);
    sum16 = pulp_nn_i4_quant(sum16, pThr);

    pThr+=16;
%endif

    *pOut = bitins(sum, n_mask, sum2, mask, off);

    pOut++;

    *pOut2 = bitins(sum5, n_mask, sum6, mask, off);

    pOut2++;

    *pOut = bitins(sum3, n_mask, sum4, mask, off);

    pOut++;

    *pOut2 = bitins(sum7, n_mask, sum8, mask, off);

    pOut2++;

%if config.kernel.matmul_fmt == '4x4':
    *pOut3 = bitins(sum9, n_mask, sum10, mask, off);

    pOut++;

    *pOut4 = bitins(sum13, n_mask, sum14, mask, off);
    pOut4++;

    *pOut3 = bitins(sum11, n_mask, sum12, mask, off);

    pOut3++;

    *pOut4 = bitins(sum15, n_mask, sum16, mask, off);

    pOut4++;
%endif
%elif config.kernel.out_data_t == 2:
    sum = pulp_nn_i2_quant(sum, pThr);
    sum5 = pulp_nn_i2_quant(sum5, pThr);

    pThr+=4;

    sum2 = pulp_nn_i2_quant(sum2, pThr);
    sum6 = pulp_nn_i2_quant(sum6, pThr);

    sum = bitins(sum, n_mask2, sum2, mask2, off2);
    sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);

    pThr+=4;

    sum3 = pulp_nn_i2_quant(sum3, pThr);
    sum7 = pulp_nn_i2_quant(sum7, pThr);

    sum = bitins(sum, n_mask4, sum3, mask4, off4);
    sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);

    pThr+=4;

    sum4 = pulp_nn_i2_quant(sum4, pThr);
    sum8 = pulp_nn_i2_quant(sum8, pThr);

    pThr+=4;

%if config.kernel.matmul_fmt == '4x4':
    sum9  = pulp_nn_i2_quant(sum9, pThr);
    sum13 = pulp_nn_i2_quant(sum13, pThr);

    pThr+=4;

    sum10 = pulp_nn_i2_quant(sum10, pThr);
    sum14 = pulp_nn_i2_quant(sum14, pThr);

    pThr+=4;

    sum11 = pulp_nn_i2_quant(sum11, pThr);
    sum15 = pulp_nn_i2_quant(sum15, pThr);

    pThr+=4;

    sum12 = pulp_nn_i2_quant(sum12, pThr);
    sum16 = pulp_nn_i2_quant(sum16, pThr);

    pThr+=4;
%endif

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

  w_rollback = 4;
  W_ROLLBACK(w_rollback);
  W_SKIP("0");
%if config.kernel.matmul_fmt == '4x2':
  MIXED_SKIP("2");
%elif config.kernel.matmul_fmt == '4x4':
  MIXED_SKIP("4");
%endif

  while(chan_left)
  {
    ${pt_in} *pB = pIn;

    int8_t *pA = pWeight + (num_col_im2col_w * (ch_out - chan_left));

    ${int_t_in} *ptrB  = (${int_t_in} *) pB;

    int32_t *ptrA  = (int32_t *) pA;

    A_ADDRESS(ptrB);
    W_ADDRESS(ptrA);

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

    int sum = 0;
    if (pBias != NULL)
    {
      sum = ((int) (*pBias++));    
    }
    int sum2 = sum;
%if config.kernel.matmul_fmt == '4x4':
    int sum3 = sum;
    int sum4 = sum;
%endif

%if config.kernel.out_data_t == 4:
    ${pt_out} out[2];
    ${pt_out} out2[2];
%if config.kernel.matmul_fmt == '4x4':
    ${pt_out} out3[2];
    ${pt_out} out4[2];
%endif
%endif
    for(int j=0; j < (num_col_im2col >> ${int(math.log2(((int(32/config.max_precision))*(int(config.max_precision/config.kernel.wt_data_t)))))}); j++)
    {
%if config.kernel.wt_data_t >= config.kernel.in_data_t:
      ptrB = MacLoadInit(0, 1, 0, 1, ptrB);

      sum  = ${mac_fn}(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x2':
      sum2 = ${mac_fn}(1, 0, 0, 1, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
%elif config.kernel.matmul_fmt == '4x4':
      sum2 = ${mac_fn}(0, 1, 0, 1, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);

      sum3 = ${mac_fn}(0, 1, 0, 0, ptrB, sum3);
      ptrB = MacLoadUpdate(ptrB);

      sum4 = ${mac_fn}(1, 0, 0, 1, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);
%endif
%elif config.kernel.wt_data_t < config.kernel.in_data_t:
%if (config.kernel.in_data_t/config.kernel.wt_data_t) == 4:
      ptrB = MacLoadInit(0, 1, 0, 1, ptrB);

      sum  = ${mac_fn}(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

      sum2 = ${mac_fn}(0, 1, 0, 1, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x4':
      sum3 = ${mac_fn}(0, 1, 0, 0, ptrB, sum3);
      ptrB = MacLoadUpdate(ptrB);

      sum4 = ${mac_fn}(0, 1, 0, 1, ptrB, sum4);
%endif

      sum  = ${mac_fn}(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

      sum2 = ${mac_fn}(0, 1, 0, 1, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x4':
      sum3 = ${mac_fn}(0, 1, 0, 0, ptrB, sum3);
      ptrB = MacLoadUpdate(ptrB);

      sum4 = ${mac_fn}(0, 1, 0, 1, ptrB, sum4);
%endif

      sum  = ${mac_fn}(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

      sum2 = ${mac_fn}(0, 1, 0, 1, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x4':
      sum3 = ${mac_fn}(0, 1, 0, 0, ptrB, sum3);
      ptrB = MacLoadUpdate(ptrB);

      sum4 = ${mac_fn}(0, 1, 0, 1, ptrB, sum4);
%endif

      sum  = ${mac_fn}(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x2':
      sum2 = ${mac_fn}(1, 0, 0, 1, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
%elif config.kernel.matmul_fmt == '4x4':
      sum2 = ${mac_fn}(0, 1, 0, 1, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);

      sum3 = ${mac_fn}(0, 1, 0, 0, ptrB, sum3);
      ptrB = MacLoadUpdate(ptrB);

      sum4 = ${mac_fn}(1, 0, 0, 1, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);
%endif
%elif (config.kernel.in_data_t/config.kernel.wt_data_t) == 2:
      ptrB  = MacLoadInit(0, 1, 0, 1, ptrB);

      sum  = ${mac_fn}(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

      sum2 = ${mac_fn}(0, 1, 0, 1, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x4':
      sum3 = ${mac_fn}(0, 1, 0, 0, ptrB, sum3);
      ptrB = MacLoadUpdate(ptrB);

      sum4 = ${mac_fn}(0, 1, 0, 1, ptrB, sum4);
%endif

      sum  = ${mac_fn}(0, 1, 0, 0, ptrB, sum);   
      ptrB = MacLoadUpdate(ptrB);

%if config.kernel.matmul_fmt == '4x2':
      sum2 = ${mac_fn}(1, 0, 0, 1, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
%elif config.kernel.matmul_fmt == '4x4':
      sum2 = ${mac_fn}(0, 1, 0, 1, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);

      sum3 = ${mac_fn}(0, 1, 0, 0, ptrB, sum3);
      ptrB = MacLoadUpdate(ptrB);

      sum4 = ${mac_fn}(1, 0, 0, 1, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);
%endif
%endif
%endif
    }
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    asm volatile ("csrr %0, 0x101" : "=r" (pA));
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
      
      ${pt_in} *pB2 = (pB + num_col_im2col_a);
%if config.kernel.matmul_fmt == '4x4':
      ${pt_in} *pB3 = (pB2 + num_col_im2col_a);
      ${pt_in} *pB4 = (pB3 + num_col_im2col_a);
%endif

      int8_t *pA2 = (pA  + num_col_im2col_w);
      int8_t *pA3 = (pA2 + num_col_im2col_w);
      int8_t *pA4 = (pA3 + num_col_im2col_w);

      do
      {
%if config.kernel.in_data_t == 8:
%if config.kernel.wt_data_t == 2:
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = *pB3++;
        ${pt_in} inB4 = *pB4++;
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 2);

        inB = *pB++;
        inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        inB3 = *pB3++;
        inB4 = *pB4++;
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 4);

        inB = *pB++;
        inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        inB3 = *pB3++;
        inB4 = *pB4++;
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB3;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 6);

        inB = *pB++;
        inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        inB3 = *pB3++;
        inB4 = *pB4++;
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        pA++;

        col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
        int8_t inA = (int8_t) bitext((int) *pA, 4, 0);

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = *pB3++;
        ${pt_in} inB4 = *pB4++;
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 4, 4);

        inB = *pB++;
        inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        inB3 = *pB3++;
        inB4 = *pB4++;
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        pA++;

        col_cnt_im2col-=2;
%else:
        int8_t inA = *pA++;

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = *pB3++;
        ${pt_in} inB4 = *pB4++;
%endif
        asm volatile("": : :"memory");
        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        col_cnt_im2col--;
%endif
%elif config.kernel.in_data_t == 4:
%if config.kernel.wt_data_t == 2:
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);

        ${pt_in} inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_in} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 0);
        ${pt_in} inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 0);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 2);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 4);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 4);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        pB++;
        pB2++;
%if config.kernel.matmul_fmt == '4x4':
        pB3++;
        pB4++;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 0);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 0);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 6);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 4);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 4);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        pA++;

        pB++;
        pB2++;
%if config.kernel.matmul_fmt == '4x4':
        pB3++;
        pB4++;
%endif

        col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
        int8_t inA = (int8_t) bitext((int) *pA, 4, 0);

        ${pt_in} inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_in} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 0);
        ${pt_in} inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 0);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA *  inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 4, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 4);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 4);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 4);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        pA++;

        pB++;
        pB2++;
%if config.kernel.matmul_fmt == '4x4':
        pB3++;
        pB4++;
%endif

        col_cnt_im2col-=2;
%else:
        int8_t inA = *pA++;

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = *pB3++;
        ${pt_in} inB4 = *pB4++;
%endif
        asm volatile("": : :"memory");
        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        col_cnt_im2col--;
%endif
%elif config.kernel.in_data_t == 2:
%if config.kernel.wt_data_t == 2:
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);

        ${pt_in} inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 0);
        ${pt_in} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 0);
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 2, 0);
        ${pt_in} inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 2, 0);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 2);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 2);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 2);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 2, 2);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 2, 2);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 4);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 4);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 2, 4);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 2, 4);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 2, 6);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 2, 6);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 2, 6);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 2, 6);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 2, 6);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        pA++;

        pB++;
        pB2++;
%if config.kernel.matmul_fmt == '4x4':
        pB3++;
        pB4++;
%endif

        col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
        int8_t inA = (int8_t) bitext((int) *pA, 4, 0);

        ${pt_in} inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        ${pt_in} inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 0);
        ${pt_in} inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 0);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        inA = (int8_t) bitext((int) *pA, 4, 4);

        inB = (${pt_in})${bex}((${int_t_in}) *pB, 4, 0);
        inB2 = (${pt_in})${bex}((${int_t_in}) *pB2, 4, 0);
%if config.kernel.matmul_fmt == '4x4':
        inB3 = (${pt_in})${bex}((${int_t_in}) *pB3, 4, 0);
        inB4 = (${pt_in})${bex}((${int_t_in}) *pB4, 4, 0);
%endif

        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        pA++;

        pB++;
        pB2++;
%if config.kernel.matmul_fmt == '4x4':
        pB3++;
        pB4++;
%endif

        col_cnt_im2col-=2;
%else:
        int8_t inA = *pA++;

        ${pt_in} inB = *pB++;
        ${pt_in} inB2 = *pB2++;
%if config.kernel.matmul_fmt == '4x4':
        ${pt_in} inB3 = *pB3++;
        ${pt_in} inB4 = *pB4++;
%endif
        asm volatile("": : :"memory");
        sum += inA * inB;

        sum2 += inA * inB2;

%if config.kernel.matmul_fmt == '4x4':
        sum3 += inA * inB3;

        sum4 += inA * inB4;
%endif

        col_cnt_im2col--;
%endif
%endif
      } while(col_cnt_im2col);
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
%if config.kernel.matmul_fmt == '4x4':
      *pOut3 = ${config.bn_fn}(sum3, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = ${config.bn_fn}(sum4, *pKappa, *pLambda, out_shift);
      pOut4++;
%endif
      pKappa++;
      pLambda++;
%elif config.kernel.out_data_t == 4:
      uint8_t i_o = i & 0x01;
      out[i_o] = ${config.bn_fn}(sum, *pKappa, *pLambda, out_shift);
      out2[i_o] = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
%if config.kernel.matmul_fmt == '4x4':
      out3[i_o] = ${config.bn_fn}(sum3, *pKappa, *pLambda, out_shift);
      out4[i_o] = ${config.bn_fn}(sum4, *pKappa, *pLambda, out_shift);
%endif
      pKappa++;
      pLambda++;
      if(i_o == 0x01)
      {
        *pOut = bitins(out[0], n_mask, out[1], mask, off);
        *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
%if config.kernel.matmul_fmt == '4x4':
        *pOut3 = bitins(out3[0], n_mask, out3[1], mask, off);
        *pOut4 = bitins(out4[0], n_mask, out4[1], mask, off);
%endif
        pOut++;
        pOut2++;
%if config.kernel.matmul_fmt == '4x4':
        pOut3++;
        pOut4++;
%endif
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
%if config.kernel.matmul_fmt == '4x4':
        *pOut3 = ${config.relu_fn}(sum3, out_mult, out_shift);
        pOut3++;
        *pOut4 = ${config.relu_fn}(sum4, out_mult, out_shift);
        pOut4++;
%endif
%elif config.kernel.out_data_t == 4:
        uint8_t i_o = i & 0x01;
        out[i_o] = ${config.relu_fn}(sum, out_mult, out_shift);
        out2[i_o] = ${config.relu_fn}(sum2, out_mult, out_shift);
%if config.kernel.matmul_fmt == '4x4':
        out3[i_o] = ${config.relu_fn}(sum3, out_mult, out_shift);
        out4[i_o] = ${config.relu_fn}(sum4, out_mult, out_shift);
%endif
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask, out[1], mask, off);
          *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
%if config.kernel.matmul_fmt == '4x4':
          *pOut3 = bitins(out3[0], n_mask, out3[1], mask, off);
          *pOut4 = bitins(out4[0], n_mask, out4[1], mask, off);
%endif
          pOut++;
          pOut2++;
%if config.kernel.matmul_fmt == '4x4':
          pOut3++;
          pOut4++;
%endif
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
%if config.kernel.matmul_fmt == '4x4':
        *pOut3 = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
        pOut3++;
        *pOut4 = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
        pOut4++;
%endif
%elif config.kernel.out_data_t == 4:
        uint8_t i_o = i & 0x01;
        out[i_o] = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
        out2[i_o] = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
%if config.kernel.matmul_fmt == '4x4':
        out3[i_o] = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
        out4[i_o] = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
%endif
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask, out[1], mask, off);
          *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
%if config.kernel.matmul_fmt == '4x4':
          *pOut3 = bitins(out3[0], n_mask, out3[1], mask, off);
          *pOut4 = bitins(out4[0], n_mask, out4[1], mask, off);
%endif
          pOut++;
          pOut2++;
%if config.kernel.matmul_fmt == '4x4':
          pOut3++;
          pOut4++;
%endif
        }
%endif
      }
    }
%elif config.kernel.out_data_t == 4:
    uint8_t i_o = i & 0x01;
    out[i_o] = pulp_nn_i4_quant(sum, pThr);
    out2[i_o] = pulp_nn_i4_quant(sum2, pThr);
%if config.kernel.matmul_fmt == '4x4':
    out3[i_o] = pulp_nn_i4_quant(sum3, pThr);
    out4[i_o] = pulp_nn_i4_quant(sum4, pThr);
%endif
    pThr+=16;
    if(i_o == 0x01)
    {
      *pOut = bitins(out[0], n_mask, out[1], mask, off);
      *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
%if config.kernel.matmul_fmt == '4x4':
      *pOut3 = bitins(out3[0], n_mask, out3[1], mask, off);
      *pOut4 = bitins(out4[0], n_mask, out4[1], mask, off);
%endif
      pOut++;
      pOut2++;
%if config.kernel.matmul_fmt == '4x4':
      pOut3++;
      pOut4++;
%endif
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
