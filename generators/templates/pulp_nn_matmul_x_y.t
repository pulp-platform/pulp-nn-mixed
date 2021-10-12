<%! import math %>
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


uint8_t *${config.fn_name}(
                        uint8_t *pIn,
                        int8_t *pBias,
                        uint8_t *pOut,
                        uint8_t *pOut2,
                        int8_t *pWeight,
%if config.kernel.act_prec == '32bit':
                        int32_t *pKappa,
                        int32_t *pLambda,
%elif config.kernel.act_prec == '64bit':
                        int64_t *pKappa,
                        int64_t *pLambda,
%endif
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t num_col_im2col,
                        uint16_t ch_out,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
%for i in range(int(8/config.kernel.out_data_t)-1):
%if config.kernel.out_data_t > 1:
  int8_t mask${(i+1)*config.kernel.out_data_t} = ${hex(((config.kernel.out_data_t*config.kernel.out_data_t)-1)<<((i+1)*config.kernel.out_data_t))};
%else:
  int8_t mask${(i+1)*config.kernel.out_data_t} = ${hex(config.kernel.out_data_t<<((i+1)*config.kernel.out_data_t))};
%endif
  int8_t n_mask${(i+1)*config.kernel.out_data_t} = ~ mask${(i+1)*config.kernel.out_data_t};
  int8_t off${(i+1)*config.kernel.out_data_t} = ${(i+1)*config.kernel.out_data_t};
%endfor

% for i in range(int(16/config.kernel.wt_data_t)):
  v4u vecB${i+1};
% endfor
%if config.kernel.wt_data_t != 8:
%if config.kernel.out_data_t != 1:
% for i in range(4):
  v4s vecA${i+1}[${int(8/config.kernel.wt_data_t)}];
% endfor
%else:
% for i in range(8):
  v4s vecA${i+1}[${int(8/config.kernel.wt_data_t)}];
% endfor
%endif
%else:
%if config.kernel.out_data_t != 1:
% for i in range(4):
  v4s vecA${i+1};
% endfor
%else:
% for i in range(8):
  v4s vecA${i+1};
% endfor
%endif
%endif

  uint16_t ch_out_r = PACK_INT${config.kernel.out_data_t}_SIZE(ch_out);

  uint16_t num_col_im2col_w = PACK_INT${config.kernel.wt_data_t}_SIZE(num_col_im2col);

  //uint8_t *pOut2 = pOut + ch_out_r;
  int8_t *pA1 = pWeight;

%if config.kernel.out_data_t != 1:
  uint16_t chan_left = ch_out & 0x3;
%else:
  uint16_t chan_left = ch_out & 0x7;
%endif

%if config.kernel.out_data_t != 1:
  for(int i=0; i < (ch_out >> 2); i++)
%else:
  for(int i=0; i < (ch_out >> 3); i++)
%endif
  {
    uint8_t *pB1 =  pIn;
    uint8_t *pB2 = (pB1 + num_col_im2col);
    int8_t *pA2 = (pA1 + num_col_im2col_w);
    int8_t *pA3 = (pA2 + num_col_im2col_w);
    int8_t *pA4 = (pA3 + num_col_im2col_w);
%if config.kernel.out_data_t == 1:
    int8_t *pA5 = (pA4 + num_col_im2col_w);
    int8_t *pA6 = (pA5 + num_col_im2col_w);
    int8_t *pA7 = (pA6 + num_col_im2col_w);
    int8_t *pA8 = (pA7 + num_col_im2col_w);
%endif

    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    int sum4 = 0;
    int sum5 = 0;
    int sum6 = 0;
    int sum7 = 0;
    int sum8 = 0;
%if config.kernel.out_data_t == 1:
    int sum9 = 0;
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
      sum1 = ((int) (*pBias++));
      sum2 = ((int) (*pBias++));
      sum3 = ((int) (*pBias++));
      sum4 = ((int) (*pBias++));
%if config.kernel.out_data_t == 1:
      sum9 = ((int) (*pBias++));
      sum10 = ((int) (*pBias++));
      sum11 = ((int) (*pBias++));
      sum12 = ((int) (*pBias++));
%endif

      sum5 = sum1;
      sum6 = sum2;
      sum7 = sum3;
      sum8 = sum4;
%if config.kernel.out_data_t == 1:
      sum13 = sum5;
      sum14 = sum6;
      sum15 = sum7;
      sum16 = sum8;
%endif
    }

    for(int j=0; j<(num_col_im2col_w >> 2); j++)
    {
% for i in range(int(16/config.kernel.wt_data_t)):
      vecB${i+1} = *((v4u*)pB${(i % 2)+1} + ${(i//2) * 4});

% endfor
      pB1+=${int(32/config.kernel.wt_data_t)};
      pB2+=${int(32/config.kernel.wt_data_t)};

% if config.kernel.wt_data_t != 8:
%if config.kernel.out_data_t != 1:
% for i in range(4):
      pA${i+1} = ${config.unpack_fn}(pA${i+1},vecA${i+1});

% for j in range(int(8/config.kernel.wt_data_t)):
      sum${i+1} = SumDotp4(vecB${int((j+1)*2-1)}, vecA${i+1}[${j}], sum${i+1});
      sum${i+5} = SumDotp4(vecB${int((j+1)*2)}, vecA${i+1}[${j}], sum${i+5});
% endfor
% endfor
%else:
% for i in range(8):
      pA${i+1} = ${config.unpack_fn}(pA${i+1},vecA${i+1});

% for j in range(int(8/config.kernel.wt_data_t)):
      sum${(i+1)+(i//4)*4} = SumDotp4(vecB${int((j+1)*2-1)}, vecA${i+1}[${j}], sum${(i+1)+(i//4)*4});
      sum${(i+5)+(i//4)*4} = SumDotp4(vecB${int((j+1)*2)}, vecA${i+1}[${j}], sum${(i+5)+(i//4)*4});
% endfor
% endfor
%endif
%else:
%if config.kernel.out_data_t != 1:
      vecA1 = *((v4s*)pA1);
      vecA2 = *((v4s*)pA2);
      vecA3 = *((v4s*)pA3);
      vecA4 = *((v4s*)pA4);

      sum1 = SumDotp4(vecB1, vecA1, sum1 );
      sum2 = SumDotp4(vecB1, vecA2, sum2);
      sum3 = SumDotp4(vecB1, vecA3, sum3);
      sum4 = SumDotp4(vecB1, vecA4, sum4);

      sum5 = SumDotp4(vecB2, vecA1, sum5);
      sum6 = SumDotp4(vecB2, vecA2, sum6);
      sum7 = SumDotp4(vecB2, vecA3, sum7);
      sum8 = SumDotp4(vecB2, vecA4, sum8);

      pA1+=4;
      pA2+=4;
      pA3+=4;
      pA4+=4;
%else:
      vecA1 = *((v4s*)pA1);
      vecA2 = *((v4s*)pA2);
      vecA3 = *((v4s*)pA3);
      vecA4 = *((v4s*)pA4);

      sum1 = SumDotp4(vecB1, vecA1, sum1 );
      sum2 = SumDotp4(vecB1, vecA2, sum2);
      sum3 = SumDotp4(vecB1, vecA3, sum3);
      sum4 = SumDotp4(vecB1, vecA4, sum4);

      sum5 = SumDotp4(vecB2, vecA1, sum5);
      sum6 = SumDotp4(vecB2, vecA2, sum6);
      sum7 = SumDotp4(vecB2, vecA3, sum7);
      sum8 = SumDotp4(vecB2, vecA4, sum8);

      pA1+=4;
      pA2+=4;
      pA3+=4;
      pA4+=4;

      vecA5 = *((v4s*)pA5);
      vecA6 = *((v4s*)pA6);
      vecA7 = *((v4s*)pA7);
      vecA8 = *((v4s*)pA8);

      sum9  = SumDotp4(vecB1, vecA5, sum9);
      sum10 = SumDotp4(vecB1, vecA6, sum10);
      sum11 = SumDotp4(vecB1, vecA7, sum11);
      sum12 = SumDotp4(vecB1, vecA8, sum12);

      sum13 = SumDotp4(vecB2, vecA5, sum13);
      sum14 = SumDotp4(vecB2, vecA6, sum14);
      sum15 = SumDotp4(vecB2, vecA7, sum15);
      sum16 = SumDotp4(vecB2, vecA8, sum16);

      pA5+=4;
      pA6+=4;
      pA7+=4;
      pA8+=4;
%endif
%endif
    }
    uint16_t col_cnt_im2col = num_col_im2col & ${hex(int(32/config.kernel.wt_data_t-1))};

    while (col_cnt_im2col)
    {
% if config.kernel.wt_data_t != 8:
%if config.kernel.out_data_t != 1:
% for i in range(int(8/config.kernel.wt_data_t)):
%if i == 0:
% for j in range(4):
	    int8_t inA${j+1} = (int8_t) bitext((int) *pA${j+1}, ${config.kernel.wt_data_t}, ${i*config.kernel.wt_data_t});
% endfor
      uint8_t inB1 = *pB1++;
      uint8_t inB2 = *pB2++;
%else:
% for j in range(4):
	    inA${j+1} = (int8_t) bitext((int) *pA${j+1}, ${config.kernel.wt_data_t}, ${i*config.kernel.wt_data_t});
% endfor
      inB1 = *pB1++;
      inB2 = *pB2++;
%endif
% for j in range(8):
      sum${j+1} += inA${(j % 4)+1} * inB${(j//4)+1};
% endfor
% endfor
%else:
% for i in range(int(8/config.kernel.wt_data_t)):
%if i == 0:
% for j in range(8):
	    int8_t inA${j+1} = (int8_t) bitext((int) *pA${j+1}, ${config.kernel.wt_data_t}, ${i*config.kernel.wt_data_t});
% endfor
      uint8_t inB1 = *pB1++;
      uint8_t inB2 = *pB2++;
%else:
% for j in range(8):
	    inA${j+1} = (int8_t) bitext((int) *pA${j+1}, ${config.kernel.wt_data_t}, ${i*config.kernel.wt_data_t});
% endfor
      inB1 = *pB1++;
      inB2 = *pB2++;
%endif
% for j in range(8):
      sum${(j+1)} += inA${(j % 4)+1} * inB${(j//4)+1};
      sum${(j+9)} += inA${(j % 4)+5} * inB${(j//4)+1};
% endfor
% endfor
%endif
      pA1++;
      pA2++;
      pA3++;
      pA4++;
%if config.kernel.out_data_t == 1:
      pA5++;
      pA6++;
      pA7++;
      pA8++;
%endif
      col_cnt_im2col-=${int(8/config.kernel.wt_data_t)};
%else:
      int8_t inA1 = *pA1++;
      int8_t inA2 = *pA2++;
      int8_t inA3 = *pA3++;
      int8_t inA4 = *pA4++;
      uint8_t inB1 = *pB1++;
      uint8_t inB2 = *pB2++;
      asm volatile("": : :"memory");
      sum1 += inA1 * inB1;
      sum2 += inA2 * inB1;
      sum3 += inA3 * inB1;
      sum4 += inA4 * inB1;
      sum5 += inA1 * inB2;
      sum6 += inA2 * inB2;
      sum7 += inA3 * inB2;
      sum8 += inA4 * inB2;
      col_cnt_im2col--;
%endif
    }
%if config.kernel.out_data_t == 8 or config.kernel.quantization == 'shift_clip':
    if (flag_batch_norm && flag_relu)
    {
      %if config.kernel.out_data_t == 8:
            *pOut = ${config.bn_fn}(sum1, *pKappa, *pLambda, out_shift);
            pOut++;
            *pOut2 = ${config.bn_fn}(sum5, *pKappa, *pLambda, out_shift);
            pOut2++;
            pKappa++;
            pLambda++;

            *pOut = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
            pOut++;
            *pOut2 = ${config.bn_fn}(sum6, *pKappa, *pLambda, out_shift);
            pOut2++;
            pKappa++;
            pLambda++;

            *pOut = ${config.bn_fn}(sum3, *pKappa, *pLambda, out_shift);
            pOut++;
            *pOut2 = ${config.bn_fn}(sum7, *pKappa, *pLambda, out_shift);
            pOut2++;
            pKappa++;
            pLambda++;

            *pOut = ${config.bn_fn}(sum4, *pKappa, *pLambda, out_shift);
            pOut++;
            *pOut2 = ${config.bn_fn}(sum8, *pKappa, *pLambda, out_shift);
            pOut2++;
            pKappa++;
            pLambda++;
      %elif config.kernel.out_data_t == 4:
            sum1 = ${config.bn_fn}(sum1, *pKappa, *pLambda, out_shift);
            sum5 = ${config.bn_fn}(sum5, *pKappa, *pLambda, out_shift);
            pKappa++;
            pLambda++;
            sum2 = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
            sum6 = ${config.bn_fn}(sum6, *pKappa, *pLambda, out_shift);
            *pOut = bitins(sum1, n_mask4, sum2, mask4, off4);
            *pOut2 = bitins(sum5, n_mask4, sum6, mask4, off4);
            pKappa++;
            pLambda++;
            pOut++;
            pOut2++;
            sum3 = ${config.bn_fn}(sum3, *pKappa, *pLambda, out_shift);
            sum7 = ${config.bn_fn}(sum7, *pKappa, *pLambda, out_shift);
            pKappa++;
            pLambda++;
            sum4 = ${config.bn_fn}(sum4, *pKappa, *pLambda, out_shift);
            sum8 = ${config.bn_fn}(sum8, *pKappa, *pLambda, out_shift);
            pKappa++;
            pLambda++;
            *pOut = bitins(sum3, n_mask4, sum4, mask4, off4);
            *pOut2 = bitins(sum7, n_mask4, sum8, mask4, off4);
            pOut++;
            pOut2++;
      %elif config.kernel.out_data_t == 2:
            sum1 = ${config.bn_fn}(sum1, *pKappa, *pLambda, out_shift);
            sum5 = ${config.bn_fn}(sum5, *pKappa, *pLambda, out_shift);
            pKappa++;
            pLambda++;
            sum2 = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
            sum6 = ${config.bn_fn}(sum6, *pKappa, *pLambda, out_shift);
            pKappa++;
            pLambda++;
            sum3 = ${config.bn_fn}(sum3, *pKappa, *pLambda, out_shift);
            sum7 = ${config.bn_fn}(sum7, *pKappa, *pLambda, out_shift);
            pKappa++;
            pLambda++;
            sum4 = ${config.bn_fn}(sum4, *pKappa, *pLambda, out_shift);
            sum8 = ${config.bn_fn}(sum8, *pKappa, *pLambda, out_shift);
            pKappa++;
            pLambda++;
            sum1 = bitins(sum1, n_mask2, sum2, mask2, off2);
            sum1 = bitins(sum1, n_mask4, sum3, mask4, off4);
            *pOut = bitins(sum1, n_mask6, sum4, mask6, off6);
            sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
            sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
            *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);
            pOut2++;
            pOut++;
            %elif config.kernel.out_data_t == 1:
                  sum1 = ${config.bn_fn}(sum1, *pKappa, *pLambda, out_shift);
                  sum5 = ${config.bn_fn}(sum5, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum2 = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
                  sum6 = ${config.bn_fn}(sum6, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum3 = ${config.bn_fn}(sum3, *pKappa, *pLambda, out_shift);
                  sum7 = ${config.bn_fn}(sum7, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum4 = ${config.bn_fn}(sum4, *pKappa, *pLambda, out_shift);
                  sum8 = ${config.bn_fn}(sum8, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum9 = ${config.bn_fn}(sum9, *pKappa, *pLambda, out_shift);
                  sum13 = ${config.bn_fn}(sum13, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum10 = ${config.bn_fn}(sum10, *pKappa, *pLambda, out_shift);
                  sum14 = ${config.bn_fn}(sum14, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum11 = ${config.bn_fn}(sum11, *pKappa, *pLambda, out_shift);
                  sum15 = ${config.bn_fn}(sum15, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum12 = ${config.bn_fn}(sum12, *pKappa, *pLambda, out_shift);
                  sum16 = ${config.bn_fn}(sum16, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum1  = bitins(sum1, n_mask1, sum2, mask1, off1);
                  sum1  = bitins(sum1, n_mask2, sum3, mask2, off2);
                  sum1  = bitins(sum1, n_mask3, sum4, mask3, off3);
                  sum1  = bitins(sum1, n_mask4, sum9, mask4, off4);
                  sum1  = bitins(sum1, n_mask5, sum10, mask5, off5);
                  sum1  = bitins(sum1, n_mask6, sum11, mask6, off6);
                  *pOut = bitins(sum1, n_mask7, sum12, mask7, off7);

                  sum5   = bitins(sum5, n_mask1, sum6, mask1, off1);
                  sum5   = bitins(sum5, n_mask2, sum7, mask2, off2);
                  sum5   = bitins(sum5, n_mask3, sum8, mask3, off3);
                  sum5   = bitins(sum5, n_mask4, sum13, mask4, off4);
                  sum5   = bitins(sum5, n_mask5, sum14, mask5, off5);
                  sum5   = bitins(sum5, n_mask6, sum15, mask6, off6);
                  *pOut2 = bitins(sum5, n_mask7, sum16, mask7, off7);
                  pOut2++;
                  pOut++;
      %endif
    }
    else
    {
      if (flag_relu == 1)
      {
%if config.kernel.out_data_t == 8:
        *pOut = ${config.relu_fn}(sum1, out_mult, out_shift);
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
%elif config.kernel.out_data_t == 4:
        sum1 = ${config.relu_fn}(sum1, out_mult, out_shift);
        sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
        *pOut = bitins(sum1, n_mask4, sum2, mask4, off4);
        pOut++;
        sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
        sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
        *pOut = bitins(sum3, n_mask4, sum4, mask4, off4);
        pOut++;

        sum5 = ${config.relu_fn}(sum5, out_mult, out_shift);
        sum6 = ${config.relu_fn}(sum6, out_mult, out_shift);
        *pOut2 = bitins(sum5, n_mask4, sum6, mask4, off4);
        pOut2++;
        sum7 = ${config.relu_fn}(sum7, out_mult, out_shift);
        sum8 = ${config.relu_fn}(sum8, out_mult, out_shift);
        *pOut2 = bitins(sum7, n_mask4, sum8, mask4, off4);
        pOut2++;
%elif config.kernel.out_data_t == 2:
        sum1 = ${config.relu_fn}(sum1, out_mult, out_shift);
        sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
        sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
        sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
        sum1 = bitins(sum1, n_mask2, sum2, mask2, off2);
        sum1 = bitins(sum1, n_mask4, sum3, mask4, off4);
        *pOut = bitins(sum1, n_mask6, sum4, mask6, off6);
        pOut++;
        sum5 = ${config.relu_fn}(sum5, out_mult, out_shift);
        sum6 = ${config.relu_fn}(sum6, out_mult, out_shift);
        sum7 = ${config.relu_fn}(sum7, out_mult, out_shift);
        sum8 = ${config.relu_fn}(sum8, out_mult, out_shift);
        sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
        sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
        *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);
        pOut2++;
        %elif config.kernel.out_data_t == 1:
                sum1 = ${config.relu_fn}(sum1, out_mult, out_shift);
                sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
                sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
                sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
                sum9 = ${config.relu_fn}(sum9, out_mult, out_shift);
                sum10 = ${config.relu_fn}(sum10, out_mult, out_shift);
                sum11 = ${config.relu_fn}(sum11, out_mult, out_shift);
                sum12 = ${config.relu_fn}(sum12, out_mult, out_shift);

                sum1  = bitins(sum1, n_mask1, sum2, mask1, off1);
                sum1  = bitins(sum1, n_mask2, sum3, mask2, off2);
                sum1  = bitins(sum1, n_mask3, sum4, mask3, off3);
                sum1  = bitins(sum1, n_mask4, sum9, mask4, off4);
                sum1  = bitins(sum1, n_mask5, sum10, mask5, off5);
                sum1  = bitins(sum1, n_mask6, sum11, mask6, off6);
                *pOut = bitins(sum1, n_mask7, sum12, mask7, off7);

                pOut++;

                sum5 = ${config.relu_fn}(sum5, out_mult, out_shift);
                sum6 = ${config.relu_fn}(sum6, out_mult, out_shift);
                sum7 = ${config.relu_fn}(sum7, out_mult, out_shift);
                sum8 = ${config.relu_fn}(sum8, out_mult, out_shift);
                sum13 = ${config.relu_fn}(sum13, out_mult, out_shift);
                sum14 = ${config.relu_fn}(sum14, out_mult, out_shift);
                sum15 = ${config.relu_fn}(sum15, out_mult, out_shift);
                sum16 = ${config.relu_fn}(sum16, out_mult, out_shift);

                sum5   = bitins(sum5, n_mask1, sum6, mask1, off1);
                sum5   = bitins(sum5, n_mask2, sum7, mask2, off2);
                sum5   = bitins(sum5, n_mask3, sum8, mask3, off3);
                sum5   = bitins(sum5, n_mask4, sum13, mask4, off4);
                sum5   = bitins(sum5, n_mask5, sum14, mask5, off5);
                sum5   = bitins(sum5, n_mask6, sum15, mask6, off6);
                *pOut2 = bitins(sum5, n_mask7, sum16, mask7, off7);

                pOut2++;
%endif
      }
      else
      {
%if config.kernel.out_data_t == 8:
        *pOut = (uint8_t) clip8(sum1 >> out_shift);
        pOut++;
        *pOut = (uint8_t) clip8(sum2 >> out_shift);
        pOut++;
        *pOut = (uint8_t) clip8(sum3 >> out_shift);
        pOut++;
        *pOut = (uint8_t) clip8(sum4 >> out_shift);
        pOut++;

        *pOut2 = (uint8_t) clip8(sum5 >> out_shift);
        pOut2++;
        *pOut2 = (uint8_t) clip8(sum6 >> out_shift);
        pOut2++;
        *pOut2 = (uint8_t) clip8(sum7 >> out_shift);
        pOut2++;
        *pOut2 = (uint8_t) clip8(sum8 >> out_shift);
        pOut2++;
%elif config.kernel.out_data_t == 4:
        sum1 = (uint8_t) clip4(sum1 >> out_shift);
        sum2 = (uint8_t) clip4(sum2 >> out_shift);
        *pOut = bitins(sum1, n_mask4, sum2, mask4, off4);
        pOut++;
        sum3 = (uint8_t) clip4(sum3 >> out_shift);
        sum4 = (uint8_t) clip4(sum4 >> out_shift);
        *pOut = bitins(sum3, n_mask4, sum4, mask4, off4);
        pOut++;

        sum5 = (uint8_t) clip4(sum5 >> out_shift);
        sum6 = (uint8_t) clip4(sum6 >> out_shift);
        *pOut2 = bitins(sum5, n_mask4, sum6, mask4, off4);
        pOut2++;
        sum7 = (uint8_t) clip4(sum7 >> out_shift);
        sum8 = (uint8_t) clip4(sum8 >> out_shift);
        *pOut2 = bitins(sum7, n_mask4, sum8, mask4, off4);
        pOut2++;
%elif config.kernel.out_data_t == 2:
        sum1 = (uint8_t) clip2(sum1 >> out_shift);
        sum2 = (uint8_t) clip2(sum2 >> out_shift);
        sum3 = (uint8_t) clip2(sum3 >> out_shift);
        sum4 = (uint8_t) clip2(sum4 >> out_shift);
        sum1 = bitins(sum1, n_mask2, sum2, mask2, off2);
        sum1 = bitins(sum1, n_mask4, sum3, mask4, off4);
        *pOut = bitins(sum1, n_mask6, sum4, mask6, off6);
        pOut++;

        sum5 = (uint8_t) clip2(sum5 >> out_shift);
        sum6 = (uint8_t) clip2(sum6 >> out_shift);
        sum7 = (uint8_t) clip2(sum7 >> out_shift);
        sum8 = (uint8_t) clip2(sum8 >> out_shift);
        sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
        sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
        *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);
        pOut2++;
        %elif config.kernel.out_data_t == 1:
                sum1  = (uint8_t) clip1(sum1 >> out_shift);
                sum2  = (uint8_t) clip1(sum2 >> out_shift);
                sum3  = (uint8_t) clip1(sum3 >> out_shift);
                sum4  = (uint8_t) clip1(sum4 >> out_shift);
                sum9  = (uint8_t) clip1(sum9 >> out_shift);
                sum10 = (uint8_t) clip1(sum10 >> out_shift);
                sum11 = (uint8_t) clip1(sum11 >> out_shift);
                sum12 = (uint8_t) clip1(sum12 >> out_shift);

                sum1  = bitins(sum1, n_mask1, sum2, mask1, off1);
                sum1  = bitins(sum1, n_mask2, sum3, mask2, off2);
                sum1  = bitins(sum1, n_mask3, sum4, mask3, off3);
                sum1  = bitins(sum1, n_mask4, sum9, mask4, off4);
                sum1  = bitins(sum1, n_mask5, sum10, mask5, off5);
                sum1  = bitins(sum1, n_mask6, sum11, mask6, off6);
                *pOut = bitins(sum1, n_mask7, sum12, mask7, off7);

                pOut++;

                sum5  = (uint8_t) clip1(sum5 >> out_shift);
                sum6  = (uint8_t) clip1(sum6 >> out_shift);
                sum7  = (uint8_t) clip1(sum7 >> out_shift);
                sum8  = (uint8_t) clip1(sum8 >> out_shift);
                sum13 = (uint8_t) clip1(sum13 >> out_shift);
                sum14 = (uint8_t) clip1(sum14 >> out_shift);
                sum15 = (uint8_t) clip1(sum15 >> out_shift);
                sum16 = (uint8_t) clip1(sum16 >> out_shift);

                sum5   = bitins(sum5, n_mask1, sum6, mask1, off1);
                sum5   = bitins(sum5, n_mask2, sum7, mask2, off2);
                sum5   = bitins(sum5, n_mask3, sum8, mask3, off3);
                sum5   = bitins(sum5, n_mask4, sum13, mask4, off4);
                sum5   = bitins(sum5, n_mask5, sum14, mask5, off5);
                sum5   = bitins(sum5, n_mask6, sum15, mask6, off6);
                *pOut2 = bitins(sum5, n_mask7, sum16, mask7, off7);
                pOut2++;
        %endif
      }
    }
%elif config.kernel.out_data_t == 4:
    sum1 = pulp_nn_i4_quant(sum1, pThr);
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

    *pOut = bitins(sum1, n_mask4, sum2, mask4, off4);

    pOut++;

    *pOut2 = bitins(sum5, n_mask4, sum6, mask4, off4);

    pOut2++;

    *pOut = bitins(sum3, n_mask4, sum4, mask4, off4);

    pOut++;

    *pOut2 = bitins(sum7, n_mask4, sum8, mask4, off4);

    pOut2++;
%elif config.kernel.out_data_t == 2:
    sum1 = pulp_nn_i2_quant(sum1, pThr);
    sum5 = pulp_nn_i2_quant(sum5, pThr);

    pThr+=4;

    sum2 = pulp_nn_i2_quant(sum2, pThr);
    sum6 = pulp_nn_i2_quant(sum6, pThr);

    sum1 = bitins(sum1, n_mask2, sum2, mask2, off2);
    sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);

    pThr+=4;

    sum3 = pulp_nn_i2_quant(sum3, pThr);
    sum7 = pulp_nn_i2_quant(sum7, pThr);

    sum1 = bitins(sum1, n_mask4, sum3, mask4, off4);
    sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);

    pThr+=4;

    sum4 = pulp_nn_i2_quant(sum4, pThr);
    sum8 = pulp_nn_i2_quant(sum8, pThr);

    pThr+=4;

    *pOut = bitins(sum1, n_mask6, sum4, mask6, off6);

    pOut++;

    *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);

    pOut2++;
%endif
%if config.kernel.out_data_t != 1:
    pA1+=(3 * num_col_im2col_w);
%else:
    pA1+=(7 * num_col_im2col_w);
%endif
  }
%if config.kernel.out_data_t > 2:
  %if config.kernel.out_data_t == 4:
   uint16_t i = 0;
  %endif
   while(chan_left)
  {
    uint8_t *pB1 = pIn;
    uint8_t *pB2 = (pB1 + num_col_im2col);
    int sum1 = 0;
    if (pBias != NULL)
      sum1 = ((int) (*pBias++));
    int sum2 = sum1;

%if config.kernel.out_data_t == 4:
    uint8_t out[2];
    uint8_t out2[2];
%endif
    for(int j=0; j < (num_col_im2col_w >> 2); j++)
    {
%if config.kernel.wt_data_t == 1:
      vecB1  = *((v4u*)pB1);
      vecB2  = *((v4u*)pB2);
      vecB3  = *((v4u*)(pB1  + 4));
      vecB4  = *((v4u*)(pB2 + 4));
      vecB5  = *((v4u*)(pB1  + 8));
      vecB6  = *((v4u*)(pB2 + 8));
      vecB7  = *((v4u*)(pB1  + 12));
      vecB8  = *((v4u*)(pB2 + 12));
      vecB9  = *((v4u*)(pB1  + 16));
      vecB10 = *((v4u*)(pB2 + 16));
      vecB11 = *((v4u*)(pB1  + 20));
      vecB12 = *((v4u*)(pB2 + 20));
      vecB13 = *((v4u*)(pB1  + 24));
      vecB14 = *((v4u*)(pB2 + 24));
      vecB15 = *((v4u*)(pB1  + 28));
      vecB16 = *((v4u*)(pB2 + 28));

      pA1 = ${config.unpack_fn}(pA1,vecA1);

      sum1 = SumDotp4(vecB1, vecA1[0], sum1);
      sum2 = SumDotp4(vecB2, vecA1[0], sum2);
      sum1 = SumDotp4(vecB3, vecA1[1], sum1);
      sum2 = SumDotp4(vecB4, vecA1[1], sum2);
      sum1 = SumDotp4(vecB5, vecA1[2], sum1);
      sum2 = SumDotp4(vecB6, vecA1[2], sum2);
      sum1 = SumDotp4(vecB7, vecA1[3], sum1);
      sum2 = SumDotp4(vecB8, vecA1[3], sum2);
      sum1 = SumDotp4(vecB9, vecA1[4], sum1);
      sum2 = SumDotp4(vecB10, vecA1[4], sum2);
      sum1 = SumDotp4(vecB11, vecA1[5], sum1);
      sum2 = SumDotp4(vecB12, vecA1[5], sum2);
      sum1 = SumDotp4(vecB13, vecA1[6], sum1);
      sum2 = SumDotp4(vecB14, vecA1[6], sum2);
      sum1 = SumDotp4(vecB15, vecA1[7], sum1);
      sum2 = SumDotp4(vecB16, vecA1[7], sum2);

      pB1+=32;
      pB2+=32;
      %elif config.kernel.wt_data_t == 2:
            vecB1 = *((v4u*)pB1);
            vecB2 = *((v4u*)pB2);
            vecB3 = *((v4u*)(pB1 + 4));
            vecB4 = *((v4u*)(pB2 + 4));
            vecB5 = *((v4u*)(pB1 + 8));
            vecB6 = *((v4u*)(pB2 + 8));
            vecB7 = *((v4u*)(pB1 + 12));
            vecB8 = *((v4u*)(pB2 + 12));

            pA1 = ${config.unpack_fn}(pA1,vecA1);

            sum1 = SumDotp4(vecB1, vecA1[0], sum1);
            sum2 = SumDotp4(vecB2, vecA1[0], sum2);
            sum1 = SumDotp4(vecB3, vecA1[1], sum1);
            sum2 = SumDotp4(vecB4, vecA1[1], sum2);
            sum1 = SumDotp4(vecB5, vecA1[2], sum1);
            sum2 = SumDotp4(vecB6, vecA1[2], sum2);
            sum1 = SumDotp4(vecB7, vecA1[3], sum1);
            sum2 = SumDotp4(vecB8, vecA1[3], sum2);

            pB1+=16;
            pB2+=16;
%elif config.kernel.wt_data_t == 4:
      vecB1 = *((v4u*)pB1);
      vecB2 = *((v4u*)pB2);
      vecB3 = *((v4u*)(pB1 + 4));
      vecB4 = *((v4u*)(pB2 + 4));

      pA1 = ${config.unpack_fn}(pA1,vecA1);

      sum1 = SumDotp4(vecB1, vecA1[0], sum1);
      sum2 = SumDotp4(vecB2, vecA1[0], sum2);

      sum1 = SumDotp4(vecB3, vecA1[1], sum1);
      sum2 = SumDotp4(vecB4, vecA1[1], sum2);

      pB1+=8;
      pB2+=8;
%else:
      vecA1 = *((v4s*) pA1);
      vecB1 = *((v4u*) pB1);
      vecB2 = *((v4u*) pB2);

      sum1 = SumDotp4(vecB1, vecA1, sum1);
      sum2 = SumDotp4(vecB2, vecA1, sum2);

      pA1+=4;
      pB1+=4;
      pB2+=4;
%endif
    }
    uint16_t col_cnt_im2col = num_col_im2col & ${hex(int(32/config.kernel.wt_data_t-1))};
    while(col_cnt_im2col)
    {
      %if config.kernel.wt_data_t == 1:
            int8_t inA1 = (int8_t) bitext((int) *pA1, 1, 0);
            uint8_t inB1 = *pB1++;
            uint8_t inB2 = *pB2++;
            sum1 += inA1 * inB1;
            sum2 += inA1 * inB2;
            inA1 = (int8_t) bitext((int) *pA1, 1, 1);
            inB1 = *pB1++;
            inB2 = *pB2++;
            sum1 += inA1 * inB1;
            sum2 += inA1 * inB2;
            inA1 = (int8_t) bitext((int) *pA1, 1, 2);
            inB1 = *pB1++;
            inB2 = *pB2++;
            sum1 += inA1 * inB1;
            sum2 += inA1 * inB2;
            inA1 = (int8_t) bitext((int) *pA1, 1, 3);
            inB1 = *pB1++;
            inB2 = *pB2++;
            sum1 += inA1 * inB1;
            sum2 += inA1 * inB2;

            inA1 = (int8_t) bitext((int) *pA1, 1, 4);
            inB1 = *pB1++;
            inB2 = *pB2++;
            sum1 += inA1 * inB1;
            sum2 += inA1 * inB2;
            inA1 = (int8_t) bitext((int) *pA1, 1, 5);
            inB1 = *pB1++;
            inB2 = *pB2++;
            sum1 += inA1 * inB1;
            sum2 += inA1 * inB2;
            inA1 = (int8_t) bitext((int) *pA1, 1, 6);
            inB1 = *pB1++;
            inB2 = *pB2++;
            sum1 += inA1 * inB1;
            sum2 += inA1 * inB2;
            inA1 = (int8_t) bitext((int) *pA1, 1, 7);
            inB1 = *pB1++;
            inB2 = *pB2++;
            sum1 += inA1 * inB1;
            sum2 += inA1 * inB2;

            pA1++;
            col_cnt_im2col-=8;
%elif config.kernel.wt_data_t == 2:
      int8_t inA1 = (int8_t) bitext((int) *pA1, 2, 0);
      uint8_t inB1 = *pB1++;
      uint8_t inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum2 += inA1 * inB2;
      inA1 = (int8_t) bitext((int) *pA1, 2, 2);
      inB1 = *pB1++;
      inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum2 += inA1 * inB2;
      inA1 = (int8_t) bitext((int) *pA1, 2, 4);
      inB1 = *pB1++;
      inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum2 += inA1 * inB2;
      inA1 = (int8_t) bitext((int) *pA1, 2, 6);
      inB1 = *pB1++;
      inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum2 += inA1 * inB2;

      pA1++;
      col_cnt_im2col-=4;
%elif config.kernel.wt_data_t == 4:
      int8_t inA1 = (int8_t) bitext((int) *pA1, 4, 0);
      uint8_t inB1 = *pB1++;
      uint8_t inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum2 += inA1 * inB2;
      inA1 = (int8_t) bitext((int) *pA1, 4, 4);
      inB1 = *pB1++;
      inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum2 += inA1 * inB2;

      pA1++;
      col_cnt_im2col-=2;
%else:
      int8_t inA1 = *pA1++;
      uint8_t inB1 = *pB1++;
      uint8_t inB2 = *pB2++;
      asm volatile("": : :"memory");
      sum1 += inA1 * inB1;
      sum2 += inA1 * inB2;

      col_cnt_im2col--;
%endif
    }
%if config.kernel.out_data_t == 8 or config.kernel.quantization == 'shift_clip':
    if (flag_batch_norm && flag_relu)
    {
%if config.kernel.out_data_t == 8:
      *pOut = ${config.bn_fn}(sum1, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
      pOut2++;
      pKappa++;
      pLambda++;
%elif config.kernel.out_data_t == 4:
      uint8_t i_o = i & 0x01;
      out[i_o] = ${config.bn_fn}(sum1, *pKappa, *pLambda, out_shift);
      out2[i_o] = ${config.bn_fn}(sum2, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      if(i_o == 0x01)
      {
        *pOut = bitins(out[0], n_mask4, out[1], mask4, off4);
        *pOut2 = bitins(out2[0], n_mask4, out2[1], mask4, off4);
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
        *pOut = ${config.relu_fn}(sum1, out_mult, out_shift);
        pOut++;
        *pOut2 = ${config.relu_fn}(sum2, out_mult, out_shift);
        pOut2++;
%elif config.kernel.out_data_t == 4:
        uint8_t i_o = i & 0x01;
        out[i_o] = ${config.relu_fn}(sum1, out_mult, out_shift);
        out2[i_o] = ${config.relu_fn}(sum2, out_mult, out_shift);
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask4, out[1], mask4, off4);
          *pOut2 = bitins(out2[0], n_mask4, out2[1], mask4, off4);
          pOut++;
          pOut2++;
        }
%endif
      }
      else
      {
%if config.kernel.out_data_t == 8:
        *pOut = (uint8_t) clip8(sum1 >> out_shift);
        pOut++;
        *pOut2 = (uint8_t) clip8(sum2 >> out_shift);
        pOut2++;
%elif config.kernel.out_data_t == 4:
        uint8_t i_o = i & 0x01;
        out[i_o] = (uint8_t) clip4(sum1 >> out_shift);
        out2[i_o] = (uint8_t) clip4(sum2 >> out_shift);
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask4, out[1], mask4, off4);
          *pOut2 = bitins(out2[0], n_mask4, out2[1], mask4, off4);
          pOut++;
          pOut2++;
        }
%endif
      }
    }
%elif config.kernel.out_data_t == 4:
    uint8_t i_o = i & 0x01;
    out[i_o] = pulp_nn_i4_quant(sum1, pThr);
    out2[i_o] = pulp_nn_i4_quant(sum2, pThr);
    pThr+=16;
    if(i_o == 0x01)
    {
      *pOut = bitins(out[0], n_mask4, out[1], mask4, off4);
      *pOut2 = bitins(out2[0], n_mask4, out2[1], mask4, off4);
      pOut++;
      pOut2++;
    }
%endif
%if config.kernel.out_data_t == 4:
    i++;
%endif
    chan_left--;
  }
%endif
  pOut+=ch_out_r;
  return pOut;
}
