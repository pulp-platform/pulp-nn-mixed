/*
 * xpulp_nn_mix_matmul_i4_i8_i8_4x4.c
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




uint8_t * __attribute__((noinline)) xpulp_nn_mix_matmul_i4_i8_i8_4x4(
                        int8_t *pIn,
                        int8_t *pBias,
                        int8_t *pOut,
                        int8_t *pOut2,
                        int8_t *pOut3,
                        int8_t *pOut4,
                        int8_t *pWeight,
                        int64_t *pKappa,
                        int64_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t num_col_im2col,
                        uint16_t ch_out,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{

  uint16_t ch_out_r = PACK_INT8_SIZE(ch_out);

  uint16_t num_col_im2col_w = PACK_INT8_SIZE(num_col_im2col);
  uint16_t num_col_im2col_a = PACK_INT8_SIZE(num_col_im2col);

  int32_t a_rollback = 4 - (num_col_im2col_a + (num_col_im2col_a << 1));
  int32_t w_rollback = 4 - (num_col_im2col_w + (num_col_im2col_w << 1));

  A_STRIDE(num_col_im2col_a);
  W_STRIDE(num_col_im2col_w);
  A_ROLLBACK(a_rollback);
  W_ROLLBACK(w_rollback);

  int8_t *pA = pWeight;

  uint16_t chan_left = ch_out & 0x3;

  for(int i=0; i < (ch_out >> 2); i++)
  {
    int8_t *pB =  pIn;

    int32_t *ptrB  = (int32_t *) pB;

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

    int sum9  = 0;
    int sum10 = 0;
    int sum11 = 0;
    int sum12 = 0;
    int sum13 = 0;
    int sum14 = 0;
    int sum15 = 0;
    int sum16 = 0;

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

      sum9 = sum;
      sum10 = sum2;
      sum11 = sum3;
      sum12 = sum4;

      sum13 = sum;
      sum14 = sum2;
      sum15 = sum3;
      sum16 = sum4;
    }

    for(int j=0; j<(num_col_im2col >> 2); j++)
    {
      ptrB = MacLoadInit(0, 1, 0, 1, ptrB);

      sum  = MacLoads4(0, 0, 0, 0, ptrA, sum);
      sum2 = MacLoads4(0, 0, 1, 0, ptrA, sum2);
      sum3 = MacLoads4(0, 0, 2, 0, ptrA, sum3);
      sum4 = MacLoads4(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);

      sum5 = MacLoads4(0, 0, 0, 1, ptrA, sum5);
      sum6 = MacLoads4(0, 0, 1, 1, ptrA, sum6);
      sum7 = MacLoads4(0, 0, 2, 1, ptrA, sum7);
      sum8 = MacLoads4(0, 1, 3, 1, ptrB, sum8);
      ptrB = MacLoadUpdate(ptrB);

      sum9  = MacLoads4(0, 0, 0, 0, ptrA, sum9);
      sum10 = MacLoads4(0, 0, 1, 0, ptrA, sum10);
      sum11 = MacLoads4(0, 0, 2, 0, ptrA, sum11);
      sum12 = MacLoads4(0, 1, 3, 0, ptrB, sum12);
      ptrB  = MacLoadUpdate(ptrB);

      sum13 = MacLoads4(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);

      sum14 = MacLoads4(1, 0, 1, 1, ptrA, sum14);
      ptrA  = MacLoadUpdate(ptrA);

      sum15 = MacLoads4(1, 0, 2, 1, ptrA, sum15);
      ptrA  = MacLoadUpdate(ptrA);

      sum16 = MacLoads4(1, 0, 3, 1, ptrA, sum16);
      ptrA  = MacLoadUpdate(ptrA);

    }

    
    int col_cnt_im2col = num_col_im2col & 0x3;

    if(col_cnt_im2col)
    {
      uint16_t loop_cnt_im2col_w = (num_col_im2col >> 2) << 2;
      pA+=loop_cnt_im2col_w;

      uint16_t loop_cnt_im2col_a = (num_col_im2col >> 2) << 2;
      
      int8_t *pA2 = (pA  + num_col_im2col_w);
      int8_t *pA3 = (pA2 + num_col_im2col_w);
      int8_t *pA4 = (pA3 + num_col_im2col_w);

      pB+=loop_cnt_im2col_a;
      
      int8_t *pB2 = (pB + num_col_im2col_a);
      int8_t *pB3 = (pB2 + num_col_im2col_a);
      int8_t *pB4 = (pB3 + num_col_im2col_a);

      do
      {
        int8_t inA = *pA++;
        int8_t inA2 = *pA2++;
        int8_t inA3 = *pA3++;
        int8_t inA4 = *pA4++;

        int8_t inB = *pB++;
        int8_t inB2 = *pB2++;
        int8_t inB3 = *pB3++;
        int8_t inB4 = *pB4++;
        asm volatile("": : :"memory");
        sum += inA * inB;
        sum2 += inA2 * inB;
        sum3 += inA3 * inB;
        sum4 += inA4 * inB;

        sum5 += inA * inB2;
        sum6 += inA2 * inB2;
        sum7 += inA3 * inB2;
        sum8 += inA4 * inB2;

        sum9 += inA * inB3;
        sum10 += inA2 * inB3;
        sum11 += inA3 * inB3;
        sum12 += inA4 * inB3;

        sum13 += inA * inB4;
        sum14 += inA2 * inB4;
        sum15 += inA3 * inB4;
        sum16 += inA4 * inB4;

        col_cnt_im2col--;
      } while(col_cnt_im2col);
      pA-=num_col_im2col_w;
    }
    if (flag_batch_norm && flag_relu)
    {
      *pOut = pulp_nn_bn_quant_i8(sum, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_i8(sum5, *pKappa, *pLambda, out_shift);
      pOut2++;
      *pOut3 = pulp_nn_bn_quant_i8(sum9, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = pulp_nn_bn_quant_i8(sum13, *pKappa, *pLambda, out_shift);
      pOut4++;
      pKappa++;
      pLambda++;

      *pOut = pulp_nn_bn_quant_i8(sum2, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_i8(sum6, *pKappa, *pLambda, out_shift);
      pOut2++;
      *pOut3 = pulp_nn_bn_quant_i8(sum10, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = pulp_nn_bn_quant_i8(sum14, *pKappa, *pLambda, out_shift);
      pOut4++;
      pKappa++;
      pLambda++;

      *pOut = pulp_nn_bn_quant_i8(sum3, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_i8(sum7, *pKappa, *pLambda, out_shift);
      pOut2++;
      *pOut3 = pulp_nn_bn_quant_i8(sum11, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = pulp_nn_bn_quant_i8(sum15, *pKappa, *pLambda, out_shift);
      pOut4++;
      pKappa++;
      pLambda++;

      *pOut = pulp_nn_bn_quant_i8(sum4, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_i8(sum8, *pKappa, *pLambda, out_shift);
      pOut2++;
      *pOut3 = pulp_nn_bn_quant_i8(sum12, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = pulp_nn_bn_quant_i8(sum16, *pKappa, *pLambda, out_shift);
      pOut4++;
      pKappa++;
      pLambda++;
    }
    else
    {
      if (flag_relu == 1)
      {
        *pOut = pulp_nn_quant_i8(sum, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_i8(sum2, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_i8(sum3, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_i8(sum4, out_mult, out_shift);
        pOut++;

        *pOut2 = pulp_nn_quant_i8(sum5, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_i8(sum6, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_i8(sum7, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_i8(sum8, out_mult, out_shift);
        pOut2++;

        *pOut3 = pulp_nn_quant_i8(sum9, out_mult, out_shift);
        pOut3++;
        *pOut3 = pulp_nn_quant_i8(sum10, out_mult, out_shift);
        pOut3++;
        *pOut3 = pulp_nn_quant_i8(sum11, out_mult, out_shift);
        pOut3++;
        *pOut3 = pulp_nn_quant_i8(sum12, out_mult, out_shift);
        pOut3++;

        *pOut4 = pulp_nn_quant_i8(sum13, out_mult, out_shift);
        pOut4++;
        *pOut4 = pulp_nn_quant_i8(sum14, out_mult, out_shift);
        pOut4++;
        *pOut4 = pulp_nn_quant_i8(sum15, out_mult, out_shift);
        pOut4++;
        *pOut4 = pulp_nn_quant_i8(sum16, out_mult, out_shift);
        pOut4++;
      }
      else
      {
        *pOut = (int8_t) clips8(sum >> out_shift);
        pOut++;
        *pOut = (int8_t) clips8(sum2 >> out_shift);
        pOut++;
        *pOut = (int8_t) clips8(sum3 >> out_shift);
        pOut++;
        *pOut = (int8_t) clips8(sum4 >> out_shift);
        pOut++;

        *pOut2 = (int8_t) clips8(sum5 >> out_shift);
        pOut2++;
        *pOut2 = (int8_t) clips8(sum6 >> out_shift);
        pOut2++;
        *pOut2 = (int8_t) clips8(sum7 >> out_shift);
        pOut2++;
        *pOut2 = (int8_t) clips8(sum8 >> out_shift);
        pOut2++;

        *pOut3 = (int8_t) clips8(sum9 >> out_shift);
        pOut3++;
        *pOut3 = (int8_t) clips8(sum10 >> out_shift);
        pOut3++;
        *pOut3 = (int8_t) clips8(sum11 >> out_shift);
        pOut3++;
        *pOut3 = (int8_t) clips8(sum12 >> out_shift);
        pOut3++;

        *pOut4 = (int8_t) clips8(sum13 >> out_shift);
        pOut4++;
        *pOut4 = (int8_t) clips8(sum14 >> out_shift);
        pOut4++;
        *pOut4 = (int8_t) clips8(sum15 >> out_shift);
        pOut4++;
        *pOut4 = (int8_t) clips8(sum16 >> out_shift);
        pOut4++;
      }
    }
    pA+=(4 * num_col_im2col_w);
  }

  W_ROLLBACK(4);
  W_SKIP("0");
  MIXED_SKIP("4");

  while(chan_left)
  {
    int8_t *pB = pIn;

    int8_t *pA = pWeight + (num_col_im2col_w * (ch_out - chan_left));

    int32_t *ptrB  = (int32_t *) pB;

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
    int sum3 = sum;
    int sum4 = sum;

    for(int j=0; j < (num_col_im2col >> 2); j++)
    {
      ptrB = MacLoadInit(0, 1, 0, 1, ptrB);

      sum  = MacLoads4(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

      sum2 = MacLoads4(0, 1, 0, 1, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);

      sum3 = MacLoads4(0, 1, 0, 0, ptrB, sum3);
      ptrB = MacLoadUpdate(ptrB);

      sum4 = MacLoads4(1, 0, 0, 1, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);
    }
    int col_cnt_im2col = num_col_im2col & 0x3;

    if(col_cnt_im2col)
    {
      uint16_t loop_cnt_im2col_w = (num_col_im2col >> 2) << 2;
      pA+=loop_cnt_im2col_w;

      uint16_t loop_cnt_im2col_a = (num_col_im2col >> 2) << 2;
      pB+=loop_cnt_im2col_a;
      
      int8_t *pB2 = (pB + num_col_im2col_a);
      int8_t *pB3 = (pB2 + num_col_im2col_a);
      int8_t *pB4 = (pB3 + num_col_im2col_a);

      int8_t *pA2 = (pA  + num_col_im2col_w);
      int8_t *pA3 = (pA2 + num_col_im2col_w);
      int8_t *pA4 = (pA3 + num_col_im2col_w);

      do
      {
        int8_t inA = *pA++;

        int8_t inB = *pB++;
        int8_t inB2 = *pB2++;
        int8_t inB3 = *pB3++;
        int8_t inB4 = *pB4++;
        asm volatile("": : :"memory");
        sum += inA * inB;

        sum2 += inA * inB2;

        sum3 += inA * inB3;

        sum4 += inA * inB4;

        col_cnt_im2col--;
      } while(col_cnt_im2col);
      pA-=num_col_im2col_w;
    }
    if (flag_batch_norm && flag_relu)
    {
      *pOut = pulp_nn_bn_quant_i8(sum, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_i8(sum2, *pKappa, *pLambda, out_shift);
      pOut2++;
      *pOut3 = pulp_nn_bn_quant_i8(sum3, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = pulp_nn_bn_quant_i8(sum4, *pKappa, *pLambda, out_shift);
      pOut4++;
      pKappa++;
      pLambda++;
    }
    else
    {
      if (flag_relu == 1)
      {
        *pOut = pulp_nn_quant_i8(sum, out_mult, out_shift);
        pOut++;
        *pOut2 = pulp_nn_quant_i8(sum2, out_mult, out_shift);
        pOut2++;
        *pOut3 = pulp_nn_quant_i8(sum3, out_mult, out_shift);
        pOut3++;
        *pOut4 = pulp_nn_quant_i8(sum4, out_mult, out_shift);
        pOut4++;
      }
      else
      {
        *pOut = (int8_t) clips8(sum >> out_shift);
        pOut++;
        *pOut2 = (int8_t) clips8(sum2 >> out_shift);
        pOut2++;
        *pOut3 = (int8_t) clips8(sum3 >> out_shift);
        pOut3++;
        *pOut4 = (int8_t) clips8(sum4 >> out_shift);
        pOut4++;
      }
    }
    pA+=num_col_im2col_w;
    chan_left--;
  }
  W_SKIP("3");
  MIXED_SKIP("16");
  pOut += 3 * ch_out_r;
  return pOut;
}
