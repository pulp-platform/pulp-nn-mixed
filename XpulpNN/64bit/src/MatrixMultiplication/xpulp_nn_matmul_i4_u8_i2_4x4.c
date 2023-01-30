/*
 * xpulp_nn_matmul_i4_u8_i2_4x4.c
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



uint8_t * __attribute__((noinline)) xpulp_nn_matmul_i4_u8_i2_4x4(
                        int8_t *pIn,
                        int8_t *pBias,
                        uint8_t *pOut,
                        uint8_t *pOut2,
                        uint8_t *pOut3,
                        uint8_t *pOut4,
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
  int32_t vecA[2];
  int32_t vecA2[2];
  int32_t vecA3[2];
  int32_t vecA4[2];

  uint16_t ch_out_r = PACK_INT8_SIZE(ch_out);

  uint16_t num_col_im2col_w = PACK_INT2_SIZE(num_col_im2col);
  uint16_t num_col_im2col_a = PACK_INT4_SIZE(num_col_im2col);

  int8_t *pA = pWeight;

  uint16_t chan_left = ch_out & 0x3;

  for(int i=0; i < (ch_out >> 2); i++)
  {
    int8_t *pB =  pIn;
    int8_t *pB2 = (pB + num_col_im2col_a);
    int8_t *pB3 = (pB2 + num_col_im2col_a);
    int8_t *pB4 = (pB3 + num_col_im2col_a);

    int32_t *ptrB  = (int32_t *) pB;
    int32_t *ptrB2 = (int32_t *) pB2;
    int32_t *ptrB3 = (int32_t *) pB3;
    int32_t *ptrB4 = (int32_t *) pB4;

    int8_t *pA2 = (pA + num_col_im2col_w);
    int8_t *pA3 = (pA2 + num_col_im2col_w);
    int8_t *pA4 = (pA3 + num_col_im2col_w);

    pA  = pulp_nn_i2_to_i4(pA , vecA);
    pA2 = pulp_nn_i2_to_i4(pA2, vecA2);
    pA3 = pulp_nn_i2_to_i4(pA3, vecA3);
    pA4 = pulp_nn_i2_to_i4(pA4, vecA4);

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

      sum9 = sum;
      sum10 = sum2;
      sum11 = sum3;
      sum12 = sum4;

      sum13 = sum;
      sum14 = sum2;
      sum15 = sum3;
      sum16 = sum4;
    }

    for(int j=0; j<(num_col_im2col >> 4); j++)
    {
      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = MacLoads8(0, 0, 0, 0, ptrA, sum);
      sum2 = MacLoads8(0, 0, 1, 0, ptrA2, sum2);
      sum3 = MacLoads8(0, 0, 2, 0, ptrA3, sum3);
      sum4 = MacLoads8(0, 1, 3, 0, ptrB3, sum4);
      ptrB3 = MacLoadUpdate(ptrB3);


      sum5 = MacLoads8(0, 0, 0, 1, ptrA, sum5);
      sum6 = MacLoads8(0, 0, 1, 1, ptrA2, sum6);
      sum7 = MacLoads8(0, 0, 2, 1, ptrA3, sum7);
      sum8 = MacLoads8(0, 1, 3, 1, ptrB4, sum8);
      ptrB4 = MacLoadUpdate(ptrB4);

      sum9  = MacLoads8(0, 0, 0, 0, ptrA, sum9);
      sum10 = MacLoads8(0, 0, 1, 0, ptrA2, sum10);
      sum11 = MacLoads8(0, 0, 2, 0, ptrA3, sum11);
      sum12 = MacLoads8(0, 1, 3, 0, ptrB, sum12);
      ptrB = MacLoadUpdate(ptrB);

      sum13 = MacLoads8(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);
      sum14 = MacLoads8(1, 0, 1, 1, ptrA2, sum14);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum15 = MacLoads8(1, 0, 2, 1, ptrA3, sum15);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum16 = MacLoads8(1, 0, 3, 1, ptrA4, sum16);
      ptrA4 = MacLoadUpdate(ptrA4);

      ptrB2  = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = MacLoads8(0, 0, 0, 0, ptrA, sum);
      sum2 = MacLoads8(0, 0, 1, 0, ptrA2, sum2);
      sum3 = MacLoads8(0, 0, 2, 0, ptrA3, sum3);
      sum4 = MacLoads8(0, 1, 3, 0, ptrB3, sum4);
      ptrB3 = MacLoadUpdate(ptrB3);

      sum5 = MacLoads8(0, 0, 0, 1, ptrA, sum5);
      sum6 = MacLoads8(0, 0, 1, 1, ptrA2, sum6);
      sum7 = MacLoads8(0, 0, 2, 1, ptrA3, sum7);
      sum8 = MacLoads8(0, 1, 3, 1, ptrB4, sum8);
      ptrB4 = MacLoadUpdate(ptrB4);

      pA  = pulp_nn_i2_to_i4(pA , vecA);
      pA2 = pulp_nn_i2_to_i4(pA2, vecA2);
      pA3 = pulp_nn_i2_to_i4(pA3, vecA3);
      pA4 = pulp_nn_i2_to_i4(pA4, vecA4);

      ptrA   = MacLoadAssign(vecA);
      ptrA2  = MacLoadAssign(vecA2);
      ptrA3  = MacLoadAssign(vecA3);
      ptrA4  = MacLoadAssign(vecA4);

      sum9  = MacLoads8(0, 0, 0, 0, ptrA, sum9);
      sum10 = MacLoads8(0, 0, 1, 0, ptrA2, sum10);
      sum11 = MacLoads8(0, 0, 2, 0, ptrA3, sum11);
      sum12 = MacLoads8(0, 1, 3, 0, ptrB, sum12);
      ptrB = MacLoadUpdate(ptrB);

      sum13 = MacLoads8(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);
      sum14 = MacLoads8(1, 0, 1, 1, ptrA2, sum14);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum15 = MacLoads8(1, 0, 2, 1, ptrA3, sum15);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum16 = MacLoads8(1, 0, 3, 1, ptrA4, sum16);
      ptrA4 = MacLoadUpdate(ptrA4);
    }
    pA-=4;
    pA2-=4;
    pA3-=4;
    pA4-=4;

    int col_cnt_im2col = num_col_im2col & 0xf;

    if(col_cnt_im2col)
    {

      uint16_t loop_cnt_im2col_a = (num_col_im2col >> 4) << 3;
      pB+=loop_cnt_im2col_a;
      pB2+=loop_cnt_im2col_a;
      pB3+=loop_cnt_im2col_a;
      pB4+=loop_cnt_im2col_a;

      do
      {
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA2, 2, 0);
        int8_t inA3 = (int8_t) bitext((int) *pA3, 2, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA4, 2, 0);

        int8_t inB = (int8_t)bitext((int32_t) *pB, 4, 0);
        int8_t inB2 = (int8_t)bitext((int32_t) *pB2, 4, 0);
        int8_t inB3 = (int8_t)bitext((int32_t) *pB3, 4, 0);
        int8_t inB4 = (int8_t)bitext((int32_t) *pB4, 4, 0);

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

        inA = (int8_t) bitext((int) *pA, 2, 2);
        inA2 = (int8_t) bitext((int) *pA2, 2, 2);
        inA3 = (int8_t) bitext((int) *pA3, 2, 2);
        inA4 = (int8_t) bitext((int) *pA4, 2, 2);

        inB = (int8_t)bitext((int32_t) *pB, 4, 4);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 4);
        inB3 = (int8_t)bitext((int32_t) *pB3, 4, 4);
        inB4 = (int8_t)bitext((int32_t) *pB4, 4, 4);

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

        pB++;
        pB2++;
        pB3++;
        pB4++;

        inA = (int8_t) bitext((int) *pA, 2, 4);
        inA2 = (int8_t) bitext((int) *pA2, 2, 4);
        inA3 = (int8_t) bitext((int) *pA3, 2, 4);
        inA4 = (int8_t) bitext((int) *pA4, 2, 4);

        inB = (int8_t)bitext((int32_t) *pB, 4, 0);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 0);
        inB3 = (int8_t)bitext((int32_t) *pB3, 4, 0);
        inB4 = (int8_t)bitext((int32_t) *pB4, 4, 0);

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

        inA = (int8_t) bitext((int) *pA, 2, 6);
        inA2 = (int8_t) bitext((int) *pA2, 2, 6);
        inA3 = (int8_t) bitext((int) *pA3, 2, 6);
        inA4 = (int8_t) bitext((int) *pA4, 2, 6);

        inB = (int8_t)bitext((int32_t) *pB, 4, 4);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 4);
        inB3 = (int8_t)bitext((int32_t) *pB3, 4, 4);
        inB4 = (int8_t)bitext((int32_t) *pB4, 4, 4);

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

        pA++;
        pA2++;
        pA3++;
        pA4++;

        pB++;
        pB2++;
        pB3++;
        pB4++;

        col_cnt_im2col-=4;
      } while(col_cnt_im2col > 0);
    }
    if (flag_batch_norm && flag_relu)
    {
      *pOut = pulp_nn_bn_quant_u8(sum, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum5, *pKappa, *pLambda, out_shift);
      pOut2++;
      *pOut3 = pulp_nn_bn_quant_u8(sum9, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = pulp_nn_bn_quant_u8(sum13, *pKappa, *pLambda, out_shift);
      pOut4++;
      pKappa++;
      pLambda++;

      *pOut = pulp_nn_bn_quant_u8(sum2, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum6, *pKappa, *pLambda, out_shift);
      pOut2++;
      *pOut3 = pulp_nn_bn_quant_u8(sum10, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = pulp_nn_bn_quant_u8(sum14, *pKappa, *pLambda, out_shift);
      pOut4++;
      pKappa++;
      pLambda++;

      *pOut = pulp_nn_bn_quant_u8(sum3, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum7, *pKappa, *pLambda, out_shift);
      pOut2++;
      *pOut3 = pulp_nn_bn_quant_u8(sum11, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = pulp_nn_bn_quant_u8(sum15, *pKappa, *pLambda, out_shift);
      pOut4++;
      pKappa++;
      pLambda++;

      *pOut = pulp_nn_bn_quant_u8(sum4, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum8, *pKappa, *pLambda, out_shift);
      pOut2++;
      *pOut3 = pulp_nn_bn_quant_u8(sum12, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = pulp_nn_bn_quant_u8(sum16, *pKappa, *pLambda, out_shift);
      pOut4++;
      pKappa++;
      pLambda++;
    }
    else
    {
      if (flag_relu == 1)
      {
        *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_u8(sum2, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_u8(sum3, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_u8(sum4, out_mult, out_shift);
        pOut++;

        *pOut2 = pulp_nn_quant_u8(sum5, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_u8(sum6, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_u8(sum7, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_u8(sum8, out_mult, out_shift);
        pOut2++;

        *pOut3 = pulp_nn_quant_u8(sum9, out_mult, out_shift);
        pOut3++;
        *pOut3 = pulp_nn_quant_u8(sum10, out_mult, out_shift);
        pOut3++;
        *pOut3 = pulp_nn_quant_u8(sum11, out_mult, out_shift);
        pOut3++;
        *pOut3 = pulp_nn_quant_u8(sum12, out_mult, out_shift);
        pOut3++;

        *pOut4 = pulp_nn_quant_u8(sum13, out_mult, out_shift);
        pOut4++;
        *pOut4 = pulp_nn_quant_u8(sum14, out_mult, out_shift);
        pOut4++;
        *pOut4 = pulp_nn_quant_u8(sum15, out_mult, out_shift);
        pOut4++;
        *pOut4 = pulp_nn_quant_u8(sum16, out_mult, out_shift);
        pOut4++;
      }
      else
      {
        *pOut = (uint8_t) clip8(sum >> out_shift);
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

        *pOut3 = (uint8_t) clip8(sum9 >> out_shift);
        pOut3++;
        *pOut3 = (uint8_t) clip8(sum10 >> out_shift);
        pOut3++;
        *pOut3 = (uint8_t) clip8(sum11 >> out_shift);
        pOut3++;
        *pOut3 = (uint8_t) clip8(sum12 >> out_shift);
        pOut3++;

        *pOut4 = (uint8_t) clip8(sum13 >> out_shift);
        pOut4++;
        *pOut4 = (uint8_t) clip8(sum14 >> out_shift);
        pOut4++;
        *pOut4 = (uint8_t) clip8(sum15 >> out_shift);
        pOut4++;
        *pOut4 = (uint8_t) clip8(sum16 >> out_shift);
        pOut4++;
      }
    }
    pA+=(3 * num_col_im2col_w);
  }
  while(chan_left)
  {
    int8_t *pB = pIn;
    int8_t *pB2 = (pB + num_col_im2col_a);
    int8_t *pB3 = (pB2 + num_col_im2col_a);
    int8_t *pB4 = (pB3 + num_col_im2col_a);

    int32_t *ptrB  = (int32_t *) pB;
    int32_t *ptrB2 = (int32_t *) pB2;
    int32_t *ptrB3 = (int32_t *) pB3;
    int32_t *ptrB4 = (int32_t *) pB4;

    pA  = pulp_nn_i2_to_i4(pA , vecA);

    int32_t *startA;

    asm volatile("mv %0, %1":"=r"(startA):"r"(vecA));

    int32_t *ptrA  = (int32_t *) vecA;

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

    int sum = 0;
    if (pBias != NULL)
    {
      sum = *((int*) pBias++);
    }
    int sum2 = sum;
    int sum3 = sum;
    int sum4 = sum;

    for(int j=0; j < (num_col_im2col >> 4); j++)
    {
      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = MacLoads8(0, 1, 0, 0, ptrB3, sum);
      ptrB3 = MacLoadUpdate(ptrB3);

      sum2 = MacLoads8(0, 1, 0, 1, ptrB4, sum2);
      ptrB4 = MacLoadUpdate(ptrB4);

      sum3 = MacLoads8(0, 1, 0, 0, ptrB, sum3);
      ptrB = MacLoadUpdate(ptrB);

      sum4 = MacLoads8(1, 0, 0, 1, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);

      ptrB2  = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = MacLoads8(0, 1, 0, 0, ptrB3, sum);
      ptrB3 = MacLoadUpdate(ptrB3);

      sum2 = MacLoads8(0, 1, 0, 1, ptrB4, sum2);
      ptrB4 = MacLoadUpdate(ptrB4);

      sum3 = MacLoads8(0, 1, 0, 0, ptrB, sum3);
      ptrB = MacLoadUpdate(ptrB);

      pA = pulp_nn_i2_to_i4(pA , vecA);

      ptrA = MacLoadAssign(vecA);

      sum4 = MacLoads8(1, 0, 0, 1, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);
    }
    pA-=4;
    int col_cnt_im2col = num_col_im2col & 0xf;

    if(col_cnt_im2col)
    {

      uint16_t loop_cnt_im2col_a = (num_col_im2col >> 4) << 3;
      pB+=loop_cnt_im2col_a;
      pB2+=loop_cnt_im2col_a;
      pB3+=loop_cnt_im2col_a;
      pB4+=loop_cnt_im2col_a;

      do
      {
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);

        int8_t inB = (int8_t)bitext((int32_t) *pB, 4, 0);
        int8_t inB2 = (int8_t)bitext((int32_t) *pB2, 4, 0);
        int8_t inB3 = (int8_t)bitext((int32_t) *pB3, 4, 0);
        int8_t inB4 = (int8_t)bitext((int32_t) *pB4, 4, 0);

        sum += inA * inB;

        sum2 += inA * inB2;

        sum3 += inA * inB3;

        sum4 += inA * inB4;

        inA = (int8_t) bitext((int) *pA, 2, 2);

        inB = (int8_t)bitext((int32_t) *pB, 4, 4);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 4);
        inB3 = (int8_t)bitext((int32_t) *pB3, 4, 4);
        inB4 = (int8_t)bitext((int32_t) *pB4, 4, 4);

        sum += inA * inB;

        sum2 += inA * inB2;

        sum3 += inA *inB3;

        sum4 += inA * inB4;

        pB++;
        pB2++;
        pB3++;
        pB4++;

        inA = (int8_t) bitext((int) *pA, 2, 4);

        inB = (int8_t)bitext((int32_t) *pB, 4, 0);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 0);
        inB3 = (int8_t)bitext((int32_t) *pB3, 4, 0);
        inB4 = (int8_t)bitext((int32_t) *pB4, 4, 0);

        sum += inA * inB;

        sum2 += inA * inB2;

        sum3 += inA * inB3;

        sum4 += inA * inB4;

        inA = (int8_t) bitext((int) *pA, 2, 6);

        inB = (int8_t)bitext((int32_t) *pB, 4, 4);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 4);
        inB3 = (int8_t)bitext((int32_t) *pB3, 4, 4);
        inB4 = (int8_t)bitext((int32_t) *pB4, 4, 4);

        sum += inA * inB;

        sum2 += inA * inB2;

        sum3 += inA * inB3;

        sum4 += inA * inB4;

        pA++;

        pB++;
        pB2++;
        pB3++;
        pB4++;

        col_cnt_im2col-=4;
      } while(col_cnt_im2col > 0);
    }
    if (flag_batch_norm && flag_relu)
    {
      *pOut = pulp_nn_bn_quant_u8(sum, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum2, *pKappa, *pLambda, out_shift);
      pOut2++;
      *pOut3 = pulp_nn_bn_quant_u8(sum3, *pKappa, *pLambda, out_shift);
      pOut3++;
      *pOut4 = pulp_nn_bn_quant_u8(sum4, *pKappa, *pLambda, out_shift);
      pOut4++;
      pKappa++;
      pLambda++;
    }
    else
    {
      if (flag_relu == 1)
      {
        *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
        pOut++;
        *pOut2 = pulp_nn_quant_u8(sum2, out_mult, out_shift);
        pOut2++;
        *pOut3 = pulp_nn_quant_u8(sum3, out_mult, out_shift);
        pOut3++;
        *pOut4 = pulp_nn_quant_u8(sum4, out_mult, out_shift);
        pOut4++;
      }
      else
      {
        *pOut = (uint8_t) clip8(sum >> out_shift);
        pOut++;
        *pOut2 = (uint8_t) clip8(sum2 >> out_shift);
        pOut2++;
        *pOut3 = (uint8_t) clip8(sum3 >> out_shift);
        pOut3++;
        *pOut4 = (uint8_t) clip8(sum4 >> out_shift);
        pOut4++;
      }
    }
    chan_left--;
  }
  pOut += 3 * ch_out_r;
  return pOut;
}
