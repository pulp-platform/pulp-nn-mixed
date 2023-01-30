/*
 * xpulp_nn_matmul_u8_i4_i8_4x4.c
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



uint8_t * __attribute__((noinline)) xpulp_nn_matmul_u8_i4_i8_4x4(
                        uint8_t *pIn,
                        int8_t *pBias,
                        int8_t *pOut,
                        int8_t *pOut2,
                        int8_t *pOut3,
                        int8_t *pOut4,
                        int8_t *pWeight,
                        int32_t *pKappa,
                        int32_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t num_col_im2col,
                        uint16_t ch_out,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  uint16_t ch_out_r = PACK_INT4_SIZE(ch_out);

  uint16_t num_col_im2col_w = PACK_INT8_SIZE(num_col_im2col);
  uint16_t num_col_im2col_a = PACK_INT8_SIZE(num_col_im2col);

  int8_t *pA = pWeight;

  uint16_t chan_left = ch_out & 0x3;

  for(int i=0; i < (ch_out >> 2); i++)
  {
    uint8_t *pB =  pIn;
    uint8_t *pB2 = (pB + num_col_im2col_a);
    uint8_t *pB3 = (pB2 + num_col_im2col_a);
    uint8_t *pB4 = (pB3 + num_col_im2col_a);

    uint32_t *ptrB  = (uint32_t *) pB;
    uint32_t *ptrB2 = (uint32_t *) pB2;
    uint32_t *ptrB3 = (uint32_t *) pB3;
    uint32_t *ptrB4 = (uint32_t *) pB4;

    int8_t *pA2 = (pA + num_col_im2col_w);
    int8_t *pA3 = (pA2 + num_col_im2col_w);
    int8_t *pA4 = (pA3 + num_col_im2col_w);

    int32_t *ptrA  = (int32_t *) pA ;
    int32_t *ptrA2 = (int32_t *) pA2;
    int32_t *ptrA3 = (int32_t *) pA3;
    int32_t *ptrA4 = (int32_t *) pA4;

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

    for(int j=0; j<(num_col_im2col >> 2); j++)
    {
      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = MacLoad4(0, 0, 0, 0, ptrA, sum);
      sum2 = MacLoad4(0, 0, 1, 0, ptrA2, sum2);
      sum3 = MacLoad4(0, 0, 2, 0, ptrA3, sum3);
      sum4 = MacLoad4(0, 1, 3, 0, ptrB3, sum4);
      ptrB3 = MacLoadUpdate(ptrB3);


      sum5 = MacLoad4(0, 0, 0, 1, ptrA, sum5);
      sum6 = MacLoad4(0, 0, 1, 1, ptrA2, sum6);
      sum7 = MacLoad4(0, 0, 2, 1, ptrA3, sum7);
      sum8 = MacLoad4(0, 1, 3, 1, ptrB4, sum8);
      ptrB4 = MacLoadUpdate(ptrB4);

      sum9  = MacLoad4(0, 0, 0, 0, ptrA, sum9);
      sum10 = MacLoad4(0, 0, 1, 0, ptrA2, sum10);
      sum11 = MacLoad4(0, 0, 2, 0, ptrA3, sum11);
      sum12 = MacLoad4(0, 1, 3, 0, ptrB, sum12);
      ptrB = MacLoadUpdate(ptrB);

      sum13 = MacLoad4(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);
      sum14 = MacLoad4(1, 0, 1, 1, ptrA2, sum14);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum15 = MacLoad4(1, 0, 2, 1, ptrA3, sum15);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum16 = MacLoad4(1, 0, 3, 1, ptrA4, sum16);
      ptrA4 = MacLoadUpdate(ptrA4);

    }

    int col_cnt_im2col = num_col_im2col & 0x3;

    if(col_cnt_im2col)
    {
      uint16_t loop_cnt_im2col_w = (num_col_im2col >> 2) << 2;
      pA+=loop_cnt_im2col_w;
      pA2+=loop_cnt_im2col_w;
      pA3+=loop_cnt_im2col_w;
      pA4+=loop_cnt_im2col_w;

      uint16_t loop_cnt_im2col_a = (num_col_im2col >> 2) << 2;
      pB+=loop_cnt_im2col_a;
      pB2+=loop_cnt_im2col_a;
      pB3+=loop_cnt_im2col_a;
      pB4+=loop_cnt_im2col_a;

      do
      {
        int8_t inA = *pA++;
        int8_t inA2 = *pA2++;
        int8_t inA3 = *pA3++;
        int8_t inA4 = *pA4++;

        uint8_t inB = *pB++;
        uint8_t inB2 = *pB2++;
        uint8_t inB3 = *pB3++;
        uint8_t inB4 = *pB4++;
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
      } while(col_cnt_im2col > 0);
      pA-=num_col_im2col_w;
    }
    if (flag_batch_norm && flag_relu)
    {
      sum = pulp_nn_bn_quant_i4(sum, *pKappa, *pLambda, out_shift);
      sum5 = pulp_nn_bn_quant_i4(sum5, *pKappa, *pLambda, out_shift);
      sum9 = pulp_nn_bn_quant_i4(sum9, *pKappa, *pLambda, out_shift);
      sum13 = pulp_nn_bn_quant_i4(sum13, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum2 = pulp_nn_bn_quant_i4(sum2, *pKappa, *pLambda, out_shift);
      sum6 = pulp_nn_bn_quant_i4(sum6, *pKappa, *pLambda, out_shift);
      sum10 = pulp_nn_bn_quant_i4(sum10, *pKappa, *pLambda, out_shift);
      sum14 = pulp_nn_bn_quant_i4(sum14, *pKappa, *pLambda, out_shift);
      *pOut = bitins(sum, n_mask, sum2, mask, off);
      *pOut2 = bitins(sum5, n_mask, sum6, mask, off);
      *pOut3 = bitins(sum9, n_mask, sum10, mask, off);
      *pOut4 = bitins(sum13, n_mask, sum14, mask, off);
      pKappa++;
      pLambda++;
      pOut++;
      pOut2++;
      pOut3++;
      pOut4++;
      sum3 = pulp_nn_bn_quant_i4(sum3, *pKappa, *pLambda, out_shift);
      sum7 = pulp_nn_bn_quant_i4(sum7, *pKappa, *pLambda, out_shift);
      sum11 = pulp_nn_bn_quant_i4(sum11, *pKappa, *pLambda, out_shift);
      sum15 = pulp_nn_bn_quant_i4(sum15, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum4 = pulp_nn_bn_quant_i4(sum4, *pKappa, *pLambda, out_shift);
      sum8 = pulp_nn_bn_quant_i4(sum8, *pKappa, *pLambda, out_shift);
      sum12 = pulp_nn_bn_quant_i4(sum12, *pKappa, *pLambda, out_shift);
      sum16 = pulp_nn_bn_quant_i4(sum16, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      *pOut = bitins(sum3, n_mask, sum4, mask, off);
      *pOut2 = bitins(sum7, n_mask, sum8, mask, off);
      *pOut3 = bitins(sum11, n_mask, sum12, mask, off);
      *pOut4 = bitins(sum15, n_mask, sum16, mask, off);
      pOut++;
      pOut2++;
      pOut3++;
      pOut4++;
    }
    else
    {
      if (flag_relu == 1)
      {
        sum = pulp_nn_quant_i4(sum, out_mult, out_shift);
        sum2 = pulp_nn_quant_i4(sum2, out_mult, out_shift);
        *pOut = bitins(sum, n_mask, sum2, mask, off);
        pOut++;
        sum3 = pulp_nn_quant_i4(sum3, out_mult, out_shift);
        sum4 = pulp_nn_quant_i4(sum4, out_mult, out_shift);
        *pOut = bitins(sum3, n_mask, sum4, mask, off);
        pOut++;

        sum5 = pulp_nn_quant_i4(sum5, out_mult, out_shift);
        sum6 = pulp_nn_quant_i4(sum6, out_mult, out_shift);
        *pOut2 = bitins(sum5, n_mask, sum6, mask, off);
        pOut2++;
        sum7 = pulp_nn_quant_i4(sum7, out_mult, out_shift);
        sum8 = pulp_nn_quant_i4(sum8, out_mult, out_shift);
        *pOut2 = bitins(sum7, n_mask, sum8, mask, off);
        pOut2++;

        sum9 = pulp_nn_quant_i4(sum9, out_mult, out_shift);
        sum10 = pulp_nn_quant_i4(sum10, out_mult, out_shift);
        *pOut3 = bitins(sum9, n_mask, sum10, mask, off);
        pOut3++;
        sum11 = pulp_nn_quant_i4(sum11, out_mult, out_shift);
        sum12 = pulp_nn_quant_i4(sum12, out_mult, out_shift);
        *pOut3 = bitins(sum11, n_mask, sum12, mask, off);
        pOut3++;

        sum13 = pulp_nn_quant_i4(sum13, out_mult, out_shift);
        sum14 = pulp_nn_quant_i4(sum14, out_mult, out_shift);
        *pOut4 = bitins(sum13, n_mask, sum14, mask, off);
        pOut4++;
        sum15 = pulp_nn_quant_i4(sum15, out_mult, out_shift);
        sum16 = pulp_nn_quant_i4(sum16, out_mult, out_shift);
        *pOut4 = bitins(sum15, n_mask, sum16, mask, off);
        pOut4++;
      }
      else
      {
        sum = (int8_t) clips4(sum >> out_shift);
        sum2 = (int8_t) clips4(sum2 >> out_shift);
        *pOut = bitins(sum, n_mask, sum2, mask, off);
        pOut++;
        sum3 = (int8_t) clips4(sum3 >> out_shift);
        sum4 = (int8_t) clips4(sum4 >> out_shift);
        *pOut = bitins(sum3, n_mask, sum4, mask, off);
        pOut++;

        sum5 = (int8_t) clips4(sum5 >> out_shift);
        sum6 = (int8_t) clips4(sum6 >> out_shift);
        *pOut2 = bitins(sum5, n_mask, sum6, mask, off);
        pOut2++;
        sum7 = (int8_t) clips4(sum7 >> out_shift);
        sum8 = (int8_t) clips4(sum8 >> out_shift);
        *pOut2 = bitins(sum7, n_mask, sum8, mask, off);
        pOut2++;

        sum9 = (int8_t) clips4(sum9 >> out_shift);
        sum10 = (int8_t) clips4(sum10 >> out_shift);
        *pOut3 = bitins(sum9, n_mask, sum10, mask, off);
        pOut3++;
        sum11 = (int8_t) clips4(sum11 >> out_shift);
        sum12 = (int8_t) clips4(sum12 >> out_shift);
        *pOut3 = bitins(sum11, n_mask, sum12, mask, off);
        pOut3++;

        sum13 = (int8_t) clips4(sum13 >> out_shift);
        sum14 = (int8_t) clips4(sum14 >> out_shift);
        *pOut4 = bitins(sum13, n_mask, sum14, mask, off);
        pOut4++;
        sum15 = (int8_t) clips4(sum15 >> out_shift);
        sum16 = (int8_t) clips4(sum16 >> out_shift);
        pOut4++;
      }
    }
    pA+=(4 * num_col_im2col_w);
  }
  int i = 0;
  while(chan_left)
  {
    uint8_t *pB = pIn;
    uint8_t *pB2 = (pB + num_col_im2col_a);
    uint8_t *pB3 = (pB2 + num_col_im2col_a);
    uint8_t *pB4 = (pB3 + num_col_im2col_a);

    uint32_t *ptrB  = (uint32_t *) pB;
    uint32_t *ptrB2 = (uint32_t *) pB2;
    uint32_t *ptrB3 = (uint32_t *) pB3;
    uint32_t *ptrB4 = (uint32_t *) pB4;

    int32_t *ptrA  = (int32_t *) pA;

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

    int8_t out[2];
    int8_t out2[2];
    int8_t out3[2];
    int8_t out4[2];
    for(int j=0; j < (num_col_im2col >> 2); j++)
    {
      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = MacLoad4(0, 1, 0, 0, ptrB3, sum);
      ptrB3 = MacLoadUpdate(ptrB3);

      sum2 = MacLoad4(0, 1, 0, 1, ptrB4, sum2);
      ptrB4 = MacLoadUpdate(ptrB4);

      sum3 = MacLoad4(0, 1, 0, 0, ptrB, sum3);
      ptrB = MacLoadUpdate(ptrB);

      sum4 = MacLoad4(1, 0, 0, 1, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);

    }
    int col_cnt_im2col = num_col_im2col & 0x3;

    if(col_cnt_im2col)
    {
      uint16_t loop_cnt_im2col_w = (num_col_im2col >> 2) << 2;
      pA+=loop_cnt_im2col_w;

      uint16_t loop_cnt_im2col_a = (num_col_im2col >> 2) << 2;
      pB+=loop_cnt_im2col_a;
      pB2+=loop_cnt_im2col_a;
      pB3+=loop_cnt_im2col_a;
      pB4+=loop_cnt_im2col_a;

      do
      {
        int8_t inA = *pA++;

        uint8_t inB = *pB++;
        uint8_t inB2 = *pB2++;
        uint8_t inB3 = *pB3++;
        uint8_t inB4 = *pB4++;
        asm volatile("": : :"memory");
        sum += inA * inB;

        sum2 += inA * inB2;

        sum3 += inA * inB3;

        sum4 += inA * inB4;

        col_cnt_im2col--;
      } while(col_cnt_im2col > 0);
      pA-=num_col_im2col_w;
    }
    if (flag_batch_norm && flag_relu)
    {
      uint8_t i_o = i & 0x01;
      out[i_o] = pulp_nn_bn_quant_i4(sum, *pKappa, *pLambda, out_shift);
      out2[i_o] = pulp_nn_bn_quant_i4(sum2, *pKappa, *pLambda, out_shift);
      out3[i_o] = pulp_nn_bn_quant_i4(sum3, *pKappa, *pLambda, out_shift);
      out4[i_o] = pulp_nn_bn_quant_i4(sum4, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      if(i_o == 0x01)
      {
        *pOut = bitins(out[0], n_mask, out[1], mask, off);
        *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
        *pOut3 = bitins(out3[0], n_mask, out3[1], mask, off);
        *pOut4 = bitins(out4[0], n_mask, out4[1], mask, off);
        pOut++;
        pOut2++;
        pOut3++;
        pOut4++;
      }
    }
    else
    {
      if (flag_relu == 1)
      {
        uint8_t i_o = i & 0x01;
        out[i_o] = pulp_nn_quant_i4(sum, out_mult, out_shift);
        out2[i_o] = pulp_nn_quant_i4(sum2, out_mult, out_shift);
        out3[i_o] = pulp_nn_quant_i4(sum3, out_mult, out_shift);
        out4[i_o] = pulp_nn_quant_i4(sum4, out_mult, out_shift);
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask, out[1], mask, off);
          *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
          *pOut3 = bitins(out3[0], n_mask, out3[1], mask, off);
          *pOut4 = bitins(out4[0], n_mask, out4[1], mask, off);
          pOut++;
          pOut2++;
          pOut3++;
          pOut4++;
        }
      }
      else
      {
        uint8_t i_o = i & 0x01;
        out[i_o] = (int8_t) clips4(sum >> out_shift);
        out2[i_o] = (int8_t) clips4(sum2 >> out_shift);
        out3[i_o] = (int8_t) clips4(sum3 >> out_shift);
        out4[i_o] = (int8_t) clips4(sum4 >> out_shift);
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask, out[1], mask, off);
          *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
          *pOut3 = bitins(out3[0], n_mask, out3[1], mask, off);
          *pOut4 = bitins(out4[0], n_mask, out4[1], mask, off);
          pOut++;
          pOut2++;
          pOut3++;
          pOut4++;
        }
      }
    }
    i++;
    pA+=num_col_im2col_w;
    chan_left--;
  }
  pOut += 3 * ch_out_r;
  return pOut;
}
