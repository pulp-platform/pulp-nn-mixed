/*
 * xpulp_nn_mix_matmul_i4_i4_i2.c
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




uint8_t * __attribute__((noinline)) xpulp_nn_mix_matmul_i4_i4_i2(
                        int8_t *pIn,
                        int8_t *pBias,
                        int8_t *pOut,
                        int8_t *pOut2,
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
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  uint16_t ch_out_r = PACK_INT4_SIZE(ch_out);

  uint16_t num_col_im2col_w = PACK_INT2_SIZE(num_col_im2col);
  uint16_t num_col_im2col_a = PACK_INT4_SIZE(num_col_im2col);

  int32_t a_rollback = 4 - num_col_im2col_a;
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


    if (pBias != NULL)
    {
      sum =  *((int*) pBias);
      pBias += 4;
      sum2 = *((int*) pBias);
      pBias += 4;
      sum3 = *((int*) pBias);
      pBias += 4;
      sum4 = *((int*) pBias);
      pBias += 4;

      sum5 = sum;
      sum6 = sum2;
      sum7 = sum3;
      sum8 = sum4;

    }

    for(int j=0; j<(num_col_im2col >> 4); j++)
    {
      ptrB  = MacLoadInit(0, 1, 0, 1, ptrB);

      sum  = MacLoads8(0, 0, 0, 0, ptrA, sum);
      sum2 = MacLoads8(0, 0, 1, 0, ptrA, sum2);
      sum3 = MacLoads8(0, 0, 2, 0, ptrA, sum3);
      sum4 = MacLoads8(0, 1, 3, 0, ptrB, sum4);
      ptrB = MacLoadUpdate(ptrB);

      sum5 = MacLoads8(0, 0, 0, 1, ptrA,  sum5);
      sum6 = MacLoads8(0, 0, 1, 1, ptrA, sum6);
      sum7 = MacLoads8(0, 0, 2, 1, ptrA, sum7);
      sum8 = MacLoads8(0, 1, 3, 1, ptrB, sum8);
      ptrB = MacLoadUpdate(ptrB);


      MemoryFence();

      sum  = MacLoads8(0, 0, 0, 0, ptrA, sum);
      sum2 = MacLoads8(0, 0, 1, 0, ptrA, sum2);
      sum3 = MacLoads8(0, 0, 2, 0, ptrA, sum3);
      sum4 = MacLoads8(0, 1, 3, 0, ptrB, sum4);      
      ptrB = MacLoadUpdate(ptrB);

      sum5 = MacLoads8(1, 0, 0, 1, ptrA, sum5);
      ptrA = MacLoadUpdate(ptrA);

      sum6 = MacLoads8(1, 0, 1, 1, ptrA, sum6);
      ptrA = MacLoadUpdate(ptrA);

      sum7 = MacLoads8(1, 0, 2, 1, ptrA, sum7);
      ptrA = MacLoadUpdate(ptrA);

      sum8 = MacLoads8(1, 0, 3, 1, ptrA, sum8);
      ptrA = MacLoadUpdate(ptrA);
    }

    asm volatile ("csrr %0, 0x101" : "=r" (pA));
    pA-=4;
    
    int col_cnt_im2col = num_col_im2col & 0xf;

    if(col_cnt_im2col)
    {

      uint16_t loop_cnt_im2col_a = (num_col_im2col >> 4) << 3;
      
      int8_t *pA2 = (pA  + num_col_im2col_w);
      int8_t *pA3 = (pA2 + num_col_im2col_w);
      int8_t *pA4 = (pA3 + num_col_im2col_w);

      pB+=loop_cnt_im2col_a;
      
      int8_t *pB2 = (pB + num_col_im2col_a);

      do
      {
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA2, 2, 0);
        int8_t inA3 = (int8_t) bitext((int) *pA3, 2, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA4, 2, 0);

        int8_t inB = (int8_t)bitext((int32_t) *pB, 4, 0);
        int8_t inB2 = (int8_t)bitext((int32_t) *pB2, 4, 0);

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

        inB = (int8_t)bitext((int32_t) *pB, 4, 4);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 4);

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

        inB = (int8_t)bitext((int32_t) *pB, 4, 0);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 0);

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

        inB = (int8_t)bitext((int32_t) *pB, 4, 4);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 4);

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
      } while(col_cnt_im2col);
    }
    if (flag_batch_norm && flag_relu)
    {
      sum   = pulp_nn_bn_quant_i4(sum, *pKappa, *pLambda, out_shift);
      sum5  = pulp_nn_bn_quant_i4(sum5, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum2  = pulp_nn_bn_quant_i4(sum2, *pKappa, *pLambda, out_shift);
      sum6  = pulp_nn_bn_quant_i4(sum6, *pKappa, *pLambda, out_shift);
      *pOut = bitins(sum, n_mask, sum2, mask, off);
      *pOut2 = bitins(sum5, n_mask, sum6, mask, off);
      pKappa++;
      pLambda++;
      pOut++;
      pOut2++;
      sum3 = pulp_nn_bn_quant_i4(sum3, *pKappa, *pLambda, out_shift);
      sum7 = pulp_nn_bn_quant_i4(sum7, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum4 = pulp_nn_bn_quant_i4(sum4, *pKappa, *pLambda, out_shift);
      sum8 = pulp_nn_bn_quant_i4(sum8, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      *pOut = bitins(sum3, n_mask, sum4, mask, off);
      *pOut2 = bitins(sum7, n_mask, sum8, mask, off);
      pOut++;
      pOut2++;
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

      }
    }
    pA+=(3 * num_col_im2col_w);
  }
  int i = 0;

  if(chan_left != 0){
    W_ROLLBACK(4);
    W_SKIP("0");
    MIXED_SKIP("2");
  }

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

    int8_t out[2];
    int8_t out2[2];
    for(int j=0; j < (num_col_im2col >> 4); j++)
    {
      ptrB  = MacLoadInit(0, 1, 0, 1, ptrB);

      sum  = MacLoads8(0, 1, 0, 0, ptrB, sum);
      ptrB = MacLoadUpdate(ptrB);

      sum2 = MacLoads8(0, 1, 0, 1, ptrB, sum2);
      ptrB = MacLoadUpdate(ptrB);


      sum  = MacLoads8(0, 1, 0, 0, ptrB, sum);   
      ptrB = MacLoadUpdate(ptrB);

      sum2 = MacLoads8(1, 0, 0, 1, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
    }
    asm volatile ("csrr %0, 0x101" : "=r" (pA));
    pA-=4;
    int col_cnt_im2col = num_col_im2col & 0xf;

    if(col_cnt_im2col)
    {

      uint16_t loop_cnt_im2col_a = (num_col_im2col >> 4) << 3;
      pB+=loop_cnt_im2col_a;
      
      int8_t *pB2 = (pB + num_col_im2col_a);

      int8_t *pA2 = (pA  + num_col_im2col_w);
      int8_t *pA3 = (pA2 + num_col_im2col_w);
      int8_t *pA4 = (pA3 + num_col_im2col_w);

      do
      {
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);

        int8_t inB = (int8_t)bitext((int32_t) *pB, 4, 0);
        int8_t inB2 = (int8_t)bitext((int32_t) *pB2, 4, 0);

        sum += inA * inB;

        sum2 += inA * inB2;


        inA = (int8_t) bitext((int) *pA, 2, 2);

        inB = (int8_t)bitext((int32_t) *pB, 4, 4);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 4);

        sum += inA * inB;

        sum2 += inA * inB2;


        pB++;
        pB2++;

        inA = (int8_t) bitext((int) *pA, 2, 4);

        inB = (int8_t)bitext((int32_t) *pB, 4, 0);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 0);

        sum += inA * inB;

        sum2 += inA * inB2;


        inA = (int8_t) bitext((int) *pA, 2, 6);

        inB = (int8_t)bitext((int32_t) *pB, 4, 4);
        inB2 = (int8_t)bitext((int32_t) *pB2, 4, 4);

        sum += inA * inB;

        sum2 += inA * inB2;


        pA++;

        pB++;
        pB2++;

        col_cnt_im2col-=4;
      } while(col_cnt_im2col);
    }
    if (flag_batch_norm && flag_relu)
    {
      uint8_t i_o = i & 0x01;
      out[i_o] = pulp_nn_bn_quant_i4(sum, *pKappa, *pLambda, out_shift);
      out2[i_o] = pulp_nn_bn_quant_i4(sum2, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      if(i_o == 0x01)
      {
        *pOut = bitins(out[0], n_mask, out[1], mask, off);
        *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
        pOut++;
        pOut2++;
      }
    }
    else
    {
      if (flag_relu == 1)
      {
        uint8_t i_o = i & 0x01;
        out[i_o] = pulp_nn_quant_i4(sum, out_mult, out_shift);
        out2[i_o] = pulp_nn_quant_i4(sum2, out_mult, out_shift);
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask, out[1], mask, off);
          *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
          pOut++;
          pOut2++;
        }
      }
      else
      {
        uint8_t i_o = i & 0x01;
        out[i_o] = (int8_t) clips4(sum >> out_shift);
        out2[i_o] = (int8_t) clips4(sum2 >> out_shift);
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask, out[1], mask, off);
          *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
          pOut++;
          pOut2++;
        }
      }
    }
    i++;
    chan_left--;
  }

  if(chan_left != 0){
    W_ROLLBACK(w_rollback);
    W_SKIP("3");
    MIXED_SKIP("8");
  }
  pOut+=ch_out_r;
  return pOut;
}
