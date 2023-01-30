/*
 * xpulp_nn_mix_linear_i2_u2_i8.c
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



void __attribute__((noinline)) xpulp_nn_mix_linear_i2_u2_i8(
                        int8_t *pIn,
                        int8_t *pBias,
                        uint8_t *pOut,
                        int8_t *pWeight,
                        int64_t *pKappa,
                        int64_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;
  uint16_t dim_vec_in = PACK_INT2_SIZE(dim_vec);
  uint16_t dim_vec_wt = PACK_INT8_SIZE(dim_vec);

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
  int neuron_left = 0;
  if (chunk & 0x3)
  {
      neuron_left = (4 - (chunk & 0x7));
  }
  int start = min((chunk + neuron_left) * core_id, num_o_neurons);
  int stop = min(start + chunk + neuron_left, num_o_neurons);

  v4s vecB[4];

  uint8_t *pOutBuffer = (uint8_t *) pOut + (start >> 2);

  int i;

  int64_t *k1 = pKappa + start;
  int64_t *lambda1 = pLambda + start;

  int32_t a_tollback = 4;
  int32_t w_rollback = 4 - (3 * dim_vec_wt);

  A_STRIDE(0);
  W_STRIDE(dim_vec_wt);
  A_ROLLBACK(a_tollback);
  W_ROLLBACK(w_rollback);

  for(i=start; i<stop; i+=4)
  {
    int sum = 0;
    int sum2 = 0;
    int sum3 = 0;
    int sum4 = 0;

    if (pBias != NULL)
    {
      sum = *(int32_t *)(pBias + 4*i);
      sum2 = (pBias[i + 1]);
      sum3 = (pBias[i + 2]);
      sum4 = (pBias[i + 3]);
    }

    int8_t *pB = pIn;
    int8_t *pA = pWeight + (i * dim_vec_wt);
    int8_t *pA2 = pA + dim_vec_wt;
    int8_t *pA3 = pA2 + dim_vec_wt;
    int8_t *pA4 = pA3 + dim_vec_wt;
    
    int32_t *ptrA  = (int32_t *) pA ;

    pB  = pulp_nn_i2_to_i8(pB , vecB);

    int32_t *startB;

    asm volatile("mv %0, %1":"=r"(startB):"r"(vecB));

    int32_t *ptrB  = (int32_t *) vecB;

    W_ADDRESS(ptrA);
    A_ADDRESS(ptrB);

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
    ptrA  = MacLoadInit(1, 0, 1, 0, ptrA);
    ptrA  = MacLoadInit(1, 0, 2, 0, ptrA);
    ptrA  = MacLoadInit(1, 0, 3, 0, ptrA);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);


    for(int j=0; j < (dim_vec >> 4); j++)
    {
      sum  = MacLoads4(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
      sum2 = MacLoads4(1, 0, 1, 0, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
      sum3 = MacLoads4(1, 0, 2, 0, ptrA, sum3);
      ptrA = MacLoadUpdate(ptrA);
      sum4 = MacLoads4(1, 0, 3, 0, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

      sum  = MacLoads4(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
      sum2 = MacLoads4(1, 0, 1, 0, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
      sum3 = MacLoads4(1, 0, 2, 0, ptrA, sum3);
      ptrA = MacLoadUpdate(ptrA);
      sum4 = MacLoads4(1, 0, 3, 0, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

      sum  = MacLoads4(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
      sum2 = MacLoads4(1, 0, 1, 0, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
      sum3 = MacLoads4(1, 0, 2, 0, ptrA, sum3);
      ptrA = MacLoadUpdate(ptrA);
      sum4 = MacLoads4(1, 0, 3, 0, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

      pB  = pulp_nn_i2_to_i8(pB , vecB);

      ptrB   = MacLoadAssign(startB);
      A_ADDRESS(ptrB);

      sum  = MacLoads4(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
      sum2 = MacLoads4(1, 0, 1, 0, ptrA, sum2);
      ptrA = MacLoadUpdate(ptrA);
      sum3 = MacLoads4(1, 0, 2, 0, ptrA, sum3);
      ptrA = MacLoadUpdate(ptrA);
      sum4 = MacLoads4(1, 0, 3, 0, ptrA, sum4);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
    }
    uint16_t col_cnt = dim_vec & 0xf;
    if(col_cnt)
    {
      pA=((dim_vec >> 4) << 4);
      pA2+=((dim_vec >> 4) << 4);
      pA3+=((dim_vec >> 4) << 4);
      pA4+=((dim_vec >> 4) << 4);
      pB-=4;
      do
      {
        int8_t inB =  (int8_t) bitext((int32_t) *pB, 2, 0);
        int8_t inB2 = (int8_t) bitext((int32_t) *pB, 2, 2);
        int8_t inB3 = (int8_t) bitext((int32_t) *pB, 2, 4);
        int8_t inB4 = (int8_t) bitext((int32_t) *pB, 2, 6);
        pB++;
        int8_t inA = *pA;
        pA++;
        int8_t inA2 = *pA;
        pA++;
        int8_t inA3 = *pA;
        pA++;
        int8_t inA4 = *pA;
        pA++;
        sum += inA * inB;
        sum += inA2 * inB2;
        sum += inA3 * inB3;
        sum += inA4 * inB4;
        inA = *pA2;
        pA2++;
        inA2 = *pA2;
        pA2++;
        inA3 = *pA2;
        pA2++;
        inA4 = *pA2;
        pA2++;
        sum2 += inA * inB;
        sum2 += inA2 * inB2;
        sum2 += inA3 * inB3;
        sum2 += inA4 * inB4;
        inA = *pA3;
        pA3++;
        inA2 = *pA3;
        pA3++;
        inA3 = *pA3;
        pA3++;
        inA4 = *pA3;
        pA3++;
        sum3 += inA * inB;
        sum3 += inA2 * inB2;
        sum3 += inA3 * inB3;
        sum3 += inA4 * inB4;
        inA = *pA4;
        pA4++;
        inA2 = *pA4;
        pA4++;
        inA3 = *pA4;
        pA4++;
        inA4 = *pA4;
        pA4++;
        sum4 += inA * inB;
        sum4 += inA2 * inB2;
        sum4 += inA3 * inB3;
        sum4 += inA4 * inB4;
        col_cnt-=4;
      }while (col_cnt);
    }
    if (flag_batch_norm && flag_relu)
    {
      sum = pulp_nn_bn_quant_u2(sum, *k1, *lambda1, out_shift);
      sum2 = pulp_nn_bn_quant_u2(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
      sum3 = pulp_nn_bn_quant_u2(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
      sum4 = pulp_nn_bn_quant_u2(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
      k1+=4;
      lambda1+=4;
      sum = bitins(sum, n_mask2, sum2, mask2, off2);
      sum = bitins(sum, n_mask4, sum3, mask4, off4);
      *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
      pOutBuffer++;
    }
    else
    {
      if (flag_relu == 1)
      {
        sum = pulp_nn_quant_u2(sum, out_mult, out_shift);
        sum2 = pulp_nn_quant_u2(sum2, out_mult, out_shift);
        sum3 = pulp_nn_quant_u2(sum3, out_mult, out_shift);
        sum4 = pulp_nn_quant_u2(sum4, out_mult, out_shift);
        sum = bitins(sum, n_mask2, sum2, mask2, off2);
        sum = bitins(sum, n_mask4, sum3, mask4, off4);
        *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
        pOutBuffer++;
      }
      else
      {
        sum = (uint8_t) clip2(sum >> out_shift);
        sum2 = (uint8_t) clip2(sum2 >> out_shift);
        sum3 = (uint8_t) clip2(sum3 >> out_shift);
        sum4 = (uint8_t) clip2(sum4 >> out_shift);
        sum = bitins(sum, n_mask2, sum2, mask2, off2);
        sum = bitins(sum, n_mask4, sum3, mask4, off4);
        *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
        pOutBuffer++;
      }
    }
  }
  pi_cl_team_barrier(0);
}
