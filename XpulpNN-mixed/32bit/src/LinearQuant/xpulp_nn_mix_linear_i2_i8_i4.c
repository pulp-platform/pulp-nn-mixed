/*
 * xpulp_nn_mix_linear_i2_i8_i4.c
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



void __attribute__((noinline)) xpulp_nn_mix_linear_i2_i8_i4(
                        int8_t *pIn,
                        int8_t *pBias,
                        int8_t *pOut,
                        int8_t *pWeight,
                        int32_t *pKappa,
                        int32_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
  uint16_t dim_vec_in = PACK_INT2_SIZE(dim_vec);
  uint16_t dim_vec_wt = PACK_INT4_SIZE(dim_vec);

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
  int start = min(chunk * core_id, num_o_neurons);
  int stop = min(start + chunk, num_o_neurons);

  v4s vecB[4];

  int8_t *pOutBuffer = (int8_t *) pOut + start;

  int i;

  int32_t *k1 = pKappa + start;
  int32_t *lambda1 = pLambda + start;

  int32_t a_tollback = 4;
  int32_t w_rollback = 4;

  A_STRIDE(0);
  W_STRIDE(0);
  A_ROLLBACK(a_tollback);
  W_ROLLBACK(w_rollback);

  for(i=start; i<stop; i++)
  {
    int sum = 0;

    if (pBias != NULL)
    {
      sum = ((int) (pBias[i]));
    }

    int8_t *pB = pIn;
    int8_t *pA = pWeight + (i * dim_vec_wt);
    
    int32_t *ptrA  = (int32_t *) pA ;

    pB  = pulp_nn_i2_to_i4(pB , vecB);

    int32_t *startB;

    asm volatile("mv %0, %1":"=r"(startB):"r"(vecB));

    int32_t *ptrB  = (int32_t *) vecB;

    W_ADDRESS(ptrA);
    A_ADDRESS(ptrB);

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);


    for(int j=0; j < (dim_vec >> 4); j++)
    {
      sum  = MacLoads8(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);


      pB  = pulp_nn_i2_to_i4(pB , vecB);

      ptrB   = MacLoadAssign(startB);
      A_ADDRESS(ptrB);

      sum  = MacLoads8(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
    }
    uint16_t col_cnt = dim_vec & 0xf;
    if(col_cnt)
    {
      pA=((dim_vec >> 4) << 3);
      pB-=4;
      do
      {
        int8_t inB =  (int8_t) bitext((int32_t) *pB, 2, 0);
        int8_t inB2 = (int8_t) bitext((int32_t) *pB, 2, 2);
        int8_t inB3 = (int8_t) bitext((int32_t) *pB, 2, 4);
        int8_t inB4 = (int8_t) bitext((int32_t) *pB, 2, 6);
        pB++;
        int8_t inA =  (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA, 4, 4);
        pA++;
        int8_t inA3 = (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA, 4, 4);
        pA++;
        sum += inA * inB;
        sum += inA2 * inB2;
        sum += inA3 * inB3;
        sum += inA4 * inB4;
        col_cnt-=4;
      }while (col_cnt);
    }
    if (flag_batch_norm && flag_relu)
    {
      *pOutBuffer = pulp_nn_bn_quant_i8(sum, *k1++, *lambda1++, out_shift);
      pOutBuffer++;
    }
    else
    {
      if (flag_relu == 1)
      {
        *pOutBuffer = pulp_nn_quant_i8(sum, out_mult, out_shift);
        pOutBuffer++;
      }
      else
      {
        *pOutBuffer = (int8_t) clips8(sum >> out_shift);
        pOutBuffer++;
      }
    }
  }
  pi_cl_team_barrier(0);
}
