/*
 * xpulp_nn_linear_u2_u1_i1.c
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


void __attribute__((noinline)) xpulp_nn_linear_u2_u1_i1(
                        uint8_t *pIn,
                        int8_t *pBias,
                        uint8_t *pOut,
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
  uint16_t dim_vec_wt = PACK_INT1_SIZE(dim_vec);

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);

  v4u vecB[8];
  v4s vecA[8];
  v4s vecA2[8];


  int i;
  int32_t *k1 = pKappa + start;
  int32_t *lambda1 = pLambda + start;

  for(i=start; i<stop; i+=8)
  {
    int sum = 0;
    int sum2 = 0;

    if (pBias != NULL)
    {
      sum = ((int) (pBias[i]));
      sum2 = (pBias[i + 1]);
    }

    uint8_t *pB = pIn;
    int8_t *pA = pWeight + (i * dim_vec_wt);
    int8_t *pA2 = pA + dim_vec_wt;
    pA  = pulp_nn_i1_to_i2(pA , vecA);
    pA2  = pulp_nn_i1_to_i2(pA2 , vecA2);

    int32_t *startA;
    int32_t *startA2;

    asm volatile("mv %0, %1":"=r"(startA):"r"(vecA));
    asm volatile("mv %0, %1":"=r"(startA2):"r"(vecA2));
    int32_t *ptrA  = (int32_t *) vecA ;
    int32_t *ptrA2  = (int32_t *) vecA2 ;
    uint32_t *ptrB  = (uint32_t *) pB ;

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
    ptrA2  = MacLoadInit(1, 0, 1, 0, ptrA2);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);


    for(int j=0; j < (dim_vec >> 5); j++)
    {
      sum = MacLoad16(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
      sum2 = MacLoad16(1, 0, 1, 0, ptrA2, sum2);
      ptrA2 = MacLoadUpdate(ptrA2);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

      pA  = pulp_nn_i1_to_i2(pA , vecA);
      pA2  = pulp_nn_i1_to_i2(pA2 , vecA2);

      ptrA   = MacLoadAssign(startA);
      ptrA2   = MacLoadAssign(startA2);
      sum = MacLoad16(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
      sum2 = MacLoad16(1, 0, 1, 0, ptrA2, sum2);
      ptrA2 = MacLoadUpdate(ptrA2);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
    }
    uint16_t col_cnt = dim_vec & 0x1f;
    if(col_cnt)
    {
      pA-=4;
      pA2-=4;
      pB=((dim_vec >> 5) << 3);
      do
      {
      }while (col_cnt);
    }
    if (flag_batch_norm && flag_relu)
    {
    }
    else
    {
      if (flag_relu == 1)
      {
      }
      else
      {
      }
    }
  }
  pi_cl_team_barrier(0);
}
