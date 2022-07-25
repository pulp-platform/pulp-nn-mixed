/*
 * xpulp_nn_mix_linear_u4_i32_i8.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Nadalini Alessandro <alessandro.nadalini3@unibo.it>
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


void __attribute__((noinline)) xpulp_nn_mix_linear_u4_i32_i8(
                  uint8_t *pIn,
                  int8_t *pBias,
                  int8_t *pOut,
                  int8_t *pWeight,
                  uint16_t dim_vec,
                  uint16_t num_o_neurons)
{
  uint16_t dim_vec_in = PACK_INT4_SIZE(dim_vec);
  uint16_t dim_vec_wt = PACK_INT8_SIZE(dim_vec);

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
  int start = min(chunk * core_id, num_o_neurons);
  int stop = min(start + chunk, num_o_neurons);

  int32_t *pOutBuffer = (int32_t *) pOut + start;

  uint32_t vecB[2];

  for(int i=start; i<stop; i++)
  {
    int sum = 0;

    if (pBias != NULL)
    {
      sum = ((int) (pBias[i]));
    }

    int8_t *pA = pWeight + (i * dim_vec_wt);

    uint8_t *pB = pIn;

    int32_t *ptrA  = (int32_t *) pA ;

    pB  = pulp_nn_u4_to_u8(pB , vecB);

    uint32_t *startB;

    asm volatile("mv %0, %1":"=r"(startB):"r"(vecB));

    uint32_t *ptrB  = vecB;

    W_ADDRESS(ptrA);
    A_ADDRESS(ptrB);

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);


    for(int j=0; j < (dim_vec >> 3); j++)
    {
      sum = MacLoad4(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
      pB = pulp_nn_u4_to_u8(pB, vecB);

      ptrB = MacLoadAssign(startB);
      A_ADDRESS(ptrB);

      sum = MacLoad4(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
    }
    uint16_t col_cnt = dim_vec & 0x7;
    if(col_cnt)
    {
      pA=((dim_vec >> 3) << 3);
      pB-=4;
      do
      {
        int8_t inA = *pA;
        pA++;
        int8_t inA2 = *pA;
        pA++;
        uint8_t inB  = (uint8_t) bitextu((uint32_t) *pB, 4, 0);
        uint8_t inB2 = (uint8_t) bitextu((uint32_t) *pB, 4, 4);
        pB++;
        sum += inA * inB;
        sum += inA2 * inB2;
        col_cnt-=2;
      }while (col_cnt);
    }
    *pOutBuffer = sum;
    pOutBuffer++;
  }
  pi_cl_team_barrier(0);
}
