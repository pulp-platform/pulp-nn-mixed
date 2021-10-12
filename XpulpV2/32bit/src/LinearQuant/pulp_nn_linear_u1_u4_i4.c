/*
 * pulp_nn_linear_u1_u4_i4.c
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


void pulp_nn_linear_u1_u4_i4(
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
    int8_t mask = 0xf0;
    int8_t n_mask = ~ mask;
    int8_t off = 0x04;
    uint16_t dim_vec_wt = dim_vec >> 1;

    int core_id = pi_core_id();
    int Log2Core = log2(NUM_CORES);
    int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
    int start = min((chunk << (chunk == 1)) * core_id, num_o_neurons);
    int stop = min(start + (chunk << (chunk == 1)), num_o_neurons);


    uint8_t *pOutBuffer = (uint8_t *) pOut + (start >> 1);

    int i;
    int32_t *k1 = pKappa + start;
    int32_t *lambda1 = pLambda + start;

    for(i=start; i<stop; i+=2)
    {
        int sum = 0;
        int sum2 = 0;

        uint8_t *pA = pIn;
        int8_t *pB = pWeight + (i * dim_vec_wt);
        int8_t *pB2 = pB + dim_vec_wt;

        {
          pA+=4;
          pB+=4;
          pB2+=4;
        }
        while (col_cnt)
        {
          col_cnt--;
        }
        if (flag_batch_norm && flag_relu)
        {
          sum = pulp_nn_bn_quant_u4(sum, *k1, *lambda1, out_shift);
          sum2 = pulp_nn_bn_quant_u4(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
          pOutBuffer++;
          k1+=2;
          lambda1+=2;
        }
        else
        {
          if (flag_relu == 1)
          {
            sum = pulp_nn_quant_u4(sum, out_mult, out_shift);
            sum2 = pulp_nn_quant_u4(sum2, out_mult, out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
            pOutBuffer++;
          }
          else
          {
            sum = (uint8_t) clip4(sum >> out_shift);
            sum2 = (uint8_t) clip4(sum2 >> out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
            pOutBuffer++;
          }
        }
    }
    pi_cl_team_barrier(0);
}
