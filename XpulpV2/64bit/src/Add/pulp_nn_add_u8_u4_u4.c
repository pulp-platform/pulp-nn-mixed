/*
 * pulp_nn_add_u8_u4_u4.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
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



void __attribute__ ((noinline)) pulp_nn_add_u8_u4_u4(
    uint8_t * pIn1,
    uint8_t * pIn2,
    uint8_t * pOut,
    uint64_t in_mult1,
    uint64_t in_add1,
    uint16_t in_shift1,
    uint64_t in_mult2,
    uint64_t in_add2,
    uint16_t in_shift2,
    uint64_t out_mult,
    uint64_t out_add,
    uint16_t out_shift,
    uint16_t dim_im_in_x,
    uint16_t dim_im_in_y,
    uint16_t ch_im_in,
    int      out_requant_flag)
{
    int core_id = pi_core_id();
    int n_cores = NUM_CORES;

    if (dim_im_in_y < NUM_CORES)
    {
      n_cores = dim_im_in_y;
    }

    int  Log2Core = log2(n_cores);
    int chunck = (dim_im_in_y >> Log2Core) + ((dim_im_in_y & (NUM_CORES-1))!=0);

    uint64_t sum1, sum2, sum3, sum4;
    uint8_t out1, out2, out3, out4;



    int ch_im_out_r = ch_im_in << 1

    int start = min(chunck * core_id, dim_im_in_y);
    int stop = min(start + chunck, dim_im_in_y);

    uint8_t *target1 = pIn1 + start * ch_im_in1_r * dim_im_in_x;
    uint8_t *target2 = pIn2 + start * ch_im_in2_r * dim_im_in_x;
    uint8_t *pOutBuffer = pOut + start * ch_im_out * dim_im_in_x;

    int a = 0;
    int b = 0;

    uint8_t *target1_ext = &a;
    uint8_t *target2_ext = &b;

    for (int i=start; i<((stop * ch_im_out_r * dim_im_in_x) >> 1); i++)
    {
        target1_ext = target1;
        target1+=4;

        *((v4u*)target2_ext) = pulp_nn_u4_to_u8_r(target2);

        target2+=2;
        sum1 = (((*target1_ext) * in1_mult + in1_add) >> in1_shift) + (((*target2_ext) * in2_mult + in2_add) >> in2_shift);
        sum2 = (((*target1_ext + 1 ) * in1_mult + in1_add) >> in1_shift) + (((*target2_ext + 1 ) * in2_mult + in2_add) >> in2_shift);
        sum3 = (((*target1_ext + 2 ) * in1_mult + in1_add) >> in1_shift) + (((*target2_ext + 2 ) * in2_mult + in2_add) >> in2_shift);
        sum4 = (((*target1_ext + 3 ) * in1_mult + in1_add) >> in1_shift) + (((*target2_ext + 3 ) * in2_mult + in2_add) >> in2_shift);

        if (out_requant_flag) {
          out1 = (out1 * out_mult + out_add) >> out_shift;
          out2 = (out2 * out_mult + out_add) >> out_shift;
          out3 = (out3 * out_mult + out_add) >> out_shift;
          out4 = (out4 * out_mult + out_add) >> out_shift;
        }
        out1 = clip4(out1);
        out2 = clip4(out2);
        out3 = clip4(out3);
        out4 = clip4(out4);

        

        *pOutBuffer = (uint8_t) bitins(out1, 0x0f, out2, 0xf0, 4);
        pOutBuffer++;
        *pOutBuffer = (uint8_t) bitins(out3, 0x0f, out4, 0xf0, 4);
        pOutBuffer++;
    }
   pi_cl_team_barrier(0);
}