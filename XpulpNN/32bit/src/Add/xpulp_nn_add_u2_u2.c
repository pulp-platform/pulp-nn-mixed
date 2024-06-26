/*
 * xpulp_nn_add_u2_u2.c
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


void __attribute__ ((noinline)) xpulp_nn_add_u2_u2(
    uint8_t * pIn1,
    uint8_t * pIn2,
    uint8_t * pOut,
    uint16_t out_mult1,
    uint16_t out_mult2,
    uint16_t out_shift,
    uint16_t dim_im_in_x,
    uint16_t dim_im_in_y,
    uint16_t ch_im_in)
{
    int core_id = pi_core_id();
    int n_cores = NUM_CORES;

    if (dim_im_in_y < NUM_CORES)
    {
      n_cores = dim_im_in_y;
    }

    int  Log2Core = log2(n_cores);
    int chunck = (dim_im_in_y >> Log2Core) + ((dim_im_in_y & (NUM_CORES-1))!=0);

    uint8_t out1, out2, out3, out4;

    int ch_im_in1_r = ch_im_in >> 2;
    int ch_im_in2_r = ch_im_in >> 2;

    int ch_im_out = ch_im_in >> 2;

    int8_t mask2 = 0x0c;
    int8_t n_mask2 = ~ mask2;
    int8_t mask4 = 0x30;
    int8_t n_mask4 = ~ mask4;
    int8_t mask6 = 0xc0;
    int8_t n_mask6 = ~ mask6;
    int8_t off2 = 2;
    int8_t off4 = 4;
    int8_t off6 = 6;

    int start = min(chunck * core_id, dim_im_in_y);
    int stop = min(start + chunck, dim_im_in_y);

    uint8_t *target1 = pIn1 + start * ch_im_in1_r * dim_im_in_x;
    uint8_t *target2 = pIn2 + start * ch_im_in2_r * dim_im_in_x;
    uint8_t *pOutBuffer = pOut + start * ch_im_out * dim_im_in_x;

    int a = 0;
    int b = 0;

    uint8_t *target1_ext = &a;
    uint8_t *target2_ext = &b;

    for (int i=start; i<(stop * ch_im_out * dim_im_in_x); i++)
    {
        *((v4u*)target1_ext) = pulp_nn_u2_to_u8_r(target1);
        target1++;


        *((v4u*)target2_ext) = pulp_nn_u2_to_u8_r(target2);
        target2++;

        out1 = pulp_nn_add_quant_u2(*target1_ext, *target2_ext, out_mult1, out_mult2, out_shift);
        out2 = pulp_nn_add_quant_u2(*(target1_ext + 1), *(target2_ext + 1), out_mult1, out_mult2, out_shift);
        out3 = pulp_nn_add_quant_u2(*(target1_ext + 2), *(target2_ext + 2), out_mult1, out_mult2, out_shift);
        out4 = pulp_nn_add_quant_u2(*(target1_ext + 3), *(target2_ext + 3), out_mult1, out_mult2, out_shift);

        out1 = bitins(out1, n_mask2, out2, mask2, off2);
        out1 = bitins(out1, n_mask4, out3, mask4, off4);
        *pOutBuffer = bitins(out1, n_mask6, out4, mask6, off6);
        pOutBuffer++;
    }
   pi_cl_team_barrier();
}