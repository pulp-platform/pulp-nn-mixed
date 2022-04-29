/*
 * ${config.filename}
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

<%
import numpy as np
act_prec = int(config.kernel.act_prec[0:2])
act_t = f"int{act_prec}_t"
%>

void __attribute__ ((noinline)) ${config.fn_name}(
    uint8_t * pIn1,
    uint8_t * pIn2,
    uint8_t * pOut,
    ${act_t} in_mult1,
    ${act_t} in_add1,
    uint16_t in_shift1,
    ${act_t} in_mult2,
    ${act_t} in_add2,
    uint16_t in_shift2,
    ${act_t} out_mult,
    ${act_t} out_add,
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

    ${act_t} sum1, sum2, sum3, sum4;
    int32_t sum_out1, sum_out2, sum_out3, sum_out4;
    uint8_t out1, out2, out3, out4;

<%
els_per_byte_in1 = 8//config.in1_data_t
els_per_byte_in2 = 8//config.in2_data_t
els_per_byte_out = 8//config.out_data_t
dw_in1 = config.in1_data_t
dw_in2 = config.in2_data_t
dw_out = config.out_data_t
in1_base_mask = 0xff >> (8-dw_in1)
in1_masks = []
for c in range(els_per_byte_in1):
    in1_masks.append(in1_base_mask << (c*dw_in1))

in1_mask_strings = [f"0x{m:02x}" for m in in1_masks]
in1_mask_n_strings = [f"0x{m^0xff:02x}" for m in in1_masks]

in2_base_mask = 0xff >> (8-dw_in2)
in2_masks = []
for c in range(els_per_byte_in2):
    in1_masks.append(in2_base_mask << (c*dw_in2))

in2_mask_strings = [f"0x{m:02x}" for m in in2_masks]
in2_mask_n_strings = [f"0x{m^0xff:02x}" for m in in2_masks]

out_base_mask = 0xff >> (8-dw_out)
out_masks = []
for c in range(els_per_byte_out):
    out_masks.append(out_base_mask << (c*dw_out))
out_mask_strings = [f"0x{m:02x}" for m in out_masks]
out_mask_n_strings = [f"0x{m^0xff:02x}" for m in out_masks]
byte_chan_shift_in1 = int(np.log2(els_per_byte_in1))
byte_chan_shift_in2 = int(np.log2(els_per_byte_in2))
byte_chan_shift_out = int(np.log2(els_per_byte_out))
%>

    int ch_im_in1 = ch_im_in << ${byte_chan_shift_in1};
    int ch_im_in2 = ch_im_in << ${byte_chan_shift_in2};
    int ch_im_out_r = ch_im_in << ${byte_chan_shift_out};

    int start = min(chunck * core_id, dim_im_in_y);
    int stop = min(start + chunck, dim_im_in_y);

    uint8_t *target1 = pIn1 + start * ch_im_in1_r * dim_im_in_x;
    uint8_t *target2 = pIn2 + start * ch_im_in2_r * dim_im_in_x;
    uint8_t *pOutBuffer = pOut + start * ch_im_out * dim_im_in_x;

    int a = 0;
    int b = 0;

    uint8_t *target1_ext = &a;
    uint8_t *target2_ext = &b;

    for (int i=start; i<((stop * ch_im_out_r * dim_im_in_x) >> ${int(np.log2(4/els_per_byte_out))}); i++)
    {
%if config.in1_data_t == 8:
        target1_ext = target1;
%elif config.in1_data_t == 4:
        *((v4u*)target1_ext) = ${config.unpack_in1_fn}(target1);
%endif
        target1+=${int(dw_in1/2)};

%if config.in2_data_t == 8:
        target2_ext = target2;
%else:
        *((v4u*)target2_ext) = ${config.unpack_in2_fn}(target2);

%endif
        target2+=${int(dw_in2/2)};
        % for pp in range(4):
        sum${pp+1} = (((*target1_ext${f" + {pp} " if pp else ""}) * in1_mult + in1_add) >> in1_shift) + (((*target2_ext${f" + {pp} " if pp else ""}) * in2_mult + in2_add) >> in2_shift);
        % endfor

        if (out_requant_flag) {
        % for pp in range(4):
          sum_out${pp+1} = (sum${pp+1} * out_mult + out_add) >> out_shift;
        % endfor
        } else {
        % for pp in range(4):
          sum_out${pp+1} = sum${pp+1};
        % endfor
        }
        % for pp in range(4):
        out${pp+1} = clip${config.out_data_t}(sum_out${pp+1});
        % endfor

        

%if dw_out == 8:
        % for pp in range(4):
        *pOutBuffer = (uint8_t) out${pp+1};
        pOutBuffer++;
        % endfor
%elif dw_out == 4:
        *pOutBuffer = (uint8_t) bitins(out1, ${out_mask_n_strings[1]}, out2, ${out_mask_strings[1]}, 4);
        pOutBuffer++;
        *pOutBuffer = (uint8_t) bitins(out3, ${out_mask_n_strings[1]}, out4, ${out_mask_strings[1]}, 4);
        pOutBuffer++;
%elif dw_out == 2:
        out1 = bitins(out1, ${out_mask_n_strings[1]}, out2, ${out_mask_strings[1]}, 2);
        out1 = bitins(out1, ${out_mask_n_strings[2]}, out3, ${out_mask_strings[2]}, 4);
        *pOutBuffer = bitins(out1, ${out_mask_n_strings[3]}, out4, ${out_mask_strings[3]}, 6);
        pOutBuffer++;
%endif
    }
   pi_cl_team_barrier(0);
}