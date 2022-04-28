/*
 * ${config.filename}
 * Georg Rutishauser <georgr@iis.ee.ethz.ch>
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
def clip_fn_name(prec, signed):
    return f"clip{'s' if signed else ''}{prec}"

clip_in1_fn = clip_fn_name(8, config.in1_signed)
clip_in2_fn = clip_fn_name(8, config.in2_signed)
clip_out_fn = clip_fn_name(config.out_data_t, config.kernel.out_signed)
def su(sgn):
    return 's' if sgn else 'u'
def u_(sgn):
    return '' if sgn else 'u'
vt_in1 = f"v4{su(config.in1_signed)}"
vt_in2 = f"v4{su(config.in2_signed)}"
pt_in1 = f"{u_(config.in1_signed)}int8_t"
pt_in2 = f"{u_(config.in2_signed)}int8_t"
pt_out = f"{u_(config.kernel.out_signed)}int8_t"
%>

void __attribute__ ((noinline)) ${config.fn_name}(
    ${pt_in1} * pIn1,
    ${pt_in2} * pIn2,
    ${pt_out} * pOut,
    ${act_t} in1_mul,
    ${act_t} in1_add,
    uint16_t in1_shift,
    ${act_t} in2_mul,
    ${act_t} in2_add,
    uint16_t in2_shift,
    ${act_t} out_mul,
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

    ${act_t} in1_rq1, in1_rq2, in1_rq3, in1_rq4,
             in2_rq1, in2_rq2, in2_rq3, in2_rq4;
    ${act_t} sum1, sum2, sum3, sum4;
    ${act_t} sum_out1, sum_out2, sum_out3, sum_out4;
    int32_t out1, out2, out3, out4,
            sum_int1, sum_int2, sum_int3, sum_int4;

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

    int ch_im_in1_r = ch_im_in >> ${byte_chan_shift_in1};
    int ch_im_in2_r = ch_im_in >> ${byte_chan_shift_in2};
    int ch_im_out_r = ch_im_in >> ${byte_chan_shift_out};

    int start = min(chunck * core_id, dim_im_in_y);
    int stop = min(start + chunck, dim_im_in_y);

    ${pt_in1} *target1 = pIn1 + start * ch_im_in1_r * dim_im_in_x;
    ${pt_in2} *target2 = pIn2 + start * ch_im_in2_r * dim_im_in_x;
    ${pt_out} *pOutBuffer = pOut + start * ch_im_out_r * dim_im_in_x;

    int a = 0;
    int b = 0;

    ${pt_in1} *target1_ext = &a;
    ${pt_in2} *target2_ext = &b;

    for (int i=0; i<(((stop-start) * ch_im_out_r * dim_im_in_x) >> ${int(np.log2(4/els_per_byte_out))}); i++)
    {
%if config.in1_data_t == 8:
        target1_ext = target1;
%else:
        *((${vt_in1}*)target1_ext) = ${config.unpack_in1_fn}(target1);
%endif
        target1+=${int(dw_in1/2)};

%if config.in2_data_t == 8:
        target2_ext = target2;
%else:
        *((${vt_in2}*)target2_ext) = ${config.unpack_in2_fn}(target2);

%endif
        target2+=${int(dw_in2/2)};
        % for pp in range(4):
#ifdef ADD_VERBOSE
        printf("core %d - in1 it${pp} before requant: %d\n", core_id, *(target1_ext${f" + {pp} " if pp else ""}));
        printf("core %d - in2 it${pp} before requant: %d\n", core_id, *(target2_ext${f" + {pp} " if pp else ""}));
#endif
        in1_rq${pp+1} = ((*(target1_ext${f" + {pp} " if pp else ""})) * in1_mul + in1_add) >> in1_shift;
        in2_rq${pp+1} = ((*(target2_ext${f" + {pp} " if pp else ""})) * in2_mul + in2_add) >> in2_shift;
        sum${pp+1} = ${clip_in1_fn}(in1_rq${pp+1}) + ${clip_in2_fn}(in2_rq${pp+1});
#ifdef ADD_VERBOSE
        printf("core %d - in1_rq${pp+1} it${pp} after requant: %d\nclipped in1_rq${pp+1}: %d\n", core_id, in1_rq${pp+1}, ${clip_in1_fn}(in1_rq${pp+1}));
        printf("core %d - in2_rq${pp+1} it${pp} after requant: %d\nclipped in2_rq${pp+1}: %d\n", core_id, in2_rq${pp+1}), ${clip_in2_fn}(in2_rq${pp+1});
        printf("core %d - sum${pp+1}: %d\n", core_id, sum${pp+1});
#endif
        % endfor

        if (out_requant_flag) {
        % for pp in range(4):
          sum${pp+1} = (sum${pp+1} * out_mul + out_add) >> out_shift;
#ifdef ADD_VERBOSE
          printf("core %d - requantized sum${pp+1}: %d\n", core_id, sum${pp+1});
#endif
        % endfor
        }
        % for pp in range(4):
          out${pp+1} = ${clip_out_fn}(sum${pp+1});
#ifdef ADD_VERBOSE
        printf("core %d - out${pp+1} clipped: %d\n", core_id, out${pp+1});
#endif
        % endfor


%if dw_out == 8:
        % for pp in range(4):
        *pOutBuffer = (${pt_out}) out${pp+1};
        pOutBuffer++;
        % endfor
%elif dw_out == 4:
        *pOutBuffer = (${pt_out}) bitins(out1, (int8_t) ${out_mask_n_strings[1]}, out2, (int8_t) ${out_mask_strings[1]}, 4);
        pOutBuffer++;
        *pOutBuffer = (${pt_out}) bitins(out3, (int8_t) ${out_mask_n_strings[1]}, out4, (int8_t) ${out_mask_strings[1]}, 4);
        pOutBuffer++;
%elif dw_out == 2:
        out1 = bitins(out1, (int8_t) ${out_mask_n_strings[1]}, out2, (int8_t) ${out_mask_strings[1]}, 2);
        out1 = bitins(out1, (int8_t) ${out_mask_n_strings[2]}, out3, (int8_t) ${out_mask_strings[2]}, 4);
        *pOutBuffer = bitins(out1, (int8_t) ${out_mask_n_strings[3]}, out4, (int8_t) ${out_mask_strings[3]}, 6);
        pOutBuffer++;
%endif
    }
   pi_cl_team_barrier(0);
}
