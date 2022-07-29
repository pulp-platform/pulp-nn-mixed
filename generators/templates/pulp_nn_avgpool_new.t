/*
 * ${config.filename}
 * Georg Rutishauser <georgr@iis.ee.ethz.ch>
 * Inspired by CMSIS-NN AvgPool at https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/Source/PoolingFunctions/arm_avgpool_s8.c
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
#include "pulp_nn_kernels.h"

<%
import numpy as np
act_prec = int(config.kernel.act_prec[0:2])
act_t = f"int{act_prec}_t"
out_clip_fn = f"clip{'s' if config.kernel.out_signed else ''}{config.kernel.out_data_t}"
pt_in = f"{'' if config.kernel.in_signed else 'u'}int8_t"
sum_t = f"{'' if config.kernel.in_signed else 'u'}int32_t"
pt_out = f"{'' if config.kernel.out_signed else 'u'}int8_t"
bext = f"bitext{'' if config.kernel.in_signed else '_u'}"
int_t = f"{('' if config.kernel.in_signed else 'unsigned ') + 'int'}"
els_per_out_byte = 8//config.kernel.out_data_t
out_base_mask = 255 >> (8-config.kernel.out_data_t) 
%>


#define bitins(dst,not_mask_imm,src,mask_imm,off) __builtin_pulp_binsert(dst,not_mask_imm,src,mask_imm,off)
#define bitext_u(x,size,off) __builtin_pulp_bextractu(x,size,off)
#define bitext(x,size,off) __builtin_pulp_bextract(x,size,off)

void __attribute__ ((noinline))  ${config.fn_name}(
  ${pt_in} * pIn,
  ${pt_out} * pOut,
  ${act_t} lambda,
  uint16_t out_shift,
  ${act_t} out_add,
  uint16_t dim_im_in_x,
  uint16_t dim_im_in_y,
  uint16_t ch_im_in,
  uint16_t dim_im_out_x,
  uint16_t dim_im_out_y,
  uint16_t dim_kernel_x,
  uint16_t dim_kernel_y,
  uint16_t padding_t,
  uint16_t padding_b,
  uint16_t padding_l,
  uint16_t padding_r,
  uint16_t stride_x,
  uint16_t stride_y,
  int flag_requant
)
{
  /* parallelization */
  int core_id = pi_core_id();
  int n_cores = NUM_CORES;
  if (dim_im_out_y < NUM_CORES)
  {
    n_cores = dim_im_out_y;
  }
  int Log2Core = log2(n_cores);
  int chunck = (dim_im_out_y >> Log2Core) + ((dim_im_out_y & (n_cores -1))!=0);
  int start = chunck * core_id;
  int stop = min(start + chunck, dim_im_out_y);
  int   i_x, i_y;
<%
els_per_byte_in = 8//config.kernel.in_data_t
els_per_byte_out = 8//config.kernel.out_data_t
dw_in = config.kernel.in_data_t
dw_out = config.kernel.out_data_t
in_base_mask = 0xff >> (8-dw_in)
in_masks = []
for c in range(els_per_byte_in):
    in_masks.append(in_base_mask << (c*dw_in))
in_mask_strings = [f"0x{m:02x}" for m in in_masks]
in_mask_n_strings = [f"0x{m^0xff:02x}" for m in in_masks]
out_base_mask = 0xff >> (8-dw_out)
out_masks = []
for c in range(els_per_byte_out):
    out_masks.append(out_base_mask << (c*dw_out))
out_mask_strings = [f"0x{m:02x}" for m in out_masks]
out_mask_n_strings = [f"0x{m^0xff:02x}" for m in out_masks]
# the following flag signals that we don't write an output every input channel block iteration
wr_not_in_every_iter = dw_in > dw_out
wr_every_in_iter = int(dw_in/dw_out)
byte_chan_shift_in = int(np.log2(els_per_byte_in))
byte_chan_shift_out = int(np.log2(els_per_byte_out))
byte_chan_shift_diff = byte_chan_shift_out - byte_chan_shift_in
%>


  uint32_t kernel_size_tot = dim_kernel_x * dim_kernel_y;
  lambda = lambda/kernel_size_tot;
  int ch_im_in_r = ch_im_in >> ${byte_chan_shift_in};
  int ch_im_out_r = ch_im_in >> ${byte_chan_shift_out};
  ${sum_t} sum[${els_per_byte_in}] = {0};
  for (i_y = start; i_y < stop; i_y++)
    {
        for (i_x = 0; i_x < dim_im_out_x; i_x++)
        {
            int k_y_start, k_y_end;
            int k_x_start, k_x_end;

            const int8_t *pTmp, *pTmpInner;
            int8_t *pDst;
% if wr_not_in_every_iter:
            // in data width: ${dw_in}
            // out data width: ${dw_out}
            // -> we need to do an output write every ${int(dw_in/dw_out)}
            //    input channel "block" (1 byte) iterations
            uint32_t in_iter_cnt = 0;
% endif

            k_y_start = maxs32(0, i_y * stride_y - padding_b);
            k_y_end = mins32(i_y * stride_y - padding_t + dim_kernel_y, dim_im_in_y);

            k_x_start = maxs32(0, i_x * stride_x - padding_l);
            k_x_end = mins32(i_x * stride_x - padding_r + dim_kernel_x, dim_im_in_x);

            pTmp = pIn;
            pDst = &pOut[ch_im_out_r * (i_x + i_y * dim_im_out_x)];
            int k_x, k_y;
% if wr_not_in_every_iter:
            ${pt_out} out_el = 0;
% endif

            for (int ch_cnt = 0; ch_cnt < ch_im_in_r; ch_cnt++)
            {
% for sum_idx in range(els_per_byte_in):
              sum[${sum_idx}] = 0;
% endfor
% if not wr_not_in_every_iter:
              ${pt_out} out_el = 0;
% endif
                for (k_y = k_y_start; k_y < k_y_end; k_y++)
                {
                    for (k_x = k_x_start; k_x < k_x_end; k_x++)
                    {
                        pTmpInner = pTmp + (ch_im_in_r * (k_x + k_y * dim_im_in_x));
                        ${pt_in} cur_chans = *pTmpInner;
                        % for c in range(els_per_byte_in):
<%
cur_mask_shift = c * dw_in
%>
                        sum[${c}] += (${sum_t}) ${bext}((${int_t}) cur_chans, ${dw_in}, ${cur_mask_shift});
                        % endfor
                    }
                }
                ${act_t} out_large;
                if (flag_requant) {
                  % for c in range(els_per_byte_in):
                  out_large = (sum[${c}] * lambda + out_add) >> out_shift;
                  % if not wr_not_in_every_iter:
                    % if els_per_byte_out == 1:
                  out_el = ${out_clip_fn}(out_large);
                    % else:
                      % if c==0 and not wr_not_in_every_iter:
                  out_el = ${out_clip_fn}(out_large);
                      % else:
                  out_el = bitins(out_el, (int8_t) ${out_mask_n_strings[c % els_per_byte_out]}, (${pt_out}) ${out_clip_fn}(out_large), (int8_t) ${out_mask_strings[c % els_per_byte_out]}, ${(c % els_per_byte_out) * dw_out});
                      % endif
                    % endif
                    % if (c % els_per_byte_out) == els_per_byte_out-1:
                  pDst[(ch_cnt ${">>" if byte_chan_shift_diff >= 0 else "<<"} (${byte_chan_shift_diff if byte_chan_shift_diff >= 0 else -byte_chan_shift_diff})) + ${c >> byte_chan_shift_out}] = out_el;
                    % endif
                    % else:
                    out_el |= (${"(" if config.kernel.out_signed else ""}${out_clip_fn}(out_large) ${f"& 0x{out_base_mask:02x})" if config.kernel.out_signed else ""} << (in_iter_cnt * ${els_per_byte_in * config.kernel.out_data_t} + ${c * config.kernel.out_data_t}));
                    % endif
                  % endfor
                  } else {
                  % for c in range(els_per_byte_in):
                  out_large = sum[${c}] / kernel_size_tot;
                  % if not wr_not_in_every_iter:
                    % if els_per_byte_out == 1:
                  out_el = ${out_clip_fn}(out_large);
                    % else:
                        % if c % els_per_byte_out == 0:
                  out_el = ${out_clip_fn}(out_large);
                        % else:
                  out_el = bitins(out_el, (int8_t) ${out_mask_n_strings[c % els_per_byte_out]}, (${pt_out}) ${out_clip_fn}(out_large), (int8_t) ${out_mask_strings[c % els_per_byte_out]}, ${(c % els_per_byte_out) * dw_out});
                        % endif
                    % endif
                    % if (c % els_per_byte_out) == els_per_byte_out-1:
                  pDst[(ch_cnt ${">>" if byte_chan_shift_diff>=0 else "<<"} (${byte_chan_shift_diff if byte_chan_shift_diff>=0 else -byte_chan_shift_diff})) + ${c >> byte_chan_shift_out}] = out_el;
                    % endif
                  % else :
                  out_el |= (${"(" if config.kernel.out_signed else ""}${out_clip_fn}(out_large) ${f"& 0x{out_base_mask:02x})" if config.kernel.out_signed else ""} << (in_iter_cnt * ${els_per_byte_in * config.kernel.out_data_t} + ${c * config.kernel.out_data_t}));
                  % endif
                  % endfor
                }
                % if wr_not_in_every_iter:
                if (in_iter_cnt++ == ${wr_every_in_iter-1}) {
                    pDst[(ch_cnt ${f">> ({byte_chan_shift_diff})" if byte_chan_shift_diff >= 0 else f"<< ({-byte_chan_shift_diff})"})] = out_el;
                    in_iter_cnt = 0;
                    out_el = 0;
                }
                % endif
                pTmp++;
            }
        }
    }
 pi_cl_team_barrier(0);
}
