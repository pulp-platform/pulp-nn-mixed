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
#include "pulp_nn_kernels.h"

<%
import numpy as np
%>

void __attribute__ ((noinline))  ${config.fn_name}(
  uint8_t * Im_in,
  uint16_t dim_im_in_x,
  uint16_t dim_im_in_y,
  uint16_t ch_im_in,
  uint16_t dim_kernel_x,
  uint16_t dim_kernel_y,
  uint16_t padding_t,
  uint16_t padding_b,
  uint16_t padding_l,
  uint16_t padding_r,
  uint16_t stride,
  uint16_t dim_im_out_x,
  uint16_t dim_im_out_y,
  uint16_t out_shift,
  uint32_t out_add,
  uint32_t lambda,
  uint8_t * Im_out,
  int flag_requant,
  unsigned int * memory_chan
)
{
  /* parallelization */
  int core_id = pi_core_id();
  int n_cores = NUM_CORES;
  int Log2Core = log2(n_cores);
  int chunck = (dim_im_out_y >> Log2Core) + (dim_im_out_y & (n_cores -1)!=0);
  int start = chunck * core_id;
  int stop = min(start + chunck, dim_im_out_y);
  int   i_x, i_y;
<%
els_per_byte_in = 8//config.kernel.in_data_t
els_per_byte_out = 8//config.kernel.out_data_t
base_mask = 0xff >> (8-config.kernel.in_data_t)
# the following flag signals that we don't write an output every input channel block iteration
wr_not_in_every_iter = config.kernel.in_data_t > config.kernel.out_data_t
wr_every_in_iter = int(config.kernel.in_data_t/config.kernel.out_data_t)
byte_chan_shift_in = int(np.log2(els_per_byte_in))
byte_chan_shift_out = int(np.log2(els_per_byte_out))
byte_chan_shift_diff = byte_chan_shift_out - byte_chan_shift_in
%>

  uint32_t kernel_size_tot = dim_kernel_x * dim_kernel_y;
  int ch_im_in_r = ch_im_in >> ${byte_chan_shift_in};
  int ch_im_out_r = ch_im_in >> ${byte_chan_shift_out};
  int oc_slice;
  uint32_t sum[${els_per_byte_in}] = {0};
  for (i_y = start; i_y < stop; i_y++)
    {
        for (i_x = 0; i_x < dim_im_out_y; i_x++)
        {

            uint16_t k_y_start, k_y_end;
            uint16_t k_x_start, k_x_end;
            int32_t chCnt;
            int32_t out_ch_cnt = 0;
            const int8_t *pTmp, *pTmpInner;
            int8_t *pDst;
            % if wr_not_in_every_iter:
            // in data width: ${config.kernel.in_data_t}
            // out data width: ${config.kernel.out_data_t}
            // -> we need to do an output write every ${int(config.kernel.in_data_t/config.kernel.out_data_t)}
            //    input channel "block" (1 byte) iterations
            uint32_t in_iter_cnt = 0;
            % endif

            k_y_start = max16(0, i_y * stride_y - padding_b);
            k_y_end = min16(i_y * stride_y - padding_t + kernel_y, input_y);

            k_x_start = max16(0, i_x * stride_x - padding_l);
            k_x_end = min16(i_x * stride_x - padding_r + kernel_x, input_x);

            pTmp = Im_in;
            pDst = &Im_out[ch_im_out_r * (i_x + i_y * output_x)];

            chCnt = ch_im_in_r;
            while (chCnt > 0)
            {
              % for sum_idx in range(els_per_byte_in):
              sum[${sum_idx}] = 0;
              % endfor
              uint8_t out_el = 0;
                for (k_y = k_y_start; k_y < k_y_end; k_y++)
                {
                    for (k_x = k_x_start; k_x < k_x_end; k_x++)
                    {
                        pTmpInner = pTmp + (ch_im_in_r * (k_x + k_y * input_x));
                        uint8_t cur_chans = *pTmpInner;
                        % for c in range(els_per_byte_in):
<%
cur_mask_shift = c * config.kernel.in_data_t
cur_mask_str = f"0x{base_mask << cur_mask_shift:02x}"
%>
                        sum[${c}] += (uint32_t) ((cur_chans & ${cur_mask_str}) >> ${cur_mask_shift});
                        % endfor
                    }
                }
                chCnt--;
                uint32_t out_large;
                if (flag_requant) {
                  % for c in range(els_per_byte_in):
                  out_large = (sum[${c}] * lambda + out_add) >> out_shift;
                  % if not wr_not_in_every_iter:
                  out_el ${"|" if els_per_byte_out != 1 else ""}= (clip${config.kernel.out_data_t}(out_large)${f" << {(c % els_per_byte_out) * config.kernel.out_data_t}" if els_per_byte_out != 1 else ""};
                    % if (c % els_per_byte_out) == els_per_byte_out-1:
                  pDst[(chCnt >> (${byte_chan_shift_diff})) + ${c >> byte_chan_shift_out}] = out_el;
                    % endif
                                                                   % else:
                  out_el |= (clip${config.kernel.out_data_t}(out_large) << in_iter_cnt * ${els_per_byte_in*config.kernel.out_data_t} + ${c*config.kernel.out_data_t});
                  if (in_iter_cnt == ${wr_every_in_iter-1}): {
                      pDst[(chCnt >> (${byte_chan_shift_diff}))] = out_el;
                      in_iter_cnt = 0;
                    }
                  % endif
                  % endfor
                } else {
                  % for c in range(els_per_byte_in):
                  out_large = sum[${c}] / kernel_size_tot;
                  % if not wr_not_in_every_iter:
                  out_el ${"|" if els_per_byte_out != 1 else ""}= (clip${config.kernel.out_data_t}(out_large)${f" << {(c % els_per_byte_out) * config.kernel.out_data_t}" if els_per_byte_out != 1 else ""};
                  % if (c % els_per_byte_out) == els_per_byte_out-1:
                  pDst[(chCnt >> (${byte_chan_shift_diff})) + ${c >> byte_chan_shift_out}] = out_el;
                  % endif
                  % endif
                  % endfor
                }
            }

 pi_cl_team_barrier(0);
}
