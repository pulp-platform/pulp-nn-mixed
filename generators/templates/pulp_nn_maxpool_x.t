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
act_prec = int(config.kernel.act_prec[0:2])
act_t = f"int{act_prec}_t"
def su(sgn):
    return 's' if sgn else 'u'
def u_(sgn):
    return '' if sgn else 'u'
def s_(sgn):
    return 's' if sgn else ''

pt_in = f"{u_(config.kernel.in_signed)}int8_t"
int_t_in = f"{u_(config.kernel.in_signed)}int32_t"
pt_out = f"{u_(config.kernel.out_signed)}int8_t"
sumdotp_fn = f"SumDotp{s_(config.kernel.in_signed)}4"
bex = f"bitext{u_(config.kernel.in_signed)}"
%>


void __attribute__ ((noinline)) ${config.fn_name}(
                                                  ${pt_in} * pIn,
                                                  ${pt_out} * pOut,
  uint16_t  dim_im_in_x,
  uint16_t  dim_im_in_y,
  uint16_t  ch_im_in,
  uint16_t  dim_im_out_x,
  uint16_t  dim_im_out_y,
  uint16_t  dim_kernel_x,
  uint16_t  dim_kernel_y,
  uint16_t  padding_t,
  uint16_t  padding_b,
  uint16_t  padding_l,
  uint16_t  padding_r,
  uint16_t  stride_x,
  uint16_t  stride_y)
{
  int core_id = pi_core_id();
  int n_cores = NUM_CORES;
  if (dim_im_in_y < NUM_CORES)
  {
    n_cores = dim_im_in_y;
  }
  int  Log2Core = log2(n_cores);
%if config.kernel.in_data_t == 8:
  int ch_im_in_r = ch_im_in;
%elif config.kernel.in_data_t == 4:
  int ch_im_in_r = ch_im_in >> 1;
%elif config.kernel.in_data_t == 2:
  int ch_im_in_r = ch_im_in >> 2;
%endif

  int chunck = (dim_im_in_y >> Log2Core) + ((dim_im_in_y & (NUM_CORES-1))!=0);

  int start = min(chunck * core_id, dim_im_in_y);
  int stop = min(start + chunck, dim_im_in_y);
  int   i_x, i_y;

  for (i_y = start; i_y < stop; i_y++)
  {
    for (i_x = 0; i_x < dim_im_out_x; i_x++)
    {
      /* for each output pixel */
      ${pt_in}     *target = pIn + (i_y * dim_im_in_x + i_x) * ch_im_in_r;
      uint8_t     *win_start;
      uint8_t     *win_stop;
      if (i_x * stride_x - padding_l < 0)
      {
        win_start = target;
      }
      else
      {
        win_start = pIn + (i_y * dim_im_in_x + i_x * stride_x - padding_l) * ch_im_in_r;
      }

      if (i_x * stride_x - padding_l + dim_kernel_x >= dim_im_in_x)
      {
        win_stop = pIn + (i_y * dim_im_in_x + dim_im_in_x) * ch_im_in_r;
      }
      else
      {
        win_stop = pIn + (i_y * dim_im_in_x + i_x * stride_x - padding_l + dim_kernel_x) * ch_im_in_r;
      }

      /* first step is to copy over initial data */
      for (int i = 0; i< ch_im_in_r; i++) target[i] = win_start[i];

      /* start the max operation from the second part */
      win_start += ch_im_in_r;
      for (; win_start < win_stop; win_start += ch_im_in_r)
      {
        ${config.comp_and_replace_fn}(target, win_start, ch_im_in_r);
      }
    }
  }

  pi_cl_team_barrier();
  if (dim_im_out_y < NUM_CORES)
  {
    n_cores = dim_im_out_y;
  }
  Log2Core = log2(n_cores);
  int chunck2 = (dim_im_out_y >> Log2Core) + ((dim_im_out_y & (NUM_CORES-1))!=0);
  int start2 = chunck2 * core_id;
  int stop2 = min(start2 + chunck2, dim_im_out_y);

    /* then does the pooling along y axis */
  for (i_y = start2; i_y < stop2; i_y++)
  {
    /* for each output row */
    ${pt_out} *target = pOut + i_y * dim_im_out_x * ch_im_in_r;
    ${pt_in} *row_start;
    ${pt_in} *row_end;
    /* setting the starting row */
    if (i_y * stride_y - padding_t < 0)
    {
      row_start = pIn;
    }
    else
    {
      row_start = pIn + (i_y * stride_y - padding_t) * dim_im_in_x * ch_im_in_r;
    }
    /* setting the stopping row */
    if (i_y * stride_y - padding_t + dim_kernel_y >= dim_im_in_y)
    {
      row_end = pIn + dim_im_in_y * dim_im_in_x * ch_im_in_r;
    }
    else
    {
      row_end = pIn + (i_y * stride_y - padding_t + dim_kernel_y) * dim_im_in_x * ch_im_in_r;
    }

    /* copy over the first row */
    for (int i = 0; i< dim_im_out_x * ch_im_in_r; i++)
    {
      target[i] = (${pt_out}) row_start[i];
    }
    /* move over to next row */
    row_start += ch_im_in_r * dim_im_in_x;

    for (; row_start < row_end; row_start += dim_im_in_x * ch_im_in_r)
    {
      ${config.comp_and_replace_fn}(target, row_start, dim_im_out_x * ch_im_in_r);
    }
  }
  pi_cl_team_barrier();
}
