/*
 * ${config.filename}
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
vt_in = f"v4{su(config.kernel.in_signed)}"
int_t_in = f"{u_(config.kernel.in_signed)}int32_t"
pt_out = f"{u_(config.kernel.out_signed)}int8_t"
macload_fn = f"MacLoad{s_(config.kernel.in_signed)}4"
sumdotp_fn = f"SumDotp{s_(config.kernel.in_signed)}4"
out_clip_fn = f"clip{s_(config.kernel.out_signed)}{config.kernel.out_data_t}"
bex = f"bitext{u_(config.kernel.in_signed)}"
%>

void ${config.fn_name}(
                        ${pt_in} *pIn,
                        ${pt_in} *pIm2ColBuffer,
                        int8_t *pBias,
                        ${pt_out} *pOut,
                        int8_t *pWeight,
                        int8_t *pWtBuffer,
                        ${act_t} *pKappa,
                        ${act_t} *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t dim_in_x,
                        uint16_t dim_in_y,
                        uint16_t ch_in,
                        uint16_t dim_out_x,
                        uint16_t dim_out_y,
                        uint16_t ch_out,
                        uint16_t dim_kernel_x,
                        uint16_t dim_kernel_y,
                        uint16_t padding_y_top,
                        uint16_t padding_y_bottom,
                        uint16_t padding_x_left,
                        uint16_t padding_x_right,
                        uint16_t stride_x,
                        uint16_t stride_y,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
  uint8_t core_id = pi_core_id();
  uint8_t Log2Core = log2(NUM_CORES);

%if config.kernel.out_data_t == 8:
  uint16_t ch_out_r = ch_out;
%elif config.kernel.out_data_t == 4:
  uint16_t ch_out_r = ch_out >> 1;
%elif config.kernel.out_data_t == 2:
  uint16_t ch_out_r = ch_out >> 2;
%endif
%if config.kernel.in_data_t == 8:
  uint16_t ch_in_r = ch_out;
%elif config.kernel.in_data_t == 4:
  uint16_t ch_in_r = ch_out >> 1;
%elif config.kernel.in_data_t == 2:
  uint16_t ch_in_r = ch_out >> 2;
%endif
%if config.kernel.wt_data_t == 8:
  uint16_t ch_wt_r = ch_out;
%elif config.kernel.wt_data_t == 4:
  uint16_t ch_wt_r = ch_out >> 1;
%elif config.kernel.wt_data_t == 2:
  uint16_t ch_wt_r = ch_out >> 2;
%endif

%if config.less_precision == 8:
  uint16_t ch_min = ch_out;
%elif config.less_precision == 4:
  uint16_t ch_min = ch_out >> 1;
%elif config.less_precision == 2:
  uint16_t ch_min = ch_out >> 2;
%endif

  int chunk = (ch_min >> Log2Core) + ((ch_min & (NUM_CORES - 1)) != 0);

  int start_channel = min(chunk * core_id, ch_min);
  int stop_channel = min(start_channel + chunk, ch_min);

  uint16_t dim_kernel_x_size_rem = dim_kernel_x & 0x3;
  uint16_t dim_kernel_x_size_padded = (dim_kernel_x >> 2) + (dim_kernel_x_size_rem != 0);
  uint16_t dim_incr = (dim_kernel_x_size_padded << 2) - dim_kernel_x;
  uint16_t dim_incr_pad_left = (dim_kernel_x_size_padded << 2) - dim_kernel_x + padding_x_left;
  uint16_t dim_incr_pad_right = (dim_kernel_x_size_padded << 2) - dim_kernel_x + 1;
  uint16_t kernel_size = dim_kernel_x * dim_kernel_y;
  uint16_t im2col_size = ((dim_kernel_x * (dim_in_y + padding_y_top + padding_y_bottom)) + dim_kernel_x);
  uint16_t in_image_size = dim_in_x * dim_in_y;

%if config.less_precision == 8:
  ${pt_in} * pIm2ColBase = pIm2ColBuffer + (core_id * im2col_size);
  int8_t * pWtBase = pWtBuffer + kernel_size;
%elif config.less_precision == 4:
  ${pt_in} * pIm2ColBase = pIm2ColBuffer + (core_id * im2col_size << 1);
  int8_t * pWtBase = pWtBuffer + (core_id * (kernel_size << 1));
%elif config.less_precision == 2:
  ${pt_in} * pIm2ColBase = pIm2ColBuffer + (core_id * im2col_size << 2);
  int8_t * pWtBase = pWtBuffer + (core_id * (kernel_size << 2));
%endif

  int i_out_x, i_buff_y;
  uint16_t colCnt = kernel_size >> 2;
  uint16_t leftCnt = kernel_size & 0x3;

%if config.kernel.out_data_t == 2:
  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;
%elif config.kernel.out_data_t == 4:
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;
%endif

%if config.less_precision == 8:
  int i_out_ch = start_channel;
  int i_in_ch = start_channel * in_image_size;
  int i_wt_ch = start_channel * kernel_size;
%elif config.less_precision == 4:
  %if config.kernel.out_data_t == 8:
  int i_out_ch = (start_channel << 1);
  %elif config.kernel.out_data_t == 4:
  int i_out_ch = start_channel;
  %endif
  %if config.kernel.in_data_t == 8:
  int i_in_ch = (start_channel << 1) * in_image_size;
  %elif config.kernel.in_data_t == 4:
  int i_in_ch = start_channel * in_image_size;
  %endif
  %if config.kernel.wt_data_t == 8:
  int i_wt_ch = (start_channel << 1) * kernel_size;
  %elif config.kernel.wt_data_t == 4:
  int i_wt_ch = start_channel * kernel_size;
  %endif
%elif config.less_precision == 2:
  %if config.kernel.out_data_t == 8:
  int i_out_ch = (start_channel << 2);
  %elif config.kernel.out_data_t == 4:
  int i_out_ch = (start_channel << 1);
  %elif config.kernel.out_data_t == 2:
  int i_out_ch = start_channel;
  %endif
  %if config.kernel.in_data_t == 8:
  int i_in_ch = (start_channel << 2) * in_image_size;
  %elif config.kernel.in_data_t == 4:
  int i_in_ch = (start_channel << 1) * in_image_size;
  %elif config.kernel.in_data_t == 2:
  int i_in_ch = start_channel * in_image_size;
  %endif
  %if config.kernel.wt_data_t == 8:
  int i_wt_ch = (start_channel << 2) * kernel_size;
  %elif config.kernel.wt_data_t == 4:
  int i_wt_ch = (start_channel << 1) * kernel_size;
  %elif config.kernel.wt_data_t == 2:
  int i_wt_ch = start_channel * kernel_size;
  %endif
%endif


  ${act_t} * k1 = pKappa + core_id * (chunk << ${4//config.less_precision});
  ${act_t} * lambda1 = pLambda + core_id * (chunk << ${4//config.less_precision});

  for(int i_ch = start_channel; i_ch < stop_channel; i_ch++)
  {
    i_out_x = 0;
%if config.less_precision == 8:
    int8_t * pWt = pWeight + i_wt_ch;
%elif config.less_precision == 4:
  %if config.kernel.wt_data_t == 8:
    int8_t * pWt = pWeight + i_wt_ch;
  %else:
    int8_t * pWt = pWtBase;
  %endif
    int8_t * pWt2 = pWt + kernel_size;
  %if config.kernel.wt_data_t == 4:
    int8_t *src_wt = pWeight + i_wt_ch;
    for(int i_unpack = 0; i_unpack < kernel_size; i_unpack++)
    {
      *pWt = (int8_t) bitext((int) *src_wt, 4, 0);
      pWt++;
      *pWt2 = (int8_t) bitext((int) *src_wt, 4, 4);
      pWt2++;
      src_wt++;
    }
  %endif
%elif config.less_precision == 2:
  %if config.kernel.wt_data_t == 8:
    int8_t * pWt = pWeight + i_wt_ch;
  %else:
    int8_t * pWt = pWtBase;
  %endif
    int8_t * pWt2 = pWt + kernel_size;
    int8_t * pWt3 = pWt2 + kernel_size;
    int8_t * pWt4 = pWt3 + kernel_size;
  %if config.kernel.wt_data_t == 4:
    int8_t *src_wt = pWeight + i_wt_ch;
    for(int i_unpack = 0; i_unpack < kernel_size; i_unpack++)
    {
      *pWt = (int8_t) bitext((int) *src_wt, 4, 0);
      pWt++;
      *pWt2 = (int8_t) bitext((int) *src_wt, 4, 4);
      pWt2++;
      src_wt++;
    }
    for(int i_unpack = 0; i_unpack < kernel_size; i_unpack++)
    {
      *pWt3 = (int8_t) bitext((int) *src_wt, 4, 0);
      pWt3++;
      *pWt4 = (int8_t) bitext((int) *src_wt, 4, 4);
      pWt4++;
      src_wt++;
    }
  %elif config.kernel.wt_data_t == 2:
    int8_t *src_wt = pWeight + i_wt_ch;
    for(int i_unpack = 0; i_unpack < kernel_size; i_unpack++)
    {
      *pWt = (int8_t) bitext((int) *src_wt, 2, 0);
      pWt++;
      *pWt2 = (int8_t) bitext((int) *src_wt, 2, 2);
      pWt2++;
      *pWt3 = (int8_t) bitext((int) *src_wt, 2, 4);
      pWt3++;
      *pWt4 = (int8_t) bitext((int) *src_wt, 2, 6);
      pWt4++;
      src_wt++;
    }
  %endif
%endif
    if(padding_x_left > 0)
    {
      do
      {
        ${pt_out} *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
        ${pt_in} *pIm2Col = pIm2ColBase;
  %if config.less_precision == 4:
        ${pt_in} *pIm2Col2 = pIm2Col + im2col_size;
  %elif config.less_precision == 2:
        ${pt_in} *pIm2Col2 = pIm2Col + im2col_size;
        ${pt_in} *pIm2Col3 = pIm2Col2 + im2col_size;
        ${pt_in} *pIm2Col4 = pIm2Col3 + im2col_size;
  %endif
        i_buff_y = - padding_y_top;
        if(padding_y_top > 0)
        {
          do
          {
            int i=0;
            do
            {
              *(${vt_in} *) pIm2Col = (${vt_in}) {0, 0, 0, 0};
              pIm2Col+=4;
    %if config.less_precision == 4:
              *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
              pIm2Col2+=4;
    %elif config.less_precision == 2:
              *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
              pIm2Col2+=4;
              *(${vt_in} *) pIm2Col3 = (${vt_in}) {0, 0, 0, 0};
              pIm2Col3+=4;
              *(${vt_in} *) pIm2Col4 = (${vt_in}) {0, 0, 0, 0};
              pIm2Col4+=4;
    %endif
              i++;
            }while(i<dim_kernel_x_size_padded);
            pIm2Col-=dim_incr;
    %if config.less_precision == 4:
            pIm2Col2-=dim_incr;
    %elif config.less_precision == 2:
            pIm2Col2-=dim_incr;
            pIm2Col3-=dim_incr;
            pIm2Col4-=dim_incr;
    %endif
            i_buff_y++;
          }while(i_buff_y < 0);
        }
        int const1 = (i_out_x * stride_x);
        int base_ptr = pIn + i_in_ch;
        do
        {
          for(int j=0; j< (padding_x_left - const1); j++)
          {
            *(${pt_in} *) pIm2Col = 0;
            pIm2Col++;
  %if config.less_precision == 4:
            *(${pt_in} *) pIm2Col2 = 0;
            pIm2Col2++;
  %elif config.less_precision == 2:
            *(${pt_in} *) pIm2Col2 = 0;
            pIm2Col2++;
            *(${pt_in} *) pIm2Col3 = 0;
            pIm2Col3++;
            *(${pt_in} *) pIm2Col4 = 0;
            pIm2Col4++;
  %endif
          }
          int idx = 0;
          int i = 0;
  %if config.less_precision == 8:
          do
          {
            *((${vt_in}*) pIm2Col) = *((${vt_in}*) (base_ptr + idx));
            pIm2Col+=4;
            idx+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
  %elif config.less_precision == 4:
    %if config.kernel.in_data_t == 8:
          do
          {
            *((${vt_in}*) pIm2Col) = *((${vt_in}*) (base_ptr + idx));
            pIm2Col+=4;
            *((${vt_in}*) pIm2Col2) = *((${vt_in}*) (base_ptr + idx + in_image_size));
            pIm2Col2+=4;
            idx+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
    %elif config.kernel.in_data_t == 4:
          do
          {
            ${vt_in} src_in = *((${vt_in}*) (base_ptr + idx));
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 0);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 4);
            pIm2Col++;
            pIm2Col2++;
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 8);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 12);
            pIm2Col++;
            pIm2Col2++;
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 16);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 20);
            pIm2Col++;
            pIm2Col2++;
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 24);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 28);
            pIm2Col++;
            pIm2Col2++;
            idx+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
    %endif
  %elif config.less_precision == 2:
    %if config.kernel.in_data_t == 8:
          do
          {
            int idc = in_image_size;
            *((${vt_in}*) pIm2Col) = *((${vt_in}*) (base_ptr + idx));
            pIm2Col+=4;
            *((${vt_in}*) pIm2Col2) = *((${vt_in}*) (base_ptr + idx + idc));
            pIm2Col2+=4;
            idc+=in_image_size;
            *((${vt_in}*) pIm2Col3) = *((${vt_in}*) (base_ptr + idx + idc));
            pIm2Col3+=4;
            idc+=in_image_size;
            *((${vt_in}*) pIm2Col4) = *((${vt_in}*) (base_ptr + idx + idc));
            pIm2Col4+=4;
            idx+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
    %elif config.kernel.in_data_t == 4:
          do
          {
            ${vt_in} src_in = *((${vt_in}*) (base_ptr + idx));
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 0);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 4);
            pIm2Col++;
            pIm2Col2++;
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 8);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 12);
            pIm2Col++;
            pIm2Col2++;
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 16);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 20);
            pIm2Col++;
            pIm2Col2++;
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 24);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 28);
            pIm2Col++;
            pIm2Col2++;
            src_in = *((${vt_in}*) (base_ptr + idx + in_image_size));
            *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 0);
            *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 4);
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 8);
            *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 12);
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 16);
            *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 20);
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 24);
            *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 28);
            pIm2Col3++;
            pIm2Col4++;
            idx+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
    %elif config.kernel.in_data_t == 2:
          do
          {
            ${vt_in} src_in = *((${vt_in}*) (base_ptr + idx));
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 0);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 2);
            *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 4);
            *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 6);
            pIm2Col++;
            pIm2Col2++;
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 8);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 10);
            *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 12);
            *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 14);
            pIm2Col++;
            pIm2Col2++;
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 16);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 18);
            *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 20);
            *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 22);
            pIm2Col++;
            pIm2Col2++;
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 24);
            *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 26);
            *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 28);
            *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 30);
            pIm2Col++;
            pIm2Col2++;
            pIm2Col3++;
            pIm2Col4++;
            idx+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
    %endif
  %endif
          pIm2Col-=(dim_incr_pad_left - const1);
  %if config.less_precision == 4:
          pIm2Col2-=(dim_incr_pad_left - const1);
  %elif config.less_precision == 2:
          pIm2Col2-=(dim_incr_pad_left - const1);
          pIm2Col3-=(dim_incr_pad_left - const1);
          pIm2Col4-=(dim_incr_pad_left - const1);
  %endif
          base_ptr+=dim_in_x;
          i_buff_y++;
        }while(i_buff_y < dim_in_y);
        for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
        {
          int i=0;
          do
          {
            *(${vt_in} *) pIm2Col = (${vt_in}) {0, 0, 0, 0};
            pIm2Col+=4;
  %if config.less_precision == 4:
            *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col2+=4;
  %elif config.less_precision == 2:
            *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col2+=4;
            *(${vt_in} *) pIm2Col3 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col3+=4;
            *(${vt_in} *) pIm2Col4 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col4+=4;
  %endif
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
  %if config.less_precision == 4:
          pIm2Col2-=dim_incr;
  %elif config.less_precision == 2:
          pIm2Col2-=dim_incr;
          pIm2Col3-=dim_incr;
          pIm2Col4-=dim_incr;
  %endif
        }

        int l=0;
        do
        {
  %if config.less_precision == 8:
    %if config.kernel.wt_data_t == 8:
          pWt = pWeight + i_wt_ch;
    %else:
          pWt = pWtBase;
    %endif
          int sum = 0;
          if (pBias != NULL)
          {
            sum = ((int) (pBias[i_ch]));
          }
          pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
  %elif config.less_precision == 4:
    %if config.kernel.wt_data_t == 8:
          pWt = pWeight + i_wt_ch;
    %else:
          pWt = pWtBase;
    %endif
          pWt2 = pWt + kernel_size;
          int sum = 0;
          int sum2 = 0;
          if (pBias != NULL)
          {
            sum = ((int) (pBias[i_ch]));
            sum2 = ((int) (pBias[i_ch + 1]));
          }
          pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
          pIm2Col2 = pIm2Col + im2col_size;
  %elif config.less_precision == 2:
    %if config.kernel.wt_data_t == 8:
          pWt = pWeight + i_wt_ch;
    %else:
          pWt = pWtBase;
    %endif
          pWt2 = pWt + kernel_size;
          pWt3 = pWt2 + kernel_size;
          pWt4 = pWt3 + kernel_size;
          int sum = 0;
          int sum2 = 0;
          int sum3 = 0;
          int sum4 = 0;
          if (pBias != NULL)
          {
            sum = ((int) (pBias[i_ch]));
            sum2 = ((int) (pBias[i_ch + 1]));
            sum3 = ((int) (pBias[i_ch + 2]));
            sum4 = ((int) (pBias[i_ch + 3]));
          }
          pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
          pIm2Col2 = pIm2Col + im2col_size;
          pIm2Col3 = pIm2Col2 + im2col_size;
          pIm2Col4 = pIm2Col3 + im2col_size;
  %endif
%if config.less_precision == 8:
          int32_t ptrA  = (int32_t *) pWt;
          ${int_t_in} ptrB = (${int_t_in} *) pIm2Col;

          ptrA = MacLoadInit(1, 0, 0, 0, ptrA);
          ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
%elif config.less_precision == 4:
          int32_t  ptrA  = (int32_t *) pWt;
          int32_t  ptrA2 = (int32_t *) pWt2;
          ${int_t_in} ptrB  = (${int_t_in} *) pIm2Col;
          ${int_t_in} ptrB2 = (${int_t_in} *) pIm2Col2;

          ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
          ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
          ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
          ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);
%endif
          int j=0;
          do
          {
  %if config.less_precision == 8:
            sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
            ptrA = MacLoadUpdate(ptrA);
            ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
  %elif config.less_precision == 4:
            sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
            ptrA = MacLoadUpdate(ptrA);
            ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
            sum2  = ${macload_fn}(1, 0, 1, 1, ptrA2, sum2);
            ptrA2 = MacLoadUpdate(ptrA2);
            ptrB2 = MacLoadInit(0, 1, 1, 1, ptrB2);
  %elif config.less_precision == 2:
            v4s w = *(v4s *) pWt;
            ${vt_in} x = *(${vt_in} *) pIm2Col;
            sum = ${sumdotp_fn}(x, w, sum);
            pWt += 4;
            pIm2Col += 4;
            v4s w2 = *(v4s *) pWt2;
            ${vt_in} x2 = *(${vt_in} *) pIm2Col2;
            sum2 = ${sumdotp_fn}(x2, w2, sum2);
            pWt2 += 4;
            pIm2Col2 += 4;
            v4s w3 = *(v4s *) pWt3;
            ${vt_in} x3 = *(${vt_in} *) pIm2Col3;
            sum3 = ${sumdotp_fn}(x3, w3, sum3);
            pWt3 += 4;
            pIm2Col3 += 4;
            v4s w4 = *(v4s *) pWt4;
            ${vt_in} x4 = *(${vt_in} *) pIm2Col4;
            sum4 = ${sumdotp_fn}(x4, w4, sum4);
            pWt4 += 4;
            pIm2Col4 += 4;
  %endif
            j++;
          }while(j<colCnt);
          if(leftCnt)
          {
    %if config.less_precision == 8:
            pWt+=(j << 2);
            pIm2Col+=(j << 2);
    %elif config.less_precision == 4:
            pWt+=(j << 2);
            pWt2+=(j << 2);
            pIm2Col+=(j << 2);
            pIm2Col2+=(j << 2);
    %endif
            do
            {
    %if config.less_precision == 8:
              int8_t w = *(int8_t *) pWt++;
              ${pt_in} x = *(${pt_in} *) pIm2Col++;
              sum += x * w;
    %elif config.less_precision == 4:
              int8_t w = *(int8_t *) pWt++;
              ${pt_in} x = *(${pt_in} *) pIm2Col++;
              sum += x * w;
              int8_t w2 = *(int8_t *) pWt2++;
              ${pt_in} x2 = *(${pt_in} *) pIm2Col2++;
              sum2 += x2 * w2;
    %elif config.less_precision == 2:
              int8_t w = *(int8_t *) pWt++;
              ${pt_in} x = *(${pt_in} *) pIm2Col++;
              sum += x * w;
              int8_t w2 = *(int8_t *) pWt2++;
              ${pt_in} x2 = *(${pt_in} *) pIm2Col2++;
              sum2 += x2 * w2;
              int8_t w3 = *(int8_t *) pWt3++;
              ${pt_in} x3 = *(${pt_in} *) pIm2Col3++;
              sum3 += x3 * w3;
              int8_t w4 = *(int8_t *) pWt4++;
              ${pt_in} x4 = *(${pt_in} *) pIm2Col4++;
              sum4 += x4 * w4;
    %endif
              j++;
            }while(j<leftCnt);
          }
          if (flag_batch_norm && flag_relu)
          {
  %if config.less_precision == 8:
            *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
  %elif config.less_precision == 4:
    %if config.kernel.out_data_t == 8:
            *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
            *(pOutBuffer + 1) = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
    %elif config.kernel.out_data_t == 4:
            sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
            sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
    %endif
  %elif config.less_precision == 2:
    %if config.kernel.out_data_t == 8:
            *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
            *(pOutBuffer + 1) = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
            *(pOutBuffer + 2) = ${config.bn_fn}(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
            *(pOutBuffer + 3) = ${config.bn_fn}(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
    %elif config.kernel.out_data_t == 4:
            sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
            sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
            sum3 = ${config.bn_fn}(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
            sum4 = ${config.bn_fn}(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
            *(pOutBuffer + 1) = bitins(sum3, n_mask, sum4, mask, off);
    %elif config.kernel.out_data_t == 2:
            sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
            sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
            sum3 = ${config.bn_fn}(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
            sum4 = ${config.bn_fn}(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
    %endif
  %endif
          }
          else
          {
            if(flag_relu == 1)
            {
  %if config.less_precision == 8:
              *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
  %elif config.less_precision == 4:
    %if config.kernel.out_data_t == 8:
              *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
              *(pOutBuffer + 1) = ${config.relu_fn}(sum2, out_mult, out_shift);
    %elif config.kernel.out_data_t == 4:
              sum = ${config.relu_fn}(sum, out_mult, out_shift);
              sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
              *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
    %endif
  %elif config.less_precision == 2:
    %if config.kernel.out_data_t == 8:
              *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
              *(pOutBuffer + 1) = ${config.relu_fn}(sum2, out_mult, out_shift);
              *(pOutBuffer + 2) = ${config.relu_fn}(sum3, out_mult, out_shift);
              *(pOutBuffer + 3) = ${config.relu_fn}(sum4, out_mult, out_shift);
    %elif config.kernel.out_data_t == 4:
              sum = ${config.relu_fn}(sum, out_mult, out_shift);
              sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
              *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
              sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
              sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
              *(pOutBuffer + 1) = bitins(sum3, n_mask, sum4, mask, off);
    %elif config.kernel.out_data_t == 2:
              sum = ${config.relu_fn}(sum, out_mult, out_shift);
              sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
              sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
              sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
              sum = bitins(sum, n_mask2, sum2, mask2, off2);
              sum = bitins(sum, n_mask4, sum3, mask4, off4);
              *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
    %endif
  %endif
            }
            else
            {
  %if config.less_precision == 8:
              *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
  %elif config.less_precision == 4:
    %if config.kernel.out_data_t == 8:
              *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
              *(pOutBuffer + 1) = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
    %elif config.kernel.out_data_t == 4:
              sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
              sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
              *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
    %endif
  %elif config.less_precision == 2:
    %if config.kernel.out_data_t == 8:
              *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
              *(pOutBuffer + 1) = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
              *(pOutBuffer + 2) = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
              *(pOutBuffer + 3) = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
    %elif config.kernel.out_data_t == 4:
              sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
              sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
              *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
              sum3 = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
              sum4 = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
              *(pOutBuffer + 1) = bitins(sum3, n_mask, sum4, mask, off);
    %elif config.kernel.out_data_t == 2:
              sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
              sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
              sum3 = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
              sum4 = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
              sum = bitins(sum, n_mask2, sum2, mask2, off2);
              sum = bitins(sum, n_mask4, sum3, mask4, off4);
              *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
    %endif
  %endif
            }
          }
          pOutBuffer+=(dim_out_x * ch_out_r);
          l++;
        }while(l<dim_out_y);
        i_out_x++;
      }while((i_out_x * stride_x) < padding_x_left);
    }
    do
    {
      ${pt_out} *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
      ${pt_in} *pIm2Col = pIm2ColBase;
%if config.less_precision == 4:
      ${pt_in} *pIm2Col2 = pIm2Col + im2col_size;
%elif config.less_precision == 2:
      ${pt_in} *pIm2Col2 = pIm2Col + im2col_size;
      ${pt_in} *pIm2Col3 = pIm2Col2 + im2col_size;
      ${pt_in} *pIm2Col4 = pIm2Col3 + im2col_size;
%endif
      i_buff_y = - padding_y_top;
      if(padding_y_top > 0)
      {
        do
        {
          int i=0;
          do
          {
            *(${vt_in} *) pIm2Col = (${vt_in}) {0, 0, 0, 0};
            pIm2Col+=4;
  %if config.less_precision == 4:
            *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col2+=4;
  %elif config.less_precision == 2:
            *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col2+=4;
            *(${vt_in} *) pIm2Col3 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col3+=4;
            *(${vt_in} *) pIm2Col4 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col4+=4;
  %endif
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
  %if config.less_precision == 4:
          pIm2Col2-=dim_incr;
  %elif config.less_precision == 2:
          pIm2Col2-=dim_incr;
          pIm2Col3-=dim_incr;
          pIm2Col4-=dim_incr;
  %endif
          i_buff_y++;
        }while(i_buff_y < 0);
      }
      int base_ptr = pIn + i_in_ch + (i_out_x * stride_x) - padding_x_left;
      do
      {
        int idx = 0;
%if config.less_precision == 8:
        for (int i=0; i<dim_kernel_x_size_padded; i++)
        {
          *((${vt_in}*) pIm2Col) = *((${vt_in}*) (base_ptr + idx));
          pIm2Col+=4;
          idx+=4;
        }
%elif config.less_precision == 4:
  %if config.kernel.in_data_t == 8:
        for (int i=0; i<dim_kernel_x_size_padded; i++)
        {
          *((${vt_in}*) pIm2Col) = *((${vt_in}*) (base_ptr + idx));
          pIm2Col+=4;
          *((${vt_in}*) pIm2Col2) = *((${vt_in}*) (base_ptr + idx + in_image_size));
          pIm2Col2+=4;
          idx+=4;
        }
  %elif config.kernel.in_data_t == 4:
        for (int i=0; i<dim_kernel_x_size_padded; i++)
        {
          ${vt_in} src_in = *((${vt_in}*) (base_ptr + idx));
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 0);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 4);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 8);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 12);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 16);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 20);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 24);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 28);
          pIm2Col++;
          pIm2Col2++;
          idx+=4;
        }
  %endif
%elif config.less_precision == 2:
  %if config.kernel.in_data_t == 8:
        for (int i=0; i<dim_kernel_x_size_padded; i++)
        {
          int idc = in_image_size;
          *((${vt_in}*) pIm2Col) = *((${vt_in}*) (base_ptr + idx));
          pIm2Col+=4;
          *((${vt_in}*) pIm2Col2) = *((${vt_in}*) (base_ptr + idx + idc));
          pIm2Col2+=4;
          idc+=in_image_size;
          *((${vt_in}*) pIm2Col3) = *((${vt_in}*) (base_ptr + idx + idc));
          pIm2Col3+=4;
          idc+=in_image_size;
          *((${vt_in}*) pIm2Col4) = *((${vt_in}*) (base_ptr + idx + idc));
          pIm2Col4+=4;
          idx+=4;
        }
  %elif config.kernel.in_data_t == 4:
        for (int i=0; i<dim_kernel_x_size_padded; i++)
        {
          ${vt_in} src_in = *((${vt_in}*) (base_ptr + idx));
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 0);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 4);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 8);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 12);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 16);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 20);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 24);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 28);
          pIm2Col++;
          pIm2Col2++;
          src_in = *((${vt_in}*) (base_ptr + idx + in_image_size));
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 0);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 4);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 8);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 12);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 16);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 20);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 24);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 28);
          pIm2Col3++;
          pIm2Col4++;
          idx+=4;
        }
  %elif config.kernel.in_data_t == 2:
        for (int i=0; i<dim_kernel_x_size_padded; i++)
        {
          ${vt_in} src_in = *((${vt_in}*) (base_ptr + idx));
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 0);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 2);
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 4);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 6);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 8);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 10);
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 12);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 14);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 16);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 18);
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 20);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 22);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 24);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 26);
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 28);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 30);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          idx+=4;
        }
  %endif
%endif
        pIm2Col-=dim_incr;
%if config.less_precision == 4:
        pIm2Col2-=dim_incr;
%elif config.less_precision == 2:
        pIm2Col2-=dim_incr;
        pIm2Col3-=dim_incr;
        pIm2Col4-=dim_incr;
%endif
        base_ptr+=dim_in_x;
        i_buff_y++;
      }while(i_buff_y < dim_in_y);
      for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
      {
        int i=0;
        do
        {
          *(${vt_in} *) pIm2Col = (${vt_in}) {0, 0, 0, 0};
          pIm2Col+=4;
%if config.less_precision == 4:
          *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
          pIm2Col2+=4;
%elif config.less_precision == 2:
          *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
          pIm2Col2+=4;
          *(${vt_in} *) pIm2Col3 = (${vt_in}) {0, 0, 0, 0};
          pIm2Col3+=4;
          *(${vt_in} *) pIm2Col4 = (${vt_in}) {0, 0, 0, 0};
          pIm2Col4+=4;
%endif
          i++;
        }while(i<dim_kernel_x_size_padded);
        pIm2Col-=dim_incr;
%if config.less_precision == 4:
        pIm2Col2-=dim_incr;
%elif config.less_precision == 2:
        pIm2Col2-=dim_incr;
        pIm2Col3-=dim_incr;
        pIm2Col4-=dim_incr;
%endif
      }
      int l=0;
      do
      {
%if config.less_precision == 8:
  %if config.kernel.wt_data_t == 8:
        pWt = pWeight + i_wt_ch;
  %else:
        pWt = pWtBase;
  %endif
        int sum = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
%elif config.less_precision == 4:
  %if config.kernel.wt_data_t == 8:
        pWt = pWeight + i_wt_ch;
  %else:
        pWt = pWtBase;
  %endif
        pWt2 = pWt + kernel_size;
        int sum = 0;
        int sum2 = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
          sum2 = ((int) (pBias[i_ch + 1]));
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        pIm2Col2 = pIm2Col + im2col_size;
%elif config.less_precision == 2:
  %if config.kernel.wt_data_t == 8:
        pWt = pWeight + i_wt_ch;
  %else:
        pWt = pWtBase;
  %endif
        pWt2 = pWt + kernel_size;
        pWt3 = pWt2 + kernel_size;
        pWt4 = pWt3 + kernel_size;
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
          sum2 = ((int) (pBias[i_ch + 1]));
          sum3 = ((int) (pBias[i_ch + 2]));
          sum4 = ((int) (pBias[i_ch + 3]));
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        pIm2Col2 = pIm2Col + im2col_size;
        pIm2Col3 = pIm2Col2 + im2col_size;
        pIm2Col4 = pIm2Col3 + im2col_size;
%endif
%if config.less_precision == 8:
        int32_t ptrA  = (int32_t *) pWt;
        ${int_t_in} ptrB = (${int_t_in} *) pIm2Col;

        ptrA = MacLoadInit(1, 0, 0, 0, ptrA);
        ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
%elif config.less_precision == 4:
        int32_t  ptrA  = (int32_t *) pWt;
        int32_t  ptrA2 = (int32_t *) pWt2;
        ${int_t_in} ptrB  = (${int_t_in} *) pIm2Col;
        ${int_t_in} ptrB2 = (${int_t_in} *) pIm2Col2;

        ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
        ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
        ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
        ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);
%endif
        int j=0;
        do
        {
%if config.less_precision == 8:
          sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
          ptrA = MacLoadUpdate(ptrA);
          ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
%elif config.less_precision == 4:
          sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
          ptrA = MacLoadUpdate(ptrA);
          ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
          sum2  = ${macload_fn}(1, 0, 1, 1, ptrA2, sum2);
          ptrA2 = MacLoadUpdate(ptrA2);
          ptrB2 = MacLoadInit(0, 1, 1, 1, ptrB2);
%elif config.less_precision == 2:
          v4s w = *(v4s *) pWt;
          ${vt_in} x = *(${vt_in} *) pIm2Col;
          sum = ${sumdotp_fn}(x, w, sum);
          pWt += 4;
          pIm2Col += 4;
          v4s w2 = *(v4s *) pWt2;
          ${vt_in} x2 = *(${vt_in} *) pIm2Col2;
          sum2 = ${sumdotp_fn}(x2, w2, sum2);
          pWt2 += 4;
          pIm2Col2 += 4;
          v4s w3 = *(v4s *) pWt3;
          ${vt_in} x3 = *(${vt_in} *) pIm2Col3;
          sum3 = ${sumdotp_fn}(x3, w3, sum3);
          pWt3 += 4;
          pIm2Col3 += 4;
          v4s w4 = *(v4s *) pWt4;
          ${vt_in} x4 = *(${vt_in} *) pIm2Col4;
          sum4 = ${sumdotp_fn}(x4, w4, sum4);
          pWt4 += 4;
          pIm2Col4 += 4;
%endif
          j++;
        }while(j<colCnt);
        if(leftCnt)
        {
  %if config.less_precision == 8:
          pWt+=(j << 2);
          pIm2Col+=(j << 2);
  %elif config.less_precision == 4:
          pWt+=(j << 2);
          pWt2+=(j << 2);
          pIm2Col+=(j << 2);
          pIm2Col2+=(j << 2);
  %endif
          do
          {
  %if config.less_precision == 8:
            int8_t w = *(int8_t *) pWt++;
            ${pt_in} x = *(${pt_in} *) pIm2Col++;
            sum += x * w;
  %elif config.less_precision == 4:
            int8_t w = *(int8_t *) pWt++;
            ${pt_in} x = *(${pt_in} *) pIm2Col++;
            sum += x * w;
            int8_t w2 = *(int8_t *) pWt2++;
            ${pt_in} x2 = *(${pt_in} *) pIm2Col2++;
            sum2 += x2 * w2;
  %elif config.less_precision == 2:
            int8_t w = *(int8_t *) pWt++;
            ${pt_in} x = *(${pt_in} *) pIm2Col++;
            sum += x * w;
            int8_t w2 = *(int8_t *) pWt2++;
            ${pt_in} x2 = *(${pt_in} *) pIm2Col2++;
            sum2 += x2 * w2;
            int8_t w3 = *(int8_t *) pWt3++;
            ${pt_in} x3 = *(${pt_in} *) pIm2Col3++;
            sum3 += x3 * w3;
            int8_t w4 = *(int8_t *) pWt4++;
            ${pt_in} x4 = *(${pt_in} *) pIm2Col4++;
            sum4 += x4 * w4;
  %endif
            j++;
          }while(j<leftCnt);
        }
        if (flag_batch_norm && flag_relu)
        {
%if config.less_precision == 8:
          *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
%elif config.less_precision == 4:
  %if config.kernel.out_data_t == 8:
          *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          *(pOutBuffer + 1) = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
  %elif config.kernel.out_data_t == 4:
          sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
  %endif
%elif config.less_precision == 2:
  %if config.kernel.out_data_t == 8:
          *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          *(pOutBuffer + 1) = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          *(pOutBuffer + 2) = ${config.bn_fn}(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
          *(pOutBuffer + 3) = ${config.bn_fn}(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
  %elif config.kernel.out_data_t == 4:
          sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          sum3 = ${config.bn_fn}(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
          sum4 = ${config.bn_fn}(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
          *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
          *(pOutBuffer + 1) = bitins(sum3, n_mask, sum4, mask, off);
  %elif config.kernel.out_data_t == 2:
          sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          sum3 = ${config.bn_fn}(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
          sum4 = ${config.bn_fn}(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
          sum = bitins(sum, n_mask2, sum2, mask2, off2);
          sum = bitins(sum, n_mask4, sum3, mask4, off4);
          *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
  %endif
%endif
        }
        else
        {
          if(flag_relu == 1)
          {
%if config.less_precision == 8:
            *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
%elif config.less_precision == 4:
  %if config.kernel.out_data_t == 8:
            *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
            *(pOutBuffer + 1) = ${config.relu_fn}(sum2, out_mult, out_shift);
  %elif config.kernel.out_data_t == 4:
            sum = ${config.relu_fn}(sum, out_mult, out_shift);
            sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
  %endif
%elif config.less_precision == 2:
  %if config.kernel.out_data_t == 8:
            *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
            *(pOutBuffer + 1) = ${config.relu_fn}(sum2, out_mult, out_shift);
            *(pOutBuffer + 2) = ${config.relu_fn}(sum3, out_mult, out_shift);
            *(pOutBuffer + 3) = ${config.relu_fn}(sum4, out_mult, out_shift);
  %elif config.kernel.out_data_t == 4:
            sum = ${config.relu_fn}(sum, out_mult, out_shift);
            sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
            sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
            sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
            *(pOutBuffer + 1) = bitins(sum3, n_mask, sum4, mask, off);
  %elif config.kernel.out_data_t == 2:
            sum = ${config.relu_fn}(sum, out_mult, out_shift);
            sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
            sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
            sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
  %endif
%endif
          }
          else
          {
%if config.less_precision == 8:
            *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
%elif config.less_precision == 4:
  %if config.kernel.out_data_t == 8:
            *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            *(pOutBuffer + 1) = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
  %elif config.kernel.out_data_t == 4:
            sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
  %endif
%elif config.less_precision == 2:
  %if config.kernel.out_data_t == 8:
            *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            *(pOutBuffer + 1) = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
            *(pOutBuffer + 2) = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
            *(pOutBuffer + 3) = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
  %elif config.kernel.out_data_t == 4:
            sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
            sum3 = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
            sum4 = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
            *(pOutBuffer + 1) = bitins(sum3, n_mask, sum4, mask, off);
  %elif config.kernel.out_data_t == 2:
            sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
            sum3 = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
            sum4 = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
  %endif
%endif
          }
        }
        pOutBuffer+=(dim_out_x * ch_out_r);
        l++;
      }while(l<dim_out_y);
      i_out_x++;
    }while((i_out_x * stride_x) < ((dim_out_x * stride_x) - padding_x_right));
    for (i_out_x; i_out_x < dim_out_x; i_out_x++)
    {
      ${pt_out} *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
      ${pt_in} *pIm2Col = pIm2ColBase;
%if config.less_precision == 4:
      ${pt_in} *pIm2Col2 = pIm2Col + im2col_size;
%elif config.less_precision == 2:
      ${pt_in} *pIm2Col2 = pIm2Col + im2col_size;
      ${pt_in} *pIm2Col3 = pIm2Col2 + im2col_size;
      ${pt_in} *pIm2Col4 = pIm2Col3 + im2col_size;
%endif
      asm volatile ("":::"memory");
      i_buff_y = - padding_y_top;
      if(padding_y_top > 0)
      {
        do
        {
          int i=0;
          do
          {
            *(${vt_in} *) pIm2Col = (${vt_in}) {0, 0, 0, 0};
            pIm2Col+=4;
  %if config.less_precision == 4:
            *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col2+=4;
  %elif config.less_precision == 2:
            *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col2+=4;
            *(${vt_in} *) pIm2Col3 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col3+=4;
            *(${vt_in} *) pIm2Col4 = (${vt_in}) {0, 0, 0, 0};
            pIm2Col4+=4;
  %endif
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
  %if config.less_precision == 4:
          pIm2Col2-=dim_incr;
  %elif config.less_precision == 2:
          pIm2Col2-=dim_incr;
          pIm2Col3-=dim_incr;
          pIm2Col4-=dim_incr;
  %endif
          i_buff_y++;
        }while(i_buff_y < 0);
      }
      int base_ptr = pIn + i_in_ch + (i_out_x * stride_x) - padding_x_left;
      do
      {
        int i = 0;
        int idx = 0;
        do
        {
%if config.less_precision == 8:
          *((${vt_in}*) pIm2Col) = *((${vt_in}*) (base_ptr + idx));
          pIm2Col+=4;
%elif config.less_precision == 4:
  %if config.kernel.in_data_t == 8:
          *((${vt_in}*) pIm2Col) = *((${vt_in}*) (base_ptr + idx));
          pIm2Col+=4;
          *((${vt_in}*) pIm2Col2) = *((${vt_in}*) (base_ptr + idx + in_image_size));
          pIm2Col2+=4;
  %elif config.kernel.in_data_t == 4:
          ${vt_in} src_in = *((${vt_in}*) (base_ptr + idx));
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 0);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 4);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 8);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 12);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 16);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 20);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 24);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 28);
          pIm2Col++;
          pIm2Col2++;
  %endif
%elif config.less_precision == 2:
  %if config.kernel.in_data_t == 8:
          int idc = in_image_size;
          *((${vt_in}*) pIm2Col) = *((${vt_in}*) (base_ptr + idx));
          pIm2Col+=4;
          *((${vt_in}*) pIm2Col2) = *((${vt_in}*) (base_ptr + idx + idc));
          pIm2Col2+=4;
          idc+=in_image_size;
          *((${vt_in}*) pIm2Col3) = *((${vt_in}*) (base_ptr + idx + idc));
          pIm2Col3+=4;
          idc+=in_image_size;
          *((${vt_in}*) pIm2Col4) = *((${vt_in}*) (base_ptr + idx + idc));
          pIm2Col4+=4;
  %elif config.kernel.in_data_t == 4:
          ${vt_in} src_in = *((${vt_in}*) (base_ptr + idx));
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 0);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 4);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 8);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 12);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 16);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 20);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 24);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 28);
          pIm2Col++;
          pIm2Col2++;
          src_in = *((${vt_in}*) (base_ptr + idx + in_image_size));
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 0);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 4);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 8);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 12);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 16);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 20);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 24);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 4, 28);
          pIm2Col3++;
          pIm2Col4++;
  %elif config.kernel.in_data_t == 2:
          ${vt_in} src_in = *((${vt_in}*) (base_ptr + idx));
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 0);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 2);
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 4);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 6);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 8);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 10);
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 12);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 14);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 16);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 18);
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 20);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 22);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 24);
          *pIm2Col2 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 26);
          *pIm2Col3 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 28);
          *pIm2Col4 = (${pt_in}) ${bex}((${int_t_in}) src_in, 2, 30);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
  %endif
%endif
          idx+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);

        pIm2Col-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
%if config.less_precision == 4:
        pIm2Col2-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
%elif config.less_precision == 2:
        pIm2Col2-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        pIm2Col3-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        pIm2Col4-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
%endif
        base_ptr+=dim_in_x;
        for(int j=0; j<(1 + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right); j++)
        {
          *(${pt_in} *) pIm2Col = 0;
          pIm2Col++;
%if config.less_precision == 4:
          *(${pt_in} *) pIm2Col2 = 0;
          pIm2Col2++;
%elif config.less_precision == 2:
          *(${pt_in} *) pIm2Col2 = 0;
          pIm2Col2++;
          *(${pt_in} *) pIm2Col3 = 0;
          pIm2Col3++;
          *(${pt_in} *) pIm2Col4 = 0;
          pIm2Col4++;
%endif
        }
        i_buff_y++;
      }while(i_buff_y < dim_in_y);
      for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
      {
        int i=0;
        do
        {
          *(${vt_in} *) pIm2Col = (${vt_in}) {0, 0, 0, 0};
          pIm2Col+=4;
%if config.less_precision == 4:
          *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
          pIm2Col2+=4;
%elif config.less_precision == 2:
          *(${vt_in} *) pIm2Col2 = (${vt_in}) {0, 0, 0, 0};
          pIm2Col2+=4;
          *(${vt_in} *) pIm2Col3 = (${vt_in}) {0, 0, 0, 0};
          pIm2Col3+=4;
          *(${vt_in} *) pIm2Col4 = (${vt_in}) {0, 0, 0, 0};
          pIm2Col4+=4;
%endif
          i++;
        }while(i<dim_kernel_x_size_padded);
        pIm2Col-=dim_incr;
%if config.less_precision == 4:
        pIm2Col2-=dim_incr;
%elif config.less_precision == 2:
        pIm2Col2-=dim_incr;
        pIm2Col3-=dim_incr;
        pIm2Col4-=dim_incr;
%endif
      }

      int l=0;
      do
      {
%if config.less_precision == 8:
  %if config.kernel.wt_data_t == 8:
        pWt = pWeight + i_wt_ch;
  %else:
        pWt = pWtBase;
  %endif
        int sum = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
%elif config.less_precision == 4:
  %if config.kernel.wt_data_t == 8:
        pWt = pWeight + i_wt_ch;
  %else:
        pWt = pWtBase;
  %endif
        pWt2 = pWt + kernel_size;
        int sum = 0;
        int sum2 = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
          sum2 = ((int) (pBias[i_ch + 1]));
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        pIm2Col2 = pIm2Col + im2col_size;
%elif config.less_precision == 2:
  %if config.kernel.wt_data_t == 8:
        pWt = pWeight + i_wt_ch;
  %else:
        pWt = pWtBase;
  %endif
        pWt2 = pWt + kernel_size;
        pWt3 = pWt2 + kernel_size;
        pWt4 = pWt3 + kernel_size;
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
          sum2 = ((int) (pBias[i_ch + 1]));
          sum3 = ((int) (pBias[i_ch + 2]));
          sum4 = ((int) (pBias[i_ch + 3]));
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        pIm2Col2 = pIm2Col + im2col_size;
        pIm2Col3 = pIm2Col2 + im2col_size;
        pIm2Col4 = pIm2Col3 + im2col_size;
%endif
%if config.less_precision == 8:
        int32_t ptrA  = (int32_t *) pWt;
        ${int_t_in} ptrB = (${int_t_in} *) pIm2Col;

        ptrA = MacLoadInit(1, 0, 0, 0, ptrA);
        ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
%elif config.less_precision == 4:
        int32_t  ptrA  = (int32_t *) pWt;
        int32_t  ptrA2 = (int32_t *) pWt2;
        ${int_t_in} ptrB  = (${int_t_in} *) pIm2Col;
        ${int_t_in} ptrB2 = (${int_t_in} *) pIm2Col2;

        ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
        ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
        ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
        ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);
%endif
        int j=0;
        do
        {
%if config.less_precision == 8:
          sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
          ptrA = MacLoadUpdate(ptrA);
          ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
%elif config.less_precision == 4:
          sum  = ${macload_fn}(1, 0, 0, 0, ptrA, sum);
          ptrA = MacLoadUpdate(ptrA);
          ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
          sum2  = ${macload_fn}(1, 0, 1, 1, ptrA2, sum2);
          ptrA2 = MacLoadUpdate(ptrA2);
          ptrB2 = MacLoadInit(0, 1, 1, 1, ptrB2);
%elif config.less_precision == 2:
          v4s w = *(v4s *) pWt;
          ${vt_in} x = *(${vt_in} *) pIm2Col;
          sum = ${sumdotp_fn}(x, w, sum);
          pWt += 4;
          pIm2Col += 4;
          v4s w2 = *(v4s *) pWt2;
          ${vt_in} x2 = *(${vt_in} *) pIm2Col2;
          sum2 = ${sumdotp_fn}(x2, w2, sum2);
          pWt2 += 4;
          pIm2Col2 += 4;
          v4s w3 = *(v4s *) pWt3;
          ${vt_in} x3 = *(${vt_in} *) pIm2Col3;
          sum3 = ${sumdotp_fn}(x3, w3, sum3);
          pWt3 += 4;
          pIm2Col3 += 4;
          v4s w4 = *(v4s *) pWt4;
          ${vt_in} x4 = *(${vt_in} *) pIm2Col4;
          sum4 = ${sumdotp_fn}(x4, w4, sum4);
          pWt4 += 4;
          pIm2Col4 += 4;
%endif
          j++;
        }while(j<colCnt);
        if(leftCnt)
        {
  %if config.less_precision == 8:
          pWt+=(j << 2);
          pIm2Col+=(j << 2);
  %elif config.less_precision == 4:
          pWt+=(j << 2);
          pWt2+=(j << 2);
          pIm2Col+=(j << 2);
          pIm2Col2+=(j << 2);
  %endif
          do
          {
  %if config.less_precision == 8:
            int8_t w = *(int8_t *) pWt++;
            ${pt_in} x = *(${pt_in} *) pIm2Col++;
            sum += x * w;
  %elif config.less_precision == 4:
            int8_t w = *(int8_t *) pWt++;
            ${pt_in} x = *(${pt_in} *) pIm2Col++;
            sum += x * w;
            int8_t w2 = *(int8_t *) pWt2++;
            ${pt_in} x2 = *(${pt_in} *) pIm2Col2++;
            sum2 += x2 * w2;
  %elif config.less_precision == 2:
            int8_t w = *(int8_t *) pWt++;
            ${pt_in} x = *(${pt_in} *) pIm2Col++;
            sum += x * w;
            int8_t w2 = *(int8_t *) pWt2++;
            ${pt_in} x2 = *(${pt_in} *) pIm2Col2++;
            sum2 += x2 * w2;
            int8_t w3 = *(int8_t *) pWt3++;
            ${pt_in} x3 = *(${pt_in} *) pIm2Col3++;
            sum3 += x3 * w3;
            int8_t w4 = *(int8_t *) pWt4++;
            ${pt_in} x4 = *(${pt_in} *) pIm2Col4++;
            sum4 += x4 * w4;
  %endif
            j++;
          }while(j<leftCnt);
        }
        if (flag_batch_norm && flag_relu)
        {
%if config.less_precision == 8:
          *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
%elif config.less_precision == 4:
  %if config.kernel.out_data_t == 8:
          *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          *(pOutBuffer + 1) = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
  %elif config.kernel.out_data_t == 4:
          sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
  %endif
%elif config.less_precision == 2:
  %if config.kernel.out_data_t == 8:
          *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          *(pOutBuffer + 1) = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          *(pOutBuffer + 2) = ${config.bn_fn}(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
          *(pOutBuffer + 3) = ${config.bn_fn}(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
  %elif config.kernel.out_data_t == 4:
          sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          sum3 = ${config.bn_fn}(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
          sum4 = ${config.bn_fn}(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
          *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
          *(pOutBuffer + 1) = bitins(sum3, n_mask, sum4, mask, off);
  %elif config.kernel.out_data_t == 2:
          sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
          sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          sum3 = ${config.bn_fn}(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
          sum4 = ${config.bn_fn}(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
          sum = bitins(sum, n_mask2, sum2, mask2, off2);
          sum = bitins(sum, n_mask4, sum3, mask4, off4);
          *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
  %endif
%endif
        }
        else
        {
          if(flag_relu == 1)
          {
%if config.less_precision == 8:
            *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
%elif config.less_precision == 4:
  %if config.kernel.out_data_t == 8:
            *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
            *(pOutBuffer + 1) = ${config.relu_fn}(sum2, out_mult, out_shift);
  %elif config.kernel.out_data_t == 4:
            sum = ${config.relu_fn}(sum, out_mult, out_shift);
            sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
  %endif
%elif config.less_precision == 2:
  %if config.kernel.out_data_t == 8:
            *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
            *(pOutBuffer + 1) = ${config.relu_fn}(sum2, out_mult, out_shift);
            *(pOutBuffer + 2) = ${config.relu_fn}(sum3, out_mult, out_shift);
            *(pOutBuffer + 3) = ${config.relu_fn}(sum4, out_mult, out_shift);
  %elif config.kernel.out_data_t == 4:
            sum = ${config.relu_fn}(sum, out_mult, out_shift);
            sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
            sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
            sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
            *(pOutBuffer + 1) = bitins(sum3, n_mask, sum4, mask, off);
  %elif config.kernel.out_data_t == 2:
            sum = ${config.relu_fn}(sum, out_mult, out_shift);
            sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
            sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
            sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
  %endif
%endif
          }
          else
          {
%if config.less_precision == 8:
            *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
%elif config.less_precision == 4:
  %if config.kernel.out_data_t == 8:
            *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            *(pOutBuffer + 1) = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
  %elif config.kernel.out_data_t == 4:
            sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
  %endif
%elif config.less_precision == 2:
  %if config.kernel.out_data_t == 8:
            *pOutBuffer = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            *(pOutBuffer + 1) = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
            *(pOutBuffer + 2) = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
            *(pOutBuffer + 3) = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
  %elif config.kernel.out_data_t == 4:
            sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
            sum3 = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
            sum4 = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
            *(pOutBuffer + 1) = bitins(sum3, n_mask, sum4, mask, off);
  %elif config.kernel.out_data_t == 2:
            sum = (${pt_out}) ${out_clip_fn}(sum >> out_shift);
            sum2 = (${pt_out}) ${out_clip_fn}(sum2 >> out_shift);
            sum3 = (${pt_out}) ${out_clip_fn}(sum3 >> out_shift);
            sum4 = (${pt_out}) ${out_clip_fn}(sum4 >> out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
  %endif
%endif
          }
        }
        pOutBuffer+=(dim_out_x * ch_out_r);
        l++;
      }while(l<dim_out_y);
    }
%if config.less_precision == 8:
    i_in_ch+=in_image_size;
%elif config.less_precision == 4:
  %if config.kernel.in_data_t == 8:
    i_in_ch+=(in_image_size << 1);
  %elif config.kernel.in_data_t == 4:
    i_in_ch+=in_image_size;
  %endif
%elif config.less_precision == 2:
  %if config.kernel.in_data_t == 8:
    i_in_ch+=(in_image_size << 2);
  %elif config.kernel.in_data_t == 4:
    i_in_ch+=(in_image_size << 1);
  %elif config.kernel.in_data_t == 2:
    i_in_ch+=in_image_size;
  %endif
%endif
%if config.less_precision == 8:
    i_wt_ch+=kernel_size;
%elif config.less_precision == 4:
  %if config.kernel.wt_data_t == 8:
    i_wt_ch+=(kernel_size << 1);
  %elif config.kernel.wt_data_t == 4:
    i_wt_ch+=kernel_size;
  %endif
%elif config.less_precision == 2:
  %if config.kernel.wt_data_t == 8:
    i_wt_ch+=(kernel_size << 2);
  %elif config.kernel.wt_data_t == 4:
    i_wt_ch+=(kernel_size << 1);
  %elif config.kernel.wt_data_t == 2:
    i_wt_ch+=kernel_size;
  %endif
%endif
%if config.less_precision == 8:
    k1++;
    lambda1++;
    i_out_ch++;
%elif config.less_precision == 4:
    k1+=2;
    lambda1+=2;
  %if config.kernel.out_data_t == 8:
    i_out_ch+=2;
  %elif config.kernel.out_data_t == 4:
    i_out_ch++;
  %endif
%elif config.less_precision == 2:
    k1+=4;
    lambda1+=4;
  %if config.kernel.out_data_t == 8:
    i_out_ch+=4;
  %elif config.kernel.out_data_t == 4:
    i_out_ch+=2;
  %elif config.kernel.out_data_t == 2:
    i_out_ch++;
  %endif
%endif
  }
  pi_cl_team_barrier();
}
