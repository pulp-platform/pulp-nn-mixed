/*
 * xpulp_nn_depthwise_u4_u2_i8.c
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


void xpulp_nn_depthwise_u4_u2_i8(
                        uint8_t *pIn,
                        uint8_t *pIm2ColBuffer,
                        int8_t *pBias,
                        uint8_t *pOut,
                        int8_t *pWeight,
                        int8_t *pWtBuffer,
                        int32_t *pKappa,
                        int32_t *pLambda,
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

  uint16_t ch_out_r = ch_out >> 2;
  uint16_t ch_in_r = ch_out >> 1;
  uint16_t ch_wt_r = ch_out;

  uint16_t ch_min = ch_out >> 2;

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

  uint8_t * pIm2ColBase = pIm2ColBuffer + (core_id * im2col_size << 2);
  int8_t * pWtBase = pWtBuffer + (core_id * (kernel_size << 2));

  int i_out_x, i_buff_y;
  uint16_t colCnt = kernel_size >> 2;
  uint16_t leftCnt = kernel_size & 0x3;

  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;

  int i_out_ch = start_channel;
  int i_in_ch = (start_channel << 1) * in_image_size;
  int i_wt_ch = (start_channel << 2) * kernel_size;


  int32_t * k1 = pKappa + core_id * (chunk << 2);
  int32_t * lambda1 = pLambda + core_id * (chunk << 2);

  for(int i_ch = start_channel; i_ch < stop_channel; i_ch++)
  {
    i_out_x = 0;
    int8_t * pWt = pWeight + i_wt_ch;
    int8_t * pWt2 = pWt + kernel_size;
    int8_t * pWt3 = pWt2 + kernel_size;
    int8_t * pWt4 = pWt3 + kernel_size;
    if(padding_x_left > 0)
    {
      do
      {
        uint8_t *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
        uint8_t *pOutBuffer2 = pOutBuffer + ((dim_out_y>>1) * (dim_out_x * ch_out_r));
        uint8_t *pIm2Col = pIm2ColBase;
        uint8_t *pIm2Col2 = pIm2Col + im2col_size;
        uint8_t *pIm2Col3 = pIm2Col2 + im2col_size;
        uint8_t *pIm2Col4 = pIm2Col3 + im2col_size;
        i_buff_y = - padding_y_top;
        if(padding_y_top > 0)
        {
          do
          {
            int i=0;
            do
            {
              *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
              pIm2Col+=4;
              *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
              pIm2Col2+=4;
              *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
              pIm2Col3+=4;
              *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
              pIm2Col4+=4;
              i++;
            }while(i<dim_kernel_x_size_padded);
            pIm2Col-=dim_incr;
            pIm2Col2-=dim_incr;
            pIm2Col3-=dim_incr;
            pIm2Col4-=dim_incr;
            i_buff_y++;
          }while(i_buff_y < 0);
        }
        int const1 = (i_out_x * stride_x);
        int base_ptr = pIn + i_in_ch;
        do
        {
          for(int j=0; j< (padding_x_left - const1); j++)
          {
            *(uint8_t *) pIm2Col = 0;
            pIm2Col++;
            *(uint8_t *) pIm2Col2 = 0;
            pIm2Col2++;
            *(uint8_t *) pIm2Col3 = 0;
            pIm2Col3++;
            *(uint8_t *) pIm2Col4 = 0;
            pIm2Col4++;
          }
          int idx = 0;
          int i = 0;
          do
          {
            v4u src_in = *((v4u*) (base_ptr + idx));
            *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 0);
            *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 4);
            pIm2Col++;
            pIm2Col2++;
            *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 8);
            *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 12);
            pIm2Col++;
            pIm2Col2++;
            *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 16);
            *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 20);
            pIm2Col++;
            pIm2Col2++;
            *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 24);
            *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 28);
            pIm2Col++;
            pIm2Col2++;
            src_in = *((v4u*) (base_ptr + idx + in_image_size));
            *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 0);
            *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 4);
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 8);
            *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 12);
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 16);
            *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 20);
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 24);
            *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 28);
            pIm2Col3++;
            pIm2Col4++;
            idx+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=(dim_incr_pad_left - const1);
          pIm2Col2-=(dim_incr_pad_left - const1);
          pIm2Col3-=(dim_incr_pad_left - const1);
          pIm2Col4-=(dim_incr_pad_left - const1);
          base_ptr+=dim_in_x;
          i_buff_y++;
        }while(i_buff_y < dim_in_y);
        for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
        {
          int i=0;
          do
          {
            *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
            pIm2Col+=4;
            *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
            pIm2Col2+=4;
            *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
            pIm2Col3+=4;
            *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
            pIm2Col4+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
          pIm2Col2-=dim_incr;
          pIm2Col3-=dim_incr;
          pIm2Col4-=dim_incr;
        }

        int l=0;
        do
        {
          pWt = pWeight + i_wt_ch;
          pWt2 = pWt + kernel_size;
          pWt3 = pWt2 + kernel_size;
          pWt4 = pWt3 + kernel_size;
          int sum = 0;
          int sum2 = 0;
          int sum3 = 0;
          int sum4 = 0;
          int sum5 = 0;
          int sum6 = 0;
          int sum7 = 0;
          int sum8 = 0;
          if (pBias != NULL)
          {
            sum = ((int) (pBias[i_ch]));
            sum2 = ((int) (pBias[i_ch + 1]));
            sum3 = ((int) (pBias[i_ch + 2]));
            sum4 = ((int) (pBias[i_ch + 3]));
            sum5 = sum;
            sum6 = sum2;
            sum7 = sum3;
            sum8 = sum4;
          }
          pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
          pIm2Col2 = pIm2Col + im2col_size;
          pIm2Col3 = pIm2Col2 + im2col_size;
          pIm2Col4 = pIm2Col3 + im2col_size;
          int32_t ptrA = (int32_t *) pWt;
          int32_t ptrA2 = (int32_t *) pWt2;
          int32_t ptrA3 = (int32_t *) pWt3;
          int32_t ptrA4 = (int32_t *) pWt4;
          uint32_t ptrB = (uint32_t *) pIm2Col;
          uint32_t ptrB_2 = (uint32_t *) (pIm2Col + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
          uint32_t ptrB2 = (uint32_t *) pIm2Col2;
          uint32_t ptrB2_2 = (uint32_t *) (pIm2Col2 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
          uint32_t ptrB3 = (uint32_t *) pIm2Col3;
          uint32_t ptrB3_2 = (uint32_t *) (pIm2Col3 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
          uint32_t ptrB4 = (uint32_t *) pIm2Col4; 
          uint32_t ptrB4_2 = (uint32_t *) (pIm2Col4 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));

          ptrA = MacLoadInit(1, 0, 0, 0, ptrA);
          ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
          ptrA3 = MacLoadInit(1, 0, 2, 0, ptrA3);
          ptrA4 = MacLoadInit(1, 0, 3, 0, ptrA4);
          ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
          int j=0;
          do
          {
            ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);
            sum = MacLoad4(0, 1, 0, 0, ptrB3, sum);
            ptrB3 = MacLoadUpdate(ptrB3);
            sum2 = MacLoad4(0, 1, 1, 1, ptrB4, sum2);
            ptrB4 = MacLoadUpdate(ptrB4);
            sum3 = MacLoad4(0, 1, 2, 0, ptrB_2, sum3);
            ptrB_2 = MacLoadUpdate(ptrB_2);
            sum4 = MacLoad4(0, 1, 3, 1, ptrB2_2, sum4);
            ptrB2_2 = MacLoadUpdate(ptrB2_2);
            sum5 = MacLoad4(0, 1, 0, 0, ptrB3_2, sum5);
            ptrB3_2 = MacLoadUpdate(ptrB3_2);
            sum6 = MacLoad4(0, 1, 1, 1, ptrB4_2, sum6);
            ptrB4_2 = MacLoadUpdate(ptrB4_2);
            sum7 = MacLoad4(0, 1, 2, 0, ptrB, sum7);
            ptrB = MacLoadUpdate(ptrB);
            sum8 = MacLoad4(1, 0, 3, 1, ptrA4, sum8);
            ptrA4 = MacLoadUpdate(ptrA4);
            ptrA = MacLoadInit(1, 0, 0, 0, ptrA);
            ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
            ptrA3 = MacLoadInit(1, 0, 2, 0, ptrA3);
            j++;
          }while(j<colCnt);
          if(leftCnt)
          {
            pWt+=(j << 2);
            pWt2+=(j << 2);
            pWt3+=(j << 2);
            pWt4+=(j << 2);
            pIm2Col+=(j << 2);
            pIm2Col2+=(j << 2);
            pIm2Col3+=(j << 2);
            pIm2Col4+=(j << 2);
            do
            {
              int8_t w = *(int8_t *) pWt++;
              uint8_t x = *(uint8_t *) pIm2Col++;
              uint8_t x_2 = *(uint8_t *) (pIm2Col - 1 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
              sum += x * w;
              sum5 += x_2 * w;
              int8_t w2 = *(int8_t *) pWt2++;
              uint8_t x2 = *(uint8_t *) pIm2Col2++;
              uint8_t x2_2 = *(uint8_t *) (pIm2Col2 - 1 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
              sum2 += x2 * w2;
              sum6 += x2_2 * w2;
              int8_t w3 = *(int8_t *) pWt3++;
              uint8_t x3 = *(uint8_t *) pIm2Col3++;
              uint8_t x3_2 = *(uint8_t *) (pIm2Col3 - 1 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
              sum3 += x3 * w3;
              sum7 += x3_2 * w3;
              int8_t w4 = *(int8_t *) pWt4++;
              uint8_t x4 = *(uint8_t *) pIm2Col4++;
              uint8_t x4_2 = *(uint8_t *) (pIm2Col4 - 1 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
              sum4 += x4 * w4;
              sum8 += x4_2 * w4;
              j++;
            }while(j<leftCnt);
          }
          if (flag_batch_norm && flag_relu)
          {
            sum = pulp_nn_bn_quant_u2(sum, *k1, *lambda1, out_shift);
            sum2 = pulp_nn_bn_quant_u2(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
            sum3 = pulp_nn_bn_quant_u2(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
            sum4 = pulp_nn_bn_quant_u2(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
            sum5 = pulp_nn_bn_quant_u2(sum5, *k1, *lambda1, out_shift);
            sum6 = pulp_nn_bn_quant_u2(sum6, *(k1 + 1), *(lambda1 + 1), out_shift);
            sum7 = pulp_nn_bn_quant_u2(sum7, *(k1 + 2), *(lambda1 + 2), out_shift);
            sum8 = pulp_nn_bn_quant_u2(sum8, *(k1 + 3), *(lambda1 + 3), out_shift);
            sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
            sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
            *pOutBuffer2 = bitins(sum5, n_mask6, sum8, mask6, off6);
          }
          else
          {
            if(flag_relu == 1)
            {
              sum = pulp_nn_quant_u2(sum, out_mult, out_shift);
              sum2 = pulp_nn_quant_u2(sum2, out_mult, out_shift);
              sum3 = pulp_nn_quant_u2(sum3, out_mult, out_shift);
              sum4 = pulp_nn_quant_u2(sum4, out_mult, out_shift);
              sum = bitins(sum, n_mask2, sum2, mask2, off2);
              sum = bitins(sum, n_mask4, sum3, mask4, off4);
              *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
              sum5 = pulp_nn_quant_u2(sum5, out_mult, out_shift);
              sum6 = pulp_nn_quant_u2(sum6, out_mult, out_shift);
              sum7 = pulp_nn_quant_u2(sum7, out_mult, out_shift);
              sum8 = pulp_nn_quant_u2(sum8, out_mult, out_shift);
              sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
              sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
              *pOutBuffer2 = bitins(sum5, n_mask6, sum8, mask6, off6);
            }
            else
            {
              sum = (uint8_t) clip2(sum >> out_shift);
              sum2 = (uint8_t) clip2(sum2 >> out_shift);
              sum3 = (uint8_t) clip2(sum3 >> out_shift);
              sum4 = (uint8_t) clip2(sum4 >> out_shift);
              sum = bitins(sum, n_mask2, sum2, mask2, off2);
              sum = bitins(sum, n_mask4, sum3, mask4, off4);
              *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
              sum5 = (uint8_t) clip2(sum5 >> out_shift);
              sum6 = (uint8_t) clip2(sum6 >> out_shift);
              sum7 = (uint8_t) clip2(sum7 >> out_shift);
              sum8 = (uint8_t) clip2(sum8 >> out_shift);
              sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
              sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
              *pOutBuffer2 = bitins(sum5, n_mask6, sum8, mask6, off6);
            }
          }
          pOutBuffer+=(dim_out_x * ch_out_r);
          pOutBuffer2+=(dim_out_x * ch_out_r);
          l++;
        }while(l<(dim_out_y>>1));
        i_out_x++;
      }while((i_out_x * stride_x) < padding_x_left);
    }
    do
    {
      uint8_t *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
      uint8_t *pOutBuffer2 = pOutBuffer + ((dim_out_y >> 1) * (dim_out_x * ch_out_r));
      uint8_t *pIm2Col = pIm2ColBase;
      uint8_t *pIm2Col2 = pIm2Col + im2col_size;
      uint8_t *pIm2Col3 = pIm2Col2 + im2col_size;
      uint8_t *pIm2Col4 = pIm2Col3 + im2col_size;
      i_buff_y = - padding_y_top;
      if(padding_y_top > 0)
      {
        do
        {
          int i=0;
          do
          {
            *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
            pIm2Col+=4;
            *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
            pIm2Col2+=4;
            *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
            pIm2Col3+=4;
            *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
            pIm2Col4+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
          pIm2Col2-=dim_incr;
          pIm2Col3-=dim_incr;
          pIm2Col4-=dim_incr;
          i_buff_y++;
        }while(i_buff_y < 0);
      }
      int base_ptr = pIn + i_in_ch + (i_out_x * stride_x) - padding_x_left;
      do
      {
        int idx = 0;
        for (int i=0; i<dim_kernel_x_size_padded; i++)
        {
          v4u src_in = *((v4u*) (base_ptr + idx));
          *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 0);
          *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 4);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 8);
          *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 12);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 16);
          *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 20);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 24);
          *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 28);
          pIm2Col++;
          pIm2Col2++;
          src_in = *((v4u*) (base_ptr + idx + in_image_size));
          *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 0);
          *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 4);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 8);
          *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 12);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 16);
          *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 20);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 24);
          *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 28);
          pIm2Col3++;
          pIm2Col4++;
          idx+=4;
        }
        pIm2Col-=dim_incr;
        pIm2Col2-=dim_incr;
        pIm2Col3-=dim_incr;
        pIm2Col4-=dim_incr;
        base_ptr+=dim_in_x;
        i_buff_y++;
      }while(i_buff_y < dim_in_y);
      for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
      {
        int i=0;
        do
        {
          *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
          pIm2Col+=4;
          *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
          pIm2Col2+=4;
          *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
          pIm2Col3+=4;
          *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
          pIm2Col4+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);
        pIm2Col-=dim_incr;
        pIm2Col2-=dim_incr;
        pIm2Col3-=dim_incr;
        pIm2Col4-=dim_incr;
      }
      int l=0;
      do
      {
        pWt = pWeight + i_wt_ch;
        pWt2 = pWt + kernel_size;
        pWt3 = pWt2 + kernel_size;
        pWt4 = pWt3 + kernel_size;
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        int sum5 = 0;
        int sum6 = 0;
        int sum7 = 0;
        int sum8 = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
          sum2 = ((int) (pBias[i_ch + 1]));
          sum3 = ((int) (pBias[i_ch + 2]));
          sum4 = ((int) (pBias[i_ch + 3]));
          sum5 = sum;
          sum6 = sum2;
          sum7 = sum3;
          sum8 = sum4;
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        pIm2Col2 = pIm2Col + im2col_size;
        pIm2Col3 = pIm2Col2 + im2col_size;
        pIm2Col4 = pIm2Col3 + im2col_size;
        int32_t ptrA = (int32_t *) pWt;
        int32_t ptrA2 = (int32_t *) pWt2;
        int32_t ptrA3 = (int32_t *) pWt3;
        int32_t ptrA4 = (int32_t *) pWt4;
        uint32_t ptrB = (uint32_t *) pIm2Col;
        uint32_t ptrB_2 = (uint32_t *) (pIm2Col + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
        uint32_t ptrB2 = (uint32_t *) pIm2Col2;
        uint32_t ptrB2_2 = (uint32_t *) (pIm2Col2 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
        uint32_t ptrB3 = (uint32_t *) pIm2Col3;
        uint32_t ptrB3_2 = (uint32_t *) (pIm2Col3 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
        uint32_t ptrB4 = (uint32_t *) pIm2Col4; 
        uint32_t ptrB4_2 = (uint32_t *) (pIm2Col4 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));

        ptrA = MacLoadInit(1, 0, 0, 0, ptrA);
        ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
        ptrA3 = MacLoadInit(1, 0, 2, 0, ptrA3);
        ptrA4 = MacLoadInit(1, 0, 3, 0, ptrA4);
        ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
        int j=0;
        do
        {
          ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);
          sum = MacLoad4(0, 1, 0, 0, ptrB3, sum);
          ptrB3 = MacLoadUpdate(ptrB3);
          sum2 = MacLoad4(0, 1, 1, 1, ptrB4, sum2);
          ptrB4 = MacLoadUpdate(ptrB4);
          sum3 = MacLoad4(0, 1, 2, 0, ptrB_2, sum3);
          ptrB_2 = MacLoadUpdate(ptrB_2);
          sum4 = MacLoad4(0, 1, 3, 1, ptrB2_2, sum4);
          ptrB2_2 = MacLoadUpdate(ptrB2_2);
          sum5 = MacLoad4(0, 1, 0, 0, ptrB3_2, sum5);
          ptrB3_2 = MacLoadUpdate(ptrB3_2);
          sum6 = MacLoad4(0, 1, 1, 1, ptrB4_2, sum6);
          ptrB4_2 = MacLoadUpdate(ptrB4_2);
          sum7 = MacLoad4(0, 1, 2, 0, ptrB, sum7);
          ptrB = MacLoadUpdate(ptrB);
          sum8 = MacLoad4(1, 0, 3, 1, ptrA4, sum8);
          ptrA4 = MacLoadUpdate(ptrA4);
          ptrA = MacLoadInit(1, 0, 0, 0, ptrA);
          ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
          ptrA3 = MacLoadInit(1, 0, 2, 0, ptrA3);
          j++;
        }while(j<colCnt);
        if(leftCnt)
        {
          pWt+=(j << 2);
          pWt2+=(j << 2);
          pWt3+=(j << 2);
          pWt4+=(j << 2);
          pIm2Col+=(j << 2);
          pIm2Col2+=(j << 2);
          pIm2Col3+=(j << 2);
          pIm2Col4+=(j << 2);
          do
          {
            int8_t w = *(int8_t *) pWt++;
            uint8_t x = *(uint8_t *) pIm2Col++;
            uint8_t x_2 = *(uint8_t *) (pIm2Col - 1 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
            sum += x * w;
            sum5 += x_2 * w;
            int8_t w2 = *(int8_t *) pWt2++;
            uint8_t x2 = *(uint8_t *) pIm2Col2++;
            uint8_t x2_2 = *(uint8_t*) (pIm2Col2 - 1 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
            sum2 += x2 * w2;
            sum6 += x2_2 * w2;
            int8_t w3 = *(int8_t *) pWt3++;
            uint8_t x3 = *(uint8_t *) pIm2Col3++;
            uint8_t x3_2 = *(uint8_t *) (pIm2Col3 - 1 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
            sum3 += x3 * w3;
            sum7 += x3_2 * w3;
            int8_t w4 = *(int8_t *) pWt4++;
            uint8_t x4 = *(uint8_t *) pIm2Col4++;
            uint8_t x4_2 = *(uint8_t *) (pIm2Col4 - 1 + ((dim_in_y >> 1) * stride_y * dim_kernel_x));
            sum4 += x4 * w4;
            sum8 += x4_2 * w4;
            j++;
          }while(j<leftCnt);
        }
        if (flag_batch_norm && flag_relu)
        {
          sum = pulp_nn_bn_quant_u2(sum, *k1, *lambda1, out_shift);
          sum2 = pulp_nn_bn_quant_u2(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          sum3 = pulp_nn_bn_quant_u2(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
          sum4 = pulp_nn_bn_quant_u2(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
          sum = bitins(sum, n_mask2, sum2, mask2, off2);
          sum = bitins(sum, n_mask4, sum3, mask4, off4);
          *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
          sum5 = pulp_nn_bn_quant_u2(sum5, *k1, *lambda1, out_shift);
          sum6 = pulp_nn_bn_quant_u2(sum6, *(k1 + 1), *(lambda1 + 1), out_shift);
          sum7 = pulp_nn_bn_quant_u2(sum7, *(k1 + 2), *(lambda1 + 2), out_shift);
          sum8 = pulp_nn_bn_quant_u2(sum8, *(k1 + 3), *(lambda1 + 3), out_shift);
          sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
          sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
          *pOutBuffer2 = bitins(sum5, n_mask6, sum8, mask6, off6);
        }
        else
        {
          if(flag_relu == 1)
          {
            sum = pulp_nn_quant_u2(sum, out_mult, out_shift);
            sum2 = pulp_nn_quant_u2(sum2, out_mult, out_shift);
            sum3 = pulp_nn_quant_u2(sum3, out_mult, out_shift);
            sum4 = pulp_nn_quant_u2(sum4, out_mult, out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
            sum5 = pulp_nn_quant_u2(sum5, out_mult, out_shift);
            sum6 = pulp_nn_quant_u2(sum6, out_mult, out_shift);
            sum7 = pulp_nn_quant_u2(sum7, out_mult, out_shift);
            sum8 = pulp_nn_quant_u2(sum8, out_mult, out_shift);
            sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
            sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
            *pOutBuffer2 = bitins(sum5, n_mask6, sum8, mask6, off6);
          }
          else
          {
            sum = (uint8_t) clip2(sum >> out_shift);
            sum2 = (uint8_t) clip2(sum2 >> out_shift);
            sum3 = (uint8_t) clip2(sum3 >> out_shift);
            sum4 = (uint8_t) clip2(sum4 >> out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
            sum5 = (uint8_t) clip2(sum5 >> out_shift);
            sum6 = (uint8_t) clip2(sum6 >> out_shift);
            sum7 = (uint8_t) clip2(sum7 >> out_shift);
            sum8 = (uint8_t) clip2(sum8 >> out_shift);
            sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
            sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
            *pOutBuffer2 = bitins(sum5, n_mask6, sum8, mask6, off6);
          }
        }
        pOutBuffer+=(dim_out_x * ch_out_r);
        pOutBuffer2+=(dim_out_x * ch_out_r);
        l++;
      }while(l<(dim_out_y>>1));
      i_out_x++;
    }while((i_out_x * stride_x) < ((dim_out_x * stride_x) - padding_x_right));
    for (i_out_x; i_out_x < dim_out_x; i_out_x++)
    {
      uint8_t *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
      uint8_t *pOutBuffer2 = pOutBuffer + ((dim_out_y>>1) * (dim_out_x * ch_out_r));
      uint8_t *pIm2Col = pIm2ColBase;
      uint8_t *pIm2Col2 = pIm2Col + im2col_size;
      uint8_t *pIm2Col3 = pIm2Col2 + im2col_size;
      uint8_t *pIm2Col4 = pIm2Col3 + im2col_size;
      asm volatile ("":::"memory");
      i_buff_y = - padding_y_top;
      if(padding_y_top > 0)
      {
        do
        {
          int i=0;
          do
          {
            *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
            pIm2Col+=4;
            *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
            pIm2Col2+=4;
            *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
            pIm2Col3+=4;
            *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
            pIm2Col4+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
          pIm2Col2-=dim_incr;
          pIm2Col3-=dim_incr;
          pIm2Col4-=dim_incr;
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
          v4u src_in = *((v4u*) (base_ptr + idx));
          *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 0);
          *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 4);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 8);
          *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 12);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 16);
          *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 20);
          pIm2Col++;
          pIm2Col2++;
          *pIm2Col = (uint8_t) bitextu((uint32_t) src_in, 4, 24);
          *pIm2Col2 = (uint8_t) bitextu((uint32_t) src_in, 4, 28);
          pIm2Col++;
          pIm2Col2++;
          src_in = *((v4u*) (base_ptr + idx + in_image_size));
          *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 0);
          *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 4);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 8);
          *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 12);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 16);
          *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 20);
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col3 = (uint8_t) bitextu((uint32_t) src_in, 4, 24);
          *pIm2Col4 = (uint8_t) bitextu((uint32_t) src_in, 4, 28);
          pIm2Col3++;
          pIm2Col4++;
          idx+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);

        pIm2Col-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        pIm2Col2-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        pIm2Col3-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        pIm2Col4-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        base_ptr+=dim_in_x;
        for(int j=0; j<(1 + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right); j++)
        {
          *(uint8_t *) pIm2Col = 0;
          pIm2Col++;
          *(uint8_t *) pIm2Col2 = 0;
          pIm2Col2++;
          *(uint8_t *) pIm2Col3 = 0;
          pIm2Col3++;
          *(uint8_t *) pIm2Col4 = 0;
          pIm2Col4++;
        }
        i_buff_y++;
      }while(i_buff_y < dim_in_y);
      for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
      {
        int i=0;
        do
        {
          *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
          pIm2Col+=4;
          *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
          pIm2Col2+=4;
          *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
          pIm2Col3+=4;
          *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
          pIm2Col4+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);
        pIm2Col-=dim_incr;
        pIm2Col2-=dim_incr;
        pIm2Col3-=dim_incr;
        pIm2Col4-=dim_incr;
      }

      int l=0;
      do
      {
        pWt = pWeight + i_wt_ch;
        pWt2 = pWt + kernel_size;
        pWt3 = pWt2 + kernel_size;
        pWt4 = pWt3 + kernel_size;
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        int sum5 = 0;
        int sum6 = 0;
        int sum7 = 0;
        int sum8 = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
          sum2 = ((int) (pBias[i_ch + 1]));
          sum3 = ((int) (pBias[i_ch + 2]));
          sum4 = ((int) (pBias[i_ch + 3]));
          sum5 = sum;
          sum6 = sum2;
          sum7 = sum3;
          sum8 = sum4;
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        pIm2Col2 = pIm2Col + im2col_size;
        pIm2Col3 = pIm2Col2 + im2col_size;
        pIm2Col4 = pIm2Col3 + im2col_size;
        int32_t ptrA = (int32_t *) pWt;
        int32_t ptrA2 = (int32_t *) pWt2;
        int32_t ptrA3 = (int32_t *) pWt3;
        int32_t ptrA4 = (int32_t *) pWt4;
        uint32_t ptrB  = (uint32_t *) pIm2Col;
        uint32_t ptrB_2 = (uint32_t *) (pIm2Col + ((dim_in_y>>1) * stride_y * dim_kernel_x));
        uint32_t ptrB2 = (uint32_t *) pIm2Col2;
        uint32_t ptrB2_2 = (uint32_t *) (pIm2Col2 + ((dim_in_y>>1) * stride_y * dim_kernel_x));
        uint32_t ptrB3 = (uint32_t *) pIm2Col3;
        uint32_t ptrB3_2 = (uint32_t *) (pIm2Col3 + ((dim_in_y>>1) * stride_y * dim_kernel_x));
        uint32_t ptrB4 = (uint32_t *) pIm2Col4;
        uint32_t ptrB4_2 = (uint32_t *) (pIm2Col4 + ((dim_in_y>>1) * stride_y * dim_kernel_x));

        ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
        ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
        ptrA3 = MacLoadInit(1, 0, 2, 0, ptrA3);
        ptrA4 = MacLoadInit(1, 0, 3, 0, ptrA4);
        ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
        int j=0;
        do
        {
          ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);
          sum = MacLoad4(0, 1, 0, 0, ptrB3, sum);
          ptrB3 = MacLoadUpdate(ptrB3);
          sum2 = MacLoad4(0, 1, 1, 1, ptrB4, sum2);
          ptrB4 = MacLoadUpdate(ptrB4);
          sum3 = MacLoad4(0, 1, 2, 0, ptrB_2, sum3);
          ptrB_2 = MacLoadUpdate(ptrB_2);
          sum4 = MacLoad4(0, 1, 3, 1, ptrB2_2, sum4);
          ptrB2_2 = MacLoadUpdate(ptrB2_2);
          sum5 = MacLoad4(0, 1, 0, 0, ptrB3_2, sum5);
          ptrB3_2 = MacLoadUpdate(ptrB3_2);
          sum6 = MacLoad4(0, 1, 1, 1, ptrB4_2, sum6);
          ptrB4_2 = MacLoadUpdate(ptrB4_2);
          sum7 = MacLoad4(0, 1, 2, 0, ptrB, sum7);
          ptrB = MacLoadUpdate(ptrB);
          sum8 = MacLoad4(1, 0, 3, 1, ptrA4, sum8);
          ptrA4 = MacLoadUpdate(ptrA4);
          ptrA = MacLoadInit(1, 0, 0, 0, ptrA);
          ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
          ptrA3 = MacLoadInit(1, 0, 2, 0, ptrA3);
          j++;
        }while(j<colCnt);
        if(leftCnt)
        {
          pWt+=(j << 2);
          pWt2+=(j << 2);
          pWt3+=(j << 2);
          pWt4+=(j << 2);
          pIm2Col+=(j << 2);
          pIm2Col2+=(j << 2);
          pIm2Col3+=(j << 2);
          pIm2Col4+=(j << 2);
          do
          {
            int8_t w = *(int8_t *) pWt++;
            uint8_t x = *(uint8_t *) pIm2Col++;
            uint8_t x_2 = *(uint8_t *) (pIm2Col - 1 + ((dim_in_y>>1) * stride_y * dim_kernel_x));
            sum += x * w;
            sum5 += x_2 * w;
            int8_t w2 = *(int8_t *) pWt2++;
            uint8_t x2 = *(uint8_t *) pIm2Col2++;
            uint8_t x2_2 = *(uint8_t *) (pIm2Col2 - 1 + ((dim_in_y>>1) * stride_y * dim_kernel_x));
            sum2 += x2 * w2;
            sum6 += x2_2 * w2;
            int8_t w3 = *(int8_t *) pWt3++;
            uint8_t x3 = *(uint8_t *) pIm2Col3++;
            uint8_t x3_2 = *(uint8_t *) (pIm2Col3 - 1 + ((dim_in_y>>1) * stride_y * dim_kernel_x));
            sum3 += x3 * w3;
            sum7 += x3_2 * w3;
            int8_t w4 = *(int8_t *) pWt4++;
            uint8_t x4 = *(uint8_t *) pIm2Col4++;
            uint8_t x4_2 = *(uint8_t *) (pIm2Col4 - 1 + ((dim_in_y>>1) * stride_y * dim_kernel_x));
            sum4 += x4 * w4;
            sum8 += x4_2 * w4;
            j++;
          }while(j<leftCnt);
        }
        if (flag_batch_norm && flag_relu)
        {
          sum = pulp_nn_bn_quant_u2(sum, *k1, *lambda1, out_shift);
          sum2 = pulp_nn_bn_quant_u2(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          sum3 = pulp_nn_bn_quant_u2(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
          sum4 = pulp_nn_bn_quant_u2(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
          sum = bitins(sum, n_mask2, sum2, mask2, off2);
          sum = bitins(sum, n_mask4, sum3, mask4, off4);
          *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
          sum5 = pulp_nn_bn_quant_u2(sum5, *k1, *lambda1, out_shift);
          sum6 = pulp_nn_bn_quant_u2(sum6, *(k1 + 1), *(lambda1 + 1), out_shift);
          sum7 = pulp_nn_bn_quant_u2(sum7, *(k1 + 2), *(lambda1 + 2), out_shift);
          sum8 = pulp_nn_bn_quant_u2(sum8, *(k1 + 3), *(lambda1 + 3), out_shift);
          sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
          sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
          *pOutBuffer2 = bitins(sum5, n_mask6, sum8, mask6, off6);
        }
        else
        {
          if(flag_relu == 1)
          {
            sum = pulp_nn_quant_u2(sum, out_mult, out_shift);
            sum2 = pulp_nn_quant_u2(sum2, out_mult, out_shift);
            sum3 = pulp_nn_quant_u2(sum3, out_mult, out_shift);
            sum4 = pulp_nn_quant_u2(sum4, out_mult, out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
            sum5 = pulp_nn_quant_u2(sum5, out_mult, out_shift);
            sum6 = pulp_nn_quant_u2(sum6, out_mult, out_shift);
            sum7 = pulp_nn_quant_u2(sum7, out_mult, out_shift);
            sum8 = pulp_nn_quant_u2(sum8, out_mult, out_shift);
            sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
            sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
            *pOutBuffer2 = bitins(sum5, n_mask6, sum8, mask6, off6);
          }
          else
          {
            sum = (uint8_t) clip2(sum >> out_shift);
            sum2 = (uint8_t) clip2(sum2 >> out_shift);
            sum3 = (uint8_t) clip2(sum3 >> out_shift);
            sum4 = (uint8_t) clip2(sum4 >> out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
            sum5 = (uint8_t) clip2(sum5 >> out_shift);
            sum6 = (uint8_t) clip2(sum6 >> out_shift);
            sum7 = (uint8_t) clip2(sum7 >> out_shift);
            sum8 = (uint8_t) clip2(sum8 >> out_shift);
            sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
            sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
            *pOutBuffer2 = bitins(sum5, n_mask6, sum8, mask6, off6);
          }
        }
        pOutBuffer+=(dim_out_x * ch_out_r);
        pOutBuffer2+=(dim_out_x * ch_out_r);
        l++;
      }while(l<(dim_out_y>>1));
    }
    i_in_ch+=(in_image_size << 1);
    i_wt_ch+=(kernel_size << 2);
    k1+=4;
    lambda1+=4;
    i_out_ch++;
  }
  pi_cl_team_barrier(0);
}
