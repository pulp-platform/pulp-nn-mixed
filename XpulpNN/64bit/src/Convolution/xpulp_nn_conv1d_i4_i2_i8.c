/*
 * xpulp_nn_conv1d_i4_i2_i8.c
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
#include "pulp_nn_kernels.h"



void __attribute__((noinline)) xpulp_nn_conv1d_i4_i2_i8(
                        int8_t *pIn,
                        int8_t *pIm2ColBuffer,
                        int8_t *pBias,
                        int8_t *pOut,
                        int8_t *pWeight,
                        int64_t *pKappa,
                        int64_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t dim_in_x,
                        uint16_t ch_in,
                        uint16_t dim_out_x,
                        uint16_t ch_out,
                        uint16_t dim_kernel_x,
                        uint16_t padding_x_left,
                        uint16_t padding_x_right,
                        uint16_t stride_x,
                        uint16_t dilation_x,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
  uint16_t ch_in_r = PACK_INT4_SIZE(ch_in);
  uint16_t ch_out_r = PACK_INT2_SIZE(ch_out);
  uint16_t dil = dilation_x - 1;

  int core_id = pi_core_id();
  int8_t * pIm2ColBase = pIm2ColBuffer + (2 * core_id * PACK_INT8_SIZE(ch_in) * dim_kernel_x);
  int i_out_x, i_ker_x;
  int Log2Core;

  uint8_t extra_chunk = ((dim_out_x & (NUM_CORES-1)) != 0);
  uint8_t extra_chunk_r;
  uint8_t section;
  int core_id_r;

  Log2Core = log2(NUM_CORES);
  core_id_r = core_id;
  section = 0;
  extra_chunk_r = extra_chunk;
  extra_chunk = 0;

  int chunk = (dim_out_x >> Log2Core) + extra_chunk_r;

  int start_pixel = min((chunk * core_id_r), dim_out_x);
  int stop_pixel = min(start_pixel + chunk, dim_out_x);

  int8_t *pIm2Col = pIm2ColBase;
  int8_t *pOutBuffer = pOut + (start_pixel * ch_out_r) + (section * ch_out_r);

  for (i_out_x = start_pixel; i_out_x < stop_pixel; i_out_x++)
  {
      if(i_out_x < padding_x_left)
      {
        for(i_ker_x=i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x * (1 + dil) - dil; i_ker_x += (1 + dil))
        {
            if((i_ker_x < 0) || (i_ker_x >= dim_in_x))
            {
              xpulp_nn_zero_mem_u8(pIm2Col, ch_in);
            }
            else
            {
              xpulp_nn_im2col_i4_to_i8((uint8_t*) pIn + i_ker_x * ch_in_r, pIm2Col, ch_in);
            }
            pIm2Col+=PACK_INT8_SIZE(ch_in);
        }
      }
      else if(i_out_x < dim_out_x - padding_x_right)
      {
          for(i_ker_x=i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x * (1 + dil) - dil; i_ker_x += (1 + dil))
          {
            xpulp_nn_im2col_i4_to_i8((uint8_t*) pIn + i_ker_x * ch_in_r, pIm2Col, ch_in);
            pIm2Col+=PACK_INT8_SIZE(ch_in);
          }
      }
      else
      {
        for(i_ker_x=i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x * (1 + dil) - dil; i_ker_x += (1 + dil))
        {
            if(i_ker_x < 0 || (i_ker_x >= dim_in_x))
            {
              xpulp_nn_zero_mem_u8(pIm2Col, ch_in);
            }
            else
            {
              xpulp_nn_im2col_i4_to_i8((uint8_t *) pIn + i_ker_x * ch_in_r, pIm2Col, ch_in);
            }
            pIm2Col+=PACK_INT8_SIZE(ch_in);
        }
      }
      if(pIm2Col == (pIm2ColBase + ((PACK_INT8_SIZE(ch_in) * dim_kernel_x) << 1)))
      {
        pOutBuffer = xpulp_nn_matmul_i4_i2_i8(
          pIm2ColBase,
          pBias,
          pOutBuffer,
          pOutBuffer + ch_out_r,
          pWeight,
          pKappa,
          pLambda,
          out_mult,
          out_shift,
          (ch_in * dim_kernel_x),
          ch_out,
          flag_relu,
          flag_batch_norm
          );

        pIm2Col = pIm2ColBase;
      }
    }
    if(pIm2Col != pIm2ColBase)
    {
      int8_t mask2 = 0x0c;
      int8_t n_mask2 = ~ mask2;
      int8_t mask4 = 0x30;
      int8_t n_mask4 = ~ mask4;
      int8_t mask6 = 0xc0;
      int8_t n_mask6 = ~ mask6;
      int8_t off2 = 2;
      int8_t off4 = 4;
      int8_t off6 = 6;
      const int8_t *pA = pWeight;
      int i;
      int64_t * k1 = pKappa;
      int64_t * lambda1 = pLambda;
      int8_t out[4];
      uint16_t num_col_im2col = ch_in * dim_kernel_x;
      uint16_t num_col_im2col_w = PACK_INT8_SIZE(ch_in) * dim_kernel_x;

      for(i = 0; i < ch_out; i++)
      {
        int sum = 0;
        if (pBias != NULL)
        {
          sum = ((int) (*pBias++));
        }

        int8_t *pB = pIm2ColBase;

        int32_t *ptrA  = (int32_t *)pA;
        int32_t *ptrB = (int32_t *)pB;

        for(int j=0; j < (num_col_im2col >> 2); j++)
        {
          sum = SumDotps4(*(v4s *)ptrB, *(v4s *)ptrA, sum);
          ptrA++;
          ptrB++;
        }

        int col_cnt_im2col = num_col_im2col & 0x3;

        if(col_cnt_im2col)
        {
          uint16_t loop_cnt_im2col_w = (num_col_im2col >> 2) << 2;
          pA+=loop_cnt_im2col_w;

          uint16_t loop_cnt_im2col_a = (num_col_im2col >> 2) << 2;
          pB+=loop_cnt_im2col_a;

          do
          {
            int8_t inA1 = *pA++;
            int8_t inB1 = *pB++;
            asm volatile("": : :"memory");
            sum += inA1 * inB1;

            col_cnt_im2col--;
          } while(col_cnt_im2col);
          pA-=num_col_im2col_w;
        }
        if (flag_batch_norm && flag_relu)
        {
          uint8_t i_o = i & 0x03;
          out[i_o] = pulp_nn_bn_quant_i2(sum, *k1, *lambda1, out_shift);
          k1++;
          lambda1++;
          if(i_o == 0x03)
          {
            out[0] = bitins(out[0], n_mask2, out[1], mask2, off2);
            out[0] = bitins(out[0], n_mask4, out[2], mask4, off4);
            *pOutBuffer = bitins(out[0], n_mask6, out[3], mask6, off6);
            pOutBuffer++;
          }
        }
        else
        {
          if(flag_relu == 1)
          {
            uint8_t i_o = i & 0x03;
            out[i_o] = pulp_nn_quant_i2(sum, out_mult, out_shift);
            if(i_o == 0x03)
            {
              out[0] = bitins(out[0], n_mask2, out[1], mask2, off2);
              out[0] = bitins(out[0], n_mask4, out[2], mask4, off4);
              *pOutBuffer = bitins(out[0], n_mask6, out[3], mask6, off6);
              pOutBuffer++;
            }
          }
          else
          {
            uint8_t i_o = i & 0x03;
            out[i_o] = (int8_t) clips2(sum >> out_shift);
            if(i_o == 0x03)
            {
              out[0] = bitins(out[0], n_mask2, out[1], mask2, off2);
              out[0] = bitins(out[0], n_mask4, out[2], mask4, off4);
              *pOutBuffer = bitins(out[0], n_mask6, out[3], mask6, off6);
              pOutBuffer++;
            }
          }
        }
        pA+=num_col_im2col_w;
      }
    pOutBuffer+=(extra_chunk * (1 - section) * ch_out_r);
    pIm2Col = pIm2ColBase;
  }
  pi_cl_team_barrier();
}
