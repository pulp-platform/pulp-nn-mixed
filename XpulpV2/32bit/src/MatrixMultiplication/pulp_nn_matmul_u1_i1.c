
/*
 * pulp_nn_matmul_u1_i1.c
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


uint8_t *pulp_nn_matmul_u1_i1(
                        uint8_t *pIn,
                        int8_t *pBias,
                        uint8_t *pOut,
                        uint8_t *pOut2,
                        int8_t *pWeight,
                        int32_t *pKappa,
                        int32_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t num_col_im2col,
                        uint16_t ch_out,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
  int8_t mask1 = 0x2;
  int8_t n_mask1 = ~ mask1;
  int8_t off1 = 1;
  int8_t mask2 = 0x4;
  int8_t n_mask2 = ~ mask2;
  int8_t off2 = 2;
  int8_t mask3 = 0x8;
  int8_t n_mask3 = ~ mask3;
  int8_t off3 = 3;
  int8_t mask4 = 0x10;
  int8_t n_mask4 = ~ mask4;
  int8_t off4 = 4;
  int8_t mask5 = 0x20;
  int8_t n_mask5 = ~ mask5;
  int8_t off5 = 5;
  int8_t mask6 = 0x40;
  int8_t n_mask6 = ~ mask6;
  int8_t off6 = 6;
  int8_t mask7 = 0x80;
  int8_t n_mask7 = ~ mask7;
  int8_t off7 = 7;

  v4u vecB1;
  v4u vecB2;
  v4u vecB3;
  v4u vecB4;
  v4u vecB5;
  v4u vecB6;
  v4u vecB7;
  v4u vecB8;
  v4u vecB9;
  v4u vecB10;
  v4u vecB11;
  v4u vecB12;
  v4u vecB13;
  v4u vecB14;
  v4u vecB15;
  v4u vecB16;
  v4s vecA1[8];
  v4s vecA2[8];
  v4s vecA3[8];
  v4s vecA4[8];
  v4s vecA5[8];
  v4s vecA6[8];
  v4s vecA7[8];
  v4s vecA8[8];

  uint16_t ch_out_r = PACK_INT1_SIZE(ch_out);

  uint16_t num_col_im2col_w = PACK_INT1_SIZE(num_col_im2col);

  //uint8_t *pOut2 = pOut + ch_out_r;
  int8_t *pA1 = pWeight;

  uint16_t chan_left = ch_out & 0x7;

  for(int i=0; i < (ch_out >> 3); i++)
  {
    uint8_t *pB1 =  pIn;
    uint8_t *pB2 = (pB1 + num_col_im2col);
    int8_t *pA2 = (pA1 + num_col_im2col_w);
    int8_t *pA3 = (pA2 + num_col_im2col_w);
    int8_t *pA4 = (pA3 + num_col_im2col_w);
    int8_t *pA5 = (pA4 + num_col_im2col_w);
    int8_t *pA6 = (pA5 + num_col_im2col_w);
    int8_t *pA7 = (pA6 + num_col_im2col_w);
    int8_t *pA8 = (pA7 + num_col_im2col_w);

    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    int sum4 = 0;
    int sum5 = 0;
    int sum6 = 0;
    int sum7 = 0;
    int sum8 = 0;
    int sum9 = 0;
    int sum10 = 0;
    int sum11 = 0;
    int sum12 = 0;
    int sum13 = 0;
    int sum14 = 0;
    int sum15 = 0;
    int sum16 = 0;

    if (pBias != NULL)
    {
      sum1 = ((int) (*pBias++));
      sum2 = ((int) (*pBias++));
      sum3 = ((int) (*pBias++));
      sum4 = ((int) (*pBias++));
      sum9 = ((int) (*pBias++));
      sum10 = ((int) (*pBias++));
      sum11 = ((int) (*pBias++));
      sum12 = ((int) (*pBias++));

      sum5 = sum1;
      sum6 = sum2;
      sum7 = sum3;
      sum8 = sum4;
      sum13 = sum5;
      sum14 = sum6;
      sum15 = sum7;
      sum16 = sum8;
    }

    for(int j=0; j<(num_col_im2col_w >> 2); j++)
    {
      vecB1 = *((v4u*)pB1 + 0);

      vecB2 = *((v4u*)pB2 + 0);

      vecB3 = *((v4u*)pB1 + 4);

      vecB4 = *((v4u*)pB2 + 4);

      vecB5 = *((v4u*)pB1 + 8);

      vecB6 = *((v4u*)pB2 + 8);

      vecB7 = *((v4u*)pB1 + 12);

      vecB8 = *((v4u*)pB2 + 12);

      vecB9 = *((v4u*)pB1 + 16);

      vecB10 = *((v4u*)pB2 + 16);

      vecB11 = *((v4u*)pB1 + 20);

      vecB12 = *((v4u*)pB2 + 20);

      vecB13 = *((v4u*)pB1 + 24);

      vecB14 = *((v4u*)pB2 + 24);

      vecB15 = *((v4u*)pB1 + 28);

      vecB16 = *((v4u*)pB2 + 28);

      pB1+=32;
      pB2+=32;

      pA1 = pulp_nn_i1_to_i8(pA1,vecA1);

      sum1 = SumDotp4(vecB1, vecA1[0], sum1);
      sum5 = SumDotp4(vecB2, vecA1[0], sum5);
      sum1 = SumDotp4(vecB3, vecA1[1], sum1);
      sum5 = SumDotp4(vecB4, vecA1[1], sum5);
      sum1 = SumDotp4(vecB5, vecA1[2], sum1);
      sum5 = SumDotp4(vecB6, vecA1[2], sum5);
      sum1 = SumDotp4(vecB7, vecA1[3], sum1);
      sum5 = SumDotp4(vecB8, vecA1[3], sum5);
      sum1 = SumDotp4(vecB9, vecA1[4], sum1);
      sum5 = SumDotp4(vecB10, vecA1[4], sum5);
      sum1 = SumDotp4(vecB11, vecA1[5], sum1);
      sum5 = SumDotp4(vecB12, vecA1[5], sum5);
      sum1 = SumDotp4(vecB13, vecA1[6], sum1);
      sum5 = SumDotp4(vecB14, vecA1[6], sum5);
      sum1 = SumDotp4(vecB15, vecA1[7], sum1);
      sum5 = SumDotp4(vecB16, vecA1[7], sum5);
      pA2 = pulp_nn_i1_to_i8(pA2,vecA2);

      sum2 = SumDotp4(vecB1, vecA2[0], sum2);
      sum6 = SumDotp4(vecB2, vecA2[0], sum6);
      sum2 = SumDotp4(vecB3, vecA2[1], sum2);
      sum6 = SumDotp4(vecB4, vecA2[1], sum6);
      sum2 = SumDotp4(vecB5, vecA2[2], sum2);
      sum6 = SumDotp4(vecB6, vecA2[2], sum6);
      sum2 = SumDotp4(vecB7, vecA2[3], sum2);
      sum6 = SumDotp4(vecB8, vecA2[3], sum6);
      sum2 = SumDotp4(vecB9, vecA2[4], sum2);
      sum6 = SumDotp4(vecB10, vecA2[4], sum6);
      sum2 = SumDotp4(vecB11, vecA2[5], sum2);
      sum6 = SumDotp4(vecB12, vecA2[5], sum6);
      sum2 = SumDotp4(vecB13, vecA2[6], sum2);
      sum6 = SumDotp4(vecB14, vecA2[6], sum6);
      sum2 = SumDotp4(vecB15, vecA2[7], sum2);
      sum6 = SumDotp4(vecB16, vecA2[7], sum6);
      pA3 = pulp_nn_i1_to_i8(pA3,vecA3);

      sum3 = SumDotp4(vecB1, vecA3[0], sum3);
      sum7 = SumDotp4(vecB2, vecA3[0], sum7);
      sum3 = SumDotp4(vecB3, vecA3[1], sum3);
      sum7 = SumDotp4(vecB4, vecA3[1], sum7);
      sum3 = SumDotp4(vecB5, vecA3[2], sum3);
      sum7 = SumDotp4(vecB6, vecA3[2], sum7);
      sum3 = SumDotp4(vecB7, vecA3[3], sum3);
      sum7 = SumDotp4(vecB8, vecA3[3], sum7);
      sum3 = SumDotp4(vecB9, vecA3[4], sum3);
      sum7 = SumDotp4(vecB10, vecA3[4], sum7);
      sum3 = SumDotp4(vecB11, vecA3[5], sum3);
      sum7 = SumDotp4(vecB12, vecA3[5], sum7);
      sum3 = SumDotp4(vecB13, vecA3[6], sum3);
      sum7 = SumDotp4(vecB14, vecA3[6], sum7);
      sum3 = SumDotp4(vecB15, vecA3[7], sum3);
      sum7 = SumDotp4(vecB16, vecA3[7], sum7);
      pA4 = pulp_nn_i1_to_i8(pA4,vecA4);

      sum4 = SumDotp4(vecB1, vecA4[0], sum4);
      sum8 = SumDotp4(vecB2, vecA4[0], sum8);
      sum4 = SumDotp4(vecB3, vecA4[1], sum4);
      sum8 = SumDotp4(vecB4, vecA4[1], sum8);
      sum4 = SumDotp4(vecB5, vecA4[2], sum4);
      sum8 = SumDotp4(vecB6, vecA4[2], sum8);
      sum4 = SumDotp4(vecB7, vecA4[3], sum4);
      sum8 = SumDotp4(vecB8, vecA4[3], sum8);
      sum4 = SumDotp4(vecB9, vecA4[4], sum4);
      sum8 = SumDotp4(vecB10, vecA4[4], sum8);
      sum4 = SumDotp4(vecB11, vecA4[5], sum4);
      sum8 = SumDotp4(vecB12, vecA4[5], sum8);
      sum4 = SumDotp4(vecB13, vecA4[6], sum4);
      sum8 = SumDotp4(vecB14, vecA4[6], sum8);
      sum4 = SumDotp4(vecB15, vecA4[7], sum4);
      sum8 = SumDotp4(vecB16, vecA4[7], sum8);
      pA5 = pulp_nn_i1_to_i8(pA5,vecA5);

      sum9 = SumDotp4(vecB1, vecA5[0], sum9);
      sum13 = SumDotp4(vecB2, vecA5[0], sum13);
      sum9 = SumDotp4(vecB3, vecA5[1], sum9);
      sum13 = SumDotp4(vecB4, vecA5[1], sum13);
      sum9 = SumDotp4(vecB5, vecA5[2], sum9);
      sum13 = SumDotp4(vecB6, vecA5[2], sum13);
      sum9 = SumDotp4(vecB7, vecA5[3], sum9);
      sum13 = SumDotp4(vecB8, vecA5[3], sum13);
      sum9 = SumDotp4(vecB9, vecA5[4], sum9);
      sum13 = SumDotp4(vecB10, vecA5[4], sum13);
      sum9 = SumDotp4(vecB11, vecA5[5], sum9);
      sum13 = SumDotp4(vecB12, vecA5[5], sum13);
      sum9 = SumDotp4(vecB13, vecA5[6], sum9);
      sum13 = SumDotp4(vecB14, vecA5[6], sum13);
      sum9 = SumDotp4(vecB15, vecA5[7], sum9);
      sum13 = SumDotp4(vecB16, vecA5[7], sum13);
      pA6 = pulp_nn_i1_to_i8(pA6,vecA6);

      sum10 = SumDotp4(vecB1, vecA6[0], sum10);
      sum14 = SumDotp4(vecB2, vecA6[0], sum14);
      sum10 = SumDotp4(vecB3, vecA6[1], sum10);
      sum14 = SumDotp4(vecB4, vecA6[1], sum14);
      sum10 = SumDotp4(vecB5, vecA6[2], sum10);
      sum14 = SumDotp4(vecB6, vecA6[2], sum14);
      sum10 = SumDotp4(vecB7, vecA6[3], sum10);
      sum14 = SumDotp4(vecB8, vecA6[3], sum14);
      sum10 = SumDotp4(vecB9, vecA6[4], sum10);
      sum14 = SumDotp4(vecB10, vecA6[4], sum14);
      sum10 = SumDotp4(vecB11, vecA6[5], sum10);
      sum14 = SumDotp4(vecB12, vecA6[5], sum14);
      sum10 = SumDotp4(vecB13, vecA6[6], sum10);
      sum14 = SumDotp4(vecB14, vecA6[6], sum14);
      sum10 = SumDotp4(vecB15, vecA6[7], sum10);
      sum14 = SumDotp4(vecB16, vecA6[7], sum14);
      pA7 = pulp_nn_i1_to_i8(pA7,vecA7);

      sum11 = SumDotp4(vecB1, vecA7[0], sum11);
      sum15 = SumDotp4(vecB2, vecA7[0], sum15);
      sum11 = SumDotp4(vecB3, vecA7[1], sum11);
      sum15 = SumDotp4(vecB4, vecA7[1], sum15);
      sum11 = SumDotp4(vecB5, vecA7[2], sum11);
      sum15 = SumDotp4(vecB6, vecA7[2], sum15);
      sum11 = SumDotp4(vecB7, vecA7[3], sum11);
      sum15 = SumDotp4(vecB8, vecA7[3], sum15);
      sum11 = SumDotp4(vecB9, vecA7[4], sum11);
      sum15 = SumDotp4(vecB10, vecA7[4], sum15);
      sum11 = SumDotp4(vecB11, vecA7[5], sum11);
      sum15 = SumDotp4(vecB12, vecA7[5], sum15);
      sum11 = SumDotp4(vecB13, vecA7[6], sum11);
      sum15 = SumDotp4(vecB14, vecA7[6], sum15);
      sum11 = SumDotp4(vecB15, vecA7[7], sum11);
      sum15 = SumDotp4(vecB16, vecA7[7], sum15);
      pA8 = pulp_nn_i1_to_i8(pA8,vecA8);

      sum12 = SumDotp4(vecB1, vecA8[0], sum12);
      sum16 = SumDotp4(vecB2, vecA8[0], sum16);
      sum12 = SumDotp4(vecB3, vecA8[1], sum12);
      sum16 = SumDotp4(vecB4, vecA8[1], sum16);
      sum12 = SumDotp4(vecB5, vecA8[2], sum12);
      sum16 = SumDotp4(vecB6, vecA8[2], sum16);
      sum12 = SumDotp4(vecB7, vecA8[3], sum12);
      sum16 = SumDotp4(vecB8, vecA8[3], sum16);
      sum12 = SumDotp4(vecB9, vecA8[4], sum12);
      sum16 = SumDotp4(vecB10, vecA8[4], sum16);
      sum12 = SumDotp4(vecB11, vecA8[5], sum12);
      sum16 = SumDotp4(vecB12, vecA8[5], sum16);
      sum12 = SumDotp4(vecB13, vecA8[6], sum12);
      sum16 = SumDotp4(vecB14, vecA8[6], sum16);
      sum12 = SumDotp4(vecB15, vecA8[7], sum12);
      sum16 = SumDotp4(vecB16, vecA8[7], sum16);
    }
    uint16_t col_cnt_im2col = num_col_im2col & 0x1f;

    while (col_cnt_im2col)
    {
	    int8_t inA1 = (int8_t) bitext((int) *pA1, 1, 0);
	    int8_t inA2 = (int8_t) bitext((int) *pA2, 1, 0);
	    int8_t inA3 = (int8_t) bitext((int) *pA3, 1, 0);
	    int8_t inA4 = (int8_t) bitext((int) *pA4, 1, 0);
	    int8_t inA5 = (int8_t) bitext((int) *pA5, 1, 0);
	    int8_t inA6 = (int8_t) bitext((int) *pA6, 1, 0);
	    int8_t inA7 = (int8_t) bitext((int) *pA7, 1, 0);
	    int8_t inA8 = (int8_t) bitext((int) *pA8, 1, 0);
      uint8_t inB1 = *pB1++;
      uint8_t inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum9 += inA5 * inB1;
      sum2 += inA2 * inB1;
      sum10 += inA6 * inB1;
      sum3 += inA3 * inB1;
      sum11 += inA7 * inB1;
      sum4 += inA4 * inB1;
      sum12 += inA8 * inB1;
      sum5 += inA1 * inB2;
      sum13 += inA5 * inB2;
      sum6 += inA2 * inB2;
      sum14 += inA6 * inB2;
      sum7 += inA3 * inB2;
      sum15 += inA7 * inB2;
      sum8 += inA4 * inB2;
      sum16 += inA8 * inB2;
	    inA1 = (int8_t) bitext((int) *pA1, 1, 1);
	    inA2 = (int8_t) bitext((int) *pA2, 1, 1);
	    inA3 = (int8_t) bitext((int) *pA3, 1, 1);
	    inA4 = (int8_t) bitext((int) *pA4, 1, 1);
	    inA5 = (int8_t) bitext((int) *pA5, 1, 1);
	    inA6 = (int8_t) bitext((int) *pA6, 1, 1);
	    inA7 = (int8_t) bitext((int) *pA7, 1, 1);
	    inA8 = (int8_t) bitext((int) *pA8, 1, 1);
      inB1 = *pB1++;
      inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum9 += inA5 * inB1;
      sum2 += inA2 * inB1;
      sum10 += inA6 * inB1;
      sum3 += inA3 * inB1;
      sum11 += inA7 * inB1;
      sum4 += inA4 * inB1;
      sum12 += inA8 * inB1;
      sum5 += inA1 * inB2;
      sum13 += inA5 * inB2;
      sum6 += inA2 * inB2;
      sum14 += inA6 * inB2;
      sum7 += inA3 * inB2;
      sum15 += inA7 * inB2;
      sum8 += inA4 * inB2;
      sum16 += inA8 * inB2;
	    inA1 = (int8_t) bitext((int) *pA1, 1, 2);
	    inA2 = (int8_t) bitext((int) *pA2, 1, 2);
	    inA3 = (int8_t) bitext((int) *pA3, 1, 2);
	    inA4 = (int8_t) bitext((int) *pA4, 1, 2);
	    inA5 = (int8_t) bitext((int) *pA5, 1, 2);
	    inA6 = (int8_t) bitext((int) *pA6, 1, 2);
	    inA7 = (int8_t) bitext((int) *pA7, 1, 2);
	    inA8 = (int8_t) bitext((int) *pA8, 1, 2);
      inB1 = *pB1++;
      inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum9 += inA5 * inB1;
      sum2 += inA2 * inB1;
      sum10 += inA6 * inB1;
      sum3 += inA3 * inB1;
      sum11 += inA7 * inB1;
      sum4 += inA4 * inB1;
      sum12 += inA8 * inB1;
      sum5 += inA1 * inB2;
      sum13 += inA5 * inB2;
      sum6 += inA2 * inB2;
      sum14 += inA6 * inB2;
      sum7 += inA3 * inB2;
      sum15 += inA7 * inB2;
      sum8 += inA4 * inB2;
      sum16 += inA8 * inB2;
	    inA1 = (int8_t) bitext((int) *pA1, 1, 3);
	    inA2 = (int8_t) bitext((int) *pA2, 1, 3);
	    inA3 = (int8_t) bitext((int) *pA3, 1, 3);
	    inA4 = (int8_t) bitext((int) *pA4, 1, 3);
	    inA5 = (int8_t) bitext((int) *pA5, 1, 3);
	    inA6 = (int8_t) bitext((int) *pA6, 1, 3);
	    inA7 = (int8_t) bitext((int) *pA7, 1, 3);
	    inA8 = (int8_t) bitext((int) *pA8, 1, 3);
      inB1 = *pB1++;
      inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum9 += inA5 * inB1;
      sum2 += inA2 * inB1;
      sum10 += inA6 * inB1;
      sum3 += inA3 * inB1;
      sum11 += inA7 * inB1;
      sum4 += inA4 * inB1;
      sum12 += inA8 * inB1;
      sum5 += inA1 * inB2;
      sum13 += inA5 * inB2;
      sum6 += inA2 * inB2;
      sum14 += inA6 * inB2;
      sum7 += inA3 * inB2;
      sum15 += inA7 * inB2;
      sum8 += inA4 * inB2;
      sum16 += inA8 * inB2;
	    inA1 = (int8_t) bitext((int) *pA1, 1, 4);
	    inA2 = (int8_t) bitext((int) *pA2, 1, 4);
	    inA3 = (int8_t) bitext((int) *pA3, 1, 4);
	    inA4 = (int8_t) bitext((int) *pA4, 1, 4);
	    inA5 = (int8_t) bitext((int) *pA5, 1, 4);
	    inA6 = (int8_t) bitext((int) *pA6, 1, 4);
	    inA7 = (int8_t) bitext((int) *pA7, 1, 4);
	    inA8 = (int8_t) bitext((int) *pA8, 1, 4);
      inB1 = *pB1++;
      inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum9 += inA5 * inB1;
      sum2 += inA2 * inB1;
      sum10 += inA6 * inB1;
      sum3 += inA3 * inB1;
      sum11 += inA7 * inB1;
      sum4 += inA4 * inB1;
      sum12 += inA8 * inB1;
      sum5 += inA1 * inB2;
      sum13 += inA5 * inB2;
      sum6 += inA2 * inB2;
      sum14 += inA6 * inB2;
      sum7 += inA3 * inB2;
      sum15 += inA7 * inB2;
      sum8 += inA4 * inB2;
      sum16 += inA8 * inB2;
	    inA1 = (int8_t) bitext((int) *pA1, 1, 5);
	    inA2 = (int8_t) bitext((int) *pA2, 1, 5);
	    inA3 = (int8_t) bitext((int) *pA3, 1, 5);
	    inA4 = (int8_t) bitext((int) *pA4, 1, 5);
	    inA5 = (int8_t) bitext((int) *pA5, 1, 5);
	    inA6 = (int8_t) bitext((int) *pA6, 1, 5);
	    inA7 = (int8_t) bitext((int) *pA7, 1, 5);
	    inA8 = (int8_t) bitext((int) *pA8, 1, 5);
      inB1 = *pB1++;
      inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum9 += inA5 * inB1;
      sum2 += inA2 * inB1;
      sum10 += inA6 * inB1;
      sum3 += inA3 * inB1;
      sum11 += inA7 * inB1;
      sum4 += inA4 * inB1;
      sum12 += inA8 * inB1;
      sum5 += inA1 * inB2;
      sum13 += inA5 * inB2;
      sum6 += inA2 * inB2;
      sum14 += inA6 * inB2;
      sum7 += inA3 * inB2;
      sum15 += inA7 * inB2;
      sum8 += inA4 * inB2;
      sum16 += inA8 * inB2;
	    inA1 = (int8_t) bitext((int) *pA1, 1, 6);
	    inA2 = (int8_t) bitext((int) *pA2, 1, 6);
	    inA3 = (int8_t) bitext((int) *pA3, 1, 6);
	    inA4 = (int8_t) bitext((int) *pA4, 1, 6);
	    inA5 = (int8_t) bitext((int) *pA5, 1, 6);
	    inA6 = (int8_t) bitext((int) *pA6, 1, 6);
	    inA7 = (int8_t) bitext((int) *pA7, 1, 6);
	    inA8 = (int8_t) bitext((int) *pA8, 1, 6);
      inB1 = *pB1++;
      inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum9 += inA5 * inB1;
      sum2 += inA2 * inB1;
      sum10 += inA6 * inB1;
      sum3 += inA3 * inB1;
      sum11 += inA7 * inB1;
      sum4 += inA4 * inB1;
      sum12 += inA8 * inB1;
      sum5 += inA1 * inB2;
      sum13 += inA5 * inB2;
      sum6 += inA2 * inB2;
      sum14 += inA6 * inB2;
      sum7 += inA3 * inB2;
      sum15 += inA7 * inB2;
      sum8 += inA4 * inB2;
      sum16 += inA8 * inB2;
	    inA1 = (int8_t) bitext((int) *pA1, 1, 7);
	    inA2 = (int8_t) bitext((int) *pA2, 1, 7);
	    inA3 = (int8_t) bitext((int) *pA3, 1, 7);
	    inA4 = (int8_t) bitext((int) *pA4, 1, 7);
	    inA5 = (int8_t) bitext((int) *pA5, 1, 7);
	    inA6 = (int8_t) bitext((int) *pA6, 1, 7);
	    inA7 = (int8_t) bitext((int) *pA7, 1, 7);
	    inA8 = (int8_t) bitext((int) *pA8, 1, 7);
      inB1 = *pB1++;
      inB2 = *pB2++;
      sum1 += inA1 * inB1;
      sum9 += inA5 * inB1;
      sum2 += inA2 * inB1;
      sum10 += inA6 * inB1;
      sum3 += inA3 * inB1;
      sum11 += inA7 * inB1;
      sum4 += inA4 * inB1;
      sum12 += inA8 * inB1;
      sum5 += inA1 * inB2;
      sum13 += inA5 * inB2;
      sum6 += inA2 * inB2;
      sum14 += inA6 * inB2;
      sum7 += inA3 * inB2;
      sum15 += inA7 * inB2;
      sum8 += inA4 * inB2;
      sum16 += inA8 * inB2;
      pA1++;
      pA2++;
      pA3++;
      pA4++;
      pA5++;
      pA6++;
      pA7++;
      pA8++;
      col_cnt_im2col-=8;
    }
    if (flag_batch_norm && flag_relu)
    {
                  sum1 = pulp_nn_bn_quant_u1(sum1, *pKappa, *pLambda, out_shift);
                  sum5 = pulp_nn_bn_quant_u1(sum5, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum2 = pulp_nn_bn_quant_u1(sum2, *pKappa, *pLambda, out_shift);
                  sum6 = pulp_nn_bn_quant_u1(sum6, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum3 = pulp_nn_bn_quant_u1(sum3, *pKappa, *pLambda, out_shift);
                  sum7 = pulp_nn_bn_quant_u1(sum7, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum4 = pulp_nn_bn_quant_u1(sum4, *pKappa, *pLambda, out_shift);
                  sum8 = pulp_nn_bn_quant_u1(sum8, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum9 = pulp_nn_bn_quant_u1(sum9, *pKappa, *pLambda, out_shift);
                  sum13 = pulp_nn_bn_quant_u1(sum13, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum10 = pulp_nn_bn_quant_u1(sum10, *pKappa, *pLambda, out_shift);
                  sum14 = pulp_nn_bn_quant_u1(sum14, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum11 = pulp_nn_bn_quant_u1(sum11, *pKappa, *pLambda, out_shift);
                  sum15 = pulp_nn_bn_quant_u1(sum15, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum12 = pulp_nn_bn_quant_u1(sum12, *pKappa, *pLambda, out_shift);
                  sum16 = pulp_nn_bn_quant_u1(sum16, *pKappa, *pLambda, out_shift);
                  pKappa++;
                  pLambda++;
                  sum1  = bitins(sum1, n_mask1, sum2, mask1, off1);
                  sum1  = bitins(sum1, n_mask2, sum3, mask2, off2);
                  sum1  = bitins(sum1, n_mask3, sum4, mask3, off3);
                  sum1  = bitins(sum1, n_mask4, sum9, mask4, off4);
                  sum1  = bitins(sum1, n_mask5, sum10, mask5, off5);
                  sum1  = bitins(sum1, n_mask6, sum11, mask6, off6);
                  *pOut = bitins(sum1, n_mask7, sum12, mask7, off7);

                  sum5   = bitins(sum5, n_mask1, sum6, mask1, off1);
                  sum5   = bitins(sum5, n_mask2, sum7, mask2, off2);
                  sum5   = bitins(sum5, n_mask3, sum8, mask3, off3);
                  sum5   = bitins(sum5, n_mask4, sum13, mask4, off4);
                  sum5   = bitins(sum5, n_mask5, sum14, mask5, off5);
                  sum5   = bitins(sum5, n_mask6, sum15, mask6, off6);
                  *pOut2 = bitins(sum5, n_mask7, sum16, mask7, off7);
                  pOut2++;
                  pOut++;
    }
    else
    {
      if (flag_relu == 1)
      {
                sum1 = pulp_nn_quant_u1(sum1, out_mult, out_shift);
                sum2 = pulp_nn_quant_u1(sum2, out_mult, out_shift);
                sum3 = pulp_nn_quant_u1(sum3, out_mult, out_shift);
                sum4 = pulp_nn_quant_u1(sum4, out_mult, out_shift);
                sum9 = pulp_nn_quant_u1(sum9, out_mult, out_shift);
                sum10 = pulp_nn_quant_u1(sum10, out_mult, out_shift);
                sum11 = pulp_nn_quant_u1(sum11, out_mult, out_shift);
                sum12 = pulp_nn_quant_u1(sum12, out_mult, out_shift);

                sum1  = bitins(sum1, n_mask1, sum2, mask1, off1);
                sum1  = bitins(sum1, n_mask2, sum3, mask2, off2);
                sum1  = bitins(sum1, n_mask3, sum4, mask3, off3);
                sum1  = bitins(sum1, n_mask4, sum9, mask4, off4);
                sum1  = bitins(sum1, n_mask5, sum10, mask5, off5);
                sum1  = bitins(sum1, n_mask6, sum11, mask6, off6);
                *pOut = bitins(sum1, n_mask7, sum12, mask7, off7);

                pOut++;

                sum5 = pulp_nn_quant_u1(sum5, out_mult, out_shift);
                sum6 = pulp_nn_quant_u1(sum6, out_mult, out_shift);
                sum7 = pulp_nn_quant_u1(sum7, out_mult, out_shift);
                sum8 = pulp_nn_quant_u1(sum8, out_mult, out_shift);
                sum13 = pulp_nn_quant_u1(sum13, out_mult, out_shift);
                sum14 = pulp_nn_quant_u1(sum14, out_mult, out_shift);
                sum15 = pulp_nn_quant_u1(sum15, out_mult, out_shift);
                sum16 = pulp_nn_quant_u1(sum16, out_mult, out_shift);

                sum5   = bitins(sum5, n_mask1, sum6, mask1, off1);
                sum5   = bitins(sum5, n_mask2, sum7, mask2, off2);
                sum5   = bitins(sum5, n_mask3, sum8, mask3, off3);
                sum5   = bitins(sum5, n_mask4, sum13, mask4, off4);
                sum5   = bitins(sum5, n_mask5, sum14, mask5, off5);
                sum5   = bitins(sum5, n_mask6, sum15, mask6, off6);
                *pOut2 = bitins(sum5, n_mask7, sum16, mask7, off7);

                pOut2++;
      }
      else
      {
                sum1  = (uint8_t) clip1(sum1 >> out_shift);
                sum2  = (uint8_t) clip1(sum2 >> out_shift);
                sum3  = (uint8_t) clip1(sum3 >> out_shift);
                sum4  = (uint8_t) clip1(sum4 >> out_shift);
                sum9  = (uint8_t) clip1(sum9 >> out_shift);
                sum10 = (uint8_t) clip1(sum10 >> out_shift);
                sum11 = (uint8_t) clip1(sum11 >> out_shift);
                sum12 = (uint8_t) clip1(sum12 >> out_shift);

                sum1  = bitins(sum1, n_mask1, sum2, mask1, off1);
                sum1  = bitins(sum1, n_mask2, sum3, mask2, off2);
                sum1  = bitins(sum1, n_mask3, sum4, mask3, off3);
                sum1  = bitins(sum1, n_mask4, sum9, mask4, off4);
                sum1  = bitins(sum1, n_mask5, sum10, mask5, off5);
                sum1  = bitins(sum1, n_mask6, sum11, mask6, off6);
                *pOut = bitins(sum1, n_mask7, sum12, mask7, off7);

                pOut++;

                sum5  = (uint8_t) clip1(sum5 >> out_shift);
                sum6  = (uint8_t) clip1(sum6 >> out_shift);
                sum7  = (uint8_t) clip1(sum7 >> out_shift);
                sum8  = (uint8_t) clip1(sum8 >> out_shift);
                sum13 = (uint8_t) clip1(sum13 >> out_shift);
                sum14 = (uint8_t) clip1(sum14 >> out_shift);
                sum15 = (uint8_t) clip1(sum15 >> out_shift);
                sum16 = (uint8_t) clip1(sum16 >> out_shift);

                sum5   = bitins(sum5, n_mask1, sum6, mask1, off1);
                sum5   = bitins(sum5, n_mask2, sum7, mask2, off2);
                sum5   = bitins(sum5, n_mask3, sum8, mask3, off3);
                sum5   = bitins(sum5, n_mask4, sum13, mask4, off4);
                sum5   = bitins(sum5, n_mask5, sum14, mask5, off5);
                sum5   = bitins(sum5, n_mask6, sum15, mask6, off6);
                *pOut2 = bitins(sum5, n_mask7, sum16, mask7, off7);
                pOut2++;
      }
    }
    pA1+=(7 * num_col_im2col_w);
  }
  pOut+=ch_out_r;
  return pOut;
}
