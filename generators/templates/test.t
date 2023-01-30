/*
 * test.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Alessandro Nadalini <alessandro.nadalini3@unibo.it>
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


#define CHECK

#define PERFORMANCE

${config.include}

void test();
void pulp_parallel();

void pulp_parallel()
{
  pi_cl_team_fork(NUM_CORES, (void *)test, NULL);
}

void test()
{
  uint32_t errors = 0;

  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;

%if (config.kernel.extensions == 'XpulpNN-mixed'):
%if (config.kernel.type == 'convolution') or (config.kernel.type == 'pointwise'):
LEGACY_MODE("0");
%if config.kernel.matmul_fmt == '4x2':
A_SKIP("1");
W_SKIP("3");
MIXED_SKIP("8");
%elif config.kernel.matmul_fmt == '4x4':
A_SKIP("3");
W_SKIP("3");
MIXED_SKIP("16");
%endif
#if (KERNEL == 888) || (KERNEL == 848) || (KERNEL == 828) || (KERNEL == 488) || (KERNEL == 448) || (KERNEL == 428) || (KERNEL == 288) || (KERNEL == 248) || (KERNEL == 228)
IVEC_FMT("2");
#elif (KERNEL == 484) || (KERNEL == 444) || (KERNEL == 424) || (KERNEL == 284) || (KERNEL == 244) || (KERNEL == 224)
IVEC_FMT("3");
#elif (KERNEL == 282) || (KERNEL == 242) || (KERNEL == 222)
IVEC_FMT("4");
#elif (KERNEL == 482) || (KERNEL == 442) || (KERNEL == 422)
IVEC_FMT("5");
#elif (KERNEL == 882) || (KERNEL == 842) || (KERNEL == 822)
IVEC_FMT("6");
#elif (KERNEL == 884) || (KERNEL == 844) || (KERNEL == 824)
IVEC_FMT("8");
#else
IVEC_FMT("0");
#endif
%elif (config.kernel.type == 'depthwise'):
  LEGACY_MODE("0");
  IVEC_FMT("2");
  #if (KERNEL == 888)
  A_SKIP("1");
  W_SKIP("0");
  #elif (KERNEL == 884) || (KERNEL == 848) || (KERNEL == 844) || (KERNEL == 488) || (KERNEL == 448) || (KERNEL == 484) || (KERNEL == 444)
  A_SKIP("1");
  W_SKIP("1");
  #elif (KERNEL == 882) || (KERNEL == 842) || (KERNEL == 828) || (KERNEL == 824) || (KERNEL == 822) || (KERNEL == 482) || (KERNEL == 442) || (KERNEL == 428) || (KERNEL == 424) || (KERNEL == 422) || (KERNEL == 288) || (KERNEL == 248) || (KERNEL == 228) || (KERNEL == 284) || (KERNEL == 244) || (KERNEL == 224) || (KERNEL == 282) || (KERNEL == 242) || (KERNEL == 222)
  A_SKIP("3");
  W_SKIP("3");
  #endif
%elif (config.kernel.type == 'linear_no_quant'):
  LEGACY_MODE("0");
  A_SKIP("0");
  W_SKIP("0");
  A_STRIDE(0);
  W_STRIDE(0);
  A_ROLLBACK(4);
  W_ROLLBACK(4);
  #if (KERNEL == 88) || (KERNEL == 48) || (KERNEL == 28)
  IVEC_FMT("2");
  #elif (KERNEL == 44) || (KERNEL == 24)
  IVEC_FMT("3");
  #elif KERNEL == 22
  IVEC_FMT("4");
  #elif KERNEL == 42
  IVEC_FMT("5");
  MIXED_SKIP("1");
  #elif KERNEL == 82
  IVEC_FMT("6");
  MIXED_SKIP("1");
  #elif KERNEL == 84
  IVEC_FMT("8");
  MIXED_SKIP("1");
  #else
  IVEC_FMT("0");
  #endif
%elif (config.kernel.type == 'linear_quant'):
  LEGACY_MODE("0");
  #if KERNEL == 222
  IVEC_FMT("4");
  A_SKIP("0");
  W_SKIP("3");
  #elif KERNEL == 224
  IVEC_FMT("3");
  A_SKIP("0");
  W_SKIP("3");
  #elif KERNEL == 228
  IVEC_FMT("2");
  A_SKIP("0");
  W_SKIP("3");
  #elif KERNEL == 242
  IVEC_FMT("4");
  A_SKIP("0");
  W_SKIP("1");
  #elif KERNEL == 244
  IVEC_FMT("3");
  A_SKIP("0");
  W_SKIP("1");
  #elif KERNEL == 248
  IVEC_FMT("2");
  A_SKIP("0");
  W_SKIP("1");
  #elif KERNEL == 282
  IVEC_FMT("4");
  A_SKIP("0");
  W_SKIP("0");
  #elif KERNEL == 284
  IVEC_FMT("3");
  A_SKIP("0");
  W_SKIP("0");
  #elif KERNEL == 288
  IVEC_FMT("2");
  A_SKIP("0");
  W_SKIP("0");
  #elif KERNEL == 422
  IVEC_FMT("5");
  MIXED_SKIP("4");
  A_SKIP("0");
  W_SKIP("3");
  #elif KERNEL == 424
  IVEC_FMT("3");
  A_SKIP("0");
  W_SKIP("3");
  #elif KERNEL == 428
  IVEC_FMT("2");
  A_SKIP("0");
  W_SKIP("3");
  #elif KERNEL == 442
  IVEC_FMT("5");
  MIXED_SKIP("2");
  A_SKIP("0");
  W_SKIP("1");
  #elif KERNEL == 444
  IVEC_FMT("3");
  A_SKIP("0");
  W_SKIP("1");
  #elif KERNEL == 448
  IVEC_FMT("2");
  A_SKIP("0");
  W_SKIP("1");
  #elif KERNEL == 482
  IVEC_FMT("5");
  MIXED_SKIP("1");
  A_SKIP("0");
  W_SKIP("0");
  #elif KERNEL == 484
  IVEC_FMT("3");
  A_SKIP("0");
  W_SKIP("0");
  #elif KERNEL == 488
  IVEC_FMT("2");
  A_SKIP("0");
  W_SKIP("0");
  #elif KERNEL == 822
  IVEC_FMT("6");
  MIXED_SKIP("4");
  A_SKIP("0");
  W_SKIP("3");
  #elif KERNEL == 824
  IVEC_FMT("8");
  MIXED_SKIP("4");
  A_SKIP("0");
  W_SKIP("3");
  #elif KERNEL == 828
  IVEC_FMT("2");
  A_SKIP("0");
  W_SKIP("3");
  #elif KERNEL == 842
  IVEC_FMT("6");
  MIXED_SKIP("2");
  A_SKIP("0");
  W_SKIP("1");
  #elif KERNEL == 844
  IVEC_FMT("8");
  MIXED_SKIP("2");
  A_SKIP("0");
  W_SKIP("1");
  #elif KERNEL == 848
  IVEC_FMT("2");
  A_SKIP("0");
  W_SKIP("1");
  #elif KERNEL == 882
  IVEC_FMT("6");
  MIXED_SKIP("1");
  A_SKIP("0");
  W_SKIP("0");
  #elif KERNEL == 884
  IVEC_FMT("8");
  MIXED_SKIP("1");
  A_SKIP("0");
  W_SKIP("0");
  #elif KERNEL == 888
  IVEC_FMT("2");
  A_SKIP("0");
  W_SKIP("0");
  #else
  IVEC_FMT("0");
  A_SKIP("0");
  W_SKIP("0");
  #endif
%endif
%endif

  if(pi_core_id()==0)
  {
#ifdef PERFORMANCE
#ifdef VERBOSE_PERF
%if config.kernel.type == 'depthwise':
    printf("MACs=%d\n", DIM_KERNEL_X * DIM_KERNEL_Y * DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT);
%elif config.kernel.type == 'convolution' or config.kernel.type == 'pointwise':
    printf("MACs=%d\n", DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT);
%elif config.kernel.type == 'linear_no_quant':
    printf("MACs=%d\n", CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT);
%elif config.kernel.type == 'linear_quant':
    printf("MACs=%d\n", CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT);
%endif
#endif /* VERBOSE */
#endif /* PERFORMANCE */
%if config.kernel.type == 'add':
#if INPUT1 == 2
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2); i++)
    {
      uint8_t c_data = bitins(IN1_INT2_L2[4*i], n_mask2, IN1_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, IN1_INT2_L2[(4*i)+2], mask4, off4);
      IN1_INT8_L1[i] = bitins(c_data, n_mask6, IN1_INT2_L2[(4*i)+3], mask6, off6);
    }
#endif /* INPUT1 */
#if INPUT1 == 4
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1); i++)
    {
      IN1_INT8_L1[i] = bitins(IN1_INT4_L2[2*i], n_mask, IN1_INT4_L2[(2*i)+1], mask, off);
    }
#endif /* INPUT1 */
#if INPUT1 == 8
    for(int i=0; i<(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN); i++)
    {
      IN1_INT8_L1[i] = IN1_INT8_L2[i];
    }
#endif /* INPUT1 */
#if INPUT2 == 2
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2); i++)
    {
      uint8_t c_data = bitins(IN2_INT2_L2[4*i], n_mask2, IN2_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, IN2_INT2_L2[(4*i)+2], mask4, off4);
      IN2_INT8_L1[i] = bitins(c_data, n_mask6, IN2_INT2_L2[(4*i)+3], mask6, off6);
    }
#endif /* INPUT2 */
#if INPUT2 == 4
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1); i++)
    {
      IN2_INT8_L1[i] = bitins(IN2_INT4_L2[2*i], n_mask, IN2_INT4_L2[(2*i)+1], mask, off);
    }
#endif /* INPUT2 */
#if INPUT2 == 8
    for(int i=0; i<(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN); i++)
    {
      IN2_INT8_L1[i] = IN2_INT8_L2[i];
    }
#endif /* INPUT2 */
%else:
#if INPUT == 2
%if config.kernel.type == 'depthwise':
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2); i++)
    {
      uint8_t c_data = bitins(IN_INT2_L2_HWC[4*i], n_mask2, IN_INT2_L2_HWC[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, IN_INT2_L2_HWC[(4*i)+2], mask4, off4);
      IN_INT8_L2_HWC[i] = bitins(c_data, n_mask6, IN_INT2_L2_HWC[(4*i)+3], mask6, off6);
    }
    for(int i=0; i<(CH_IM_IN >> 2); i++)
    {
      for(int j=0; j<(DIM_IM_IN_X * DIM_IM_IN_Y); j++)
      {
        IN_INT8_L1_CHW[(i * DIM_IM_IN_X * DIM_IM_IN_Y) + j] = IN_INT8_L2_HWC[(j * (CH_IM_IN >> 2)) + i];
      }
    }
%elif config.kernel.type == 'convolution' or config.kernel.type == 'pointwise' or config.kernel.type == 'linear_no_quant' or config.kernel.type == 'linear_quant' or config.kernel.type == 'maxpool' or config.kernel.type == 'avgpool':
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2); i++)
    {
      uint8_t c_data = bitins(IN_INT2_L2[4*i], n_mask2, IN_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, IN_INT2_L2[(4*i)+2], mask4, off4);
      IN_INT8_L1[i] = bitins(c_data, n_mask6, IN_INT2_L2[(4*i)+3], mask6, off6);
    }
%endif
#elif INPUT == 4
%if config.kernel.type == 'depthwise':
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1); i++)
    {
      IN_INT8_L2_HWC[i] = bitins(IN_INT4_L2_HWC[2*i], n_mask, IN_INT4_L2_HWC[(2*i)+1], mask, off);
    }
    for(int i=0; i<(CH_IM_IN >> 1); i++)
    {
      for(int j=0; j<(DIM_IM_IN_X * DIM_IM_IN_Y); j++)
      {
        IN_INT8_L1_CHW[(i * DIM_IM_IN_X * DIM_IM_IN_Y) + j] = IN_INT8_L2_HWC[(j * (CH_IM_IN >> 1)) + i];
      }
    }
%elif config.kernel.type == 'convolution' or config.kernel.type == 'pointwise' or config.kernel.type == 'linear_no_quant' or config.kernel.type == 'linear_quant' or config.kernel.type == 'maxpool' or config.kernel.type == 'avgpool':
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1); i++)
    {
      IN_INT8_L1[i] = bitins(IN_INT4_L2[2*i], n_mask, IN_INT4_L2[(2*i)+1], mask, off);
    }
%endif
#elif INPUT == 8
%if config.kernel.type == 'depthwise':
    for(int i=0; i<CH_IM_IN; i++)
    {
      for(int j=0; j<(DIM_IM_IN_X * DIM_IM_IN_Y); j++)
      {
        IN_INT8_L1_CHW[(i * DIM_IM_IN_X * DIM_IM_IN_Y) + j] = IN_INT8_L2_HWC[(j * CH_IM_IN) + i];
      }
    }
%elif config.kernel.type == 'convolution' or config.kernel.type == 'pointwise' or config.kernel.type == 'linear_no_quant' or config.kernel.type == 'linear_quant' or config.kernel.type == 'maxpool' or config.kernel.type == 'avgpool':
    for(int i=0; i<(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN); i++)
    {
      IN_INT8_L1[i] = IN_INT8_L2[i];
    }
%endif
#endif /* INPUT */
%if config.kernel.type == 'depthwise':
#if WEIGHTS == 2
    for(int i=0; i<(DIM_KERNEL_X * DIM_KERNEL_Y); i++)
    {
      for(int j=0; j<CH_IM_IN; j++)
      {
        WEIGHT_INT2_L2_HWC[(i * CH_IM_IN) + j] = WEIGHT_INT2_L2_CHW[(j * (DIM_KERNEL_X * DIM_KERNEL_Y)) + i];
      }
    }
    for(int i=0; i<((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 2); i++)
    {
      int8_t c_data = bitins(WEIGHT_INT2_L2_HWC[4*i], n_mask2, WEIGHT_INT2_L2_HWC[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, WEIGHT_INT2_L2_HWC[(4*i)+2], mask4, off4);
      WEIGHT_INT8_L2_HWC[i] = bitins(c_data, n_mask6, WEIGHT_INT2_L2_HWC[(4*i)+3], mask6, off6);
    }
    for(int i=0; i<(CH_IM_IN >> 2); i++)
    {
      for(int j=0; j<(DIM_KERNEL_X * DIM_KERNEL_Y); j++)
      {
        WEIGHT_INT8_L1_CHW[(i * DIM_KERNEL_X * DIM_KERNEL_Y) + j] = WEIGHT_INT8_L2_HWC[(j * (CH_IM_IN >> 2)) + i];
      }
    }
#elif WEIGHTS == 4
    for(int i=0; i<(DIM_KERNEL_X * DIM_KERNEL_Y); i++)
    {
      for(int j=0; j<CH_IM_IN; j++)
      {
        WEIGHT_INT4_L2_HWC[(i * CH_IM_IN) + j] = WEIGHT_INT4_L2_CHW[(j * (DIM_KERNEL_X * DIM_KERNEL_Y)) + i];
      }
    }
    for(int i=0; i<((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 1); i++)
    {
      WEIGHT_INT8_L2_HWC[i] = bitins(WEIGHT_INT4_L2_HWC[2*i], n_mask, WEIGHT_INT4_L2_HWC[(2*i)+1], mask, off);
    }
    for(int i=0; i<(CH_IM_IN >> 1); i++)
    {
      for(int j=0; j<(DIM_KERNEL_X * DIM_KERNEL_Y); j++)
      {
        WEIGHT_INT8_L1_CHW[(i * DIM_KERNEL_X * DIM_KERNEL_Y) + j] = WEIGHT_INT8_L2_HWC[(j * (CH_IM_IN >> 1)) + i];
      }
    }
#elif WEIGHTS == 8
    for(int i=0; i<(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN); i++)
    {
      WEIGHT_INT8_L1_CHW[i] = WEIGHT_INT8_L2_CHW[i];
    }
#endif /* WEIGHTS */
%elif config.kernel.type == 'convolution' or config.kernel.type == 'pointwise':
#if WEIGHTS == 2
    for(int i=0; i<((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 2); i++)
    {
      int8_t c_data = bitins(WEIGHT_INT2_L2[4*i], n_mask2, WEIGHT_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, WEIGHT_INT2_L2[(4*i)+2], mask4, off4);
      WEIGHT_INT8_L1[i] = bitins(c_data, n_mask6, WEIGHT_INT2_L2[(4*i)+3], mask6, off6);
    }
#elif WEIGHTS == 4
    for(int i=0; i<((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 1); i++)
    {
      WEIGHT_INT8_L1[i] = bitins(WEIGHT_INT4_L2[2*i], n_mask, WEIGHT_INT4_L2[(2*i)+1], mask, off);
    }
#elif WEIGHTS == 8
    for(int i=0; i<(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT); i++)
    {
      WEIGHT_INT8_L1[i] = WEIGHT_INT8_L2[i];
    }
#endif /* WEIGHTS */
%elif config.kernel.type == 'linear_no_quant' or config.kernel.type == 'linear_quant':
#if WEIGHTS == 2
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN * CH_IM_OUT) >> 2); i++)
    {
      int8_t c_data = bitins(WEIGHT_INT2_L2[4*i], n_mask2, WEIGHT_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, WEIGHT_INT2_L2[(4*i)+2], mask4, off4);
      WEIGHT_INT8_L1[i] = bitins(c_data, n_mask6, WEIGHT_INT2_L2[(4*i)+3], mask6, off6);
    }
#elif WEIGHTS == 4
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN * CH_IM_OUT) >> 1); i++)
    {
      WEIGHT_INT8_L1[i] = bitins(WEIGHT_INT4_L2[2*i], n_mask, WEIGHT_INT4_L2[(2*i)+1], mask, off);
    }
#elif WEIGHTS == 8
    for(int i=0; i<(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN * CH_IM_OUT); i++)
    {
      WEIGHT_INT8_L1[i] = WEIGHT_INT8_L2[i];
    }
#endif /* WEIGHTS */
%endif
%if config.layer.bn == True and config.kernel.type not in ['add', 'avgpool']:
    for(int i=0; i<CH_IM_OUT; i++)
    {
      KAPPA_L1[i] = KAPPA_L2[i];
    }
    for(int i=0; i<CH_IM_OUT; i++)
    {
      LAMBDA_L1[i] = LAMBDA_L2[i];
    }
%endif
%if config.kernel.quantization == 'thresholds':
#if OUTPUT == 2
    for(int i=0; i<(CH_IM_OUT << 2); i++)
    {
      THR_INT2_L1[i] = THR_INT2_L2[i];
    }
#elif OUTPUT == 4
    for(int i=0; i<(CH_IM_OUT << 4); i++)
    {
      THR_INT4_L1[i] = THR_INT4_L2[i];
    }
#endif /*OUTPUT */
%endif
%endif
  }
  pi_cl_team_barrier(0);

#ifdef PERFORMANCE
#ifdef VERBOSE_PERF
for (int k=0; k < 13; k++)
{
  if ( (k>=3))
  {         
    if(k==3) pi_perf_conf(1<<PI_PERF_CYCLES);     
    if(k==4) pi_perf_conf(1<<PI_PERF_IMISS);      
    if(k==5) pi_perf_conf(1<<PI_PERF_LD_EXT);   
    if(k==6) pi_perf_conf(1<<PI_PERF_ST_EXT);    
    if(k==7) pi_perf_conf(1<<PI_PERF_TCDM_CONT);   
    if(k==8) pi_perf_conf(1<<PI_PERF_INSTR);     
    if(k==9) pi_perf_conf(1<<PI_PERF_ACTIVE_CYCLES);    
    if(k==10) pi_perf_conf(1<<PI_PERF_LD_STALL);    
    if(k==11) pi_perf_conf(1<<PI_PERF_JR_STALL);   
    if(k==12) pi_perf_conf(1<<PI_PERF_BRANCH);   
    pi_perf_reset();            
    pi_perf_stop();           
    pi_perf_start(); 
  }
#else
  pi_perf_conf(1<<PI_PERF_CYCLES);          
  pi_perf_reset();                      
  pi_perf_stop();                       
  pi_perf_start(); 
#endif /* VERBOSE */
#endif /* PERFORMANCE */

${config.call}

#ifdef PERFORMANCE
#ifdef VERBOSE_PERF
  if( (k>=3 ))
  {
    pi_perf_stop();      
    int cid = pi_core_id();   
    if(k==3) printf("[%d] : num_cycles: %d\n",cid,pi_perf_read(PI_PERF_CYCLES) ); 
    if(k==4) printf("[%d] : num_instr_miss: %d\n",cid,pi_perf_read(PI_PERF_IMISS) ); 
    if(k==5) printf("[%d] : num_ext_load: %d\n",cid,pi_perf_read(PI_PERF_LD_EXT) ); 
    if(k==6) printf("[%d] : num_ext_Store: %d\n",cid,pi_perf_read(PI_PERF_ST_EXT) ); 
    if(k==7) printf("[%d] : num_tcdm_contentions: %d\n",cid,pi_perf_read(PI_PERF_TCDM_CONT) ); 
    if(k==8) printf("[%d] : num_instrs: %d\n",cid,pi_perf_read(PI_PERF_INSTR) ); 
    if(k==9) printf("[%d] : num_active_cycles: %d\n",cid,pi_perf_read(PI_PERF_ACTIVE_CYCLES) ); 
    if(k==10) printf("[%d] : num_load_stalls: %d\n",cid,pi_perf_read(PI_PERF_LD_STALL ) ); 
    if(k==11) printf("[%d] : num_jumpr_stalls: %d\n",cid,pi_perf_read(PI_PERF_JR_STALL ) ); 
    if(k==12) printf("[%d] : num_branch: %d\n",cid,pi_perf_read(PI_PERF_BRANCH ) ); 
   }             
 }
#else
  pi_perf_stop();          
  int cid = pi_core_id();   
  int perf_cyc =  pi_perf_read(PI_PERF_CYCLES);
%if config.kernel.type == 'depthwise':
  int MACs = DIM_KERNEL_X * DIM_KERNEL_Y * DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT;
%elif config.kernel.type == 'convolution' or config.kernel.type == 'pointwise':
  int MACs = DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT;
%elif config.kernel.type == 'linear_no_quant':
  int MACs = CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT;
%elif config.kernel.type == 'linear_quant':
  int MACs = CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT;
%endif
%if config.kernel.type != 'maxpool' and config.kernel.type != 'avgpool' and config.kernel.type != 'add':
  float perf_MAC =  (float)MACs/perf_cyc;
  if (cid == 0)
  {
    printf("\n[%d] : num_cycles: %d\n",cid,perf_cyc); 
    printf("[%d] : MACs: %d\n",cid,MACs ); 
    printf("[%d] : MAC/cycle: %f\n",cid,perf_MAC ); 
    printf("[%d] : n. of Cores: %d\n",cid,NUM_CORES); 
  }
%else:
  if (cid == 0)
  {
    printf("\n[%d] : num_cycles: %d\n",cid,perf_cyc); 
  }
%endif
#endif /* VERBOSE */
  pi_cl_team_barrier(0);
#endif /* PERFORMANCE */
#ifdef CHECK
  if(pi_core_id()==0)
  {
%if config.kernel.type == 'linear_no_quant':
    for (int i=0; i<CH_IM_OUT; i++)
    {
      if(OUT_L1[i] != OUT_L2[i])
      {
#ifdef VERBOSE_CHECK
        printf("error at index %d, %d instead of %d\n", i, OUT_L1[i], OUT_L2[i]);
#endif /* VERBOSE */
        errors++;
      }
    }
    printf("errors: %d\n", errors);
%elif config.kernel.type == 'linear_quant':
#if OUTPUT == 2
    for (int i=0; i<(CH_IM_OUT >> 2); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<(CH_IM_OUT >> 2); i++)
    {
      uint8_t c_data = bitins(OUT_INT2_L2[4*i], n_mask2, OUT_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, OUT_INT2_L2[(4*i)+2], mask4, off4);
      OUT_INT8_L2[i] = bitins(c_data, n_mask6, OUT_INT2_L2[(4*i)+3], mask6, off6);
#elif OUTPUT == 4
    for (int i=0; i<(CH_IM_OUT >> 1); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<(CH_IM_OUT >> 1); i++)
    {
      OUT_INT8_L2[i] = bitins(OUT_INT4_L2[2*i], n_mask, OUT_INT4_L2[(2*i)+1], mask, off);
#elif OUTPUT == 8
    for (int i=0; i<CH_IM_OUT; i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<CH_IM_OUT; i++)
    {
#endif /* OUTPUT */
      if(OUT_L2[i] != OUT_INT8_L2[i])
      {
#ifdef VERBOSE_CHECK
        printf("error at index %d, %d instead of %d\n", i, OUT_L2[i], OUT_INT8_L2[i]);
#endif /* VERBOSE */
        errors++;
      }
    }
    printf("errors: %d\n", errors);
%else:
#if OUTPUT == 2
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2); i++)
    {
      uint8_t c_data = bitins(OUT_INT2_L2[4*i], n_mask2, OUT_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, OUT_INT2_L2[(4*i)+2], mask4, off4);
      OUT_INT8_L2[i] = bitins(c_data, n_mask6, OUT_INT2_L2[(4*i)+3], mask6, off6);
#elif OUTPUT == 4
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1); i++)
    {
      OUT_INT8_L2[i] = bitins(OUT_INT4_L2[2*i], n_mask, OUT_INT4_L2[(2*i)+1], mask, off);
#elif OUTPUT == 8
    for (int i=0; i<(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT); i++)
    {
#endif /* OUTPUT */
      if(OUT_L2[i] != OUT_INT8_L2[i])
      {
%if config.kernel.type in ['maxpool', 'avgpool']:
#if defined(VERBOSE_CHECK) && defined(VERBOSE_PERF)
#else
#ifdef VERBOSE_CHECK
        printf("error at index %d, %d instead of %d\n", i, OUT_L2[i], OUT_INT8_L2[i]);
#endif
#endif /* VERBOSE */
%else:
#ifdef VERBOSE_CHECK
        printf("error at index %d, %d instead of %d\n", i, OUT_L2[i], OUT_INT8_L2[i]);
#endif /* VERBOSE */
%endif
        errors++;
      }
    }
%if config.kernel.type == 'maxpool':
    printf("errors: %d\nNOTE: Errors detection may not work if you have used perf=1. Pooling kernels overwrites the inputs located in L1\n", errors);
%else:
    printf("errors: %d\n", errors);
%endif
%endif
  }
  pi_cl_team_barrier(0);
#endif /* CHECK */
}

///////////////////////////////////////////////////////////////////
////------------------------MAIN------------------------------/////
///////////////////////////////////////////////////////////////////

int main()
{
  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  // task parameters allocation
  pi_cluster_task(&cluster_task, pulp_parallel, NULL);
  cluster_task.stack_size = 1024;
  cluster_task.slave_stack_size = 1024;
  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return -1;
  // Then offload an entry point, this will get executed on the cluster controller
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  // closing of the cluster
  pi_cluster_close(&cluster_dev);

  return 0;
}
