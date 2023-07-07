<%
def iu(sgn):
    return 'i' if sgn else 'u'
in_w = config.kernel.in_data_t
out_w = config.kernel.out_data_t
in_t = f"{iu(config.kernel.in_signed)}{config.kernel.in_data_t}"
out_t = f"{iu(config.kernel.out_signed)}{config.kernel.out_data_t}"

%>\
%if config.api == 'PULPNNMatMul':
#if (KERNEL == ${in_w}${out_w}${config.kernel.wt_data_t}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
#define OUTPUT ${config.kernel.out_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelMatMul/golden_${in_t}_${out_t}_${config.kernel.wt_data_t}.h"
#include "DataAllocationMatMul/data_allocation_${in_t}_${out_t}_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNConvolve':
#if (KERNEL == ${in_w}${out_w}${config.kernel.wt_data_t}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.out_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelConvolution/golden_${in_t}_${out_t}_${config.kernel.wt_data_t}.h"
#include "DataAllocationConvolution/data_allocation_${in_t}_${out_t}_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNConvolvePointwise':
#if (KERNEL == ${in_w}${out_w}${config.kernel.wt_data_t}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.out_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelPointwise/golden_${in_t}_${out_t}_${config.kernel.wt_data_t}.h"
#include "DataAllocationPointwise/data_allocation_${in_t}_${out_t}_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNConvolveDepthwise':
#if (KERNEL == ${in_w}${out_w}${config.kernel.wt_data_t}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.out_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelDepthwise/golden_${in_t}_${out_t}_${config.kernel.wt_data_t}.h"
#include "DataAllocationDepthwise/data_allocation_${in_t}_${out_t}_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNLinearNoQuant':
#if (KERNEL == ${in_w}${config.kernel.wt_data_t}) && (SIGNED == ${int(config.kernel.in_signed)})
#define INPUT ${config.kernel.in_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelLinearNoQuant/golden_${in_t}_i32_${config.kernel.wt_data_t}.h"
#include "DataAllocationLinearNoQuant/data_allocation_${in_t}_i32_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNLinearQuant':
#if (KERNEL == ${in_w}${out_w}${config.kernel.wt_data_t}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.out_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelLinearQuant/golden_${in_t}_${out_t}_${config.kernel.wt_data_t}.h"
#include "DataAllocationLinearQuant/data_allocation_${in_t}_${out_t}_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNMaxPool':
#if (KERNEL == ${config.kernel.in_data_t}) && (SIGNED== ${int(config.kernel.in_signed)})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.in_data_t}
#include "GoldenModelMaxPool/golden_${in_t}_${in_t}.h"
#include "DataAllocationMaxPool/data_allocation_${in_t}_${in_t}.h"
#endif
%elif config.api == 'PULPNNAvgPoolNew':
#if (KERNEL == ${in_w}${out_w}) && (SIGNED == ${f"{int(config.kernel.in_signed)}{int(config.kernel.out_signed)}"})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.out_data_t}
#include "GoldenModelAvgPool/golden_${in_t}_${out_t}.h"
#include "DataAllocationAvgPool/data_allocation_${in_t}_${out_t}.h"
#endif
%elif config.api == 'PULPNNAdd':
#if (KERNEL == ${config.in1_data_t}${config.in2_data_t})
#define INPUT1 ${config.in1_data_t}
#define INPUT2 ${config.in2_data_t}
#define OUTPUT ${config.max_precision}
#include "GoldenModelAdd/golden_${config.in1_data_t}_${config.in2_data_t}.h"
#include "DataAllocationAdd/data_allocation_${config.in1_data_t}_${config.in2_data_t}.h"
#endif
%endif
