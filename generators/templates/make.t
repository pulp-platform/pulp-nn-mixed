APP = test

APP_SRCS = test.c

%if config.kernel.type == 'maxpool':
ifndef signed
signed=1
endif
ifndef kernel
kernel=${config.kernel.in_data_t}
%elif config.kernel.type == 'linear_no_quant':
ifndef signed
signed=1
endif
ifndef kernel
kernel=88
%elif config.api == 'PulpNNAdd':
ifndef kernel
kernel = 88
%elif config.api == 'PulpNNQuantAdd':
ifndef signed
signed = 111
endif
ifndef kernel
kernel = 888
%elif config.kernel.type == 'avgpool':
ifndef signed
signed = 11
endif
ifndef kernel
kernel = 88
%elif config.kernel.type == 'matmul' or config.kernel.type == 'convolution' or config.kernel.type == 'pointwise' or config.kernel.type == 'depthwise' or config.kernel.type == 'linear_quant':
ifndef signed
signed=11
endif
ifndef kernel
kernel=888
%endif
else
kernel = $(kernel)
endif

${config.make}

ifndef cores
cores=1
else
cores = $(cores)
endif

ifeq ($(perf), 1)
APP_CFLAGS += -DVERBOSE_PERF
endif

ifeq ($(check), 1)
APP_CFLAGS += -DVERBOSE_CHECK
endif

APP_CFLAGS += -O3 -Iinclude -w #-flto
APP_CFLAGS += -DNUM_CORES=$(cores) -DKERNEL=$(kernel) -DSIGNED=$(signed)

APP_LDFLAGS += -lm #-flto


include $(RULES_DIR)/pmsis_rules.mk
