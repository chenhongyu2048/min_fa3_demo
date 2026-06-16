ARCH ?= SM90

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
TK_ROOT ?= $(abspath $(THIS_DIR)/../../third_party/ThunderKittens)

SRC := offset_ipc_repro.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
NPROC ?= 2
CMD := OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$(NPROC) test_repro.py
CONFIG := pytorch
PYBIND_INCLUDES ?=

include $(TK_ROOT)/kernels/common.mk
