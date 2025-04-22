#
# Makefile (Windows & POSIX compatible)
#
# - 빌드:  make build
# - 실행:  make run
# - 정리:  make clean
# - 포맷:  make format
#

# --------------------------------------------------------------------
# 1) 사용자 설정
# --------------------------------------------------------------------
# CUDA 설치 경로 (Windows)
CUDA_PATH     := C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8
NVCC          := "$(CUDA_PATH)/bin/nvcc.exe"

# 소스 / 출력
SRC           := main.cu
BUILD_DIR     := build
TARGET        := $(BUILD_DIR)/main.exe

# 컴파일 플래그
NVCCFLAGS     := -std=c++11 -arch=sm_86

# --------------------------------------------------------------------
# 2) 기본 타겟
# --------------------------------------------------------------------
.PHONY: all
all: build

# --------------------------------------------------------------------
# 3) 빌드
# --------------------------------------------------------------------
.PHONY: build
build:
ifeq ($(OS),Windows_NT)
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
else
	@mkdir -p $(BUILD_DIR)
endif
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

# --------------------------------------------------------------------
# 4) 실행
# --------------------------------------------------------------------
.PHONY: run
run: build
ifeq ($(OS),Windows_NT)
	@echo "---- 실행: $(TARGET) ----"
	@$(TARGET)
else
	@echo "---- 실행: ./${TARGET} ----"
	@./$(TARGET)
endif

# --------------------------------------------------------------------
# 5) 정리
# --------------------------------------------------------------------
.PHONY: clean
clean:
ifeq ($(OS),Windows_NT)
	@if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
else
	@rm -rf $(BUILD_DIR)
endif

# --------------------------------------------------------------------
# 6) 포맷 (clang-format)
# --------------------------------------------------------------------
.PHONY: format
format:
	@clang-format -i $(SRC)
