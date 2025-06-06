﻿
## 목차
1. [About CUDA](#about-cuda)  
2. [GPU 확인](#gpu-확인)  
3. [CUDA 개발 환경 세팅](#cuda-개발-환경-세팅)  
4. [Troubleshooting](#troubleshooting)  
   - [cl.exe를 못 찾음](#1-clexe를-못-찾음)  
   - [cuda_runtime.h를 못 찾음](#2-cudaruntimeh를-못-찾음)  
   - [warning C4819 인코딩 경고](#3-warning-c4819-인코딩-경고)  
5. [Reference](#reference)  

---

## About CUDA  
CUDA(Compute Unified Device Architecture)는 NVIDIA에서 개발한 병렬 컴퓨팅 플랫폼 겸 프로그래밍 모델이다.  
GPU의 대규모 병렬 처리 능력을 일반 애플리케이션에서 사용할 수 있게 해 주며, NVIDIA GPU에서만 동작한다.

---

## GPU 확인  
터미널(또는 명령 프롬프트)에서 다음 명령어로 현재 시스템의 NVIDIA GPU 정보를 확인할 수 있다:

```bash
nvidia-smi
```
![GPU 정보](https://github.com/user-attachments/assets/93ef13b6-fd53-414d-be6a-5dfc8c95003d)

- **GPU 모델**  
- **메모리 사용량**  
- **드라이버 버전**  
- 등 다양한 정보를 제공한다.

---

## CUDA 개발 환경 세팅  
1. **CUDA 툴킷 다운로드 & 설치**  
   - [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)  
   - 설치가 완료되면, 터미널에서 버전을 확인하고 다음과 같이 나오면 설치가 완료된 것이다:
     ```bash
     nvcc --version
     ```
   ![nvcc 버전 확인](https://github.com/user-attachments/assets/b5f9d332-381a-48c2-8de3-92d6ba279ff3)


2. **환경 변수 설정**  
   Windows 환경에서 `PATH`에 다음 두 경로를 추가한다:
   ```text
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
   C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<버전>\bin\Hostx64\x64
   ```

---

## Troubleshooting

### 1. cl.exe를 못 찾음
```text
nvcc fatal   : Cannot find compiler 'cl.exe' in PATH
```
- **원인**: `nvcc`가 호출하는 호스트 컴파일러 `cl.exe` 경로가 시스템 `PATH`에 없음  
- **해결**:
  1. 시스템 환경 변수 `PATH`에 `cl.exe` 폴더 추가 후 터미널 재시작
     ```text
     C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64
     ```

---

### 2. cuda_runtime.h를 못 찾음
```text
#include errors detected. Please update your includePath.
cannot open source file "cuda_runtime.h"
```
- **원인**: VS Code IntelliSense 설정(`includePath`)에 CUDA `include` 폴더가 빠져 있음  
- **해결**: `.vscode/c_cpp_properties.json`에 다음을 추가:
  ```jsonc
  "includePath": [
    "${workspaceFolder}/**",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
  ],
  ```

---

### 3. warning C4819 인코딩 경고
```text
warning C4819: The file contains a character that cannot be represented in the current code page (949).
Save the file in Unicode format to prevent data loss
```
- **원인**: MSVC 기본 코드 페이지(CP‑949)로 외부 헤더의 유니코드 문자를 처리 못함  
- **해결**:
  - VS Code에서 **Files: Encoding**을 `UTF-8 with BOM`으로 변경

---

## Reference
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)  
- [nvidia-smi User Guide](https://developer.nvidia.com/nvidia-system-management-interface)  
- [VS Code C/C++ IntelliSense 설정](https://code.visualstudio.com/docs/cpp/config-msvc)

