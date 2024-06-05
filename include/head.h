#include "acl/acl.h"
#include "runtime/rt.h"
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// AICORE: various checks for different function calls.
#define ACLRT_CHECK(ret)                                                       \
  do {                                                                         \
    aclError error = ret;                                                    \
    if (error != RT_ERROR_NONE) {                                              \
      std::cerr << "\033[31m"; /* 开始红色文字 */                        \
      std::cerr                                                                \
          << "\n========================= ERROR ============================"  \
          << std::endl;                                                        \
      std::cerr << "File: " << __FILE__ << std::endl;                          \
      std::cerr << "Line: " << __LINE__ << std::endl;                          \
      std::cerr << "[Error]" << error << std::endl;                            \
      std::cerr                                                                \
          << "============================================================\n"  \
          << std::endl;                                                        \
      std::cerr << "\033[0m"; /* 重置文字颜色 */                         \
    }                                                                          \
  } while (0)

#define RT_CHECK(ret)                                                          \
  do {                                                                         \
    rtError_t error = ret;                                                     \
    if (error != RT_ERROR_NONE) {                                              \
      std::cerr << "\033[31m"; /* 开始红色文字 */                        \
      std::cerr                                                                \
          << "\n========================= ERROR ============================"  \
          << std::endl;                                                        \
      std::cerr << "File: " << __FILE__ << std::endl;                          \
      std::cerr << "Line: " << __LINE__ << std::endl;                          \
      std::cerr << "[Error]" << error << std::endl;                            \
      std::cerr                                                                \
          << "============================================================\n"  \
          << std::endl;                                                        \
      std::cerr << "\033[0m"; /* 重置文字颜色 */                         \
    }                                                                          \
  } while (0)

std::vector<char> readBinFile(const char *file);

class AiCoreKernel {
public:
  AiCoreKernel(std::string _kernelFilePath, std::string _kernelFileName,
               std::string _kernelName, uint32_t _blockDim = 1)
      : kernelFilePath(_kernelFilePath), kernelFileName(_kernelFileName),
        kernelName(_kernelName), blockDim(_blockDim) {
    stub = nullptr;
  }
  std::string kernelFilePath;
  std::string kernelFileName;
  std::string kernelName;
  uint32_t blockDim;
  char *stub;
  aclrtBinHandle *funcHandle;
  void loadKernel();
  void launchKernel(void *args, uint32_t argsSize, rtSmDesc_t *smDesc,
                    rtStream_t stream);
};


