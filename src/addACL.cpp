#include "acl/acl.h"
#include "runtime/rt.h"
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// AICORE: various checks for different function calls.
// AICORE: various checks for different function calls.
#define AICORE_CHECK(condition)                                                \
  /* Code block avoids redefinition of rtError_t error */                      \
  do {                                                                         \
    rtError_t error = condition;                                               \
    if (error != RT_ERROR_NONE) {                                              \
      std::cout << __LINE__ << "    " << error << std::endl;                   \
    }                                                                          \
  } while (0)

// Stub out GPU calls as unavailable.
class AicoreKerel {
public:
  AicoreKerel(std::string _kernelfile, std::string _kernelname,
              unsigned int _block_num = 1)
      : kernelfile(_kernelfile), kernelname(_kernelname),
        block_num(_block_num) {}
  std::string kernelfile;
  std::string kernelname;
  unsigned int block_num;
};

class AICoreKernelInfo {
public:
  AICoreKernelInfo(char *kernel, int block_num)
      : kernel_(kernel), block_num_(block_num) {}
  const char *const kernel_;
  const int block_num_;
};

std::string kernel_dir = "";
std::map<std::string, std::vector<char>> kernels_holder;
std::vector<AICoreKernelInfo> aicore_kernel_info_;
char *readBinFile(const char *file_name, uint64_t *fileSize) {
  std::filebuf *pbuf;
  std::ifstream filestr;
  size_t size;
  filestr.open(file_name, std::ios::binary);
  if (!filestr) {
    std::cout << "file:" << file_name << " open failed!" << std::endl;
    return NULL;
  }

  pbuf = filestr.rdbuf();
  size = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
  pbuf->pubseekpos(0, std::ios::in);
  char *buffer = (char *)malloc(size);
  if (NULL == buffer) {
    std::cout << "NULL == buffer!" << std::endl;
    return NULL;
  }
  // new char[size];
  pbuf->sgetn(buffer, size);
  *fileSize = size;

  filestr.close();
  return buffer;
}

char *new_load_aicore_kernel(std::string kernel_file, std::string kernel_name) {

  auto iter = kernels_holder.find(kernel_file);
  if (iter == kernels_holder.end()) {
    void *binHandle;
    rtDevBinary_t binary;
    binary.data = readBinFile(
        ("/root/xkyin/TIK/AscendTikSimple/kernel_meta/" + kernel_file).c_str(),
        &binary.length);
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;

    kernels_holder[kernel_file] =
        std::vector<char>(kernel_name.begin(), kernel_name.end());
    kernels_holder[kernel_file].push_back(0);
    char *stub = kernels_holder[kernel_file].data();
    AICORE_CHECK(rtDevBinaryRegister(&binary, &binHandle));
    AICORE_CHECK(rtFunctionRegister(binHandle, stub, stub, (void *)stub, 0));
    return stub;
  } else {
    return iter->second.data();
  }
}

int main() {

  rtStream_t aicore_stream;
  AICORE_CHECK(rtSetDevice(0));
  AICORE_CHECK(rtStreamCreate(&aicore_stream, 0));

  AicoreKerel fw_param(std::string("simple_add.o"),
                       std::string("simple_add__kernel0"), 2);

  aclrtEvent start_event;
  aclrtEvent end_event;
  aclrtCreateEvent(&start_event);
  aclrtCreateEvent(&end_event);

  char *fw_stub =
      new_load_aicore_kernel(fw_param.kernelfile, fw_param.kernelname);
  aicore_kernel_info_.push_back(AICoreKernelInfo(fw_stub, fw_param.block_num));

  int data_len = 128;

  float *d_input1;
  float *d_input2;
  float *d_output;

  AICORE_CHECK(
      rtMalloc((void **)&d_input1, data_len * sizeof(float), RT_MEMORY_HBM));
  AICORE_CHECK(
      rtMalloc((void **)&d_input2, data_len * sizeof(float), RT_MEMORY_HBM));
  AICORE_CHECK(
      rtMalloc((void **)&d_output, data_len * sizeof(float), RT_MEMORY_HBM));

  std::vector<void *> args = {(void *)d_input1, (void *)d_input2,
                              (void *)d_output};

  AICORE_CHECK(rtStreamSynchronize(aicore_stream));

  aclrtRecordEvent(start_event, aicore_stream);
  AICORE_CHECK(rtKernelLaunch(
      aicore_kernel_info_[0].kernel_, aicore_kernel_info_[0].block_num_,
      args.data(), args.size() * sizeof(void *), NULL, aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(aicore_stream));
  aclrtRecordEvent(end_event, aicore_stream);
  aclrtSynchronizeStream(aicore_stream);
  float time = 0;
  aclrtEventElapsedTime(&time, start_event, end_event);

  std::cout << time << "ms" << std::endl;

  AICORE_CHECK(rtStreamSynchronize(aicore_stream));
  
  return 0;
}