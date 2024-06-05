# include "head.h"

std::vector<char> readBinFile(const char *file) {
  std::ifstream filestr(file, std::ios::binary);
  if (!filestr) {
    std::cerr << "file:" << file << " open failed!" << std::endl;
    return {};
  }
  filestr.seekg(0, std::ios::end);
  size_t size = filestr.tellg();
  filestr.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  filestr.read(buffer.data(), size);
  filestr.close();
  return buffer;
}


void AiCoreKernel::loadKernel() {
  if (stub == nullptr) {
    void *binHandle;
    rtDevBinary_t binary;
    auto file = readBinFile((kernelFilePath + kernelFileName).c_str());
    binary.data = file.data();
    binary.length = file.size();
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    binary.version = 0;
    auto temp = std::vector<char>(kernelName.begin(), kernelName.end());
    stub = temp.data();
    RT_CHECK(rtDevBinaryRegister(&binary, &binHandle));
    RT_CHECK(rtFunctionRegister(binHandle, stub, stub, (void *)stub, 0));
  }
}

void AiCoreKernel::launchKernel(void *args, uint32_t argsSize,
                                rtSmDesc_t *smDesc, rtStream_t stream) {
  RT_CHECK(rtKernelLaunch(stub, blockDim, args, argsSize, smDesc, stream));
}