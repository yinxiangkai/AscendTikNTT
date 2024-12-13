/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef __CLANG_CCE_RUNTIME_H__
#define __CLANG_CCE_RUNTIME_H__
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// This interface is provided by runtime, it needs to be kept the same as their.
/**
 * @ingroup dvrt_clang_cce_runtime
 * @brief Config kernel launch parameters
 * @param [in] numBlocks block dimemsions
 * @param [in|out] smDesc  L2 memory usage control information
 * @param [in|out] stream associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for invalid resource handle
 */
#ifdef __cplusplus
unsigned int rtConfigureCall(uint32_t numBlocks, void *smDesc = nullptr, void *stream = nullptr);
#else  // __cplusplus
unsigned int rtConfigureCall(uint32_t numBlocks, void *smDesc, void *stream);
#endif

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __CLANG_CCE_RUNTIME_H__
