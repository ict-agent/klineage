/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_FUSED_INFER_ATTENTION_SCORE_INNER_H_
#define ACLNN_FUSED_INFER_ATTENTION_SCORE_INNER_H_
#define ACLNN_API __attribute__((visibility("default")))

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void TensorPreProcess(const aclTensorList *&tensorListKey, const aclTensorList *&tensorListValue);
void PrefixTensorPreProcess(const aclTensor *&tensorKey, const aclTensor *&tensorValue);
aclnnStatus FakeArray(const aclIntArray *inArray, aclTensor *&outArray);

void FusedInferAttentionScoreProcessSoftmaxLse(bool softmaxLseFlag, const aclTensor *softmaxLse,
                                               const aclTensor *&tempTensor, const aclTensor *&placeHolder);


#ifdef __cplusplus
}
#endif

#endif