#pragma once
#include <torch/extension.h>
#include <acl/acl.h>

void launch(const std::vector<torch::Tensor>& inputs,
            const std::vector<torch::Tensor>& outputs,
            aclrtStream stream);
