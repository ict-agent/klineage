# Ascend generation reproduction

Published experiments are complete. See each operator README for measured results,
protocol corrections and limitations. No automatic new run is authorized.

Environment: ssh 910b, container pjj-fmha-opt, Ascend910B3, NPU2/3, CANN9.1.0.
Use scripts/ascend/task2.sh for environment sync/probe. Configure the provider locally
using scripts/ascend/deepseek.example.toml; never commit credentials.

For a newly authorized FusedAddRmsNorm experiment:

```sh
KLINEAGE_LANGUAGE=ascendc scripts/ascend/task2.sh start --kernels fused_add_rmsnorm
```

Both settings have7200seconds and the same fixed gate. A has no injected expert pack;
B receives the separately supplied expert context. Private context must be provisioned
locally before reproduction; the public artifact does not include all model inputs.
The controller resumes the same session until deadline and stores raw traces locally.
Public sharing requires a separate confidentiality review. Never overwrite completed runs
without explicit authorization. Inputs can be regenerated with scripts/ascend/gen_inputs.py.
