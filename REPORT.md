# Configuration Decision Notes

## NB3 OOM on Colab L4

The BIGGPU path trains Qwen3.5-9B on an NVIDIA L4 with about 22 GB of VRAM. NB1
measured p95 = 192 tokens, while the rounded suggestion was 256. The 256-token run
still OOMed at the first NB3 optimizer step, so the BIGGPU tier now uses
`max_length=192`, which matches the measured p95 exactly.

After that, NB3 still OOMed with a physical batch of 2. The next change keeps the
effective batch at 16 for experiment fairness, but lowers the physical batch:
`per_device_batch=1`, `grad_accum=16`.

The remaining OOM happened inside TRL's `chunked_nll` path during a large fp32
`lm_head` matmul. For this short 192-token BIGGPU/L4 path, standard `loss_type="nll"`
is used instead. T4 and the normal lab path keep `chunked_nll`.
