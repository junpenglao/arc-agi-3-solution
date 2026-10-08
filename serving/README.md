# SGLang serving build

This folder contains the offline wheelhouse builder and custom serving patches for the ARC-AGI-3 notebook. It builds on [John Pezzulli's Pennyroyal SGLang fork](https://github.com/jpezzulli/sglang-rtxpro6000), pinned to **v2.5.3**, commit `d00d88efc8d6281b12be4f4073126aec95038c55`.

See the [repository README](../README.md) for the solution overview and reproduction path, and the [write-up](../WRITEUP.md) for the reasoning behind the serving and prefix-cache changes.

## Included patches

The [builder](build_bundle_pennyroyal.sh) embeds and applies all five patches automatically:

1. **Low-M BF16 GEMM:** adapted from [Gabriel Olympie's patch 0004](https://github.com/gabrielolympie/sglang-flashnext-sm120/blob/main/patches/0004-sm120-lowm-triton-gemm.patch).
2. **Speculative-state memory budgeting:** adapted from the budget-accounting changes in [patch 0002 in Mamy Ratsimbazafy's repository](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo/blob/master/patches/0002-sm120-gdn-recover-ssm.patch), avoiding reservations for intermediate SSM buffers that are not allocated.
3. **Marlin scale-dtype compatibility:** a local fix for the BF16/FP16 mismatch when loading GPTQ MoE weights.
4. **Prefix-cache retention:** my sparse Mamba prefill checkpoints and LRU refresh on final unlock, helping retain prefixes while games leave generation to execute Python tools.
5. **Bounded checkpoint prefetch:** my [lookahead patch](patches/patch-sglang-prefetch-lookahead.patch), which follows the loader's shard order and prefetches only the next shard while the current one loads.

Standalone patches are provided for inspection and reuse; do not apply them again when using the builder.

## Build the offline bundle

Build online, inside the Kaggle GPU Docker image matching your notebook. No GPU is needed for the build. From the repository root:

```bash
export KAGGLE_IMAGE='gcr.io/kaggle-gpu-images/python:<matching-tag>'
mkdir -p out-penny
docker run --name penny-builder -it \
  -v "$PWD/serving:/src:ro" \
  -v "$PWD/out-penny:/out" \
  "$KAGGLE_IMAGE" bash /src/build_bundle_pennyroyal.sh
```

Upload `out-penny/bundle-pennyroyal` as a Kaggle dataset. The builder checks a fresh offline installation before publishing the bundle. Model weights, tokenizer files, and the draft checkpoint are supplied separately. Use the [competition notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) for the complete launch configuration.

## Run launcher tests

From the repository root, run `make -C serving test`. This uses the locked ARC3-Inference Python 3.12 test dependencies and runs the serving launcher and QSA probe CPU tests without a GPU or model download.

## Prefetch settings

Enable prefetch with `--weight-loader-prefetch-checkpoints`. Lookahead is on by default; `SGLANG_WEIGHT_LOADER_PREFETCH_LOOKAHEAD=0` restores the original eager prefetch behavior.

The patched prefetcher uses `--weight-loader-prefetch-num-threads` (default **4**) to read separate ranges of each shard, in **16 MiB** blocks by default (`SGLANG_PREFETCH_BLOCK_SIZE_MB`). It finishes staging the current shard before loading its tensors, then stages the next shard concurrently. It requires the standard safetensors mmap loader and replaces the independent eager prefetch pass for general deployments. The published competition notebook intentionally also starts its external background prefetcher while enabling `--weight-loader-prefetch-checkpoints`; the [write-up](../WRITEUP.md) records that paired configuration.
