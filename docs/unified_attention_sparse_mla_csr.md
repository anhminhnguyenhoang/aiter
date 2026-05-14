# Unified Attention Sparse MLA CSR

## Summary

`unified_attention_sparse_mla` now supports a CSR-style sparse KV list in addition
to the existing dense `topk_indices` matrix. The CSR path is per flattened query
token:

- `kv_indptr`: int32 tensor with shape `[num_tokens + 1]`
- `kv_indices`: int32 tensor with shape `[nnz]`
- `kv_indices` entries are physical KV-cache token positions, matching the old
`topk_indices` convention.

The kernel maps each physical token position with:

```text
physical_block_idx = kv_pos // block_size
slot = kv_pos % block_size
```

Rows with no KV entries are valid and produce zero output.

## Verification

Added CSR correctness coverage in
`op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py` for:

- decode (`s_q == 1`) and multi-query rows
- variable CSR row lengths
- explicit zero-length rows
- both tested block sizes, head counts, and LoRA dimensions

Added benchmark entry point:

```bash
python op_tests/op_benchmarks/triton/bench_unified_attention_sparse_mla.py --validate
```

Runtime verification was run in the `anguyenh-dev` devcontainer with
`amd-aiter` installed in editable mode from `file:///home/anguyenh/aiter_ua`.

Passed:

```bash
python3 -m pytest -v op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py -k csr
python3 -m pytest -v op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py
```

The full sparse MLA test file passed with `176 passed`.

Benchmark results:

```text
batch=8 sq=1 sk=2048 heads=32 lora=512 rope=64 block=64 top_k=64
nnz=512 max_sparse_len=64
topk_matrix_ms=0.0125
csr_ms=0.0138
speedup_vs_topk=0.907x

batch=8 sq=1 sk=32 heads=32 lora=512 rope=64 block=64 top_k=64
nnz=256 max_sparse_len=32
topk_matrix_ms=0.0122
csr_ms=0.0133
speedup_vs_topk=0.915x
```

For these small decode cases, CSR is slightly slower than the fixed-width top-k
matrix path because both launch the same amount of work at the tile level while
CSR adds row-pointer loads. CSR mainly improves representation flexibility and
removes padded index storage.

## Next Steps

- Run the full sparse MLA pytest target in a ROCm/PyTorch/Triton environment.
- Run the new benchmark and record top-k vs CSR latency for representative row
length distributions.
- Add page-level CSR support if index bandwidth becomes a bottleneck.
- Add logical-token CSR support by mapping through `block_table`.
- Consider split-K/reduction for large sparse rows.
- Port MXFP4/RoPE split ideas from `unified_mxfp4_attention` once the BF16 CSR
path is validated.

