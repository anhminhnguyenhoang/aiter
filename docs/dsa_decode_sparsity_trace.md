# End-to-end trace: how DSA sparsity is handled in decode

This document traces, top to bottom, how DeepSeek Sparse Attention (DSA) — the
top-K token selection used by **DeepSeek V3.2** and **GLM-5** — is handled
during decode when SGLang is launched with `--nsa-decode-backend aiter`.

The story is simple: **the indexer-selected top-K page IDs flow through several
layers to land in a single ASM kernel argument that controls which KV pages are
read.** The kernel itself is the standard dense MLA decode kernel; sparsity is
realized purely by what gets put into one pointer.

Split-K (KV-partition parallelization) is intentionally omitted here — it is an
orthogonal GPU-utilization concern and not required to understand sparsity.

## Layer 1 — Indexer produces the top-K page table (SGLang)

The DSA "lightning indexer" runs in the model layer, scores all preceding tokens
against the current query, and SGLang's `topk_transform` writes the top-K page
IDs into a 2-D table called `page_table_1`:

```python
# sglang/srt/layers/attention/nsa_backend.py
@dataclass(frozen=True)
class NSAMetadata:
    # Page table, the index of KV Cache Tables/Blocks
    # this table is always with page_size = 1
    page_table_1: torch.Tensor
    ...
```

Shape is `[batch, max_top_k]`, dtype `int32`. Slots that the indexer didn't
select are filled with `-1`. This is the only place where "sparsity" actually
gets decided — every step below just transports the result.

## Layer 2 — SGLang flattens valid IDs into `kv_indices` (`_forward_aiter`)

`_forward_aiter` is the entry point when `--nsa-decode-backend aiter` is
selected. It compacts `page_table_1` into a CSR-style pair
`(kv_indptr, kv_indices)`:

```python
# sglang/srt/layers/attention/nsa_backend.py, _forward_aiter
non_minus1_mask = page_table_1 != -1
non_minus1_counts = non_minus1_mask.sum(dim=1)
kv_indptr[1 : bs + 1] = torch.cumsum(non_minus1_counts, dim=0)

kv_indices = self.kv_indices
get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)
```

After this:

- `kv_indices` = flat list of *valid* page IDs (the actual top-K, with no
`-1` padding).
- `kv_indptr[b..b+1]` slices into `kv_indices` to give batch `b`'s page IDs.

This is the "sparse subset" the kernel will operate on.

## Layer 3 — Hand off to aiter (`_forward_aiter` → `mla.mla_decode_fwd`)

```python
# sglang/srt/layers/attention/nsa_backend.py, _forward_aiter
mla_decode_fwd(
    q.view(-1, layer.tp_q_head_num, layer.head_dim),
    kv_cache.view(-1, 1, 1, layer.head_dim),
    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
    metadata.cu_seqlens_q,
    kv_indptr,
    kv_indices,            # <-- the sparse top-K page list
    metadata.cu_seqlens_q,
    metadata.max_seq_len_q,
    sm_scale=layer.scaling,
    logit_cap=layer.logit_cap,
)
```

Note that `kv_cache` is still the **full** paged buffer — sparsity is not
enforced by what's stored, only by which page IDs are referenced.

## Layer 4 — aiter Python wrapper passes through (`aiter/mla.py`)

`mla_decode_fwd` receives the sparse `kv_indices` and forwards it to the ASM
kernel binding as the `kv_page_indices` argument:

```python
# aiter/mla.py, mla_decode_fwd, lines 254-275
        aiter.mla_decode_stage1_asm_fwd(
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            num_kv_splits_indptr,
            None,
            None,
            None,
            max_seqlen_q,
            page_size,
            nhead_kv,
            sm_scale,
            logits,
            attn_lse,
            o,
            final_lse,
            q_scale,
            kv_scale,
        )
```

## Layer 5 — Python-to-C ABI binding (`aiter/ops/attention.py`)

The signature documents the contract: `kv_page_indices` is a flat 1-D list of
length `num_page_used` — i.e., the count of pages the kernel will attend to
(top-K, not L):

```python
# aiter/ops/attention.py, lines 655-670
@compile_ops(MD_NAME, ffi_type="ctypes")
def mla_decode_stage1_asm_fwd(
    # [num_seqs, num_heads, head_size]
    Q: torch.Tensor,
    # [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    KV: torch.Tensor,
    # [batch_size+1]
    qo_indptr: torch.Tensor,
    # [batch_size+1]
    kv_indptr: torch.Tensor,
    # [num_page_used]
    kv_page_indices: torch.Tensor,
    # [batch_size]
    kv_last_page_lens: torch.Tensor,
```

If the indexer selected 2048 pages per query, `num_page_used` is roughly
`batch * 2048`. If you ran the same kernel without DSA, `num_page_used` would be
roughly `batch * (seq_len / page_size)`.

## Layer 6 — C++ shim wires the pointer into the ASM kernel arg struct (`csrc/py_itfs_cu/asm_mla.cu`)

The HIP shim takes the `kv_page_indices` tensor and stuffs its raw pointer into
`args.ptr_LTD` (note: `LTD` = "list of pages, indices"):

```cpp
// csrc/py_itfs_cu/asm_mla.cu, mla_decode_stage1_asm_fwd, lines 120-151
{
    int batch           = qo_indptr->size(0) - 1;
    int num_heads       = Q->size(1);
    int head_size       = Q->size(2);
    int num_kv_heads    = nhead_kv;
    int kv_split        = splitData->size(1);
    const int gqa_ratio = num_heads / num_kv_heads;

    bool persistent = (num_kv_splits_indptr == nullptr);

    const HipDeviceGuard device_guard(Q->device_id);

    int stride_Q       = Q->stride(0) * Q->element_size() * max_seqlen_q;
    int stride_Page    = KV->stride(0) * KV->element_size();
    uint32_t log2_page = (uint32_t)log2f(page_size);

    KernelArgs args = {};
    size_t arg_size  = sizeof(args);
    args.ptr_R       = splitData->data_ptr();
    args.ptr_LSE     = splitLse->data_ptr();
    args.ptr_Q       = Q->data_ptr();
    args.ptr_KV      = KV->data_ptr();
    args.ptr_LTP     = kv_indptr->data_ptr();
    args.ptr_LTD     = kv_page_indices->data_ptr();
    args.ptr_LTL     = kv_last_page_lens->data_ptr();
    args.ptr_QTP     = qo_indptr->data_ptr();
    args.scalar      = softmax_scale;
```

The three "list" pointers in the kernel ABI are:

- `ptr_LTP` = `kv_indptr` — for each batch, where in `LTD` its page IDs start.
- `ptr_LTD` = `kv_page_indices` — the flat top-K page-ID list.
**This is the sparsity payload.**
- `ptr_LTL` = `kv_last_page_lens` — per-batch length of the last page (handles
non-multiples of `page_size`).

The kernel selection is also gated to decode:

```cpp
// csrc/py_itfs_cu/asm_mla.cu, lines 260-267
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    int ps = persistent ? 1 : 0;
    int prefill = 0; // decode stage
    int causal = 0;
    int config_max_seqlen_q = max_seqlen_q;
    int config_gqa_ratio = gqa_ratio;
    int sub_Q = 128; // default value
```

## Layer 7 — ASM kernel iterates the page list

The hand-written assembly kernel walks `ptr_LTD` from `ptr_LTP[b]` to
`ptr_LTP[b+1]` and, for each page ID `p`, loads the KV slice at
`ptr_KV + p * stride_Page`. It never touches pages whose IDs aren't in the list.
So if the indexer wrote only 2048 IDs, only 2048 pages are read, not the full
sequence.

This is the only place where "sparsity" physically reduces work — by not
loading those KV bytes from HBM and not doing the corresponding QK / PV math.

## Single-picture summary

```
[lightning indexer per layer]
        |
        v
  page_table_1 [batch, max_top_k]  (-1 padded)
        |   (SGLang nsa_backend._forward_aiter)
        v
  kv_indices  [num_page_used]      (flat top-K page IDs)
  kv_indptr   [batch+1]            (CSR boundaries)
        |   (aiter.mla.mla_decode_fwd)
        v
  passes both unchanged into:
        |
        v
  mla_decode_stage1_asm_fwd(...)   (aiter/ops/attention.py + asm_mla.cu)
        |
        v
  KernelArgs.ptr_LTD = kv_indices  (csrc/py_itfs_cu/asm_mla.cu:143)
  KernelArgs.ptr_LTP = kv_indptr   (csrc/py_itfs_cu/asm_mla.cu:142)
        |
        v
  ASM kernel iterates only the page IDs in ptr_LTD,
  loading only those pages from ptr_KV
        => Q · K, softmax, P · V over the top-K subset
```

The sparsity is realized *entirely* through the contents of one pointer
(`ptr_LTD`). The ASM kernel itself is identical to the dense-MLA decode kernel —
sparsity is upstream-only, and from the kernel's point of view the only
difference is "the index list happens to be shorter."

## Reference: file roles


| File                                         | Role                                                                                    |
| -------------------------------------------- | --------------------------------------------------------------------------------------- |
| `sglang/srt/layers/attention/nsa_backend.py` | Builds `kv_indices` from `page_table_1`; dispatches to `aiter.mla.mla_decode_fwd`       |
| `aiter/mla.py`                               | Python wrapper `mla_decode_fwd`; passes `kv_indices` through to the ASM stage-1 binding |
| `aiter/ops/attention.py`                     | Python `@compile_ops` declaration of `mla_decode_stage1_asm_fwd` (the C ABI)            |
| `csrc/py_itfs_cu/asm_mla.cu`                 | HIP shim that fills `KernelArgs.ptr_LTD = kv_page_indices` and launches the ASM kernel  |


