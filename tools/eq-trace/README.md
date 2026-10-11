# llama-eq-trace

Teacher-forced router tracing for Mixture-of-Experts GGUF models. Runs the
prompt tokens through the model (no generation) and records, for every token
position and every MoE layer, the router logits, the selected experts and
(sampled) the router input hidden state.

```
llama-eq-trace -m model.gguf --prompts prompts.jsonl --out PREFIX
               [--max-tokens N] [--hidden-sample F] [--seed S]
               [-ngl ...] [-ot ...] [other common llama.cpp args]
```

- `--prompts`: JSONL, one object per line: `{"prompt_id": <u32>, "text": "<string>"}`
  (other keys ignored, blank lines skipped). Repeated `prompt_id`, malformed
  lines or a missing/mistyped field abort before the model runs.
- `--out PREFIX`: writes `PREFIX.routes.bin` and `PREFIX.hidden.bin`.
- `--max-tokens N` (default 512): prompts are truncated to N tokens; the
  context size is N.
- `--hidden-sample F` (default 0.1), `--seed S` (default 0): see "Sampling".
- All other arguments go to the common llama.cpp parser (`-ngl`, `-ot`, `-b`,
  `-ub`, `-t`, ...). `-c` and `-np` are overridden.

Tokenization: `add_special=true, parse_special=false`. One prompt per decode,
split into `n_batch` chunks (`token_pos` stays global within the prompt); the
memory is cleared between prompts. Every position is an output, so the last
layer is not pruned.

Exit code is nonzero with a message on stderr if: the model has no experts
(`<arch>.expert_count` missing or 0), the prompts file is malformed or repeats
a `prompt_id`, a prompt tokenizes to nothing, an expected tensor never showed
up for a layer or token, the set of MoE layers changes between prompts, or a
write fails. At the end a summary line (prompts, tokens, records, tokens/s of
the decode loop) goes to stderr.

## Captured graph tensors

Per MoE layer `<il>` (the model's block index):

| tensor | content |
| --- | --- |
| `ffn_moe_logits-<il>` | raw router logits, f32 `[n_expert, n_tokens]` (`gate_inp * ffn_norm`), before any bias/softmax/mask |
| `ffn_moe_topk-<il>` | selected expert ids, i32 `[n_expert_used, n_tokens]`, in model order |
| `ffn_norm-<il>` | router input, f32 `[n_embd, n_tokens]`: RMS-normed `ffn_inp`, the tensor passed to `build_moe_ffn` |

Layers are discovered from the tensors seen (dense layers of mixed models only
have `ffn_norm-<il>` and are ignored).

## File format

Little-endian, no padding. Two files, each a 32 byte header followed by fixed
size records. Codec: `eq-trace-format.h`.

Header (both files):

| offset | type | field |
| --- | --- | --- |
| 0 | char[4] | magic: `EQTR` (routes), `EQTH` (hidden) |
| 4 | u32 | version = 1 |
| 8 | u32 | n_layer: number of MoE layers traced |
| 12 | u32 | n_expert |
| 16 | u32 | top_k (n_expert_used) |
| 20 | u32 | n_embd |
| 24 | u64 | n_records, patched at close (header 0 but data present = run was cut) |

`PREFIX.routes.bin` record: `u32 prompt_id, u32 token_pos, u16 layer,
u16 expert_ids[top_k], f32 logits[n_expert]`; size `10 + 2*top_k + 4*n_expert`.
`expert_ids` are in the order the model selected them. One record per
(prompt, token_pos, layer).

`PREFIX.hidden.bin` record: `u32 prompt_id, u32 token_pos, u16 layer,
f16 hidden[n_embd]` (IEEE binary16, round to nearest even, from f32); size
`10 + 2*n_embd`.

`layer` is the model's block index. Records are grouped by prompt in input
order and ordered by (token_pos, layer) within a prompt, over the MoE layers
only.

## Sampling

Hidden states are sampled per token: a (prompt_id, token_pos) pair is recorded
for all MoE layers or for none. With `splitmix64` as

```
splitmix64(x):                      # all arithmetic mod 2^64
    x += 0x9E3779B97F4A7C15
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)
```

the token is recorded iff

```
h = splitmix64(seed)
h = splitmix64(h ^ prompt_id)
h = splitmix64(h ^ token_pos)
(h >> 11) * 2^-53  <  F
```

so F=1 records everything, F=0 nothing (header-only hidden file), and reruns
are byte-identical for the same model/placement. Routes are never sampled.

## Notes

Router logits depend slightly on numerics (backend, `-ub`, chunking), so
near-tied experts can flip between runs with different placement or batch
settings; keep `-ub`/`-b` fixed within one study.
