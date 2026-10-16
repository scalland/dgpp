# Batch-size variance in greedy decoding

Status: accepted (decision recorded 2026-10-16). Reference for future work.

## Finding

Greedy outputs for a given prompt can depend on how many other requests are in
flight. This is not new to the decode-row change (A3); the build before it shows
the same behaviour.

Measured on the 2-node GB10 TP=2 deployment, MTP depth 2, FP8_e4m3 KV, 31 seats,
with 10 fixed prompts, temperature 0, 256 output tokens:

- The pre-change build and the decode-row build both reproduce their own replies
  with a single user and with three concurrent users.
- At 31 concurrent requests, the same three prompts (indices 2, 3 and 7) produce a
  different reply from their single-user reply. The other seven match.
- The pre-change build shows the same three prompts changing at 31 concurrent
  requests. The decode-row build changes the same three, not new ones.
- The divergence is a single word choice in each case, and both versions read as
  coherent text.

## Cause

Floating-point addition is not associative. The engine chooses kernels, tile
shapes and reduction orders by the number of rows in a step. With more requests
in a batch, the same token's logits are computed with a different summation order,
so they differ in the last few bits. Greedy decoding takes the arg-max token, so a
near-tie between the top two tokens can flip, after which the text follows the
other branch.

The code selects kernels by row count in several places, for example the MTP
hidden projection switches to the MMA path above 32 tokens (see the call in
`mtp_run_rows` in `src/models/qwen/forward.cpp`). This is consistent with the
measurements, but the specific kernel responsible for each diverging prompt has
not been profiled.

## Consequences

- Greedy output is not guaranteed identical across batch sizes. Only single-request
  behaviour is stable against the pre-change build.
- Sampling (temperature above 0) is not affected in distribution. The variance is
  one realisation of the same arithmetic noise that sampling already randomises.
- Bit-exact regression tests must compare at a fixed concurrency. Comparing a
  1-request reference against a multi-request run will show differences.

## Decision

Accepted for the decode-row change. Batch-invariant decoding is not a requirement
of this deployment. If it becomes one, it needs a deterministic reduction order
for every kernel on the decode path, which is a separate, larger piece of work.
