"""Replace _forward_aiter (lines starting at "    def _forward_aiter(" up to
"    def _forward_aiter_extend(") in nsa_backend.py with the new body in
new_forward_aiter.py.

This is Run F's version: identical to Run-D/E's apply_patch.py but the new
body adds a third env-gated branch for SGLANG_NSA_USE_UA_SPARSE_MLA.
"""
import sys, pathlib

target = pathlib.Path(sys.argv[1])
new_body = pathlib.Path(sys.argv[2]).read_text()

text = target.read_text()
lines = text.splitlines(keepends=True)

start = end = None
for i, line in enumerate(lines):
    if start is None and line.startswith("    def _forward_aiter(") and "_extend" not in line:
        start = i
    elif start is not None and line.startswith("    def _forward_aiter_extend("):
        end = i
        break
if start is None or end is None:
    print(f"FATAL: could not locate _forward_aiter span (start={start}, end={end})", file=sys.stderr)
    sys.exit(1)

print(f"Replacing lines {start+1}..{end} (1-indexed) of {target}")
print(f"  old span: {end - start} lines")
print(f"  new span: {len(new_body.splitlines())} lines")

new_text = new_body
if not new_text.endswith("\n"):
    new_text += "\n"
new_text += "\n"

out = "".join(lines[:start]) + new_text + "".join(lines[end:])
target.write_text(out)
print("OK")
