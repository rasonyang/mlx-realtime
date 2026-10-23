"""Offline check (no GPU): would a lenient variant of mlx-lm's qwen3_coder parser recover the calls that
Qwen3.6 emitted as `<tool_call>\n<NAME>\n<parameter=...>...</function>` (missing `function=`)?

Reads spikes/results/s1_toolparse_qwen3.6.json (dropped_raw_from_log), rewrites a bare `<NAME>` opening
tag to `<function=NAME>` when NAME is a declared tool, and runs the stock parser on the result.

    uv run --project spikes/llm python spikes/llm/s1_toolparse_lenient.py
"""

import json
import re

from mlx_lm.tool_parsers import qwen3_coder

import llmlib as L

tools = L.aicc()["tools"]
names = {t["function"]["name"] for t in tools}
d = json.loads((L.RESULTS / "s1_toolparse_qwen3.6.json").read_text())
out = []
for raw in d["dropped_raw_from_log"]:
    body = raw.split("<tool_call>", 1)[-1].split("</tool_call>", 1)[0]
    try:
        qwen3_coder.parse_tool_call(body, tools)
        stock = "parsed"
    except ValueError as e:
        stock = f"ValueError: {e}"
    fixed = re.sub(r"^(\s*)<([A-Za-z_][\w.-]*)>", lambda m: f"{m.group(1)}<function={m.group(2)}>"
                   if m.group(2) in names else m.group(0), body, count=1)
    try:
        lenient = qwen3_coder.parse_tool_call(fixed, tools)
    except ValueError as e:
        lenient = f"ValueError: {e}"
    out.append({"stock": stock, "lenient": lenient})
    print(stock, "->", json.dumps(lenient, ensure_ascii=False)[:160])
d["lenient_check"] = out
(L.RESULTS / "s1_toolparse_qwen3.6.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
