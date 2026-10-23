"""A1 LLM smoke check (M0): mlx_lm.server 0.32.2 with a Qwen3.x-35B-A3B-4bit build, thinking off.

Correctness only; timings are single observations, not benchmarks. The script starts

    uv run --project spikes/llm mlx_lm.server --model <repo> --host 127.0.0.1 --port 8101 \
        --chat-template-args '{"enable_thinking":false}'

waits for /health, then sends: a non-streaming zh question, a streaming en question (its raw SSE is
kept), a streaming zh question, and a warm repeat of the first request (to separate load time from
request time). With --quick it sends only the non-streaming zh question. No request carries
chat_template_kwargs, so only the server-wide flag keeps thinking off. Each answer is checked for
text, for no `reasoning` field and for no <think> markup. The server is stopped at the end.

Run from the repository root (TMPDIR receives the kept server log):

    TMPDIR=<scratch> uv run --project spikes/llm python spikes/llm/smoke_llm.py \
        --model mlx-community/Qwen3.6-35B-A3B-4bit
    TMPDIR=<scratch> uv run --project spikes/llm python spikes/llm/smoke_llm.py \
        --model mlx-community/Qwen3.5-35B-A3B-4bit --quick

Writes spikes/results/a1_llm_smoke_<repo name>.json; the full run also writes the first 40 raw SSE lines
of the en streaming request to spikes/results/a1_llm_smoke.txt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import constants

HOST, PORT = "127.0.0.1", 8101
SPIKE_DIR = Path(__file__).resolve().parent
RESULTS = SPIKE_DIR.parent / "results"
CHAT_TEMPLATE_ARGS = '{"enable_thinking":false}'
MAX_TOKENS = 200

Q_ZH = "请用一两句话介绍一下杭州。"
Q_EN = "In one or two sentences: what is the capital of France, and what is it famous for?"
Q_ZH_SERVICE = "我想退货，一般需要哪些步骤？请简要回答。"
THINK = re.compile(r"</?think>")


def local_sha(repo: str) -> str | None:
    ref = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    return ref.read_text().strip() if ref.is_file() else None


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def get(path: str) -> tuple[int, str]:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    conn.request("GET", path)
    resp = conn.getresponse()
    return resp.status, resp.read().decode("utf-8", "replace")


def footprint_bytes() -> int | None:
    out = subprocess.run(["lsof", "-t", f"-iTCP:{PORT}", "-sTCP:LISTEN"], capture_output=True, text=True)
    pids = out.stdout.split()
    if not pids:
        return None
    out = subprocess.run(["footprint", "-p", pids[0], "--noCategories", "-f", "bytes"],
                         capture_output=True, text=True)
    m = re.search(r"Footprint:\s*(\d+)\s*B", out.stdout)
    return int(m.group(1)) if m else None


def chat(model: str, question: str, stream: bool) -> dict:
    body: dict = {
        "model": model,  # the exact --model string (any other value makes the server look it up and swap)
        "messages": [{"role": "user", "content": question}],
        "max_tokens": MAX_TOKENS,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=900)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body, ensure_ascii=False).encode(),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    out: dict = {"request": body, "status": resp.status, "content_type": resp.getheader("content-type"),
                 "ms_to_headers": round((time.perf_counter() - t0) * 1000, 1)}
    if resp.status != 200:
        out["error_body"] = resp.read().decode("utf-8", "replace")[:2000]
        out |= {"content": "", "reasoning": None, "finish_reason": None, "usage": None,
                "ms_total": round((time.perf_counter() - t0) * 1000, 1)}
    elif not stream:
        payload = json.loads(resp.read())
        out["ms_total"] = round((time.perf_counter() - t0) * 1000, 1)
        message = payload["choices"][0]["message"]
        out |= {
            "response": payload,
            "content": message.get("content") or "",
            "reasoning": message.get("reasoning"),
            "finish_reason": payload["choices"][0].get("finish_reason"),
            "usage": payload.get("usage"),
        }
    else:
        raw_lines: list[str] = []
        content, reasoning, keepalives, usage, finish, done = "", "", 0, None, None, False
        first_content_ms = None
        while line := resp.readline():
            text = line.decode("utf-8", "replace").rstrip("\r\n")
            raw_lines.append(text)
            if text.startswith(":"):
                keepalives += 1
            elif text.startswith("data: "):
                data = text[len("data: "):]
                if data == "[DONE]":
                    done = True
                    continue
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        if first_content_ms is None:
                            first_content_ms = round((time.perf_counter() - t0) * 1000, 1)
                        content += delta["content"]
                    if delta.get("reasoning"):
                        reasoning += delta["reasoning"]
                    finish = choice.get("finish_reason") or finish
        out |= {
            "ms_to_first_content": first_content_ms,
            "ms_total": round((time.perf_counter() - t0) * 1000, 1),
            "raw_sse_lines": raw_lines,
            "content": content,
            "reasoning": reasoning or None,
            "finish_reason": finish,
            "usage": usage,
            "keepalive_comment_lines": keepalives,
            "done_sentinel": done,
            "usage_chunk_before_done": usage is not None and done,
        }
    conn.close()
    out["checks"] = {
        "http_200": out["status"] == 200,
        "has_text": bool(out["content"].strip()),
        "no_reasoning_field": not out["reasoning"],
        "no_think_markup": not THINK.search(out["content"]),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--quick", action="store_true", help="load + one non-streaming request only")
    args = parser.parse_args()
    if port_open():
        raise SystemExit(f"port {PORT} is already in use")

    name = args.model.split("/")[-1]
    result: dict = {
        "check": "A1 LLM smoke: mlx_lm.server, thinking off server-wide (correctness only, not a benchmark)",
        "started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": args.model,
        "model_sha_local": local_sha(args.model),
        "server_cmd": f"uv run --project spikes/llm mlx_lm.server --model {args.model} --host {HOST} "
        f"--port {PORT} --chat-template-args '{CHAT_TEMPLATE_ARGS}'",
        "server_env": {"HF_HUB_OFFLINE": "1"},
        "iogpu.wired_limit_mb": subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                                               capture_output=True, text=True).stdout.strip(),
        "mx_device_info": subprocess.run(
            ["uv", "run", "--project", str(SPIKE_DIR), "python", "-c",
             "import json, mlx.core as mx; print(json.dumps(mx.device_info()))"],
            capture_output=True, text=True).stdout.strip(),
        "requests": {},
    }
    reqs = result["requests"]

    def run_step(key: str, question: str, stream: bool) -> None:
        try:
            reqs[key] = chat(args.model, question, stream)
        except Exception as exc:  # noqa: BLE001 - a smoke check records failures
            reqs[key] = {"failed": f"{type(exc).__name__}: {exc}", "checks": {"completed": False},
                         "content": "", "finish_reason": None, "ms_total": None}

    cmd = ["uv", "run", "--project", str(SPIKE_DIR), "mlx_lm.server", "--model", args.model,
           "--host", HOST, "--port", str(PORT), "--chat-template-args", CHAT_TEMPLATE_ARGS]
    log_path = Path(tempfile.mkdtemp(prefix="a1_mlx_lm_")) / "server.out"  # kept for inspection
    result["server_log_file"] = str(log_path)
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log_path.open("w"), stderr=subprocess.STDOUT,
                            env=dict(os.environ, HF_HUB_OFFLINE="1"), start_new_session=True)
    try:
        while True:
            if proc.poll() is not None:
                raise SystemExit(f"server exited early:\n{log_path.read_text()[-3000:]}")
            try:
                if get("/health")[0] == 200:
                    break
            except OSError:
                pass
            if time.perf_counter() - t0 > 300:
                raise SystemExit("no /health within 300 s")
            time.sleep(0.5)
        # /health only says the generation thread is alive; the default model loads inside that
        # thread, so the first answer below includes the rest of the load.
        result["seconds_to_health_200"] = round(time.perf_counter() - t0, 1)
        run_step("zh_non_stream", Q_ZH, stream=False)
        result["seconds_start_to_first_answer"] = round(time.perf_counter() - t0, 1)
        if not args.quick:
            run_step("en_stream", Q_EN, stream=True)
            run_step("zh_stream", Q_ZH_SERVICE, stream=True)
            run_step("zh_non_stream_warm_repeat", Q_ZH, stream=False)
        result["footprint_bytes_after_requests"] = footprint_bytes()
        result["health_after"] = get("/health")
        status, body = get("/v1/models")
        result["v1_models_after"] = {"status": status, "body": body[:2000]}
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        for _ in range(40):
            if not port_open():
                break
            time.sleep(0.25)
        result["server_stop"] = {"returncode": proc.returncode, "port_free_after": not port_open()}
        lines = log_path.read_text(errors="replace").splitlines()
        result["server_log_line_count"] = len(lines)
        result["server_log_tail"] = [ln for ln in lines[-60:] if "token" not in ln.lower()]
        result["server_log_warnings"] = [
            ln for ln in lines if re.search(r"warn|error|exception|traceback", ln, re.I)
        ][:40]

    result["all_checks_pass"] = bool(reqs) and all(all(r["checks"].values()) for r in reqs.values())
    result["finished_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    RESULTS.mkdir(parents=True, exist_ok=True)
    if reqs.get("en_stream", {}).get("raw_sse_lines"):
        r = reqs["en_stream"]
        header = [
            f"# A1 LLM smoke: raw SSE of the streaming en request, first 40 lines ({len(r['raw_sse_lines'])} total)",
            f"# model {args.model} @ {result['model_sha_local']}",
            f"# server: {result['server_cmd']}",
            f"# request: POST /v1/chat/completions {json.dumps(r['request'], ensure_ascii=False)}",
            "# ----- raw SSE lines follow -----",
        ]
        (RESULTS / "a1_llm_smoke.txt").write_text("\n".join(header + r["raw_sse_lines"][:40]) + "\n")
    out = RESULTS / f"a1_llm_smoke_{name}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for key, r in reqs.items():
        print(f"{key}: {r['checks']} finish={r['finish_reason']} ms_total={r['ms_total']} {r.get('failed', '')}")
        print(f"   {r['content'][:300]!r}")
    print({k: result.get(k) for k in ("seconds_to_health_200", "seconds_start_to_first_answer",
                                      "footprint_bytes_after_requests", "all_checks_pass", "server_stop")})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
