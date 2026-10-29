"""Measurement of the hold (barge_in.confirm_while_playing, issue #5) on the simulated AICC call.

    uv run --locked python spikes/live_echo/bursts.py run [--only A,B,Be,E0,C,D] [--smoke]
    uv run --locked python spikes/live_echo/bursts.py report

``run`` starts the ASR and TTS backends (config.yaml's, not the LLM) unless they already listen,
two gateways in this process (baseline: confirm_while_playing false; fix: true; the rest of
config.yaml as is, capture on), and runs the scenarios in real time, one chunk of trials at a time,
alternating the conditions chunk by chunk so that drift (thermal, load) hits both. Raw rows go to
captures/hold/rows/ (git-ignored; finished jobs are skipped on a rerun), then ``report`` writes
results/hold.json and results/hold.md.

Scenarios (the answers are verbatim lines, so the bot's speech is the real TTS; create_response is
off, so nothing but the harness starts an answer):
  A   residual-echo bursts (hold_lib's model) on a line with no steady echo: A1 the last words of an
      answer, A2 the first words of an answer after >= 8.5 s of silence (burst ends 0.45-0.9 s past
      the 2 s onset guard), A3 mid-answer (starts 4-12 s into a 25 s answer)
  B   real barge-ins: FLEURS zh / en and ASCEND mixed clips, 3-15 s into an answer, no echo line
  Be  the same on a line with steady linear echo (ERL 10 dB, 100 / 180 ms)
  E0  ordinary turns over silence (the commit -> transcript latency, no playback)
  C   short utterances over playback (synthesised: Qwen3-TTS voices uncle_fu / aiden, not the bot's)
  D   non-speech noise over playback (cup, keys, clicks)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from hold_backends import backends_running  # noqa: E402
from hold_lib import (  # noqa: E402
    HOLD,
    Plan,
    Rig,
    capture_items,
    cleanup,
    client_events,
    data_root,
    drain,
    load_events,
    normalise_active,
    onset_s,
    play_answer,
    read_wav,
    session_dir,
    synth_noise,
    wait_event,
    write_wav,
)

ROWS = HOLD / "rows"
SEED = 20261029

# Answers in the style of AICC's long Chinese bot flows (about 25 s of speech, 110-140 characters).
LONG = [
    "您好，这里是新网络客服中心。我先跟您说明一下目前的套餐情况：基础套餐每月九十九元，包含两百分钟通话和二十吉流量，超出部分按标准资费计算。如果您需要更多流量，可以升级到高级套餐，同时还能享受免费的来电显示和语音信箱服务，请问您想了解哪一种呢？",
    "关于您刚才提到的账单问题，我来详细解释一下。账单分为三个部分：第一部分是固定月租费，第二部分是超出套餐的使用费用，第三部分是各项税费。您可以在手机应用程序里查看每一笔明细，如果发现有不明扣费，我们会在三个工作日内为您核实并处理。",
    "为了保障您的账户安全，我们需要先确认几个信息。请您准备好注册时使用的手机号码、身份证后四位，以及最近一次充值的大概金额。验证通过以后，我就可以帮您办理换卡、改套餐或者暂停服务等业务，整个过程大约需要五分钟左右的时间。",
    "我们的宽带安装服务覆盖了市区的大部分小区。师傅上门之前会提前一个小时给您打电话，安装过程包括布线、调试路由器和测试网速三个步骤。安装完成后一周内如果出现任何网络不稳定的情况，都可以免费上门维修，不需要额外支付任何费用。",
    "感谢您的耐心等待。经过系统查询，您的订单目前处于配送途中，预计明天下午五点前送达。快递员会先打电话联系您，如果当时不方便接收，可以改约到后天或者选择放到附近的自提柜，取件码会通过短信发送到您预留的手机号码上。",
    "如果您想办理携号转网，需要满足几个条件：号码已经使用满一百二十天，账户没有欠费，也没有正在履行的合约。符合条件的话，只要发送短信申请授权码，再带上身份证到新的营业厅办理就可以，全程大约十分钟，原来的号码和余额都不会改变。",
    "今天给您介绍的是我们新推出的家庭共享套餐。一个主卡可以添加最多四张副卡，所有成员共用流量和通话时长，每个月只需要一百五十九元。另外还赠送一年的视频会员，到期后可以按原价续费，也可以随时取消，不会产生任何违约金。",
    "您反映的手机信号不好的问题，我们已经记录下来了。可能的原因包括基站维护、室内墙体遮挡或者手机本身的设置。建议您先重启手机，再检查一下飞行模式是否关闭。如果问题持续存在，我们会安排网络工程师上门检测，并在两个工作日内给您回复。",
]
SHORT = [
    "好的，我已经为您记录下来了，稍后会有专人跟进，请您保持电话畅通，谢谢。",
    "没问题，您的申请已经提交成功，处理结果会在一个工作日内通过短信通知您。",
    "请您稍等片刻，我正在为您查询账户信息，大概需要十几秒钟的时间，感谢您的耐心。",
    "已经为您办理完成，新的套餐从下个月一号开始生效，本月的费用不会受到任何影响。",
    "非常抱歉给您带来了不便，我们会尽快核实情况，并在今天下班之前给您一个明确的答复。",
    "您的预约已经确认，师傅会在明天上午九点到十点之间上门，请您提前准备好相关证件。",
]
WARMUP = "您好，欢迎致电新网络客服中心，我是您的智能助手，请问有什么可以帮您？"
SHORT_UTTERANCES = [  # (text, voice, lang)
    ("喂", "uncle_fu", "zh"), ("嗯", "uncle_fu", "zh"), ("对", "uncle_fu", "zh"),
    ("好的", "uncle_fu", "zh"), ("等一下", "uncle_fu", "zh"), ("不是", "uncle_fu", "zh"),
    ("可以", "uncle_fu", "zh"), ("OK", "aiden", "en"), ("wait", "aiden", "en"),
    ("hello", "aiden", "en"), ("hold on", "aiden", "en"),
]


# Plans


def plan_a(rng: random.Random, reps: int, *, profile: str = "default", name: str = "A") -> list[dict[str, Any]]:
    trials = []
    for kind in ("A1", "A2", "A3"):
        for delay in (0.316, 0.506):
            for level in (-5.0, -20.0, -35.0):
                for _ in range(reps):
                    trials.append(
                        {
                            "scenario": name, "profile": profile, "kind": kind, "delay_s": delay, "level_db": level,
                            "length_s": round(rng.uniform(1.5, 2.5), 2),
                            "seed": rng.randrange(1 << 30),
                            "offset_s": round(rng.uniform(4.0, 12.0), 2),
                            "past_guard_s": round(rng.uniform(0.45, 0.9), 2),
                            "text": (rng.choice(SHORT) if kind == "A1" else rng.choice(LONG)),
                            "gap_s": 8.5 if kind == "A2" else 3.0,
                        }
                    )
    rng.shuffle(trials)
    for index, trial in enumerate(trials):
        trial["id"] = f"{name}{index:02d}"
    return trials


def dataset_clips() -> dict[str, list[dict[str, Any]]]:
    root = data_root()
    rows = [json.loads(line) for line in (HERE.parent / "datasets" / "manifest.jsonl").read_text().splitlines()]
    out: dict[str, list[dict[str, Any]]] = {"zh": [], "en": [], "mix": []}
    for row in rows:
        out[row["set"]].append({"id": row["id"], "set": row["set"], "text": row["text"],
                                "path": str(root / row["path_16k"]), "duration_s": row["duration_s"]})
    return out


def plan_b(rng: random.Random, per_set: int, prefix: str, *, offsets: bool = True) -> list[dict[str, Any]]:
    clips = dataset_clips()
    trials = []
    for name in ("zh", "en", "mix"):
        for clip in rng.sample(clips[name], per_set):
            trials.append(
                {"scenario": prefix, "clip": clip, "text": rng.choice(LONG), "gap_s": 1.5,
                 "offset_s": round(rng.uniform(3.0, 15.0), 2) if offsets else 0.0}
            )
    rng.shuffle(trials)
    for index, trial in enumerate(trials):
        trial["id"] = f"{prefix}{index:02d}"
    return trials


def plan_c(rng: random.Random, reps: int) -> list[dict[str, Any]]:
    trials = []
    for text, voice, lang in SHORT_UTTERANCES:
        for _ in range(reps):
            trials.append({"scenario": "C", "utterance": text, "voice": voice, "lang": lang,
                           "text": rng.choice(LONG), "gap_s": 1.5, "offset_s": round(rng.uniform(3.0, 12.0), 2)})
    rng.shuffle(trials)
    for index, trial in enumerate(trials):
        trial["id"] = f"C{index:02d}"
    return trials


def plan_d(rng: random.Random, reps: int) -> list[dict[str, Any]]:
    trials = []
    for kind in ("cup", "keys", "clicks"):
        for _ in range(reps):
            trials.append({"scenario": "D", "noise": kind, "peak_dbfs": rng.choice([-3.0, -12.0, -24.0]),
                           "seed": rng.randrange(1 << 30), "text": rng.choice(LONG), "gap_s": 1.5,
                           "offset_s": round(rng.uniform(3.0, 12.0), 2)})
    rng.shuffle(trials)
    for index, trial in enumerate(trials):
        trial["id"] = f"D{index:02d}"
    return trials


def plan_f(rng: random.Random, reps: int) -> list[dict[str, Any]]:
    """Double talk: a caller (B's clips) speaks while a leak burst is on the line. F1: the caller's
    speech starts with the burst; F2: 0.5-1 s into it, so both fall in one VAD turn."""
    clips = dataset_clips()
    trials = []
    for kind in ("F1", "F2"):
        for level in (-5.0, -20.0):
            for delay in (0.316, 0.506):
                for _ in range(reps):
                    name = rng.choice(["zh", "en", "mix"])
                    trials.append(
                        {"scenario": "F", "kind": kind, "level_db": level, "delay_s": delay,
                         "length_s": round(rng.uniform(2.0, 2.5), 2), "seed": rng.randrange(1 << 30),
                         "offset_s": round(rng.uniform(4.0, 10.0), 2),
                         "shift_s": 0.0 if kind == "F1" else round(rng.uniform(0.5, 1.0), 2),
                         "clip": rng.choice(clips[name]), "text": rng.choice(LONG), "gap_s": 1.5}
                    )
    rng.shuffle(trials)
    for index, trial in enumerate(trials):
        trial["id"] = f"F{index:02d}"
    return trials


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_jobs(only: set[str], smoke: bool, conditions: tuple[str, ...] = ("baseline", "fix")) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    plans: list[tuple[str, list[list[dict[str, Any]]], dict[str, Any]]] = []
    if "A" in only:
        trials = plan_a(rng, 1 if smoke else 2)
        plans.append(("A", chunks(trials[:3] if smoke else trials, 12), {}))
    if "Am" in only:
        trials = plan_a(random.Random(SEED + 1), 1, profile="mild", name="Am")  # 18 per condition: time
        plans.append(("Am", chunks(trials[:3] if smoke else trials, 9), {}))
    # Every plan draws from the one rng in this order whatever ``only`` says, so a subset of
    # scenarios gets the trials of the full run.
    trials = plan_b(rng, 1 if smoke else 10, "B")
    if "B" in only:
        plans.append(("B", chunks(trials, 15), {}))
    trials = plan_b(rng, 1 if smoke else 4, "Be")[: 3 if smoke else 10]
    if "Be" in only:
        plans.append(("Be", chunks(trials, 5 if not smoke else 3), {"echo_db": 10.0}))
    trials = plan_b(rng, 1 if smoke else 3, "E0", offsets=False)[: 2 if smoke else 8]
    if "E0" in only:
        plans.append(("E0", [trials], {}))
    trials = plan_c(rng, 1 if smoke else 2)
    if "C" in only:
        plans.append(("C", chunks(trials[:3] if smoke else trials, 11), {}))
    trials = plan_d(rng, 1 if smoke else 5)
    if "D" in only:
        plans.append(("D", [trials[:3] if smoke else trials], {}))
    if "F" in only:
        trials = plan_f(random.Random(SEED + 2), 1 if smoke else 3)
        plans.append(("F", chunks(trials, 12), {}))
    jobs = []
    for name, groups, extra in plans:
        for index, group in enumerate(groups):
            order = ("baseline", "fix") if index % 2 == 0 else ("fix", "baseline")
            kwargs = dict(extra)
            if name == "Be":
                kwargs["delay_s"] = (0.1, 0.18)[index % 2]
            for condition in order:
                if condition not in conditions:
                    continue
                jobs.append({"name": f"{name}-{index}-{condition}", "scenario": name,
                             "condition": condition, "trials": group, "session": kwargs})
    return jobs


# Trials


async def settle_until(call: Any, n: int) -> None:
    async with asyncio.timeout(120.0):
        while call.n < n:  # noqa: ASYNC110
            await asyncio.sleep(0.05)


def client_summary(call: Any, since: int, lo_t: float, hi_t: float) -> dict[str, Any]:
    events = client_events(call, since, {"input_audio_buffer.speech_started", "response.done"})
    return {
        "speech_started": [{"t": e["t"], "item": e["item_id"], "audio_start_ms": e["audio_start_ms"]}
                           for e in events if e["type"] == "input_audio_buffer.speech_started"],
        "done": [{"t": e["t"], "status": e["status"]} for e in events if e["type"] == "response.done"],
        "ss_snap": [s for s in call.ss_snap if lo_t <= s["t"] <= hi_t],
        "flushes": [f for f in call.flushes if lo_t <= f["t"] <= hi_t and not f["cleanup"]],
    }


async def trial_a(call: Any, client: Any, spec: dict[str, Any]) -> dict[str, Any]:
    plan = Plan(spec["kind"], spec["delay_s"], spec["level_db"], spec["length_s"], spec["seed"],
                spec["offset_s"], spec["past_guard_s"], spec.get("profile", "default"))
    await asyncio.sleep(spec["gap_s"])
    bursts = len(call.bursts)
    if plan.kind != "A1":
        call.plan_at_start(plan)
    since, t_start = await play_answer(call, client, spec["text"])
    if plan.kind == "A1":
        call.plan_at_end(plan)
        async with asyncio.timeout(60.0):
            while len(call.bursts) == bursts:  # noqa: ASYNC110
                await asyncio.sleep(0.05)
    burst = call.bursts[-1]
    await settle_until(call, burst.end_frame + round(4.5 / 0.02))
    t0 = call.tick_time(burst.first_frame)
    t1 = call.tick_time(min(burst.end_frame, call.n - 1))
    ends = [e for e in client_events(call, since, {"response.done"})]
    answer_bytes = sum(len(e.get("delta", "")) for _, e in call.events[since:] if e["type"] == "response.output_audio.delta")
    row = {
        "burst": {"first_n": burst.first_frame, "end_n": burst.end_frame,
                  "start_ms": burst.first_frame * 20, "end_ms": burst.end_frame * 20,
                  "t0": t0, "t1": t1, "realised_db": burst.realised_db(),
                  "playback_pos_s": burst.playback_pos_s, "to_end_s": burst.to_end_s,
                  "playback_start_ms": call.start_n[-1] * 20 if call.start_n else None},
        "answer_s": answer_bytes * 3 / 4 / 48000,
        "done_before_burst": any(e["t"] < t0 for e in ends),
        "client": client_summary(call, since, t0 - 0.5, t1 + 4.6),
        "window_ms": [burst.first_frame * 20 - 500, burst.end_frame * 20 + 4600],
    }
    await cleanup(call, client)
    return row


async def trial_inject(call: Any, client: Any, spec: dict[str, Any], pcm: bytes, rig: Any,
                       *, playing: bool) -> dict[str, Any]:
    await asyncio.sleep(spec["gap_s"])
    since = len(call.events)
    t_play = None
    onset = onset_s(pcm)
    burst_info = None
    if "kind" in spec and spec["scenario"] == "F":  # a leak burst the caller talks into
        call.plan_at_start(Plan("A3", spec["delay_s"], spec["level_db"], spec["length_s"], spec["seed"], spec["offset_s"]))
        since, t_play = await play_answer(call, client, spec["text"])
        burst = call.bursts[-1]
        target = burst.first_frame + round(spec["shift_s"] / 0.02) - round(onset / 0.02)
        async with asyncio.timeout(60.0):
            while call.n < target - 1:  # noqa: ASYNC110
                await asyncio.sleep(0.004)
        burst_info = {"first_n": burst.first_frame, "end_n": burst.end_frame, "start_ms": burst.first_frame * 20,
                      "end_ms": burst.end_frame * 20, "realised_db": None}
        since = len(call.events)
    elif playing:
        since, t_play = await play_answer(call, client, spec["text"])
        await asyncio.sleep(max(t_play + spec["offset_s"] - asyncio.get_running_loop().time(), 0.0))
        since = len(call.events)  # events from here
    duration = len(pcm) / 32000
    first = call.inject(pcm)
    await settle_until(call, first + 2)
    t_inj = call.tick_time(first)
    t_onset = t_inj + onset
    ss = await wait_event(call, "input_audio_buffer.speech_started", since, duration + 6.0)
    flush_t = None
    if ss is not None:
        deadline = ss[0] + 2.5
        while asyncio.get_running_loop().time() < deadline:
            fl = [f for f in call.flushes if f["t"] >= t_inj and not f["cleanup"]]
            if fl:
                flush_t = fl[0]["t"]
                break
            await asyncio.sleep(0.02)
    item = None if ss is None else ss[1]["item_id"]
    stopped = await wait_event(call, "input_audio_buffer.speech_stopped", since, duration + 8.0, item=item) if ss else None
    transcript = await wait_event(call, "conversation.item.input_audio_transcription.completed", since,
                                  duration + 8.0, item=item) if ss else None
    # Let a held turn that is never announced resolve (its drop has no client event).
    await settle_until(call, first + round((onset + duration + 5.0) / 0.02))
    end_n = first + round(duration / 0.02)
    row = {
        "onset_s": onset, "duration_s": duration, "n_inject": first, "t_inject": t_inj, "t_onset": t_onset,
        "ss": None if ss is None else {"t": ss[0], "dt_ms": (ss[0] - t_onset) * 1000, "item": item},
        "flush_dt_ms": None if flush_t is None else (flush_t - t_onset) * 1000,
        "stopped_dt_ms": None if stopped is None else (stopped[0] - t_onset) * 1000,
        "transcript": None if transcript is None else {
            "text": transcript[1].get("transcript", ""),
            "dt_from_stopped_ms": None if stopped is None else (transcript[0] - stopped[0]) * 1000,
            "dt_from_onset_ms": (transcript[0] - t_onset) * 1000},
        "client": client_summary(call, since, t_inj - 0.5, t_inj + duration + 6.0),
        "window_ms": [first * 20 - 500, end_n * 20 + 8000],
    }
    if burst_info is not None:
        burst_info["realised_db"] = call.bursts[-1].realised_db()
        row["burst"] = burst_info
    await cleanup(call, client)
    return row


async def run_trial(rig: Rig, call: Any, client: Any, spec: dict[str, Any], clips: dict[str, bytes]) -> dict[str, Any]:
    scenario = spec["scenario"]
    if scenario in ("A", "Am"):
        return await trial_a(call, client, spec)
    if scenario in ("B", "Be", "E0", "F"):
        pcm = read_wav(Path(spec["clip"]["path"]))
        return await trial_inject(call, client, spec, pcm, rig, playing=scenario != "E0")
    if scenario == "C":
        return await trial_inject(call, client, spec, clips[spec["utterance"]], rig, playing=True)
    pcm = synth_noise(spec["noise"], spec["seed"], spec["peak_dbfs"])
    return await trial_inject(call, client, spec, pcm, rig, playing=True)


async def short_clips(rig: Rig) -> dict[str, bytes]:
    out = {}
    for text, voice, lang in SHORT_UTTERANCES:
        path = HOLD / "clips" / f"short_{voice}_{text.replace(' ', '_')}.wav"
        if path.exists():
            out[text] = read_wav(path)
            continue
        pcm = await rig.synth(text, voice, lang)
        write_wav(path, pcm)
        out[text] = pcm
    return out


async def run_job(rig: Rig, job: dict[str, Any], clips: dict[str, bytes]) -> dict[str, Any]:
    condition = job["condition"]
    rows = []
    started = time.time()
    async with rig.session(condition, warmup_text=WARMUP, **job["session"]) as (client, call, session_id):
        for spec in job["trials"]:
            began = time.time()
            try:
                row = await run_trial(rig, call, client, spec, clips)
            except Exception as error:  # a failed trial is a row, not the end of the run
                traceback.print_exc()
                row = {"error": repr(error)}
                await cleanup(call, client)
            rows.append({**spec, **row, "condition": condition, "session": session_id,
                         "job": job["name"], "wall_s": round(time.time() - began, 1)})
            print(f"  {job['name']} {spec['id']} {rows[-1].get('error', 'ok')} {rows[-1]['wall_s']} s", flush=True)
    await asyncio.sleep(2.0)  # the capture writer thread finishes the files
    directory = session_dir(rig.root / condition, session_id)
    events = load_events(directory) if directory else []
    for row in rows:
        if "window_ms" in row:
            row["capture"] = capture_items(events, *row["window_ms"])
        row["capture_dir"] = None if directory is None else directory.name
    return {"job": job["name"], "condition": condition, "session": session_id, "rows": rows,
            "wall_s": round(time.time() - started, 1)}


def write_meta(smoke: bool) -> None:
    import subprocess

    import hold_lib
    import yaml

    if smoke:
        return
    config = yaml.safe_load((HERE.parents[1] / "config.yaml").read_text())
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short", "src", "config.yaml"], capture_output=True, text=True, cwd=HERE.parents[1]).stdout.strip().splitlines()
    meta = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "pmset": subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout,
        "git": f"HEAD {git} plus the uncommitted fix ({len(dirty)} changed paths under src/ and config.yaml)",
        "models": {"asr": config["providers"]["asr"]["model"], "tts": config["providers"]["tts"]["primary"]["model"],
                   "bot voice": config["voices"]["default"], "vad": "Silero + Smart Turn (config.yaml turn:)"},
        "config": {"barge_in (fix)": config["barge_in"], "aec": {k: config["aec"][k] for k in ("enabled", "engine", "asr_input")}, "output.max_lead_ms": config["output"]["max_lead_ms"]},
        "seed": SEED,
        "burst_model": {"profiles": hold_lib.PROFILES, "gain_rho": hold_lib.GAIN_RHO, "ir_taps": hold_lib.IR_TAPS, "delays_ms": [316, 506], "levels_db": [-5, -20, -35], "length_s": "1.5-2.5"},
    }
    ROWS.mkdir(parents=True, exist_ok=True)
    path = ROWS / "meta.json"
    if path.exists():  # a later run of more scenarios: keep the first run's record, add this one
        first = json.loads(path.read_text())
        first.setdefault("later_runs", []).append({"started": meta["started"], "pmset": meta["pmset"]})
        meta = first
    path.write_text(json.dumps(meta, indent=1))


async def run(only: set[str], smoke: bool, conditions: tuple[str, ...]) -> None:
    jobs = build_jobs(only, smoke, conditions)
    ROWS.mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not (ROWS / f"{j['name']}{'.smoke' if smoke else ''}.json").exists()]
    print(f"{len(jobs)} jobs, {len(pending)} to run", flush=True)
    write_meta(smoke)
    with backends_running(HOLD / "logs"):
        async with Rig() as rig:
            clips = await short_clips(rig) if "C" in only else {}
            for job in pending:
                print(f"== {job['name']} ({len(job['trials'])} trials)", flush=True)
                result = await run_job(rig, job, clips)
                (ROWS / f"{job['name']}{'.smoke' if smoke else ''}.json").write_text(json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--only", default="A,Am,B,Be,E0,C,D")
    runner.add_argument("--conditions", default="baseline,fix")
    runner.add_argument("--smoke", action="store_true", help="two or three trials per scenario")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "run":
        asyncio.run(run(set(args.only.split(",")), args.smoke, tuple(args.conditions.split(","))))
    else:
        from hold_report import report

        report()


if __name__ == "__main__":
    main()
