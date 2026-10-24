"""Term inventory and the context strings of the ASR context-biasing measurement (issue #1).

All brands and codes are fictional. ``DEPLOYMENT_TERMS`` is what a deployment would put in the
ASR context: brand, product and plan names, and the English jargon its callers mix into Chinese.
Order numbers cannot be listed one by one; the context only shows their format, with examples
that are NOT in the corpus.
"""

from __future__ import annotations

BRANDS = ["NovaNet", "Orbix", "Lumio", "Veltra", "Zentra", "SkyWave"]
BRAND_NOUN = {  # what the brand names, for the long-context padding
    "NovaNet": "宽带和移动业务", "Orbix": "云盘", "Lumio": "智能音箱", "Veltra": "路由器",
    "Zentra": "支付", "SkyWave": "电视盒子",
}
PRODUCTS = ["Wi-Fi 7", "Wi-Fi 6", "eSIM", "5G 畅享 199", "5G 尊享 299", "5G-A", "NovaNet Home Pro",
            "NovaNet Max", "NovaCloud", "NovaTV", "Mesh 路由器", "VoLTE", "IPTV", "AX3000"]
JARGON = ["App", "SIM 卡", "VPN", "APN", "OTA 升级", "PIN 码", "DNS", "IP 地址", "SSID", "Ping 值",
          "VIP", "LTE"]

# Format examples shown in the context (not used as test codes).
CODE_EXAMPLES = ["NV20240101", "TK-10001", "SO-100-200", "RMA-0001"]

CONTEXT = (
    "这是一通宽带和手机业务的客服热线电话。通话中可能出现的专有名词：" + "、".join(BRANDS) + "、"
    + "、".join(PRODUCTS) + "、" + "、".join(JARGON) + "。订单号、工单号和维修单号的格式如 "
    + "、".join(CODE_EXAMPLES) + "。"
)

# The bare comma-separated list (what a config example or a typical deployment sends).
CONTEXT_TERMS = ", ".join(BRANDS + PRODUCTS + JARGON)

# The sentence and the term list but no order-number examples (the examples in CONTEXT are copied
# into the transcript of order-number clips: see the report).
CONTEXT_NOEX = CONTEXT.split("订单号、工单号")[0]

# A different language from the speech: describes the call, lists no term.
CONTEXT_EN = ("This is a customer-service phone call about home broadband and mobile plans. "
              "The caller asks questions about their account, devices and orders.")


def _padding() -> list[str]:
    """Extra (unused) glossary lines: fictional accessory SKUs, service tiers and store names."""
    lines = []
    for i in range(1, 80):
        lines.append(f"配件型号 NX-{100 + 7 * i}、服务档位 Tier-{i % 9 + 1}、门店 Nova Store {i:02d}")
    return lines


def _grow(base: str, target: int, *, utf8: bool = False) -> str:
    size = (lambda s: len(s.encode())) if utf8 else len
    out = base
    for line in _padding() * 40:
        if size(out) + size(line) + 1 > target:
            break
        out += "\n" + line
    return out


CONTEXT_LONG = _grow(CONTEXT, 1000)  # ~1000 characters, the deployment terms first
CONTEXT_HUGE = _grow(CONTEXT, 10 * 1024, utf8=True)  # just under 10 KiB of UTF-8

assert len(CONTEXT) < 600
assert 950 <= len(CONTEXT_LONG) <= 1000, len(CONTEXT_LONG)
assert 10 * 1024 - 100 <= len(CONTEXT_HUGE.encode()) <= 10 * 1024
