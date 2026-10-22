from __future__ import annotations

import pytest

from mlx_realtime.text.segmenter import Segmenter, text_units

PARAMS = {"first_min_chars": 4, "first_max_wait_ms": 400, "min_chars": 8, "max_chars": 80}


def segment(text: str, step: int = 1, **overrides: int) -> list[str]:
    """Stream ``text`` in ``step``-character pieces, all at t=0 (the timer never fires)."""
    segmenter = Segmenter(**{**PARAMS, **overrides})
    segments: list[str] = []
    for start in range(0, len(text), step):
        segments += segmenter.push(text[start : start + step], 0)
    segments += segmenter.flush()
    assert "".join(segments) == text
    return segments


@pytest.mark.parametrize("step", [1, 3, 1000])
def test_chinese(step: int) -> None:
    text = "你好！我是客服小王，很高兴为您服务。请问有什么可以帮您？"
    assert segment(text, step) == [
        "你好！",
        "我是客服小王，很高兴为您服务。",
        "请问有什么可以帮您？",
    ]


@pytest.mark.parametrize("step", [1, 5])
def test_english(step: int) -> None:
    text = "Hello there! How can I help you today? I can check your balance right now."
    assert segment(text, step) == [
        "Hello there! ",
        "How can I help you today? I can check your balance right now.",
    ]


def test_mixed_language_and_ascii_punctuation_before_cjk() -> None:
    assert segment("OK!我们开始吧。这是 Qwen 模型的测试，请稍等。") == [
        "OK!",
        "我们开始吧。这是 Qwen 模型的测试，请稍等。",
    ]


def test_first_segment_ends_at_a_soft_boundary_once_long_enough() -> None:
    assert (
        segment("好的，我来帮您查询一下。")[0] == "好的，我来帮您查询一下。"
    )  # "好的" is too short
    assert segment("您好，欢迎致电客服中心，请问有什么可以帮您？") == [
        "您好，欢迎致电客服中心，",
        "请问有什么可以帮您？",
    ]
    assert segment("Well, I think so: that works for me.")[0] == "Well, I think so: "


def test_later_segments_need_min_chars() -> None:
    assert segment("好。对。是的。没问题，马上为您办理。") == [
        "好。",
        "对。是的。没问题，马上为您办理。",
    ]


@pytest.mark.parametrize(
    ("text", "intact"),
    [
        ("价格是3.5元，总共1,000元。", "3.5"),
        ("总共1,000,000元，谢谢您。", "1,000,000"),
        ("请访问 https://example.com/a?b=1&c=2, 然后告诉我。", "https://example.com/a?b=1&c=2"),
        (
            "Email me at first.last@example.co.uk. Thanks a lot for that.",
            "first.last@example.co.uk",
        ),
        ("订单号 ORD-2024-001.A 已经发货，请注意查收。", "ORD-2024-001.A"),
        ("The meeting is at 10:30 today, see you there.", "10:30"),
        ("Pi is 3.14159 and e is 2.71828, both irrational.", "3.14159"),
    ],
)
def test_never_splits_numbers_urls_emails_or_ids(text: str, intact: str) -> None:
    for step in (1, 2, 7):
        assert any(intact in piece for piece in segment(text, step)), (step, segment(text, step))


def test_a_trailing_dot_waits_for_the_next_character() -> None:
    segmenter = Segmenter(**PARAMS)
    assert segmenter.push("价格是3.", 0) == []
    assert segmenter.push("5元。", 0) == ["价格是3.5元。"]


def test_abbreviations_initials_and_list_numbers_do_not_end_sentences() -> None:
    assert segment("Mr. Smith met Dr. J. Watson at noon. Then they left together quietly.") == [
        "Mr. Smith met Dr. J. Watson at noon. ",
        "Then they left together quietly.",
    ]
    assert segment("1. First item here\n2. Second item here\n") == [
        "1. First item here\n",
        "2. Second item here\n",
    ]


def test_ellipsis_and_closing_quotes_stay_with_their_sentence() -> None:
    assert segment('Wait... what? He said "fine." Then he left the room quietly.') == [
        "Wait... ",
        'what? He said "fine." Then he left the room quietly.',
    ]
    quoted = "他说：“好的。”然后就离开了会议室，没有再回来。"
    assert segment(quoted, step=len(quoted))[0] == "他说：“好的。”"
    # Streamed, a CJK full stop ends the segment at once; the late quote opens the next one.
    assert segment(quoted, step=1)[:2] == ["他说：“好的。", "”然后就离开了会议室，没有再回来。"]


def test_newline_is_a_hard_boundary_and_blank_lines_never_form_a_segment() -> None:
    assert segment("\n\n好的。") == ["\n\n好的。"]
    assert segment("第一行内容\n第二行内容在这里结束了。") == [
        "第一行内容\n",
        "第二行内容在这里结束了。",
    ]


def test_max_chars_splits_at_the_latest_soft_boundary() -> None:
    clause = "这是一个比较长的分句"  # 10 units
    text = "，".join([clause] * 10) + "。"
    pieces = segment(text, max_chars=35)
    assert all(text_units(piece) <= 35 for piece in pieces)
    assert all(piece.endswith(("，", "。")) for piece in pieces)


def test_max_chars_falls_back_to_spaces_then_cjk_pairs() -> None:
    words = " ".join(["word"] * 50)
    assert all(text_units(piece) <= 20 for piece in segment(words, max_chars=20))
    chinese = "没有标点符号的长句子" * 12
    pieces = segment(chinese, max_chars=30)
    assert [text_units(piece) for piece in pieces] == [30, 30, 30, 30]


def test_first_segment_timer_takes_the_best_boundary_so_far() -> None:
    segmenter = Segmenter(**PARAMS)
    assert segmenter.push("嗯，让我", 1000) == []  # "嗯" is below first_min_chars
    assert segmenter.deadline_ms() == 1400
    assert segmenter.poll(1399) == []
    assert segmenter.poll(1400) == ["嗯，"]
    assert segmenter.deadline_ms() is None


def test_first_segment_timer_falls_back_to_a_space() -> None:
    segmenter = Segmenter(**PARAMS)
    assert segmenter.push("Hello world how are", 0) == []
    assert segmenter.poll(400) == ["Hello world how "]
    assert segmenter.flush() == ["are"]


def test_timer_without_any_boundary_waits_for_more_text() -> None:
    segmenter = Segmenter(**PARAMS)
    assert segmenter.push("好", 0) == []
    assert segmenter.poll(1000) == []
    assert segmenter.push("的。", 1001) == ["好的。"]


def test_text_units_count_cjk_characters_and_latin_words() -> None:
    assert text_units("你好 world, it's 3.5 元") == 3 + 4


def test_spaced_digit_groups_are_never_cut() -> None:
    # The first segment's timer and the max_chars fallback cut at spaces, but never between the
    # groups of a spaced phone or card number, even before the next group has arrived.
    segmenter = Segmenter(**PARAMS)
    assert segmenter.push("请拨打 138 ", 0) == []
    assert segmenter.poll(400) == ["请拨打 "]
    assert segmenter.push("0013 8000 联系我们。", 450) == []  # 7 units: below min_chars
    assert segmenter.flush() == ["138 0013 8000 联系我们。"]
    english = Segmenter(**PARAMS)
    assert english.push("Call 400 820 8820 or", 0) == []
    assert english.poll(400) == ["Call 400 820 8820 "]
    pieces = segment("卡号 6222 0212 3456 7890 123 请核对。" * 3, max_chars=12)
    assert all("6222 0212 3456 7890 123" in piece for piece in pieces if "6222" in piece)
