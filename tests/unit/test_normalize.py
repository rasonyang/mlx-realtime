from __future__ import annotations

from typing import Literal

import pytest

from mlx_realtime.text.normalize import NormalizerOptions, normalize, normalize_with_map

DEFAULT = NormalizerOptions()
NARROW_OFF: dict[str, bool] = {
    "cardinals": False,
    "emails": False,
    "en_dates": False,
    "en_amounts": False,
}
PHONES_AND_IDS = NormalizerOptions(**NARROW_OFF)  # the M3 defaults, before G5
NOTHING = NormalizerOptions(phones=False, ids=False, **NARROW_OFF)
EVERYTHING = NormalizerOptions(numbers=True, money=True, dates=True, units=True, spacing=True)
NUMBERS = NormalizerOptions(numbers=True)
MONEY = NormalizerOptions(money=True)
DATES = NormalizerOptions(dates=True)
UNITS = NormalizerOptions(units=True)
UNITS_AND_NUMBERS = NormalizerOptions(units=True, numbers=True)
SPACING = NormalizerOptions(spacing=True)

COMBOS = {
    "default": DEFAULT,
    "nothing": NOTHING,
    "everything": EVERYTHING,
    "numbers": NUMBERS,
    "money-dates-units": NormalizerOptions(money=True, dates=True, units=True),
    "spacing": SPACING,
    "numbers-without-yao": NormalizerOptions(numbers=True, phone_one_as_yao=False),
    "phones-and-ids": PHONES_AND_IDS,
    "cardinals-from-3": NormalizerOptions(min_cardinal_digits=3),
    "disabled": NormalizerOptions(enabled=False, numbers=True, spacing=True),
}


# --- Unchanged text -------------------------------------------------------------------------


@pytest.mark.parametrize("options", list(COMBOS.values()), ids=list(COMBOS))
@pytest.mark.parametrize(
    "text",
    [
        "你好，世界。",
        "Thanks for calling NovaNet.",
        "Good morning, NovaNet support.",
        "请讲一个很长很长的故事。",
        "我在用 Qwen3 模型。",
        "iPhone15 很好用",
        "COVID-19",
        "MP3",
        "5G 网络",
        "H2O",
        "A1B2 和 GPT-4o",
        "第一行内容\n第二行内容。",
        "Hello there! ",
        "",
    ],
)
def test_text_without_patterns_is_unchanged(text: str, options: NormalizerOptions) -> None:
    assert normalize(text, options) == text


def test_disabled_returns_the_text_unchanged() -> None:
    options = NormalizerOptions(
        enabled=False, numbers=True, money=True, dates=True, units=True, spacing=True
    )
    text = (
        "请拨打 138-0013-8000，订单号 ORD-2024-001，¥12.5，2024-10-23 14:30，5km，50%，我用iPhone"
    )
    assert normalize(text, options) == text
    assert normalize(text, options, lang="en") == text


def test_defaults_follow_g5() -> None:
    expected_defaults = NormalizerOptions(
        enabled=True,
        numbers=False,
        cardinals=True,
        min_cardinal_digits=5,
        money=False,
        dates=False,
        phones=True,
        ids=True,
        emails=True,
        units=False,
        spacing=False,
        en_dates=True,
        en_amounts=True,
        en_amount_min=1000,
        phone_one_as_yao=True,
    )
    assert expected_defaults == DEFAULT
    text = "订单 ORD20241023001 金额 ¥12.5，2024-10-23 送达 5km，共 3 件，电话 13800138000"
    assert normalize(text) == (
        "订单 O R D 二零二四，一零二三，零零一 金额 ¥12.5，2024-10-23 送达 5km，共 3 件，"
        "电话 幺三八，零零幺三，八零零零"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # M0 S9 sentences the TTS read wrongly by itself (spikes/audio/b2_texts.py).
        ("您的积分余额是15600分。", "您的积分余额是一万五千六百分。"),
        ("您的手机号码尾号是5678。", "您的手机号码尾号是五六七八。"),
        ("请回拨021-12345678转8001分机。", "请回拨零二幺，幺二三四，五六七八转八零零幺分机。"),
        (
            "您登记的手机号是138 0013 8000，对吗？",
            "您登记的手机号是幺三八，零零幺三，八零零零，对吗？",
        ),
        ("您的工单编号为TK-2024-0789。", "您的工单编号为T K，二零二四，零七八九。"),
        ("退货单号是RT20241102A3。", "退货单号是R T 二零二四，一一零二 A 三。"),
        (
            "请把截图发到我们的邮箱service@example.com。",
            "请把截图发到我们的邮箱service艾特example点com。",
        ),
        (
            "Your order total is $1,280.50.",
            "Your order total is one thousand two hundred eighty dollars and fifty cents.",
        ),
        (
            "Please call us at 1-800-555-0199.",
            "Please call us at one, eight zero zero, five five five, zero one nine nine.",
        ),
        (
            "Your appointment is on 03/15/2024 at 2:30 PM.",
            "Your appointment is on March fifteenth, twenty twenty-four at 2:30 PM.",
        ),
        (
            "Your order number is A20241015001.",
            "Your order number is A two zero two four, one zero one five, zero zero one.",
        ),
    ],
)
def test_defaults_read_what_the_tts_misreads(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # M0 S9 sentences the TTS read right by itself: G5 leaves them as written.
        "您本月一共有3笔订单待处理。",
        "当前排队人数为128人，请您耐心等待。",
        "本次活动全场8.5折。",
        "您的好评率达到了99.6%。",
        "您是今天第2位来电的客户。",
        "您本期账单应还金额为¥1,280.50。",
        "您充值的100元已经到账。",
        "该套餐的月费为$29.99。",
        "违约金为合同金额的5%，即2500元。",
        "您的账单金额是12,800元。",
        "您的预约时间是2024年3月15日下午2:30。",
        "活动截止到12月31日24:00。",
        "您的新套餐将于2025-01-01生效。",
        "日期是03/15/2024。",
        "包裹重量为2.5kg。",
        "您家的宽带速率是500Mbps。",
        "您可以登录官网www.example.com查询。",
        "您的iPhone 15需要送修吗？",
        "请先连接Wi-Fi再下载更新包。",
        "The package weighs 2.5 kg.",
        "You get 15% off orders over $200.",
        "It costs $999.99 at 10:30 on May 5.",
        "She came 2nd, with 3.5 points and 1234 votes.",
        "我们支持PayPal和Apple Pay付款。",
    ],
)
def test_defaults_leave_what_the_tts_reads_right(text: str) -> None:
    assert normalize(text) == text


# --- Phones ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("请拨打 138-0013-8000 联系我们", "请拨打 幺三八，零零幺三，八零零零 联系我们"),
        ("手机13800138000", "手机幺三八，零零幺三，八零零零"),
        ("手机 138 0013 8000", "手机 幺三八，零零幺三，八零零零"),
        ("手机 +86 138 0013 8000", "手机 加八六，幺三八，零零幺三，八零零零"),
        ("手机 +8613800138000", "手机 加八六，幺三八，零零幺三，八零零零"),
        ("手机 0086-13800138000", "手机 加八六，幺三八，零零幺三，八零零零"),
        ("手机 86 13800138000", "手机 加八六，幺三八，零零幺三，八零零零"),
        ("座机 010-12345678", "座机 零幺零，幺二三四，五六七八"),
        ("座机 (0755) 1234 5678", "座机 零七五五，幺二三四，五六七八"),
        ("座机 021 6123 4567", "座机 零二幺，六幺二三，四五六七"),
        ("座机 0755-1234567", "座机 零七五五，幺二三，四五六七"),
        ("座机01012345678", "座机零幺零，幺二三四，五六七八"),
        ("客服 400-820-8820", "客服 四零零，八二零，八八二零"),
        ("客服 4008208820", "客服 四零零，八二零，八八二零"),
        ("客服 800 810 8888", "客服 八零零，八幺零，八八八八"),
        ("客服 4008-208-820", "客服 四零零，八二零，八八二零"),
        ("请拨打95588", "请拨打九五五八八"),
        ("请拨打110", "请拨打幺幺零"),
        ("热线5588", "热线五五八八"),
        ("致电 12345 市民热线", "致电 幺二三四五 市民热线"),
        ("移动客服10086", "移动客服幺零零八六"),
        ("客服热线是 95588。", "客服热线是 九五五八八。"),
        (
            "电话：010-12345678，手机：13800138000",
            "电话：零幺零，幺二三四，五六七八，手机：幺三八，零零幺三，八零零零",
        ),
    ],
)
def test_phones_in_chinese(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Call 400-820-8820 now.",
            "Call four zero zero, eight two zero, eight eight two zero now.",
        ),
        (
            "My number is 138-0013-8000.",
            "My number is one three eight, zero zero one three, eight zero zero zero.",
        ),
        (
            "Dial +86 138 0013 8000 today",
            "Dial plus eight six, one three eight, zero zero one three, eight zero zero zero today",
        ),
        (
            "Office: (010) 1234 5678.",
            "Office: zero one zero, one two three four, five six seven eight.",
        ),
        ("Tel:13800138000", "Tel:one three eight, zero zero one three, eight zero zero zero"),
        ("Call 95588 now", "Call nine five five eight eight now"),
        ("The hotline is 96123.", "The hotline is nine six one two three."),
    ],
)
def test_phones_in_english(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_phone_one_as_yao_toggle() -> None:
    assert normalize("拨打13812345671") == "拨打幺三八，幺二三四，五六七幺"
    no_yao = NormalizerOptions(phone_one_as_yao=False)
    assert normalize("拨打13812345671", no_yao) == "拨打一三八，一二三四，五六七一"
    assert normalize("Call 13812345671", DEFAULT) == (
        "Call one three eight, one two three four, five six seven one"
    )


@pytest.mark.parametrize(
    "text",
    [
        "共138001380001个",  # 12 digits
        "共1380013800个",  # 10 digits
        "号码12345678901",  # 1 is not followed by 3-9
        "95588",  # a hotline needs a phone keyword
        "热线电话12",  # too short
        "价格13800138000.5",
        "¥13800138000",
        "13800138000元",
        "会议2024-10-23开",
        "14:30开会",
    ],
)
def test_not_phones(text: str) -> None:
    assert normalize(text, PHONES_AND_IDS) == text


def test_phones_off_leaves_phones_as_written() -> None:
    assert normalize("请拨打 138-0013-8000", NormalizerOptions(phones=False)) == (
        "请拨打 138-0013-8000"
    )


# --- IDs ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("您的订单号是 ORD-2024-001。", "您的订单号是 O R D，二零二四，零零一。"),
        ("订单 ORD20241023001 已发货", "订单 O R D 二零二四，一零二三，零零一 已发货"),
        ("顺丰单号SF1234567890", "顺丰单号S F 一二三四，五六七八，九零"),
        ("京东 JD0012345678", "京东 J D 零零一二，三四五六，七八"),
        ("编号 ORD-2024-001.A", "编号 O R D，二零二四，零零一，A"),
        ("激活码 A1B2C3D4", "激活码 A 一 B 二 C 三 D 四"),
        ("显卡是 RTX4090", "显卡是 R T X 四零九零"),
        ("航班CA1234", "航班C A 一二三四"),
        ("单号 sf1234567", "单号 S F 一二三四，五六七"),
        ("发票 INV.2024.0001", "发票 I N V，二零二四，零零零一"),
        ("订单号：20241023001", "订单号：二零二四，一零二三，零零一"),
        ("快递单号是 1234-5678-9012", "快递单号是 一二三四，五六七八，九零一二"),
        ("卡号 6222 0212 3456 7890", "卡号 六二二二，零二一二，三四五六，七八九零"),
        ("验证码为123456", "验证码为一二三四，五六"),
        ("会员号 #12345678", "会员号 #一二三四，五六七八"),
        ("订单号 No. 20241023 已发货", "订单号 No. 二零二四，一零二三 已发货"),
        ("您的ID是123456", "您的ID是一二三四，五六"),
    ],
)
def test_ids_in_chinese(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Your order SF1234567890 has shipped.",
            "Your order S F one two three four, five six seven eight, nine zero has shipped.",
        ),
        ("Order ORD-2024-001 is ready.", "Order O R D, two zero two four, zero zero one is ready."),
        (
            "Your order number is 20241023001.",
            "Your order number is two zero two four, one zero two three, zero zero one.",
        ),
        ("Order #123456 shipped", "Order #one two three four, five six shipped"),
        ("Order No. 12345678", "Order No. one two three four, five six seven eight"),
        (
            "Tracking number: 1234567890",
            "Tracking number: one two three four, five six seven eight, nine zero",
        ),
        ("Ref ABC-123456", "Ref A B C, one two three four, five six"),
        ("Your ID is 987654.", "Your ID is nine eight seven six, five four."),
        ("Confirmation code 482913", "Confirmation code four eight two nine, one three"),
        ("The RTX4090 is fast", "The R T X four zero nine zero is fast"),
    ],
)
def test_ids_in_english(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "我在用 Qwen3 模型。",
        "iPhone15 很好用",
        "A1B2 和 X1",
        "号码 123456",  # digits without an ID keyword
        "订单号 123",  # too short
        "A1B2C",  # fewer than 3 digits
        "No.123456",
        "Windows2000",  # a word, not an ID
        "report2024.pdf",
        "1000km 5000mAh",
        "the 1990s",
        "open 10am-12pm",
        "USD2000 1000RMB",
        "1920x1080",
        "我在用 Qwen2.5-72B 模型",  # a version number
        "Llama-3.1-70B",
        "车次G1234",  # fewer than 6 letters and digits
        "https://example.com/ORD20241023001",
    ],
)
def test_not_ids(text: str) -> None:
    assert normalize(text, PHONES_AND_IDS) == text


def test_ids_off_leaves_ids_as_written() -> None:
    assert normalize("订单 ORD-2024-001", NormalizerOptions(ids=False)) == "订单 ORD-2024-001"


# --- Numbers --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", "零"),
        ("10", "十"),
        ("11", "十一"),
        ("20", "二十"),
        ("101", "一百零一"),
        ("110", "一百一十"),
        ("1005", "一千零五"),
        ("1010", "一千零一十"),
        ("10010", "一万零一十"),
        ("100000", "十万"),
        ("110000", "十一万"),
        ("1,234,567", "一百二十三万四千五百六十七"),
        ("100000000", "一亿"),
        ("100010000", "一亿零一万"),
        ("120000000", "一亿二千万"),
        ("1000000001", "十亿零一"),
        ("1234567890123456", "一千二百三十四万五千六百七十八亿九千零一十二万三千四百五十六"),
        ("007", "零零七"),
        ("3.14", "三点一四"),
        ("0.5", "零点五"),
        ("10.05", "十点零五"),
        ("50%", "百分之五十"),
        ("3.5%", "百分之三点五"),
        ("100%", "百分之一百"),
        ("-5", "负五"),
        ("-0.5", "负零点五"),
        ("-5%", "负百分之五"),
        ("12345678901234567", "12345678901234567"),  # past 16 digits: left as written
    ],
)
def test_numbers_in_chinese(text: str, expected: str) -> None:
    assert normalize(text, NUMBERS, lang="zh") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("我要2个苹果", "我要两个苹果"),
        (
            "2位 2件 2次 2天 2小时 2分钟 2张 2本 2台",
            "两位 两件 两次 两天 两小时 两分钟 两张 两本 两台",
        ),
        ("大约2万人", "大约两万人"),
        ("第2个", "第二个"),
        ("2年级", "二年级"),
        ("2分之1", "二分之一"),
        ("共22个", "共二十二个"),
        ("等了2.5小时", "等了二点五小时"),
        ("3-5天送达", "三到五天送达"),
        ("3~5天送达", "三到五天送达"),
        ("2-3个", "两到三个"),
        ("增长了3-5%", "增长了百分之三到百分之五"),
        ("温度为-5度", "温度为负五度"),
        ("气温-3度", "气温负三度"),
        ("今天-3度", "今天负三度"),
        ("气温10度-15度", "气温十度-十五度"),  # a dash after a number with a unit
        ("增长了3.5%", "增长了百分之三点五"),
        ("他得了1st", "他得了第一"),
        ("第1st名", "第一名"),
        ("版本5-3不是范围", "版本5-3不是范围"),
        ("日期10/11/12", "日期10/11/12"),
        ("版本1.2.3", "版本1.2.3"),
        ("地址192.168.1.1", "地址192.168.1.1"),
    ],
)
def test_numbers_in_chinese_text(text: str, expected: str) -> None:
    assert normalize(text, NUMBERS) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", "zero"),
        ("15", "fifteen"),
        ("101", "one hundred one"),
        ("999", "nine hundred ninety-nine"),
        ("1234", "one thousand two hundred thirty-four"),
        ("1,000,000", "one million"),
        (
            "1,234,567,890",
            "one billion two hundred thirty-four million five hundred sixty-seven "
            "thousand eight hundred ninety",
        ),
        ("007", "zero zero seven"),
        ("3.14", "three point one four"),
        ("0.5", "zero point five"),
        ("50%", "fifty percent"),
        ("-5", "minus five"),
        ("1st 2nd 3rd 21st", "first second third twenty-first"),
        (
            "11th 12th 13th 22nd 101st",
            "eleventh twelfth thirteenth twenty-second one hundred first",
        ),
        ("1th 2st", "1th 2st"),  # the suffix must fit
    ],
)
def test_numbers_in_english(text: str, expected: str) -> None:
    assert normalize(text, NUMBERS, lang="en") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It takes 3-5 days.", "It takes three to five days."),
        ("Prices rose 3.5% this year.", "Prices rose three point five percent this year."),
        ("Set x = -5 now", "Set x = minus five now"),
        ("Compute 3 - 5 now", "Compute three - five now"),  # subtraction, not a negative
        ("She came 2nd of 10 runners", "She came second of ten runners"),
    ],
)
def test_numbers_in_english_text(text: str, expected: str) -> None:
    assert normalize(text, NUMBERS) == expected


@pytest.mark.parametrize(
    "text",
    [
        "我在用 Qwen3 模型。",
        "A1B2",
        "COVID-19",
        "v1.2",
        "H2O",
        "5G 网络",
        "3D 打印",
        "a 19-year-old",
        "1,23",
        "1e5",
    ],
)
def test_numbers_inside_tokens_are_not_read(text: str) -> None:
    assert normalize(text, NUMBERS) == text


# --- Money ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("价格¥1,234.56", "价格一千二百三十四元五角六分"),
        ("价格1234.56元", "价格一千二百三十四元五角六分"),
        ("只要￥88", "只要八十八元"),
        ("只要12.5元", "只要十二元五角"),
        ("只要12.05元", "只要十二元零五分"),
        ("只要0.5元", "只要五角"),
        ("只要2元", "只要两元"),
        ("只要5块钱", "只要五块钱"),
        ("只要12.5块", "只要十二块五毛"),
        ("只要3块5", "只要3块5"),
        ("只要$12.99", "只要十二点九九美元"),
        ("只要12.99美元", "只要十二点九九美元"),
        ("只要USD 20", "只要二十美元"),
        ("只要20 USD", "只要二十美元"),
        ("只要100日元", "只要一百日元"),
        ("只要€5", "只要五欧元"),
        ("只要HK$10", "只要十港元"),
        ("融资1.5万元", "融资一点五万元"),
        ("融资$1.5 million", "融资一百五十万美元"),
        ("价格在¥100-200之间", "价格在一百到二百元之间"),
        ("只要¥100元", "只要一百元"),
    ],
)
def test_money_in_chinese(text: str, expected: str) -> None:
    assert normalize(text, MONEY) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It costs $12.99.", "It costs twelve dollars and ninety-nine cents."),
        ("Only $1!", "Only one dollar!"),
        ("That is ¥100.", "That is one hundred yuan."),
        ("Pay $0.99 now", "Pay ninety-nine cents now"),
        ("Pay $1.01 now", "Pay one dollar and one cent now"),
        ("about €2.50 each", "about two euros and fifty cents each"),
        ("£1 and £3.50", "one pound and three pounds and fifty pence"),
        ("USD 20 total", "twenty dollars total"),
        ("raised $1.5 million", "raised one point five million dollars"),
        ("save $10-20 now", "save ten to twenty dollars now"),
        ("a $1,000,000 prize", "a one million dollars prize"),
        ("about 5万元 in total", "about fifty thousand yuan in total"),
        ("a $5k bonus", "a $5k bonus"),
    ],
)
def test_money_in_english(text: str, expected: str) -> None:
    assert normalize(text, MONEY) == expected


# --- Dates and times ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("会议定在2024-10-23", "会议定在二零二四年十月二十三日"),
        ("会议定在2024/10/23", "会议定在二零二四年十月二十三日"),
        ("会议定在2024.10.23", "会议定在二零二四年十月二十三日"),
        ("会议定在2024年10月23日", "会议定在二零二四年十月二十三日"),
        ("会议定在2024年10月23号", "会议定在二零二四年十月二十三号"),
        ("10月23日见", "十月二十三日见"),
        ("2024年10月开始", "二零二四年十月开始"),
        ("2024年的计划", "二零二四年的计划"),
        ("2月29日", "二月二十九日"),
        ("下午14:30开会", "下午十四点三十分开会"),
        ("早上09:05出发", "早上九点零五分出发"),
        ("8:00上班", "八点上班"),
        ("2:00见", "两点见"),
        ("14:30:15到达", "十四点三十分十五秒到达"),
        ("营业时间9:00-17:30", "营业时间九点到十七点三十分"),
        ("明天9am见", "明天上午九点见"),
        ("日期2023-02-29不存在", "日期2023-02-29不存在"),
        ("日期2024-13-01不存在", "日期2024-13-01不存在"),
        ("时间25:00不存在", "时间25:00不存在"),
        ("日期10/11/12有歧义", "日期10/11/12有歧义"),
    ],
)
def test_dates_and_times_in_chinese(text: str, expected: str) -> None:
    assert normalize(text, DATES) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See you on 2024-10-23.", "See you on October twenty-third, twenty twenty-four."),
        ("Born 1999-01-01", "Born January first, nineteen ninety-nine"),
        ("Since 2000-02-02", "Since February second, two thousand"),
        ("Since 2005-03-03", "Since March third, two thousand five"),
        ("Since 1905-06-06", "Since June sixth, nineteen oh five"),
        ("at 14:30 today", "at fourteen thirty today"),
        ("at 9:05 today", "at nine oh five today"),
        ("at 10:00 today", "at ten o'clock today"),
        ("at 14:00 today", "at fourteen hundred today"),
        ("at 14:30:15 today", "at fourteen thirty and fifteen seconds today"),
        ("open 9am-5pm daily", "open nine A M to five P M daily"),
        ("at 9:30 pm.", "at nine thirty P M."),
        ("meet at 9 a.m. tomorrow", "meet at nine A M. tomorrow"),
        ("on 10/11/12", "on 10/11/12"),
    ],
)
def test_dates_and_times_in_english(text: str, expected: str) -> None:
    assert normalize(text, DATES) == expected


# --- Units ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("全程5km", "全程5公里"),
        ("重2 kg", "重2公斤"),
        ("气温-5°C", "气温-5摄氏度"),
        ("内存8GB", "内存8G B"),
        ("It is 5km away.", "It is 5 kilometers away."),
        ("Only 1 km left", "Only 1 kilometer left"),
    ],
)
def test_units_keep_the_number_without_numbers(text: str, expected: str) -> None:
    assert normalize(text, UNITS) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("全程5km", "全程五公里"),
        ("重2kg", "重两公斤"),
        ("容量2L", "容量两升"),
        ("气温-5°C", "气温负五摄氏度"),
        ("气温2℃", "气温二摄氏度"),
        ("限速100km/h", "限速一百公里每小时"),
        ("风速5m/s", "风速五米每秒"),
        ("面积50㎡", "面积五十平方米"),
        ("面积50m²", "面积五十平方米"),
        ("体积3m³", "体积三立方米"),
        ("耗电1.5kWh", "耗电一点五千瓦时"),
        ("功率60W", "功率六十瓦"),
        ("电池5000mAh", "电池五千毫安时"),
        ("频率2.4GHz", "频率二点四吉赫兹"),
        ("内存8GB", "内存八G B"),
        ("距离3-5km", "距离三到五公里"),
        ("身高180cm", "身高一百八十厘米"),
        ("水500ml", "水五百毫升"),
    ],
)
def test_units_in_chinese(text: str, expected: str) -> None:
    assert normalize(text, UNITS_AND_NUMBERS) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It is 5km away.", "It is five kilometers away."),
        ("Only 1 km left", "Only one kilometer left"),
        ("Only 0.5 km left", "Only zero point five kilometers left"),
        ("It is -5°C outside", "It is minus five degrees Celsius outside"),
        ("It is 1°C outside", "It is one degree Celsius outside"),
        ("Up to 100km/h now", "Up to one hundred kilometers per hour now"),
        ("It has 8GB of RAM", "It has eight gigabytes of RAM"),
        ("The 2.4GHz band", "The two point four gigahertz band"),
        ("Uses 1kWh daily", "Uses one kilowatt hour daily"),
    ],
)
def test_units_in_english(text: str, expected: str) -> None:
    assert normalize(text, UNITS_AND_NUMBERS) == expected


@pytest.mark.parametrize("text", ["跑了5min", "延迟5ms", "5G 网络", "I have 5 more"])
def test_not_units(text: str) -> None:
    assert normalize(text, UNITS) == text


# --- Spacing --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("我用iPhone手机", "我用 iPhone 手机"),
        ("你 好", "你好"),
        ("我有3个", "我有 3 个"),
        ("Hello世界", "Hello 世界"),
        ("我用  iPhone   手机", "我用 iPhone 手机"),
        ("a   b", "a b"),
        ("你好\n世界", "你好\n世界"),
        ("中文，English。", "中文，English。"),
    ],
)
def test_spacing(text: str, expected: str) -> None:
    assert normalize(text, SPACING) == expected


def test_spacing_runs_last_on_the_normalized_text() -> None:
    assert normalize("我用iPhone买了2个", EVERYTHING) == "我用 iPhone 买了两个"
    assert normalize("请拨打 138-0013-8000 联系我们", SPACING) == (
        "请拨打幺三八，零零幺三，八零零零联系我们"
    )


# --- G5: last four digits, extensions, North American numbers -------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("尾号1234的卡", "尾号幺二三四的卡"),
        ("手机尾号为 8801。", "手机尾号为 八八零幺。"),
        ("卡号末四位是0011", "卡号末四位是零零幺幺"),
        ("后四位：2468", "后四位：二四六八"),
        (
            "The card ending in 4321 is blocked.",
            "The card ending in four three two one is blocked.",
        ),
        ("last four digits are 1001", "last four digits are one zero zero one"),
        ("尾号12345", "尾号一万二千三百四十五"),  # five digits are no last four: a cardinal
    ],
)
def test_last_four_digits(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("请拨010-12345678转8023", "请拨零幺零，幺二三四，五六七八转八零二三"),
        ("请拨010-12345678 转 8023", "请拨零幺零，幺二三四，五六七八 转八零二三"),
        ("电话 010-12345678 分机 23", "电话 零幺零，幺二三四，五六七八 分机二三"),
        ("请转分机号是8023。", "请转分机号是八零二三。"),
        (
            "Call 800-555-0199 ext. 123",
            "Call eight zero zero, five five five, zero one nine nine extension one two three",
        ),
        (
            "Call 800-555-0199 x12",
            "Call eight zero zero, five five five, zero one nine nine extension one two",
        ),
        ("Extension 4567, please.", "extension four five six seven, please."),
        ("请拨 010-12345678 ext 8", "请拨 零幺零，幺二三四，五六七八 分机八"),
        ("转3次", "转3次"),  # 转 alone is a verb, not an extension
    ],
)
def test_extensions(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Call (212) 555-0147.", "Call two one two, five five five, zero one four seven."),
        ("Call 212.555.0147 now", "Call two one two, five five five, zero one four seven now"),
        (
            "Call +1 212 555 0147",
            "Call plus one, two one two, five five five, zero one four seven",
        ),
        ("美国号码 1-212-555-0147", "美国号码 幺，二幺二，五五五，零幺四七"),
        ("美国号码 +1 212 555 0147", "美国号码 加幺，二幺二，五五五，零幺四七"),
    ],
)
def test_north_american_numbers(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize("text", ["2125550147", "123-456-7890", "212-55-0147", "No 21-555-0147"])
def test_not_north_american_numbers(text: str) -> None:
    assert normalize(text, PHONES_AND_IDS) == text


# --- G5: IDs after their keyword, short alphanumeric IDs -------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("验证码1234", "验证码一二三四"),
        ("订单号 12345", "订单号 一二三四，五"),
        ("Your code is 482913", "Your code is four eight two nine, one three"),
        ("取件码 5821", "取件码 五八二一"),
        ("Confirmation code 4829", "Confirmation code four eight two nine"),
        ("激活码 A1B2C3", "激活码 A 一 B 二 C 三"),
        ("账户 abc123", "账户 A B C 一二三"),
        ("网速500Mbps", "网速500Mbps"),  # three digits first: a quantity
    ],
)
def test_short_ids(text: str, expected: str) -> None:
    assert normalize(text) == expected


# --- G5: long cardinals ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("积分15600分", "积分一万五千六百分"),
        ("共有12,800人", "共有一万二千八百人"),
        ("差额-15600", "差额负一万五千六百"),
        ("共1234人", "共1234人"),  # four digits: as written
        ("增长15600.5", "增长15600.5"),  # a decimal: as written
        ("占比12345%", "占比12345%"),  # a percent: as written
        ("从10000-20000", "从10000-20000"),  # a range: as written
        ("We have 15600 points.", "We have fifteen thousand six hundred points."),
        ("We have 1500 points.", "We have 1500 points."),
    ],
)
def test_long_cardinals(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_the_cardinal_threshold_and_switch() -> None:
    assert normalize("共1234人", NormalizerOptions(min_cardinal_digits=4)) == "共一千二百三十四人"
    assert normalize("积分15600分", NormalizerOptions(cardinals=False)) == "积分15600分"
    assert normalize("共3人", NormalizerOptions(min_cardinal_digits=0)) == "共三人"


# --- G5: English numeric dates and amounts --------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See you on 2024-10-23.", "See you on October twenty-third, twenty twenty-four."),
        ("Due 2024/10/23", "Due October twenty-third, twenty twenty-four"),
        ("Due 10/23/2024", "Due October twenty-third, twenty twenty-four"),
        ("Due 3/5/2024 please", "Due March fifth, twenty twenty-four please"),
        ("Due 23/10/2024", "Due October twenty-third, twenty twenty-four"),  # day first
        ("Due 31/31/2024", "Due 31/31/2024"),  # no such date
        ("Due 10/11/12", "Due 10/11/12"),  # ambiguous
        ("Open at 14:30 on May 5", "Open at 14:30 on May 5"),  # times and word dates stay
        ("日期是2024-10-23", "日期是2024-10-23"),  # Chinese: as written
    ],
)
def test_english_numeric_dates(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_english_numeric_dates_switch() -> None:
    assert normalize("Due 10/23/2024", NormalizerOptions(en_dates=False)) == "Due 10/23/2024"
    assert normalize("截止03/15/2024", DATES) == "截止二零二四年三月十五日"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A fee of $1,000.", "A fee of one thousand dollars."),
        ("Pay USD 2500 now", "Pay two thousand five hundred dollars now"),
        ("raised $1.5 million", "raised one point five million dollars"),
        ("save $500-1500 now", "save five hundred to one thousand five hundred dollars now"),
        ("It is $999.99.", "It is $999.99."),
        ("That is ¥100.", "That is ¥100."),
        ("应还¥1,280.50元", "应还¥1,280.50元"),  # Chinese: as written
    ],
)
def test_english_amounts(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_english_amounts_threshold_and_switch() -> None:
    assert normalize("A fee of $1,000.", NormalizerOptions(en_amounts=False)) == (
        "A fee of $1,000."
    )
    assert normalize("It is $50.", NormalizerOptions(en_amount_min=10)) == "It is fifty dollars."


@pytest.mark.parametrize(
    "text",
    [
        "您的手机号码尾号是5678。",
        "请回拨021-12345678转8001分机。",
        "请把截图发到我们的邮箱service@example.com。",
        "您的积分余额是15600分。",
        "Your order total is $1,280.50.",
        "Due 03/15/2024, call 1-800-555-0199 ext. 12.",
    ],
)
def test_the_character_map_covers_the_g5_readings(text: str) -> None:
    spoken, source = normalize_with_map(text)
    assert spoken == normalize(text)
    assert spoken != text
    assert len(source) == len(spoken) + 1
    assert source[-1] == len(text)
    assert list(source) == sorted(source)
    assert all(0 <= index <= len(text) for index in source)


# --- Recognition order and protection -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "options", "expected"),
    [
        ("13800138000", NormalizerOptions(numbers=True, phones=False), "13800138000"),
        ("电话13800138000", NUMBERS, "电话幺三八，零零幺三，八零零零"),
        ("单号SF13800138000", NUMBERS, "单号S F 一三八零，零一三八，零零零"),
        ("ORD20241023001", NormalizerOptions(numbers=True, ids=False), "ORD20241023001"),
        ("订单号 20241023001", NormalizerOptions(numbers=True, ids=False), "订单号 20241023001"),
        ("订单号 20241023001", NUMBERS, "订单号 二零二四，一零二三，零零一"),
        ("价格¥1,234.56", NUMBERS, "价格¥1,234.56"),
        ("会议2024-10-23", NUMBERS, "会议2024-10-23"),
        (
            "会议2024-10-23",
            NormalizerOptions(numbers=True, dates=True),
            "会议二零二四年十月二十三日",
        ),
        ("会议14:30", NUMBERS, "会议14:30"),
        ("全程5km", NUMBERS, "全程5km"),
        ("10月23日-25日", NormalizerOptions(numbers=True, dates=True), "十月二十三日-二十五日"),
        ("温度-5°C", NormalizerOptions(numbers=True, units=True), "温度负五摄氏度"),
        ("增长50%", NormalizerOptions(numbers=True, units=True), "增长百分之五十"),
    ],
)
def test_a_recognized_span_is_never_read_by_a_later_category(
    text: str, options: NormalizerOptions, expected: str
) -> None:
    assert normalize(text, options) == expected


# --- Language by context --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        ("我买了3个apple", "en", "我买了三个apple"),  # a CJK neighbour wins
        ("共3 items", "en", "共三 items"),
        ("I have 3 个", "en", "I have 三 个"),
        ("Order 3 now", "zh", "Order three now"),  # Latin on both sides beats the segment tag
        ("Qwen3 模型有 7 个版本", "zh", "Qwen3 模型有 七 个版本"),
        ("3", "zh", "三"),  # no letter at all: the segment tag decides
        ("3", "en", "three"),
        ("3", None, "three"),  # detect_lang("3") is "en"
        ("Call 400-820-8820", "zh", "Call four zero zero, eight two zero, eight eight two zero"),
        ("请拨打 400-820-8820", "en", "请拨打 四零零，八二零，八八二零"),
        ("13800138000", "zh", "幺三八，零零幺三，八零零零"),
        ("13800138000", None, "one three eight, zero zero one three, eight zero zero zero"),
    ],
)
def test_language_is_decided_by_the_nearest_letters(
    text: str, lang: Literal["zh", "en"] | None, expected: str
) -> None:
    assert normalize(text, NUMBERS, lang=lang) == expected


# --- URLs and e-mail addresses --------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "请访问 https://example.com/order/12345678 查询",
        "官网 www.novanet.com/help?id=13800138000",
        "Visit https://example.com/a?b=1&c=2, then call.",
        "See https://example.com/?to=help@novanet.cn now",  # an address inside a URL
        "See 163.com or novanet.cn/2024",
        "文件 report2024.pdf",
    ],
)
def test_urls_are_untouched(text: str) -> None:
    assert normalize(text, EVERYTHING) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("发邮件到 support123456@163.com", "发邮件到 support一二三四五六艾特一六三点com"),
        ("邮箱是zhang_san@novanet.com.cn。", "邮箱是zhang下划线san艾特novanet点com点cn。"),
        ("请发到123456789@qq.com", "请发到一二三四五六七八九艾特qq点com"),
        (
            "Email first.last2024@example.co.uk.",
            "Email first dot last two zero two four at example dot co dot uk.",
        ),
        ("mail ORD2024@example.com", "mail ORD two zero two four at example dot com"),
        (
            "Write to help-desk+vip@nova-net.io",
            "Write to help dash desk plus vip at nova dash net dot io",
        ),
    ],
)
def test_emails_are_spoken(text: str, expected: str) -> None:
    assert normalize(text) == expected
    assert normalize(text, NormalizerOptions(emails=False)) == text


def test_emails_are_never_read_by_another_category() -> None:
    text = "邮箱 13800138000@163.com"
    assert normalize(text, NormalizerOptions(emails=False, numbers=True)) == text
    assert normalize(text) == "邮箱 一三八零零一三八零零零艾特一六三点com"


@pytest.mark.parametrize(
    "text",
    [
        "请讲一个很长很长的故事。",
        "请拨打13800138000，谢谢。",
        "订单号ABC123456已发货",
        "call me at 138 0013 8000 ok",
        "我用iPhone手机，价格¥12.5，5km",
    ],
)
@pytest.mark.parametrize("spacing", [False, True])
def test_the_character_map_points_each_spoken_character_at_its_source(
    text: str, spacing: bool
) -> None:
    options = NormalizerOptions(numbers=True, money=True, units=True, spacing=spacing)
    spoken, source = normalize_with_map(text, options)
    assert spoken == normalize(text, options)
    assert len(source) == len(spoken) + 1
    assert source[-1] == len(text)
    assert list(source) == sorted(source)  # in order: a prefix maps to a prefix


def test_unchanged_text_maps_one_to_one() -> None:
    spoken, source = normalize_with_map("你好，世界。")
    assert spoken == "你好，世界。"
    assert source == (0, 1, 2, 3, 4, 5, 6)


def test_a_reading_spreads_over_the_span_it_replaces() -> None:
    text = "拨打13800138000"
    spoken, source = normalize_with_map(text)
    assert spoken.startswith("拨打幺三八")
    digits = [source[index] for index in range(2, len(spoken))]
    assert digits[0] == 2
    assert digits[-1] <= len(text) - 1
    assert len(set(digits)) > 5  # spread over the number, not piled on its first digit
