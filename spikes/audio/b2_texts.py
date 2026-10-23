"""Text sets for B2 (S3 TTS, S9 normalization probe). Written for M0; call-center flavoured.

Length is counted as the segmenter will count it (ARCHITECTURE §7): CJK characters plus Latin words
(`b2_common.units`). The brief's short / medium / long (<= 8 chars or <= 4 words / ~20 / ~60) is read
as: zh and mix by units (<= 8 / ~20 / ~60); en by words at about the same speaking time as zh
(<= 4 / ~12 / ~35 words), because 20 English *letters* would be no longer than the short texts.
"""

SWEEP = {
    "zh": {
        "short": ["您好，请问有什么事？", "好的，马上为您查询。", "请您稍等一下。"],
        "medium": [
            "您好，这里是客服中心，请问有什么可以帮您的吗？",
            "您的问题我已经记录下来了，稍后会有专人联系您。",
            "非常抱歉让您久等了，我现在帮您查一下订单状态。",
        ],
        "long": [
            "您好，感谢您致电我们的客服中心。为了更好地为您服务，本次通话可能会被录音。请问您是想查询订单、办理退换货，还是咨询其他问题呢？",
            "我已经帮您查到了，您的包裹目前正在配送途中，预计明天下午送达。如果您明天不方便收货，可以告诉我，我帮您改约其他时间。",
            "关于您反映的扣费问题，我们会在三个工作日内核实处理，结果会通过短信通知您。请您保持电话畅通，还有其他需要帮助的吗？",
        ],
    },
    "en": {
        "short": ["How can I help?", "Please hold on.", "Thank you for calling."],
        "medium": [
            "Thank you for calling customer service. How can I help you today?",
            "I have noted your request, and a specialist will call you back soon.",
            "I'm sorry for the wait. Let me check the status of your order.",
        ],
        "long": [
            "Thank you for calling our customer service center. To help us serve you better, this call may be "
            "recorded. Are you calling about an order, a return or exchange, or something else today?",
            "I found your package. It is on its way and should arrive tomorrow afternoon. If you will not be "
            "home tomorrow, just let me know and I can schedule a different delivery time for you.",
            "About the charge you mentioned, we will review it within three business days and send you the "
            "result by text message. Please keep your phone available. Is there anything else I can help with?",
        ],
    },
    "mix": {
        "short": ["好的，我发您email。", "请打开App看一下。", "您的VIP已经开通。"],
        "medium": [
            "您可以在App的个人中心里修改密码，或者联系online客服。",
            "我们会把invoice发到您注册的email邮箱，请注意查收一下。",
            "您的account目前是正常状态，可以直接login使用了。",
        ],
        "long": [
            "您好，您反映的App闪退问题我们已经收到了，技术团队正在排查。建议您先把App更新到最新version，"
            "然后清除一下cache再重新login试试。如果还有问题，可以发email给我们的support团队。",
            "关于您的refund申请，我们已经审核通过了，款项会在三到五个工作日内退回到您的credit card。"
            "到账以后系统会自动发一条SMS提醒您，您也可以在App的订单页面查看进度。",
            "这次的promotion活动只针对VIP会员，您目前是普通会员，如果想参加的话，可以先在App里upgrade一下"
            "会员等级，升级以后就能看到活动入口了，有问题随时联系我们。",
        ],
    },
}

# Quality guardrail (S3.5): 10 sentences, 10 runs each.
GUARDRAIL = [
    ("zh", "您好，感谢您致电客服中心，请问有什么可以帮您的吗？"),
    ("zh", "您的订单已经发货了，预计明天下午就能送到您手上。"),
    ("zh", "非常抱歉给您带来不便，我们会尽快为您处理退款。"),
    ("zh", "请您保持电话畅通，我们的工作人员稍后会联系您。"),
    ("zh", "为了保障您的账户安全，请不要把验证码告诉任何人。"),
    ("en", "Thank you for calling customer service. How can I help you today?"),
    ("en", "Your order has shipped and should arrive tomorrow afternoon."),
    ("en", "We are sorry for the trouble, and we will process your refund soon."),
    ("en", "Please keep your phone on. Our staff will call you back shortly."),
    ("en", "For your security, please do not share your verification code with anyone."),
]

# Kokoro vs Qwen3-TTS (S3.6, D13): 20 sentences per language.
D13 = {
    "zh": [
        "您好，这里是客服中心，请问有什么可以帮您？",
        "好的，我马上帮您查询一下订单状态。",
        "非常抱歉让您久等了，感谢您的耐心。",
        "您的包裹已经到达配送站，今天下午会送到。",
        "请问您是想办理退货还是换货呢？",
        "您反映的问题我已经记录下来了，会尽快处理。",
        "请您保持电话畅通，稍后会有专人联系您。",
        "退款会原路返回到您的支付账户。",
        "为了保障您的账户安全，请不要向他人透露验证码。",
        "您的会员积分可以在下次购物时抵扣现金。",
        "这个问题需要转接到技术部门，请您稍等。",
        "我们的工作人员会在明天上午上门安装。",
        "如果您对这次服务满意，请给我们一个好评。",
        "很抱歉，这款商品目前暂时缺货。",
        "您可以在订单详情页面申请开具发票。",
        "本次通话可能会被录音，以便提升服务质量。",
        "请您核对一下收货地址是否正确。",
        "系统显示您的快递已经签收了。",
        "感谢您的来电，祝您生活愉快，再见。",
        "您的投诉我们已经受理，处理结果会短信通知您。",
    ],
    "en": [
        "Hello, this is customer service. How can I help you today?",
        "Sure, let me check the status of your order right away.",
        "I'm sorry to keep you waiting. Thank you for your patience.",
        "Your package has arrived at the local depot and will be delivered this afternoon.",
        "Would you like to return the item or exchange it?",
        "I have recorded your issue, and we will handle it as soon as possible.",
        "Please keep your phone on. A specialist will call you back shortly.",
        "The refund will go back to your original payment method.",
        "For your security, please never share your verification code with anyone.",
        "You can use your reward points as a discount on your next purchase.",
        "This needs to go to our technical team. Please hold for a moment.",
        "Our technician will come to install it tomorrow morning.",
        "If you are happy with our service today, please leave us a good review.",
        "I'm sorry, this item is currently out of stock.",
        "You can request an invoice on the order details page.",
        "This call may be recorded to help us improve our service.",
        "Could you please confirm that your delivery address is correct?",
        "Our system shows that your parcel has already been signed for.",
        "Thank you for calling, and have a nice day. Goodbye.",
        "We have received your complaint, and we will text you the result.",
    ],
    "mix": [
        "请您打开App，在首页点击客服入口。",
        "我们已经把确认邮件发到您的email了。",
        "您的VIP会员下个月就要到期了。",
        "您可以把截图发给我们的online客服。",
        "请先连接Wi-Fi再下载这个update。",
        "您的refund申请已经审核通过了。",
        "这个promotion活动只针对新用户。",
        "请您先logout，然后重新login一下。",
        "您的account目前处于正常状态。",
        "我们会把invoice发到您登记的邮箱。",
        "这款iPhone目前有现货，可以当天发货。",
        "您的order已经安排发货了，请留意物流信息。",
        "请问您用的是Android手机还是iPhone？",
        "系统显示您的password已经修改成功。",
        "您可以在settings里面关闭消息推送。",
        "我们的support团队会在今天之内回复您。",
        "这个bug我们已经反馈给技术部门了。",
        "您的credit card额度暂时不够，可以换一张卡试试。",
        "请您在App里重新upload一下身份证照片。",
        "好的，没问题，我现在帮您check一下。",
    ],
}

# S9 (G5): raw text as an LLM would write it, and the intended spoken form written out in words.
# Phone numbers use 幺 for 1, the usual telephone reading (configurable in the planned normalizer).
S9 = [
    # --- cardinal, decimal, percent, ordinal
    {"id": "num1", "cat": "number", "text": "您本月一共有3笔订单待处理。", "spoken": "您本月一共有三笔订单待处理。"},
    {"id": "num2", "cat": "number", "text": "当前排队人数为128人，请您耐心等待。", "spoken": "当前排队人数为一百二十八人，请您耐心等待。"},
    {"id": "num3", "cat": "number", "text": "您的积分余额是15600分。", "spoken": "您的积分余额是一万五千六百分。"},
    {"id": "num4", "cat": "number", "text": "本次活动全场8.5折。", "spoken": "本次活动全场八点五折。"},
    {"id": "num5", "cat": "number", "text": "您的好评率达到了99.6%。", "spoken": "您的好评率达到了百分之九十九点六。"},
    {"id": "num6", "cat": "number", "text": "您是今天第2位来电的客户。", "spoken": "您是今天第二位来电的客户。"},
    # --- money
    {"id": "money1", "cat": "money", "text": "您本期账单应还金额为¥1,280.50。", "spoken": "您本期账单应还金额为一千二百八十元五角。"},
    {"id": "money2", "cat": "money", "text": "您充值的100元已经到账。", "spoken": "您充值的一百元已经到账。"},
    {"id": "money3", "cat": "money", "text": "这单的运费是12.5元。", "spoken": "这单的运费是十二点五元。"},
    {"id": "money4", "cat": "money", "text": "该套餐的月费为$29.99。", "spoken": "该套餐的月费为二十九点九九美元。"},
    {"id": "money5", "cat": "money", "text": "您有一张满200减30的优惠券。", "spoken": "您有一张满二百减三十的优惠券。"},
    {"id": "money6", "cat": "money", "text": "违约金为合同金额的5%，即2500元。", "spoken": "违约金为合同金额的百分之五，即两千五百元。"},
    # --- dates and times
    {"id": "date1", "cat": "datetime", "text": "您的预约时间是2024年3月15日下午2:30。", "spoken": "您的预约时间是二零二四年三月十五日下午两点三十分。"},
    {"id": "date2", "cat": "datetime", "text": "活动截止到12月31日24:00。", "spoken": "活动截止到十二月三十一日二十四点。"},
    {"id": "date3", "cat": "datetime", "text": "我们的服务时间是每天9:00到21:00。", "spoken": "我们的服务时间是每天九点到二十一点。"},
    {"id": "date4", "cat": "datetime", "text": "您的新套餐将于2025-01-01生效。", "spoken": "您的新套餐将于二零二五年一月一日生效。"},
    {"id": "date5", "cat": "datetime", "text": "请在今天15:45之前完成支付。", "spoken": "请在今天十五点四十五分之前完成支付。"},
    {"id": "date6", "cat": "datetime", "text": "您的账单日是每月5号。", "spoken": "您的账单日是每月五号。"},
    # --- phone numbers
    {"id": "phone1", "cat": "phone", "text": "您可以拨打我们的客服热线400-820-8820。", "spoken": "您可以拨打我们的客服热线四零零，八二零，八八二零。"},
    {"id": "phone2", "cat": "phone", "text": "请拨打银行客服电话95588。", "spoken": "请拨打银行客服电话九五五八八。"},
    {"id": "phone3", "cat": "phone", "text": "您登记的手机号是138 0013 8000，对吗？", "spoken": "您登记的手机号是幺三八，零零幺三，八零零零，对吗？"},
    {"id": "phone4", "cat": "phone", "text": "您的手机号码尾号是5678。", "spoken": "您的手机号码尾号是五六七八。"},
    {"id": "phone5", "cat": "phone", "text": "我们的办公电话是010-62345678。", "spoken": "我们的办公电话是零幺零，六二三四五六七八。"},
    {"id": "phone6", "cat": "phone", "text": "请回拨021-12345678转8001分机。", "spoken": "请回拨零二幺，幺二三四五六七八，转八零零幺分机。"},
    {"id": "phone7", "cat": "phone", "text": "您留的联系电话是13912345678。", "spoken": "您留的联系电话是幺三九，幺二三四，五六七八。"},
    # --- order and tracking IDs
    {"id": "id1", "cat": "id", "text": "您的订单号是A20241015001。", "spoken": "您的订单号是A，二零二四，幺零幺五，零零幺。"},
    {"id": "id2", "cat": "id", "text": "您的快递单号是SF1234567890。", "spoken": "您的快递单号是S F，幺二三四，五六七八，九零。"},
    {"id": "id3", "cat": "id", "text": "您的工单编号为TK-2024-0789。", "spoken": "您的工单编号为T K，二零二四，零七八九。"},
    {"id": "id4", "cat": "id", "text": "您的验证码是386425，五分钟内有效。", "spoken": "您的验证码是三八六，四二五，五分钟内有效。"},
    {"id": "id5", "cat": "id", "text": "请提供您的会员卡号VIP8866。", "spoken": "请提供您的会员卡号V I P，八八六六。"},
    {"id": "id6", "cat": "id", "text": "退货单号是RT20241102A3。", "spoken": "退货单号是R T，二零二四，幺幺零二，A，三。"},
    # --- units
    {"id": "unit1", "cat": "unit", "text": "包裹重量为2.5kg。", "spoken": "包裹重量为二点五公斤。"},
    {"id": "unit2", "cat": "unit", "text": "您本月的流量已经用了15GB。", "spoken": "您本月的流量已经用了十五G B。"},
    {"id": "unit3", "cat": "unit", "text": "我们的配送范围是5km以内。", "spoken": "我们的配送范围是五公里以内。"},
    {"id": "unit4", "cat": "unit", "text": "这款手机的屏幕是6.1英寸。", "spoken": "这款手机的屏幕是六点一英寸。"},
    {"id": "unit5", "cat": "unit", "text": "空调温度建议设定在26℃。", "spoken": "空调温度建议设定在二十六摄氏度。"},
    {"id": "unit6", "cat": "unit", "text": "您家的宽带速率是500Mbps。", "spoken": "您家的宽带速率是五百兆。"},
    # --- mixed zh/en
    {"id": "mix1", "cat": "mixed", "text": "请在App里点击“我的订单”查看物流信息。", "spoken": "请在App里点击我的订单查看物流信息。"},
    {"id": "mix2", "cat": "mixed", "text": "您可以登录官网www.example.com查询。", "spoken": "您可以登录官网三W点example点com查询。"},
    {"id": "mix3", "cat": "mixed", "text": "请把截图发到我们的邮箱service@example.com。", "spoken": "请把截图发到我们的邮箱service艾特example点com。"},
    {"id": "mix4", "cat": "mixed", "text": "您的iPhone 15需要送修吗？", "spoken": "您的iPhone十五需要送修吗？"},
    {"id": "mix5", "cat": "mixed", "text": "请先连接Wi-Fi再下载更新包。", "spoken": "请先连接Wi-Fi再下载更新包。"},
    {"id": "mix6", "cat": "mixed", "text": "我们支持PayPal和Apple Pay付款。", "spoken": "我们支持PayPal和Apple Pay付款。"},
]

# S9 extra (not in the brief's count): the same categories in English, synthesized with an English
# speaker and lang_code english, to see whether English segments need the normalizer too.
S9_EN = [
    {"id": "en_money", "cat": "money", "text": "Your order total is $1,280.50.",
     "spoken": "Your order total is one thousand two hundred eighty dollars and fifty cents."},
    {"id": "en_phone", "cat": "phone", "text": "Please call us at 1-800-555-0199.",
     "spoken": "Please call us at one, eight hundred, five five five, zero one nine nine."},
    {"id": "en_date", "cat": "datetime", "text": "Your appointment is on 03/15/2024 at 2:30 PM.",
     "spoken": "Your appointment is on March fifteenth, twenty twenty-four, at two thirty PM."},
    {"id": "en_id", "cat": "id", "text": "Your order number is A20241015001.",
     "spoken": "Your order number is A, two zero two four, one zero one five, zero zero one."},
    {"id": "en_unit", "cat": "unit", "text": "The package weighs 2.5 kg.",
     "spoken": "The package weighs two point five kilograms."},
    {"id": "en_percent", "cat": "number", "text": "You get 15% off orders over $200.",
     "spoken": "You get fifteen percent off orders over two hundred dollars."},
]

# S9 calibration: a plausible WRONG reading of the key item, written out in words. If the ASR writes the
# spoken (intended) form and this alternative differently, it can tell the two readings apart, and a raw
# run can be classified by which one it matches. If it writes both the same way, the item needs a human
# listening check.
S9_ALT = {
    "num2": "当前排队人数为一二八人，请您耐心等待。",
    "num3": "您的积分余额是一五六零零分。",
    "date1": "您的预约时间是两千零二十四年三月十五日下午二比三十。",
    "date2": "活动截止到十二月三十一日二十四比零零。",
    "date3": "我们的服务时间是每天九比零零到二十一比零零。",
    "date4": "您的新套餐将于二零二五减零一减零一生效。",
    "date5": "请在今天十五比四十五之前完成支付。",
    "phone1": "您可以拨打我们的客服热线四百，八百二十，八千八百二十。",
    "phone2": "请拨打银行客服电话九万五千五百八十八。",
    "phone3": "您登记的手机号是一百三十八，十三，八千，对吗？",
    "phone4": "您的手机号码尾号是五千六百七十八。",
    "phone5": "我们的办公电话是十，六千二百三十四万五千六百七十八。",
    "phone6": "请回拨二十一，一千二百三十四万五千六百七十八，转八千零一分机。",
    "phone7": "您留的联系电话是一百三十九亿一千二百三十四万五千六百七十八。",
    "id1": "您的订单号是A两百零二亿四千一百零一万五千零一。",
    "id2": "您的快递单号是SF十二亿三千四百五十六万七千八百九十。",
    "id3": "您的工单编号为TK，两千零二十四，七百八十九。",
    "id4": "您的验证码是三十八万六千四百二十五，五分钟内有效。",
    "id5": "请提供您的会员卡号VIP八千八百六十六。",
    "id6": "退货单号是RT两千零二十四万一千一百零二A三。",
}
