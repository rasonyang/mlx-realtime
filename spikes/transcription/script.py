"""The call scripts of the corpus (issue #2): 24 two-party customer-service calls, written to
``calls.jsonl`` (versioned; the audio is generated from it by ``synth.py``).

    uv run --locked --project spikes/transcription python spikes/transcription/script.py

A call is a list of events. Utterance events (``kind: "utt"``) carry the leg (``agent`` or
``customer``), the reference ``text`` (what the speaker says, terms as written), ``tts`` (what the
TTS reads when it differs: codes are spoken character by character), the voice, and ``gap_s``:
the pause between the end of the previous utterance (of either leg) and the start of this one;
negative means the speaker starts before the other has finished (crosstalk). Long speeches
(``long: true``) are 30-60 s of text that ``synth.py`` speaks sentence by sentence.
Other events: ``hold`` (``leg_audio``: ``silence`` on both legs, ``music`` on the customer leg),
``dtmf`` (the customer presses digits) and ``silence`` (a stretch of one or both legs).

Per call: the line noise of each leg (``noise_rms``, a 16-bit RMS before G.711: 20 clean, 80
moderate, 200 noisy), the voices, and the planned features. Every text is checked against the
words ASR writes differently from the script (numbers above nine: see ``FORBIDDEN``).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "asr_context"))  # the term inventory of issue #1
import terms as T  # noqa: E402

SEED = 20261024
N_CALLS = 24
AGENT_VOICES = ["vivian", "serena"]
CUSTOMER_VOICES = ["uncle_fu", "aiden"]
CN_DIGIT = "零一二三四五六七八九"
FORBIDDEN = "十百千万两"  # ASR writes them as digits: a spurious error in the reference

# slots: {b} brand, {p} product, {j} jargon, {c} an order code, {c2} another
TOPICS: dict[str, dict[str, list]] = {
    "broadband": {
        "open": ["你好，我家的宽带从昨天晚上开始就一直断断续续的，想报修。",
                 "喂，是 {b} 客服吗？我家 {p} 的网络老是掉线，你们能处理吗？",
                 "你好，我家网速特别慢，看视频都卡，是不是线路有问题？"],
        "ex": [
            ("路由器上的指示灯一直是红色的。", "红灯说明设备没有连上我们的网络。请您先把路由器的电源拔掉，等半分钟再重新插上，看看指示灯会不会变成绿色。"),
            ("我试过了，还是红灯。", "好的，那我这边帮您远程检测一下线路。请问您家的宽带账号是手机号码吗？"),
            ("是的，就是我现在打电话的这个号码。", "我查到了，您的线路在昨天晚上有一次异常，现在信号不太稳定，需要工程师上门看一下。"),
            ("工程师什么时候能来？", "最快是明天上午，九点到中午之间，您看方便吗？"),
            ("明天上午我在家，可以。", "好的，我已经为您预约了。维修单号是 {c}，工程师出发前会提前给您打电话。"),
            ("上门要收费吗？", "宽带线路故障的上门维修是免费的，如果是您自己的设备损坏才会收取检测费用。"),
            ("那我这几天没法上网，有什么补偿吗？", "您的账号这个月连续断网超过三天，我会提交申请，按天数返还相应的宽带费用，一般三个工作日内到账。"),
            ("好吧，谢谢你。", "不客气，还有其他可以帮您的吗？"),
        ],
        "long_agent": ["为了让您更清楚上门维修的流程，我给您详细说明一下。", "工程师到达以后，会先检查入户的光纤和光猫的状态。",
                       "如果发现光纤本身有损坏，会当场更换，整个过程大概需要一个小时。", "如果问题出在路由器，也会为您测试信号强度，并且调整摆放的位置。",
                       "维修结束以后，工程师会请您在手机上确认服务结果，您可以对服务进行评价。", "如果三天之内同样的问题再次出现，您可以直接拨打我们的热线，我们会免费二次上门。",
                       "另外，维修期间如果您需要临时的网络，可以在 App 里领取一张流量包，当月有效。"],
        "long_customer": ["我跟你说一下情况，这个问题其实已经持续一个多星期了。", "最开始只是晚上看视频的时候偶尔会卡一下，我以为是网站的问题。",
                          "后来发现连着路由器的手机、电脑，都会隔一会儿就掉线，需要重新连接。", "上个周末我还专门给你们打过电话，当时的客服让我重启路由器，说观察一下。",
                          "我重启了好几次，好一阵子又坏了，我家里孩子上网课都受到了影响。", "所以这次我希望你们一次把问题彻底解决，不要再让我反复折腾了。"],
    },
    "plan": {
        "open": ["你好，我想了解一下现在有什么套餐，我想换一个。",
                 "喂，我现在用的是 {p}，感觉流量不太够用，能换个大一点的吗？",
                 "你好，{b} 最近是不是出了新的套餐？我想问问。"],
        "ex": [
            ("我每个月流量都不够用，大概要多一倍。", "明白。您可以考虑 5G 尊享 299，流量比现在的套餐多很多，还包含视频会员。"),
            ("那个套餐每个月多少钱？", "套餐名字里的数字就是月费，包含充足的通话和流量。"),
            ("有没有便宜一点的？", "5G 畅享 199 也不错，流量够日常使用，适合您这样的用户。"),
            ("这些套餐的网速有区别吗？", "网速是一样的，区别主要在流量和附加权益。尊享套餐可以免费使用 Orbix 的云盘空间。"),
            ("换了套餐以后，原来的合约怎么办？", "您原来的合约不受影响，新套餐从下个月一号开始生效，当月仍按原套餐计费。"),
            ("好，那就换成 5G 畅享 199 吧。", "好的，我现在帮您办理。需要您确认一下，办理以后会有一条短信，请您回复确认。"),
            ("短信收到了，我回复了。", "收到了，办理成功。新套餐下个月一号生效，您可以在 App 里查看详情。"),
        ],
        "long_agent": ["我给您介绍一下这几个套餐的区别。", "5G 畅享 199 包含充足的国内流量，通话时长足够日常使用，并且赠送一个视频会员。",
                       "5G 尊享 299 则包含更多的流量，还可以和家人共享，最多可以添加三位家庭成员。", "这些套餐都支持 eSIM，您可以不用换卡，在手机里直接开通。",
                       "超出套餐流量以后，网速不会断，但是会降低，您也可以随时购买加油包。", "套餐的变更需要您本人办理，办理成功后会收到短信通知。"],
        "long_customer": ["是这样的，我平时在外面跑业务，主要靠手机上网，开会、发文件都用流量。", "现在的套餐一到月底就不够用，要么去买加油包，要么就得找地方连 Wi-Fi。",
                          "我同事用的是你们家别的套餐，说流量很多，价格也差不多，所以我才想来问问。", "我希望换了以后，原来的手机号码和卡都不用变，也不用跑营业厅。"],
    },
    "esim": {
        "open": ["你好，我新买了手机，想开通 eSIM，应该怎么操作？",
                 "喂，我想把现在的号码转到 eSIM 上，可以吗？",
                 "你好，我的手机支持 eSIM，你们这边怎么开通？"],
        "ex": [
            ("我现在用的是实体 SIM 卡。", "可以的。您可以在 App 里选择号码转 eSIM，不需要去营业厅。"),
            ("需要准备什么吗？", "您需要准备身份证，并保证手机连着 Wi-Fi，因为开通过程中会下载一个二维码配置。"),
            ("二维码在哪里看？", "在 App 的 eSIM 页面，点击生成，二维码有效期是半小时，过期需要重新生成。"),
            ("我扫了，但是手机提示无法添加。", "请您检查一下手机系统是不是最新版本，有的机型需要先做一次 OTA 升级才能识别。"),
            ("我现在去升级，等一下再打给你们。", "好的，不着急。如果升级以后还是不行，您把手机型号告诉我，我帮您核对是否支持。"),
            ("手机型号是 {p} 配套的那款。", "这个型号是支持的。我先帮您把二维码重新发送到您的邮箱，您可以直接在手机上打开。"),
            ("收到了，好像可以了。", "太好了。激活以后，实体卡会自动停用，请您确认手机信号是不是正常。"),
        ],
        "long_agent": ["我给您说明一下 eSIM 开通以后需要注意的几点。", "第一，开通成功以后，原来的实体 SIM 卡会失效，请不要再把它插回手机里。",
                       "第二，如果手机丢失，可以通过 App 立刻挂失 eSIM，别人无法再使用。", "第三，换手机的时候，需要重新生成二维码，旧手机上的配置会被清除。",
                       "第四，出国使用的时候，eSIM 同样支持漫游，您可以提前在 App 里开通国际流量。", "如果开通过程中遇到任何问题，随时联系我们，我们全天都有人值守。"],
        "long_customer": ["我之前一直担心 eSIM 不太稳定，因为身边有朋友开通以后信号变差了。", "而且我经常出差，要在不同的国家使用，如果能一个手机里装三个号码就最好了。",
                          "我现在的手机是双卡的，一个卡槽放工作号码，另一个想用 eSIM 放私人号码。", "所以我想确认一下，你们的 eSIM 能不能和别家的实体卡同时使用，会不会冲突。"],
    },
    "bill": {
        "open": ["你好，我这个月的账单好像不对，比平时多了很多。",
                 "喂，我看了一下话费，多了一笔我不认识的扣费，你们帮我查一下。",
                 "你好，{b} 这个月的扣费怎么回事？我想核对一下账单。"],
        "ex": [
            ("多出来的是一笔增值业务的费用。", "好的，我帮您查一下。请问是哪一天扣的？"),
            ("好像是上个月底。", "我看到了，上个月底确实有一笔视频会员的续费，是您之前开通的自动续费。"),
            ("我没有开通过啊。", "系统显示是三个月前通过 App 开通的，当时有一个免费体验，到期以后自动续费了。"),
            ("那这个能退吗？", "因为您没有使用过这个会员，我可以帮您申请退还这一个月的费用，同时关闭自动续费。"),
            ("好的，麻烦你帮我关掉。", "已经关闭了。退款会在五个工作日内原路退回到您的话费账户。"),
            ("还有别的我不知道的自动续费吗？", "我帮您查了一下，目前没有其他的了。以后您可以在 App 的我的订阅里面查看和管理。"),
            ("明白了，谢谢。", "不客气，请问还有什么可以帮您的吗？"),
        ],
        "long_agent": ["我把您这个月账单的构成给您说明一下。", "首先是套餐月费，这是固定的，每个月都一样。", "其次是超出套餐的部分，您这个月通话没有超出，流量也没有超出。",
                       "然后是增值业务，包括视频会员和云盘空间，这几项都是您自己开通的。", "最后是税费和优惠抵扣，优惠已经体现在最终的应付金额里面了。",
                       "如果您对其中任何一项有疑问，我都可以帮您查看详细的使用记录。"],
        "long_customer": ["我一直都是按时缴费的，每个月的话费差不多都是固定的，这次突然多了一笔，我心里很不踏实。", "我最近也没有办理过什么新业务，也没有打过国际电话。",
                          "我老婆用的是家庭共享的副卡，不知道是不是她开通了什么服务，但是她也说没有。", "你们能不能把这笔费用的明细发给我，让我自己再核对一下，我担心是不是被盗用了。"],
    },
    "logistics": {
        "open": ["你好，我在你们这里买的 Lumio 智能音箱，到现在还没有收到。",
                 "喂，我的订单 {c} 显示已经发货三天了，怎么还没到？",
                 "你好，我想查一下我的订单到哪里了。"],
        "ex": [
            ("订单号是 {c}。", "好的，请稍等，我帮您查询一下。我看到包裹昨天到达了您所在城市的分拨中心。"),
            ("那为什么还没有送出来？", "最近快递量比较大，可能会延迟一天。预计明天下午会安排派送。"),
            ("能不能改一下收货地址？", "可以的。包裹还没有派送，我可以帮您修改。请您告诉我新的详细地址。"),
            ("改到我公司，地址我发短信给你们。", "好的，您可以直接回复订单短信，我这边收到以后会立刻处理。"),
            ("改完以后会不会更慢？", "同城修改地址一般不会影响时效，改好以后我会给您发一条确认短信。"),
            ("好，那我等你们的短信。", "好的，还有什么需要帮助的吗？"),
        ],
        "long_agent": ["关于您的包裹，我把物流的情况完整地给您说明一下。", "您的订单是三天前从华东仓库发出的，经过了三个中转站，目前停留在本地的分拨中心。",
                       "按照我们的承诺，在发货以后的五个工作日内都属于正常送达时间。", "如果超过这个时间还没有收到，您可以申请补偿，我们会赠送一张优惠券。",
                       "另外包裹到达以后请您当面检查，确认外包装没有破损再签收。", "如果有任何问题，可以直接拒收，我们会安排免费的退货。"],
        "long_customer": ["我是为了送给父母才买的，本来想赶在他们生日之前送到，结果拖到现在还没有消息。", "下单的时候页面写的是三天内送达，我专门看了的。",
                          "而且发货以后，物流信息有三天都没有更新，我给快递员打电话也没人接。", "我想知道，如果这个周末之前还收不到，能不能直接退款，我再去别的地方买。"],
    },
    "return": {
        "open": ["你好，我买的 Veltra 路由器有问题，想申请退换货。",
                 "喂，我收到的 {p} 是坏的，开不了机，怎么处理？",
                 "你好，我想退货，上周买的东西不太满意。"],
        "ex": [
            ("通电以后指示灯完全不亮。", "这种情况可能是设备本身的故障。您还在七天无理由退换的时间内，可以选择换货。"),
            ("我想换一个新的。", "好的，我帮您创建换货单，单号是 {c}。请您保留好原来的包装。"),
            ("换货要我自己寄回去吗？", "不需要，我们会安排快递员上门取件，同时寄出新的设备，您只需要在家等待就可以。"),
            ("快递员什么时候来？", "最快明天下午，快递员会提前一个小时给您打电话。"),
            ("新的设备多久能到？", "新设备会在取件以后的第二天发出，预计三天内送到。"),
            ("运费是我出吗？", "质量问题造成的换货，运费全部由我们承担，您不需要支付任何费用。"),
            ("好的，那就这样吧。", "好的，换货单已经生成，您会收到一条短信，里面有取件的详细信息。"),
        ],
        "long_agent": ["我给您说明一下退换货的详细规则。", "自收货之日起七天内，商品没有使用痕迹，包装完整，可以申请无理由退货。",
                       "如果是质量问题，半个月内可以换货，超过这个时间会交给售后中心检测。", "退货的款项会在我们收到商品并验货以后，三个工作日内退回到原支付账户。",
                       "如果您是用 Zentra 支付的，款项会直接退到您的 Zentra 账户里面。", "赠品需要和主商品一起寄回，缺失的话，会从退款里面按价格扣除。"],
        "long_customer": ["这个路由器我是上个星期搬家的时候专门买的，因为之前那个信号不好。", "买回来以后接上电源，指示灯闪了一下就灭了，再怎么按按钮也没有反应。",
                          "我换了好几个插座，也试过换电源线，还是一样，所以我基本可以确定是设备本身的问题。", "我现在家里完全没有网络，孩子要上网课，很着急，希望你们能尽快处理。"],
    },
    "wifi": {
        "open": ["你好，我家 Wi-Fi 信号不好，有的房间根本连不上。",
                 "喂，听说你们有 Mesh 路由器，我家面积比较大，想问问。",
                 "你好，我换了 {p} 以后，信号反而变差了，这是怎么回事？"],
        "ex": [
            ("我家是三室一厅，路由器放在客厅。", "这种户型建议使用 Mesh 路由器组网，在卧室再放一个节点，信号就能覆盖全屋。"),
            ("Mesh 路由器要多少钱？", "我们有三只装和四只装的套装，具体价格您可以在 App 的商城里面查看，现在有活动。"),
            ("能和我现在的路由器一起用吗？", "最好使用同一品牌的设备，这样可以统一管理，切换房间的时候也不会断网。"),
            ("安装复杂吗？", "不复杂，插上电，用 App 扫码，大约五分钟就可以完成。如果您需要，我们也可以安排工程师上门安装。"),
            ("上门安装收费吗？", "购买套装包含一次免费上门安装，单独预约的话需要收取少量的服务费。"),
            ("好，我先看看商城吧。", "好的，有任何问题随时联系我们。"),
        ],
        "long_agent": ["我给您一些改善 Wi-Fi 信号的小建议。", "第一，路由器尽量放在家里中间的位置，离地面一米以上，周围不要有金属物品。",
                       "第二，尽量不要和微波炉、蓝牙音箱放得太近，它们会干扰信号。", "第三，如果您的手机支持 Wi-Fi 6 或者 Wi-Fi 7，要在路由器设置里打开对应的频段。",
                       "第四，距离比较远的设备，建议连接五赫兹之外的频段，穿墙能力更强。", "如果这些调整以后效果仍然不理想，再考虑增加 Mesh 节点。"],
        "long_customer": ["我家是复式的，一楼是客厅和厨房，二楼是卧室和书房，路由器放在一楼的电视柜后面。", "到了二楼的书房，信号就只剩一格，视频会议的时候经常卡住。",
                          "我自己在网上买过信号放大器，用了以后效果也不明显，反而有时候会掉线。", "所以我想听听你们专业的建议，到底是换路由器，还是重新布一下网线。"],
    },
    "tv": {
        "open": ["你好，我家的 SkyWave 电视盒子打不开了，一直卡在开机画面。",
                 "喂，我的 IPTV 没有信号，是不是欠费了？",
                 "你好，{b} 的电视盒子怎么看不了回放了？"],
        "ex": [
            ("开机以后一直显示正在加载。", "这种情况多半是网络连接异常。请您检查盒子的网线或者 Wi-Fi 是否连接正常。"),
            ("网线是插着的。", "好的，那请您长按遥控器的设置键五秒，选择恢复出厂设置，看看能不能重新加载。"),
            ("恢复以后需要重新登录吗？", "需要的。账号是您的宽带账号，密码在您开通的时候收到的短信里面。"),
            ("我找不到那条短信了。", "没关系，我可以帮您重置密码，新密码会发到您的手机上。"),
            ("好了，现在可以了。", "太好了。如果之后再出现类似的问题，您也可以在 App 里面直接做远程诊断。"),
            ("那回放功能怎么用？", "在直播的时候按遥控器的左键，就可以回看过去七天的节目。"),
        ],
        "long_agent": ["我给您介绍一下电视盒子的几个常用功能。", "首先是回看，直播过程中按左键，可以拖动进度条回看七天内的节目。",
                       "其次是投屏，手机和盒子连在同一个网络里，在 App 里点击投屏就可以了。", "再次是语音搜索，按住遥控器上的麦克风键说出节目名称，就能直接找到。",
                       "最后是儿童模式，可以限制每天的观看时间，也可以设置家长密码。", "如果您还想了解更多，我可以把使用手册的链接发到您的手机上。"],
        "long_customer": ["我家的电视盒子已经用了三年多，一直都挺好的，就是最近半个月开始越来越慢。", "开机要等好几分钟，进了主页以后点什么都要转圈。",
                          "之前有一次直接黑屏了，拔掉电源重新插才恢复，后来又出现了三次。", "我怀疑是不是盒子老化了，如果需要换新的话，能不能以旧换新，价格会不会便宜一点。"],
    },
    "cloud": {
        "open": ["你好，我的 Orbix 云盘空间满了，想扩容。",
                 "喂，我在 Orbix 上传的照片不见了，能帮我找回来吗？",
                 "你好，我想问一下 NovaCloud 和 Orbix 有什么区别。"],
        "ex": [
            ("我现在的空间只剩下一点了。", "您可以在 App 的云盘页面选择扩容，有按月和按年几种方式，按年更划算。"),
            ("按年的话是怎么收费的？", "按年购买相当于打八折，并且送一个月的使用时间。"),
            ("扩容以后原来的文件会受影响吗？", "不会的，您的所有文件都会保留，只是可用的空间变大了。"),
            ("那我删掉的照片能恢复吗？", "云盘有回收站，删除的文件会保留一个月，您可以在回收站里面还原。"),
            ("我找到了，谢谢。", "不客气。建议您打开自动备份功能，这样新拍的照片会自动上传。"),
            ("自动备份会不会耗很多流量？", "您可以设置为仅在 Wi-Fi 下备份，这样就不会消耗手机流量了。"),
        ],
        "long_agent": ["我给您介绍一下云盘的几项安全措施。", "您的文件在上传和存储的时候都会加密，只有您自己的账号可以访问。", "我们建议您开启二次验证，登录的时候除了密码，还需要手机验证码。",
                       "如果更换了手机，只需要登录同一个账号，所有文件都会同步下来。", "共享文件的时候，您可以设置链接的有效期和提取码，到期以后自动失效。",
                       "如果您忘记了密码，可以通过绑定的手机号码找回。"],
        "long_customer": ["我平时喜欢拍照，孩子的每一个成长瞬间我都想留下来，所以手机里的照片特别多。", "以前都存在手机里，后来手机内存不够了，才开始用云盘备份。",
                          "现在云盘也快满了，我又不舍得删，所以想问问有没有更大的容量。", "另外我还想确认一下，如果以后我不续费了，这些照片会不会被删除，我很担心。"],
    },
    "pay": {
        "open": ["你好，我用 Zentra 交话费的时候一直失败，提示支付异常。",
                 "喂，我的 Zentra 被扣了三次款，但是话费只充了一次。",
                 "你好，我想用 Zentra 绑定自动缴费，应该怎么设置？"],
        "ex": [
            ("提示的是银行卡限额。", "可能是您银行卡的单笔限额比较低。您可以换一张卡，或者在 Zentra 里调整限额。"),
            ("我换了一张卡也不行。", "那我帮您查一下交易记录。请问您大概是什么时候支付的？"),
            ("就是刚才，大概五分钟前。", "我看到了，有一笔交易处理中，状态没有更新，所以显示支付异常。通常半小时内会自动到账。"),
            ("如果没有到账怎么办？", "如果超过一个小时还没有到账，款项会自动原路退回，您也可以随时联系我们人工核实。"),
            ("好的，那我再等一等。", "好的，到账以后我们会给您发送短信提醒。"),
            ("自动缴费怎么设置？", "在 App 里选择自动缴费，绑定 Zentra，设置余额低于多少时自动充值就可以了。"),
        ],
        "long_agent": ["关于支付安全，我再多说几句。", "我们不会通过电话或者短信向您索要银行卡密码、验证码，请您一定提高警惕。", "如果您收到了陌生链接，请不要点击，可以直接转发给我们的客服核实。",
                       "使用 Zentra 支付的时候，每一笔交易您都会收到通知，如有异常请立刻冻结账户。", "冻结以后，所有的自动缴费都会暂停，您可以随时联系我们解冻。"],
        "long_customer": ["前几天有人给我打电话，自称是你们公司的工作人员，说我的话费账户有异常，让我提供验证码。", "我当时就觉得不对劲，直接把电话挂了。",
                          "后来我又收到一条短信，里面有一个链接，说是什么积分兑换，我没有点。", "我想问问，这些是不是你们发的，有没有可能我的信息被泄露了，我需要做什么。"],
    },
    "network": {
        "open": ["你好，我手机连着 Wi-Fi 能上网，用流量就不行，提示 APN 有问题。",
                 "喂，我想设置 VPN，公司要求的，但是总是连不上。",
                 "你好，我家的 DNS 好像有问题，有些网页打不开。"],
        "ex": [
            ("我换了新手机，什么都是默认的。", "那应该是 APN 设置没有同步。请您进入设置，找到移动网络，选择接入点名称，选择我们的默认配置。"),
            ("没有看到你们的配置。", "没关系，您可以手动新建一个。名称随便填，接入点填 nova，其他的保持默认就可以了。"),
            ("好了，保存了。", "请您重启一下手机，然后打开移动数据，看看是否可以上网。"),
            ("可以了，谢谢。", "不客气。如果还有网页打不开，可以把 DNS 改成自动获取试试。"),
            ("DNS 在哪里改？", "在 Wi-Fi 的详细信息里面，选择高级设置，把 IP 地址和 DNS 都改成自动，然后保存。"),
            ("我看到了，改好了。", "好的，请您再刷新一下页面确认。"),
        ],
        "long_agent": ["我把几个常见的网络设置问题给您整理一下。", "如果手机连接 Wi-Fi 但不能上网，先检查是否需要登录认证，有些公共网络需要在浏览器里确认。",
                       "如果是 DNS 的问题，可以尝试把它改为公共的地址，通常能解决网页打不开的情况。", "如果使用 VPN 以后速度变慢，可能是服务器距离较远，建议更换节点。",
                       "如果 Ping 值很高并且不稳定，可能是路由器的问题，可以尝试重启或者更换频道。", "如果以上都不行，请联系我们，我们可以做远程诊断。"],
        "long_customer": ["我在家办公，公司要求通过 VPN 连接内网，之前一直都是好好的。", "这个星期开始，连接的时候总是提示超时，偶尔连上了也特别卡，开个文档要等半分钟。",
                          "我问过公司的 IT，他们说公司那边没有问题，让我检查自己家里的网络。", "我已经重启过路由器，也换过手机热点试了，热点反而是正常的，所以我觉得是宽带的问题。"],
    },
    "sim": {
        "open": ["你好，我的手机丢了，想挂失 SIM 卡。",
                 "喂，我 SIM 卡的 PIN 码输错了三次，现在被锁住了，怎么办？",
                 "你好，我想补办一张 SIM 卡，原来的坏了。"],
        "ex": [
            ("手机是昨天晚上丢的。", "您别着急。为了安全，我先帮您临时停机，这样别人就不能使用您的号码了。"),
            ("好的，麻烦你帮我停一下。", "已经停机了。请问您需要补办新的 SIM 卡，还是直接开通 eSIM？"),
            ("补办实体卡要去营业厅吗？", "可以去任何一家营业厅，带上身份证就行。也可以在 App 里申请邮寄，三天内送到。"),
            ("邮寄要收费吗？", "补卡本身收取少量工本费，邮寄是免费的。"),
            ("新卡的 PIN 码是多少？", "新卡的初始 PIN 码在卡片的背面，您收到以后可以在手机设置里修改。"),
            ("好的，那我选邮寄吧。", "好的，请您告诉我收货地址和联系电话。"),
        ],
        "long_agent": ["我给您说明一下补卡以后需要注意的事情。", "新卡激活以后，旧卡会立刻失效，原来的短信和通话记录不会保留。", "您的号码、套餐和余额都不会变化，也不需要重新办理合约。",
                       "如果您的手机里绑定了很多账号，建议您尽快修改密码，防止验证码被他人收到。", "银行卡和支付软件，也建议您联系对应的客服进行冻结，确认安全以后再解冻。",
                       "如果您找回了旧手机，请不要再使用旧卡。"],
        "long_customer": ["我当时是在地铁上，人很多，下车以后才发现口袋是空的，回去找也没有找到。", "手机里有很多重要的东西，银行卡、支付软件、工作的邮箱，全都绑定了这个号码。",
                          "我现在是借朋友的手机给你们打电话，所以特别着急，担心有人拿去乱用。", "你们能不能马上帮我把所有的服务都先暂停，之后我再慢慢处理。"],
    },
}

GREET_AGENT = ["您好，{b} 客服中心，很高兴为您服务，请问有什么可以帮您？", "您好，这里是 {b} 客服热线，请问有什么可以帮您？",
               "您好，{b}，我是客服专员，请问您需要办理什么业务？"]
VERIFY = [("请问您的手机号码是多少？", "我的号码是 {phone}。"),
          ("为了保障您的账户安全，需要核实一下您的身份，请问您的姓氏是？", "我姓王。"),
          ("请问是机主本人在使用吗？", "对，是我本人。")]
BACKCHANNEL_C = ["嗯。", "好的。", "对。", "是的。", "哦，明白了。", "行。", "可以。"]
BACKCHANNEL_A = ["好的。", "嗯嗯。", "明白。", "稍等。", "收到。"]
CLOSE = [("请问还有其他需要帮助的吗？", "没有了，谢谢你。"),
         ("今天的问题是否已经为您解决了？", "解决了，谢谢。"),
         ("感谢您的来电，祝您生活愉快，再见。", "好的，再见。")]
HOLD_AGENT = ["请您稍等一下，我需要和技术部门核实一下，请不要挂断电话。", "好的，我这边需要查询一下系统，大约需要一分钟，请您在线等待。"]
HOLD_BACK = ["非常抱歉让您久等了，我已经核实清楚了。", "谢谢您的耐心等待，我这边查到结果了。"]
DTMF_PROMPT = "为了核对您的信息，请您用手机键盘输入八位服务码，输入完成后按井号键。"


def spoken_code(code: str) -> str:
    return " ".join("".join(CN_DIGIT[int(ch)] if ch.isdigit() else ch if ch.isalpha() else " " for ch in code).split())


def spaced(sp: str) -> str:
    return " ".join(sp.replace(" ", ""))


def make_code(rnd: random.Random) -> str:
    return rnd.choice([
        lambda: f"NV2024{rnd.randint(1, 12):02d}{rnd.randint(1, 28):02d}",
        lambda: f"TK-{rnd.randint(10000, 99999)}",
        lambda: f"SO-{rnd.randint(100, 999)}-{rnd.randint(100, 999)}",
        lambda: f"RMA-{rnd.randint(1000, 9999)}",
        lambda: f"NN{rnd.randint(1000, 9999)}{rnd.choice('XYZABK')}",
    ])()


class Slots:
    """The per-call fills of {b} {p} {j} {c} {c2} {phone}; each is a (text, tts) pair."""

    def __init__(self, rnd: random.Random) -> None:
        self.rnd = rnd
        self.b = rnd.choice(T.BRANDS)
        self.p = rnd.choice(T.PRODUCTS)
        self.codes = [make_code(rnd), make_code(rnd)]
        self.phone = "1" + "".join(str(rnd.randint(0, 9)) for _ in range(10))

    def fill(self, template: str) -> tuple[str, str]:
        text = template.replace("{c2}", self.codes[1]).replace("{c}", self.codes[0])
        tts = template.replace("{c2}", spaced(spoken_code(self.codes[1]))).replace("{c}", spaced(spoken_code(self.codes[0])))
        phone_tts = spaced(spoken_code(self.phone))
        text, tts = text.replace("{phone}", self.phone), tts.replace("{phone}", phone_tts)
        for key, value in (("{b}", self.b), ("{p}", self.p)):
            text, tts = text.replace(key, value), tts.replace(key, value)
        return text, tts


def check(text: str) -> None:
    bad = [ch for ch in text if ch in FORBIDDEN]
    assert not bad, (bad, text)


def utt(slots: Slots, leg: str, template: str, voice: str, gap: float, *, long: bool = False) -> dict:
    text, tts = slots.fill(template)
    check(text)
    row = {"kind": "utt", "leg": leg, "text": text, "voice": voice, "gap_s": round(gap, 2)}
    if tts != text:
        row["tts"] = tts
    if long:
        row["long"] = True
    return row


# Per call: target minutes, noise of the (agent, customer) legs, features. 24 calls, 3 to 10 minutes.
NOISE = {"clean": 20, "moderate": 80, "noisy": 200}
FEATURES: list[tuple[float, str, str, list[str]]] = [
    # minutes, agent noise, customer noise, features
    (3.2, "clean", "clean", []),
    (3.6, "clean", "moderate", ["long"]),
    (4.0, "moderate", "clean", ["hold_silence"]),
    (4.4, "clean", "clean", ["crosstalk"]),
    (4.8, "moderate", "moderate", ["dtmf"]),
    (5.0, "clean", "moderate", ["hold_music"]),
    (5.2, "clean", "noisy", ["long", "crosstalk"]),
    (5.5, "noisy", "moderate", ["hold_silence", "dtmf"]),
    (5.8, "clean", "clean", ["hold_music", "long"]),
    (6.0, "moderate", "clean", ["crosstalk", "dtmf"]),
    (6.3, "clean", "moderate", ["hold_silence_both"]),
    (6.6, "moderate", "noisy", ["hold_music", "crosstalk"]),
    (6.9, "clean", "clean", ["long", "hold_silence"]),
    (7.2, "moderate", "moderate", ["dtmf", "hold_music"]),
    (7.5, "clean", "moderate", ["crosstalk", "long"]),
    (7.8, "noisy", "clean", ["hold_silence_both", "dtmf"]),
    (8.1, "clean", "clean", ["hold_music", "crosstalk", "long"]),
    (8.4, "moderate", "moderate", ["hold_silence", "long"]),
    (8.7, "clean", "noisy", ["hold_music", "dtmf"]),
    (9.0, "moderate", "clean", ["hold_silence_both", "crosstalk", "long"]),
    (9.3, "clean", "moderate", ["hold_music", "hold_silence", "dtmf"]),
    (9.6, "clean", "clean", ["crosstalk", "long", "hold_music"]),
    (9.8, "moderate", "moderate", ["hold_silence", "dtmf", "long", "crosstalk"]),
    (10.0, "clean", "noisy", ["hold_music", "hold_silence_both", "long", "dtmf"]),
]
assert len(FEATURES) == N_CALLS

CHARS_PER_S = 4.3  # TTS speaking rate, for planning only (synth.py measures the real durations)


def est(text: str) -> float:
    return sum(1 for ch in text if not ch.isspace()) / CHARS_PER_S + 0.4


def build_call(index: int) -> dict:
    rnd = random.Random(SEED + index)
    minutes, agent_noise, customer_noise, features = FEATURES[index]
    slots = Slots(rnd)
    av, cv = AGENT_VOICES[index % 2], CUSTOMER_VOICES[(index // 2) % 2]
    target = minutes * 60
    events: list[dict] = []
    clock = 0.0
    topics = rnd.sample(list(TOPICS), 5)

    def add(row: dict, spent: float) -> None:
        nonlocal clock
        events.append(row)
        clock += spent

    def say(leg: str, template: str, gap: float | None = None, *, long: bool = False) -> None:
        gap = rnd.uniform(0.4, 1.4) if gap is None else gap
        row = utt(slots, leg, template, av if leg == "agent" else cv, gap, long=long)
        add(row, max(gap, 0) + est(row["text"]))

    say("agent", rnd.choice(GREET_AGENT), 1.0)
    say("customer", rnd.choice(["你好。", "喂，你好。", "你好，我想咨询一下。"]), 0.5)
    for q, a in rnd.sample(VERIFY, 2):
        say("agent", q)
        say("customer", a)
    holds = [f for f in features if f.startswith("hold")]
    dtmf_done = long_done = cross_done = False
    pending_holds = list(holds)
    for topic in topics:
        data = TOPICS[topic]
        say("customer", rnd.choice(data["open"]))
        exchanges = list(data["ex"])
        rnd.shuffle(exchanges)
        for k, (q, a) in enumerate(exchanges):
            if clock > target - 25:
                break
            say("customer", q)
            # crosstalk: the agent starts before the customer is done, or the customer cuts in
            if "crosstalk" in features and rnd.random() < 0.3:
                say("agent", a, -rnd.uniform(0.6, 1.5))
                cross_done = True
            else:
                say("agent", a)
            if rnd.random() < 0.22:
                say("customer", rnd.choice(BACKCHANNEL_C))
            if rnd.random() < 0.12:
                say("agent", rnd.choice(BACKCHANNEL_A))
            if "long" in features and not long_done and k == 1 and clock > target * 0.25:
                key = "long_agent" if rnd.random() < 0.65 else "long_customer"
                leg = "agent" if key == "long_agent" else "customer"
                say(leg, "".join(data[key]), long=True)
                long_done = True
            if pending_holds and clock > target * 0.4 and rnd.random() < 0.35:
                hold = pending_holds.pop(0)
                say("agent", rnd.choice(HOLD_AGENT))
                dur = rnd.uniform(30, 90)
                audio = "music" if hold == "hold_music" else "silence"
                add({"kind": "hold", "seconds": round(dur, 1), "audio": audio,
                     "both_silent": hold == "hold_silence_both"}, dur)
                say("agent", rnd.choice(HOLD_BACK), 0.8)
            if "dtmf" in features and not dtmf_done and clock > target * 0.3 and rnd.random() < 0.3:
                say("agent", DTMF_PROMPT)
                digits = "".join(str(rnd.randint(0, 9)) for _ in range(8)) + "#"
                add({"kind": "dtmf", "digits": digits, "gap_s": round(rnd.uniform(1.0, 2.5), 2)}, 6.0)
                dtmf_done = True
        if clock > target - 25:
            break
    # whatever the random draws did not reach: place it near the end
    for hold in pending_holds:
        say("agent", rnd.choice(HOLD_AGENT))
        dur = rnd.uniform(30, 90)
        add({"kind": "hold", "seconds": round(dur, 1), "audio": "music" if hold == "hold_music" else "silence",
             "both_silent": hold == "hold_silence_both"}, dur)
        say("agent", rnd.choice(HOLD_BACK), 0.8)
    if "dtmf" in features and not dtmf_done:
        say("agent", DTMF_PROMPT)
        digits = "".join(str(rnd.randint(0, 9)) for _ in range(8)) + "#"
        add({"kind": "dtmf", "digits": digits, "gap_s": 1.5}, 6.0)
    if "long" in features and not long_done:
        data = TOPICS[topics[0]]
        say("agent", "".join(data["long_agent"]), long=True)
    if "crosstalk" in features and not cross_done:
        q, a = rnd.choice(TOPICS[topics[1]]["ex"])
        say("customer", q)
        say("agent", a, -1.0)
    q, a = rnd.choice(CLOSE[:2])
    say("agent", q)
    say("customer", a)
    q, a = CLOSE[2]
    say("agent", q)
    say("customer", a, 0.4)
    return {"id": f"call{index + 1:02d}", "minutes": minutes, "features": features, "agent_voice": av,
            "customer_voice": cv, "noise_rms": {"agent": NOISE[agent_noise], "customer": NOISE[customer_noise]},
            "est_s": round(clock), "events": events}


def main() -> None:
    calls = [build_call(i) for i in range(N_CALLS)]
    path = HERE / "calls.jsonl"
    path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in calls))
    n_utts = sum(1 for c in calls for e in c["events"] if e["kind"] == "utt")
    chars = sum(len(e["text"]) for c in calls for e in c["events"] if e["kind"] == "utt")
    print(f"{len(calls)} calls, {n_utts} utterances, {chars} characters, planned {sum(c['est_s'] for c in calls) / 60:.0f} min")
    for c in calls:
        print(c["id"], f"plan {c['minutes']} min est {c['est_s'] / 60:.1f} min", c["features"], c["noise_rms"])


if __name__ == "__main__":
    main()
