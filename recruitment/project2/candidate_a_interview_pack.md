# Deep-Dive Interview Pack

## 1) Candidate Fit Snapshot (De-identified)

- Candidate ID: `[Candidate-A]`
- Target role: 图像算法工程师（偏业务落地、可独立负责、CV+工程化）
- Primary match level: `medium-high`
- Resume extraction confidence: `high`（项目指标真实性与部分细节口径为 `medium`，需面试核验）
- Top matching strengths:
- OCR 全流程落地经验（检测/版面分析/SDK 集成/C++ ONNXRuntime 推理）
- 有生成式方向实践（Stable Diffusion + DreamBooth/LoRA，FID 指标与数据构造）
- 有部署与工程问题处理经历（内存泄漏、warmup 显存占位、推理时延优化）
- Primary concerns:
- 与招聘优先级中的“视频编解码/ffmpeg”基本无直接证据
- 多项目指标较好但缺少统一对照基线与线上稳定性长期数据

## 1.2) Assumptions & Confidence Notes

- 脱敏策略：已去除姓名、手机号、邮箱、头像等个人身份信息。
- 提取置信度：
- 教育/技能/工作时间线：`high`
- 项目职责与技术路线：`high`
- 指标与收益（如 BLEU/FID/召回率）：`medium`（需核验实验协议、测试集构成、是否可复现）
- 不确定片段（`low/medium`）：
- 个别项目中“数据规模与指标口径”的统计边界未完全展开（训练集/测试集分布、线上样本比例等）

## 1.5) Resume Image Ordering

- Input type: `directory`
- Applied ordering rule: `natural filename numeric order`
- Ordered files: `1.png, 2.png, 3.png, 4.png, 5.png`

## 2) JD-to-Resume Evidence Matrix


| JD Target            | Resume Evidence                            | Evidence Strength | Verification Need        |
| -------------------- | ------------------------------------------ | ----------------- | ------------------------ |
| 独立需求承接与方案设计          | OCR 平台、古籍切分、RAG 问答均描述了从数据到部署链路             | strong            | 追问需求输入、取舍依据、里程碑与失败复盘     |
| CV 核心算法研发（检测/分割/OCR） | DBNet/PSENet、YOLO 系列、版面分析与文本行排序            | strong            | 是否亲自改网络与训练策略，还是以调参为主     |
| 生成式模型落地（AIGC）        | SD v1-4 + DreamBooth/LoRA 生成古籍样本，FID 优化    | strong            | 数据闭环是否真正提升主任务泛化，收益可量化性   |
| 工程化与生产代码质量           | ONNXRuntime C++ 推理、SDK 集成、接口设计、Valgrind 排障 | strong            | 代码质量标准、监控告警、发布回滚机制       |
| 模型全生命周期管理            | 数据清洗、训练、评估、部署、在线问答服务                       | partial           | 缺模型监控/漂移检测/自动化重训证据       |
| 多模态/LLM 融合           | Qwen3-4B + RAG + RAGAS 评估                  | partial           | 幻觉抑制机制、检索失败兜底、事实一致性保障    |
| 视频编解码/ffmpeg         | 未见明确项目描述                                   | missing           | 必问是否做过视频抽帧、转码、时序对齐、低延迟链路 |
| Web 后端服务协作           | Streamlit 交互与在线服务描述                        | partial           | API 设计、并发、鉴权、日志链路等后端工程能力 |


## 3) Priority Risk Signals

- Risk 1: 指标“好看”但实验协议不透明（如 FID/BLEU/召回率口径）
- How to verify: 要求现场写出评估协议（数据切分、负样本、置信区间、A/B 对照）
- Risk 2: 可能偏“模型训练+离线验证”，线上可靠性体系证据不足
- How to verify: 深问监控指标、告警阈值、灰度发布、回滚与故障复盘
- Risk 3: 视频编解码/ffmpeg 与招聘优先级存在缺口
- How to verify: 追加定向题（ffmpeg pipeline、码率/清晰度/时延 trade-off）
- Risk 4: 复杂项目中的个人 ownership 边界可能模糊
- How to verify: 要求逐项说明“本人负责模块、关键决策、失败修复”

## 4) Balanced Question Plan

- Total questions: `12`
- Top-level questions: `4`
- Deep-detail questions: `8`
- Model-principle questions: `6`（满足 >=4）
- Troubleshooting questions: `3`（满足 >=2）

## 5) Deep-Dive Questions and Answer Keys

### Q1. 你在“现代档案 OCR 平台”中独立负责了哪些环节？请按需求->方案->上线->迭代复盘讲清楚。

- Level: `top_level`
- Intent: 验证端到端 ownership 与交付能力
- Source anchor: OCR 平台 + JD“独立承接与方案设计”
- Canonical answer (`verified`): 强答案应包含业务目标量化、方案候选与取舍、里程碑、上线验收指标、失败案例与修复闭环。
- Resume-expected answer: 应提到版面分析/检测模型迭代、SDK 集成、接口设计、线上问题定位与更新策略。
- Acceptable answer: 能明确本人负责边界并给出至少一个关键决策及结果。
- Red flags: 全程“我们做了”无个人边界；只讲模型不讲上线约束。
- Follow-up prompts: 你拍板的关键技术决策是什么？如果重来一次会改哪一步？
- Scoring guide (1-5): 1=无 ownership；3=有边界但浅；5=端到端且有复盘证据。

### Q2. 你的项目优先级如何匹配“视觉生成 > 视频编解码/ffmpeg > 传统机器视觉”？你有什么短板补齐计划？

- Level: `top_level`
- Intent: 验证岗位匹配与自我认知
- Source anchor: 生成+传统视觉项目；招聘目标
- Canonical answer (`verified`): 应如实承认 ffmpeg 缺口，并给可执行补齐路径（抽帧/转码/时序同步/硬编解码实践计划）。
- Resume-expected answer: 生成方向可举 SD 数据增强闭环；传统视觉可举 OCR/YOLO 落地。
- Acceptable answer: 匹配分析客观，补齐计划具体到技术点与时间。
- Red flags: 回避缺口；空泛“可快速学习”。
- Follow-up prompts: 如果两周内补齐 ffmpeg，你会做什么 mini-project？
- Scoring guide (1-5): 1=回避；3=承认但计划泛；5=计划可执行可验证。

### Q3. 你为何在 DBNet 中加分类分支？它对损失函数、收敛和误检/漏检有什么影响？

- Level: `deep_detail`
- Intent: 检验模型机制理解（原理题1）
- Source anchor: DBNet 改造
- Canonical answer (`verified`): DBNet本质是可微二值化文本检测；增分类头会引入多任务学习，需平衡检测与分类损失权重，可能提升区分相近文本类型但也可能造成梯度竞争和收敛不稳。
- Resume-expected answer: 应说清“为何要区分手写/标准体/背景”、标签构建方式、loss 权重调节及收益。
- Acceptable answer: 解释新增分支的业务动机+训练代价+误差变化。
- Red flags: 仅说“加头精度更高”；不理解多任务权衡。
- Follow-up prompts: 你怎么设 loss 权重？如何判断是模型问题还是标注噪声问题？
- Scoring guide (1-5): 1=机制错误；3=概念对但不完整；5=能讲梯度/权衡/实验。

### Q4. 你把 ResNet50 换成 MobileNetV3 后时延从 ~5s 降到 ~1s，精度与吞吐如何权衡？

- Level: `deep_detail`
- Intent: 检验工程化模型压缩思维（原理题2）
- Source anchor: OCR 推理优化
- Canonical answer (`verified`): 轻量 backbone 降 FLOPs 与参数量；需通过蒸馏、输入分辨率策略、后处理优化维持精度；报告应同时给 P50/P95 延迟、吞吐、内存占用和准确率变化。
- Resume-expected answer: 应提到在性能下降可接受前提下达成时延目标，并有场景化阈值。
- Acceptable answer: 能给出至少两维指标与阈值而非只报单点时延。
- Red flags: 只谈平均时延，不谈尾延迟和稳定性。
- Follow-up prompts: 你会如何做回归测试防止精度暗降？
- Scoring guide (1-5): 1=无指标框架；3=有基础权衡；5=多指标体系完整。

### Q5. 解释 LayoutLMv3 做“文本行阅读顺序预测”的可行性边界：何时优于规则法，何时不如规则法？

- Level: `deep_detail`
- Intent: 检验结构化文档建模理解（原理题3）
- Source anchor: LayoutLMv3 排序项目
- Canonical answer (`verified`): LayoutLMv3联合文本与版面信息，对复杂布局更鲁棒；但在规则稳定、数据少、分布单一时规则法成本更低且可解释性更强。
- Resume-expected answer: 应提到规则法性能边界、为何转模型法、数据标注与类别设计。
- Acceptable answer: 能说明适用场景与失败场景。
- Red flags: 绝对化“模型一定比规则好”。
- Follow-up prompts: 你如何设计错误类型 taxonomy（跨栏、注释、图文混排）？
- Scoring guide (1-5): 1=无边界意识；3=有对比；5=有系统决策框架。

### Q6. 你在 SD/DreamBooth/LoRA 项目里，为什么冻结 VAE/TextEncoder（除 token）并只训部分 UNet？

- Level: `deep_detail`
- Intent: 检验生成模型微调机制（原理题4）
- Source anchor: 古籍样本生成
- Canonical answer (`verified`): 冻结大部分模块可降低过拟合与显存成本；LoRA在注意力层注入低秩更新，保留基模能力同时学习风格概念；DreamBooth通过实例+先验损失平衡特异性与泛化。
- Resume-expected answer: 应能讲清训练参数选择、过拟合控制、FID变化与人工筛选关系。
- Acceptable answer: 说清冻结策略和 LoRA/DreamBooth 各自作用。
- Red flags: 把 LoRA 说成“只提速不影响表示”；不懂先验保持。
- Follow-up prompts: 什么时候你会放开 TextEncoder？如何避免概念漂移？
- Scoring guide (1-5): 1=机制混乱；3=基本正确；5=可解释训练稳定性与泛化。

### Q7. 你的 RAG 用“向量+B M25+RRF”，请解释 RRF 为什么能稳健提升召回，何时会失效？

- Level: `deep_detail`
- Intent: 检验检索融合原理（原理题5）
- Source anchor: 民国档案问答
- Canonical answer (`verified`): RRF通过对多个排序结果做倒数秩融合，弱化分数标定差异，常提升召回鲁棒性；若候选池质量差或查询理解失败，融合收益有限。
- Resume-expected answer: 应说明索引缓存、查询重写、RAGAS评估与召回提升链路。
- Acceptable answer: 能说明融合逻辑与失效边界。
- Red flags: 只会背“混合检索更好”。
- Follow-up prompts: 你如何调 RRF 参数并做离线/在线一致性验证？
- Scoring guide (1-5): 1=不会解释；3=会原理；5=会做验证闭环。

### Q8. 你提到部署中遇到内存泄漏和崩溃，用 Valgrind 扫描后修复。请给出一次完整排障过程。

- Level: `deep_detail`
- Intent: 检验生产故障处理（排障题1）
- Source anchor: SDK C++ 部署
- Canonical answer (`verified`): 合理流程包括复现条件固定、最小化复现、工具定位（Valgrind/ASan）、对象生命周期排查、修复后压测与回归。
- Resume-expected answer: 应讲清泄漏点类型（buffer/handle/context）、修复策略与验证结果。
- Acceptable answer: 能说出“定位-修复-验证”三段证据。
- Red flags: 只说“升级库版本就好了”。
- Follow-up prompts: 你如何避免同类问题再次发生？
- Scoring guide (1-5): 1=无方法；3=有步骤；5=可复用排障框架。

### Q9. 如果 OCR 新场景出现误检激增，你如何在 72 小时内止损并给出长期方案？

- Level: `deep_detail`
- Intent: 检验应急与长期优化（排障题2）
- Source anchor: 多场景 OCR 泛化问题
- Canonical answer (`verified`): 短期可阈值/规则兜底+高风险样本人工复核；中期做错误分桶与数据回流；长期重构数据分布与模型结构并建立监控。
- Resume-expected answer: 应结合其“补充数据+模型迭代+流程设计”经历。
- Acceptable answer: 同时覆盖短中长期，不只谈训新模型。
- Red flags: 只强调“加数据”无优先级。
- Follow-up prompts: 你会先看哪三个监控指标？
- Scoring guide (1-5): 1=拍脑袋；3=有分层；5=止损与根因并重。

### Q10. 你如何证明“生成古籍样本”真实提升了主任务（检测/识别）而非仅提升离线 FID？

- Level: `deep_detail`
- Intent: 检验因果验证意识
- Source anchor: AIGC 数据增强
- Canonical answer (`verified`): 必须用任务指标 A/B（不加生成样本 vs 加生成样本），控制变量（模型/训练步数/数据量），并做子分布分析与显著性检验。
- Resume-expected answer: 应能给出下游任务收益曲线，而不只汇报 FID。
- Acceptable answer: 明确“生成质量指标 != 任务收益指标”。
- Red flags: 用单一 FID 直接推断业务收益。
- Follow-up prompts: 你如何防止生成样本引入偏差并伤害线上表现？
- Scoring guide (1-5): 1=因果混淆；3=理解A/B；5=验证设计严谨。

### Q11. 你在 OCR SDK 中怎样设计接口，既支持多模型扩展又不破坏线上兼容性？

- Level: `top_level`
- Intent: 检验工程抽象能力
- Source anchor: SDK 集成与接口设计
- Canonical answer (`verified`): 关键是版本化 API、配置驱动、模型注册机制、前后处理解耦、灰度与回滚能力。
- Resume-expected answer: 应能描述“版面分析+检测”模块边界及 C++ 推理封装。
- Acceptable answer: 有模块边界、版本策略和回滚方案。
- Red flags: 接口与模型强耦合，无法平滑升级。
- Follow-up prompts: 你如何做兼容性测试矩阵？
- Scoring guide (1-5): 1=无设计；3=可用但脆弱；5=可扩展且可回滚。

### Q12. 请现场设计一个最小可行的视频 OCR 管线（含 ffmpeg），并说明性能瓶颈与优化点。

- Level: `top_level`
- Intent: 补齐 JD 优先级缺口验证（视频编解码/ffmpeg）
- Source anchor: JD priority gap coverage
- Canonical answer (`verified`): 合理方案应含解复用/抽帧、关键帧策略、去重、检测识别、时序跟踪、结果聚合；优化点含硬解、批处理、异步队列与缓存。
- Resume-expected answer: 候选人若无直接经验，至少应给结构化方案与学习迁移路径。
- Acceptable answer: 架构完整、瓶颈识别准确、指标可量化。
- Red flags: 只会“逐帧跑OCR”，无时序与性能意识。
- Follow-up prompts: 你会如何在 200ms 延迟预算下取舍精度？
- Scoring guide (1-5): 1=不会设计；3=方案基本可跑；5=可落地并有优化路线。

## 6) Interview Decision Draft

- Recommendation: `yes`（偏“可录用，但需补齐视频链路能力”）
- Confidence: `medium`
- Decision reasons:
- 视觉算法与工程落地证据较扎实，且有 AIGC 实操，符合你们“视觉生成优先”方向
- 有两年以上工作经历并有独立交付痕迹，匹配岗位对自主性要求
- Unresolved items for next step:
- 视频编解码/ffmpeg 实战能力需通过定向技术面或作业验证
- 指标真实性与可复现性需通过实验协议追问与现场 case 复盘确认

