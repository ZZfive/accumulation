# 视觉算法工程师深挖面试包（去标识化）

## 1) 候选人匹配快照（De-identified）

- Candidate ID: `[Candidate-001]`
- Target role: `图像算法工程师（CV/遥感/AIGC可扩展）`
- Primary match level: `medium-high`
- Resume extraction confidence: `medium`
- Top matching strengths:
- `有0到1落地经历：遥感去云/去阴影功能已进入生产预处理流程`
- `覆盖算法全流程：数据集制作-训练-测试-推理-部署-优化`
- `具备工程化意识：多卡推理加速、国产显卡适配、接口与性能优化`
- Primary concerns:
- `多项指标缺少统一口径（数据集规模、基线、统计显著性未完整给出）`
- `部分技术表述存在命名不清（如“DDFM”）与机制描述不完整`

### 1.1 提取置信度（按版块）

- 教育/技能: `high`
- 工作经历主项目（去云去阴影、去条带）: `high`
- 质检系统/云雪检测: `medium`
- 实习项目（ReCLIP/UNITER）: `medium`
- 不确定片段: `OCR对少量术语与数字有噪声，需面试中逐项复核`

## 1.5) 简历图片排序说明

- Input type: `directory`
- Applied ordering rule: `natural filename numeric order（1,2,...)，无歧义时不触发后续规则`
- Ordered files: `1.png, 2.png`

## 2) JD-简历证据矩阵


| JD Target                    | Resume Evidence                 | Evidence Strength | Verification Need        |
| ---------------------------- | ------------------------------- | ----------------- | ------------------------ |
| 独立承接需求并做方案设计                 | 遥感去云/去阴影从0到1，含路线选择与部署           | strong            | 让候选人复盘“业务约束→技术方案→上线收益”闭环 |
| CV核心算法研发（分割/增强/识别）           | 去云去阴影、条带噪声去除、云雪检测、质检            | strong            | 校验任务定义、损失函数、误差类型与边界场景    |
| 深度学习框架实战（PyTorch/TensorFlow） | 明确PyTorch/OpenCV/Python；训练与推理优化 | strong            | 要求给出关键训练配置和可复现实验设置       |
| 工程化与系统集成                     | 接入公司预处理流程，算子化，多卡推理，显卡适配         | strong            | 追问服务化架构、回滚策略、线上监控指标      |
| 模型全生命周期管理                    | 提到数据制作、训练、测试、推理到微调              | partial           | 验证数据版本化、模型版本管理、A/B与漂移监控  |
| 前沿CV/AIGC技术落地                | 提到Transformer、扩散模型思路            | partial           | 核验为何选择、相对传统方法收益与代价       |
| 沟通协作与跨团队交付                   | 有“客户按需启用/生产单位应用”描述              | partial           | 核验跨团队协作冲突与决策机制           |
| 后端/平台能力（加分项）                 | 有接口配置与系统优化经历                    | partial           | 追问API设计、并发控制、异常处理与SLA    |
| 论文与研究能力（加分项）                 | 声称SCI一区/WACV一作                  | partial           | 要求1篇代表作技术贡献与复现/落地关系      |


## 3) 优先风险信号与核验清单

- 风险1: `指标可比性风险（PSNR/SSIM、84.3%、88.6%缺少统一基线与数据集口径）`
- How to verify: `要求现场写出实验协议：数据划分、基线模型、统计区间、失败样本分布`
- 风险2: `“独立负责”边界不清（可能偏实现，未必主导架构）`
- How to verify: `追问被否决过的方案、拍板依据、上线后复盘`
- 风险3: `术语不标准（DDFM/ReCLIP）可能存在包装表述`
- How to verify: `要求从目标函数与前向/反向路径讲机制，不允许只讲名词`
- 风险4: `工程可靠性证据不足（监控、告警、灰度、回滚）`
- How to verify: `让候选人给出线上故障案例及MTTR改进`

## 4) 平衡式题目规划

- Total questions: `12`
- Top-level questions: `4`
- Deep-detail questions: `8`
- Model-principle questions: `5`（Q5/Q6/Q7/Q8/Q9）
- Troubleshooting questions: `3`（Q10/Q11/Q12）

## 5) 深挖问题与标准答案

### Q1. 你在“去云/去阴影”项目中从需求到上线的完整闭环是什么？你个人拍板了哪些关键决策？

- Level: `top_level`
- Intent: `验证独立性、业务抽象能力、端到端交付能力`
- Source anchor: `遥感影像增强项目 + JD(需求承接/方案设计)`
- Canonical answer (`verified`):
- `强答案应覆盖：业务目标量化(KPI/SLA)→数据与标注策略→模型与基线→离线评估→灰度上线→监控与回滚→收益复盘。关键决策需能说明约束、备选方案与取舍。`
- Resume-expected answer:
- `应提及预处理流程集成、多星源适配、大图切片一致性问题及其解决。`
- Acceptable answer:
- `能说明完整链路并明确个人负责环节与结果。`
- Red flags:
- `只讲“我做了模型”，不讲上线约束`
- `无法说明为何不是另一条技术路线`
- Follow-up prompts:
- `如果客户要求延迟降30%，你先改哪里？`
- `一次你做错的技术决策是什么，如何纠偏？`
- `灰度阶段你设置了哪些拦截阈值？`
- Scoring guide (1-5): `1=无闭环；3=闭环基本完整；5=闭环+可量化收益+反思迭代`

### Q2. “算法算子化+多卡并行+国产显卡适配”这三块你分别做了什么，如何保证可维护性？

- Level: `top_level`
- Intent: `验证工程化深度与生产代码能力`
- Source anchor: `工程化描述 + JD(工具/系统开发)`
- Canonical answer (`verified`):
- `应包含：模块边界定义、统一I/O协议、配置化参数、算子版本控制、性能基准、硬件抽象层、CI回归测试。`
- Resume-expected answer:
- `应能举出1-2个具体性能瓶颈（显存/IO/算子耗时）及优化收益。`
- Acceptable answer:
- `能讲清至少一种可复用抽象和一种性能优化证据。`
- Red flags:
- `把脚本堆叠称为“系统化”`
- `无版本管理与回归验证`
- Follow-up prompts:
- `给出你定义的算子接口最小字段集合。`
- `跨硬件一致性如何验收？`
- `上线后如何防止配置漂移？`
- Scoring guide (1-5): `1=口号化；3=有工程实践；5=工程规范+量化优化+稳定性策略`

### Q3. 你如何向非技术业务方解释 PSNR/SSIM 与真实业务价值的关系？

- Level: `top_level`
- Intent: `验证沟通能力与指标治理能力`
- Source anchor: `PSNR/SSIM指标 + JD(沟通协作)`
- Canonical answer (`verified`):
- `PSNR/SSIM反映重建相似度，但不等于下游任务收益；需补充任务指标（检测/解译精度）、人工质检通过率、错误代价。应建立离线指标与线上业务KPI映射。`
- Resume-expected answer:
- `应给出“图像质量提升→下游识别更稳→业务效率提升”的证据链。`
- Acceptable answer:
- `能说明“单一指标不足”并给出补充评估维度。`
- Red flags:
- `把PSNR/SSIM当作唯一成功标准`
- `无法定义业务可接受阈值`
- Follow-up prompts:
- `如果SSIM升高但下游识别下降，你怎么排查？`
- `业务方只关心误检率时你如何重设目标？`
- Scoring guide (1-5): `1=指标误用；3=知道多指标；5=能建立指标-业务映射与决策机制`

### Q4. 你在跨团队协作中最难的一次冲突是什么？最终如何达成交付？

- Level: `top_level`
- Intent: `验证协作、风险管理与主人翁意识`
- Source anchor: `生产接入与客户启用 + JD(协作交付)`
- Canonical answer (`verified`):
- `高质量回答应包含冲突对象、冲突根因、对齐机制（RFC/评审/里程碑）、决策记录与复盘。`
- Resume-expected answer:
- `应能讲出算法、后端、业务三方的约束平衡。`
- Acceptable answer:
- `能给出具体冲突实例和可验证结果。`
- Red flags:
- `将问题归因他人，无自我改进`
- `没有过程证据（文档/评审/时间线）`
- Follow-up prompts:
- `如果重来一次，流程上你会改什么？`
- `你如何提前识别高风险依赖？`
- Scoring guide (1-5): `1=无案例；3=有案例但方法弱；5=机制化协作与可复用经验`

### Q5. 你提到“Transformer双分支注意力+特征细化前馈模块”，请从机制上解释相较CNN的收益与代价。

- Level: `deep_detail`
- Intent: `模型原理理解（注意力机制、归纳偏置、复杂度）`
- Source anchor: `去云去阴影技术创新 + JD(技术深度)`
- Canonical answer (`verified`):
- `Transformer通过全局注意力建模长程依赖，适合大尺度上下文；代价是O(N^2)注意力开销与数据需求上升。双分支可理解为全局语义分支+局部细节分支，前馈细化用于补边缘与纹理。若用于遥感，需结合窗口化/金字塔降低计算与显存成本。`
- Resume-expected answer:
- `应说明双分支输入/融合位置、损失项、在云边界与阴影区域的收益。`
- Acceptable answer:
- `能正确解释注意力作用与复杂度代价，并给出工程降本方法。`
- Red flags:
- `把注意力解释成“自动调参”`
- `说不清分支各自职责`
- Follow-up prompts:
- `你的分支融合是concat还是cross-attn，为什么？`
- `如何避免全局分支吞噬局部纹理？`
- `在16bit遥感数据上做过哪些归一化处理？`
- Scoring guide (1-5): `1=概念错误；3=机制基本正确；5=机制+实现+场景化权衡完整`

### Q6. 条带噪声去除里你提到“扩散模型+频域分解”，请写出训练目标并解释为何能抑制条带。

- Level: `deep_detail`
- Intent: `验证生成式去噪原理与可解释性`
- Source anchor: `去条带噪声项目 + JD(AIGC/前沿CV)`
- Canonical answer (`needs-verification`):
- `通用扩散去噪目标是预测噪声或x0重建，结合频域分解可将条带主导频带（方向性高频/周期噪声）单独约束；常见做法是空间域重建损失+频域一致性损失联合训练。条带抑制有效的关键是噪声先验与频带可分性假设成立。`
- Resume-expected answer:
- `应明确“DDFM”具体定义、噪声调度、频带分解方式（FFT/小波/可学习滤波）与推理耗时。`
- Acceptable answer:
- `能正确描述扩散训练目标，并说清频域约束如何作用于条带噪声。`
- Red flags:
- `只会讲“扩散效果好”，讲不出目标函数`
- `无法说明为何不是UNet监督去噪`
- Follow-up prompts:
- `你的噪声模拟器如何覆盖乘性/加性与周期/随机条带？`
- `为什么选该噪声调度而不是cosine/linear另一种？`
- `如何评估“去条带”同时不过度平滑纹理？`
- Scoring guide (1-5): `1=原理不清；3=有正确框架；5=目标函数、先验与失败模式都能讲透`

### Q7. 云雪检测里的 copy-paste 增强，为什么对小目标有效？什么时候会适得其反？

- Level: `deep_detail`
- Intent: `验证数据策略与长尾问题理解`
- Source anchor: `云雪检测技术创新 + JD(算法优化)`
- Canonical answer (`verified`):
- `copy-paste通过提高稀有类别出现频次与上下文多样性缓解长尾；对小目标可提升召回。风险在于粘贴边缘伪影、上下文不自然与标签噪声，会导致过拟合伪模式。需配合掩码平滑、上下文约束和难例挖掘。`
- Resume-expected answer:
- `应解释样本选择策略、粘贴位置规则、与难例挖掘如何联动。`
- Acceptable answer:
- `能说明收益机制与至少两类副作用及缓解办法。`
- Red flags:
- `把增强视为无副作用`
- `没有离线消融实验`
- Follow-up prompts:
- `你做过哪些ablation（仅copy-paste/仅难例/联合）？`
- `如何避免云与雪在光谱/纹理上被模型混淆？`
- Scoring guide (1-5): `1=经验化；3=机制清晰；5=有消融证据与风险控制`

### Q8. 实习里“ReCLIP/UNITER目标指代理解”具体怎么做跨模态对齐？

- Level: `deep_detail`
- Intent: `验证多模态模型机制理解与落地能力`
- Source anchor: `实习项目 + JD(前沿技术跟踪)`
- Canonical answer (`needs-verification`):
- `UNITER类方法通常基于区域特征与文本token做联合编码，并通过MLM/MRM/ITM等任务学习对齐；若结合CLIP思路，常用对比学习或相似度检索筛候选，再进行细粒度匹配。目标指代任务关键是候选区域生成、文本消歧与指代一致性评分。`
- Resume-expected answer:
- `应给出输入表示、对齐头设计、损失函数、推理流程和错误案例。`
- Acceptable answer:
- `能讲清至少一条完整“文本→候选框→匹配评分→输出”链路。`
- Red flags:
- `混淆检索式与生成式指代方法`
- `只报准确率，不会做错误归因`
- Follow-up prompts:
- `UNITER预训练任务与你任务迁移时保留了哪些？`
- `家庭场景里遮挡和指代歧义如何处理？`
- `为何不用纯CLIP zero-shot直接做？`
- Scoring guide (1-5): `1=概念混乱；3=流程正确；5=机制/迁移/误差分析完整`

### Q9. 去云去阴影任务你最终用什么损失组合？每项损失对应哪类失败模式？

- Level: `deep_detail`
- Intent: `验证目标函数设计与误差驱动优化`
- Source anchor: `遥感影像增强项目 + JD(算法研发)`
- Canonical answer (`verified`):
- `常见组合：像素重建(L1/L2)保整体一致性，结构损失(SSIM)保结构细节，感知/频域损失抑制纹理伪影；可加掩码加权突出云阴影区域。失败模式映射应明确：过平滑、色偏、边缘伪影、残留云雾。`
- Resume-expected answer:
- `应对应其“色彩校正+小波校正+切片一致性”经验解释损失权重设置。`
- Acceptable answer:
- `能说明至少三种损失及各自作用边界。`
- Red flags:
- `损失堆砌但无法说明贡献`
- `不做权重敏感性分析`
- Follow-up prompts:
- `你如何自动调loss权重？`
- `为何L1优于L2或反之？`
- `线下最常见失败样本是哪类？`
- Scoring guide (1-5): `1=拍脑袋调参；3=有基本映射；5=损失-失败模式-指标三者闭环`

### Q10. 上线后出现“颜色漂移/拼接接缝明显”，你如何定位是模型问题还是工程问题？

- Level: `deep_detail`
- Intent: `故障诊断与系统性排障能力`
- Source anchor: `大图切片色彩一致性优化 + JD(稳定性优化)`
- Canonical answer (`verified`):
- `先分层定位：数据预处理一致性→模型输出统计→后处理/拼接策略→编码与显示链路。对照实验应固定三段并逐段替换；常见修复包括重叠窗口+加权融合、色彩空间统一、tile-level归一化对齐、参考图颜色迁移约束。`
- Resume-expected answer:
- `应结合其“小波色彩校正”说明定位证据与修复前后对比。`
- Acceptable answer:
- `能给出可执行排障路径与验证实验。`
- Red flags:
- `直接重训模型，不做分层排查`
- `无可复现故障样本集`
- Follow-up prompts:
- `如何构造最小可复现实验？`
- `你会记录哪些线上诊断日志？`
- Scoring guide (1-5): `1=无方法；3=有排查步骤；5=分层诊断+可复现实验+回归策略`

### Q11. 同精度目标下，推理延迟超预算40%，你如何做模型与系统协同优化？

- Level: `deep_detail`
- Intent: `性能优化与工程权衡能力`
- Source anchor: `多卡并行/显卡适配 + JD(性能与可扩展性)`
- Canonical answer (`verified`):
- `先profiling定位瓶颈（算子、内存、IO、后处理）；再做分级优化：图优化与算子融合、混合精度、批处理策略、并行管线、缓存与预取、服务层并发控制。以P95延迟、吞吐、成本与精度共同约束。`
- Resume-expected answer:
- `应给出至少一次“优化前后指标对比”。`
- Acceptable answer:
- `有性能剖析依据，能提出按优先级执行的优化清单。`
- Red flags:
- `无剖析数据直接“换更大卡”`
- `只看平均延迟，不看P95/P99`
- Follow-up prompts:
- `如果INT8掉点明显，你怎么回退/补偿？`
- `国产卡上算子不支持时你的替代策略？`
- Scoring guide (1-5): `1=经验主义；3=有方法；5=指标驱动+跨层优化+可回滚`

### Q12. 自动质检系统上线后误报高，业务方失去信任，你的修复策略是什么？

- Level: `deep_detail`
- Intent: `线上质量闭环与产品意识`
- Source anchor: `影像质检系统 + JD(模型全生命周期)`
- Canonical answer (`verified`):
- `应建立“误报分层账本”：数据偏移、标注噪声、阈值失配、场景漂移。短期通过阈值分档+人工复核兜底；中期做主动学习回流与校准；长期建设监控（漂移、分布、置信度）和周期再训练。`
- Resume-expected answer:
- `应结合其质检维度（边界异常、直方图异常、云雪检查）逐项给修复动作。`
- Acceptable answer:
- `能提出短中长期策略并可量化。`
- Red flags:
- `只靠“调阈值”`
- `没有数据闭环与责任机制`
- Follow-up prompts:
- `你如何定义“可运营”的告警策略？`
- `误报与漏报冲突时如何按业务代价权衡？`
- Scoring guide (1-5): `1=被动救火；3=有阶段方案；5=工程化质量治理闭环`

## 6) 面试结论草案

- Recommendation: `hold（可进入下一轮技术实操）`
- Confidence: `medium`
- Decision reasons:
- `项目覆盖面与工程落地能力较强，符合“能独当一面”的JD主线`
- `前沿模型与指标治理表述存在不确定点，需要机制级追问验证真深度`
- Unresolved items for next step:
- `要求现场白板：去云去阴影模型结构与损失函数，含复杂度与部署预算`
- `要求提供1个真实故障复盘：监控信号、定位路径、修复与回归结果`

## 7) 评分执行建议（按题统一）

- 维度: `Ownership and Clarity / Technical Depth / Engineering Judgment / Validation and Debugging / Technical Correctness`
- 规则: `每题1-5分，最终按深挖题加权（deep_detail权重1.2）`
- 判定建议:
- `>=4.3：强通过`
- `3.5-4.2：通过，附定向培养点`
- `2.8-3.4：补面/加试`
- `<2.8：不建议继续`

## 8) 技术基线参考（用于“canonical answer”校验）

- Transformer: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- DDPM: [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- UNITER: [https://arxiv.org/abs/1909.11740](https://arxiv.org/abs/1909.11740)
- Copy-Paste (CVPR 2021): [https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html](https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html)
- SSIM (TIP 2004): [https://ece.uwaterloo.ca/~z70wang/publications/ssim.html](https://ece.uwaterloo.ca/~z70wang/publications/ssim.html)

