# 图片
## 图片交互
- https://huggingface.co/spaces/AI4Editing/MagicQuill

## 图片修复
- https://ali-vilab.github.io/largen-page/
- https://aba-ir.github.io/
- 将涂抹区域中的物体清除，类似于智能擦除：https://hustvl.github.io/PixelHacker/

## 图片outpainting
- https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint

## 图片隐藏水印
- https://huggingface.co/papers/2408.10446

## GUI图片内容结构化识别
- https://huggingface.co/spaces/jadechoghari/OmniParser
  - https://microsoft.github.io/OmniParser/

## 抠图
- https://github.com/plemeri/InSPyReNet
- https://lightchaserx.github.io/matting-by-generation/
  - https://huggingface.co/papers/2407.21017
- https://huggingface.co/ZhengPeng7/BiRefNet
- https://github.com/ViTAE-Transformer/P3M-Net
- https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting
- https://huggingface.co/spaces/not-lain/background-removal/blob/main/app.py
- https://colab.research.google.com/drive/10z-pNKRnVNsp0Lq9tH1J_XPZ7CBC_uHm?usp=sharing#scrollTo=XQS1RNu3IEl2
- https://huggingface.co/spaces/finegrain/finegrain-object-cutter
- https://huggingface.co/spaces/Hedro/room_cleaner
- https://huggingface.co/spaces/innova-ai/video-background-removal-抠视频背景，这个也不错
- https://huggingface.co/briaai/RMBG-2.0
- 支持视频抠背景，效果比现在灵创上更好-https://huggingface.co/PramaLLC/BEN2
- 视频抠背景，好像可以控制：https://pq-yang.github.io/projects/MatAnyone/

## 语义分割
- https://github.com/vladan-stojnic/LPOSS

## 图片编辑
- https://huggingface.co/papers/2408.09702
- https://github.com/fallenshock/FlowEdit，可以跟换细节，改变整体风格
- https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing
- https://fluxspace.github.io/
- 交互式图片编辑：https://huggingface.co/papers/2501.08225
- https://github.com/primecai/diffusion-self-distillation
- 将目标对象融入到目标图片中：https://huggingface.co/spaces/WensongSong/Insert-Anything

## 提示词生成
- https://huggingface.co/spaces/gokaygokay/FLUX-Prompt-Generator
- https://github.com/dagthomas/comfyui_dagthomas

## 图片Caption
- https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha

## 图片超分
- https://huggingface.co/spaces/finegrain/finegrain-image-enhancer
- https://huggingface.co/fal/AuraSR
  - https://huggingface.co/spaces/gokaygokay/AuraSR-v2
- https://huggingface.co/spaces/finegrain/finegrain-image-enhancer
- https://github.com/ArcticHare105/S3Diff
- https://github.com/shallowdream204/DreamClear
- https://github.com/zsyOAOA/InvSR
- https://therasr.github.io/
- https://huggingface.co/papers/2503.18446
- https://huggingface.co/papers/2504.08591
- https://huggingface.co/spaces/gokaygokay/Tile-Upscaler
## 图片编辑
- https://huggingface.co/spaces/LiruiZhao/Diffree
- 去除物体：https://huggingface.co/spaces/finegrain/finegrain-object-eraser
- https://swift-edit.github.io/
- 什么都好像能做：https://xavierchen34.github.io/UniReal-Page/
- https://github.com/TencentARC/BrushEdit，与VLM模型结合，分主动和手动两种模式，可添加、修改和删除图片内容
- https://xilluill.github.io/projectpages/KV-Edit/，尽可能保存mask外的区域不改变
- 人像编辑：https://huggingface.co/spaces/tight-inversion/tight-inversion-pulid-demo
- https://github.com/yeates/OmniPaint
- 对输入的图片进行一定的编辑，角色保持的图片编辑：https://github.com/yairshp/SISO
- https://github.com/LOCATEdit/LOCATEdit
- https://github.com/peterljq/Concept-Lancet
- https://github.com/CUC-MIPG/UnifyEdit
- https://river-zhang.github.io/ICEdit-gh-pages/
人物风格迁移
- 发型（不好搞）、发色、年龄等
  - https://huggingface.co/spaces/AIRI-Institute/StyleFeatureEditor
    - https://github.com/AIRI-Institute/StyleFeatureEditor
  - https://mytimemachine.github.io/--年龄变化

## 图片布局生成
- https://huggingface.co/papers/2407.15233
- https://roictrl.github.io/--可控制图片区域生成
- https://github.com/limuloo/3DIS--可控制图片区域生成
- https://art-msra.github.io/
- flux的无训练：https://github.com/limuloo/DreamRenderer

## 图片风格迁移
- https://github.com/songrise/artist
  - https://huggingface.co/papers/2407.15842
- https://huggingface.co/spaces/fffiloni/RB-Modulation
- https://ciarastrawberry.github.io/stylecodes.github.io/
- https://stylestudio-official.github.io/，将参考图片风格结合提示词生成对应风格图片
- https://easyref-gen.github.io/，也是好像什么都能做点
- https://consislora.github.io/
- https://huggingface.co/spaces/fotographerai/Zen-Style-Shape

## 图片多概念融合
- https://github.com/ali-vilab/In-Context-LoRA
- https://diptychprompting.github.io/--可以风格迁移、物体生成背景等
- 偏向于风格融合：https://github.com/wutong16/FiVA
- 物体融合：https://kakituken.github.io/affordance-any.github.io/
- 通过提示词将多张图片中的多个概念融合到一张图片中：https://token-verse.github.io/
- 基于Flux的模型，可以将多张参考图片中的对象保真生成到一张图片中：https://github.com/bytedance/UNO
- 多张图片结合提示词，将不同概念融合
  - https://huggingface.co/spaces/IP-composer/ip-composer
  - https://github.com/bytedance/DreamO

## 图片是否由AI生成的检测模型
- https://huggingface.co/HPAI-BSC/SuSy

## Try-ON（虚拟试衣）
- https://humanaigc.github.io/outfit-anyone/
  - https://huggingface.co/papers/2407.16224
- https://huggingface.co/spaces/yisol/IDM-VTON
- https://byjiang.com/FitDiT/
  - https://huggingface.co/spaces/BoyuanJiang/FitDiT
- https://yisol.github.io/BootComp/
- https://github.com/franciszzj/Leffa，还可以控制人体质态
- https://sihuiji.github.io/FashionComposer-Page/
- https://github.com/Zheng-Chong/CatV2TON
- https://github.com/logn-2024/Any2anyTryon
- https://huggingface.co/spaces/WeShopAI/WeShopAI-Virtual-Try-On
- https://huggingface.co/spaces/VIDraft/Fashion-Fit
## Try-off（脱衣）
- https://rizavelioglu.github.io/tryoffdiff/

## 图片上色
- https://huggingface.co/spaces/fffiloni/text-guided-image-colorization
- 基于参考图片，可细颗粒度对线条图片上色：https://github.com/ali-vilab/MangaNinjia
- 基于参考图给线条图上色：https://zhuang2002.github.io/Cobra/

## 图片relight
- https://luminet-relight.github.io/#
- https://huggingface.co/spaces/jasperai/LBM_relighting
- https://huggingface.co/papers/2504.03011

## 多视角图片生成
- https://github.com/huanngzh/MV-Adapter

## 人物换发型、发色
- 换发色：https://openart.ai/workflows/akihungac/color-hair-stylist-workflow/rWSPc3GSZaCzTIoC0Hno
- 换发型：https://openart.ai/workflows/dugumatai/super-natural-hair-design/1p8U4xX3QC8VchdYsAoZ

## 图片中文本精确生成
- https://github.com/NJU-PCALab/TextCrafter

## 矢量图生成
- 将图片转换为矢量图
  - https://huggingface.co/spaces/ovi054/image-to-vector
  - https://github.com/joanrod/star-vector
- 文生矢量图/text2svg：https://github.com/showlab/LayerTracer
- https://omnisvg.github.io/

## 其他
- 文本生成个性化漫画：https://github.com/jianzongwu/DiffSensei
- 生成式图片隐藏水印：https://huggingface.co/papers/2412.04653
- 超高分辨率图片生成
  - https://github.com/ali-vilab/FreeScale
  - https://zhenyangcs.github.io/RectifiedHR-Diffusion/
  - https://github.com/Huage001/URAE
  - https://github.com/Bujiazi/HiFlow
- 图片或视频中人体关键点检测：https://huggingface.co/spaces/hysts/ViTPose-transformers
- 人物一致性的文生图或绘本
  - https://aigcdesigngroup.github.io/AnyStory/
  - https://github.com/byliutao/1Prompt1Story
  - https://flexip-tech.github.io/flexip
- 文本生成live 2D图像：https://github.com/Human3DAIGC/Textoon
- 图片修复：https://github.com/KangLiao929/Noise-DA
-  3d风格图像生成：https://huggingface.co/spaces/ginigen/text3d-R1
- 图片控制生成指定文本：https://t2i-text-loc.github.io/
- 直接换头：https://github.com/ai-forever/ghost-2.0
- 一个人物的多张多视角图片为参考，生成人物形象保持的图片：https://github.com/nupurkmr9/syncd
- 内容控制：https://github.com/Xiaojiu-z/EasyControl
- 图片物体移动：https://xinyu-andy.github.io/ObjMover/
- 物体分割：https://github.com/hustvl/GroundingSuite
- 人物个性化生成：https://github.com/fenghora/personalize-anything
- 画面迁移，先通过两张前后对比的图片学习一种图片画面差异，再将此种差异应用到新的图片中：https://github.com/CUC-MIPG/Edit-Transfer
- 统一、灵活的元素级图像生成或编辑：https://liyaowei-stu.github.io/project/BlobCtrl/
- 基于图片生成立体图：https://qjizhi.github.io/genstereo/
- 合成图片判定：https://opendatalab.github.io/LEGION
- 人物面部保留(类似于Instant)：
  - https://huggingface.co/spaces/ByteDance/InfiniteYou-FLUX
  - https://instantcharacter.github.io/
- 图片生成遵循相机移动视角的图片或视频：https://huggingface.co/spaces/stabilityai/stable-virtual-camera
- 合成图片检测及理由说明：https://github.com/opendatalab/FakeVLM
- 组合图片搜索：https://collm-cvpr25.github.io/
- 说是支持各种任务：https://visualcloze.github.io/
- 图片去雾：https://castlechen339.github.io/DehazeXL.github.io/
- 准确的文本渲染：https://reptext.github.io/
- 图片的法线图/normal map估计：https://stable-x.github.io/StableNormal/
- 图片中法线图、深度图、各种渲染分解等图片预测：https://www.obukhov.ai/marigold
  - https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage

# 视频
## 视频去水印
- https://github.com/YaoFANGUK/video-subtitle-remover

## 视频生成
- https://ssyang2020.github.io/zerosmooth.github.io/
- https://github.com/NUS-HPC-AI-Lab/VideoSys
- 控制生成
  - https://huggingface.co/papers/2408.11475
  - https://github.com/dvlab-research/ControlNeXt
  - https://github.com/bytedance/X-Dyna
  - https://shiyi-zh0408.github.io/projectpages/FlexiAct/
- 图片前后帧生成视频
  - https://svd-keyframe-interpolation.github.io/
- 简笔画生成动图：https://hmrishavbandy.github.io/flipsketch-web/
- 卖货视频：https://cangcz.github.io/Anchor-Crafter
- 基于图片多角度、参数生成：https://huggingface.co/spaces/l-li/NVComposer
- 基于图片和拖动方向生成动图：https://ppetrichor.github.io/levitor.github.io/
- 生成带透明背景的视频：https://github.com/wileewang/TransPixar
- 以任意视觉元素为参考对象，生成的视频中保持元素一致性：https://huggingface.co/papers/2504.02436

## 基于音频拼接视频
- https://huggingface.co/papers/2408.10998?collection=true

## 视频上色
- 线稿视频参考上色：https://luckyhzt.github.io/lvcd
- 将图片中的风格迁移到视频：https://github.com/KwaiVGI/StyleMaster
- https://yihao-meng.github.io/AniDoc_demo/

## 数字人
- https://huggingface.co/papers/2408.03284
- https://github.com/lipku/metahuman-stream
  - https://zhuanlan.zhihu.com/p/696337285
- https://deepbrainai-research.github.io/float/
- https://github.com/memoavatar/memo
- https://github.com/Hanbo-Cheng/DAWN-pytorch
- https://huggingface.co/spaces/Skywork/skyreels-a1-talking-head
- https://huggingface.co/spaces/DyrusQZ/LHM
- 阿里通义，4090实现30fps：https://humanaigc.github.io/chat-anyone/
- https://github.com/harlanhong/ACTalker
- https://github.com/Fantasy-AMAP/fantasy-talking
- https://antonibigata.github.io/KeySync/

## 视频特征map提取
- https://huggingface.co/papers/2408.12569

## 视频补帧
- https://github.com/MCG-NJU/EMA-VFI

## 视频超分
- https://iceclear.github.io/projects/seedvr/
- https://github.com/NJU-PCALab/STAR

## 视频修复
- 人脸视频恢复的统一框架，支持 BFR、着色、修复等任务
  - https://huggingface.co/spaces/fffiloni/SVFR-demo
  - https://github.com/wangzhiyaoo/SVFR

## 视频编辑
- 视频各种编辑能力：https://genprop.github.io/
- 视频中对象擦除：https://github.com/lixiaowen-xw/DiffuEraser
- https://github.com/TencentARC/VideoPainter

## 其他
- 人物一致性的视频生成
  - https://huggingface.co/papers/2501.13452
  - https://echopluto.github.io/MagicID-project/
- 视频理解：https://github.com/bytedance/tarsier
- 循环视频直接生成：https://github.com/YisuiTT/Mobius
- 视频inpaint：https://mtv-inpaint.github.io/
- 视频法线估计：https://normalcrafter.github.io/
- 视频虚拟穿衣：https://2y7c3.github.io/3DV-TON/

# 音频
## 音频修复
- https://huggingface.co/papers/2409.08514
- 音质调高、超分等：https://github.com/modelscope/ClearerVoice-Studio

## 音频生成
- https://huggingface.co/papers/2407.15060
- 文本生音频
  - https://huggingface.co/cvssp/audioldm-m-full
  - https://huggingface.co/spaces/OpenSound/EzAudio
  - https://github.com/shaopengw/Awesome-Music-Generation
  - https://huggingface.co/spaces/asigalov61/Giant-Music-Transformer
  - https://huggingface.co/declare-lab/tango2-full
  - 可以单文本或文本+音频作为输入：https://huggingface.co/spaces/facebook/MelodyFlow
  - https://github.com/declare-lab/Tangoflux
    - https://huggingface.co/spaces/declare-lab/TangoFlux
  - https://huggingface.co/spaces/amphion/Vevo
- 视频生成音频
  - https://github.com/open-mmlab/FoleyCrafter
  - https://github.com/ariesssxu/vta-ldm
  - https://yannqi.github.io/Draw-an-Audio/
  - https://v-aura.notion.site/Temporally-Aligned-Audio-with-Autoregression-90793f75ff3c4ff69d1d248ea2000836
  - 视频+文本生成音频，https://ificl.github.io/MultiFoley/
  - https://github.com/hkchengrex/MMAudio
- 文本/音频/视频生成音频
  - https://zeyuet.github.io/AudioX/
  - 可生成音效或音乐：https://huggingface.co/spaces/Zeyue7/AudioX

## 音频分离
- https://huggingface.co/spaces/r3gm/Audio_separator
- https://github.com/kwatcharasupat/source-separation-landing
- https://github.com/WangHelin1997/SoloAudio-通过提示词将一种目标声音从一段音频中提取出去

## 音频情绪识别
- https://modelscope.cn/models/iic/emotion2vec_base_finetuned/summary

## TTS
- https://github.com/fishaudio/fish-speech?tab=readme-ov-file
- https://fun-audio-llm.github.io/#SenseVoice-overview
  - TTS+ASR
- https://huggingface.co/hexgrad/Kokoro-82M
- 读绘本：https://github.com/DrewThomasson/ebook2audiobook
- 字节开源：https://huggingface.co/ByteDance/MegaTTS3

## 声音克隆
- https://huggingface.co/papers/2409.02245
- https://huggingface.co/spaces/mrfakename/E2-F5-TTS
- https://mullivc.github.io/
  - https://huggingface.co/papers/2408.04708
- 类cosyvoice：https://github.com/edwko/OuteTTS
- https://huggingface.co/spaces/srinivasbilla/llasa-3b-tts
- https://huggingface.co/SparkAudio/Spark-TTS-0.5B

## 同声传译
- https://huggingface.co/papers/2407.21646
- https://huggingface.co/papers/2408.05101

## 音乐分离
- https://github.com/kwatcharasupat/query-bandit
- https://github.com/kwatcharasupat/source-separation-landing

## 音乐生成
- https://github.com/feizc/FluxMusic
- 基于视频生成对其bgm音乐：https://muvi-v2m.github.io/?utm_source=diffusiondigest.beehiiv.com&utm_medium=referral&utm_campaign=sd-3-5-midjourney-editor-act-one-animation-this-week-in-ai-art
- https://github.com/FunAudioLLM/InspireMusic
- https://xmusic-project.github.io/
- 基于歌词生成整首歌曲，可以设置语言、风格、情感等，文本正确性不是非常好：https://github.com/multimodal-art-projection/YuE
  - https://huggingface.co/spaces/fffiloni/YuE
- https://github.com/LiuZH-19/SongGen
- https://versband.github.io/
- https://github.com/ace-step/ACE-Step
  - https://www.reddit.com/r/comfyui/comments/1kgyf4o/acestep_is_now_supported_in_comfyui/
- Stable audio open的后续迭代：https://arc-text2audio.github.io/web/

## 音频可视化
- 传入图片，基于音频，渲染出视频：https://github.com/yvann-ba/ComfyUI_Yvann-Nodes

# 3D
## 3D生成
- https://buaacyw.github.io/meshanything-v2/
- https://huggingface.co/spaces/sudo-ai/SpaRP
- https://huggingface.co/spaces/sudo-ai/MeshLRM
- https://github.com/Microsoft/TRELLIS
  - https://huggingface.co/spaces/ginipick/SORA-3D
  - https://huggingface.co/spaces/JeffreyXiang/TRELLIS
- 图片生成3D：https://thuzhaowang.github.io/projects/DI-PCG/
- 对3D模型及进行超分：https://github.com/DHPark98/SequenceMatters
- 局部级的3D模型生成或编辑：https://silent-chen.github.io/PartGen/
- 图片生成3D模型：https://spar3d.github.io/
- 3D头像生成：https://xiangyueliu.github.io/GaussianAvatar-Editor/
- stability新的方案，先生成点云，再生成mesh：https://huggingface.co/spaces/stabilityai/stable-point-aware-3d
- 文本或图片生成模型，开源：https://ltt-o.github.io/Kiss3dgen.github.io/
- https://huggingface.co/papers/2503.11629
- 这个和HunYuan3d-2、Trellis对标更好：https://huggingface.co/spaces/Stable-X/Hi3DGen
- 基于图片和粗糙的3D模型生成细节更丰富的3D模型：https://detailgen3d.github.io/DetailGen3D/

## 3D场景生成
- https://ys-imtech.github.io/projects/LayerPano3D/
- https://huggingface.co/papers/2412.04827
- https://snap-research.github.io/wonderland/
- https://huggingface.co/spaces/VAST-AI/MIDI-3D

## 贴图生成
- 图片+提示词+模型生成贴图：https://github.com/CVMI-Lab/TEXGen
- 文或图生贴图，开源：https://github.com/OpenTexture/Paint3D
- 文生贴图：https://github.com/LIU-Yuxin/SyncMVD
- 文生贴图、图生贴图，就是没开源：https://flexitex.github.io/FlexiTex/
- 文生贴图，没开源：https://dong-huo.github.io/TexGen/
- 在进行图片生3D模型后，将原始图片和生成的3D模型作为输入，对3D模型进行贴图：https://huggingface.co/spaces/VAST-AI/MV-Adapter-Img2Texture

## 其他
- 人物图片转身体草图：https://huggingface.co/spaces/yeq6x/Image2Body_gradio
- 超大画幅图片生成：https://huggingface.co/spaces/takarajordan/CineDiffusion
- 基于视频生成3D场景：https://huggingface.co/spaces/facebook/vggt
- 通过mask对图片中的区域生成caption：https://cvlab-kaist.github.io/URECA/
- 好像是可以将图片中的风格生成模型的贴图：https://styleme3d.github.io/
- 将3d模型进行拆分：https://www.reddit.com/r/comfyui/comments/1kgwb80/i_implemented_a_new_mit_license_3d_model/

# 心理学相关
- https://huggingface.co/papers/2409.11733--情绪识别？
- https://huggingface.co/papers/2409.12106
