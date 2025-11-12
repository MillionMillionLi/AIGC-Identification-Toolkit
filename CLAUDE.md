# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
### Starting Large Tasks

  When exiting plan mode with an accepted plan: 1.**Create Task Directory**:
  mkdir -p ./[task-name]/

  2.**Create Documents**:

  - `[task-name]-plan.md` - The accepted plan
  - `[task-name]-context.md` - Key files, decisions
  - `[task-name]-tasks.md` - Checklist of work

  3.**Update Regularly**: Mark tasks complete immediately

  ### Continuing Tasks

  - Check `/dev/active/` for existing tasks
  - Read all three files before proceeding
  - Update "Last Updated" timestamps



## Project Overview

This is an AIGC (AI-Generated Content) identification system that provides comprehensive marking and tracing solutions for AI-generated content. The system integrates multiple identification technologies including:

- **Hidden Watermarking Technologies**: Invisible technical identification for copyright protection and content tracing
  - **Text Hidden Watermarking**: Dual-algorithm support (PostMark default, CredID optional)
    - **PostMark** (默认): 后处理水印，支持黑盒LLM（GPT-4等）
    - **CredID** (可选): 生成时水印，需要访问模型logits
  - **Image Hidden Watermarking**: VideoSeal backend by default, PRC-Watermark optional
  - **Audio Hidden Watermarking**: AudioSeal algorithm for robust audio watermarking with optional Bark text-to-speech integration
  - **Video Hidden Watermarking**: Wan2.1 generation + VideoSeal watermarking

- **Visible Marking Technologies**: Visible compliance markers for regulatory requirements and user awareness
  - **Text Visible Marking**: Insert standard compliance text markers
  - **Image Visible Marking**: Overlay visible text marks with customizable position and style
  - **Audio Visible Marking**: Voice marker insertion (based on Bark TTS)
  - **Video Visible Marking**: Visible text overlay on video frames (based on FFmpeg)

- **Implicit Metadata Marking**: (Planned) Structured metadata embedding for generation model, timestamp, parameters, etc.

## Architecture

The codebase follows a modular architecture with the following components:

### Core Modules
- `src/unified/unified_engine.py`: Core AIGC content identification engine for multimodal marking (text/image/audio/video)
- `src/unified/watermark_tool.py`: High-level facade over the engine; preferred entry-point for AIGC identification
- `src/text_watermark/`: Text hidden watermarking with dual-algorithm support (PostMark + CredID)
  - `text_watermark.py`: Unified text watermark facade supporting algorithm switching
  - `postmark_watermark.py`: PostMark post-processing watermark (default)
  - `credid_watermark.py`: CredID generation-time watermark (optional)
  - `PostMark/`: PostMark algorithm repository
  - `credid/`: CredID multi-party watermarking framework
- `src/image_watermark/`: Image identification (VideoSeal backend by default; PRC available)
- `src/audio_watermark/`: Audio identification (AudioSeal; optional Bark TTS)
- `src/video_watermark/`: Video generation (Wan2.1) + VideoSeal identification
- `src/utils/`: Shared utilities for configuration, model management, and visible marking

### Unified Content Identification Engine (Enhanced with Dual-Operation Support)
Location: `src/unified/unified_engine.py`

Key features:
- **Unified AIGC Identification API**: `embed(content, message, modality, operation='watermark|visible_mark', **kwargs)` and `extract(content, modality, operation='watermark|visible_mark', **kwargs)` for comprehensive content marking across `text|image|audio|video`
- **Dual-mode support**: AI generation mode (prompt-based content creation) and upload file mode (existing content processing)
- **Dual-operation support**: Hidden watermarking (`operation='watermark'`) for technical protection and visible marking (`operation='visible_mark'`) for compliance and transparency
- **Smart routing**: Automatically selects appropriate identification technology based on modality and operation type
- **Original content preservation**: returns both original and identified content for before/after comparison display
- **Backward compatibility**: `operation` parameter defaults to `'watermark'` to maintain existing API compatibility
- Technology defaults: `text=postmark`, `image=videoseal`, `audio=audioseal`, `video=hunyuan+videoseal`
- Offline-first: lazily initializes text model/tokenizer from local cache; falls back to `sshleifer/tiny-gpt2` if configured model not found (still offline)
- Config-driven: reads `config/text_config.yaml` and modality-specific configs for comprehensive AIGC identification

### AIGC Content Identification Operation Types
- **`operation='watermark'`** (default): Hidden watermarking for technical protection and content tracing
  - Uses advanced deep learning algorithms (CredID, VideoSeal, AudioSeal, etc.)
  - Invisible to users but detectable with proper identification tools
  - Robust against various attacks and transformations
  - Suitable for copyright protection, content authentication, and technical tracing

- **`operation='visible_mark'`**: Visible compliance marking for AIGC content transparency
  - Adds visible text/audio markers for regulatory compliance and user awareness
  - Clearly indicates AI-generated content to users, ensuring transparency
  - Supports customizable position, style, duration, and content
  - Meets regulatory requirements and supports user informed consent

Quick start:
```python
from src.unified.watermark_tool import WatermarkTool

tool = WatermarkTool()

# ===== AIGC内容隐式标识 (Hidden Identification, operation='watermark') =====
# 适用于版权保护、内容追踪、技术防护等场景

# Text (文本隐式标识，PostMark后处理模式 - 默认)
# PostMark适用于任何LLM生成的文本（包括GPT-4、Claude等黑盒API）
generated_text = "Your LLM generated text here..."  # 任何LLM生成的文本
txt = tool.embed(generated_text, "wm_msg", 'text')  # 后处理添加水印
res = tool.extract(txt, 'text')

# 可选：使用CredID算法（需要修改config/text_config.yaml: algorithm: "credid"）
# CredID适用于自部署的开源模型，生成时嵌入水印
# txt = tool.embed("prompt", "wm_msg", 'text', model=model, tokenizer=tokenizer)

# Image (图像AI生成 + 隐式标识，自动保存原图和标识图用于对比)
img = tool.embed("a cat", "hello_vs", 'image')  # 返回标识图像

# Audio (音频AI生成 + 隐式标识，自动保存原音频和标识音频)
aud = tool.embed("audio content", "hello_audio", 'audio', output_path="outputs/audio/a.wav")

# Video (视频AI生成 + 隐式标识，自动保存原视频和标识视频)
vid = tool.embed("阳光洒在海面上", "video_wm", 'video')

# AIGC内容上传文件标识 (Existing AIGC Content Identification)
img_wm = tool.embed("watermark message", "hello_img", 'image',
                    image_input="/path/to/image.jpg")
aud_wm = tool.embed("watermark message", "hello_audio", 'audio',
                    audio_input="/path/to/audio.wav", output_path="outputs/watermarked.wav")
vid_wm = tool.embed("watermark message", "hello_video", 'video',
                    video_input="/path/to/video.mp4")

# ===== AIGC内容显式标识 (Visible Marking, operation='visible_mark') =====
# 适用于监管合规、用户告知、透明标识等场景
# 文本显式标识
original_text = "这是一段原始文本内容。"
marked_text = tool.embed(original_text, "本内容由AI生成", 'text',
                        operation='visible_mark', position='start')

# 图像显式标识
marked_img = tool.embed("/path/to/image.jpg", "测试标识", 'image',
                       operation='visible_mark',
                       position='bottom_right', font_percent=5.0)

# 音频显式标识 (需要Bark TTS)
marked_audio = tool.embed("/path/to/audio.wav", "本内容由AI生成", 'audio',
                         operation='visible_mark')

# 视频显式标识
marked_video = tool.embed("/path/to/video.mp4", "本内容由AI生成", 'video',
                         operation='visible_mark')

# AIGC内容标识便捷接口
marked_content = tool.add_visible_mark(content="原始内容",
                                      message="本内容由人工智能生成", modality='text')
detection = tool.detect_visible_mark(content=marked_content, modality='text')
```

AIGC Content Identification Interface Parameters and Returns:
- **Text Content Identification (Hidden + Visible)**:
  - **隐式标识**: `embed(prompt, message, 'text')` → 返回标识文本 `str` （仅AI生成模式，适用于版权保护）
  - **显式标识**: `embed(original_text, mark_text, 'text', operation='visible_mark', position='start|end')` → 返回带合规标识文本 `str`
  - Extraction returns `{detected: bool, message: str, confidence: float}` for both operations

- **Image Content Identification (Hidden + Visible)**:
  - **隐式标识**: 基于Stable Diffusion生成图像后嵌入技术标识，返回`PIL.Image` （AI生成模式）或直接对上传图像嵌入标识（上传模式）
  - **显式标识**: `embed('/path/to/image.jpg', mark_text, 'image', operation='visible_mark', position='bottom_right', font_percent=5.0, font_color='#FFFFFF')` → 返回带合规标识图像 `PIL.Image`
  - **Technology backends**: VideoSeal (default), PRC-Watermark (optional)
  - **Effect comparison**: Automatically saves original and identified images, Web interface shows before/after comparison
  - `extract` supports `operation='watermark|visible_mark'`, `replicate/chunk_size` for enhanced detection confidence

- **Audio Content Identification (Hidden + Visible)**:
  - **隐式标识**: Bark TTS生成音频 + AudioSeal技术标识嵌入（AI生成模式）或直接对上传音频嵌入标识（上传模式）
  - **显式标识**: `embed('/path/to/audio.wav', mark_text, 'audio', operation='visible_mark', position='start', voice_preset='v2/zh_speaker_6')` → 返回带语音合规标识的音频
  - **Format support**: WAV, MP3, FLAC等主流音频格式
  - **Effect comparison**: 自动保存原音频和标识音频，支持Web播放器对比
  - Returns `torch.Tensor | str`; extraction returns `{detected, message, confidence}`

- **Video Content Identification (Hidden + Visible)**:
  - **隐式标识**: HunyuanVideo生成视频 + VideoSeal技术标识嵌入（AI生成模式）或直接对上传视频嵌入标识（上传模式）
  - **显式标识**: `embed('/path/to/video.mp4', mark_text, 'video', operation='visible_mark', position='bottom_right', font_percent=4.0, duration_seconds=2.0)` → 返回带合规文字标识的视频
  - **Browser compatibility**: 自动转码为H.264+AAC+faststart格式确保Web播放
  - **Effect comparison**: 自动保存原视频和标识视频，支持并排播放对比
  - Returns saved video path; `extract` returns `{detected, message, confidence, metadata}`

**AIGC Content Identification Convenience Methods**:
- `tool.add_visible_mark(content, message, modality, **kwargs)` → 一键添加AIGC显式合规标识
- `tool.detect_visible_mark(content, modality, **kwargs)` → 检测AIGC显式标识
- `tool.get_supported_operations()` → `['watermark', 'visible_mark']`
- `tool.get_operation_info()` → 返回AIGC标识操作类型详细信息

Offline cache hints:
- Set `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`; store models under `models/` or point `HF_HOME/HF_HUB_CACHE` to local hub

### Text Watermarking Framework (Dual-Algorithm Support)

**架构概览**：
```
src/text_watermark/
├── text_watermark.py          # 统一门面，支持多算法切换
├── postmark_watermark.py      # PostMark后处理水印（默认）
├── credid_watermark.py        # CredID生成时水印（可选）
├── PostMark/                  # PostMark算法仓库
└── credid/                    # CredID多方水印框架
```

#### PostMark 后处理水印（默认算法）

**特点**：
- 黑盒LLM支持：无需访问模型logits，适用于GPT-4、Claude等API
- 后处理模式：对已生成的文本进行水印嵌入
- 高灵活性：任何LLM生成的文本都可以添加水印

**核心文件**：
- `src/text_watermark/postmark_watermark.py`: PostMark封装类
- `src/text_watermark/PostMark/postmark/`: 算法核心实现
  - `models.py`: Watermarker, embedder, inserter模型
  - `utils.py`: 水印词提取和存在率计算
  - `watermark.py`: 水印嵌入脚本
  - `detect.py`: 水印检测脚本

**依赖模型**（已下载到本地）：
- `nomic-ai/nomic-embed-text-v1`: 文本嵌入模型
- `mistralai/Mistral-7B-Instruct-v0.2`: 水印词插入LLM
- `paragram_xxl.pkl`: Paragram词嵌入
- `filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl`: 预计算嵌入

**配置参数** (`config/text_config.yaml`):
```yaml
postmark:
  embedder: "nomic"              # 嵌入模型
  inserter: "mistral-7b-inst"    # 插入LLM
  ratio: 0.12                    # 水印词比例（12%）
  iterate: "v2"                  # 迭代插入版本
  threshold: 0.7                 # 检测相似度阈值
```

**使用示例**：
```python
from src.text_watermark.postmark_watermark import PostMarkWatermark

# 初始化
watermark = PostMarkWatermark({
    'embedder': 'nomic',
    'inserter': 'mistral-7b-inst',
    'ratio': 0.12
})

# 嵌入水印（后处理）
text = "Your LLM generated text..."
result = watermark.embed(text, message="watermark_id")

# 提取水印
detection = watermark.extract(
    result['watermarked_text'],
    original_words=result['watermark_words']
)
print(f"检测成功: {detection['detected']}, 置信度: {detection['confidence']}")
```

#### CredID 生成时水印（可选算法）

Located in `src/text_watermark/credid/`, this is a comprehensive multi-party watermarking framework:

**特点**：
- 白盒LLM：需要访问模型logits，适用于自部署开源模型
- 生成时嵌入：在文本生成过程中同步嵌入水印
- 高准确率：深度集成到生成过程，检测率更高

**核心文件**：
- `watermarking/`: Core watermarking algorithms (CredID, KGW, MPAC, etc.)
- `attacks/`: Attack implementations (copy-paste, deletion, homoglyph, substitution)
- `evaluation/`: Evaluation pipelines and metrics for quality, speed, robustness analysis
- `experiments/`: Experimental scripts for research validation
- `demo/`: Example scripts for single-party and multi-party scenarios

#### 算法切换

**方法1: 配置文件切换**
编辑 `config/text_config.yaml`:
```yaml
# 使用PostMark（默认）
algorithm: "postmark"

# 或使用CredID
# algorithm: "credid"
```

**方法2: 代码动态切换**
```python
from src.text_watermark.text_watermark import TextWatermark

watermark = TextWatermark()
watermark.set_algorithm('postmark')  # 或 'credid'
```

**算法对比**：

| 特性 | PostMark（默认） | CredID |
|------|-----------------|--------|
| 模型访问 | 黑盒（无需logits） | 白盒（需要logits） |
| 适用场景 | 第三方API | 自部署模型 |
| 嵌入方式 | 后处理 | 生成时 |
| 灵活性 | 高 | 中 |
| 检测率 | 高 | 高 |
| 处理成本 | 需二次LLM调用 | 单次生成 |

### AudioSeal Audio Watermarking Framework
Located in `src/audio_watermark/`, this provides comprehensive audio watermarking capabilities:

- `audioseal_wrapper.py`: Core AudioSeal watermarking implementation with 16-bit message encoding/decoding and 3D tensor handling
- `bark_generator.py`: Bark text-to-speech integration with intelligent cache management and local model priority loading
- `audio_watermark.py`: Unified audio watermarking interface supporting both direct audio and TTS workflows with batch processing
- `utils.py`: Audio processing utilities for I/O, quality assessment, visualization, and noise robustness testing
- `audioseal/`: AudioSeal algorithm submodule (Meta's official implementation)

## Common Development Commands

### Installation and Setup
```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Install CredID-specific dependencies (if working with text watermarking)
pip install -r src/text_watermark/credid/watermarking/MPAC/requirements.txt

# Install PRC-Watermark dependencies (if working with image watermarking)  
pip install -r src/image_watermark/PRC-Watermark/requirements.txt

# Install AudioSeal dependencies (if working with audio watermarking)
pip install torch torchaudio julius soundfile librosa scipy matplotlib

# Install Bark for text-to-speech (optional, for advanced audio features)
pip install git+https://github.com/suno-ai/bark.git
```

### Running the Tool
```python
from src.unified.watermark_tool import WatermarkTool

tool = WatermarkTool()

# 隐式水印（默认）
text_wm = tool.embed("这是一个测试文本", "msg", 'text')
text_res = tool.extract(text_wm, 'text')

# 显式标识
marked_text = tool.embed("原始文本", "本内容由AI生成", 'text',
                        operation='visible_mark', position='start')
mark_res = tool.extract(marked_text, 'text', operation='visible_mark')

# 便捷接口
marked_content = tool.add_visible_mark("原始内容", "显式标识", 'text')
detection = tool.detect_visible_mark(marked_content, 'text')

# Optional: choose algorithms
tool.set_algorithm('image', 'videoseal')
img = tool.generate_image_with_watermark("a cat", message="hello")
img_res = tool.extract_image_watermark(img, replicate=16, chunk_size=16)
```

### 使用视频水印（VideoSeal）最小封装

已将 VideoSeal 以子包形式集成于 `src/video_watermark/videoseal`，并在 `src/video_watermark/__init__.py` 暴露公共入口：

```python
from video_watermark import load

# 加载默认 256-bit 模型（会按需下载权重到运行目录 ckpts/）
model = load("videoseal")

# 对图像/视频执行嵌入或检测（参见 videoseal 文档）
# 例：对视频帧张量进行嵌入（FxCxHxW, 值域[0,1]）
# outputs = model.embed(frames, is_video=True)
# msgs = outputs["msgs"]
# frames_w = outputs["imgs_w"]
```

依赖提示：需要 `ffmpeg` 可执行和以下 Python 包（若缺请安装）
`ffmpeg-python av omegaconf timm==0.9.16 lpips pycocotools PyWavelets tensorboard calflops pytorch-msssim scikit-image scipy tqdm safetensors`

### 使用音频水印（AudioSeal）

已将 AudioSeal 集成于 `src/audio_watermark/`，提供完整的音频水印解决方案：

```python
from src.audio_watermark import create_audio_watermark

# 创建音频水印工具
watermark_tool = create_audio_watermark()

# 基础音频水印嵌入
import torch
audio = torch.randn(1, 16000)  # 1秒音频
message = "test_message_2025"

# 嵌入水印
watermarked_audio = watermark_tool.embed_watermark(audio, message)

# 提取水印
result = watermark_tool.extract_watermark(watermarked_audio)
print(f"检测成功: {result['detected']}, 消息: {result['message']}")

# 文本转语音 + 水印（需要安装Bark）
generated_audio = watermark_tool.generate_audio_with_watermark(
    prompt="Hello, this is a test",
    message="bark_watermark",
    voice_preset="v2/en_speaker_6"
)
```

**核心特性**：
- **16位消息编码**: 支持字符串消息的哈希编码
- **高质量嵌入**: SNR > 40dB，几乎无听觉差异
- **鲁棒检测**: 对噪声、压缩等攻击有良好抗性
- **多语言TTS**: 集成Bark支持中英文等多语言语音生成
- **批处理支持**: 支持批量音频处理
- **文件I/O**: 支持多种音频格式读写

**依赖要求**：
- 基础功能: `torch torchaudio julius soundfile librosa scipy matplotlib`
- 高级功能（TTS）: `pip install git+https://github.com/suno-ai/bark.git`
- 注意：Bark安装后会自动下载模型到指定缓存目录（约5GB）

### Testing and Development

#### 🚀 推荐的测试运行方式

我们提供了一个统一的测试脚本 `run_tests.py` 来简化测试流程，自动处理路径设置和依赖检查：

```bash
# 运行所有测试
python run_tests.py

# 运行特定模态的测试
python run_tests.py --audio             # 音频水印测试
python run_tests.py --image             # 图像水印测试  
python run_tests.py --text              # 文本水印测试

# 运行特定测试文件
python run_tests.py test_audio_watermark.py

# 详细输出
python run_tests.py --audio -v

# 快速测试（跳过耗时测试）
python run_tests.py --quick
```

#### 📁 传统的测试运行方式

如果你喜欢直接运行测试文件，我们已经修复了导入问题，以下命令都可以正常工作：

```bash
# PRC图像水印测试 (推荐)
python test_prc_only.py                 # 完整PRC水印系统测试
python test_modes_comparison.py         # 不同模式性能对比

# CredID文本水印测试
cd src/text_watermark/credid/demo
python test_method_single_party.py      # Single vendor scenario
python test_method_multi_party.py       # Multi-vendor scenario  
python test_real_word.py                # Real-world mixed text scenario

# AudioSeal音频水印测试
python tests/test_audio_watermark.py    # 完整音频水印测试套件
python audio_watermark_demo.py          # 端到端演示脚本
```

#### 🛠️ 故障排除

如果遇到导入问题：

1. **确保从项目根目录运行**：
   ```bash
   cd /path/to/unified_watermark_tool
   python run_tests.py --audio
   ```

2. **检查环境**：
   ```bash
   python -c "import torch; print('✅ PyTorch 可用')"
   python -c "import transformers; print('✅ Transformers 可用')"
   ```

3. **使用我们的便利脚本**：
   ```bash
   python run_tests.py --audio  # 自动设置路径和检查依赖
   ```

### Configuration Management

The tool uses YAML configuration files and supports both AI generation and file upload modes for all supported modalities:

#### 📁 主要配置文件位置
- `config/default_config.yaml`: 统一配置文件，包含所有模态的默认设置
- `config/text_config.yaml`: 文本水印专用配置
- `src/text_watermark/credid/config/`: 算法特定的JSON配置 (CredID.json, KGW.json, etc.)

#### 🔧 各模态参数配置详解

##### 文本水印配置 (Text Watermarking)
**修改文件**: `config/text_config.yaml` 或 `src/text_watermark/credid/config/CredID.json`
```yaml
# config/text_config.yaml
text_watermark:
  algorithm: credid
  model_name: gpt2-medium               # 或 sshleifer/tiny-gpt2 (离线回退)
  device: cuda                          # 设备选择：cpu/cuda
  watermark_method: credid
  hf_cache_dir: ~/.cache/huggingface   # 模型缓存目录
  offline_mode: true                   # 强制离线加载
```

**核心参数说明**:
- `model_name`: LLM模型路径，优先本地缓存
- `offline_mode`: 启用时强制`local_files_only=True`
- `watermark_method`: 支持credid, kgw, mpac等算法

##### 图像水印配置 (Image Watermarking)  
**修改文件**: `config/default_config.yaml` (image_watermark section)
```yaml
# config/default_config.yaml - 图像水印部分
image_watermark:
  algorithm: videoseal                  # 算法选择：videoseal（默认）, prc
  model_name: stabilityai/stable-diffusion-2-1-base
  resolution: 512                       # AI生成模式：图像分辨率
  num_inference_steps: 30               # AI生成模式：推理步数
  guidance_scale: 7.5                   # AI生成模式：引导系数
  lowres_attenuation: true              # VideoSeal：低分辨率衰减
  device: cuda
  
  # PRC-Watermark 特有配置（当algorithm=prc时生效）
  prc_config:
    decoder_model_path: models/dec_48b_whit.torchscript.pt
    noise_step: 50
    mode: exact                         # 模式选择：fast/accurate/exact
    
  # 上传文件模式配置
  upload_config:
    max_file_size: 10485760            # 最大上传文件大小 (10MB)
    supported_formats: [jpg, jpeg, png, bmp, webp]
```

**核心参数说明**:
- **AI生成模式**: `resolution`, `num_inference_steps`, `guidance_scale`控制生成质量
- **上传文件模式**: `max_file_size`和`supported_formats`控制文件上传限制
- **VideoSeal**: `lowres_attenuation`启用低分辨率优化，`replicate`和`chunk_size`提升检测精度
- **PRC**: `mode`选择检测精度级别，`noise_step`影响水印强度

##### 音频水印配置 (Audio Watermarking)
**修改文件**: `config/default_config.yaml` (audio_watermark section)
```yaml  
# config/default_config.yaml - 音频水印部分
audio_watermark:
  algorithm: audioseal                  # AudioSeal算法
  device: cuda
  nbits: 16                            # 消息编码位数
  sample_rate: 16000                   # 采样率
  
  # Bark TTS配置 (AI生成模式)
  bark_config:
    model_size: large                  # 模型大小：large/medium/small
    temperature: 0.8                   # 生成温度，控制随机性
    default_voice: v2/en_speaker_6     # 默认说话人音色
    cache_dir: ~/.cache/bark           # Bark模型缓存目录
    
  # 上传文件模式配置
  upload_config:
    max_file_size: 52428800           # 最大上传文件大小 (50MB)
    supported_formats: [wav, mp3, flac, aac, m4a]
    auto_resample: true               # 自动重采样到目标采样率
```

**核心参数说明**:
- **AI生成模式**: `bark_config`控制TTS质量，`temperature`影响语音自然度
- **上传文件模式**: `auto_resample`自动处理采样率不匹配问题  
- **AudioSeal**: `nbits=16`支持字符串消息编码，`sample_rate`需与输入音频匹配

##### 视频水印配置 (Video Watermarking)
**修改文件**: `config/default_config.yaml` (video_watermark section)
```yaml
# config/default_config.yaml - 视频水印部分  
video_watermark:
  # Wan2.1视频生成配置 (AI生成模式)
  wan_config:
    model_name: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    num_frames: 81                     # 视频帧数（5秒@15fps，Wan2.1推荐）
    height: 480                        # 视频高度（Wan2.1推荐480p）
    width: 832                         # 视频宽度（16:9比例）
    num_inference_steps: 50            # 推理步数（Wan2.1推荐）
    guidance_scale: 5.0                # 引导系数（Wan2.1推荐）
    device: cuda
    vram_requirement: 8GB              # 显存需求（Wan2.1仅需8GB）
    
  # VideoSeal水印配置  
  videoseal_config:
    model_path: ckpts/videoseal         # VideoSeal模型路径
    lowres_attenuation: true           # 低分辨率衰减
    device: cuda
    
  # 上传文件模式配置
  upload_config:
    max_file_size: 104857600          # 最大上传文件大小 (100MB)
    supported_formats: [mp4, avi, mov, mkv, flv, webm]
    transcode_for_web: true           # 自动转码为浏览器兼容格式
    target_codec: libx264             # 目标视频编解码器
    target_audio_codec: aac           # 目标音频编解码器
    enable_faststart: true            # 启用快速开始（Web优化）
```

**核心参数说明**:
- **AI生成模式**: `num_frames`控制视频长度，`height/width`控制分辨率，数值越高质量越好但耗时更长
- **上传文件模式**: `transcode_for_web=true`自动转码为H.264+AAC+faststart确保浏览器兼容
- **内存优化**: `enable_cpu_offload`在GPU内存不足时启用CPU卸载
- **质量平衡**: 降低分辨率和帧数可减少内存占用和处理时间

#### 🎯 快速参数调优指南

**提升生成质量**:
```yaml
# 图像：提高分辨率和推理步数
resolution: 1024
num_inference_steps: 50

# 视频：提高分辨率和帧数  
height: 1024
width: 1024
num_frames: 75

# 音频：使用更大的Bark模型
bark_config:
  model_size: large
  temperature: 0.7
```

**优化性能和内存**:
```yaml
# 降低分辨率和步数
resolution: 320
num_inference_steps: 20
height: 320
width: 512

# 启用内存优化
enable_cpu_offload: true
lowres_attenuation: true
```

**增强检测准确率**:
```yaml
# 图像VideoSeal检测优化
replicate: 16          # 单图复制为多帧 
chunk_size: 16         # 分块检测

# PRC精确模式
prc_config:
  mode: exact          # 最高精度检测
  noise_step: 50       # 标准噪声步数
```

## Web Demo Interface Features

### 🌐 统一Web界面 (templates/index.html)
项目提供了完整的Web演示界面，支持所有模态的水印操作：

**核心功能**:
- **双模式切换**: "AI生成内容" 和 "上传现有文件" 模式无缝切换
- **实时状态反馈**: 任务状态实时更新，支持进度显示和错误提示  
- **对比显示界面**: 自动显示原文件vs水印文件的并排对比
- **多媒体支持**: 支持文本、图像、音频、视频的Web播放和显示
- **文件下载**: 支持原文件和水印文件的独立下载

**界面特性**:
```javascript
// 动态模板切换
function toggleMode(modality) {
    // 根据选择的模态和模式显示相应的输入界面
    showTemplate(modality, selectedMode);
}

// 结果对比显示
function showComparison(modality, originalUrl, watermarkedUrl) {
    // 自动加载对比模板并显示before/after效果
    loadComparisonTemplate(modality, originalUrl, watermarkedUrl);
}
```


### 🔧 后端API支持 (app.py)
Flask后端提供完整的RESTful API：

**主要端点**:
```python
@app.route('/api/watermark/<modality>', methods=['POST'])
def watermark_endpoint(modality):
    # 统一水印处理端点，支持text/image/audio/video
    
@app.route('/api/files/<task_id>/original')
@app.route('/api/files/<task_id>/watermarked') 
def serve_file(task_id, file_type):
    # 文件服务端点，支持原文件和水印文件访问
    
@app.route('/api/status/<task_id>')
def get_task_status(task_id):
    # 任务状态查询，实时返回处理进度
```

## AIGC Content Identification Implementation Details

### Text Content Identification (CredID Hidden Watermarking)
- **仅支持AI生成模式**: 基于LLM的统计特征进行文本内容标识
- Multi-bit identification framework supporting multiple LLM vendors for AIGC text tracing
- Privacy-preserving design with Trusted Third Party (TTP) architecture
- Error correction codes (ECC) for robustness against various attacks and transformations
- Joint-voting mechanism for multi-party AIGC content identification
- **离线优先**: 优先使用本地缓存模型，支持完全离线AIGC识别运行

### Image Content Identification (Dual Backend Support)
**支持后端**: VideoSeal (默认), PRC-Watermark (可选)
**双模式支持**: AI生成模式 + 上传文件模式

#### VideoSeal 图像标识技术 (默认)
- **单帧视频处理**: 将图像视作单帧视频，复用VideoSeal深度学习标识算法
- **检测增强**: 支持`replicate`和`chunk_size`参数，通过多帧复制提升AIGC内容检测置信度
- **低分辨率优化**: `lowres_attenuation`参数优化低分辨率AIGC图像处理
- **AI生成模式**: Stable Diffusion 2.1 + VideoSeal标识，自动保存原图和标识图
- **上传文件模式**: 直接对上传的AIGC图像嵌入VideoSeal标识，支持多种图像格式

#### PRC-Watermark (可选后端)
- **完整的PRC标识系统**: 基于Stable Diffusion的伪随机纠错码AIGC内容标识
- **统一的exact_inversion实现**: 所有模式都使用相同的核心逆向函数，仅通过参数调节
- **多精度逆向模式**:
  - FAST模式: 20步推理，decoder_inv=False，快速AIGC内容检测
  - ACCURATE模式: 50步推理，decoder_inv=True，精度平衡检测
  - EXACT模式: 50步推理，decoder_inv=True，最高精度检测
- **100%检测成功率**: 所有模式都能完美检测并解码AIGC标识消息
- **本地模型支持**: 离线模式使用本地Stable Diffusion 2.1模型
- **简洁架构**: 统一的`_image_to_latents()`函数，消除代码冗余

#### 技术实现特点
- **懒加载架构**: 按需初始化具体后端，避免无关依赖加载
- **离线优先**: 强制本地模型加载，避免网络依赖
- **对比显示**: Web界面自动显示原图vs水印图的并排对比
- **格式支持**: JPG, PNG, BMP, WebP等主流图像格式

### Audio Content Identification (AudioSeal)
**双模式支持**: AI生成模式(Bark TTS) + 上传文件模式
**核心算法**: Meta AudioSeal 深度学习音频内容标识

#### 技术特性
- **16位消息编码**: 基于SHA256哈希的字符串消息编码系统，确保AIGC音频标识一致性
- **高保真嵌入**: SNR>40dB（实测44.45dB），听觉质量几乎无损失
- **双处理模式**:
  - **AI生成模式**: Bark TTS文本转语音 + AudioSeal标识嵌入（3-8秒生成）
  - **上传文件模式**: 直接对上传AIGC音频文件进行标识嵌入（0.93秒嵌入，0.04秒提取）
- **原文件保存**: 两种模式都自动保存原音频和标识音频，支持Web界面对比播放
- **设备自适应**: 支持CPU/CUDA自动切换和设备张量管理，修复设备不匹配问题
- **批处理支持**: 高效的批量AIGC音频处理能力（3个音频2.8秒）
- **格式兼容**: 支持WAV、MP3、FLAC、AAC、M4A等主流音频格式读写
- **鲁棒性验证**: 对不同SNR级别噪声的抗性测试（SNR≥10dB可靠检测）

#### Bark TTS集成
- **多语言支持**: 支持中英文等多语言高质量语音合成
- **音色选择**: 支持多种预设说话人音色，默认使用`v2/en_speaker_6`
- **智能缓存**: 优先使用本地缓存，支持符号链接和自定义缓存目录
- **参数控制**: `temperature`控制生成随机性，`model_size`控制模型质量

### Video Content Identification (Wan2.1 + VideoSeal)
**双模式支持**: AI生成模式(Wan2.1) + 上传文件模式
**技术栈**: Wan2.1文生视频 + VideoSeal视频内容标识

#### 核心特性
- **双处理流程**:
  - **AI生成模式**: Wan2.1文本生成视频 + VideoSeal标识嵌入
  - **上传文件模式**: 直接对上传AIGC视频文件进行VideoSeal标识嵌入
- **原文件保存**: 两种模式都自动保存原视频和标识视频，支持Web界面并排播放对比
- **浏览器兼容**: 自动转码为H.264+AAC+faststart格式，确保跨浏览器Web播放
- **内存优化**: 支持CPU内存卸载和VAE tiling，处理大分辨率AIGC视频，仅需8GB显存
- **离线优先**: 优先使用本地Wan2.1模型快照，避免网络下载

#### Wan2.1集成
- **模型支持**: 使用`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`（仅1.3B参数，更轻量）
- **帧数控制**: 推荐81帧（5秒@15fps），无严格格式限制
- **分辨率配置**: 推荐480p (832x480)，支持720p但不稳定
- **显存需求**: 仅需8GB VRAM，几乎兼容所有消费级GPU
- **生成时间**: RTX 4090上生成5秒480p视频约4分钟

#### VideoSeal视频水印
- **256位水印**: 支持长消息的字符串编码
- **分块检测**: 支持`max_frames`和`chunk_size`参数优化大视频处理
- **置信度评估**: 提供检测置信度和元数据信息
- **格式支持**: MP4, AVI, MOV, MKV, FLV, WebM等视频格式

#### 浏览器兼容转码
- **自动转码**: `transcode_for_web=true`启用H.264+AAC+faststart转码
- **编码器选择**: 默认使用libx264视频编码器和AAC音频编码器  
- **快速开始**: 启用faststart优化Web流媒体播放
- **文件管理**: 智能处理转码后的文件命名和访问

### AIGC Content Identification Unified Interface (Enhanced)
The `WatermarkTool` class in `src/unified/watermark_tool.py` provides comprehensive AIGC content identification capabilities:
- **Consistent AIGC API**: 统一的`embed()`和`extract()`接口支持所有模态的AIGC内容标识
- **Dual-mode support**: 每个模态都支持AI生成和文件上传两种AIGC内容处理模式
- **Dual-operation support**: 支持隐式标识(watermark)和显式标识(visible_mark)两种操作类型
- **Original content preservation**: 自动保存原文件和标识文件供before/after效果对比显示
- **Batch processing**: 批处理能力支持大规模AIGC内容处理
- **Technology switching**: 运行时标识技术切换（如VideoSeal/PRC图像后端）
- **Configuration management**: 跨模态配置管理和AIGC标识参数优化
- **Web integration**: 与Flask Web界面的无缝集成，提供完整AIGC标识展示
- **Error handling**: 完善的错误处理和降级策略，确保AIGC标识系统稳定性

## Working with Different Components

### 🔧 各模态开发和修改指南

#### 修改文本水印 (Text Watermarking)
**主要文件位置**:
- **核心算法**: `src/text_watermark/credid/` - CredID多方水印框架
- **配置文件**: `src/text_watermark/credid/config/` - 算法特定参数（CredID.json等）
- **统一配置**: `config/text_config.yaml` - 全局文本水印设置
- **测试脚本**: `src/text_watermark/credid/demo/` - 单方和多方场景演示

**修改流程**:
1. 算法参数调整 → 修改`config/text_config.yaml`或相应JSON配置文件
2. 模型路径设置 → 配置`model_name`和`hf_cache_dir`参数
3. 离线模式 → 设置`offline_mode: true`和相应环境变量
4. 测试验证 → 运行`src/text_watermark/credid/demo/`下的测试脚本

#### 修改图像水印 (Image Watermarking)  
**主要文件位置**:
- **统一接口**: `src/image_watermark/image_watermark.py` - 双后端支持的基类
- **VideoSeal后端**: `src/image_watermark/videoseal_image_watermark.py` - 默认后端实现
- **PRC后端**: `src/image_watermark/prc_watermark.py` - 可选PRC水印实现
- **配置文件**: `config/default_config.yaml` (image_watermark section)

**修改流程**:
1. **切换后端** → 修改`algorithm: videoseal|prc`配置
2. **AI生成参数** → 调整`resolution`, `num_inference_steps`, `guidance_scale`
3. **上传文件限制** → 修改`upload_config`中的`max_file_size`和`supported_formats`
4. **检测优化** → 配置VideoSeal的`replicate`和`chunk_size`参数
5. **测试验证** → 运行`python test_image_videoseal_root.py`或`python test_prc_only.py`

#### 修改音频水印 (Audio Watermarking)
**主要文件位置**:
- **统一接口**: `src/audio_watermark/audio_watermark.py` - 双模式音频水印基类
- **AudioSeal核心**: `src/audio_watermark/audioseal_wrapper.py` - 深度学习水印实现
- **Bark TTS**: `src/audio_watermark/bark_generator.py` - AI语音生成集成
- **工具函数**: `src/audio_watermark/utils.py` - 音频处理和质量评估
- **配置文件**: `config/default_config.yaml` (audio_watermark section)

**修改流程**:
1. **基础参数** → 调整`nbits`, `sample_rate`, `device`配置
2. **TTS设置** → 修改`bark_config`中的`model_size`, `temperature`, `default_voice`
3. **上传支持** → 配置`upload_config`的格式支持和文件大小限制  
4. **设备优化** → 根据硬件配置选择CPU/CUDA设备
5. **测试验证** → 运行`python tests/test_audio_watermark.py`完整测试套件

#### 修改视频水印 (Video Watermarking)
**主要文件位置**:
- **统一接口**: `src/video_watermark/video_watermark.py` - 双模式视频水印
- **Wan2.1生成器**: `src/video_watermark/wan_video_generator.py` - AI视频生成（新）
- **VideoSeal**: `src/video_watermark/videoseal_wrapper.py` - 视频水印算法
- **视频处理**: `src/video_watermark/utils.py` - 转码和I/O工具
- **配置文件**: `config/default_config.yaml` (video_watermark section)

**修改流程**:
1. **生成质量** → 调整`num_frames`, `height`, `width`, `num_inference_steps`
2. **内存优化** → 配置`enable_cpu_offload`和设备映射策略  
3. **浏览器兼容** → 设置`transcode_for_web`, `target_codec`等转码参数
4. **上传支持** → 修改`upload_config`的视频格式和大小限制
5. **测试验证** → 运行`python tests/test_video_watermark_demo.py`

#### 扩展统一接口 (Unified Interface)
**主要文件位置**:
- **核心工具**: `src/unified/watermark_tool.py` - 高层API封装
- **引擎核心**: `src/unified/unified_engine.py` - 底层执行引擎
- **Web集成**: `app.py` - Flask Web应用后端
- **前端界面**: `templates/index.html` - 用户交互界面

**扩展流程**:
1. **新功能接口** → 在`watermark_tool.py`中添加新方法
2. **引擎支持** → 在`unified_engine.py`中实现底层逻辑
3. **配置更新** → 修改`config/`目录下的YAML配置文件
4. **Web集成** → 更新`app.py`的API端点和`templates/index.html`的界面
5. **测试覆盖** → 添加相应的测试用例和演示脚本
## AudioSeal音频水印系统状态

### ✅ 已完成功能
- **完整AudioSeal集成**: Meta官方AudioSeal算法的完整Python封装
- **消息编码系统**: 基于SHA256哈希的16位消息编码，支持字符串到二进制的可靠转换
- **Bark TTS集成**: 完整的文本转语音功能，支持多语言和多音色
- **统一接口设计**: AudioWatermark基类提供与图像、文本水印一致的API
- **设备自适应**: 自动CPU/CUDA检测，内存优化和设备张量管理
- **批处理支持**: 高效的批量音频处理和水印操作
- **质量评估工具**: SNR、MSE、相关性等音频质量指标计算
- **多格式支持**: WAV、MP3、FLAC等音频格式的读写支持


## 变更摘要（2025-08）

### 背景
- 为兼容 Hunyuan 视频模型，环境升级至 `diffusers==0.34`。该版本与现有 PRC 图像水印路径存在不兼容风险（自定义管线模块注册差异）。

### 新增：VideoSeal 作为图像水印第二后端
- 在 `src/image_watermark/` 新增 `videoseal_image_watermark.py`，将单张图像视作单帧视频，复用 `src/video_watermark/videoseal_wrapper.py` 的 `embed/detect`。
- `src/image_watermark/image_watermark.py` 增加 `algorithm: videoseal` 分支，保持统一接口：
  - 直接对输入图像嵌入/提取
  - 或使用 Stable Diffusion 先生成图像，再用 VideoSeal 嵌入
- `src/unified/watermark_tool.py` 的 `get_supported_algorithms()['image']` 增加 `videoseal`。
- 检测增强：`VideoSealImageWatermark.extract(..., replicate=N, chunk_size=N)` 支持单图复制为多帧均值，提高读出稳定性与置信度。

### 懒加载与离线加载
- 懒加载：`ImageWatermark` 改为按需初始化具体后端，避免在构造时无关依赖（如 PRC/SD 管线）被加载。
- 离线加载（Stable Diffusion）：`src/utils/model_manager.py` 强制离线并解析本地 HF Hub 目录：
  - 优先解析 `.../huggingface/hub/models--stabilityai--stable-diffusion-2-1-base`（与 PRC 路径一致）
  - `from_pretrained(local_files_only=True)`，不触网

### 导入与测试可用性
- 统一 `src.*` 绝对导入，确保以项目根运行脚本时稳定。
- `tests/conftest.py` 将 `src/` 注入 `sys.path`，保证测试环境下 `unified.*` 可导入。
- 新增单测与演示：
  - `tests/test_image_videoseal.py`（最小验证）
  - 根级 `test_image_videoseal_root.py`（可 `python` 直接运行）：
    - `--mode pil`：现有图像嵌入/提取
    - `--mode gen`：生成→嵌入→提取（完全离线，需本地 SD 权重）

### 使用指引（VideoSeal 图像水印）
- 配置（示例）：
```yaml
image_watermark:
  algorithm: videoseal
  model_name: stabilityai/stable-diffusion-2-1-base
  resolution: 512
  num_inference_steps: 30
  lowres_attenuation: true
  device: cuda
```
- 代码：
```python
from src.unified.watermark_tool import WatermarkTool
tool = WatermarkTool()
tool.set_algorithm('image', 'videoseal')
img = tool.generate_image_with_watermark(prompt='a cat', message='hello_videoseal')
res = tool.extract_image_watermark(img, replicate=16, chunk_size=16)
```
- 命令行演示：
```bash
python test_image_videoseal_root.py --mode pil  --device cuda
python test_image_videoseal_root.py --mode gen  --device cuda --resolution 512 --steps 30
```

### 提升检测置信度建议
- 生成侧：提高 `resolution` 与 `num_inference_steps`；简化 prompt；使用 GPU。
- 检测侧：`replicate` 设置为 8~32 并与 `chunk_size` 对齐，用多帧均值稳定读出。



### 重要约定。
- 不要进行任何测试或者下载的操作，告诉我应该怎样执行，我会自己进行
- 回答问题用中文，写代码用英文