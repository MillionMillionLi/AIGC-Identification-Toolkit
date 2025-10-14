# AIGC标识技术开发套件

<div align="center">
  <!-- 项目logo占位符 - 需要logo图片 -->
  <!-- <a href="https://github.com/your-repo-link">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->


  <h3 align="center">AIGC标识技术开发套件</h3>

  <p align="center">
    一站式开源标识技术开发套件，支持文本、图像、音频和视频内容的显式标识，隐式标识和隐水印功能
    <br />
    <a href="#使用方法"><strong>快速开始 »</strong></a>
    <br />
    <br />
    <a href="pictures/watermark.mp4">在线演示</a>
    ·
    <a href="https://github.com/your-repo-link/issues">报告问题</a>
    ·
    <a href="https://github.com/your-repo-link/issues">请求功能</a>
  </p>

</div>

<!-- 目录 -->

<details>
  <summary>目录</summary>
  <ol>
    <li>
      <a href="#关于项目">关于项目</a>
      <ul>
        <li><a href="#构建技术">构建技术</a></li>
      </ul>
    </li>
    <li>
      <a href="#开始使用">开始使用</a>
      <ul>
        <li><a href="#前置要求">前置要求</a></li>
        <li><a href="#安装">安装</a></li>
      </ul>
    </li>
    <li><a href="#使用方法">使用方法</a></li>
    <li><a href="#网页演示">网页演示</a></li>
    <li><a href="#api参考">API参考</a></li>
    <li><a href="#性能">性能</a></li>
    <li><a href="#发展路线">发展路线</a></li>
    <li><a href="#贡献">贡献</a></li>
    <li><a href="#许可证">许可证</a></li>
    <li><a href="#联系">联系</a></li>
    <li><a href="#致谢">致谢</a></li>
  </ol>
</details>


## 关于项目

<!-- 项目截图占位符 - 需要网页界面截图 -->
<!-- [![产品截图][product-screenshot]](https://example.com) -->

本项目提供一站式开源标识技术开发套件。支持文本、图像、音频和视频四大模态，具备显式标识、隐式标识和隐水印功能，全面覆盖GB 45438-2025《网络安全技术 人工智能生成合成内容标识方法》"标准规定的标识范围。

### 为什么选择我们？

- **全面覆盖**：支持GB 45438-2025标准要求的所有标识方法
- **多模态支持**：统一处理文本、图像、音频和视频内容
- **双模式操作**：既支持AI内容生成，也支持现有文件处理
- **生产就绪**：配备完整的网页界面、批量处理和性能优化

### 构建技术

* [![Python][Python.org]][Python-url]
[![PyTorch][PyTorch.org]][PyTorch-url]
[![Flask][Flask.palletsprojects.com]][Flask-url] [![Transformers][Transformers-badge]][Transformers-url] [![Diffusers][Diffusers-badge]][Diffusers-url]

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 开始使用

### 前置要求

- Python 3.8或更高版本
- 支持CUDA的GPU（推荐用于最佳性能）
- FFmpeg（视频处理必需）

### 安装

1. 克隆仓库

   ```bash
   git clone https://github.com/your-repo-link/unified_watermark_tool.git
   cd unified_watermark_tool
   ```

2. 安装核心依赖

   ```bash
   pip install -r requirements.txt
   ```

3. 安装模态特定依赖（可选）

   ```bash
   # 文本水印
   pip install -r src/text_watermark/credid/watermarking/MPAC/requirements.txt
   
   # 图像水印（PRC后端）
   pip install -r src/image_watermark/PRC-Watermark/requirements.txt
   
   # 音频水印
   pip install torch torchaudio julius soundfile librosa scipy matplotlib
   
   # 高级音频功能（Bark TTS）
   pip install git+https://github.com/suno-ai/bark.git
   ```

4. 配置环境（离线模式可选）

   ```bash
   export TRANSFORMERS_OFFLINE=1
   export HF_HUB_OFFLINE=1
   export HF_ENDPOINT=https://hf-mirror.com  # 中国用户
   ```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 使用方法






## API参考

本节提供统一水印工具的完整API参考，包括Python API和REST API接口。

### 统一水印工具API (WatermarkTool)

`WatermarkTool`是推荐的主要入口点，提供统一的接口支持所有模态的水印和显式标识操作。

#### 初始化

```python
from src.unified.watermark_tool import WatermarkTool

# 使用默认配置初始化
tool = WatermarkTool()

# 使用自定义配置初始化
tool = WatermarkTool(config_path="path/to/config.yaml")
```

#### 核心方法

##### embed() - 嵌入水印或显式标识

```python
def embed(self,
          content: Union[str, Path],
          message: str,
          modality: str,
          operation: str = 'watermark',
          **kwargs) -> Union[str, PIL.Image, torch.Tensor, Path]:
    """
    嵌入水印或添加显式标识

    Args:
        content: 输入内容
            - 文本模态: 提示文本(AI生成模式)或原始文本(显式标识)
            - 图像模态: 提示文本(AI生成)或图像文件路径(上传模式)
            - 音频模态: 提示文本(AI生成)或音频文件路径(上传模式)
            - 视频模态: 提示文本(AI生成)或视频文件路径(上传模式)
        message: 要嵌入的水印信息或显式标识文本
        modality: 模态类型 ('text', 'image', 'audio', 'video')
        operation: 操作类型 ('watermark' 或 'visible_mark')
        **kwargs: 模态特定参数

    Returns:
        处理后的内容（格式根据模态而定）
    """
```

**使用示例：**

```python
# 隐式水印（默认operation='watermark'）
text_wm = tool.embed("Please write a story", "my_message", 'text')
img_wm = tool.embed("a cat under the sun", "img_msg", 'image')
audio_wm = tool.embed("Hello world", "audio_msg", 'audio')
video_wm = tool.embed("阳光洒在海面上", "video_msg", 'video')

# 上传文件模式
img_wm = tool.embed("", "file_msg", 'image', image_input="/path/to/image.jpg")
audio_wm = tool.embed("", "audio_msg", 'audio', audio_input="/path/to/audio.wav")

# 显式标识
marked_text = tool.embed("原始文本", "本内容由AI生成", 'text',
                        operation='visible_mark', position='start')
marked_img = tool.embed("/path/to/image.jpg", "AI标识", 'image',
                       operation='visible_mark', position='bottom_right')
```

##### extract() - 提取水印或检测显式标识

```python
def extract(self,
           content: Union[str, PIL.Image, torch.Tensor, Path],
           modality: str,
           operation: str = 'watermark',
           **kwargs) -> Dict[str, Any]:
    """
    提取水印或检测显式标识

    Args:
        content: 待检测的内容
        modality: 模态类型
        operation: 操作类型 ('watermark' 或 'visible_mark')
        **kwargs: 检测参数

    Returns:
        检测结果字典:
        {
            'detected': bool,      # 是否检测到水印/标识
            'message': str,        # 提取的消息内容
            'confidence': float,   # 置信度 (0.0-1.0)
            'metadata': dict       # 额外的元数据信息
        }
    """
```

**使用示例：**

```python
# 提取隐式水印
text_result = tool.extract(watermarked_text, 'text')
img_result = tool.extract(watermarked_image, 'image', replicate=16)
audio_result = tool.extract(watermarked_audio, 'audio')
video_result = tool.extract(watermarked_video, 'video')

# 检测显式标识
mark_result = tool.extract(marked_content, 'text', operation='visible_mark')
```


## 发展路线

- [x] **水印实现**
  - [x] CredID文本水印
  - [x] VideoSeal图像/视频水印
  - [x] AudioSeal音频水印
  - [x] PRC-Watermark替代后端

- [x] **显式标识实现**
  - [x] 文本内容标注
  - [x] 图像叠加标记
  - [x] 视频叠加标记（FFmpeg）
  - [x] 音频语音标注（Bark TTS）

- [x] **网页界面**
  - [x] 双模式支持（AI生成+文件上传）
  - [x] 识别方法选择（不可见/可见）
  - [x] 实时对比显示
  - [x] 响应式设计
  - [x] 浏览器兼容的媒体转码

- [x] **合规与标准**
  - [x] GB 45438-2025合规
  - [x] 标准标记文本模板
  - [x] 可配置的定位和样式
  - [x] 多模态统一方法

- [ ] **未来增强**
  - [ ] 其他水印算法
  - [ ] 移动应用界面
  - [ ] 云部署选项
  - [ ] 高级分析仪表板
  - [ ] 界面多语言支持

查看[开放问题](https://github.com/your-repo-link/issues)获取完整的功能提议和已知问题列表。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 致谢

* [Meta AudioSeal](https://github.com/facebookresearch/audioseal) - 音频水印算法
* [VideoSeal](https://github.com/facebookresearch/videoseal) - 视频水印技术
* [Bark TTS](https://github.com/suno-ai/bark) - 文本转语音合成
* [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) - 文本转视频生成
* [Hugging Face](https://huggingface.co) - 模型托管和transformers库
* [PyTorch](https://pytorch.org) - 深度学习框架

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

<!-- MARKDOWN链接和图像 -->

[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org/
[PyTorch.org]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Flask.palletsprojects.com]: https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/
[Transformers-badge]: https://img.shields.io/badge/🤗%20Transformers-FFD700?style=for-the-badge
[Transformers-url]: https://huggingface.co/transformers/
[Diffusers-badge]: https://img.shields.io/badge/🧨%20Diffusers-FF6B6B?style=for-the-badge
[Diffusers-url]: https://huggingface.co/docs/diffusers/