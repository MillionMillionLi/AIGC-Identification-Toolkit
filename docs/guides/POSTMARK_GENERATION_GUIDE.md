# PostMark AI文本生成+水印 使用指南

## 📖 概述

本指南介绍如何使用PostMark算法进行AI文本生成并自动嵌入水印。PostMark现在支持两种模式：

1. **AI生成模式** (新增): prompt → LLM生成 → 水印嵌入
2. **文件上传模式** (原有): 已有文本 → 水印嵌入

## 🔑 核心特性

### PostMark vs CredID

| 特性 | PostMark (默认) | CredID |
|------|----------------|--------|
| 嵌入方式 | 后处理（两步式） | 生成时嵌入 |
| LLM要求 | 黑盒LLM（无需logits） | 白盒LLM（需要logits） |
| 适用场景 | 任何LLM API | 自部署开源模型 |
| 生成模型 | Mistral-7B-Instruct | 用户提供 |
| 灵活性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### PostMark工作流程

```
用户Prompt
    ↓
Mistral-7B-Instruct 生成原始文本
    ↓
PostMark 选择水印词
    ↓
Mistral-7B-Instruct 插入水印词
    ↓
带水印文本 + 原始文本
```

## 🚀 快速开始

### 1. 配置文件设置

编辑 `config/text_config.yaml`:

```yaml
# 选择PostMark算法
algorithm: "postmark"

postmark:
  # 基础配置
  embedder: "nomic"
  inserter: "mistral-7b-inst"
  ratio: 0.12
  threshold: 0.7

  # AI生成配置 (新增)
  llm_for_generation: "mistral-7b-inst"  # 生成文本的LLM
  max_tokens: 600                        # 默认生成长度
  generation_temperature: 0.7            # 采样温度
```

### 2. 基础用法示例

#### 方式1: 使用WatermarkTool (推荐)

```python
from src.unified.watermark_tool import WatermarkTool

# 初始化工具
tool = WatermarkTool()

# AI生成模式：传入prompt
prompt = "Write a paragraph about machine learning"
watermarked_text = tool.embed(prompt, "my_watermark", 'text')

print(f"生成的带水印文本:\n{watermarked_text}")
```

#### 方式2: 使用UnifiedEngine

```python
from src.unified.unified_engine import UnifiedEngine

engine = UnifiedEngine()

# AI生成模式
watermarked = engine.embed(
    content="Explain neural networks",  # 这是prompt
    message="watermark_id",
    modality='text',
    operation='watermark',
    max_tokens=200  # 可选：覆盖默认值
)
```

#### 方式3: 使用TextWatermark

```python
from src.text_watermark.text_watermark import TextWatermark

watermark = TextWatermark(algorithm='postmark')

# 生成带水印文本
result = watermark.generate_with_watermark(
    prompt="Describe deep learning",
    message="test_msg",
    max_tokens=150,
    temperature=0.8  # 可选：调整创造性
)

print(result)  # 直接返回字符串
```

#### 方式4: 直接使用PostMarkWatermark

```python
from src.text_watermark.postmark_watermark import PostMarkWatermark

config = {
    'embedder': 'nomic',
    'inserter': 'mistral-7b-inst',
    'llm_for_generation': 'mistral-7b-inst',
    'ratio': 0.12,
    'max_tokens': 600,
    'generation_temperature': 0.7
}

watermark = PostMarkWatermark(config)

# 生成并嵌入水印
result = watermark.generate_with_watermark(
    prompt="Write about AI ethics",
    message="watermark_2025"
)

print(f"原始文本: {result['original_text']}")
print(f"带水印文本: {result['watermarked_text']}")
print(f"水印词: {result['watermark_words']}")
```

### 3. 文件上传模式（原有功能）

```python
from src.unified.watermark_tool import WatermarkTool

tool = WatermarkTool()

# 文件上传模式：传入已有文本 + text_input参数
existing_text = "This is a pre-generated text..."
watermarked = tool.embed(
    existing_text,
    "watermark_msg",
    'text',
    text_input=True  # 关键：标识这是文件上传模式
)
```

## 🔍 水印检测

```python
from src.text_watermark.postmark_watermark import PostMarkWatermark

watermark = PostMarkWatermark(config)

# 生成
result = watermark.generate_with_watermark(prompt, message)

# 检测（需要原始水印词列表）
detection = watermark.extract(
    watermarked_text=result['watermarked_text'],
    original_words=result['watermark_words']
)

print(f"检测到水印: {detection['detected']}")
print(f"置信度: {detection['confidence']:.2%}")
print(f"存在率: {detection['presence_score']:.2%}")
```

## ⚙️ 高级配置

### 自定义生成参数

```python
result = watermark.generate_with_watermark(
    prompt="Your prompt here",
    message="watermark_id",
    # 自定义参数
    llm="mistral-7b-inst",      # 选择LLM模型
    max_tokens=800,              # 生成更长文本
    temperature=0.9,             # 更高的创造性
)
```

### 可用的LLM模型

PostMark内置支持以下LLM模型：

1. **mistral-7b-inst** (推荐)
   - 7B参数，高质量生成
   - 已作为inserter使用，无需额外加载

2. **llama-3-8b-chat**
   - 8B参数，Meta最新模型
   - 需要额外加载

3. **llama-3-70b-chat**
   - 70B参数，最高质量
   - 需要API密钥（Together AI）

配置方法：

```yaml
# config/text_config.yaml
postmark:
  llm_for_generation: "llama-3-8b-chat"  # 切换到Llama-3
```

或在代码中动态指定：

```python
result = watermark.generate_with_watermark(
    prompt="...",
    message="...",
    llm="llama-3-8b-chat"  # 动态切换
)
```

## 🧪 测试

运行完整测试套件：

```bash
# 从项目根目录运行
python tests/test_postmark_generation.py

# 或者进入tests目录运行
cd tests
python test_postmark_generation.py
```

测试内容包括：
1. PostMarkWatermark直接调用
2. TextWatermark统一接口
3. UnifiedEngine AI生成模式
4. UnifiedEngine 文件上传模式

## 📊 性能参考

基于Mistral-7B-Instruct (GPU: RTX 4090):

| 操作 | 时间 | 备注 |
|------|------|------|
| 模型加载 | ~15秒 | 首次加载 |
| 文本生成 (100 tokens) | ~5秒 | Mistral-7B |
| 水印嵌入 | ~3秒 | PostMark后处理 |
| 水印检测 | <1秒 | 基于词存在率 |
| **总计** | **~8秒** | 单次完整流程 |

## 🔧 故障排除

### 1. LLM加载失败

**问题**: `RuntimeError: LLM初始化失败`

**解决方案**:
- 确保模型已下载到本地缓存
- 检查`cache_dir`配置是否正确
- 验证GPU/CUDA可用性

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 2. 生成文本过短

**问题**: 生成的文本只有几个词

**解决方案**:
- 增加`max_tokens`参数
- 调整prompt使其更具体
- 检查temperature设置（过低可能导致重复）

```python
result = watermark.generate_with_watermark(
    prompt="Write a detailed explanation of...",  # 更具体的prompt
    message="msg",
    max_tokens=800,  # 增加生成长度
    temperature=0.7   # 平衡确定性和创造性
)
```

### 3. 水印检测失败

**问题**: `detection['detected'] == False`

**解决方案**:
- 确保传入了`original_words`参数
- 检查水印文本是否被修改过
- 降低threshold阈值

```python
detection = watermark.extract(
    watermarked_text,
    original_words=result['watermark_words'],  # 必需
    threshold=0.6  # 降低阈值
)
```

## 📚 API参考

### generate_with_watermark()

```python
def generate_with_watermark(
    prompt: str,           # 生成提示词
    message: str,          # 水印消息（用于标识）
    **kwargs              # 额外参数
) -> Dict[str, Any]:
    """
    返回:
        {
            'watermarked_text': str,     # 带水印文本
            'original_text': str,        # 原始生成文本
            'watermark_words': List[str],# 水印词列表
            'message': str,              # 水印消息
            'success': bool,             # 是否成功
            'metadata': dict             # 元数据
        }
    """
```

**kwargs参数**:
- `llm`: LLM模型名称 (默认: config中的llm_for_generation)
- `max_tokens`: 最大生成长度 (默认: 600)
- `temperature`: 采样温度 (默认: 0.7)

## 🎯 最佳实践

1. **Prompt设计**
   - 使用具体、详细的prompt
   - 避免过于开放的问题
   - 示例: ❌ "AI" → ✅ "Explain how neural networks learn from data"

2. **参数调优**
   - 短文本: `max_tokens=100-200, temperature=0.6`
   - 中长文本: `max_tokens=400-600, temperature=0.7`
   - 创意内容: `max_tokens=800+, temperature=0.8-0.9`

3. **水印检测**
   - 始终保存`watermark_words`用于检测
   - 在数据库中存储`message`和`watermark_words`
   - 定期验证水印完整性

4. **性能优化**
   - 使用GPU加速 (`device: cuda`)
   - 批量处理时复用模型实例
   - 缓存常用模型到本地

## 🔗 相关资源

- 配置文件: `config/text_config.yaml`
- 核心实现: `src/text_watermark/postmark_watermark.py`
- 统一接口: `src/text_watermark/text_watermark.py`
- 引擎集成: `src/unified/unified_engine.py`
- 测试脚本: `test_postmark_generation.py`

## 💡 常见问题

**Q: PostMark和CredID哪个更好？**

A: 取决于使用场景：
- 使用第三方API（GPT-4, Claude）→ PostMark
- 自部署开源模型 → CredID或PostMark
- 需要黑盒支持 → PostMark
- 追求最高检测率 → CredID

**Q: 可以混合使用PostMark和CredID吗？**

A: 可以！在`config/text_config.yaml`中切换`algorithm`参数即可。两种算法的接口是统一的。

**Q: PostMark生成的文本质量如何？**

A: 取决于底层LLM模型：
- Mistral-7B-Instruct: 高质量，适合大多数场景
- Llama-3-8B: 更新更强，质量更高
- 用户可根据需求选择或自定义LLM

**Q: 水印会影响文本质量吗？**

A: PostMark设计为最小化影响：
- 水印词比例仅12%
- 使用语义相关词
- LLM自然插入，保持流畅性

## 🆕 更新日志

### 2025-11-04
- ✅ 新增 `generate_with_watermark()` 方法到 `PostMarkWatermark`
- ✅ 更新 `TextWatermark.generate_with_watermark()` 支持PostMark
- ✅ 更新 `UnifiedEngine.embed()` 区分AI生成和文件上传
- ✅ 增强 `config/text_config.yaml` AI生成配置
- ✅ 添加完整测试套件 `test_postmark_generation.py`

---

**维护者**: AI-Generated-Content-Identification-Toolkit Team
**最后更新**: 2025-11-04
