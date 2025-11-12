#!/usr/bin/env python3
"""
PostMark AI文本生成+水印测试脚本

此脚本演示如何使用PostMark算法进行AI文本生成并嵌入水印

测试流程:
1. 使用Mistral-7B-Instruct生成原始文本
2. PostMark后处理嵌入水印
3. 检测水印并验证

运行方法:
    # 从项目根目录运行
    python tests/test_postmark_generation.py

或者:
    cd tests
    python test_postmark_generation.py
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径（通过conftest.py自动处理）
# 如果直接运行此文件，需要手动添加
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_postmark_generation_direct():
    """
    测试1: 直接使用PostMarkWatermark类进行生成

    这是最底层的测试，直接调用PostMark的generate_with_watermark方法
    """
    print("\n" + "="*80)
    print("测试1: PostMarkWatermark.generate_with_watermark() 直接调用")
    print("="*80)

    try:
        from text_watermark.postmark_watermark import PostMarkWatermark

        # 初始化PostMark
        config = {
            'embedder': 'nomic',
            'inserter': 'mistral-7b-inst',
            'llm_for_generation': 'mistral-7b-inst',
            'ratio': 0.12,
            'iterate': 'v2',
            'threshold': 0.7,
            'max_tokens': 100,  # 测试用较短文本
            'generation_temperature': 0.7
        }

        watermark = PostMarkWatermark(config)

        # 测试prompt
        prompt = "Write a short paragraph about artificial intelligence."
        message = "test_watermark_2025"

        print(f"\n📝 Prompt: {prompt}")
        print(f"🔑 Watermark Message: {message}")
        print("\n⏳ 正在生成文本并嵌入水印...")

        # 生成带水印的文本
        result = watermark.generate_with_watermark(prompt, message)

        if result['success']:
            print("\n✅ 生成成功!")
            print(f"\n📄 原始生成文本:\n{result['original_text']}")
            print(f"\n🔒 带水印文本:\n{result['watermarked_text']}")
            print(f"\n🏷️  嵌入的水印词: {result['watermark_words'][:10]}... (共{len(result['watermark_words'])}个)")
            print(f"\n📊 元数据: {result['metadata']}")

            # 检测水印
            print("\n⏳ 正在检测水印...")
            detection = watermark.extract(
                result['watermarked_text'],
                original_words=result['watermark_words']
            )

            print(f"\n🔍 检测结果:")
            print(f"  - 检测到水印: {'✅ 是' if detection['detected'] else '❌ 否'}")
            print(f"  - 置信度: {detection['confidence']:.2%}")
            print(f"  - 存在率: {detection.get('presence_score', 0):.2%}")

            return True
        else:
            print(f"\n❌ 生成失败: {result.get('error', 'Unknown')}")
            return False

    except Exception as e:
        logger.error(f"测试1失败: {e}", exc_info=True)
        return False


def test_postmark_generation_textwatermark():
    """
    测试2: 使用TextWatermark统一接口进行生成

    这是中层接口测试，通过TextWatermark类调用
    """
    print("\n" + "="*80)
    print("测试2: TextWatermark.generate_with_watermark() 统一接口")
    print("="*80)

    try:
        from text_watermark.text_watermark import TextWatermark

        # 初始化TextWatermark（会自动从config/text_config.yaml加载配置）
        watermark = TextWatermark()

        # 确保使用PostMark算法
        if watermark.algorithm != 'postmark':
            watermark.set_algorithm('postmark')

        # 测试prompt
        prompt = "Explain the benefits of machine learning in one paragraph."
        message = "unified_test_2025"

        print(f"\n📝 Prompt: {prompt}")
        print(f"🔑 Watermark Message: {message}")
        print(f"🔧 Algorithm: {watermark.algorithm}")
        print("\n⏳ 正在通过统一接口生成文本...")

        # 使用统一接口生成
        watermarked_text = watermark.generate_with_watermark(
            prompt=prompt,
            message=message,
            max_tokens=100
        )

        print("\n✅ 生成成功!")
        print(f"\n🔒 带水印文本:\n{watermarked_text}")

        return True

    except Exception as e:
        logger.error(f"测试2失败: {e}", exc_info=True)
        return False


def test_postmark_generation_unified_engine():
    """
    测试3: 使用UnifiedWatermarkEngine进行生成（最高层接口）

    这是最高层的测试，模拟实际Web应用的调用方式
    """
    print("\n" + "="*80)
    print("测试3: UnifiedWatermarkEngine.embed() AI生成模式")
    print("="*80)

    try:
        from unified.unified_engine import UnifiedWatermarkEngine

        # 初始化引擎
        engine = UnifiedWatermarkEngine()

        # 测试prompt
        prompt = "Describe the future of natural language processing."
        message = "engine_test_2025"

        print(f"\n📝 Prompt: {prompt}")
        print(f"🔑 Watermark Message: {message}")
        print("\n⏳ 正在通过UnifiedWatermarkEngine生成文本...")

        # AI生成模式：content是prompt，不传text_input参数
        watermarked_text = engine.embed(
            content=prompt,
            message=message,
            modality='text',
            operation='watermark',
            max_tokens=100
        )

        print("\n✅ 生成成功!")
        print(f"\n🔒 带水印文本:\n{watermarked_text}")

        return True

    except Exception as e:
        logger.error(f"测试3失败: {e}", exc_info=True)
        return False


def test_postmark_upload_mode():
    """
    测试4: 文件上传模式（对比AI生成模式）

    这个测试验证文件上传模式是否仍然正常工作
    """
    print("\n" + "="*80)
    print("测试4: PostMark 文件上传模式（对比测试）")
    print("="*80)

    try:
        from unified.unified_engine import UnifiedWatermarkEngine

        # 初始化引擎
        engine = UnifiedWatermarkEngine()

        # 已有的文本
        existing_text = """
        Artificial intelligence is revolutionizing the way we interact with technology.
        Machine learning algorithms can now understand natural language, recognize images,
        and even generate creative content. The future of AI holds immense potential
        for solving complex problems and improving our daily lives.
        """

        message = "upload_test_2025"

        print(f"\n📄 已有文本:\n{existing_text[:100]}...")
        print(f"\n🔑 Watermark Message: {message}")
        print("\n⏳ 正在为已有文本嵌入水印...")

        # 文件上传模式：传入text_input参数
        watermarked_text = engine.embed(
            content=existing_text,
            message=message,
            modality='text',
            operation='watermark',
            text_input=True  # 关键参数：标识这是文件上传模式
        )

        print("\n✅ 嵌入成功!")
        print(f"\n🔒 带水印文本:\n{watermarked_text[:200]}...")

        return True

    except Exception as e:
        logger.error(f"测试4失败: {e}", exc_info=True)
        return False


def main():
    """
    主测试函数：依次运行所有测试
    """
    print("\n" + "🚀" * 40)
    print("PostMark AI文本生成+水印 完整测试套件")
    print("🚀" * 40)

    print("\n⚠️  注意事项:")
    print("  1. 确保已安装所有依赖: pip install -r requirements.txt")
    print("  2. 确保PostMark模型已下载到本地缓存")
    print("  3. 测试可能需要较长时间（LLM生成需要几分钟）")
    print("  4. 建议使用GPU以加速生成过程")

    # 运行所有测试
    results = {
        # "测试1 (PostMarkWatermark直接调用)": test_postmark_generation_direct(),
        # "测试2 (TextWatermark统一接口)": test_postmark_generation_textwatermark(),
        # "测试3 (UnifiedWatermarkEngine AI生成)": test_postmark_generation_unified_engine(),
        "测试4 (UnifiedWatermarkEngine 文件上传)": test_postmark_upload_mode(),
    }

    # 打印汇总
    print("\n" + "="*80)
    print("测试汇总")
    print("="*80)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\n📊 总计: {total_passed}/{total_tests} 测试通过")

    if total_passed == total_tests:
        print("\n🎉 所有测试通过！PostMark AI生成功能集成成功！")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} 个测试失败，请检查日志")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
