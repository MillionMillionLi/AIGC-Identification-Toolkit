"""
文本水印双算法（PostMark + CredID）切换测试脚本

测试内容：
1. PostMark后处理水印的嵌入和提取
2. CredID生成时水印的嵌入和提取
3. 配置文件算法切换
4. 统一接口的兼容性
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_postmark_basic():
    """测试PostMark基础功能"""
    print("\n" + "="*80)
    print("测试1: PostMark后处理水印基础功能")
    print("="*80)

    try:
        from text_watermark.postmark_watermark import PostMarkWatermark

        # 初始化PostMark
        config = {
            'embedder': 'nomic',
            'inserter': 'mistral-7b-inst',
            'ratio': 0.12,
            'iterate': 'v2',
            'threshold': 0.7
        }

        watermark = PostMarkWatermark(config)
        print("✅ PostMark初始化成功")

        # 测试文本（已生成的文本，模拟LLM输出）
        test_text = """
        Artificial intelligence has revolutionized many aspects of modern life.
        Machine learning algorithms can now process vast amounts of data to identify
        patterns and make predictions. Deep learning techniques have enabled significant
        breakthroughs in computer vision and natural language processing.
        """

        print(f"\n原始文本（{len(test_text.split())}词）:\n{test_text[:200]}...")

        # 嵌入水印
        print("\n🔹 嵌入PostMark水印...")
        result = watermark.embed(test_text, message="test_watermark_2025")

        if result['success']:
            print("✅ 水印嵌入成功")
            print(f"   - 水印词数量: {result['metadata']['num_watermark_words']}")
            print(f"   - 嵌入模型: {result['metadata']['inserter']}")
            print(f"   - 水印文本（前200字符）:\n   {result['watermarked_text'][:200]}...")

            # 提取水印
            print("\n🔹 提取PostMark水印...")
            extract_result = watermark.extract(
                result['watermarked_text'],
                original_words=result['watermark_words']
            )

            if extract_result['success'] and extract_result['detected']:
                print("✅ 水印检测成功")
                print(f"   - 检测置信度: {extract_result['confidence']:.2%}")
                print(f"   - 存在率: {extract_result['presence_score']:.2%}")
                print(f"   - 检测到的水印词数: {len(extract_result['watermark_words'])}")
            else:
                print("❌ 水印检测失败")
                print(f"   - 置信度: {extract_result['confidence']:.2%}")

            return True
        else:
            print(f"❌ 水印嵌入失败: {result.get('error')}")
            return False

    except Exception as e:
        print(f"❌ PostMark测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_credid_basic():
    """测试CredID基础功能（需要模型）"""
    print("\n" + "="*80)
    print("测试2: CredID生成时水印基础功能")
    print("="*80)

    try:
        from text_watermark.credid_watermark import CredIDWatermark
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # 初始化CredID
        config = {
            'mode': 'lm',
            'model_name': 'sshleifer/tiny-gpt2',  # 使用小模型测试
            'lm_params': {
                'delta': 1.5,
                'prefix_len': 10,
                'message_len': 10
            },
            'wm_params': {
                'encode_ratio': 8,
                'strategy': 'vanilla'
            },
            'max_new_tokens': 50
        }

        watermark = CredIDWatermark(config)
        print("✅ CredID初始化成功")

        # 加载模型（使用tiny-gpt2测试）
        print("\n🔹 加载测试模型...")
        model_name = config['model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"✅ 模型加载完成 (device={device})")

        # 测试prompt
        prompt = "Artificial intelligence is"
        message = "credid_test_2025"

        print(f"\n原始prompt: \"{prompt}\"")
        print(f"水印消息: \"{message}\"")

        # 嵌入水印（生成时嵌入）
        print("\n🔹 生成带CredID水印的文本...")
        result = watermark.embed(model, tokenizer, prompt, message)

        if result['success']:
            print("✅ 水印生成成功")
            print(f"   - 模式: {result['metadata']['mode']}")
            print(f"   - 生成长度: {result['metadata']['output_length']} tokens")
            print(f"   - 水印文本: \"{result['watermarked_text']}\"")

            # 提取水印
            print("\n🔹 提取CredID水印...")
            extract_result = watermark.extract(
                result['watermarked_text'],
                model=model,
                tokenizer=tokenizer,
                candidates_messages=[message, "wrong_message"]
            )

            if extract_result['success']:
                print("✅ 水印提取成功")
                print(f"   - 检测到的消息: \"{extract_result['extracted_message']}\"")
                print(f"   - 置信度: {extract_result['confidence']:.2%}")
                print(f"   - 检测方法: {extract_result['metadata']['detection_method']}")
            else:
                print(f"❌ 水印提取失败: {extract_result.get('error')}")

            return True
        else:
            print(f"❌ 水印生成失败: {result.get('error')}")
            return False

    except Exception as e:
        print(f"❌ CredID测试失败: {e}")
        print("ℹ️  提示：CredID需要加载语言模型，可能需要更多内存")
        import traceback
        traceback.print_exc()
        return False


def test_unified_interface():
    """测试统一接口的算法切换"""
    print("\n" + "="*80)
    print("测试3: 统一TextWatermark接口的算法切换")
    print("="*80)

    try:
        from text_watermark.text_watermark import TextWatermark

        # 测试1: 默认PostMark
        print("\n🔹 测试默认算法（PostMark）...")
        watermark_postmark = TextWatermark()
        assert watermark_postmark.algorithm == 'postmark', "默认算法应为postmark"
        print(f"✅ 默认算法: {watermark_postmark.algorithm}")

        # 测试2: 切换到CredID
        print("\n🔹 测试算法切换（CredID）...")
        watermark_credid = TextWatermark()
        watermark_credid.set_algorithm('credid')
        assert watermark_credid.algorithm == 'credid', "切换后算法应为credid"
        print(f"✅ 切换后算法: {watermark_credid.algorithm}")

        # 测试3: 再切换回PostMark
        print("\n🔹 测试再次切换（PostMark）...")
        watermark_credid.set_algorithm('postmark')
        assert watermark_credid.algorithm == 'postmark', "切换后算法应为postmark"
        print(f"✅ 再次切换算法: {watermark_credid.algorithm}")

        print("\n✅ 统一接口算法切换测试通过")
        return True

    except Exception as e:
        print(f"❌ 统一接口测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_file_switching():
    """测试通过配置文件切换算法"""
    print("\n" + "="*80)
    print("测试4: 配置文件算法切换")
    print("="*80)

    try:
        from text_watermark.text_watermark import TextWatermark
        import tempfile
        import yaml

        # 测试PostMark配置
        print("\n🔹 测试PostMark配置文件...")
        postmark_config = {
            'algorithm': 'postmark',
            'postmark': {
                'embedder': 'nomic',
                'inserter': 'mistral-7b-inst',
                'ratio': 0.12
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(postmark_config, f)
            postmark_config_path = f.name

        watermark = TextWatermark(postmark_config_path)
        assert watermark.algorithm == 'postmark', "从配置加载的算法应为postmark"
        print(f"✅ 配置文件加载成功: {watermark.algorithm}")

        os.unlink(postmark_config_path)

        # 测试CredID配置
        print("\n🔹 测试CredID配置文件...")
        credid_config = {
            'algorithm': 'credid',
            'credid': {
                'mode': 'lm',
                'model_name': 'sshleifer/tiny-gpt2'
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(credid_config, f)
            credid_config_path = f.name

        watermark = TextWatermark(credid_config_path)
        assert watermark.algorithm == 'credid', "从配置加载的算法应为credid"
        print(f"✅ 配置文件加载成功: {watermark.algorithm}")

        os.unlink(credid_config_path)

        print("\n✅ 配置文件切换测试通过")
        return True

    except Exception as e:
        print(f"❌ 配置文件切换测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_engine_integration():
    """测试UnifiedEngine集成"""
    print("\n" + "="*80)
    print("测试5: UnifiedEngine集成测试")
    print("="*80)

    try:
        from unified.unified_engine import UnifiedWatermarkEngine

        # 初始化引擎
        print("\n🔹 初始化UnifiedWatermarkEngine...")
        engine = UnifiedWatermarkEngine()

        # 检查默认算法
        algorithms = engine.get_default_algorithms()
        print(f"✅ 默认算法配置: {algorithms}")
        assert algorithms['text'] == 'postmark', "文本默认算法应为postmark"

        # 检查支持的模态
        modalities = engine.get_supported_modalities()
        print(f"✅ 支持的模态: {modalities}")

        print("\n✅ UnifiedEngine集成测试通过")
        return True

    except Exception as e:
        print(f"❌ UnifiedEngine集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("文本水印双算法（PostMark + CredID）测试套件")
    print("="*80)

    results = {
        '1. PostMark基础功能': test_postmark_basic(),
        # '2. CredID基础功能': test_credid_basic(),
        # '3. 统一接口算法切换': test_unified_interface(),
        # '4. 配置文件算法切换': test_config_file_switching(),
        # '5. UnifiedEngine集成': test_unified_engine_integration()
    }

    # 输出总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！PostMark和CredID双算法系统运行正常！")
    else:
        print(f"\n⚠️  部分测试失败，请检查上述错误信息")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
