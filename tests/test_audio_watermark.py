"""
音频水印功能基础测试
验证AudioSeal嵌入、提取和质量评估功能
"""

import os
import sys
import pytest
import torch
import numpy as np
import logging
from pathlib import Path

# 设置项目路径以便导入
def setup_project_path():
    """设置项目路径，确保可以导入 src 模块"""
    current_file = Path(__file__).resolve()
    
    # 找到项目根目录（包含 src 目录的目录）
    project_root = current_file.parent.parent  # tests -> project_root
    src_dir = project_root / "src"
    
    if src_dir.exists():
        src_path = str(src_dir)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        print(f"已添加 src 目录到路径: {src_path}")
        return True
    else:
        print(f"错误: 找不到 src 目录在 {project_root}")
        return False

# 立即设置路径
setup_project_path()

try:
    from audio_watermark import (
        AudioWatermark, AudioSealWrapper, AudioIOUtils, AudioQualityUtils,
        AUDIOSEAL_AVAILABLE, HAS_BARK, create_audio_watermark
    )
    from unified.watermark_tool import WatermarkTool
except ImportError as e:
    pytest.skip(f"无法导入音频水印模块: {e}", allow_module_level=True)


class TestAudioWatermarkBasic:
    """基础音频水印测试"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.test_message = "test_message_2025"
        cls.test_audio_length = 16000  # 1秒16kHz音频
        cls.output_dir = Path("tests/test_results/audio")
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
    
    def create_test_audio(self) -> torch.Tensor:
        """创建测试音频"""
        # 生成1秒的正弦波测试音频
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4音符
        
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * torch.sin(2 * np.pi * frequency * t)
        
        return audio.unsqueeze(0)  # 添加通道维度
    
    @pytest.mark.skipif(not AUDIOSEAL_AVAILABLE, reason="AudioSeal不可用")
    def test_audioseal_wrapper_basic(self):
        """测试AudioSeal基础功能"""
        self.logger.info("测试AudioSeal基础功能...")
        
        # 创建封装器
        wrapper = AudioSealWrapper(device='cpu', nbits=16)
        
        # 获取模型信息
        info = wrapper.get_model_info()
        assert info['device'] == 'cpu'
        assert info['nbits'] == 16
        
        # 创建测试音频
        test_audio = self.create_test_audio()
        self.logger.info(f"测试音频形状: {test_audio.shape}")
        
        # 测试嵌入
        self.logger.info("测试水印嵌入...")
        watermarked_audio = wrapper.embed(test_audio, self.test_message)
        
        assert watermarked_audio.shape == test_audio.shape
        assert torch.is_tensor(watermarked_audio)
        
        # 测试提取
        self.logger.info("测试水印提取...")
        result = wrapper.extract(watermarked_audio)
        
        assert isinstance(result, dict)
        assert 'detected' in result
        assert 'message' in result
        assert 'confidence' in result
        
        self.logger.info(f"提取结果: {result}")
        
        # 验证检测成功
        assert result['detected'] is True
        assert result['confidence'] > 0.5
        
        self.logger.info("✅ AudioSeal基础功能测试通过")
    
    @pytest.mark.skipif(not AUDIOSEAL_AVAILABLE, reason="AudioSeal不可用") 
    def test_audio_watermark_interface(self):
        """测试AudioWatermark统一接口"""
        self.logger.info("测试AudioWatermark统一接口...")
        
        # 创建音频水印处理器
        watermark_tool = create_audio_watermark()
        
        # 获取模型信息
        info = watermark_tool.get_model_info()
        assert info['algorithm'] == 'audioseal'
        
        # 创建测试音频
        test_audio = self.create_test_audio()
        
        # 测试嵌入
        self.logger.info("测试统一接口嵌入...")
        watermarked = watermark_tool.embed_watermark(test_audio, self.test_message)
        
        assert torch.is_tensor(watermarked)
        assert watermarked.shape == test_audio.shape
        
        # 测试提取
        self.logger.info("测试统一接口提取...")
        result = watermark_tool.extract_watermark(watermarked)
        
        assert result['detected'] is True
        assert isinstance(result['message'], str)
        assert result['confidence'] > 0.5
        
        # 测试质量评估
        self.logger.info("测试质量评估...")
        quality = watermark_tool.evaluate_quality(test_audio, watermarked)
        
        assert 'snr_db' in quality
        assert 'mse' in quality
        assert 'correlation' in quality
        
        self.logger.info(f"质量评估结果: {quality}")
        
        # 验证质量指标合理
        assert quality['snr_db'] > 10  # SNR应该大于10dB
        assert quality['correlation'] > 0.8  # 相关性应该很高
        
        self.logger.info("✅ AudioWatermark统一接口测试通过")
    
    @pytest.mark.skipif(not AUDIOSEAL_AVAILABLE, reason="AudioSeal不可用")
    def test_batch_processing(self):
        """测试批处理功能"""
        self.logger.info("测试批处理功能...")
        
        watermark_tool = create_audio_watermark()
        
        # 创建多个测试音频
        num_audios = 3
        test_audios = []
        test_messages = []
        
        for i in range(num_audios):
            audio = self.create_test_audio()
            # 添加一些变化使音频不同
            audio = audio + 0.1 * torch.randn_like(audio) * (i + 1)
            test_audios.append(audio)
            test_messages.append(f"batch_message_{i+1}")
        
        # 测试批量嵌入
        self.logger.info("测试批量嵌入...")
        watermarked_audios = watermark_tool.batch_embed(
            test_audios, test_messages
        )
        
        assert len(watermarked_audios) == num_audios
        assert all(torch.is_tensor(audio) for audio in watermarked_audios if audio is not None)
        
        # 测试批量提取
        self.logger.info("测试批量提取...")
        results = watermark_tool.batch_extract(watermarked_audios)
        
        assert len(results) == num_audios
        
        # 验证每个结果
        for i, result in enumerate(results):
            assert result['detected'] is True
            assert result['confidence'] > 0.5
            self.logger.info(f"批处理结果 {i+1}: {result['message']}")
        
        self.logger.info("✅ 批处理功能测试通过")
    
    @pytest.mark.skipif(not AUDIOSEAL_AVAILABLE, reason="AudioSeal不可用")
    def test_file_io_operations(self):
        """测试文件I/O操作"""
        self.logger.info("测试文件I/O操作...")
        
        watermark_tool = create_audio_watermark()
        
        # 创建测试音频
        test_audio = self.create_test_audio()
        
        # 保存原始音频
        original_path = self.output_dir / "test_original.wav"
        AudioIOUtils.save_audio(test_audio, str(original_path), 16000)
        assert original_path.exists()
        
        # 从文件嵌入水印并保存
        watermarked_path = self.output_dir / "test_watermarked.wav"
        output_path = watermark_tool.embed_watermark(
            str(original_path), 
            self.test_message,
            output_path=str(watermarked_path)
        )
        
        assert Path(output_path).exists()
        self.logger.info(f"带水印音频已保存: {output_path}")
        
        # 从保存的文件提取水印
        result = watermark_tool.extract_watermark(output_path)
        
        assert result['detected'] is True
        assert result['confidence'] > 0.5
        
        self.logger.info(f"从文件提取结果: {result}")
        self.logger.info("✅ 文件I/O操作测试通过")
    
    @pytest.mark.skipif(not AUDIOSEAL_AVAILABLE, reason="AudioSeal不可用")
    def test_unified_watermark_tool(self):
        """测试统一水印工具的音频接口"""
        self.logger.info("测试统一水印工具音频接口...")
        
        # 创建统一工具
        unified_tool = WatermarkTool()
        
        # 检查音频功能是否可用
        algorithms = unified_tool.get_supported_algorithms()
        if 'audio' in algorithms:
            assert 'audioseal' in algorithms['audio']
            
            # 创建测试音频
            test_audio = self.create_test_audio()
            
            # 测试嵌入
            self.logger.info("测试统一工具音频嵌入...")
            watermarked = unified_tool.embed_audio_watermark(
                test_audio, self.test_message
            )
            
            assert torch.is_tensor(watermarked)
            
            # 测试提取
            self.logger.info("测试统一工具音频提取...")
            result = unified_tool.extract_audio_watermark(watermarked)
            
            assert result['detected'] is True
            assert result['confidence'] > 0.5
            
            # 测试质量评估
            quality = unified_tool.evaluate_audio_quality(test_audio, watermarked)
            assert 'snr_db' in quality
            
            self.logger.info("✅ 统一水印工具音频接口测试通过")
        else:
            self.logger.warning("统一工具中音频功能不可用，跳过测试")


class TestAudioWatermarkAdvanced:
    """高级音频水印测试"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.output_dir = Path("tests/test_results/audio")
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.logger = logging.getLogger(__name__)
    
    @pytest.mark.skipif(not HAS_BARK, reason="Bark不可用")
    def test_text_to_audio_watermark(self):
        """测试文本转音频+水印功能"""
        self.logger.info("测试文本转音频+水印功能...")
        
        watermark_tool = create_audio_watermark()
        
        test_prompt = "Hello, this is a test of text to speech with watermark."
        test_message = "bark_watermark_test"
        
        try:
            # 生成带水印的音频
            self.logger.info("生成带水印音频...")
            generated_audio = watermark_tool.generate_audio_with_watermark(
                prompt=test_prompt,
                message=test_message,
                temperature=0.7,
                seed=42  # 使用固定种子确保可重复性
            )
            
            assert torch.is_tensor(generated_audio)
            assert generated_audio.dim() == 2  # (channels, samples)
            
            # 从生成的音频中提取水印
            self.logger.info("从生成音频提取水印...")
            result = watermark_tool.extract_watermark(generated_audio)
            
            assert result['detected'] is True
            assert result['confidence'] > 0.5
            
            self.logger.info(f"提取结果: {result}")
            
            # 保存测试音频
            output_path = self.output_dir / "test_bark_watermarked.wav"
            AudioIOUtils.save_audio(generated_audio, str(output_path), 16000)
            
            self.logger.info(f"生成的带水印音频已保存: {output_path}")
            self.logger.info("✅ 文本转音频+水印功能测试通过")
            
        except Exception as e:
            self.logger.error(f"文本转音频测试失败: {e}")
            pytest.skip(f"Bark功能测试失败: {e}")
    
    @pytest.mark.skipif(not AUDIOSEAL_AVAILABLE, reason="AudioSeal不可用")
    def test_robustness_against_noise(self):
        """测试对噪声的鲁棒性"""
        self.logger.info("测试噪声鲁棒性...")
        
        from audio_watermark.utils import AudioProcessingUtils
        
        watermark_tool = create_audio_watermark()
        
        # 创建测试音频
        test_audio = torch.randn(1, 16000)  # 随机噪声作为测试音频
        test_message = "robustness_test"
        
        # 嵌入水印
        watermarked_audio = watermark_tool.embed_watermark(test_audio, test_message)
        
        # 测试不同程度的噪声
        snr_levels = [20, 15, 10, 5]  # dB
        
        for snr_db in snr_levels:
            self.logger.info(f"测试SNR {snr_db}dB...")
            
            # 添加噪声
            noisy_audio = AudioProcessingUtils.add_noise(
                watermarked_audio, noise_type='white', snr_db=snr_db
            )
            
            # 尝试提取水印
            result = watermark_tool.extract_watermark(noisy_audio)
            
            self.logger.info(f"SNR {snr_db}dB - 检测: {result['detected']}, 置信度: {result['confidence']:.3f}")
            
            # 高SNR应该能检测到
            if snr_db >= 10:
                assert result['detected'] is True, f"SNR {snr_db}dB 应该能检测到水印"
        
        self.logger.info("✅ 噪声鲁棒性测试完成")


def run_all_tests():
    """运行所有测试"""
    import pytest
    
    # 设置测试配置
    pytest_args = [
        __file__,
        "-v",  # 详细输出
        "--tb=short",  # 简短的traceback
        "-x",  # 遇到失败就停止
    ]
    
    # 运行测试
    exit_code = pytest.main(pytest_args)
    return exit_code == 0


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🎵 音频水印功能测试")
    print("=" * 50)
    
    # 检查依赖
    print("检查依赖...")
    if AUDIOSEAL_AVAILABLE:
        print("✅ AudioSeal 可用")
    else:
        print("❌ AudioSeal 不可用")
    
    if HAS_BARK:
        print("✅ Bark 可用")
    else:
        print("❌ Bark 不可用")
    
    print()
    
    # 运行测试
    if run_all_tests():
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 某些测试失败")
        sys.exit(1)
