"""
Wan2.1视频生成器基础功能测试
测试Wan2.1模型的加载、视频生成和基本功能
"""

import os
import sys
import logging
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.video_watermark.wan_video_generator import WanVideoGenerator, create_wan_generator
from src.video_watermark.model_manager import ModelManager


class TestWanVideoBasic(unittest.TestCase):
    """Wan2.1视频生成器基础功能测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        cls.logger.info("=" * 70)
        cls.logger.info("开始Wan2.1视频生成器基础功能测试")
        cls.logger.info("=" * 70)

        # 设置缓存目录
        cls.cache_dir = "/fs-computility/wangxuhong/limeilin/.cache/huggingface/hub"

        # 创建测试结果目录
        cls.test_results_dir = project_root / "tests" / "test_results"
        cls.test_results_dir.mkdir(parents=True, exist_ok=True)

    def test_01_model_manager_wan_detection(self):
        """测试1: ModelManager检测Wan2.1本地模型"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试1: 检测本地Wan2.1模型")
        self.logger.info("=" * 60)

        model_manager = ModelManager(self.cache_dir)

        # 检查模型是否存在
        model_exists = model_manager._check_wan_model_exists()
        self.logger.info(f"本地Wan2.1模型存在: {model_exists}")

        if model_exists:
            # 获取模型路径
            model_path = model_manager.get_wan_model_path()
            self.logger.info(f"模型路径: {model_path}")
            self.assertTrue(Path(model_path).exists(), "模型路径应该存在")

            # 获取模型信息
            model_info = model_manager.get_wan_model_info()
            self.logger.info("模型信息:")
            for key, value in model_info.items():
                self.logger.info(f"  {key}: {value}")

            self.assertTrue(model_info["exists"], "模型应该存在")
            self.assertEqual(model_info["repo_id"], "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        else:
            self.logger.warning("⚠ Wan2.1模型未找到，请先下载模型到:")
            self.logger.warning(f"  {model_manager.wan_model_dir}")
            self.skipTest("Wan2.1模型未找到")

    def test_02_wan_generator_creation(self):
        """测试2: 创建Wan2.1生成器"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试2: 创建Wan2.1生成器")
        self.logger.info("=" * 60)

        try:
            generator = create_wan_generator(cache_dir=self.cache_dir)
            self.assertIsNotNone(generator, "生成器应该成功创建")
            self.logger.info("✓ Wan2.1生成器创建成功")

            # 获取生成器信息
            info = generator.get_pipeline_info()
            self.logger.info("生成器信息:")
            for key, value in info.items():
                self.logger.info(f"  {key}: {value}")

            self.assertEqual(info["model"], "Wan2.1-T2V-1.3B-Diffusers")
            self.logger.info("✓ 生成器信息验证通过")

        except Exception as e:
            self.logger.error(f"❌ 生成器创建失败: {e}")
            raise

    def test_03_pipeline_loading(self):
        """测试3: Wan2.1管道加载"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试3: Wan2.1管道加载")
        self.logger.info("=" * 60)

        try:
            generator = create_wan_generator(cache_dir=self.cache_dir)

            # 触发管道加载
            self.logger.info("开始加载Wan2.1管道...")
            generator._load_pipeline(allow_download=False)

            self.assertIsNotNone(generator.pipeline, "管道应该成功加载")
            self.logger.info("✓ Wan2.1管道加载成功")

            # 验证管道组件
            info = generator.get_pipeline_info()
            self.assertTrue(info["pipeline_loaded"], "管道应该已加载")
            self.logger.info(f"✓ 管道设备: {info['device']}")
            self.logger.info(f"✓ 数据类型: {info.get('dtype', 'unknown')}")

        except Exception as e:
            self.logger.error(f"❌ 管道加载失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_04_video_generation(self):
        """测试4: 基础视频生成（短视频）"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试4: 基础视频生成")
        self.logger.info("=" * 60)

        try:
            generator = create_wan_generator(cache_dir=self.cache_dir)

            # 生成短视频（81帧，480p，用于快速测试）
            test_prompt = "A cat walks on the grass, realistic"
            self.logger.info(f"提示词: {test_prompt}")
            self.logger.info("参数: 81帧, 480x832分辨率, 50推理步数")

            video_tensor = generator.generate_video_tensor(
                prompt=test_prompt,
                num_frames=81,
                height=480,
                width=832,
                num_inference_steps=50,
                guidance_scale=5.0,
                seed=42
            )

            # 验证输出
            self.assertIsInstance(video_tensor, torch.Tensor, "输出应该是torch.Tensor")
            self.assertEqual(video_tensor.dim(), 4, "应该是4维tensor (frames, channels, H, W)")
            self.logger.info(f"✓ 视频tensor形状: {video_tensor.shape}")
            self.logger.info(f"✓ 数据类型: {video_tensor.dtype}")
            self.logger.info(f"✓ 值域: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")

            # 验证值域在[0, 1]之间
            self.assertTrue(video_tensor.min() >= 0.0, "最小值应该 >= 0")
            self.assertTrue(video_tensor.max() <= 1.0, "最大值应该 <= 1")
            self.logger.info("✓ 值域验证通过")

            # 验证帧数和分辨率
            frames, channels, height, width = video_tensor.shape
            self.assertEqual(frames, 81, "帧数应该为81")
            self.assertEqual(channels, 3, "通道数应该为3 (RGB)")
            self.assertEqual(height, 480, "高度应该为480")
            self.assertEqual(width, 832, "宽度应该为832")
            self.logger.info("✓ 视频参数验证通过")

            # 检查是否非黑屏（平均像素值应该 > 0.1）
            mean_value = video_tensor.mean().item()
            self.logger.info(f"✓ 平均像素值: {mean_value:.3f}")
            self.assertGreater(mean_value, 0.1, "视频不应该是黑屏")
            self.logger.info("✓ 非黑屏验证通过")

        except Exception as e:
            self.logger.error(f"❌ 视频生成失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_05_video_generation_with_save(self):
        """测试5: 视频生成并保存文件"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试5: 视频生成并保存文件")
        self.logger.info("=" * 60)

        try:
            generator = create_wan_generator(cache_dir=self.cache_dir)

            # 生成视频并保存
            test_prompt = "A cat walks on the grass, realistic"
            output_path = self.test_results_dir / "test_wan_basic_output.mp4"

            self.logger.info(f"提示词: {test_prompt}")
            self.logger.info(f"输出路径: {output_path}")

            result_path = generator.generate_video(
                prompt=test_prompt,
                num_frames=81,
                height=480,
                width=832,
                num_inference_steps=50,
                guidance_scale=5.0,
                seed=42,
                output_path=str(output_path)
            )

            # 验证文件存在
            self.assertTrue(Path(result_path).exists(), "输出文件应该存在")
            file_size = Path(result_path).stat().st_size / (1024 * 1024)  # MB
            self.logger.info(f"✓ 视频已保存: {result_path}")
            self.logger.info(f"✓ 文件大小: {file_size:.2f} MB")
            self.assertGreater(file_size, 0.1, "文件大小应该 > 0.1 MB")

        except Exception as e:
            self.logger.error(f"❌ 视频保存失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_06_pipeline_info(self):
        """测试6: 获取管道详细信息"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试6: 获取管道详细信息")
        self.logger.info("=" * 60)

        try:
            generator = create_wan_generator(cache_dir=self.cache_dir)
            generator._load_pipeline(allow_download=False)

            info = generator.get_pipeline_info()
            self.logger.info("完整管道信息:")
            for key, value in info.items():
                self.logger.info(f"  {key}: {value}")

            # 验证关键信息
            self.assertEqual(info["model"], "Wan2.1-T2V-1.3B-Diffusers")
            self.assertTrue(info["pipeline_loaded"])
            self.assertTrue(info["trust_remote_code"])
            self.assertIn("device", info)
            self.logger.info("✓ 管道信息验证通过")

        except Exception as e:
            self.logger.error(f"❌ 获取管道信息失败: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        cls.logger.info("\n" + "=" * 70)
        cls.logger.info("Wan2.1视频生成器基础功能测试完成")
        cls.logger.info("=" * 70)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
