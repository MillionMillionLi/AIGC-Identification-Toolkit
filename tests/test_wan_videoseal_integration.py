"""
Wan2.1 + VideoSeal 水印完整集成测试
测试Wan2.1视频生成与VideoSeal水印嵌入、提取的完整流程
"""
# python3 tests/test_wan_videoseal_integration.py -v
import os
import sys
import logging
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.video_watermark.video_watermark import VideoWatermark


class TestWanVideoSealIntegration(unittest.TestCase):
    """Wan2.1 + VideoSeal 完整集成测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        cls.logger.info("=" * 70)
        cls.logger.info("开始 Wan2.1 + VideoSeal 完整集成测试")
        cls.logger.info("=" * 70)

        # 创建测试结果目录
        cls.test_results_dir = project_root / "tests" / "test_results"
        cls.test_results_dir.mkdir(parents=True, exist_ok=True)

        # 创建VideoWatermark实例
        cls.video_watermark = VideoWatermark()
        cls.logger.info("✓ VideoWatermark实例创建成功")

    def test_01_basic_watermark_embedding(self):
        """测试1: 基础水印嵌入（文生视频 + 水印）"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试1: 基础水印嵌入（文生视频 + 水印）")
        self.logger.info("=" * 60)

        try:
            # 生成视频并嵌入水印
            test_prompt = "A cat walks on the grass, realistic"
            test_message = "integration_test_2025"

            self.logger.info(f"提示词: {test_prompt}")
            self.logger.info(f"水印消息: {test_message}")
            self.logger.info("参数: 81帧, 480x832分辨率, 50推理步数")

            output_path = str(self.test_results_dir / "test_integration_basic.mp4")

            watermarked_path = self.video_watermark.generate_video_with_watermark(
                prompt=test_prompt,
                message=test_message,
                num_frames=81,
                height=480,
                width=832,
                num_inference_steps=50,
                guidance_scale=5.0,
                seed=42,
                output_path=output_path,
                return_original=False
            )

            # 验证输出
            self.assertTrue(Path(watermarked_path).exists(), "水印视频文件应该存在")
            file_size = Path(watermarked_path).stat().st_size / (1024 * 1024)  # MB
            self.logger.info(f"✓ 水印视频已保存: {watermarked_path}")
            self.logger.info(f"✓ 文件大小: {file_size:.2f} MB")
            self.assertGreater(file_size, 0.5, "文件大小应该 > 0.5 MB")

            # 保存路径供后续测试使用
            self.__class__.watermarked_video_path = watermarked_path

        except Exception as e:
            self.logger.error(f"❌ 基础水印嵌入失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_02_watermark_extraction(self):
        """测试2: 水印提取与验证"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试2: 水印提取与验证")
        self.logger.info("=" * 60)

        try:
            # 确保测试1已运行
            if not hasattr(self.__class__, 'watermarked_video_path'):
                self.skipTest("需要先运行测试1")

            watermarked_path = self.__class__.watermarked_video_path
            self.logger.info(f"从视频提取水印: {watermarked_path}")

            # 提取水印
            result = self.video_watermark.extract_watermark(watermarked_path)

            # 验证结果
            self.logger.info("提取结果:")
            self.logger.info(f"  detected: {result['detected']}")
            self.logger.info(f"  message: {result['message']}")
            self.logger.info(f"  confidence: {result.get('confidence', 'N/A')}")

            self.assertTrue(result['detected'], "应该检测到水印")
            self.assertEqual(result['message'], "integration_test_2025", "水印消息应该匹配")
            self.logger.info("✓ 水印提取验证通过")

        except Exception as e:
            self.logger.error(f"❌ 水印提取失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    # def test_03_with_original_video_saved(self):
    #     """测试3: 保存原始视频和水印视频对比"""
    #     self.logger.info("\n" + "=" * 60)
    #     self.logger.info("测试3: 保存原始视频和水印视频对比")
    #     self.logger.info("=" * 60)

    #     try:
    #         test_prompt = "A dog running on the beach"
    #         test_message = "original_comparison_test"

    #         self.logger.info(f"提示词: {test_prompt}")
    #         self.logger.info(f"水印消息: {test_message}")
    #         self.logger.info("return_original=True - 同时保存原始视频和水印视频")

    #         output_path = str(self.test_results_dir / "test_integration_with_original.mp4")

    #         result = self.video_watermark.generate_video_with_watermark(
    #             prompt=test_prompt,
    #             message=test_message,
    #             num_frames=81,
    #             height=480,
    #             width=832,
    #             num_inference_steps=50,
    #             guidance_scale=5.0,
    #             seed=123,
    #             output_path=output_path,
    #             return_original=True  # 关键：保存原始视频
    #         )

    #         # 验证返回结果
    #         self.assertIsInstance(result, dict, "应该返回字典")
    #         self.assertIn('original', result, "应该包含原始视频路径")
    #         self.assertIn('watermarked', result, "应该包含水印视频路径")

    #         original_path = result['original']
    #         watermarked_path = result['watermarked']

    #         self.logger.info(f"✓ 原始视频: {original_path}")
    #         self.logger.info(f"✓ 水印视频: {watermarked_path}")

    #         # 验证两个文件都存在
    #         self.assertTrue(Path(original_path).exists(), "原始视频应该存在")
    #         self.assertTrue(Path(watermarked_path).exists(), "水印视频应该存在")

    #         # 验证文件大小
    #         original_size = Path(original_path).stat().st_size / (1024 * 1024)
    #         watermarked_size = Path(watermarked_path).stat().st_size / (1024 * 1024)

    #         self.logger.info(f"  原始视频大小: {original_size:.2f} MB")
    #         self.logger.info(f"  水印视频大小: {watermarked_size:.2f} MB")

    #         self.assertGreater(original_size, 0.5, "原始视频大小应该 > 0.5 MB")
    #         self.assertGreater(watermarked_size, 0.5, "水印视频大小应该 > 0.5 MB")

    #         # 验证水印检测
    #         watermark_result = self.video_watermark.extract_watermark(watermarked_path)
    #         self.assertTrue(watermark_result['detected'], "水印视频应该检测到水印")
    #         self.assertEqual(watermark_result['message'], test_message, "消息应该匹配")
    #         self.logger.info("✓ 水印视频检测通过")

    #         # 验证原始视频无水印（可选）
    #         try:
    #             original_result = self.video_watermark.extract_watermark(original_path)
    #             self.logger.info(f"  原始视频检测结果: {original_result['detected']}")
    #             if original_result['detected']:
    #                 self.logger.warning("⚠ 原始视频也检测到水印（可能是误报）")
    #         except Exception as e:
    #             self.logger.info(f"  原始视频检测失败（预期）: {e}")

    #     except Exception as e:
    #         self.logger.error(f"❌ 原始视频保存测试失败: {e}")
    #         import traceback
    #         self.logger.error(traceback.format_exc())
    #         raise

    # def test_04_different_parameters(self):
    #     """测试4: 不同参数配置的水印嵌入"""
    #     self.logger.info("\n" + "=" * 60)
    #     self.logger.info("测试4: 不同参数配置的水印嵌入")
    #     self.logger.info("=" * 60)

    #     # 测试较小的参数配置（更快）
    #     configs = [
    #         {
    #             "name": "快速配置（49帧，320p）",
    #             "params": {
    #                 "num_frames": 49,
    #                 "height": 320,
    #                 "width": 512,
    #                 "num_inference_steps": 30,
    #             },
    #             "message": "fast_config_test"
    #         },
    #     ]

    #     for config in configs:
    #         try:
    #             self.logger.info(f"\n测试配置: {config['name']}")
    #             self.logger.info(f"  参数: {config['params']}")

    #             output_path = str(self.test_results_dir / f"test_integration_{config['message']}.mp4")

    #             watermarked_path = self.video_watermark.generate_video_with_watermark(
    #                 prompt="A beautiful sunset over mountains",
    #                 message=config['message'],
    #                 output_path=output_path,
    #                 seed=456,
    #                 **config['params']
    #             )

    #             # 验证生成
    #             self.assertTrue(Path(watermarked_path).exists(), f"{config['name']} 视频应该存在")
    #             file_size = Path(watermarked_path).stat().st_size / (1024 * 1024)
    #             self.logger.info(f"  ✓ 视频已保存: {watermarked_path} ({file_size:.2f} MB)")

    #             # 验证水印
    #             result = self.video_watermark.extract_watermark(watermarked_path)
    #             self.assertTrue(result['detected'], f"{config['name']} 应该检测到水印")
    #             self.assertEqual(result['message'], config['message'], "消息应该匹配")
    #             self.logger.info(f"  ✓ 水印验证通过")

    #         except Exception as e:
    #             self.logger.error(f"❌ {config['name']} 测试失败: {e}")
    #             # 不抛出异常，继续下一个配置
    #             continue

    # def test_05_system_info(self):
    #     """测试5: 获取系统信息"""
    #     self.logger.info("\n" + "=" * 60)
    #     self.logger.info("测试5: 获取系统信息")
    #     self.logger.info("=" * 60)

    #     try:
    #         info = self.video_watermark.get_system_info()

    #         self.logger.info("系统信息:")
    #         for key, value in info.items():
    #             if isinstance(value, dict):
    #                 self.logger.info(f"  {key}:")
    #                 for sub_key, sub_value in value.items():
    #                     self.logger.info(f"    {sub_key}: {sub_value}")
    #             else:
    #                 self.logger.info(f"  {key}: {value}")

    #         # 验证关键信息存在
    #         self.assertIn('device', info, "应该包含设备信息")
    #         self.assertIn('wan_model', info, "应该包含Wan模型信息")

    #         # 验证Wan模型信息
    #         wan_info = info['wan_model']
    #         self.assertTrue(wan_info['exists'], "Wan2.1模型应该存在")
    #         self.assertEqual(wan_info['repo_id'], "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    #         self.logger.info("✓ 系统信息验证通过")

    #     except Exception as e:
    #         self.logger.error(f"❌ 获取系统信息失败: {e}")
    #         raise

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        cls.logger.info("\n" + "=" * 70)
        cls.logger.info("Wan2.1 + VideoSeal 完整集成测试完成")
        cls.logger.info("=" * 70)

        # 清理管道以释放内存
        if hasattr(cls, 'video_watermark'):
            try:
                cls.video_watermark.clear_cache()
                cls.logger.info("✓ 缓存已清理")
            except Exception as e:
                cls.logger.warning(f"清理缓存时出错: {e}")

        # 显示生成的文件
        cls.logger.info("\n生成的文件:")
        test_results = cls.test_results_dir
        if test_results.exists():
            for video_file in sorted(test_results.glob("test_integration_*.mp4")):
                size = video_file.stat().st_size / (1024 * 1024)
                cls.logger.info(f"  - {video_file.name} ({size:.2f} MB)")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
