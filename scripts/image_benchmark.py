#!/usr/bin/env python3
"""
W-Bench 图像水印鲁棒性评估脚本

用法:
    # 使用默认配置（VideoSeal）
    python image_benchmark.py

    # 使用自定义配置文件
    python image_benchmark.py --config benchmarks/W-Bench/configs/my_algorithm.yaml

    # 快速测试（仅10张图像）
    python image_benchmark.py --max-images 10

    # 指定GPU设备
    python image_benchmark.py --device cuda:1
"""

import sys
import argparse
import importlib.util
from pathlib import Path


def load_benchmark_runner():
    """动态加载 BenchmarkRunner（处理带连字符的目录名）"""
    spec = importlib.util.spec_from_file_location(
        'benchmark_runner',
        'benchmarks/W-Bench/evaluators/benchmark_runner.py'
    )
    runner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner_module)
    return runner_module.BenchmarkRunner


def main():
    """主函数：解析参数并运行评估"""
    parser = argparse.ArgumentParser(
        description='W-Bench 图像水印鲁棒性评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # VideoSeal 完整评估（1000张图像）
  python image_benchmark.py

  # PRC-Watermark 评估
  python image_benchmark.py --config benchmarks/W-Bench/configs/prc_distortion.yaml

  # 快速测试（10张图像）
  python image_benchmark.py --max-images 10

  # 自定义输出目录
  python image_benchmark.py --output benchmarks/W-Bench/results/my_test
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='benchmarks/W-Bench/configs/videoseal_distortion.yaml',
        help='配置文件路径（默认: VideoSeal配置）'
    )

    parser.add_argument(
        '--max-images', '-n',
        type=int,
        default=None,
        help='最大测试图像数量（用于快速测试，默认: 使用配置文件中的设置）'
    )

    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='计算设备（cuda/cpu，默认: 使用配置文件中的设置）'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出目录（默认: 使用配置文件中的设置）'
    )

    args = parser.parse_args()

    # 确保在项目根目录
    sys.path.insert(0, '.')

    # 检查配置文件是否存在
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        print(f"   可用配置: benchmarks/W-Bench/configs/")
        return 1

    print("=" * 70)
    print("🚀 W-Bench 图像水印鲁棒性评估")
    print("=" * 70)
    print(f"📄 配置文件: {args.config}")
    if args.max_images:
        print(f"🖼️  测试图像: {args.max_images} 张（快速测试模式）")
    if args.device:
        print(f"💻 计算设备: {args.device}")
    if args.output:
        print(f"📂 输出目录: {args.output}")
    print("=" * 70)
    print()

    try:
        # 加载 BenchmarkRunner
        BenchmarkRunner = load_benchmark_runner()

        # 创建运行器
        runner = BenchmarkRunner(str(config_path))

        # TODO: 支持命令行参数覆盖配置
        # 这需要修改 BenchmarkRunner 的实现来支持运行时配置覆盖

        # 运行评估
        results = runner.run(max_images=args.max_images)

        print("\n" + "=" * 70)
        print("🎉 评估完成!")
        print("=" * 70)
        print(f"\n📊 结果已保存到: {runner.output_dir / 'metrics.json'}")
        print(f"📁 生成的文件:")
        print(f"   - 水印图像: {runner.output_dir / 'watermarked/'}")
        print(f"   - 攻击图像: {runner.output_dir / 'attacked/'}")
        print(f"   - 评估指标: {runner.output_dir / 'metrics.json'}")
        print()

        # 显示简要结果摘要
        if results and 'quality_metrics' in results:
            print("📈 质量指标摘要（原图 vs 水印图）:")
            for metric, value in results['quality_metrics'].items():
                print(f"   {metric.upper()}: {value:.4f}")

        if results and 'robustness_by_attack' in results:
            print("\n🎯 鲁棒性摘要:")
            for attack_type, strengths in results['robustness_by_attack'].items():
                if strengths:
                    rates = [s['detection_rate'] for s in strengths.values()]
                    avg_rate = sum(rates) / len(rates)
                    print(f"   {attack_type:<12}: {avg_rate:.2%} 平均检测率")

        print()
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
        return 130
    except Exception as e:
        print(f"\n\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())