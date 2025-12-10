#!/usr/bin/env python3
"""
一键运行所有实验
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import yaml


def load_config():
    """加载配置文件"""
    config_path = Path("config/experiment_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def run_experiment(script_path, log_file=None):
    """运行单个实验脚本"""
    script_path = Path(script_path)
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return False

    print(f"\n{'=' * 60}")
    print(f"🔬 运行实验: {script_path.name}")
    print('=' * 60)

    try:
        # 运行脚本
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        # 记录日志
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"脚本: {script_path}\n")
                f.write(f"返回码: {result.returncode}\n")
                f.write("\n--- STDOUT ---\n")
                f.write(result.stdout)
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
                f.write("\n" + "=" * 60 + "\n")

        if result.returncode == 0:
            print(f"✅ 完成: {script_path.name}")
            if result.stdout.strip():
                print("📝 输出:", result.stdout[:300] + "..." if len(result.stdout) > 300 else result.stdout)
            return True
        else:
            print(f"❌ 失败: {script_path.name}")
            if result.stderr.strip():
                print("💥 错误:", result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
            return False

    except Exception as e:
        print(f"💥 异常: {e}")
        return False


def main():
    """主函数：按顺序运行所有实验"""
    parser = argparse.ArgumentParser(description="运行GBDT反事实分析实验")
    parser.add_argument("--skip-need", action="store_true", help="跳过Need部分实验")
    parser.add_argument("--skip-approach", action="store_true", help="跳过Approach部分实验")
    parser.add_argument("--skip-benefits", action="store_true", help="跳过Benefits部分实验")
    parser.add_argument("--skip-competitors", action="store_true", help="跳过Competitors部分实验")
    parser.add_argument("--only", choices=["need", "approach","benefits", "competitors"], help="只运行指定部分")
    args = parser.parse_args()

    # 加载配置
    config = load_config()

    print("=" * 60)
    print("🎯 GBDT Counterfactual Analysis - 实验运行器")
    print("=" * 60)

    # 实验脚本列表
    experiments = {
        "need": [
            "experiments/01_need/exp_2_1_data_challenges.py",
            "experiments/01_need/exp_2_2_traditional_limitations.py",
        ],
        "approach": [
            "experiments/02_approach/exp_3_1_structure_analysis.py",

        ],
        "benefits": [

            "experiments/03_benefits/exp_4_1_learning_curve_data_time.py",
            "experiments/03_benefits/exp_4_2_all_metrics.py",

        ],
        "competitors": [
            "experiments/04_competitors/exp_5_1_performance.py",

        ]
    }

    # 根据参数决定运行哪些实验
    if args.only:
        parts_to_run = [args.only]
    else:
        parts_to_run = []
        if not args.skip_need:
            parts_to_run.append("need")
        if not args.skip_approach:
            parts_to_run.append("approach")
        if not args.skip_benefits:
            parts_to_run.append("benefits")
        if not args.skip_competitors:
            parts_to_run.append("competitors")

    # 运行实验
    successful = []
    failed = []

    for part in parts_to_run:
        print(f"\n📂 运行 {part.upper()} 部分:")
        print("-" * 40)

        for script in experiments[part]:
            log_file = f"results/logs/{Path(script).stem}.log"
            if run_experiment(script, log_file):
                successful.append(script)
            else:
                failed.append(script)

    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 实验完成汇总")
    print("=" * 60)
    print(f"✅ 成功: {len(successful)} 个")
    print(f"❌ 失败: {len(failed)} 个")

    if failed:
        print("\n失败的实验:")
        for f in failed:
            print(f"  • {f}")

    # 生成图表
    print("\n🖼️  生成图表...")
    subprocess.run([sys.executable, "generate_figures.py"])

    print("\n" + "=" * 60)
    print("🎉 所有任务完成！")
    print("=" * 60)
    print("📁 实验结果:")
    print(f"  • 图表: results/figures/")
    print(f"  • 日志: results/logs/")
    print(f"  • 数据: results/tables/")
    print("\n📋 后续步骤:")
    print("  1. 查看生成图表: ls results/figures/")
    print("  2. 检查实验日志: cat results/logs/*.log | head")
    print("=" * 60)


if __name__ == "__main__":
    main()