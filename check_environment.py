#!/usr/bin/env python3
"""
环境验证脚本：检查所有必要包是否已安装
"""

import importlib
import sys

# 必需包列表（包名: 测试导入的名称）
REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'scikit-learn': 'sklearn',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost',
    'torch': 'torch',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'plotly': 'plotly',
    'networkx': 'networkx',
    'graphviz': 'graphviz',
    'imbalanced-learn': 'imblearn',
    'deap': 'deap',
}

# 可选包列表
OPTIONAL_PACKAGES = {
    'mlflow': 'mlflow',
    'optuna': 'optuna',
    'wandb': 'wandb',
    'jupyter': 'jupyter',
    'pyyaml': 'yaml',
}

def check_package(package_name, import_name=None):
    """检查单个包是否可用"""
    try:
        if import_name is None:
            import_name = package_name
        importlib.import_module(import_name)
        return True, f"✓ {package_name}"
    except ImportError:
        return False, f"✗ {package_name}"

def main():
    print("=" * 60)
    print("GBDT Counterfactual Analysis - 环境检查")
    print("=" * 60)
    
    print("\nPython版本:", sys.version[:6])
    print("Python路径:", sys.executable)
    
    print("\n必需包检查:")
    print("-" * 40)
    
    required_failed = []
    for pkg, import_name in REQUIRED_PACKAGES.items():
        success, message = check_package(pkg, import_name)
        print(message)
        if not success:
            required_failed.append(pkg)
    
    print("\n可选包检查:")
    print("-" * 40)
    for pkg, import_name in OPTIONAL_PACKAGES.items():
        success, message = check_package(pkg, import_name)
        print(message)
    
    print("\n" + "=" * 60)
    if not required_failed:
        print("✅ 所有必需包都已安装！")
        print("环境准备就绪，可以开始项目。")
    else:
        print("❌ 以下必需包未安装:")
        for pkg in required_failed:
            print(f"  - {pkg}")
        print("\n请运行: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()