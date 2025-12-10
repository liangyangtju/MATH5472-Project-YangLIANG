# experiments/01_need/exp_2_2_traditional_limitations.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml
import sys
# 设置NeurIPS要求的字体（Times New Roman）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  #
# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


def load_config():
    """加载配置文件 - 修复编码问题"""
    config_path = Path(__file__).parent.parent.parent / "config" / "experiment_config.yaml"
    if config_path.exists():
        try:
            # 使用UTF-8编码读取
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试其他编码
            try:
                with open(config_path, 'r', encoding='gbk') as f:
                    return yaml.safe_load(f)
            except:
                print("⚠️  Cannot read config file, using default values")
                return {}
    else:
        print(f"⚠️  Config file not found: {config_path}")
        return {}


def plot_decision_boundary(ax, X, y, model, title, cmap_background, cmap_points, alpha=0.5):
    """
    在指定坐标轴上绘制决策边界
    """
    # 设置网格范围
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 预测整个网格
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和区域
    ax.contourf(xx, yy, Z, alpha=alpha, cmap=cmap_background)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)

    # 绘制数据点
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=cmap_points, s=30)

    # 设置标题和坐标轴
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])


def create_figure_2(output_path=None):
    """
    创建图2：传统方法在处理复杂数据时的局限性

    三个子图展示不同方法在合成数据集上的决策边界：
    1. 线性模型（逻辑回归）
    2. 单决策树
    3. 随机森林（作为背景对比）
    """

    # 加载配置
    config = load_config()

    # 设置中文字体（改为英文避免编码问题）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 从配置文件获取参数
    if config and 'experiments' in config and 'need' in config['experiments']:
        exp_config = config['experiments']['need']['traditional_limitations']
        n_samples = exp_config.get('n_samples', 300)
        noise_level = exp_config.get('noise_level', 0.25)
        random_state = exp_config.get('random_state', 42)
        test_size = exp_config.get('test_size', 0.3)
    else:
        n_samples = 300
        noise_level = 0.25
        random_state = 42
        test_size = 0.3

    # 创建合成数据集：月牙形数据（非线性可分）
    np.random.seed(random_state)
    X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=random_state)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 创建图形和子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 从配置文件获取颜色，或使用默认值
    if config and 'visualization' in config and 'colors' in config['visualization']:
        color_config = config['visualization']['colors']['decision_boundary']
        cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
        cmap_points = ListedColormap([color_config['class0'], color_config['class1']])
    else:
        cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
        cmap_points = ListedColormap(['#FF4444', '#4444FF'])

    # ========== 子图1：线性模型（逻辑回归） ==========
    ax1 = axes[0]

    # 创建并训练逻辑回归模型
    model_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=random_state, max_iter=1000))
    ])
    model_lr.fit(X_train, y_train)

    # 计算准确率
    score_lr = model_lr.score(X_test, y_test)

    # 绘制决策边界
    plot_decision_boundary(ax1, X, y, model_lr,
                           f'(a) Logistic Regression\nAccuracy: {score_lr:.3f}',
                           cmap_background, cmap_points)

    # ========== 子图2：单决策树 ==========
    ax2 = axes[1]

    # 创建并训练决策树模型
    model_dt = DecisionTreeClassifier(max_depth=3, random_state=random_state)
    model_dt.fit(X_train, y_train)

    # 计算准确率
    score_dt = model_dt.score(X_test, y_test)

    # 绘制决策边界
    plot_decision_boundary(ax2, X, y, model_dt,
                           f'(b) Single Decision Tree (max_depth=3)\nAccuracy: {score_dt:.3f}',
                           cmap_background, cmap_points)

    # ========== 子图3：随机森林 ==========
    ax3 = axes[2]

    # 创建并训练随机森林模型
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    model_rf.fit(X_train, y_train)

    # 计算准确率
    score_rf = model_rf.score(X_test, y_test)

    # 绘制决策边界
    plot_decision_boundary(ax3, X, y, model_rf,
                           f'(c) Random Forest (100 trees)\nAccuracy: {score_rf:.3f}',
                           cmap_background, cmap_points)

    # ========== 图形整体美化 ==========
    # plt.suptitle('Figure 2: Decision Boundaries of Traditional Methods on Non-Linear Data',
    #              fontsize=14, fontweight='bold', y=1.05)

    # 添加整体图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#FF4444', markersize=10, label='Class 0'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#4444FF', markersize=10, label='Class 1'),
    ]

    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0), ncol=2, fontsize=11)

    plt.tight_layout()

    # 保存或显示图形
    if output_path:
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure 2 saved to: {output_path}")

        # 同时保存为PDF格式
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure 2 (PDF) saved to: {pdf_path}")
    else:
        plt.show()

    plt.close()

    # 返回模型性能数据
    performance_data = {
        'Logistic Regression': {
            'accuracy': score_lr,
            'description': 'Linear decision boundary, cannot fit non-linear relationships'
        },
        'Decision Tree': {
            'accuracy': score_dt,
            'description': 'Step-like decision boundary, prone to overfitting'
        },
        'Random Forest': {
            'accuracy': score_rf,
            'description': 'Smooth decision boundary, but has performance plateau'
        }
    }

    return performance_data


def generate_figure_2(output_path=None):
    """生成图2的包装函数"""
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / "results" / "figures" / "need" / "figure2.png"

    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    return create_figure_2(output_path)


if __name__ == "__main__":
    # 测试运行
    print("=" * 60)
    print("Generating Figure 2: Limitations of Traditional Methods on Complex Data")
    print("=" * 60)

    # 指定输出路径
    output_path = Path(__file__).parent.parent.parent / "results" / "figures" / "need" / "figure2.png"

    # 生成图形
    try:
        data = generate_figure_2(output_path)

        # 打印性能摘要
        print("\n📊 Figure 2 Performance Summary:")
        print("-" * 40)
        for model_name, info in data.items():
            print(f"{model_name}:")
            print(f"  Accuracy: {info['accuracy']:.3f}")
            print(f"  Characteristics: {info['description']}")
            print()

        print(f"✅ Figure 2 saved to: {output_path}")
        print(f"📄 PDF version saved to: {output_path.with_suffix('.pdf')}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error generating Figure 2: {e}")
        import traceback

        traceback.print_exc()