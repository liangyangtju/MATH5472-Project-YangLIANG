import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 设置新罗马字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'


def plot_gbdt_three_layer_summary_optimized():
    """优化布局的GBDT三层架构图表"""
    # 创建图形，设置适当的大小和DPI
    fig, ax = plt.subplots(figsize=(13, 7), dpi=100)

    # 设置坐标轴范围（基于黄金比例1.618:1）
    ax.set_xlim(0, 13)
    ax.set_ylim(1.7, 9.5)
    ax.axis('off')

    # 主标题
    # plt.title('Fig. 4: Three-Layer Architecture of GBDT for Tabular Data',
    #           fontsize=14, fontweight='bold', pad=25, color='navy')

    # ==================== 顶层：表格数据特性 ====================
    # 位置：从y=8.3开始，高度0.65
    ax.add_patch(patches.Rectangle((1.5, 8.3), 10, 1.0,
                                   facecolor='#F5F5F5',
                                   edgecolor='gray', linewidth=1.5,
                                   zorder=1))
    plt.text(6.5, 9.03, "Characteristics of Tabular Data",
             fontsize=12, ha='center', va='center', fontweight='bold', zorder=2)

    # 表格数据特性图标
    features = ["Mixed Features", "Missing Values", "Feature Interactions", "Large Scale"]
    colors = ['#66B2FF', '#99FF99', '#FF9966', '#FFCC99']

    for i, (feat, col) in enumerate(zip(features, colors)):
        x_pos = 2.8 + i * 2.2  # 调整间距为2.2
        # 图标
        ax.add_patch(patches.Circle((x_pos, 8.7), 0.12,
                                    facecolor=col, edgecolor='black', linewidth=1, zorder=2))
        # 标签
        plt.text(x_pos, 8.48, feat, fontsize=8.5, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                           edgecolor=col, alpha=0.7, linewidth=0.8), zorder=2)

    # ==================== 第一层：决策树基学习器 ====================
    # 位置：从y=6.6开始，高度1.6（之前为1.8）
    layer1_box = patches.Rectangle((1.5, 6.55), 10, 1.5,
                                   facecolor='#E6F3FF',
                                   edgecolor='#0066CC', linewidth=2, zorder=1)
    ax.add_patch(layer1_box)
    plt.text(6.5, 7.75, 'Layer 1: Decision Tree Base Learners',
             fontsize=12, ha='center', va='center', fontweight='bold',
             color='#0066CC', zorder=2)

    # 特征处理机制（左侧，x从2.0开始）
    tree_diagram_x = 3.2

    # 决策节点（更大一些）
    ax.add_patch(patches.Rectangle((tree_diagram_x-0.425, 7.5), 0.85, 0.45,
                                facecolor='lightblue', edgecolor='black',
                                linewidth=1, zorder=2))

    # 分裂分支
    ax.plot([tree_diagram_x, tree_diagram_x - 0.725], [7.5, 7.0], 'k-', lw=1.5, zorder=2)
    ax.plot([tree_diagram_x, tree_diagram_x + 0.675], [7.5, 7.0], 'k-', lw=1.5, zorder=2)

    # 叶子节点
    ax.add_patch(patches.Rectangle((tree_diagram_x - 1.3, 6.75), 0.9, 0.45,
                                   facecolor='#99FF99', edgecolor='darkgreen',
                                   linewidth=1, zorder=2))
    ax.add_patch(patches.Rectangle((tree_diagram_x + 0.5, 6.75), 0.9, 0.45,
                                   facecolor='#99FF99', edgecolor='darkgreen',
                                   linewidth=1, zorder=2))

    # 特征处理标注
    feature_types = [
        ("Numerical:\nThreshold Split", tree_diagram_x - 0.85, 6.95, '#66B2FF'),
        ("Categorical:\nEncoding/Gini", tree_diagram_x, 7.7, '#99FF99'),
        ("Missing:\nSeparate Branch", tree_diagram_x + 0.95, 6.95, '#FF9966')
    ]

    for text, x, y, color in feature_types:
        plt.text(x, y, text, fontsize=8, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor=color, alpha=0.8, linewidth=1.2), zorder=2)

    # 特征交互说明（右侧）
    interaction_text = "Implicit Feature Interaction:\nSplit on Income → then Age\n= Income × Age effect"
    plt.text(9.8, 7.4, interaction_text, fontsize=8.5, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='#FFF0F5',
                       edgecolor='purple', alpha=0.8, linewidth=1.2), zorder=2)

    # ==================== 第二层：梯度提升框架 ====================
    # 位置：从y=4.3开始，高度1.6
    layer2_box = patches.Rectangle((1.5, 4.85), 10, 1.45,
                                   facecolor='#E6FFE6',
                                   edgecolor='#009933', linewidth=2, zorder=1)
    ax.add_patch(layer2_box)
    plt.text(6.5, 6.0, 'Layer 2: Gradient Boosting Framework',
             fontsize=12, ha='center', va='center', fontweight='bold',
             color='#009933', zorder=2)

    # 迭代过程示意
    iterations = 3
    tree_width = 0.9  # 树的宽度
    tree_spacing = 2.5  # 树之间的间距

    for i in range(iterations):
        x_start = 2.8 + i * tree_spacing

        # 树图标（更像树的形状）
        ax.add_patch(patches.Rectangle((x_start, 5.0), tree_width, 0.5,
                                       facecolor='#FFE5CC', edgecolor='orange',
                                       linewidth=1.5, zorder=2))

        # 树的"树冠"
        ax.add_patch(patches.Polygon(
            [(x_start + 0.1, 5.5), (x_start + 0.45, 5.8), (x_start + 0.8, 5.5)],
            facecolor='#99CC66', edgecolor='darkgreen', linewidth=1, zorder=2
        ))

        plt.text(x_start + tree_width / 2, 5.25, f"Tree {i + 1}",
                 fontsize=9, ha='center', va='center', fontweight='bold', zorder=3)

        # 残差箭头（只在树之间画）
        if i < iterations - 1:
            arrow_start = x_start + tree_width + 0.1
            arrow_end = x_start + tree_spacing - 0.1

            ax.annotate('', xy=(arrow_end, 5.25), xytext=(arrow_start, 5.25),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                        connectionstyle="arc3,rad=0"), zorder=2)

            # 残差标签
            plt.text((arrow_start + arrow_end) / 2, 5.45, "Fit\nResiduals",
                     fontsize=8, ha='center', va='center',
                     color='darkred', fontweight='bold', zorder=2)

    # 数学表示（放在右侧）
    math_eq = r"$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$"
    plt.text(10.1, 5.6, math_eq, fontsize=10.5, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='white',
                       edgecolor='black', linewidth=1), zorder=2)

    # ==================== 第三层：工程优化结构 ====================
    # 位置：从y=2.0开始，高度1.6
    layer3_box = patches.Rectangle((1.5, 2.9), 10, 1.7,
                                   facecolor='#FFF0E6',
                                   edgecolor='#FF6600', linewidth=2, zorder=1)
    ax.add_patch(layer3_box)
    plt.text(6.5, 4.4, 'Layer 3: Engineering Optimization Structure',
             fontsize=12, ha='center', va='center', fontweight='bold',
             color='#FF6600', zorder=2)

    # 时间线式展示
    implementations = [
        ("XGBoost\n(2016)", 2.8, 3.6, "Regularization\nWeighted Quantile"),
        ("LightGBM\n(2017)", 5.3, 3.6, "Histogram\nGOSS & EFB"),
        ("CatBoost\n(2018)", 7.8, 3.6, "Ordered Boosting\nCategorical Features"),
        ("Scaling", 10.3, 3.6, "Millions of\nSamples")
    ]

    for name, x, y, desc in implementations:
        # 实现图标框
        box_width = 1.6 if name != "Scaling" else 1.4
        box_height = 0.9

        ax.add_patch(patches.FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2), box_width, box_height,
            boxstyle="round,pad=0.1", facecolor='white',
            edgecolor='#666666', linewidth=1.5, zorder=2
        ))

        # 实现名称
        plt.text(x, y + 0.15, name, fontsize=9.5, ha='center', va='center',
                 fontweight='bold', zorder=3)

        # 实现描述
        plt.text(x, y - 0.2, desc, fontsize=8, ha='center', va='center',
                 linespacing=1.2, zorder=3)

    # ==================== 连接箭头 ====================
    # 从表格数据到第一层
    ax.annotate('', xy=(6.5, 8.4), xytext=(6.5, 8.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue',
                                connectionstyle="arc3,rad=0"), zorder=2)

    # 从第一层到第二层
    ax.annotate('', xy=(6.5, 6.6), xytext=(6.5, 6.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='green',
                                connectionstyle="arc3,rad=0"), zorder=2)

    # 从第二层到第三层
    ax.annotate('', xy=(6.5, 4.9), xytext=(6.5, 4.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='orange',
                                connectionstyle="arc3,rad=0"), zorder=2)

    # 箭头标签
    arrow_labels = [
        ("Handles", 6.8, 8.22),
        ("Combines via", 6.95, 6.45),
        ("Optimized by", 6.9, 4.75)
    ]

    for text, x, y in arrow_labels:
        plt.text(x, y, text, fontsize=9, ha='center', va='center',
                 fontweight='bold', style='italic', color='darkred', zorder=2)

    # ==================== 底部总结 ====================
    summary_box = patches.FancyBboxPatch((4.25, 1.9), 4.5, 0.7,
                                         boxstyle="round,pad=0.1",
                                         facecolor='#F8F8FF',
                                         edgecolor='navy', linewidth=1.5, zorder=1)
    ax.add_patch(summary_box)

    summary_text = "Synergistic Architecture → Preferred for Tabular Data"
    plt.text(6.5, 2.25, summary_text, fontsize=11, ha='center', va='center',
             fontweight='bold', color='darkred', zorder=2)

    # ==================== 引用编号 ====================
    # plt.text(0.8, 9.3, "Fig. 4", fontsize=10, ha='left', va='top',
    #          fontweight='bold', style='italic', zorder=3)

    # ==================== 网格线（仅用于调试布局） ====================
    # 取消下面的注释可以显示网格线帮助调试
    # for y in np.arange(0, 10, 0.5):
    #     ax.axhline(y, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    # for x in np.arange(0, 13, 0.5):
    #     ax.axvline(x, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig('D:/course/math 5472 bayes/R file/final project/GBDT-Counterfactual-Analysis/results/figures/approach/figure3.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('D:/course/math 5472 bayes/R file/final project/GBDT-Counterfactual-Analysis/results/figures/approach/figure3.png', dpi=300, bbox_inches='tight')
    # plt.show()


# 生成优化后的图表
plot_gbdt_three_layer_summary_optimized()