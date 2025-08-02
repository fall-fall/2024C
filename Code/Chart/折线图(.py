# -*- coding: utf-8 -*-
# 文件名: plot_ga_log.py
# 功能：读取 ga_log.csv 并绘制遗传算法收敛曲线（中文零乱码版）
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 1. 指定字体文件（微软雅黑，Windows 自带）
my_font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=14)

def plot_convergence(csv_path='ga_log.csv'):
    """读取 ga_log.csv 并绘制收敛曲线"""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'找不到 {csv_path}，请先运行训练脚本生成该文件！')

    # 2. 读数据
    df = pd.read_csv(csv_path, names=['代数', '最优适应度', '平均适应度'])

    # 3. 输出目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    os.makedirs(output_dir, exist_ok=True)

    # 4. 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df['代数'], df['最优适应度'], marker='o', markersize=4,
            linestyle='-', color='#1f77b4', label='最优适应度')
    ax.plot(df['代数'], df['平均适应度'], marker='x', markersize=4,
            linestyle='--', color='#d62728', label='平均适应度')
    ax.fill_between(df['代数'], df['平均适应度'], df['最优适应度'],
                    color='skyblue', alpha=0.3)

    # 5. 中文标题/轴标签（使用硬编码字体）
    ax.set_title('遗传算法收敛曲线（问题三）', fontproperties=my_font, fontsize=18, pad=15)
    ax.set_xlabel('迭代代数', fontproperties=my_font, fontsize=14)
    ax.set_ylabel('适应度（预期平均总利润 / 元）', fontproperties=my_font, fontsize=14)
    ax.legend(prop=my_font, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 6. 保存
    save_path = os.path.join(output_dir, '6_GA收敛过程_折线图.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'图像已保存至: {save_path}')
    plt.show()

if __name__ == '__main__':
    plot_convergence()