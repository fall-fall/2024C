# 文件名: 小提琴图.py, 功能同上, 路径已修正
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_violin_plot():
    # 路径设置
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'result')
        os.makedirs(output_dir, exist_ok=True)
    except NameError:
        output_dir = 'result'
        os.makedirs(output_dir, exist_ok=True)

    # !!重要提示!!: 此处需要您提供蒙特卡洛模拟后的利润数据。
    # 为保证代码能运行，我们生成示例数据。
    print("注意：正在使用内置的示例数据。")
    np.random.seed(42)
    q1_profits = np.random.normal(loc=100, scale=30, size=300)
    q2_profits = np.random.normal(loc=90, scale=10, size=300)
    df_profits = pd.DataFrame({
        '利润': np.concatenate([q1_profits, q2_profits]),
        '方案类型': ['Q1: 确定性解 (随机环境评估)'] * 300 + ['Q2: 风险规避解'] * 300
    })

    # 可视化
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 'SimHei' 字体未找到。")
        
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    sns.violinplot(
        data=df_profits, x='方案类型', y='利润',
        palette='Pastel1', inner='quartile', linewidth=2
    )
    plt.title('不同方案在随机场景下的利润分布对比', fontsize=18, pad=15)
    plt.xlabel('方案类型', fontsize=14)
    plt.ylabel('模拟总利润 (万元)', fontsize=14)

    # 保存
    output_filename = os.path.join(output_dir, '5_方案风险对比_小提琴图.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图像已成功保存至: {output_filename}")
    plt.show()

if __name__ == '__main__':
    generate_violin_plot()