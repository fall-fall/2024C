# -*- coding: utf-8 -*-
# 文件名: 饼图.py (v1.3 - 修正API兼容性问题)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_pie_chart():
    """主函数：加载数据、处理并生成图表"""
    
    # --- 1. 路径设置 ---
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        data_path = os.path.join(project_root, 'Data', '附件1.xlsx')
        output_dir = os.path.join(current_dir, 'result')
        os.makedirs(output_dir, exist_ok=True)
    except NameError:
        project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
        data_path = os.path.join(project_root, 'Data', '附件1.xlsx')
        output_dir = 'result'
        os.makedirs(output_dir, exist_ok=True)

    # --- 2. 读取并处理真实数据 ---
    try:
        print(f"正在从 '{data_path}' 读取地块数据...")
        df_plots_raw = pd.read_excel(data_path, sheet_name='乡村的现有耕地')
        df_plots_raw.columns = df_plots_raw.columns.str.strip()
        df_plots = df_plots_raw.groupby('地块类型')['地块面积/亩'].sum().reset_index()
        df_plots.rename(columns={'地块面积/亩': '总面积'}, inplace=True)
        print("数据处理完成。")
    except Exception as e:
        print(f"错误: 读取或处理数据失败。请检查 '附件1.xlsx'。具体错误: {e}")
        return

    # --- 3. 可视化 ---
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    df_plots = df_plots.sort_values(by='总面积', ascending=False)
    labels = df_plots['地块类型']
    sizes = df_plots['总面积']
    
    explode = [0.05] * len(labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # --- 核心修正：使用新版matplotlib API获取颜色 ---
    cmap = plt.get_cmap('Pastel2')
    colors = cmap(np.linspace(0, 1, len(labels)))
    
    def autopct_generator(limit):
        def inner_autopct(pct):
            return f'{pct:.1f}%' if pct > limit else ''
        return inner_autopct

    wedges, texts, autotexts = ax.pie(sizes, explode=explode,
                                      autopct=autopct_generator(1),
                                      shadow=False, startangle=90, colors=colors,
                                      pctdistance=0.75)

    plt.setp(autotexts, size=10, color="black", fontweight="bold")
    
    ax.axis('equal')
    ax.set_title('地块类型面积分布图', fontsize=20, pad=20)
    
    legend_labels = [f'{l} ({s:.1f} 亩)' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels,
              title="地块类型",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    
    # --- 4. 保存图像 ---
    output_filename = os.path.join(output_dir, '1_地块类型分布_饼图.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图像已成功保存至: {output_filename}")
    plt.show()

if __name__ == '__main__':
    generate_pie_chart()