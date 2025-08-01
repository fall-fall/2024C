# -*- coding: utf-8 -*-
# 文件名: 热力图.py
# 功能: 生成2023年种植结构热力图

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_heatmap():
    # --- 1. 路径设置 ---
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        data_path_f1 = os.path.join(project_root, 'Data', '附件1.xlsx')
        data_path_f2 = os.path.join(project_root, 'Data', '附件2.xlsx')
        output_dir = os.path.join(current_dir, 'result')
        os.makedirs(output_dir, exist_ok=True)
    except NameError:
        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        data_path_f1 = os.path.join(project_root, 'Data', '附件1.xlsx')
        data_path_f2 = os.path.join(project_root, 'Data', '附件2.xlsx')
        output_dir = 'result'
        os.makedirs(output_dir, exist_ok=True)
        
    # --- 2. 读取并处理真实数据 ---
    try:
        print("正在读取2023年种植数据...")
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')[['地块名称', '地块类型']]
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')[['作物名称', '作物类型']]
        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
        
        # 合并以获取每个种植记录的地块类型和作物类型
        df_2023 = pd.merge(past_planting_df, plots_df, left_on='种植地块', right_on='地块名称')
        df_2023 = pd.merge(df_2023, crops_info_df, on='作物名称')
        
        # 使用pivot_table创建矩阵
        heatmap_data = df_2023.pivot_table(
            index='地块类型', columns='作物类型', values='种植面积/亩',
            aggfunc='sum', fill_value=0
        )
        print("数据处理完成。")
    except Exception as e:
        print(f"错误: 读取或处理数据失败。具体错误: {e}")
        return

    # --- 3. 可视化 ---
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 'SimHei' 字体未找到。")
        
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".1f", linewidths=.5, cmap='YlGnBu'
    )
    plt.title('2023年各地块类型-作物类型种植面积热力图 (单位: 亩)', fontsize=18, pad=20)
    plt.xlabel('作物类型', fontsize=12)
    plt.ylabel('地块类型', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # --- 4. 保存图像 ---
    output_filename = os.path.join(output_dir, '3_2023种植结构_热力图.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图像已成功保存至: {output_filename}")
    plt.show()

if __name__ == '__main__':
    generate_heatmap()