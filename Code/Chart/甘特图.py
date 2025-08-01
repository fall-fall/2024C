# -*- coding: utf-8 -*-
# 文件名: 甘特图.py
# 功能: 可视化作物轮作计划

import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_gantt_chart():
    # --- 1. 路径设置 ---
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        # 甘特图的数据来源于优化结果，这里我们以 result2.xlsx 为例
        # !!请确保您的结果文件在此路径下!!
        result_path = os.path.join(project_root, 'Data', 'result2.xlsx') 
        data_path_f1 = os.path.join(project_root, 'Data', '附件1.xlsx')
        output_dir = os.path.join(current_dir, 'result')
        os.makedirs(output_dir, exist_ok=True)
    except NameError:
        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        result_path = os.path.join(project_root, 'Data', 'result2.xlsx')
        data_path_f1 = os.path.join(project_root, 'Data', '附件1.xlsx')
        output_dir = 'result'
        os.makedirs(output_dir, exist_ok=True)
        
    # --- 2. 读取并处理真实数据 ---
    # !!重要提示!!: 此处需要您提供一个优化结果文件(如result2.xlsx)。
    # 为保证代码能运行，我们先创建一个示例结果DataFrame。
    # 在您的实际使用中，请注释掉下面的示例数据，并取消注释 pd.read_excel(...) 部分。
    
    # --- 示例数据 ---
    print("注意：正在使用内置的示例数据。请取消注释文件读取部分以使用您的真实结果。")
    plan_data = {
        '年份': [2024, 2025, 2026, 2027, 2024, 2024, 2025, 2026, 2027, 2027],
        '地块编号': ['平旱地1', '平旱地1', '平旱地1', '平旱地1', '普通大棚1', '普通大棚1', '普通大棚1', '普通大棚1', '普通大棚1', '普通大棚1'],
        '季节': [1, 1, 1, 1, 1, 2, 1, 2, 1, 2],
        '作物名称': ['小麦', '大豆', '玉米', '小麦', '番茄', '香菇', '大白菜', '羊肚菌', '番茄', '香菇']
    }
    df_plan = pd.DataFrame(plan_data)
    # --- 真实数据读取 (请取消注释) ---
    # try:
    #     print(f"正在从 '{result_path}' 读取优化结果...")
    #     df_plan = pd.read_excel(result_path)
    # except FileNotFoundError:
    #     print(f"错误: 结果文件 '{result_path}' 未找到。将使用示例数据。")
    # except Exception as e:
    #     print(f"读取结果文件时出错: {e}")
    #     return
        
    try:
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')[['作物名称', '作物类型']]
        df_plan = pd.merge(df_plan, crops_info_df, on='作物名称', how='left')
    except Exception as e:
        print(f"读取作物类型失败: {e}")
        return
        
    # 为使图表清晰，只选几个地块展示
    plots_to_show = df_plan['地块编号'].unique()[:4] # 最多显示4个
    df_plan = df_plan[df_plan['地块编号'].isin(plots_to_show)]

    # --- 3. 可视化 ---
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 'SimHei' 字体未找到。")
    
    crop_types = sorted(df_plan['作物类型'].dropna().unique())
    colors = plt.cm.get_cmap('viridis', len(crop_types))
    color_map = {crop_type: colors(i) for i, crop_type in enumerate(crop_types)}
    
    fig, ax = plt.subplots(figsize=(15, 2 * len(plots_to_show)))
    
    y_labels = list(plots_to_show)
    y_pos = range(len(y_labels))
    
    for i, plot_name in enumerate(y_labels):
        plot_data = df_plan[df_plan['地块编号'] == plot_name].sort_values(by=['年份', '季节'])
        for _, row in plot_data.iterrows():
            start_time = row['年份'] + (row['季节'] - 1) * 0.5
            ax.barh(y=i, left=start_time, width=0.5, height=0.6,
                    color=color_map.get(row['作物类型'], 'grey'), edgecolor='black')
            ax.text(start_time + 0.25, i, row['作物名称'], ha='center', va='center', color='white', fontweight='bold')
            
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlabel('年份与季节', fontsize=14)
    ax.set_ylabel('地块编号', fontsize=14)
    ax.set_title('代表性地块作物轮作计划甘特图', fontsize=18, pad=15)
    
    years = sorted(df_plan['年份'].unique())
    ax.set_xticks(range(min(years), max(years) + 2))
    ax.set_xticks([y + 0.5 for y in range(min(years), max(years) + 1)], minor=True)
    ax.grid(axis='x', linestyle='--')
    
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color_map[ct], label=ct) for ct in crop_types]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', title='作物类型')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # --- 4. 保存 ---
    output_filename = os.path.join(output_dir, '4_轮作计划_甘特图.png')
    plt.savefig(output_filename, dpi=300)
    print(f"图像已成功保存至: {output_filename}")
    plt.show()

if __name__ == '__main__':
    generate_gantt_chart()