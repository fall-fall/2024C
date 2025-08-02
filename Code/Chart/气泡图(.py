# -*- coding: utf-8 -*-
# 文件名: 生成产量图.py
# 功能: 最终版脚本，自动读取xlsx文件，使用精确匹配方法计算真实总产量，并生成图表。

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

def load_and_process_data_from_excel(data_path_f1, data_path_f2):
    """
    从真实的Excel文件中加载数据，并使用最精确的方法计算总产量。
    """
    try:
        # 1. 读取所有需要的Excel工作表
        print("正在读取Excel文件...")
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')
        stats_df_detailed = pd.read_excel(data_path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
        print(" -> Excel文件读取成功。")
        
        # --- 数据清洗 ---
        plots_df.columns = plots_df.columns.str.strip()
        crops_info_df.columns = crops_info_df.columns.str.strip()
        stats_df_detailed.columns = stats_df_detailed.columns.str.strip()
        past_planting_df.columns = past_planting_df.columns.str.strip()
        print(" -> 数据清洗完成。")

        # --- 核心计算逻辑：直接查询精确亩产 ---
        
        # 2. 准备2023年的种植记录，并附上每个地块的类型
        planting_details = pd.merge(past_planting_df, plots_df[['地块名称', '地块类型']], 
                                    left_on='种植地块', right_on='地块名称', how='left')

        # 3. 准备精确的亩产查询表 (来自 stats_df_detailed)
        # 清理可能存在的范围值
        for col in ['亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)']:
            if col in stats_df_detailed.columns:
                 def clean_and_convert(value):
                    if isinstance(value, str) and '-' in value:
                        parts = re.split(r'[-–—]', value.strip())
                        if len(parts) == 2:
                            try: return (float(parts[0]) + float(parts[1])) / 2
                            except ValueError: return pd.NA
                    return value
                 stats_df_detailed[col] = stats_df_detailed[col].apply(clean_and_convert)
                 stats_df_detailed[col] = pd.to_numeric(stats_df_detailed[col], errors='coerce')
        
        yield_lookup_table = stats_df_detailed[['作物名称', '地块类型', '亩产量/斤']].copy()
        yield_lookup_table.dropna(inplace=True)

        # 4. 将精确的亩产信息，通过“作物名称”和“地块类型”双重索引，匹配到每一条种植记录上
        final_df = pd.merge(planting_details, 
                            yield_lookup_table,
                            on=['作物名称', '地块类型'],
                            how='left')
        
        # 检查并报告数据不一致问题
        if final_df['亩产量/斤'].isnull().any():
            print("\n警告：发现数据不一致！以下种植记录没能从统计表中找到对应的亩产量，这些记录将被忽略。")
            print("请检查您的Excel文件，确保所有种植过的'作物-地块类型'组合都有对应的统计数据。")
            missing_records = final_df[final_df['亩产量/斤'].isnull()]
            print(missing_records[['种植地块', '作物名称', '地块类型']].to_string())
            final_df.dropna(subset=['亩产量/斤'], inplace=True)

        # 5. 计算每个地块的实际产量（斤），然后汇总
        final_df['地块产量(斤)'] = final_df['亩产量/斤'] * final_df['种植面积/亩']
        total_yield_df = final_df.groupby('作物名称')['地块产量(斤)'].sum().reset_index()

        # 6. 换算单位并合并作物类型信息
        total_yield_df['2023年总产量(kg)'] = total_yield_df['地块产量(斤)'] / 2
        
        result_df = pd.merge(total_yield_df, crops_info_df[['作物名称', '作物类型']], on='作物名称', how='left')
        
        return result_df.dropna(subset=['2023年总产量(kg)', '作物类型'])
        
    except FileNotFoundError as e:
        print(f"致命错误：文件未找到。请确认脚本路径和数据文件路径是否正确。")
        print(f"脚本期望找到文件: {e.filename}")
        return None
    except Exception as e:
        print(f"错误: 加载或处理数据失败。")
        print(f"具体错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_true_yield_barchart_final(df_crops, output_dir):
    """根据处理好的数据生成最终图表"""
    if df_crops is None or df_crops.empty:
        print("数据为空，无法生成图表。")
        return
        
    df_crops.sort_values(by=['作物类型', '2023年总产量(kg)'], ascending=[True, True], inplace=True)
    
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_theme(style="whitegrid", font='Microsoft YaHei')
    except Exception as e:
        print(f"设置字体'Microsoft YaHei'失败: {e}。请确保您的系统已安装该字体。")

    unique_types_sorted = sorted(df_crops['作物类型'].unique())
    palette = sns.color_palette("Set2", n_colors=len(unique_types_sorted))
    color_map = dict(zip(unique_types_sorted, palette))
    bar_colors = df_crops['作物类型'].map(color_map)
    fig, ax = plt.subplots(figsize=(12, max(8, 0.5 * len(df_crops))))
    ax.barh(df_crops['作物名称'], df_crops['2023年总产量(kg)'], color=bar_colors, height=0.7)
    max_yield = df_crops['2023年总产量(kg)'].max()
    if max_yield > 0:
        ax.set_xlim(right=max_yield * 1.2)
    for index, (label, value) in enumerate(zip(df_crops['作物名称'], df_crops['2023年总产量(kg)'])):
        if value > 0:
            offset = max_yield * 0.01 if max_yield > 0 else 1
            ax.text(value + offset, index, f'{value:,.0f} kg', va='center', ha='left', fontsize=9)
    ax.set_title('2023年各类作物真实总产量对比图 (按类型分组)', fontsize=20, pad=20)
    ax.set_xlabel('2023年总产量 (公斤)', fontsize=14)
    ax.set_ylabel('')
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color_map[ctype], label=ctype) for ctype in unique_types_sorted]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, title='作物类型')
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.grid(axis='y', linestyle='', alpha=0)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    output_filename = os.path.join(output_dir, 'final_true_yield_chart.png')
    plt.savefig(output_filename, dpi=300)
    print(f"图像已成功保存至: {output_filename}")
    plt.show()

# --- 主程序 ---
if __name__ == '__main__':
    # 假设脚本在 .../Code/Chart/ 文件夹下
    # 数据在 .../Code/Data/ 文件夹下
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        # 构建数据文件路径
        path_f1 = os.path.join(project_root, 'Data', '附件1.xlsx')
        path_f2 = os.path.join(project_root, 'Data', '附件2.xlsx')
        
        # 构建结果输出路径
        output_dir = os.path.join(current_dir, 'result')
        os.makedirs(output_dir, exist_ok=True)
        
        # 执行计算和绘图
        processed_data = load_and_process_data_from_excel(path_f1, path_f2)
        generate_true_yield_barchart_final(processed_data, output_dir)
        
    except NameError:
         # 如果在某些不支持 __file__ 的环境中运行
        print("无法自动确定文件路径，请确保数据文件在正确的相对位置。")
        # 假设数据文件在 '../Data/'
        path_f1 = os.path.join('..', 'Data', '附件1.xlsx')
        path_f2 = os.path.join('..', 'Data', '附件2.xlsx')
        output_dir = 'result'
        os.makedirs(output_dir, exist_ok=True)
        processed_data = load_and_process_data_from_excel(path_f1, path_f2)
        generate_true_yield_barchart_final(processed_data, output_dir)