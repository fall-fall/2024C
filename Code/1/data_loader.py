# common_data_loader.py (概念上)

import pandas as pd
import os
import re
import numpy as np

def load_and_prepare_data(data_path_f1, data_path_f2):
    """
    最终版数据加载与处理函数。
    读取所有Excel表，使用精确匹配方法处理数据，为优化模型准备参数字典。
    """
    try:
        print("正在读取Excel文件...")
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')
        stats_df_detailed = pd.read_excel(data_path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
        print(" -> Excel文件读取成功。")

        for df in [plots_df, crops_info_df, stats_df_detailed, past_planting_df]:
            df.columns = df.columns.str.strip()
        
        params = {}
        
        # 1. 地块参数
        params['I_plots'] = plots_df['地块名称'].tolist()
        params['P_area'] = dict(zip(plots_df['地块名称'], plots_df['地块面积/亩']))
        params['P_plot_type'] = dict(zip(plots_df['地块名称'], plots_df['地块类型']))
        
        # 2. 作物参数
        params['J_crops'] = crops_info_df['作物名称'].unique().tolist()
        params['P_crop_type'] = dict(zip(crops_info_df['作物名称'], crops_info_df['作物类型']))
        bean_keywords = ['豆', '豆类']
        params['J_bean'] = [j for j, ctype in params['P_crop_type'].items() if any(keyword in ctype for keyword in bean_keywords)]

        # 3. 2023年种植历史
        params['P_past'] = {(i, j): 0 for i in params['I_plots'] for j in params['J_crops']}
        for _, row in past_planting_df.iterrows():
            if row['种植地块'] in params['I_plots'] and row['作物名称'] in params['J_crops']:
                params['P_past'][row['种植地块'], row['作物名称']] = 1
                
        # 4. 经济与产量参数 (分地块类型)
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
        stats_df_detailed.dropna(inplace=True)

        params['P_yield'] = {}
        params['P_cost'] = {}
        params['P_price'] = {}
        
        for _, row in stats_df_detailed.iterrows():
            key = (row['作物名称'], row['地块类型'])
            params['P_cost'][key] = row['种植成本/(元/亩)']
            params['P_yield'][key] = row['亩产量/斤'] / 2  # 统一换算为公斤
            params['P_price'][key] = row['销售单价/(元/斤)'] * 2 # 统一换算为公斤
            
        # 5. 估算预期销售量 (基于2023年真实总产量)
        params['P_demand'] = {j: 0 for j in params['J_crops']}
        temp_planting_details = pd.merge(past_planting_df, plots_df, left_on='种植地块', right_on='地块名称')
        
        for j in params['J_crops']:
            total_yield_j = 0
            crop_plantings = temp_planting_details[temp_planting_details['作物名称'] == j]
            for _, planting_row in crop_plantings.iterrows():
                plot_type = planting_row['地块类型']
                area = planting_row['种植面积/亩']
                key = (j, plot_type)
                if key in params['P_yield']:
                    total_yield_j += params['P_yield'][key] * area
            params['P_demand'][j] = total_yield_j if total_yield_j > 0 else 1000 # 避免为0

        # 6. 种植适宜性矩阵
        params['S_suitability'] = {}
        for i in params['I_plots']:
            plot_t = params['P_plot_type'][i]
            for j in params['J_crops']:
                crop_t = params['P_crop_type'][j]
                is_bean = j in params['J_bean']
                for k in [1, 2]:
                    suitable = 0
                    if plot_t in ['平旱地', '梯田', '山坡地'] and (crop_t == '粮食' or is_bean or '豆类' in crop_t) and k == 1: suitable = 1
                    elif plot_t == '水浇地':
                        if (crop_t == '水稻' and k==1): suitable = 1
                        if (crop_t == '蔬菜'): suitable = 1
                    elif plot_t == '普通大棚' and ((crop_t == '蔬菜' and k == 1) or (crop_t == '食用菌' and k == 2)): suitable = 1
                    elif plot_t == '智慧大棚' and crop_t == '蔬菜': suitable = 1
                    params['S_suitability'][i, j, k] = suitable

        print(" -> 数据参数准备完成。")
        return params
        
    except Exception as e:
        print(f"错误: 加载或处理数据失败。具体错误: {e}")
        import traceback
        traceback.print_exc()
        return None