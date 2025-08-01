# -*- coding: utf-8 -*-
# 文件名: solve_q1_final.py (v1.1 - 修正TypeError)

import pandas as pd
import pyomo.environ as pyo
import os
import time
import re
import numpy as np
import copy

def load_and_prepare_data(data_path_f1, data_path_f2):
    """
    最终版数据加载与处理函数。
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
        
        # --- 核心修正：增加类型检查，防止TypeError ---
        params['J_bean'] = [
            j for j, ctype in params['P_crop_type'].items() 
            if isinstance(ctype, str) and any(keyword in ctype for keyword in bean_keywords)
        ]

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
            params['P_yield'][key] = row['亩产量/斤'] / 2
            params['P_price'][key] = row['销售单价/(元/斤)'] * 2
            
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
            params['P_demand'][j] = total_yield_j if total_yield_j > 0 else 1000

        # 6. 种植适宜性矩阵
        params['S_suitability'] = {}
        for i in params['I_plots']:
            plot_t = params['P_plot_type'][i]
            for j in params['J_crops']:
                crop_t = params['P_crop_type'].get(j)
                is_bean = j in params['J_bean']
                for k in [1, 2]:
                    suitable = 0
                    if isinstance(crop_t, str): # 确保crop_t是字符串
                        if plot_t in ['平旱地', '梯田', '山坡地'] and ('粮食' in crop_t or is_bean) and k == 1: suitable = 1
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

def solve_q1_model(params, case_type='waste'):
    print(f"\n--- 正在构建并求解问题一 (情况: {case_type}) ---")
    
    model = pyo.ConcreteModel(f"Q1_Model_{case_type}")
    
    # --- 集合与参数 ---
    model.I = pyo.Set(initialize=params['I_plots'])
    model.J = pyo.Set(initialize=params['J_crops'])
    model.Y = pyo.Set(initialize=list(range(2024, 2031)))
    model.K = pyo.Set(initialize=[1, 2])
    
    # --- 决策变量 ---
    model.x = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.NonNegativeReals, bounds=(0, 150))
    model.u = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.Binary)
    model.z = pyo.Var(model.I, model.J, model.Y, domain=pyo.Binary)
    model.Sales = pyo.Var(model.J, model.K, model.Y, domain=pyo.NonNegativeReals)
    if case_type == 'discount':
        model.OverSales = pyo.Var(model.J, model.K, model.Y, domain=pyo.NonNegativeReals)

    # --- 目标函数 ---
    def objective_rule(m):
        avg_price = {j: np.mean([p for (crop, _), p in params['P_price'].items() if crop == j] or [0]) for j in m.J}
        revenue = sum(avg_price[j] * m.Sales[j,k,y] for j in m.J for k in m.K for y in m.Y)
        if case_type == 'discount':
            revenue += sum(0.5 * avg_price[j] * m.OverSales[j,k,y] for j in m.J for k in m.K for y in m.Y)
        
        total_cost = sum(params['P_cost'].get((j, params['P_plot_type'][i]), 1e9) * m.x[i,j,k,y] 
                         for i in m.I for j in m.J for k in m.K for y in m.Y)
        return revenue - total_cost
    model.profit = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # --- 约束 ---
    print(" -> 正在构建约束...")
    def production_rule(m, j, k, y):
        total_prod = sum(params['P_yield'].get((j, params['P_plot_type'][i]), 0) * m.x[i,j,k,y] for i in m.I)
        if case_type == 'waste': return m.Sales[j,k,y] <= total_prod
        else: return m.Sales[j,k,y] + m.OverSales[j,k,y] == total_prod
    model.prod_con = pyo.Constraint(model.J, model.K, model.Y, rule=production_rule)
    
    def demand_rule(m, j, k, y): return m.Sales[j,k,y] <= params['P_demand'].get(j, 0)
    model.demand_con = pyo.Constraint(model.J, model.K, model.Y, rule=demand_rule)
    
    def area_rule(m, i, k, y): return sum(m.x[i,j,k,y] for j in m.J) <= params['P_area'][i]
    model.area_con = pyo.Constraint(model.I, model.K, model.Y, rule=area_rule)

    def suitability_rule(m, i, j, k, y):
        if params['S_suitability'].get((i,j,k), 0) == 0: return m.x[i,j,k,y] == 0
        return pyo.Constraint.Skip
    model.suitability_con = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=suitability_rule)

    A_min = 0.1
    def u_link_upper_rule(m, i, j, k, y): return m.x[i,j,k,y] <= params['P_area'][i] * m.u[i,j,k,y]
    model.u_link_upper = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=u_link_upper_rule)
    def u_link_lower_rule(m, i, j, k, y): return m.x[i,j,k,y] >= A_min * m.u[i,j,k,y]
    model.u_link_lower = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=u_link_lower_rule)

    def z_link_rule(m, i, j, y): return m.z[i,j,y] >= sum(m.u[i,j,k,y] for k in m.K)
    model.z_link_con = pyo.Constraint(model.I, model.J, model.Y, rule=z_link_rule)
    def rotation_past_rule(m, i, j):
        if params['P_past'].get((i,j), 0) > 0: return m.z[i,j,2024] <= 0
        return pyo.Constraint.Skip
    model.rotation_past_con = pyo.Constraint(model.I, model.J, rule=rotation_past_rule)
    def rotation_future_rule(m, i, j, y):
        if y < 2030: return m.z[i,j,y] + m.z[i,j,y+1] <= 1
        return pyo.Constraint.Skip
    model.rotation_future_con = pyo.Constraint(model.I, model.J, model.Y, rule=rotation_future_rule)

    model.bean_con = pyo.ConstraintList()
    for i in model.I:
        past_bean = sum(params['P_past'].get((i,j), 0) for j in params['J_bean'])
        model.bean_con.add(past_bean + sum(model.z[i,j,y] for j in params['J_bean'] for y in [2024, 2025]) >= 1)
        for y_start in range(2024, 2031 - 2):
            model.bean_con.add(sum(model.z[i,j,y] for j in params['J_bean'] for y in range(y_start, y_start+3)) >= 1)

    # --- 求解 ---
    print("模型构建完成，开始求解...")
    solver = pyo.SolverFactory('cbc')
    solver.options['sec'] = 600
    results = solver.solve(model, tee=True)

    # --- 结果处理 ---
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]):
        print("求解成功！正在整理结果...")
        output = []
        for v in model.x.values():
            if pyo.value(v, exception=False) is not None and pyo.value(v, exception=False) > 0.01:
                i, j, k, y = v.index()
                output.append({'年份': y, '季节': k, '地块编号': i, '作物名称': j, '种植面积（亩）': round(pyo.value(v), 4)})
        return pd.DataFrame(output)
    else:
        print(f"求解失败: {results.solver.termination_condition}")
        return None

# --- 主程序 ---
if __name__ == '__main__':
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        path_f1 = os.path.join(project_root, 'Data', '附件1.xlsx')
        path_f2 = os.path.join(project_root, 'Data', '附件2.xlsx')
        output_dir = os.path.join(project_root, 'Code', 'Chart', 'result')
        os.makedirs(output_dir, exist_ok=True)
    except NameError:
        project_root = os.getcwd()
        path_f1 = os.path.join(project_root, 'Data', '附件1.xlsx')
        path_f2 = os.path.join(project_root, 'Data', '附件2.xlsx')
        output_dir = os.path.join(project_root, 'Code', 'Chart', 'result')

    params = load_and_prepare_data(path_f1, path_f2)
    
    if params:
        result1_1 = solve_q1_model(copy.deepcopy(params), case_type='waste')
        if result1_1 is not None and not result1_1.empty:
            output_path1 = os.path.join(output_dir, 'result1_1.xlsx')
            result1_1.to_excel(output_path1, index=False)
            print(f"情况一的结果已成功保存至: {output_path1}")
        else:
            print("情况一未能找到可行的种植方案。")

        result1_2 = solve_q1_model(copy.deepcopy(params), case_type='discount')
        if result1_2 is not None and not result1_2.empty:
            output_path2 = os.path.join(output_dir, 'result1_2.xlsx')
            result1_2.to_excel(output_path2, index=False)
            print(f"情况二的结果已成功保存至: {output_path2}")
        else:
            print("情况二未能找到可行的种植方案。")