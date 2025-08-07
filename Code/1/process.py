# -*- coding: utf-8 -*-
# 文件名: solve_q1_final.py
# 描述: 官方题意最终版 (v11.0) - 完全基于官方PDF文件对第一问的要求进行建模。

import pandas as pd
import pyomo.environ as pyo
import os
import time
import re
import copy
import warnings
from pathlib import Path

def load_and_prepare_data(data_path):
    """数据加载与处理函数。"""
    try:
        print("正在读取官方Excel文件...")
        path_f1 = data_path / '附件1.xlsx'
        path_f2 = data_path / '附件2.xlsx'
        if not path_f1.exists() or not path_f2.exists():
            print(f"错误: 文件不存在。请检查路径 {path_f1} 和 {path_f2}"); return None
        plots_df = pd.read_excel(path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(path_f1, sheet_name='乡村种植的农作物')
        stats_df_detailed = pd.read_excel(path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(path_f2, sheet_name='2023年的农作物种植情况')
        print(" -> Excel文件读取成功。")
    except Exception as e:
        print(f"错误: 读取Excel文件失败。具体错误: {e}"); return None

    for df in [plots_df, crops_info_df, stats_df_detailed, past_planting_df]:
        df.columns = df.columns.str.strip()
    params = {}
    params['I_plots'] = plots_df['地块名称'].tolist()
    params['P_area'] = dict(zip(plots_df['地块名称'], plots_df['地块面积/亩']))
    params['P_plot_type'] = dict(zip(plots_df['地块名称'], plots_df['地块类型']))
    params['J_crops'] = crops_info_df['作物名称'].unique().tolist()
    params['P_crop_type'] = dict(zip(crops_info_df['作物名称'], crops_info_df['作物类型']))
    bean_keywords = ['豆', '豆类']
    params['J_bean'] = [j for j, ctype in params['P_crop_type'].items() if isinstance(ctype, str) and any(keyword in ctype for keyword in bean_keywords)]
    params['P_past'] = {(i, j): 0 for i in params['I_plots'] for j in params['J_crops']}
    for _, row in past_planting_df.iterrows():
        if row['种植地块'] in params['I_plots'] and row['作物名称'] in params['J_crops']:
            params['P_past'][(row['种植地块'], row['作物名称'])] = 1
    def clean_and_convert_price(value):
        if isinstance(value, str) and ('-' in value or '–' in value or '—' in value):
            parts = re.split(r'[-–—]', value.strip())
            try:
                if len(parts) == 2: return (float(parts[0]) + float(parts[1])) / 2
            except ValueError: return pd.NA
        try: return float(value)
        except (ValueError, TypeError): return pd.NA
    for col in ['亩产量/斤', '种植成本/(元/亩)']:
        stats_df_detailed[col] = pd.to_numeric(stats_df_detailed[col], errors='coerce')
    stats_df_detailed['销售单价/(元/斤)'] = stats_df_detailed['销售单价/(元/斤)'].apply(clean_and_convert_price)
    stats_df_detailed.dropna(subset=['亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)'], inplace=True)
    params['P_yield'], params['P_cost'], params['P_price'] = {}, {}, {}
    for _, row in stats_df_detailed.iterrows():
        key = (row['作物名称'], row['地块类型'])
        params['P_cost'][key] = row['种植成本/(元/亩)']
        params['P_yield'][key] = row['亩产量/斤'] / 2
        params['P_price'][key] = row['销售单价/(元/斤)'] * 2
    for crop in params['J_crops']:
        for plot_type in plots_df['地块类型'].unique():
            key = (crop, plot_type)
            if key not in params['P_yield']:
                params['P_yield'][key], params['P_cost'][key], params['P_price'][key] = 0, 9e9, 0 # 未知参数设为极高成本，避免被选中
    params['P_demand'] = {j: 0 for j in params['J_crops']}
    temp_planting_details = pd.merge(past_planting_df, plots_df, left_on='种植地块', right_on='地块名称')
    for j in params['J_crops']:
        total_yield_j = 0
        crop_plantings = temp_planting_details[temp_planting_details['作物名称'] == j]
        for _, planting_row in crop_plantings.iterrows():
            plot_type, area = planting_row['地块类型'], planting_row['种植面积/亩']
            key = (j, plot_type)
            if key in params['P_yield']: total_yield_j += params['P_yield'][key] * area
        params['P_demand'][j] = total_yield_j if total_yield_j > 0 else 1000
    params['S_suitability'] = {}
    restricted_veg = ['大白菜', '白萝卜', '红萝卜']
    for i in params['I_plots']:
        plot_t = params['P_plot_type'].get(i)
        if not plot_t: continue
        for j in params['J_crops']:
            crop_t_val = params['P_crop_type'].get(j, '')
            crop_t_str, is_bean, is_veg = str(crop_t_val), j in params['J_bean'], '蔬菜' in str(crop_t_val)
            for k in [1, 2]:
                suitable = 0
                if plot_t in ['平旱地', '梯田', '山坡地']:
                    if ('粮食' in crop_t_str or is_bean) and k == 1: suitable = 1
                elif plot_t == '水浇地':
                    if '水稻' in crop_t_str and k == 1: suitable = 1
                    elif is_veg:
                        if j not in restricted_veg and k == 1: suitable = 1
                        elif j in restricted_veg and k == 2: suitable = 1
                elif plot_t == '普通大棚':
                    if is_veg and j not in restricted_veg and k == 1: suitable = 1
                    elif '食用菌' in crop_t_str and k == 2: suitable = 1
                elif plot_t == '智慧大棚':
                    if is_veg and j not in restricted_veg: suitable = 1
                params['S_suitability'][(i, j, k)] = suitable
    params['N_dispersion'] = {j: 10 for j in params['J_crops']}
    print(" -> 数据参数准备完成。")
    return params

def build_model(params, case_type):
    """构建优化模型 (v11.0 最终版)"""
    model = pyo.ConcreteModel(f"Q1_Model_{case_type}")
    model.I, model.J, model.Y, model.K = (pyo.Set(initialize=params[s]) for s in ['I_plots', 'J_crops', 'Y_years', 'K_seasons'])
    model.x = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.Binary)
    model.z = pyo.Var(model.I, model.J, model.Y, domain=pyo.Binary)

    if case_type == 'waste':
        model.profit = pyo.Objective(rule=lambda m: sum((params['P_price'].get((j, params['P_plot_type'][i]), 0) * params['P_yield'].get((j, params['P_plot_type'][i]), 0) - params['P_cost'].get((j, params['P_plot_type'][i]), 0)) * m.x[i, j, k, y] for i,j,k,y in m.I*m.J*m.K*m.Y), sense=pyo.maximize)
        model.demand_con = pyo.Constraint(model.J, model.K, model.Y, rule=lambda m, j, k, y: sum(params['P_yield'].get((j, params['P_plot_type'][i]), 0) * m.x[i, j, k, y] for i in m.I) <= params['P_demand'].get(j, 0))
    else: # case_type == 'discount'
        model.x_normal = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.NonNegativeReals)
        model.x_over = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.NonNegativeReals)
        def objective_rule_discount(m):
            profit_normal = sum((params['P_price'].get((j, params['P_plot_type'][i]), 0) * params['P_yield'].get((j, params['P_plot_type'][i]), 0) - params['P_cost'].get((j, params['P_plot_type'][i]), 0)) * m.x_normal[i,j,k,y] for i,j,k,y in m.I*m.J*m.K*m.Y)
            profit_over = sum((0.5 * params['P_price'].get((j, params['P_plot_type'][i]), 0) * params['P_yield'].get((j, params['P_plot_type'][i]), 0) - params['P_cost'].get((j, params['P_plot_type'][i]), 0)) * m.x_over[i,j,k,y] for i,j,k,y in m.I*m.J*m.K*m.Y)
            return profit_normal + profit_over
        model.profit = pyo.Objective(rule=objective_rule_discount, sense=pyo.maximize)
        model.x_sum_con = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=lambda m,i,j,k,y: m.x[i,j,k,y] == m.x_normal[i,j,k,y] + m.x_over[i,j,k,y])
        model.demand_con = pyo.Constraint(model.J, model.K, model.Y, rule=lambda m,j,k,y: sum(params['P_yield'].get((j, params['P_plot_type'][i]), 0) * m.x_normal[i,j,k,y] for i in m.I) <= params['P_demand'].get(j,0))
        # 根据题意，不设降价销售的上限
    
    model.area_con = pyo.Constraint(model.I, model.K, model.Y, rule=lambda m,i,k,y: sum(m.x[i,j,k,y] for j in m.J) <= params['P_area'][i])
    model.suitability_con = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=lambda m,i,j,k,y: m.x[i,j,k,y] <= params['S_suitability'][(i,j,k)] * params['P_area'][i])
    model.u_link_upper = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=lambda m,i,j,k,y: m.x[i,j,k,y] <= params['P_area'][i] * m.u[i,j,k,y])
    model.u_link_lower = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=lambda m,i,j,k,y: m.x[i,j,k,y] >= 0.1 * m.u[i,j,k,y])
    model.z_link_con = pyo.Constraint(model.I, model.J, model.Y, rule=lambda m,i,j,y: m.z[i,j,y] * 2 >= sum(m.u[i,j,k,y] for k in m.K))
    model.rotation_past_con = pyo.Constraint(model.I, model.J, rule=lambda m,i,j: pyo.Constraint.Skip if params['P_past'].get((i,j),0) == 0 else m.z[i,j,2024] == 0)
    # 【最终修正】将忌重茬约束应用于所有地块
    model.rotation_future_con = pyo.Constraint(model.I, model.J, model.Y, rule=lambda m,i,j,y: pyo.Constraint.Skip if y >= 2030 else m.z[i,j,y] + m.z[i,j,y+1] <= 1)
    model.bean_con = pyo.ConstraintList()
    for i in model.I:
        if params['P_plot_type'].get(i) in ['平旱地', '梯田', '山坡地', '水浇地']:
            past_bean = sum(params['P_past'].get((i,j),0) for j in params['J_bean'])
            model.bean_con.add(past_bean + sum(model.z[i,j,y] for j in params['J_bean'] for y in [2024, 2025]) >= 1)
            for y_start in range(2024, 2029): model.bean_con.add(sum(model.z[i,j,y] for j in params['J_bean'] for y in range(y_start, y_start+3)) >= 1)
    print(f" -> 模型(情况: {case_type})构建完成。")
    return model

def build_and_solve_once(params, case_type, solver_path, timeout=600):
    """构建并求解优化模型"""
    print(f"\n--- 正在构建并求解问题一 (情况: {case_type}) ---")
    model = build_model(params, case_type)
    solver = pyo.SolverFactory('cbc', executable=str(solver_path))
    solver.options['sec'] = timeout
    start_time = time.time()
    results = solver.solve(model, tee=True)
    end_time = time.time()
    print(f"\n求解器运行完毕。")
    print(f" -> 耗时: {end_time - start_time:.2f} 秒")
    print(f" -> 求解器状态: {results.solver.status}")
    print(f" -> 终止条件: {results.solver.termination_condition}")
    if pyo.value(model.profit, exception=False) is not None:
        print("已在模型中找到一个可行的解，正在保存结果...")
        final_profit = pyo.value(model.profit)
        print(f"最终目标值 (7年总利润): {final_profit:,.2f} 元")
        output = []
        for v in model.x.values():
            val = pyo.value(v, exception=False)
            if val is not None and val > 0.01:
                i, j, k, y = v.index()
                output.append({'年份': y, '季节': k, '地块编号': i, '作物名称': j, '种植面积（亩）': round(val, 4)})
        return pd.DataFrame(output), final_profit
    else:
        print("\n求解失败：在规定时间内未能找到任何可行的种植方案。")
        return None, 0

# --- 主程序 ---
if __name__ == '__main__':
    try:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent 
        data_path = project_root / 'Data'
        output_dir = current_dir / 'results'; output_dir.mkdir(parents=True, exist_ok=True)
        solver_path = project_root / 'CBC' / 'bin' / 'cbc.exe'
        if not solver_path.exists():
            print(f"错误：未找到求解器！请检查路径：{solver_path}"); exit()
        params = load_and_prepare_data(data_path)
        if params is None: raise RuntimeError("数据加载失败")
        params['Y_years'] = list(range(2024, 2031))
        params['K_seasons'] = [1, 2]
        for case in ['waste', 'discount']:
            result_df, profit = build_and_solve_once(copy.deepcopy(params), case, solver_path)
            if result_df is not None and not result_df.empty:
                result_df.to_excel(output_dir / f'result1_{case}.xlsx', index=False)
                print(f"情况({case})的结果已成功保存。")
            else: print(f"情况({case})未能找到可行的种植方案。")
        print("\n所有求解任务完成！")
    except Exception as e:
        print(f"\n程序主流程运行出错: {e}"); import traceback; traceback.print_exc()