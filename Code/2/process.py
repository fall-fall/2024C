# -*- coding: utf-8 -*-
# 文件名: solve_q2_official_final.py
# 功能: 最终版，补全所有缺失的关键约束，解决不可行问题
# 版本: 7.4 (约束补全最终版)

import pandas as pd
import pyomo.environ as pyo
import os
import time
import re
import copy
import numpy as np
from pathlib import Path

def load_and_prepare_data(data_path_f1, data_path_f2):
    """数据加载与处理函数 (采用Q1的精细化逻辑)"""
    try:
        print("正在读取Excel文件...")
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')
        stats_df_detailed = pd.read_excel(data_path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
        for df in [plots_df, crops_info_df, stats_df_detailed, past_planting_df]:
            df.columns = df.columns.str.strip()
        params = {}
        params['I_plots'], params['P_area'], params['P_plot_type'] = plots_df['地块名称'].tolist(), dict(zip(plots_df['地块名称'], plots_df['地块面积/亩'])), dict(zip(plots_df['地块名称'], plots_df['地块类型']))
        params['J_crops'], params['P_crop_type'] = crops_info_df['作物名称'].unique().tolist(), dict(zip(crops_info_df['作物名称'], crops_info_df['作物类型']))
        params['J_bean'] = [j for j, ctype in params['P_crop_type'].items() if isinstance(ctype, str) and '豆' in ctype]
        params['P_past'] = {(i, j): 0 for i in params['I_plots'] for j in params['J_crops']}
        for _, row in past_planting_df.iterrows():
            if row['种植地块'] in params['I_plots'] and row['作物名称'] in params['J_crops']:
                params['P_past'][(row['种植地块'], row['作物名称'])] = 1
        def clean_and_convert_price(value):
            if isinstance(value, str) and any(c in value for c in '-–—'):
                parts = re.split(r'[-–—]', value.strip())
                if len(parts) == 2:
                    try: return (float(parts[0]) + float(parts[1])) / 2
                    except ValueError: return pd.NA
            return pd.to_numeric(value, errors='coerce')
        for col in ['亩产量/斤', '种植成本/(元/亩)']: stats_df_detailed[col] = pd.to_numeric(stats_df_detailed[col], errors='coerce')
        stats_df_detailed['销售单价/(元/斤)'] = stats_df_detailed['销售单价/(元/斤)'].apply(clean_and_convert_price)
        stats_df_detailed.dropna(subset=['亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)'], inplace=True)
        params['P_yield_base'], params['P_cost_base'], params['P_price_base'] = {}, {}, {}
        for _, row in stats_df_detailed.iterrows():
            key = (row['作物名称'], row['地块类型'])
            params['P_cost_base'][key], params['P_yield_base'][key], params['P_price_base'][key] = row['种植成本/(元/亩)'], row['亩产量/斤'] / 2, row['销售单价/(元/斤)'] * 2
        params['P_demand_base'] = {j: 0 for j in params['J_crops']}
        temp_planting_details = pd.merge(past_planting_df, plots_df, left_on='种植地块', right_on='地块名称')
        for j in params['J_crops']:
            total_yield_j = sum(params['P_yield_base'].get((j, row['地块类型']), 0) * row['种植面积/亩'] for _, row in temp_planting_details[temp_planting_details['作物名称'] == j].iterrows())
            params['P_demand_base'][j] = total_yield_j if total_yield_j > 0 else 1000
        params['S_suitability'] = {}
        restricted_veg = ['大白菜', '白萝卜', '红萝卜']
        for i in params['I_plots']:
            plot_t = params['P_plot_type'].get(i, '')
            for j in params['J_crops']:
                crop_t_val, is_bean, is_veg = params['P_crop_type'].get(j, ''), j in params['J_bean'], '蔬菜' in str(params['P_crop_type'].get(j, ''))
                for k in [1, 2]:
                    suitable = 0
                    if plot_t in ['平旱地', '梯田', '山坡地']:
                        if ('粮食' in str(crop_t_val) or is_bean) and k == 1: suitable = 1
                    elif plot_t == '水浇地':
                        if '水稻' in str(crop_t_val) and k == 1: suitable = 1
                        elif is_veg:
                            if j not in restricted_veg and k == 1: suitable = 1
                            elif j in restricted_veg and k == 2: suitable = 1
                    elif plot_t == '普通大棚':
                        if is_veg and j not in restricted_veg and k == 1: suitable = 1
                        elif '食用菌' in str(crop_t_val) and k == 2: suitable = 1
                    elif plot_t == '智慧大棚':
                        if is_veg and j not in restricted_veg: suitable = 1
                    params['S_suitability'][(i, j, k)] = suitable
        print(" -> 数据参数准备完成。")
        return params
    except Exception as e:
        print(f"错误: 加载或处理数据失败。具体错误: {e}")
        return None

def generate_scenarios(params):
    """生成随机情景"""
    print("--- 正在生成未来发展情景 ---")
    scenarios = {'avg': {'prob': 0.50, 'params': copy.deepcopy(params)}, 'high': {'prob': 0.25, 'params': copy.deepcopy(params)}, 'low': {'prob': 0.25, 'params': copy.deepcopy(params)}}
    Y_years = list(range(2024, 2031))
    for s_name, s_data in scenarios.items():
        s_params = s_data['params']
        s_params['P_yield'], s_params['P_cost'], s_params['P_price'], s_params['P_demand'] = {}, {}, {}, {}
        for y in Y_years:
            for key, val in params['P_cost_base'].items(): s_params['P_cost'][key + (y,)] = val * (1.05 ** (y - 2023))
            yield_shock = {'avg': 1.0, 'high': 1.1, 'low': 0.9}[s_name]
            for key, val in params['P_yield_base'].items(): s_params['P_yield'][key + (y,)] = val * yield_shock
            for j in params['J_crops']:
                crop_type, price_base_avg = params['P_crop_type'].get(j), np.mean([p for (c, _), p in params['P_price_base'].items() if c == j] or [0])
                if crop_type == '蔬菜': price_base_avg *= (1.05 ** (y - 2023))
                elif j == '羊肚菌': price_base_avg *= (0.95 ** (y - 2023))
                elif crop_type == '食用菌': price_base_avg *= ({'avg': 0.97, 'high': 0.99, 'low': 0.95}[s_name] ** (y-2023))
                s_params['P_price'][j, y] = price_base_avg
                demand_base = params['P_demand_base'].get(j, 1000)
                if j in ['小麦', '玉米']: s_params['P_demand'][j, y] = demand_base * ({'avg': 1.075, 'high': 1.10, 'low': 1.05}[s_name] ** (y - 2023))
                else: s_params['P_demand'][j, y] = demand_base * {'avg': 1.0, 'high': 1.05, 'low': 0.95}[s_name]
    print(f" -> 已生成 {len(scenarios)} 个情景。")
    return scenarios

def solve_q2_model(params, scenarios, solver_path):
    """【最终版】以官方“策略二”为核心，补全所有约束的Q2模型求解函数"""
    print("\n--- 正在构建并求解问题二 (官方策略最终版) ---")
    model = pyo.ConcreteModel("Q2_Official_Strategy_Final_Model")
    
    model.I, model.J, model.J_bean = pyo.Set(initialize=params['I_plots']), pyo.Set(initialize=params['J_crops']), pyo.Set(initialize=params['J_bean'])
    model.Y, model.K, model.S = pyo.Set(initialize=list(range(2024, 2031))), pyo.Set(initialize=[1, 2]), pyo.Set(initialize=list(scenarios.keys()))
    model.x = pyo.Var(model.I, model.J, model.K, model.Y, model.S, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.I, model.J, model.K, model.Y, model.S, domain=pyo.Binary)
    model.z = pyo.Var(model.I, model.J, model.Y, model.S, domain=pyo.Binary)
    model.Sales_normal = pyo.Var(model.J, model.K, model.Y, model.S, domain=pyo.NonNegativeReals)
    model.Sales_over = pyo.Var(model.J, model.K, model.Y, model.S, domain=pyo.NonNegativeReals)

    def objective_rule(m):
        revenue_normal = pyo.quicksum(scenarios[s]['prob'] * scenarios[s]['params']['P_price'].get((j, y), 0) * m.Sales_normal[j, k, y, s] for s in m.S for j in m.J for k in m.K for y in m.Y)
        revenue_over = pyo.quicksum(scenarios[s]['prob'] * 0.5 * scenarios[s]['params']['P_price'].get((j, y), 0) * m.Sales_over[j, k, y, s] for s in m.S for j in m.J for k in m.K for y in m.Y)
        total_cost = pyo.quicksum(scenarios[s]['prob'] * scenarios[s]['params']['P_cost'].get((j, params['P_plot_type'][i], y), 0) * m.x[i, j, k, y, s] for s in m.S for i in m.I for j in m.J for k in m.K for y in m.Y)
        return revenue_normal + revenue_over - total_cost
    model.profit = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    print(" -> 正在构建约束...")
    # --- 约束 (补全所有) ---
    first_s = list(scenarios.keys())[0]
    @model.Constraint(model.I, model.J, model.K, model.Y, model.S - {first_s})
    def non_anticipativity_x(m, i, j, k, y, s):
        if y == 2024: return m.x[i,j,k,y,s] == m.x[i,j,k,y,first_s]
        return pyo.Constraint.Skip
    @model.Constraint(model.I, model.J, model.K, model.Y, model.S - {first_s})
    def non_anticipativity_u(m, i, j, k, y, s):
        if y == 2024: return m.u[i,j,k,y,s] == m.u[i,j,k,y,first_s]
        return pyo.Constraint.Skip
    @model.Constraint(model.I, model.J, model.Y, model.S - {first_s})
    def non_anticipativity_z(m, i, j, y, s):
        if y == 2024: return m.z[i,j,y,s] == m.z[i,j,y,first_s]
        return pyo.Constraint.Skip
    @model.Constraint(model.J, model.K, model.Y, model.S)
    def production_rule(m, j, k, y, s):
        total_production = pyo.quicksum(scenarios[s]['params']['P_yield'].get((j, params['P_plot_type'][i], y), 0) * m.x[i,j,k,y,s] for i in m.I)
        return m.Sales_normal[j,k,y,s] + m.Sales_over[j,k,y,s] <= total_production
    @model.Constraint(model.J, model.Y, model.S)
    def demand_normal_rule(m, j, y, s):
        return m.Sales_normal[j,1,y,s] + m.Sales_normal[j,2,y,s] <= scenarios[s]['params']['P_demand'].get((j,y), 0)
    @model.Constraint(model.J, model.Y, model.S)
    def demand_over_rule(m, j, y, s):
        return m.Sales_over[j,1,y,s] + m.Sales_over[j,2,y,s] <= scenarios[s]['params']['P_demand'].get((j,y), 0)
    @model.Constraint(model.I, model.K, model.Y, model.S)
    def area_rule(m, i, k, y, s):
        return sum(m.x[i,j,k,y,s] for j in m.J) <= params['P_area'][i]
    @model.Constraint(model.I, model.K, model.Y, model.S)
    def one_crop_per_plot_season_rule(m, i, k, y, s):
        return sum(m.u[i,j,k,y,s] for j in m.J) <= 1
    @model.Constraint(model.I, model.J, model.K, model.Y, model.S)
    def suitability_con(m, i, j, k, y, s):
        return m.u[i,j,k,y,s] <= params['S_suitability'].get((i,j,k), 0)
    @model.Constraint(model.I, model.J, model.K, model.Y, model.S)
    def u_link_upper(m, i, j, k, y, s):
        return m.x[i,j,k,y,s] <= params['P_area'][i] * m.u[i,j,k,y,s]
    @model.Constraint(model.I, model.J, model.K, model.Y, model.S)
    def u_link_lower(m, i, j, k, y, s):
        return m.x[i,j,k,y,s] >= 0.1 * m.u[i,j,k,y,s]
    @model.Constraint(model.I, model.J, model.K, model.Y, model.S)
    def z_u_link_lower(m, i, j, k, y, s):
        return m.z[i,j,y,s] >= m.u[i,j,k,y,s]
    @model.Constraint(model.I, model.J, model.Y, model.S)
    def z_u_link_upper(m, i, j, y, s):
        return m.z[i,j,y,s] <= sum(m.u[i,j,k,y,s] for k in m.K)
    @model.Constraint(model.I, model.J, model.S)
    def rotation_past_con(m, i, j, s):
        if params['P_past'].get((i,j), 0) == 1: return m.z[i,j,2024,s] == 0
        return pyo.Constraint.Skip
    @model.Constraint(model.I, model.J, model.Y, model.S)
    def rotation_future_con(m, i, j, y, s):
        if y >= 2030: return pyo.Constraint.Skip
        if params['P_plot_type'].get(i) not in ['普通大棚', '智慧大棚']:
            return m.z[i, j, y, s] + m.z[i, j, y + 1, s] <= 1
        return pyo.Constraint.Skip
    model.bean_con = pyo.ConstraintList()
    for s in model.S:
        for i in model.I:
            if params['P_plot_type'].get(i) not in ['普通大棚', '智慧大棚']:
                past_bean = sum(1 for j in params['J_bean'] if params['P_past'].get((i,j),0) == 1)
                model.bean_con.add(past_bean + sum(model.z[i,j,y,s] for j in model.J_bean for y in [2024,2025]) >= 1)
                for y_start in range(2024, 2029):
                    model.bean_con.add(sum(model.z[i,j,y,s] for j in model.J_bean for y in range(y_start, y_start+3)) >= 1)
    
    print("模型构建完成，开始求解...")
    solver = pyo.SolverFactory('cbc', executable=solver_path)
    solver.options['sec'] = 600
    results = solver.solve(model, tee=True, load_solutions=False)
    print("\n求解时间到，提取当前解...")
    try:
        if not (results.solution and len(results.solution.l) > 0): return None
        model.solutions.load_from(results)
        final_profit = pyo.value(model.profit, exception=False)
        if final_profit is None: return None
        print(f"模型提取到的预期总利润为: {final_profit:,.2f} 元")
        output = []
        for v in model.x.component_data_objects(active=True):
            if v.value is not None and v.value > 0.01 and v.index()[4] == 'avg':
                i, j, k, y, s = v.index()
                output.append({'年份': y, '季节': k, '地块编号': i, '作物名称': j, '种植面积（亩）': round(v.value, 4)})
        if not output: return None
        return pd.DataFrame(output)
    except Exception as e:
        print(f"\n在提取结果时发生未知错误: {e}")
        return None

# --- 主程序 ---
if __name__ == '__main__':
    try:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        data_path_f1 = project_root / 'Data' / '附件1.xlsx'
        data_path_f2 = project_root / 'Data' / '附件2.xlsx'
        output_dir = current_dir / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        solver_path = str(project_root / 'CBC' / 'bin' / 'cbc.exe')
        if not os.path.exists(solver_path): solver_path = 'cbc'
    except Exception as e:
        print(f"路径设置错误: {e}"); exit()

    base_params = load_and_prepare_data(data_path_f1, data_path_f2)
    if base_params:
        scenarios = generate_scenarios(base_params)
        result_df = solve_q2_model(base_params, scenarios, solver_path)
        if result_df is not None and not result_df.empty:
            output_path = output_dir / 'result2_official_strategy.xlsx'
            result_df.to_excel(output_path, index=False)
            print(f"\n问题二的结果已成功保存至: {output_path}")
        else:
            print("\n问题二未能找到可行的种植方案。请检查模型约束是否过于严格或存在冲突。")