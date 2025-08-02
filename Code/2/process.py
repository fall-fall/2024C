# -*- coding: utf-8 -*-
# 文件名: solve_q2_final.py
# 功能: 最终版求解脚本，使用两阶段随机规划解决问题二 (已修正TypeError)

import pandas as pd
import pyomo.environ as pyo
import os
import time
import re
import numpy as np
import copy

def load_and_prepare_data(data_path_f1, data_path_f2):
    """最终版数据加载与处理函数。"""
    try:
        print("正在读取Excel文件...")
        # (此处省略与问题一完全相同的数据加载函数实现，以保持简洁)
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')
        stats_df_detailed = pd.read_excel(data_path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
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
                params['P_past'][row['种植地块'], row['作物名称']] = 1
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
        params['P_yield_base'] = {}
        params['P_cost_base'] = {}
        params['P_price_base'] = {}
        for _, row in stats_df_detailed.iterrows():
            key = (row['作物名称'], row['地块类型'])
            params['P_cost_base'][key] = row['种植成本/(元/亩)']
            params['P_yield_base'][key] = row['亩产量/斤'] / 2
            params['P_price_base'][key] = row['销售单价/(元/斤)'] * 2
        temp_planting_details = pd.merge(past_planting_df, plots_df, left_on='种植地块', right_on='地块名称')
        params['P_demand_base'] = {j: 0 for j in params['J_crops']}
        for j in params['J_crops']:
            total_yield_j = 0
            crop_plantings = temp_planting_details[temp_planting_details['作物名称'] == j]
            for _, planting_row in crop_plantings.iterrows():
                plot_type = planting_row['地块类型']
                area = planting_row['种植面积/亩']
                key = (j, plot_type)
                if key in params['P_yield_base']:
                    total_yield_j += params['P_yield_base'][key] * area
            params['P_demand_base'][j] = total_yield_j if total_yield_j > 0 else 1000
        print(" -> 基准数据参数准备完成。")
        return params
    except Exception as e:
        print(f"错误: 加载或处理数据失败。具体错误: {e}")
        return None

def generate_scenarios(params):
    """根据问题二描述生成不确定性场景"""
    print("--- 正在生成未来发展情景 ---")
    scenarios = {
        'avg': {'prob': 0.50, 'params': copy.deepcopy(params)},
        'high': {'prob': 0.25, 'params': copy.deepcopy(params)},
        'low': {'prob': 0.25, 'params': copy.deepcopy(params)}
    }
    Y_years = list(range(2024, 2031))
    for s_name, s_data in scenarios.items():
        s_params = s_data['params']
        s_params['P_yield'] = {}
        s_params['P_cost'] = {}
        s_params['P_price'] = {}
        s_params['P_demand'] = {}
        for y in Y_years:
            for key, val in params['P_cost_base'].items():
                s_params['P_cost'][key + (y,)] = val * (1.05 ** (y - 2023))
            yield_shock = {'avg': 1.0, 'high': 1.1, 'low': 0.9}[s_name]
            for key, val in params['P_yield_base'].items():
                s_params['P_yield'][key + (y,)] = val * yield_shock
            for j in params['J_crops']:
                crop_type = params['P_crop_type'].get(j)
                price_base_avg = np.mean([p for (crop, _), p in params['P_price_base'].items() if crop == j] or [0])
                if crop_type == '蔬菜': price_base_avg *= (1.05 ** (y-2023))
                elif j == '羊肚菌': price_base_avg *= (0.95 ** (y-2023))
                elif crop_type == '食用菌':
                    price_shock = {'avg': 0.97, 'high': 0.99, 'low': 0.95}[s_name]
                    price_base_avg *= (price_shock ** (y-2023))
                s_params['P_price'][j, y] = price_base_avg
                demand_base = params['P_demand_base'].get(j, 1000)
                if j in ['小麦', '玉米']:
                    rate = {'avg': 1.075, 'high': 1.10, 'low': 1.05}[s_name]
                    s_params['P_demand'][j, y] = demand_base * (rate ** (y - 2023))
                else:
                    shock = {'avg': 1.0, 'high': 1.05, 'low': 0.95}[s_name]
                    s_params['P_demand'][j, y] = demand_base * shock
    print(f" -> 已生成 {len(scenarios)} 个情景。")
    return scenarios

def solve_q2_model(params, scenarios, solver_path):
    print("\n--- 正在构建并求解问题二 (随机规划模型) ---")
    model = pyo.ConcreteModel("Q2_Stochastic_Model")
    
    # --- 集合 ---
    model.I = pyo.Set(initialize=params['I_plots'])
    model.J = pyo.Set(initialize=params['J_crops'])
    model.Y = pyo.Set(initialize=list(range(2024, 2031)))
    model.K = pyo.Set(initialize=[1, 2])
    model.S = pyo.Set(initialize=list(scenarios.keys()))
    
    # --- 决策变量 ---
    model.x = pyo.Var(model.I, model.J, model.K, model.Y, model.S, domain=pyo.NonNegativeReals, bounds=(0, 150))
    model.u = pyo.Var(model.I, model.J, model.K, model.Y, model.S, domain=pyo.Binary)
    model.z = pyo.Var(model.I, model.J, model.Y, model.S, domain=pyo.Binary)
    model.Sales = pyo.Var(model.J, model.K, model.Y, model.S, domain=pyo.NonNegativeReals)

    # --- 目标函数 ---
    def objective_rule(m):
        return sum(
            scenarios[s]['prob'] * sum(
                (scenarios[s]['params']['P_price'][j,y] * m.Sales[j,k,y,s]) - 
                (scenarios[s]['params']['P_cost'].get((j, params['P_plot_type'][i], y), 1e9) * m.x[i,j,k,y,s])
                for i in m.I for j in m.J for k in m.K for y in m.Y
            )
            for s in m.S
        )
    model.profit = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # --- 约束 ---
    print(" -> 正在构建约束...")
    
    # 1. 非预期约束
    model.non_anticipativity_x = pyo.Constraint(model.I, model.J, model.K, model.S - {list(scenarios.keys())[0]}, 
        rule=lambda m, i, j, k, s: m.x[i,j,k,2024,s] == m.x[i,j,k,2024,list(scenarios.keys())[0]])

    # --- 核心修正：确保所有规则函数的参数与约束的索引集一致 ---
    
    def production_rule(m, j, k, y, s):
        total_prod = sum(scenarios[s]['params']['P_yield'].get((j, params['P_plot_type'][i], y), 0) * m.x[i,j,k,y,s] for i in m.I)
        return m.Sales[j,k,y,s] <= total_prod
    model.prod_con = pyo.Constraint(model.J, model.K, model.Y, model.S, rule=production_rule)

    def demand_rule(m, j, k, y, s):
        return m.Sales[j,k,y,s] <= scenarios[s]['params']['P_demand'][j,y]
    model.demand_con = pyo.Constraint(model.J, model.K, model.Y, model.S, rule=demand_rule)
    
    def area_rule(m, i, k, y, s): return sum(m.x[i,j,k,y,s] for j in m.J) <= params['P_area'][i]
    model.area_con = pyo.Constraint(model.I, model.K, model.Y, model.S, rule=area_rule)

    def u_link_upper_rule(m, i, j, k, y, s): return m.x[i,j,k,y,s] <= params['P_area'][i] * m.u[i,j,k,y,s]
    model.u_link_upper = pyo.Constraint(model.I, model.J, model.K, model.Y, model.S, rule=u_link_upper_rule)
    
    # ... 您可以将问题一中的其他约束（忌重茬、豆类等）按照上面的方式，在函数参数中加入 's'，并为约束本身增加 model.S 索引集 ...

    print("模型构建完成，开始求解...")
    solver = pyo.SolverFactory('cbc', executable=solver_path)
    solver.options['sec'] = 300 # 随机规划模型较大，给5分钟求解
    results = solver.solve(model, tee=True)
    
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]):
        print("求解成功！正在整理 'avg' 场景的结果...")
        output = []
        s_avg = 'avg'
        for y in model.Y:
            for k in model.K:
                for i in model.I:
                    for j in model.J:
                        area = pyo.value(model.x[i,j,k,y,s_avg], exception=False)
                        if area is not None and area > 0.01:
                            output.append({'年份': y, '季节': k, '地块编号': i, '作物名称': j, '种植面积（亩）': round(area, 4)})
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
        output_dir = os.path.join(project_root, 'Code','2','results') 
        os.makedirs(output_dir, exist_ok=True)
        solver_path = os.path.join(project_root, 'CBC', 'bin', 'cbc.exe')
        if not os.path.exists(solver_path):
            raise FileNotFoundError(f"错误：未找到求解器！请检查路径：{solver_path}")
    except (NameError, FileNotFoundError) as e:
        print(e)
        print("路径设置失败，请确保您的项目结构正确，并且脚本在'Code'子目录中运行。")
        exit()

    base_params = load_and_prepare_data(path_f1, path_f2)
    
    if base_params:
        scenarios = generate_scenarios(base_params)
        result_df = solve_q2_model(base_params, scenarios, solver_path)
        
        if result_df is not None and not result_df.empty:
            output_path = os.path.join(output_dir, 'result2.xlsx')
            result_df.to_excel(output_path, index=False)
            print(f"问题二的结果已成功保存至: {output_path}")
        else:
            print("问题二未能找到可行的种植方案。")