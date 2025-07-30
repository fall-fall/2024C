# -*- coding: utf-8 -*-
"""
==============================================================================
2024年数学建模C题 - 问题二：鲁棒优化模型独立求解脚本 (v1.1)
==============================================================================
版本: 1.1
作者: Gemini (Modified for user's file structure)

变更日志:
- v1.1: 根据用户提供的文件结构 ，调整了项目根目录的定位逻辑。

功能:
1.  从指定的项目结构中加载并预处理数据。
2.  根据问题二的不确定性描述，计算各参数的年度名义值与偏差。
3.  构建并求解基于“不确定性预算”的鲁棒优化模型。
4.  将结果保存到 "results/result2.xlsx" 文件中。
"""

import pandas as pd
import pyomo.environ as pyo
import os
import time

# ==============================================================================
# 函数1：加载和预处理数据
# ==============================================================================
def load_and_preprocess_data(project_root_dir):
    """
    从指定路径加载并预处理所有数据。
    """
    print("--- 开始加载和预处理数据 ---")
    
    try:
        data_dir = os.path.join(project_root_dir, 'Data')
        
        path_f1 = os.path.join(data_dir, '附件1.xlsx')
        path_f2 = os.path.join(data_dir, '附件2.xlsx')
        
        plots_df = pd.read_excel(path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(path_f1, sheet_name='乡村种植的农作物')
        stats_df = pd.read_excel(path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(path_f2, sheet_name='2023年的农作物种植情况')

    except Exception as e:
        print(f"文件读取阶段发生错误，请检查文件路径和工作表名称: {e}")
        print(f"错误路径: {data_dir}")
        print("请确保您的 'Data' 文件夹及附件存在于项目根目录中。")
        return None

    print("Excel文件读取成功，开始数据预处理...")

    plots_df.columns = plots_df.columns.str.strip()
    crops_info_df.columns = crops_info_df.columns.str.strip()
    stats_df.columns = stats_df.columns.str.strip()
    past_planting_df.columns = past_planting_df.columns.str.strip()
    
    stats_df.dropna(subset=['作物名称'], inplace=True)
    stats_df = stats_df[stats_df['作物名称'].astype(str).str.strip() != '']
    
    numeric_cols = ['亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)']
    for col in numeric_cols:
        def clean_and_convert(value):
            if isinstance(value, str) and '-' in value:
                try:
                    low, high = map(float, value.split('-'))
                    return (low + high) / 2
                except: return None
            return value
        stats_df[col] = stats_df[col].apply(clean_and_convert)
        stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')

    if stats_df[numeric_cols].isnull().values.any():
        raise ValueError("数据预处理失败：源文件中包含非数字的成本、产量或价格。")
    
    stats_df['亩产量(kg/亩)'] = stats_df['亩产量/斤'] / 2
    stats_df['销售价格(元/kg)'] = stats_df['销售单价/(元/斤)'] * 2

    if '预期销售量(kg/季)' not in stats_df.columns:
        print("根据2023年产量，自动生成“预期销售量”...")
        past_production_df = pd.merge(past_planting_df, stats_df[['作物名称', '亩产量(kg/亩)']], on='作物名称', how='left')
        past_production_df.dropna(subset=['亩产量(kg/亩)'], inplace=True)
        past_production_df['2023年产量(kg)'] = past_production_df['种植面积/亩'] * past_production_df['亩产量(kg/亩)']
        demand_df = past_production_df.groupby('作物名称')['2023年产量(kg)'].sum().reset_index()
        demand_df.rename(columns={'2023年产量(kg)': '预期销售量(kg/季)'}, inplace=True)
        stats_df = pd.merge(stats_df, demand_df, on='作物名称', how='left')
        stats_df['预期销售量(kg/季)'] = stats_df['预期销售量(kg/季)'].fillna(1000)
    
    I_plots = plots_df['地块名称'].tolist()
    J_crops = stats_df['作物名称'].tolist()
    crop_data = pd.merge(stats_df, crops_info_df[['作物名称', '作物类型']], on='作物名称', how='left')
    
    params = {}
    params['P_area'] = dict(zip(plots_df['地块名称'], plots_df['地块面积/亩'])) 
    params['P_plot_type'] = dict(zip(plots_df['地块名称'], plots_df['地块类型'])) 
    params['P_crop_type'] = dict(zip(crop_data['作物名称'], crop_data['作物类型']))
    params['P_price'] = dict(zip(crop_data['作物名称'], crop_data['销售价格(元/kg)'])) 
    params['P_yield'] = dict(zip(crop_data['作物名称'], crop_data['亩产量(kg/亩)'])) 
    params['P_cost'] = dict(zip(crop_data['作物名称'], crop_data['种植成本/(元/亩)'])) 
    params['P_demand'] = dict(zip(crop_data['作物名称'], crop_data['预期销售量(kg/季)'])) 
    
    bean_names_list = ['大豆', '绿豆', '红豆', '豌豆', '蚕豆', '黄豆']
    params['J_bean'] = [j for j in J_crops if j in bean_names_list]

    P_past = {(i,j): 0 for i in I_plots for j in J_crops}
    for _, row in past_planting_df.iterrows():
        if row['种植地块'] in I_plots and row['作物名称'] in J_crops:
            P_past[row['种植地块'], row['作物名称']] = 1
    params['P_past'] = P_past

    S_suitability = {}
    for i in I_plots:
        for j in J_crops:
            for k in [1, 2]:
                plot_t = params['P_plot_type'].get(i)
                crop_t = params['P_crop_type'].get(j)
                is_bean = j in params['J_bean']
                
                suitable = 0
                if plot_t in ['平旱地', '梯田', '山坡地'] and (crop_t == '粮食' or is_bean) and k == 1: suitable = 1
                elif plot_t == '水浇地' and crop_t in ['水稻', '蔬菜']: suitable = 1
                elif plot_t == '普通大棚' and ((crop_t == '蔬菜' and k == 1) or (crop_t == '食用菌' and k == 2)): suitable = 1
                elif plot_t == '智慧大棚' and crop_t == '蔬菜': suitable = 1
                S_suitability[i,j,k] = suitable
    params['S_suitability'] = S_suitability

    print("--- 数据加载和预处理完成 ---")
    return I_plots, J_crops, params

# ==============================================================================
# 函数2：准备鲁棒优化的参数 (名义值 和 偏差)
# ==============================================================================
def prepare_robust_parameters(params, J_crops, Y_years):
    """
    根据问题二的描述，计算每年各参数的名义值和偏差。
    """
    print("--- 正在准备鲁棒优化所需参数 ---")
    params_robust = { 'nominal': {}, 'deviation': {} }
    base_year = 2023

    P_crop_type, P_cost_base = params['P_crop_type'], params['P_cost']
    P_price_base, P_yield_base, P_demand_base = params['P_price'], params['P_yield'], params['P_demand']

    for param_type in ['cost', 'yield', 'price', 'demand', 'profit_per_acre']:
        params_robust['nominal'][param_type] = {}
        params_robust['deviation'][param_type] = {}

    for j in J_crops:
        for y in Y_years:
            years_from_base = y - base_year
            crop_t = P_crop_type.get(j, '其他')
            
            cost_nominal = P_cost_base.get(j, 0) * (1.05 ** years_from_base)
            params_robust['nominal']['cost'][j, y] = cost_nominal
            
            yield_nominal = P_yield_base.get(j, 0)
            yield_dev = yield_nominal * 0.10
            params_robust['nominal']['yield'][j, y] = yield_nominal
            params_robust['deviation']['yield'][j, y] = yield_dev

            price_nominal, price_dev = 0, 0
            price_base_j = P_price_base.get(j, 0)
            if crop_t == '粮食': price_nominal, price_dev = price_base_j, 0
            elif crop_t == '蔬菜': price_nominal, price_dev = price_base_j * (1.05 ** years_from_base), 0
            elif crop_t == '食用菌':
                if j == '羊肚菌': price_nominal, price_dev = price_base_j * (0.95 ** years_from_base), 0
                else:
                    price_nominal = price_base_j * (0.97 ** years_from_base)
                    price_dev = price_nominal * 0.02
            else: price_nominal, price_dev = price_base_j, 0
            params_robust['nominal']['price'][j, y] = price_nominal
            params_robust['deviation']['price'][j, y] = price_dev
            
            demand_nominal, demand_dev = 0, 0
            demand_base_j = P_demand_base.get(j,0)
            if j in ['小麦', '玉米']:
                demand_nominal = demand_base_j * (1.075 ** years_from_base)
                demand_dev = demand_nominal * 0.025
            else:
                demand_nominal = demand_base_j
                demand_dev = demand_nominal * 0.05
            params_robust['nominal']['demand'][j, y] = demand_nominal
            params_robust['deviation']['demand'][j, y] = demand_dev
            
            profit_nominal = price_nominal * yield_nominal - cost_nominal
            profit_dev = abs(price_nominal * yield_dev) + abs(price_dev * yield_nominal)
            params_robust['nominal']['profit_per_acre'][j, y] = profit_nominal
            params_robust['deviation']['profit_per_acre'][j, y] = profit_dev

    print("--- 鲁棒参数准备完成 ---")
    return params_robust

# ==============================================================================
# 函数3：构建并求解鲁棒优化模型
# ==============================================================================
def solve_robust_model(I_plots, J_crops, params, params_robust, Gamma, solver_path):
    """
    构建并求解问题二的鲁棒优化模型。
    """
    print(f"\n--- 开始构建鲁棒模型, Gamma = {Gamma} ---")
    
    P_area, J_bean, P_past = params['P_area'], params['J_bean'], params['P_past']
    S_suitability, P_plot_type = params['S_suitability'], params['P_plot_type']
    P_demand_nominal, P_yield_nominal = params_robust['nominal']['demand'], params_robust['nominal']['yield']
    P_profit_nominal, P_profit_dev = params_robust['nominal']['profit_per_acre'], params_robust['deviation']['profit_per_acre']

    Y_years, K_seasons = list(range(2024, 2031)), [1, 2]
    A_min, N_j_val = 0.1, 15

    model = pyo.ConcreteModel("Robust_Planting_Optimization")

    model.I, model.J, model.Y, model.K = pyo.Set(initialize=I_plots), pyo.Set(initialize=J_crops), pyo.Set(initialize=Y_years), pyo.Set(initialize=K_seasons)
    model.x = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.Binary)
    model.z = pyo.Var(model.I, model.J, model.Y, domain=pyo.Binary)
    model.Z, model.lambda_ = pyo.Var(domain=pyo.Reals), pyo.Var(domain=pyo.NonNegativeReals)
    model.p = pyo.Var(model.J, model.Y, domain=pyo.NonNegativeReals)

    model.objective = pyo.Objective(expr=model.Z, sense=pyo.maximize)

    def robust_profit_rule(m):
        nominal_profit = sum(P_profit_nominal[j,y] * sum(m.x[i,j,k,y] for i in m.I for k in m.K) for j in m.J for y in m.Y)
        protection_cost = Gamma * m.lambda_ + sum(m.p[j,y] for j in m.J for y in m.Y)
        return m.Z <= nominal_profit - protection_cost
    model.robust_profit_con = pyo.Constraint(rule=robust_profit_rule)

    def protection_rule(m, j, y):
        total_area = sum(m.x[i,j,k,y] for i in m.I for k in m.K)
        return m.lambda_ + m.p[j,y] >= P_profit_dev[j,y] * total_area
    model.protection_con = pyo.Constraint(model.J, model.Y, rule=protection_rule)

    def production_rule(m, j, k, y):
        total_area = sum(m.x[i,j,k,y] for i in m.I)
        if P_demand_nominal.get((j,y), 0) > 0 :
            return P_yield_nominal.get((j,y), 0) * total_area <= P_demand_nominal.get((j,y), 0)
        return pyo.Constraint.Skip
    model.demand_con = pyo.Constraint(model.J, model.K, model.Y, rule=production_rule)
    
    def area_rule(m, i, k, y): return sum(m.x[i,j,k,y] for j in m.J) <= P_area.get(i, 0)
    model.area_con = pyo.Constraint(model.I, model.K, model.Y, rule=area_rule)
    def suitability_rule(m, i, j, k, y): return m.x[i,j,k,y] <= P_area.get(i, 0) * S_suitability.get((i,j,k), 0)
    model.suitability_con = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=suitability_rule)
    def u_link_upper_rule(m, i, j, k, y): return m.x[i,j,k,y] <= P_area.get(i, 0) * m.u[i,j,k,y]
    model.u_link_upper = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=u_link_upper_rule)
    def u_link_lower_rule(m, i, j, k, y): return m.x[i,j,k,y] >= A_min * m.u[i,j,k,y]
    model.u_link_lower = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=u_link_lower_rule)
    def dispersion_rule(m, j, k, y): return sum(m.u[i,j,k,y] for i in m.I) <= N_j_val
    model.dispersion_con = pyo.Constraint(model.J, model.K, model.Y, rule=dispersion_rule)
    def z_link_rule(m,i,j,y): return m.z[i,j,y] >= sum(m.u[i,j,k,y] for k in m.K)/len(m.K) if len(m.K)>0 else pyo.Constraint.Skip
    model.z_link_con = pyo.Constraint(model.I, model.J, model.Y, rule=z_link_rule)
    def rotation_past_rule(m, i, j):
        if P_past.get((i,j), 0) > 0: return m.z[i,j,2024] == 0
        return pyo.Constraint.Skip
    model.rotation_past_con = pyo.Constraint(model.I, model.J, rule=rotation_past_rule)
    def rotation_future_rule(m, i, j, y):
        if y < Y_years[-1]: return m.z[i,j,y] + m.z[i,j,y+1] <= 1
        return pyo.Constraint.Skip
    model.rotation_future_con = pyo.Constraint(model.I, model.J, model.Y, rule=rotation_future_rule)
    
    model.bean_con = pyo.ConstraintList()
    for i in model.I:
        past_bean_planted = 1 if sum(P_past.get((i, j), 0) for j in J_bean) > 0 else 0
        if past_bean_planted == 0:
            model.bean_con.add(sum(model.z[i, j, y] for j in J_bean for y in [2024, 2025]) >= 1)
        for y_start in range(2024, 2031 - 2):
            window = [y_start, y_start + 1, y_start + 2]
            model.bean_con.add(sum(model.z[i, j, y] for j in J_bean for y in window) >= 1)

    print("--- 鲁棒模型构建完成，开始求解 ---")
    solver = pyo.SolverFactory('cbc', executable=solver_path)
    solver.options['sec'] = 7200
    start_time = time.time()
    results = solver.solve(model, tee=True)
    end_time = time.time()
    print(f"--- 求解完成，耗时: {end_time - start_time:.2f} 秒 ---")
    
    solution_found = (results.solver.status == pyo.SolverStatus.ok) and \
                     (results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible])

    if solution_found:
        model.solutions.load_from(results)
        profit_value = pyo.value(model.Z)
        print("求解成功！")
        print(f"最坏情况下保证的总利润: {profit_value:,.2f} 元")
        output = []
        for y in model.Y:
            for k in model.K:
                for i in model.I:
                    for j in model.J:
                        area = pyo.value(model.x[i,j,k,y], exception=False)
                        if area is not None and area > 0.001:
                            output.append({'年份': y, '季节': k, '地块编号': i, '作物名称': j, '种植面积（亩）': round(area, 4)})
        return pd.DataFrame(output)
    else:
        print("求解失败，未找到任何可行的解。状态:", results.solver.termination_condition)
        return None

# ==============================================================================
# 主执行函数
# ==============================================================================
if __name__ == '__main__':
    # --- 关键：路径设置 ---
    # 获取当前脚本所在的目录 (即 .../Code/2/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # **核心调整**: 从当前脚本位置向上寻找两级，定位到项目根目录
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # 基于项目根目录构建求解器路径
    solver_executable_path = os.path.join(project_root, 'CBC', 'bin', 'cbc.exe')
    
    if not os.path.exists(solver_executable_path):
        print("="*60)
        print(f"错误：未找到求解器！请检查路径：\n{solver_executable_path}")
        print("请确保您已按照说明设置文件结构，并将CBC求解器放在项目根目录的'CBC'文件夹下。")
        print("="*60)
    else:
        # --- 执行流程 ---
        data_package = load_and_preprocess_data(project_root)
        
        if data_package:
            I_plots, J_crops, params = data_package
            Y_years = list(range(2024, 2031))

            params_robust = prepare_robust_parameters(params, J_crops, Y_years)
            
            # --- 核心参数：不确定性预算 Gamma ---
            num_uncertain_profits = len(J_crops) * len(Y_years)
            Gamma_val = round(num_uncertain_profits * 0.10) 
            
            print("="*60)
            print(f"总不确定利润项数量 (作物数 x 年份数): {num_uncertain_profits}")
            print(f"设定不确定性预算 Gamma = {Gamma_val}")
            print("您可以修改脚本中的 Gamma_val 值来测试不同的风险偏好。")
            print("="*60)
            
            result_robust = solve_robust_model(I_plots, J_crops, params, params_robust, 
                                               Gamma=Gamma_val, 
                                               solver_path=solver_executable_path)
            
            if result_robust is not None:
                # 结果保存在当前脚本所在目录下的 'results' 文件夹内
                results_dir = os.path.join(current_dir, 'results')
                os.makedirs(results_dir, exist_ok=True)
                output_path = os.path.join(results_dir, 'result2.xlsx')
                print(f"\n=== 正在保存 问题二 鲁棒优化结果至: {output_path} ===")
                result_robust.to_excel(output_path, index=False)
                print("保存成功！")
                
    print("\n所有任务已完成。")