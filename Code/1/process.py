import pandas as pd
import pyomo.environ as pyo
import os
import time

def load_and_preprocess_data():
    """
    从指定路径加载并预处理所有数据。
    """
    print("--- 开始加载和预处理数据 ---")
    
    try:
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, '..', '..', 'Data')
        
        path_f1 = os.path.join(data_dir, '附件1.xlsx')
        path_f2 = os.path.join(data_dir, '附件2.xlsx')
        
        plots_df = pd.read_excel(path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(path_f1, sheet_name='乡村种植的农作物')
        stats_df = pd.read_excel(path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(path_f2, sheet_name='2023年的农作物种植情况')

    except Exception as e:
        print(f"文件读取阶段发生错误，请检查文件路径和工作表名称: {e}")
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
        print("\n错误：在'2023年统计的相关数据'表中发现非数字值！")
        problematic_rows = stats_df[stats_df[numeric_cols].isnull().any(axis=1)]
        print("\n请打开 '附件2.xlsx' -> 工作表 '2023年统计的相关数据'，检查并修正以下作物的数值：")
        print(problematic_rows[['作物名称'] + numeric_cols])
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
    print(f"识别到的豆类作物: {params['J_bean']}")

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

def solve_planting_model(I_plots, J_crops, params, case_type='waste'):
    """
    (终极修正版) 构建并求解农作物种植优化模型。
    """
    print(f"\n--- 开始构建模型 Case: {case_type} ---")
    
    # (参数解包部分与之前一致)
    P_area, P_price, P_cost = params['P_area'], params['P_price'], params['P_cost']
    P_yield, P_demand = params['P_yield'], params['P_demand']
    J_bean, P_past, S_suitability = params['J_bean'], params['P_past'], params['S_suitability']

    Y_years, K_seasons = list(range(2024, 2031)), [1, 2]
    A_min, N_j = 0.1, 15 

    model = pyo.ConcreteModel(f"Planting_Optimization_{case_type}")

    # (模型变量和约束定义部分与之前一致，这里省略以保持简洁)
    model.I, model.J, model.Y, model.K = pyo.Set(initialize=I_plots), pyo.Set(initialize=J_crops), pyo.Set(initialize=Y_years), pyo.Set(initialize=K_seasons)
    model.x = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.Binary)
    model.z = pyo.Var(model.I, model.J, model.Y, domain=pyo.Binary)
    model.Sales = pyo.Var(model.J, model.K, model.Y, domain=pyo.NonNegativeReals)
    if case_type == 'discount':
        model.OverSales = pyo.Var(model.J, model.K, model.Y, domain=pyo.NonNegativeReals)
    def objective_rule(m):
        revenue = sum(P_price[j] * m.Sales[j,k,y] for j in m.J for k in m.K for y in m.Y)
        if case_type == 'discount':
            revenue += sum(0.5 * P_price[j] * m.OverSales[j,k,y] for j in m.J for k in m.K for y in m.Y)
        total_cost = sum(P_cost[j] * m.x[i,j,k,y] for i in m.I for j in m.J for k in m.K for y in m.Y)
        return revenue - total_cost
    model.profit = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    def production_rule(m, j, k, y):
        total_prod = P_yield.get(j,0) * sum(m.x[i,j,k,y] for i in m.I)
        if case_type == 'waste': return m.Sales[j,k,y] <= total_prod
        else: return m.Sales[j,k,y] + m.OverSales[j,k,y] == total_prod
    model.prod_con = pyo.Constraint(model.J, model.K, model.Y, rule=production_rule)
    def demand_rule(m, j, k, y): return m.Sales[j,k,y] <= P_demand.get(j, 0)
    model.demand_con = pyo.Constraint(model.J, model.K, model.Y, rule=demand_rule)
    def area_rule(m, i, k, y): return sum(m.x[i,j,k,y] for j in m.J) <= P_area.get(i, 0)
    model.area_con = pyo.Constraint(model.I, model.K, model.Y, rule=area_rule)
    def suitability_rule(m, i, j, k, y): return m.x[i,j,k,y] <= P_area.get(i, 0) * S_suitability.get((i,j,k), 0)
    model.suitability_con = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=suitability_rule)
    def u_link_upper_rule(m, i, j, k, y): return m.x[i,j,k,y] <= P_area.get(i, 0) * m.u[i,j,k,y]
    model.u_link_upper = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=u_link_upper_rule)
    def u_link_lower_rule(m, i, j, k, y): return m.x[i,j,k,y] >= A_min * m.u[i,j,k,y]
    model.u_link_lower = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=u_link_lower_rule)
    def dispersion_rule(m, j, k, y): return sum(m.u[i,j,k,y] for i in m.I) <= N_j
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
    def bean_rule(m, i, y):
        if not J_bean: return pyo.Constraint.Skip
        if y <= Y_years[-3]:
            if y == 2024:
                past_bean = sum(P_past.get((i,j), 0) for j in J_bean)
                return past_bean + sum(m.z[i,j,y_] for j in J_bean for y_ in [2024, 2025]) >= 1
            return sum(m.z[i,j,y_] for j in J_bean for y_ in [y, y+1, y+2]) >= 1
        return pyo.Constraint.Skip
    model.bean_con = pyo.Constraint(model.I, model.Y, rule=bean_rule)
    
    print("--- 模型构建完成，开始求解 ---")
    solver_path = r'CBC\bin\cbc.exe' # 请确保路径正确
    solver = pyo.SolverFactory('cbc', executable=solver_path)
    solver.options['sec'] = 6000
    start_time = time.time()
    results = solver.solve(model, tee=True)
    end_time = time.time()
    print(f"--- 求解完成，耗时: {end_time - start_time:.2f} 秒 ---")
    
    print(f"Solver Status: {results.solver.status}, Termination Condition: {results.solver.termination_condition}")

    # ##################################################################
    # 关键修正：最终版的判断逻辑，直接尝试从结果中加载并读取数值
    # ##################################################################
    solution_found = False
    try:
        # 步骤1：直接尝试加载解到模型中
        model.solutions.load_from(results)
        # 步骤2：直接尝试读取目标函数值
        profit_value = pyo.value(model.profit)
        # 步骤3：如果以上两步都没有报错，说明一个有效的解确实存在
        solution_found = True
    except (ValueError, TypeError, AttributeError):
        # 如果加载或读取失败，说明没有有效的解
        solution_found = False

    if solution_found:
        print("求解成功，找到最优解或近似最优解！")
        print(f"最大总利润: {profit_value:,.2f} 元")

        output = []
        for y in model.Y:
            for k in model.K:
                for i in model.I:
                    for j in model.J:
                        if (i,j,k,y) in model.x:
                            area = pyo.value(model.x[i,j,k,y], exception=False)
                            if area is not None and area > 0.001:
                                output.append({'年份': y, '季节': k, '地块编号': i, '作物名称': j, '种植面积（亩）': round(area, 4)})
        return pd.DataFrame(output)
    else:
        print("求解失败，未找到任何可行的解。状态:", results.solver.termination_condition)
        return None

if __name__ == '__main__':
    data_package = load_and_preprocess_data()
    
    if data_package:
        I_plots, J_crops, params = data_package
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)

        result_case1 = solve_planting_model(I_plots, J_crops, params, case_type='waste')
        if result_case1 is not None:
            output_path1 = os.path.join(results_dir, 'result1_1.xlsx')
            print(f"\n=== 正在保存 情况一 结果至: {output_path1} ===")
            result_case1.to_excel(output_path1, index=False)
            print("保存成功！")

        result_case2 = solve_planting_model(I_plots, J_crops, params, case_type='discount')
        if result_case2 is not None:
            output_path2 = os.path.join(results_dir, 'result1_2.xlsx')
            print(f"\n=== 正在保存 情况二 结果至: {output_path2} ===")
            result_case2.to_excel(output_path2, index=False)
            print("保存成功！")
            
    print("\n所有任务已完成。")