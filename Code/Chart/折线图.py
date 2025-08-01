# 文件名: 折线图.py, 功能同上, 路径已修正
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_line_chart():
    # 路径设置
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'result')
        os.makedirs(output_dir, exist_ok=True)
    except NameError:
        output_dir = 'result'
        os.makedirs(output_dir, exist_ok=True)
        
    # !!重要提示!!: 此处需要您运行GA后记录的数据。
    print("注意：正在使用内置的示例数据。")
    generations = range(1, 51)
    base_fitness = 100 + 20 * np.log(generations)
    best_fitness = base_fitness + np.random.rand(50) * 10
    avg_fitness = base_fitness - 10 + np.random.rand(50) * 5
    df_convergence = pd.DataFrame({
        '代数': generations, '最优适应度': best_fitness, '平均适应度': avg_fitness
    })

    # 可视化
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 'SimHei' 字体未找到。")
        
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="darkgrid")
    plt.plot(df_convergence['代数'], df_convergence['最优适应度'], marker='o', markersize=4, linestyle='-', label='最优适应度')
    plt.plot(df_convergence['代数'], df_convergence['平均适应度'], marker='x', markersize=4, linestyle='--', label='平均适应度')
    plt.fill_between(df_convergence['代数'], df_convergence['平均适应度'], df_convergence['最优适应度'], color='skyblue', alpha=0.3)
    
    plt.title('遗传算法收敛曲线', fontsize=18, pad=15)
    plt.xlabel('迭代代数', fontsize=14)
    plt.ylabel('适应度 (预期平均总利润)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # 保存
    output_filename = os.path.join(output_dir, '6_GA收敛过程_折线图.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图像已成功保存至: {output_filename}")
    plt.show()

if __name__ == '__main__':
    generate_line_chart()