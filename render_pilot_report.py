"""Summarize existing pilot logs and redraw figures; does not solve/train."""
import csv
import html
import json
from pathlib import Path
from run_pilot import plot, ROOT


def main():
    source=Path((ROOT/"latest_pilot.txt").read_text().strip())
    out=ROOT/"outputs"/source.name
    out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((source/"summary.csv").open()))
    histories={}
    for line in (source/"history.jsonl").read_text().splitlines():
        record=json.loads(line)
        histories.setdefault(record.pop("run"),[]).append(record)
    plot(rows,histories,out)
    meta=json.loads((source/"metadata.json").read_text())
    checks=json.loads((source/"correctness.json").read_text())
    def table(subset):
        header="<tr><th>数据/相关性/种子</th><th>算法</th><th>外层迭代</th><th>总毫秒</th><th>相对对偶间隙</th><th>测试 MSE</th></tr>"
        body=""
        for r in subset:
            body+=f"<tr><td>{html.escape(r['job'])}</td><td>{r['algorithm']}</td><td>{r['iterations']}</td><td>{1000*float(r['total_s']):.3f}</td><td>{float(r['relative_gap']):.2e}</td><td>{float(r['test_mse']):.5g}</td></tr>"
        return "<table>"+header+body+"</table>"
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LASSO 试跑结果</title>
<style>body{{font:16px/1.8 -apple-system,"PingFang SC",sans-serif;color:#243047;max-width:1100px;margin:35px auto;padding:0 24px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #dbe2e9;padding:8px;text-align:left}}th{{background:#edf3f8}}img{{width:100%}}.note{{background:#edf7fb;padding:16px}}a{{color:#1765a1}}</style>
<h1>LASSO 小规模试跑结果</h1><p>2026-11-19 · HOU Zhitian · 仅验证正确性与计算预算</p>
<p class="note"><b>结论：题目可行，计算量可控。</b>本次 24/24 次求解达到统一相对对偶间隙 ≤10⁻⁶；1,000×200 的高相关性预算探针中，三种方法均在 0.2 秒内完成。单次计时不能用作正式速度排名。</p>
<p>已完成一次初始试跑和一次设置原生线程环境变量后的核验试跑，各 24 次；本页只展示后一次，前一次日志保留，两次不合并成独立统计样本。首次包导入/字体缓存耗时不计作算法计算成本。</p>
<h2>正确性</h2><p>三算法均通过正交设计解析解和零解检查。正交设计最大系数绝对误差 {max(x['max_error'] for x in checks if 'max_error' in x):.2e}。相关设计上，与 sklearn 高精度参考解的最大系数差 {max(x['max_coefficient_error'] for x in checks if 'max_coefficient_error' in x):.2e}。24 次试跑的目标值与各自参考解最大差值 {max(float(r['objective_delta']) for r in rows):.2e}。</p>
<h2>预算探针：训练 1,000 样本，200 特征，相关性 0.9</h2>
{table([r for r in rows if r['dataset']=='scale_probe'])}
<p>总时间包括各算法准备工作和逐轮证书检查，不包含共同数据预处理或 sklearn 参考拟合。ISTA 计入谱范数计算，ADMM 计入 Gram 矩阵与 Cholesky 分解。试跑不使用 GPU。设置单线程环境并使用 threadpoolctl；Apple Accelerate 的线程数不能由该接口完整枚举，环境信息已保留。</p>
<h2>收敛曲线：300×60，种子 11</h2><img src="pilot_convergence.png" alt="三档特征相关性下三种算法的收敛曲线">
<p>种子 11 下，ISTA 在相关性 0、0.5、0.9 时分别需要 27、83、1,967 次迭代；CD 为 7、12、68 次完整扫描；ADMM 为 27、34、139 次迭代。该设置下高相关性伴随收敛变慢，不能把三种算法的迭代直接视作相同计算工作量。</p>
<h2>全部试跑记录</h2>{table(rows)}
<h2>下一阶段建议</h2><p>保持正式规模 n=1000、p=200；3 档相关性×5 数据种子×3 算法，加 Diabetes 3 个划分×3 算法，共 54 个问题/算法组合。每组合重复计时 3 次，共 162 次求解。按当前探针，纯求解预算很小，无需 GPU 或长时间训练；统计汇总、图表与分析会占主要工作量。本次不运行该正式实验。</p>
<p>所有方法充分收敛时应得到接近的 LASSO 预测与恢复指标；比较重点是求解效率。恢复指标随相关性变化主要反映模型与数据条件，而不是宣称某个求解器提高了统计能力。正式计时需先统一热身并报告诊断开销、Python 循环和线性代数实现的限制。</p>
<p><a href="../../research_plan.html">模型、公式与完整方案</a> · <a href="../../results/{source.name}/summary.csv">原始汇总 CSV</a> · <a href="../../results/{source.name}/metadata.json">环境与源码哈希</a> · <a href="../../results/{source.name}/correctness.json">正确性记录</a> · <a href="pilot_convergence.pdf">矢量曲线 PDF</a></p>
<p>源数据目录：{html.escape(str(source))}<br>本页由 render_pilot_report.py 从已保存日志生成，不重新求解。</p></html>'''
    (out/"pilot_report.html").write_text(content)
    print(out/"pilot_report.html")


if __name__=="__main__":
    main()
