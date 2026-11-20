"""Aggregate and visualize saved formal data; never launches a solver."""
from run_pilot import ROOT, np, plt, MaxNLocator
from study_io import write_csv, resolve_run
import argparse
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import html
import json
import shutil

ALGOS=['ISTA','CD','ADMM']
COLORS=dict(ISTA='#377eb8',CD='#e07a26',ADMM='#2b995b')
METRICS=['total_s','setup_s','update_s','diagnostic_s','compute_s','diagnostic_fraction',
         'iterations','relative_gap','kkt_scaled','objective_delta','test_mse',
         'coefficient_relative_error','support_f1','max_coefficient_difference']


def savefig(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=180)
    fig.savefig(out/f'{name}.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, help='Saved run directory; absolute or relative to study root')
    args=parser.parse_args()
    source=resolve_run(args.run)
    out=ROOT/'outputs'/source.name
    raw=list(csv.DictReader((source/'raw_runs.csv').open()))
    problems=list(csv.DictReader((source/'problems.csv').open()))
    meta=json.loads((source/'metadata.json').read_text())
    if not raw or not meta.get('all_converged') or any(r['converged'] != 'True' for r in raw):
        raise ValueError('Formal summary requires all runs to converge. Preserve failures in raw logs; do not treat capped times as time-to-tolerance.')
    out.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for r in raw:
        groups[(r['job'],r['algorithm'])].append(r)
    per=[]
    representative={}
    for (job,algo),items in groups.items():
        assert len(items)==3
        r=items[0]
        row={k:r[k] for k in ['job','dataset','correlation','seed','algorithm','lam','design_condition','mean_absolute_correlation']}
        for key in METRICS:
            row[key]=None if r[key]=='' else float(np.median([float(x[key]) for x in items]))
        row['all_repeats_converged']=all(x['converged']=='True' for x in items)
        row['timing_repetitions']=len(items)
        per.append(row)
        chosen=min(items,key=lambda x:abs(float(x['total_s'])-row['total_s']))
        representative[(job,algo)]=f"{job}_{algo}_r{chosen['repeat']}"
    write_csv(out/'per_problem.csv',per)
    grouping=defaultdict(list)
    for row in per:
        grouping[(row['dataset'],row['correlation'],row['algorithm'])].append(row)
    agg=[]
    for (ds,c,algo),items in grouping.items():
        row=dict(dataset=ds,correlation=c,algorithm=algo,n_data_seeds=len(items),
                 all_converged=all(x['all_repeats_converged'] for x in items))
        for key in METRICS:
            values=[x[key] for x in items if x[key] is not None]
            row[key+'_mean']=float(np.mean(values)) if values else None
            row[key+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        agg.append(row)
    agg.sort(key=lambda r:(r['dataset'],r['correlation'],ALGOS.index(r['algorithm'])))
    write_csv(out/'aggregate.csv',agg)
    hists={}
    needed=set(representative[(f'synthetic_c{c}_s1',a)] for c in [0.,.5,.9] for a in ALGOS)
    for line in (source/'history.jsonl').open():
        r=json.loads(line)
        if r['run'] in needed:
            hists.setdefault(r['run'],[]).append(r)
    fig,axes=plt.subplots(2,3,figsize=(12,6.3),constrained_layout=True)
    for col,c in enumerate([0.,.5,.9]):
        for algo in ALGOS:
            hist=hists[representative[(f'synthetic_c{c}_s1',algo)]]
            gap=np.maximum([v['relative_gap'] for v in hist],1e-14)
            axes[0,col].semilogy([v['iteration'] for v in hist],gap,label=algo,color=COLORS[algo])
            axes[1,col].semilogy([1000*v['elapsed_s'] for v in hist],gap,color=COLORS[algo])
        axes[0,col].set_title(f'Correlation c = {c}')
        axes[0,col].set_xlabel('Outer iteration (CD: full sweep)')
        axes[1,col].set_xlabel('Elapsed milliseconds, including setup')
        axes[1,col].xaxis.set_major_locator(MaxNLocator(4))
        for ax in axes[:,col]:
            ax.axhline(1e-6,color='gray',linestyle='--',linewidth=.8)
            ax.grid(alpha=.2)
    axes[0,0].legend();axes[0,0].set_ylabel('Duality gap / F(0)');axes[1,0].set_ylabel('Duality gap / F(0)')
    fig.suptitle('Formal study: n=1,000, p=200; seed 1; median-time repetition')
    savefig(fig,out,'formal_convergence')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for algo in ALGOS:
        a=sorted([r for r in agg if r['dataset']=='synthetic' and r['algorithm']==algo],key=lambda r:float(r['correlation']))
        x=[float(r['correlation']) for r in a]
        axes[0].errorbar(x,[1000*r['total_s_mean'] for r in a],yerr=[1000*r['total_s_sd'] for r in a],
                        marker='o',capsize=3,label=algo,color=COLORS[algo])
        axes[1].errorbar(x,[r['iterations_mean'] for r in a],yerr=[r['iterations_sd'] for r in a],
                        marker='o',capsize=3,label=algo,color=COLORS[algo])
    for ax in axes:
        ax.set_xlabel('AR(1) correlation c');ax.set_xticks([0,.5,.9]);ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0].set_ylabel('Total milliseconds (log scale)');axes[1].set_ylabel('Outer iterations (log scale)')
    axes[0].legend()
    fig.suptitle('Mean +/- sample SD across 5 datasets; median of 3 timings per dataset')
    savefig(fig,out,'formal_timing')
    model=[]
    for c in [0.,.5,.9]:
        items=[r for r in problems if r['dataset']=='synthetic' and float(r['correlation'])==c]
        row=dict(correlation=c,n_seeds=len(items))
        for key in ['design_condition','reference_test_mse','reference_coefficient_relative_error','reference_support_f1']:
            vals=[float(r[key]) for r in items]
            row[key+'_mean']=float(np.mean(vals));row[key+'_sd']=float(np.std(vals,ddof=1))
        model.append(row)
    write_csv(out/'model_recovery.csv',model)
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),constrained_layout=True)
    for ax,key,label in zip(axes,['reference_test_mse','reference_coefficient_relative_error','reference_support_f1'],
                            ['Test MSE','Relative coefficient error','Support F1']):
        ax.errorbar([r['correlation'] for r in model],[r[key+'_mean'] for r in model],
                    yerr=[r[key+'_sd'] for r in model],marker='o',capsize=4,color='#5452a0')
        ax.set_xlabel('AR(1) correlation c');ax.set_ylabel(label);ax.set_xticks([0,.5,.9]);ax.grid(alpha=.2)
        if key=='reference_support_f1':ax.set_ylim(0,1.05)
    fig.suptitle('Statistical recovery at the high-accuracy reference solution; 5 seeds')
    savefig(fig,out,'formal_recovery')
    # Generated tables are the sole numerical source used by the paper.
    table='\\begin{tabular}{llrrr}\n\\toprule\nData & Method & Time (ms) & Iterations & Diagnostic (\\%) \\\\\n\\midrule\n'
    for r in agg:
        ds='Diabetes' if r['dataset']=='diabetes' else '$c='+r['correlation']+'$'
        table+=f"{ds} & {r['algorithm']} & {1000*r['total_s_mean']:.2f} $\\pm$ {1000*r['total_s_sd']:.2f} & {r['iterations_mean']:.1f} $\\pm$ {r['iterations_sd']:.1f} & {100*r['diagnostic_fraction_mean']:.1f} \\\\\n"
    table+='\\bottomrule\n\\end{tabular}\n'
    (out/'timing_table.tex').write_text(table)
    table='\\begin{tabular}{lrrrr}\n\\toprule\n$c$ & $\\kappa_2(X)$ & Test MSE & Rel. coeff. error & Support F1 \\\\\n\\midrule\n'
    for r in model:
        values=[]
        for key in ['design_condition','reference_test_mse','reference_coefficient_relative_error','reference_support_f1']:
            values.append(f"{r[key+'_mean']:.3f} $\\pm$ {r[key+'_sd']:.3f}")
        table+=str(r['correlation'])+' & '+' & '.join(values)+' \\\\\n'
    table+='\\bottomrule\n\\end{tabular}\n'
    (out/'recovery_table.tex').write_text(table)
    (out/'analysis_data.json').write_text(json.dumps(dict(aggregate=agg,recovery=model,metadata=meta),indent=2))
    def tr(r):
        ds='Diabetes' if r['dataset']=='diabetes' else '合成 c='+r['correlation']
        return f"<tr><td>{ds}</td><td>{r['algorithm']}</td><td>{r['n_data_seeds']}</td><td>{1000*r['total_s_mean']:.3f} ± {1000*r['total_s_sd']:.3f}</td><td>{r['iterations_mean']:.1f} ± {r['iterations_sd']:.1f}</td><td>{r['test_mse_mean']:.4g} ± {r['test_mse_sd']:.4g}</td><td>{100*r['diagnostic_fraction_mean']:.1f}%</td></tr>"
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LASSO 正式结果</title><style>body{font:16px/1.8 -apple-system,"PingFang SC",sans-serif;max-width:1100px;margin:35px auto;padding:0 24px;color:#243047}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border:1px solid #dbe2e9;text-align:left}th{background:#edf3f8}img{width:100%}a{color:#1765a1}.note{background:#edf7fb;padding:18px}</style><h1>LASSO 正式实验结果</h1><p>2026-11-19 · HOU Zhitian · 26017658r</p>'''
    report+=f"<p class='note'>18 个数据问题、54 个问题/算法组合、162 次计时求解全部达到预设相对对偶间隙 ≤10⁻⁶。预设参数未因正式结果修改。跨重复的迭代数和目标值完全一致；均值与样本 SD 按独立数据种子计算，不把计时重复当成额外数据样本。</p>"
    report+='<h2>时间与精度</h2><p>每个数据问题先对 3 次计时取中位数，再跨 5 个合成数据种子或 3 个真实数据划分报告均值 ± 样本 SD（ddof=1）。时间包括方法准备与逐轮证书；CD 的一次迭代是一次完整扫描。单线程被请求，Apple Accelerate 实际线程数无法完全枚举。</p><table><tr><th>数据</th><th>方法</th><th>种子数</th><th>毫秒</th><th>外层迭代</th><th>测试 MSE</th><th>诊断时间占比</th></tr>'
    report+=''.join(tr(r) for r in agg)+'</table><img src="formal_timing.png" alt="正式计时汇总">'
    report+='<h2>收敛曲线</h2><p>展示固定种子 1 下各方法的中位耗时重复。跨种子的总体结论以上表为准。</p><img src="formal_convergence.png" alt="正式收敛曲线">'
    report+='<h2>恢复与泛化</h2><p>本图使用独立高精度参考解，隔离模型统计行为与求解器停止误差。三算法达到相同目标精度后，预测与恢复指标应接近；不能把更快收敛解释为改变了 LASSO 模型的统计能力。</p><img src="formal_recovery.png" alt="统计恢复指标">'
    report+='<h2>分析边界</h2><ul><li>研究的是这套小型稠密 NumPy/SciPy 实现，不是所有库或硬件上的普遍速度排名。CD 使用 Python 循环；ADMM 缓存稠密 Cholesky；ISTA 的精确谱范数准备成本已计入。</li><li>逐轮证书占用的时间单独列出，不应被误解为算法更新本身的全部成本。小型真实数据任务尤其容易受系统计时噪声影响。</li><li>相关性提高同时改变条件数、信号分布与 λmax。本研究采用 λ=0.05λmax 的固定相对正则化协议，不声称单独识别了条件数的因果效应。</li><li>没有大规模稀疏数据、过参数化模型或 λ 搜索；真实数据的验证划分保留未用，未根据测试结果调参。Diabetes 只补充优化求解实验，不用于临床结论。</li></ul>'
    report+=f'<p>所有求解的最大相对对偶间隙：{meta["max_relative_gap"]:.3e}；相对各自 sklearn 参考解的最大目标值差：{meta["max_objective_delta"]:.3e}。</p>'
    report+=f'<p><a href="aggregate.csv">正式汇总 CSV</a> · <a href="per_problem.csv">54 行逐问题中位数</a> · <a href="../../results/{source.name}/raw_runs.csv">162 行原始计时</a> · <a href="../../results/{source.name}/metadata.json">环境与验证状态</a> · <a href="../../research_plan.html">模型与实验方案</a></p></html>'
    (out/'formal_report.html').write_text(report)
    shutil.copy2(ROOT/'export_formal.py',out/'export_formal.py')
    manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    (out/'raw_sha256.json').write_text(json.dumps(manifest,indent=2))
    print(out)
    print(json.dumps(dict(aggregate=agg,recovery=model),indent=2))


if __name__=='__main__':
    main()
