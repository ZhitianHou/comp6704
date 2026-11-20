"""Export the saved sensitivity study, without optimization or tuning."""
from run_pilot import ROOT, np, plt
from study_io import write_csv
from pathlib import Path
from collections import defaultdict
import csv
import json


def main():
    source=ROOT/'results'/Path((ROOT/'latest_ablations.txt').read_text().strip()).name
    out=ROOT/'outputs'/source.name;out.mkdir(exist_ok=True,parents=True)
    rows=list(csv.DictReader((source/'raw_runs.csv').open()))
    meta=json.loads((source/'metadata.json').read_text())
    assert len(rows)==210 and meta['all_converged']
    groups=defaultdict(list)
    for r in rows:
        groups[(float(r['correlation']),int(r['seed']),r['algorithm'],float(r['rho']),int(r['check_every']))].append(r)
    per=[]
    for (c,seed,algo,rho,q),items in groups.items():
        assert len(items)==3 and all(r['converged']=='True' for r in items)
        assert len({r['iterations'] for r in items})==1
        row=dict(correlation=c,seed=seed,algorithm=algo,rho=rho,check_every=q)
        for key in ['total_s','iterations','certificate_checks','relative_gap','diagnostic_s']:
            row[key]=float(np.median([float(r[key]) for r in items]))
        per.append(row)
    write_csv(out/'per_problem.csv',per)
    groups=defaultdict(list)
    for r in per:
        groups[(r['correlation'],r['algorithm'],r['rho'],r['check_every'])].append(r)
    agg=[]
    for (c,a,rho,q),items in groups.items():
        assert len(items)==5
        row=dict(correlation=c,algorithm=a,rho=rho,check_every=q,n_seeds=5)
        for key in ['total_s','iterations','certificate_checks','relative_gap','diagnostic_s']:
            row[key+'_mean']=float(np.mean([r[key] for r in items]))
            row[key+'_sd']=float(np.std([r[key] for r in items],ddof=1))
        agg.append(row)
    agg.sort(key=lambda r:(r['correlation'],['ISTA','CD','ADMM'].index(r['algorithm']),r['rho'],r['check_every']))
    write_csv(out/'aggregate.csv',agg)
    pairs=[]
    for c in [0.,.9]:
        for a in ['ISTA','CD','ADMM']:
            for seed in range(1,6):
                x=next(r for r in per if (r['correlation'],r['algorithm'],r['seed'],r['rho'],r['check_every'])==(c,a,seed,1.,1))
                y=next(r for r in per if (r['correlation'],r['algorithm'],r['seed'],r['rho'],r['check_every'])==(c,a,seed,1.,10))
                pairs.append(dict(correlation=c,algorithm=a,seed=seed,
                                  speedup=x['total_s']/y['total_s'],extra_updates=y['iterations']-x['iterations']))
    write_csv(out/'frequency_pairs.csv',pairs)
    summary=[]
    for c in [0.,.9]:
        for a in ['ISTA','CD','ADMM']:
            p=[r for r in pairs if r['correlation']==c and r['algorithm']==a]
            summary.append(dict(correlation=c,algorithm=a,speedup_mean=float(np.mean([r['speedup'] for r in p])),
                                speedup_min=min(r['speedup'] for r in p),speedup_max=max(r['speedup'] for r in p),
                                max_extra_updates=max(r['extra_updates'] for r in p)))
    chosen=[r for r in agg if r['correlation']==.9]
    table='\\begin{tabular}{lrrrr}\n\\toprule\nMethod & $q$ & $\\rho$ & Total ms & Outer iterations \\\\\n\\midrule\n'
    for r in chosen:
        table+=f"{r['algorithm']} & {r['check_every']} & {r['rho']:g} & {1000*r['total_s_mean']:.2f} $\\pm$ {1000*r['total_s_sd']:.2f} & {r['iterations_mean']:.1f} $\\pm$ {r['iterations_sd']:.1f} \\\\\n"
    (out/'ablation_table.tex').write_text(table+'\\bottomrule\n\\end{tabular}\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    for q,offset,color in [(1,-.18,'#377eb8'),(10,.18,'#e07a26')]:
        a=[next(r for r in chosen if r['algorithm']==name and r['rho']==1. and r['check_every']==q) for name in ['ISTA','CD','ADMM']]
        axes[0].bar(np.arange(3)+offset,[1000*r['total_s_mean'] for r in a],yerr=[1000*r['total_s_sd'] for r in a],width=.34,label=f'Check every {q}',color=color,capsize=3)
    axes[0].set_xticks(range(3),['ISTA','CD','ADMM']);axes[0].set_ylabel('Total milliseconds');axes[0].legend()
    a=[next(r for r in chosen if r['algorithm']=='ADMM' and r['rho']==rho and r['check_every']==1) for rho in [.1,1.,10.]]
    axes[1].errorbar([r['rho'] for r in a],[1000*r['total_s_mean'] for r in a],yerr=[1000*r['total_s_sd'] for r in a],marker='o',capsize=4,color='#2b995b')
    axes[1].set_xscale('log');axes[1].set_yscale('log');axes[1].set_xlabel('ADMM penalty rho (q=1)');axes[1].set_ylabel('Total milliseconds (log scale)')
    fig.suptitle('Post-hoc sensitivity: correlation 0.9; mean +/- sample SD over 5 seeds')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'ablations.png',dpi=180);fig.savefig(out/'ablations.pdf');plt.close(fig)
    (out/'analysis.json').write_text(json.dumps(dict(aggregate=agg,frequency_summary=summary,metadata=meta),indent=2))
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LASSO 补充实验</title><style>body{font:16px/1.8 sans-serif;max-width:1000px;margin:40px auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>诊断频率消融与 ADMM 参数敏感性</h1><p>这是主实验之后设计的稳健性分析。原主实验参数和表格保持不变。210 次求解均达原预设精度；所有比较使用本轮同时测量的配对基线，不与上次运行的耗时直接混比。</p><img src="ablations.png"><p>三次计时取中位数，再跨五种子计算均值和样本 SD。q=10 只改变检查时点，最终仍必须通过同样的对偶间隙判据；它可能多做更新。rho 不改变 LASSO 目标，但改变 ADMM 的数值收敛速度。这些敏感性结果不用于替换主实验参数。</p><pre>'+json.dumps(dict(aggregate=agg,frequency_summary=summary),ensure_ascii=False,indent=2)+'</pre><p><a href="aggregate.csv">汇总 CSV</a> · <a href="frequency_pairs.csv">逐种子配对分析</a></p></html>'
    (out/'ablation_report.html').write_text(html)
    print(out)
    print(json.dumps(dict(aggregate=agg,frequency_summary=summary),indent=2))


if __name__=='__main__':
    main()
