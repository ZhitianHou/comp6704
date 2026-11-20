"""Summaries/figures from the saved additional real dataset; no training."""
from study_common import ROOT, np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from study_io import write_csv
from pathlib import Path
from collections import defaultdict
import csv
import json


def main():
    source=ROOT/'results'/Path((ROOT/'latest_realdata.txt').read_text().strip()).name
    out=ROOT/'outputs'/source.name;out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((source/'raw_runs.csv').open()))
    metadata=json.loads((source/'metadata.json').read_text())
    if len(rows)!=27 or not metadata['all_converged'] or any(r['converged']!='True' for r in rows):
        raise ValueError('Incomplete or unconverged additional-dataset run')
    grouping=defaultdict(list)
    for r in rows:
        grouping[(r['seed'],r['algorithm'])].append(r)
    metrics=['total_s','setup_s','update_s','diagnostic_s','iterations','relative_gap','test_mse','test_r2','objective_delta','design_condition']
    per=[]
    for (seed,a),items in grouping.items():
        assert len(items)==3 and len({r['iterations'] for r in items})==1
        row=dict(seed=int(seed),algorithm=a,n_train=int(items[0]['n_train']),n_test=int(items[0]['n_test']))
        for key in metrics:
            row[key]=float(np.median([float(r[key]) for r in items]))
        per.append(row)
    write_csv(out/'per_problem.csv',per)
    aggregate=[]
    for a in ['ISTA','CD','ADMM']:
        items=[r for r in per if r['algorithm']==a]
        row=dict(algorithm=a,n_splits=len(items))
        for key in metrics:
            row[key+'_mean']=float(np.mean([r[key] for r in items]))
            row[key+'_sd']=float(np.std([r[key] for r in items],ddof=1))
        aggregate.append(row)
    write_csv(out/'aggregate.csv',aggregate)
    table='\\begin{tabular}{lrrrr}\n\\toprule\nMethod & Time (ms) & Iterations & Test MSE ($\\mathrm{K}^2$) & Test $R^2$ \\\\\n\\midrule\n'
    for r in aggregate:
        table+=f"{r['algorithm']} & {1000*r['total_s_mean']:.1f} $\\pm$ {1000*r['total_s_sd']:.1f} & {r['iterations_mean']:.0f} $\\pm$ {r['iterations_sd']:.0f} & {r['test_mse_mean']:.2f} $\\pm$ {r['test_mse_sd']:.2f} & {r['test_r2_mean']:.3f} $\\pm$ {r['test_r2_sd']:.3f} \\\\\n"
    (out/'realdata_table.tex').write_text(table+'\\bottomrule\n\\end{tabular}\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),constrained_layout=True)
    for idx,metric in enumerate(['total_s','test_mse']):
        scale=1000 if metric=='total_s' else 1
        axes[idx].bar(['ISTA','CD','ADMM'],[scale*r[metric+'_mean'] for r in aggregate],
            yerr=[scale*r[metric+'_sd'] for r in aggregate],capsize=4,color=['#377eb8','#e07a26','#2b995b'])
        axes[idx].set_ylabel('Time (ms, log scale)' if metric=='total_s' else 'Test MSE (K squared)')
        if metric=='total_s':axes[idx].set_yscale('log')
    fig.suptitle('Superconductivity: 3 duplicate-aware splits; median of 3 timings per split')
    fig.savefig(out/'realdata.png',dpi=180);fig.savefig(out/'realdata.pdf');plt.close(fig)
    (out/'analysis.json').write_text(json.dumps(dict(aggregate=aggregate,metadata=metadata),indent=2))
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>新增真实数据实验</title><style>body{font:16px/1.8 sans-serif;max-width:1000px;margin:40px auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>UCI 超导体临界温度回归实验</h1><p>官方原始数据 21,263 行、81 特征。按完全相同的特征向量建立 15,170 个组，组别在训练/保留验证/测试间不重叠。保留全部行；仅用训练集做标准化，不用测试集调参。三算法使用原主实验参数和精度，27 次求解全部收敛。</p><img src="realdata.png"><p>时间包括准备与精度检查。先对每划分三次计时取中位数，再跨三个划分计算均值与样本 SD。原数据已工程化；分组不等同于按化学家族划分。本实验验证求解效率，不宣称材料科学预测性能最优。</p><pre>'+json.dumps(aggregate,ensure_ascii=False,indent=2)+'</pre><p><a href="aggregate.csv">汇总 CSV</a> · <a href="https://doi.org/10.24432/C53P47">UCI 数据来源</a></p></html>'
    (out/'realdata_report.html').write_text(html)
    print(out);print(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    main()
