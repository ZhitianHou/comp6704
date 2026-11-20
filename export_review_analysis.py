"""Derive conditioning/paired-time summaries from existing problems; no optimization."""
from study_common import ROOT, np, synthetic, threadpool_limits
from study_io import resolve_run
import csv
import json


def main():
    source=resolve_run()
    config=json.loads((source/'config.json').read_text())
    out=ROOT/'outputs'/source.name
    syn=config['synthetic']
    records=[]
    with threadpool_limits(limits=1):
        for c in syn['correlations']:
            for seed in syn['seeds']:
                X,*_=synthetic(syn['n_train'],syn['p'],c,seed,n_test=syn['n_test'],sparsity=syn['sparsity'],noise_sd=syn['noise_sd'])
                sv=np.linalg.svd(X,compute_uv=False)
                records.append(dict(c=c,seed=seed,L=float(sv[0]**2/len(X)),mu=float(sv[-1]**2/len(X))))
    problems=list(csv.DictReader((source/'problems.csv').open()))
    rows=list(csv.DictReader((out/'per_problem.csv').open()))
    summary=[]
    for c in syn['correlations']:
        spectra=[r for r in records if r['c']==c]
        lambdas=[float(r['lam']) for r in problems if r['dataset']=='synthetic' and float(r['correlation'])==c]
        ratios=[]
        for seed in syn['seeds']:
            d={r['algorithm']:r for r in rows if r['job']==f'synthetic_c{c}_s{seed}'}
            ratios.append(float(d['ISTA']['total_s'])/float(d['ADMM']['total_s']))
        summary.append(dict(c=c,L_mean=float(np.mean([r['L'] for r in spectra])),
            mu_mean=float(np.mean([r['mu'] for r in spectra])),lambda_mean=float(np.mean(lambdas)),
            ista_over_admm_paired_ratio_min=min(ratios),ista_over_admm_paired_ratio_max=max(ratios)))
    table='\\begin{tabular}{lrrrr}\n\\toprule\n$c$ & Mean $L$ & Mean $\\mu$ & Mean $\\lambda$ & ISTA/ADMM time ratio \\\\\n\\midrule\n'
    for r in summary:
        table+=f"{r['c']} & {r['L_mean']:.3f} & {r['mu_mean']:.3f} & {r['lambda_mean']:.3f} & {r['ista_over_admm_paired_ratio_min']:.2f}--{r['ista_over_admm_paired_ratio_max']:.2f} \\\\\n"
    (out/'spectral_table.tex').write_text(table+'\\bottomrule\n\\end{tabular}\n')
    (out/'review_analysis.json').write_text(json.dumps(dict(spectra=records,summary=summary),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
