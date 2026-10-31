"""Paired statistics: SparseAttnNet ('ours') versus every other finished run on the SAME outer folds.

Per dataset and comparison: mean fold difference (ours minus other, accuracy in percentage points) with a 95 %
confidence interval (t distribution, df = 4), paired t-test, Nadeau-Bengio corrected resampled t-test
(variance x (1/J + n_test/n_train)) and Wilcoxon signed-rank test (with 5 folds the smallest attainable two-sided
p is 0.0625). Also the macro-F1 fold difference with the same t-based interval.
Output: results/paired_stats.md and results/paired_stats.json
Usage: python paired_stats.py [--results results]
"""
import os, json, glob, argparse
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_run(d):
    folds = {}
    for fd in sorted(glob.glob(os.path.join(d, 'fold_*'))):
        mf = os.path.join(fd, 'metrics.json')
        if not os.path.exists(mf): continue
        f = int(fd.rsplit('_', 1)[1]); m = json.load(open(mf)); sp = json.load(open(os.path.join(fd, 'split.json')))
        folds[f] = {'acc': m['accuracy'], 'macro_f1': m['macro']['f1'], 'hash': sp['test_hash'],
                    'train_hash': sp['train_hash'], 'val_hash': sp['val_hash'], 'n_test': sp['n_test'],
                    'n_train': sp['n_train'] + sp['n_val']}
    return folds


def tci(d):
    d = np.asarray(d, float); J = len(d); m = d.mean(); s = d.std(ddof=1)
    h = stats.t.ppf(0.975, J - 1) * s / np.sqrt(J) if J > 1 else np.nan
    return m, m - h, m + h


def compare(a, b, name=''):
    # both runs must have all 5 folds with identical outer-test, inner-train and inner-validation splits
    assert sorted(a) == list(range(5)) and sorted(b) == list(range(5)), f'{name}: not 5 folds ({sorted(a)} vs {sorted(b)})'
    fs = list(range(5))
    for f in fs:
        for h in ('hash', 'train_hash', 'val_hash'):
            assert a[f][h] == b[f][h], f'{name} fold {f}: {h} differs ({a[f][h]} vs {b[f][h]})'
    d = np.array([a[f]['acc'] - b[f]['acc'] for f in fs]) * 100
    df1 = np.array([a[f]['macro_f1'] - b[f]['macro_f1'] for f in fs])
    J = len(fs)
    out = {'folds': fs, 'diff_pp_per_fold': d.tolist()}
    m, lo, hi = tci(d); out.update({'mean_diff_pp': m, 'ci95_pp': [lo, hi]})
    if d.std(ddof=1) > 0:
        t, p = stats.ttest_rel([a[f]['acc'] for f in fs], [b[f]['acc'] for f in fs])
        ratio = np.mean([a[f]['n_test'] / a[f]['n_train'] for f in fs])
        tc = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1))
        pc = 2 * stats.t.sf(abs(tc), J - 1)
        out.update({'t': float(t), 'p_t': float(p), 't_corrected_NB': float(tc), 'p_corrected_NB': float(pc),
                    'n_test_over_n_train': float(ratio)})
        try:
            out['p_wilcoxon'] = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            out['p_wilcoxon'] = None
    mf, flo, fhi = tci(df1); out.update({'mean_diff_macroF1': mf, 'ci95_macroF1': [flo, fhi]})
    return out


def fmt_p(p):
    return 'n/a' if p is None or (isinstance(p, float) and np.isnan(p)) else (f'{p:.3f}' if p >= 0.001 else '<0.001')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results'); a = ap.parse_args()
    base = os.path.join(ROOT, a.results)
    allres = {}
    md = ['# Paired comparisons: SparseAttnNet versus each run on the same outer folds', '',
          'Difference = ours minus other, accuracy in percentage points (pp). 95 % CI from the t distribution over folds. ',
          'NB = Nadeau-Bengio corrected resampled t-test. Wilcoxon: with 5 folds the smallest possible two-sided p is 0.0625.', '']
    for ds in sorted(os.listdir(base)):
        dsd = os.path.join(base, ds)
        if not os.path.isdir(os.path.join(dsd, 'ours')): continue
        ours = load_run(os.path.join(dsd, 'ours'))
        if not ours: continue
        allres[ds] = {}
        md += [f'## {ds} (ours accuracy {np.mean([v["acc"] for v in ours.values()])*100:.2f} %)', '',
               '| other | ours minus other (pp) | 95 % CI | p (t) | p (NB) | p (Wilcoxon) | macro-F1 difference |',
               '|---|---|---|---|---|---|---|']
        for tag in sorted(os.listdir(dsd)):
            if tag == 'ours' or not os.path.isdir(os.path.join(dsd, tag)): continue
            other = load_run(os.path.join(dsd, tag))
            if len(other) < 5: continue
            r = compare(ours, other, f'{ds}/ours vs {tag}'); allres[ds][tag] = r
            md.append(f"| {tag} | {r['mean_diff_pp']:+.2f} | [{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}] | "
                      f"{fmt_p(r.get('p_t'))} | {fmt_p(r.get('p_corrected_NB'))} | {fmt_p(r.get('p_wilcoxon'))} | "
                      f"{r['mean_diff_macroF1']:+.4f} |")
        md.append('')
    json.dump(allres, open(os.path.join(base, 'paired_stats.json'), 'w'), indent=1)
    open(os.path.join(base, 'paired_stats.md'), 'w').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
