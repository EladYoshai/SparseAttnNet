"""FLOPs of SparseAttnNet for every pixel budget k used in the paper (for the accuracy-versus-FLOPs figure).
k values: the mean selected k of each dataset's full model (results/<dataset>/ours) and the fixed-k / full-image
ablations. FLOPs counted with fvcore end to end, as in benchmark.py. Runs on the CPU.
Output: results/benchmark/flops_by_k.json
Usage: python flops_by_k.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from core import Config
import benchmark as B

ABLATION_K = {'event': [1500, 3000, 5000], 'wbc': [4000, 8000, 12000]}


def main():
    out = {}
    for ds in ['event', 'holos', 'opd', 'wbc']:
        p = os.path.join(core.ROOT, 'results', ds, 'ours', 'summary.json')
        ks = list(ABLATION_K.get(ds, []))
        if os.path.exists(p):
            ks.append(int(round(json.load(open(p))['k']['mean'])))
        Config.update(ds, 'ours')
        if ds in ABLATION_K:
            ks.append(core.full_k())
        m = core.get_model('ours').eval(); x = B.make_input('ours', ds, 1, 'cpu')
        out[ds] = {str(k): B.flops(B.EndToEnd(m, k).eval(), (x,))[0] / 1e9 for k in sorted(set(ks))}
        print(ds, {k: round(v, 4) for k, v in out[ds].items()}, flush=True)
    od = os.path.join(core.ROOT, 'results', 'benchmark'); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, 'flops_by_k.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
