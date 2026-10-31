"""End-to-end efficiency benchmark (FLOPs, parameters, latency, throughput, memory).

SparseAttnNet is timed END TO END from the raw image: coarse network -> border mask + top-k -> gather ->
point embedding -> fine attention -> coarse-feature pooling -> fusion head.
FLOPs are counted per stage with fvcore (one fused multiply-add = 1 FLOP, the same convention as thop/fvcore and
most model-zoo tables), including the einsum attention operations. Top-k and gather are memory operations and are
timed but have no multiply-adds.
Baselines run at the input size they were trained with: 135x135 for ResNet50/MobileNetV2, 224x224 for
ConvNeXt-Tiny/DeiT-Small/ViT-Small/Swin-Tiny.
Protocol: FP32, eval mode, torch.no_grad, 50 warm-up + 200 timed iterations, torch.cuda.synchronize around timing,
batch 1 (latency) and 32 (throughput), 5 repetitions -> mean +- sd, peak GPU memory at batch 32.
CPU: batch 1 latency, 10 warm-up + 50 timed, 5 repetitions.
Usage: python benchmark.py --dataset holos            (k = mean selected k of results/holos/ours)
       python benchmark.py --dataset wbc --k 7685
Output: results/benchmark/<dataset>_k<k>.json
"""
import os, sys, json, time, argparse, platform
os.environ['TIMM_FUSED_ATTN'] = '0'  # count attention matmuls explicitly (fused SDPA is invisible to fvcore)
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from core import Config


class Coarse(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x4): return self.m.attention_mask_model(x4)


class Embed(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, pts): return self.m.point_encoder(pts, self.m.image_size)


class FineAttn(nn.Module):
    """Fine attention given embedded points (everything in model.fine after the encoder)."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, enc):
        m = self.m; b = enc.shape[0]
        enc = torch.cat([enc, m.cls_token.expand(b, -1, -1)], 1)
        v = m.value_proj(enc)
        q = torch.einsum('bnd,hde->bhne', enc, m.query_proj)
        kk = torch.einsum('bnd,hde->bhne', enc, m.key_proj)
        kp = torch.relu(kk) + 1e-6
        aw = kp / kp.sum(dim=2, keepdim=True).clamp(min=1e-6)
        ctx = torch.einsum('bhne,bnd->bhed', aw, v)
        ho = torch.einsum('bhne,bhed->bhnd', q, ctx)
        return ho[:, :, -1, :].mean(1)


class Head(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, inter, cls):
        mf = self.m.mask_feature_extractor(inter)
        return self.m.fc_out(self.m.fusion_layer(torch.cat([cls, mf], -1)))


class EndToEnd(nn.Module):
    def __init__(self, m, k):
        super().__init__(); self.m = m; self.k = k
    def forward(self, imgs):
        x4 = core.to4d(imgs)
        mask = self.m.attention_mask_model(x4).squeeze(1)
        inter = self.m.attention_mask_model.intermediate
        pts, _ = core.select_points(imgs, mask, self.k)
        cls, _ = self.m.fine(pts)
        mf = self.m.mask_feature_extractor(inter)
        return self.m.fc_out(self.m.fusion_layer(torch.cat([cls, mf], -1)))


UNSUPPORTED = {}


def flops(module, inputs, name=''):
    from fvcore.nn import FlopCountAnalysis
    fa = FlopCountAnalysis(module, inputs); fa.uncalled_modules_warnings(False)  # unsupported-op warnings stay on
    total = fa.total()
    UNSUPPORTED[name or type(module).__name__] = dict(fa.unsupported_ops())  # logged in the output json
    return total, dict(fa.by_operator())


def time_it(fn, device, warm, iters, reps):
    out = []
    for _ in range(reps):
        with torch.no_grad():
            for _ in range(warm): fn()
            if device == 'cuda': torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(iters): fn()
            if device == 'cuda': torch.cuda.synchronize()
        out.append((time.perf_counter() - t) / iters)
    return float(np.mean(out)), float(np.std(out))


def make_input(model_type, ds, bs, device):
    Config.update(ds, model_type)
    s = Config.IMAGE_SIZE
    if model_type == 'ours':
        if ds == 'wbc': return torch.rand(bs, 3, s, s, device=device)
        if ds == 'opd': return torch.rand(bs, 1, s, s, device=device)
        return torch.rand(bs, s, s, device=device)
    return torch.rand(bs, 3, s, s, device=device)


def bench(model_type, ds, k, res):
    Config.update(ds, model_type)
    m = core.get_model(model_type).eval()
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    net = EndToEnd(m, k).eval() if model_type == 'ours' else m
    row = {'model': model_type, 'input': Config.IMAGE_SIZE, 'params_M': params / 1e6}
    x1 = make_input(model_type, ds, 1, 'cpu')
    total, byop = flops(net, (x1,), model_type)
    row['fvcore_unsupported_ops'] = UNSUPPORTED.get(model_type, {})
    row['GFLOPs_total'] = total / 1e9; row['flops_by_operator'] = {kk: v / 1e9 for kk, v in byop.items()}
    if model_type == 'ours':
        with torch.no_grad():
            x4 = core.to4d(x1); mask = m.attention_mask_model(x4).squeeze(1); inter = m.attention_mask_model.intermediate
            pts, _ = core.select_points(x1, mask, k); enc = m.point_encoder(pts, m.image_size); cls, _ = m.fine(pts)
        row['stages_GFLOPs'] = {
            'coarse_network': flops(Coarse(m), (x4,))[0] / 1e9,
            'topk_and_gather': 0.0,
            'point_embedding': flops(Embed(m), (pts,))[0] / 1e9,
            'fine_attention': flops(FineAttn(m), (enc,))[0] / 1e9,
            'coarse_pooling_fusion_head': flops(Head(m), (inter, cls))[0] / 1e9}
        row['k'] = k
    # CPU latency
    torch.set_num_threads(os.cpu_count())
    row['cpu_latency_ms'] = [1e3 * v for v in time_it(lambda: net(x1), 'cpu', 10, 50, 5)]
    # GPU
    netg = net.cuda()
    for bs, key in [(1, 'gpu_latency_ms'), (32, 'gpu_bs32')]:
        xg = make_input(model_type, ds, bs, 'cuda')
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        mean, sd = time_it(lambda: netg(xg), 'cuda', 50, 200, 5)
        if bs == 1:
            row[key] = [1e3 * mean, 1e3 * sd]
        else:
            row['gpu_throughput_img_s'] = [32 / mean, 32 * sd / mean ** 2]
            row['gpu_peak_mem_MB_bs32'] = torch.cuda.max_memory_allocated() / 2**20
    del netg; torch.cuda.empty_cache()
    if model_type == 'ours':  # time of each GPU stage at batch 32 (top-k + gather included explicitly)
        mg = m.cuda(); xg = make_input('ours', ds, 32, 'cuda')
        with torch.no_grad():
            x4 = core.to4d(xg); mask = mg.attention_mask_model(x4).squeeze(1); inter = mg.attention_mask_model.intermediate
            pts, _ = core.select_points(xg, mask, k); enc = mg.point_encoder(pts, mg.image_size); cls, _ = mg.fine(pts)
        st = {'coarse_network': lambda: mg.attention_mask_model(x4),
              'topk_and_gather': lambda: core.select_points(xg, mask, k),
              'point_embedding': lambda: mg.point_encoder(pts, mg.image_size),
              'fine_attention': lambda: FineAttn(mg)(enc),
              'coarse_pooling_fusion_head': lambda: Head(mg)(inter, cls)}
        row['gpu_stage_ms_bs32'] = {n: 1e3 * time_it(f, 'cuda', 50, 200, 5)[0] for n, f in st.items()}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='holos'); ap.add_argument('--k', type=int)
    ap.add_argument('--models', nargs='+', default=['ours', 'resnet50', 'mobilenet', 'ConvNeXt', 'DeiT', 'ViTSmall', 'SwinTiny'])
    a = ap.parse_args()
    if a.k is None:
        a.k = int(round(json.load(open(os.path.join(core.ROOT, 'results', a.dataset, 'ours', 'summary.json')))['k']['mean']))
    import timm.layers
    timm.layers.set_fused_attn(False)
    env = {'timm_fused_attn': False, 'note': 'run on an otherwise idle machine',
           'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cpu': platform.processor() or platform.machine(),
           'cpu_threads': os.cpu_count(), 'precision': 'FP32', 'cudnn_benchmark': torch.backends.cudnn.benchmark}
    rows = []
    for mt in a.models:
        r = bench(mt, a.dataset, a.k, None); rows.append(r)
        print(json.dumps({kk: v for kk, v in r.items() if kk != 'flops_by_operator'}), flush=True)
    od = os.path.join(core.ROOT, 'results', 'benchmark'); os.makedirs(od, exist_ok=True)
    json.dump({'env': env, 'dataset': a.dataset, 'rows': rows, 'fvcore_unsupported_ops': UNSUPPORTED}, open(os.path.join(od, f'{a.dataset}_k{a.k}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
