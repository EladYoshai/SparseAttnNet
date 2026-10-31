"""Download the ImageNet weights of the four pretrained transformer baselines WITHOUT Hugging Face.
Sources are the original authors' release files (Facebook AI file server, the Swin GitHub release, Google Cloud Storage).
Each file is checked against its SHA-256. Usage:  python fetch_weights.py   (files go to weights/)
If this machine has no internet at all, copy the four files into weights/ by any other route; they are checked the same way."""
import os, sys, hashlib, urllib.request

WEIGHTS = {  # timm model name: (file name, url, sha256)
    'deit_small_patch16_224': ('deit_small_patch16_224-cd65a155.pth',
        'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        'cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f'),
    'swin_tiny_patch4_window7_224': ('swin_tiny_patch4_window7_224.pth',
        'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
        '9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3'),
    'convnext_tiny': ('convnext_tiny_1k_224_ema.pth',
        'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth',
        '14f3164e3ea6ac32ab3f574f528ce817696c9176fad4221e0a77a905a7360595'),
    'vit_small_patch16_224': ('S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        'https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        '545815b4e770d2fa6ca4b3ccba7c16b035e474354e52d17ca197ea4efecbf4d3'),
}
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def path_for(timm_name):
    """Verified local file for a timm model, or raise."""
    fn, _, digest = WEIGHTS[timm_name]
    p = os.path.join(DIR, fn)
    if not os.path.exists(p) or sha(p) != digest:
        raise FileNotFoundError(f'{p} missing or wrong checksum; run  python fetch_weights.py')
    return p


def main():
    os.makedirs(DIR, exist_ok=True); ok = True
    for name, (fn, url, digest) in WEIGHTS.items():
        p = os.path.join(DIR, fn)
        if os.path.exists(p) and sha(p) == digest:
            print('ok (already here)', fn); continue
        try:
            print('downloading', url, flush=True)
            urllib.request.urlretrieve(url, p + '.part'); os.replace(p + '.part', p)
        except Exception as e:
            print('FAILED', name, e); ok = False; continue
        if sha(p) != digest:
            print('CHECKSUM MISMATCH', fn); os.remove(p); ok = False
        else:
            print('ok', fn)
    print('ALL WEIGHTS OK' if ok else 'SOME WEIGHTS MISSING'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
