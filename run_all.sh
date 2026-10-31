#!/bin/bash
# Reproduces every experiment of the paper, one GPU job at a time.
# Resumable: a job whose results/<dataset>/<tag>/summary.json exists is skipped, so the script can simply be
# started again after any interruption. Datasets that are not present under data/ are skipped.
# Usage: bash run_all.sh            (all datasets)
#        bash run_all.sh wbc        (one dataset, e.g. the public WBC dataset only)
set -u
cd "$(dirname "$0")"
mkdir -p logs
DATASETS=${*:-event holos opd wbc}

job() {  # job <dataset> <tag> <run.py arguments...>
  ds=$1; tag=$2; shift 2
  if [ -f results/$ds/$tag/summary.json ]; then echo "skip $ds/$tag (done)"; return; fi
  echo "[$(date '+%m-%d %H:%M')] start $ds/$tag"
  python3 run.py --dataset $ds --tag $tag --fast "$@" > logs/${ds}_${tag}.log 2>&1 || { echo "FAILED $ds/$tag (see logs/${ds}_${tag}.log)"; return; }
  python3 -c "import json;s=json.load(open('results/$ds/$tag/summary.json'));a=s['accuracy'];print('   done $ds/$tag accuracy %.2f +- %.2f'%(a['mean']*100,a['std']*100))"
}

have() {  # is the dataset on disk?
  python3 -c "import core,os,sys;sys.exit(0 if all(os.path.isdir(os.path.join(core.DATA,d)) for d in core.CLASS_DIRS['$1']) else 1)"
}

for ds in $DATASETS; do
  if ! have $ds; then echo "dataset $ds not found under data/, skipped"; continue; fi
  # main comparison: SparseAttnNet and all baselines trained from scratch with the same protocol
  job $ds ours --model ours
  for m in resnet50 mobilenet ConvNeXt DeiT ViTSmall SwinTiny; do job $ds $m --model $m; done
  if [ $ds = event ]; then
    # ImageNet-pretrained transformers (weights: python fetch_weights.py)
    python3 fetch_weights.py > logs/fetch_weights.log 2>&1 || echo "pretrained weights missing, see logs/fetch_weights.log"
    for m in ConvNeXt DeiT ViTSmall SwinTiny; do job event ${m}_pretrained --model $m --pretrained; done
    # ablations (need results/event/ours for the matched-k variants)
    job event fixed_matched --model ours --k_from ours
    for k in 1500 3000 5000; do job event fixed_k$k --model ours --fixed_k $k; done
  fi
  if [ $ds = wbc ]; then
    job wbc fixed_matched --model ours --k_from ours
    for k in 4000 8000 12000; do job wbc fixed_k$k --model ours --fixed_k $k; done
  fi
  if [ $ds = event ] || [ $ds = wbc ]; then
    job $ds full_image --model ours --full_image
    job $ds no_distill --model ours --no_distill
    job $ds no_contrast --model ours --no_contrast
    job $ds coarse_only --model ours --coarse_only
    job $ds fine_only --model ours --fine_only
    job $ds random_matched --model ours --random_select --k_from ours
  fi
done

echo "[$(date '+%m-%d %H:%M')] statistics, faithfulness, efficiency, figures"
python3 paired_stats.py > logs/paired_stats.log 2>&1 || echo "FAILED paired_stats"
DONE=$(for ds in event holos opd wbc; do [ -f results/$ds/ours/summary.json ] && echo -n "$ds "; done)
[ -n "$DONE" ] && { python3 deletion_test.py --datasets $DONE > logs/deletion_test.log 2>&1 || echo "FAILED deletion_test"; }
for ds in holos wbc; do
  [ -f results/$ds/ours/summary.json ] && { python3 benchmark.py --dataset $ds > logs/benchmark_$ds.log 2>&1 || echo "FAILED benchmark $ds"; }
done
python3 flops_by_k.py > logs/flops_by_k.log 2>&1 || echo "FAILED flops_by_k"
python3 make_figures.py > logs/make_figures.log 2>&1 || echo "FAILED make_figures (needs all four datasets)"
python3 summarize.py | tee results/summary_tables.txt
echo "ALL DONE $(date)"
