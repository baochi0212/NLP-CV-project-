# python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip/eval/textvqa_eval.yaml
python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/git/eval/textvqa_eval_large.yaml
