# python -m torch.distributed.run --nproc_per_node=16 train.py --cfg-path lavis/projects/blip/vqav2_ft.yaml
python train.py --cfg-path lavis/projects/blip/train/textvqa_ft.yaml
