# LAVIS-VQA

- Installation:
```
conda create -n vqa python=3.9 -y
conda activate vqa
cd LAVIS-VQA
pip install -e .
```
- Download TextVQA images:
```
python lavis/datasets/download_scripts/download_textvqa.py
```

## TODO: 
- Finetune BLIP on TextVQA:
```
bash run_scripts/blip/train/train_textvqa.sh
```
- Evaluate PnP-VQA-3b on TextVQA:
```
bash run_scripts/pnp-vqa/eval/eval_textvqa_3b.sh
``` 
- Evaluate BLIP2 (FLANT5XL or OPT backbone) on TextVQA:
```
bash run_scripts/blip2/eval/eval_textvqa_zeroshot_flant5xl.sh
bash run_scripts/blip2/eval/eval_textvqa_zeroshot_opt.sh
```
- Evaluate Img2LLm on TextVQA:
```
bash run_scripts/img2llm/eval_textvqa.sh
```
