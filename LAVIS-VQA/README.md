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
`
- Train BLIP: Go to ```LAVIS-VQA/lavis/projects/blip/train/textvqa_ft.yaml``` and change ```inference_method``` to ```generate``` for generation or ```rank``` for classification (for inferecne only not training), then run the following command:
```
bash run_scripts/blip/train/train_textvqa.sh
```
