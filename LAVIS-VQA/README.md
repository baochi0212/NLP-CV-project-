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
cd lavis/datasets/download_scripts && python download_text.py
cd ../../..
```
`
- Train BLIP: Go to ```LAVIS-VQA/lavis/projects/blip/train/textvqa_ft.yaml``` and change ```inference_method``` to ```generate``` for generation or ```rank``` for classification, then run the following command:
```
bash run_scripts/blip/train/train_textvqa.sh
```
