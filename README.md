# Visual Question Answering
- Installation:
```
conda create -n vqa python=3.9 -y
conda activate vqa
cd LAVIS-VQA
pip install -e .
npm install localtunnel
```
- For vocab-based models: Go to [vocab-based](https://github.com/baochi0212/visual-qa-/tree/master/vocab-based) and follow the instructions.
- For generation-based models: Go to [LAVIS-VQA](https://github.com/baochi0212/visual-qa-/tree/master/LAVIS-VQA) and follow the instructions.
- For demo, run the following command:
```
cd LAVIS-VQA/app
streamlit run main.py & npx localtunnel --port 8501
```
