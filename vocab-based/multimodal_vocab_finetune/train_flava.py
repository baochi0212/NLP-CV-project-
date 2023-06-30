
with open("./vocabs/answers_textvqa_8k.txt") as f:
  vocab = f.readlines()

answer_to_idx = {}
for idx, entry in enumerate(vocab):
  answer_to_idx[entry.strip("\n")] = idx
print(len(vocab))
print(vocab[:5])

from datasets import load_dataset
dataset = load_dataset("textvqa")
import torch
from torchvision import transforms
from collections import defaultdict
from transformers import BertTokenizer
from functools import partial

def transform(tokenizer, input):
  batch = {}
  image_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224])])
  image = image_transform(input["image"][0].convert("RGB"))
  batch["image"] = [image]

  tokenized=tokenizer(input["question"],return_tensors='pt',padding="max_length",max_length=512)
  batch.update(tokenized)


  ans_to_count = defaultdict(int)
  for ans in input["answers"][0]:
    ans_to_count[ans] += 1
  max_value = max(ans_to_count, key=ans_to_count.get)
  ans_idx = answer_to_idx.get(max_value,0)
  batch["answers"] = torch.as_tensor([ans_idx])
  return batch

tokenizer=BertTokenizer.from_pretrained("bert-base-uncased",padding="max_length",max_length=512)
transform=partial(transform,tokenizer)
dataset.set_transform(transform)
# from torchmultimodal.models.flava.model import flava_model_for_classification
# model = flava_model_for_classification(num_classes=len(vocab)).cuda()
with open("./vocabs/answers_textvqa_8k.txt") as f:
  vocab = [answer.strip() for answer in f.readlines()]
print("UNK index", vocab[0])

from tqdm.auto import tqdm
def test_model(loader):
  pred = []
  label = []
  for batch in tqdm(loader):
    out = model(text = batch["input_ids"].cuda(), image = batch["image"].cuda(), labels = batch["answers"].cuda()).logits
    idxs = torch.argmax(out, -1)
    pred += [vocab[idx.item()] for idx in idxs]
    label += [vocab[batch['answers'][i].item()] for i in range(batch['input_ids'].shape[0])]
  print("Val Acc", len([i for i in range(len(pred)) if pred[i] == label[i] if label[i] != '<unk>'])/len(pred))
  return pred, label


from torch import nn
BATCH_SIZE = 64
from torch.utils.data import DataLoader, Subset, Dataset
class TextVQA(Dataset):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __getitem__(self, index: int):
        return self.inner[index]

    def __len__(self):
        return len(self.inner)

    

print("check len", len(TextVQA(dataset["train"])), len(TextVQA(dataset["test"])))
# train_dataloader = DataLoader(TextVQA(dataset["train"]), batch_size=BATCH_SIZE)
# val_dataloader = DataLoader(TextVQA(dataset["validation"]), batch_size=BATCH_SIZE)
# for item in dataset['validation']:
#    print(item['answers'])
#    break
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


# epochs = 30
# for _ in range(epochs):
#   for idx, batch in tqdm(enumerate(train_dataloader)):
#     optimizer.zero_grad()
#     out = model(text = batch["input_ids"].cuda(), image = batch["image"].cuda(), labels = batch["answers"].cuda())
#     loss = out.loss
#     loss.backward()
#     optimizer.step()
#     if idx % 1000 == 0:
#         model.eval()
#         print(f"Loss at step {idx} = {loss}")
#         test_model(val_dataloader)
#         model.train()