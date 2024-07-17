from google.colab import drive
drive.mount('/content/drive')
!pip install git+https://github.com/openai/CLIP.git
import torch

# GPUが利用可能か確認
gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")

# 利用可能なGPUの名前を表示
if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")
else:
    print("GPU is not available.")

from google.colab import drive
import re
import random
import time
from statistics import mode
import shutil
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import os
import clip
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# Section 1: Helper Functions
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    text = text.lower()
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    text = re.sub(r'(a|an|the)', '', text)
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)
    text = re.sub(r"[^\w\s':]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename):
    return torch.load(filename)

class VQADataset(Dataset):
    def __init__(self, df_path, image_dir, preprocess, model, tokenizer, bert_model, device, transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer
        self.preprocess = preprocess
        self.model = model
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device

        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}

        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = process_text(answer["answer"])
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image_path = f"{self.image_dir}/{self.df['image'][idx]}"
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_feature = self.model.encode_image(image).squeeze(0).float()

        question = process_text(self.df["question"][idx])
        inputs = self.tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=20)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            question_feature = self.bert_model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)
            return image_feature, question_feature, torch.Tensor(answers), torch.tensor(int(mode_answer_idx))
        else:
            return image_feature, question_feature

    def __len__(self):
        return len(self.df)

# Section 3: Model Definition
class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        self.text_encoder = nn.Linear(768, 512)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image_feature, question):
        question_feature = self.text_encoder(question)
        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)
        return x

# Section 4: Training and Evaluation Functions
def VQA_criterion(pred, answers):
    batch_size = pred.size(0)
    acc = 0
    for i in range(batch_size):
        acc += (pred[i].item() in answers[i].cpu().numpy())
    return acc / batch_size

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer =             image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer =             image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.long())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

#【固定】
import os
import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def copy_file_with_progress(src, dst, buffer_size=16*1024*1024):
    total_size = os.path.getsize(src)
    copied_size = 0

    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        while True:
            buf = fsrc.read(buffer_size)
            if not buf:
                break
            fdst.write(buf)
            copied_size += len(buf)
            if copied_size % (total_size // 100) < buffer_size:
                print(f'Copying {os.path.basename(src)}: {copied_size / total_size * 100:.2f}% complete', end='')

    print(f'
{os.path.basename(src)} copied successfully.')

def copy_directory_with_progress(src_dir, dst_dir, buffer_size=16*1024*1024):
    total_files = sum([len(files) for r, d, files in os.walk(src_dir)])
    copied_files = 0

    os.makedirs(dst_dir, exist_ok=True)

    def copy_single_file(file_tuple):
        src_file, dst_file = file_tuple
        copy_file_with_progress(src_file, dst_file, buffer_size)
        return src_file

    files_to_copy = []
    for root, dirs, files in os.walk(src_dir):
        relative_path = os.path.relpath(root, src_dir)
        dst_root = os.path.join(dst_dir, relative_path)
        os.makedirs(dst_root, exist_ok=True)
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_root, file)
            files_to_copy.append((src_file, dst_file))

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(copy_single_file, file_tuple): file_tuple[0] for file_tuple in files_to_copy}
        for future in as_completed(future_to_file):
            copied_files += 1
            if copied_files % (total_files // 100) == 0:
                print(f'Copying directory {os.path.basename(src_dir)}: {copied_files / total_files * 100:.2f}% complete', end='')

    print(f'
{os.path.basename(src_dir)} copied successfully.')

train_json_path_drive = "/content/drive/MyDrive/Colab Notebooks/DLBasics2024_colab/最終課題/dl_lecture_competition_pub/data/train.json"
valid_json_path_drive = "/content/drive/MyDrive/Colab Notebooks/DLBasics2024_colab/最終課題/dl_lecture_competition_pub/data/valid.json"
train_image_dir_drive = "/content/drive/MyDrive/Colab Notebooks/DLBasics2024_colab/最終課題/dl_lecture_competition_pub/data/train/train"
valid_image_dir_drive = "/content/drive/MyDrive/Colab Notebooks/DLBasics2024_colab/最終課題/dl_lecture_competition_pub/data/valid"
local_dir = "/content/local_data"
os.makedirs(local_dir, exist_ok=True)

train_json_path = os.path.join(local_dir, "train.json")
valid_json_path = os.path.join(local_dir, "valid.json")
train_image_dir = os.path.join(local_dir, "train")
valid_image_dir = os.path.join(local_dir, "valid")

if not os.path.exists(train_json_path):
    print(f"Copying train.json from {train_json_path_drive} to {train_json_path}")
    copy_file_with_progress(train_json_path_drive, train_json_path)
else:
    print("train.json already exists.")

if not os.path.exists(valid_json_path):
    print(f"Copying valid.json from {valid_json_path_drive} to {valid_json_path}")
    copy_file_with_progress(valid_json_path_drive, valid_json_path)
else:
    print("valid.json already exists.")

if not os.path.exists(train_image_dir):
    print(f"Copying train images from {train_image_dir_drive} to {train_image_dir}")
    copy_directory_with_progress(train_image_dir_drive, train_image_dir)
else:
    print("Train images already exist.")

if not os.path.exists(valid_image_dir):
    print(f"Copying valid images from {valid_image_dir_drive} to {valid_image_dir}")
    copy_directory_with_progress(valid_image_dir_drive, valid_image_dir)
else:
    print("Valid images already exist.")

print(f"Train JSON exists: {os.path.exists(train_json_path)}")
print(f"Valid JSON exists: {os.path.exists(valid_json_path)}")
print(f"Train images exist: {os.path.exists(train_image_dir)}")
print(f"Valid images exist: {os.path.exists(valid_image_dir)}")

def check_copied_files(image_dir, json_path):
    df = pd.read_json(json_path)
    missing_files = []
    for image_file in df['image']:
        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            missing_files.append(image_path)
    return missing_files

missing_train_files = check_copied_files(train_image_dir, train_json_path)
missing_valid_files = check_copied_files(valid_image_dir, valid_json_path)

if missing_train_files:
    print(f"Missing train files: {missing_train_files}")
else:
    print("All train files are copied successfully.")

if missing_valid_files:
    print(f"Missing valid files: {missing_valid_files}")
else:
    print("All valid files are copied successfully.")

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor()
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = VQADataset(
        df_path=train_json_path,
        image_dir=train_image_dir,
        preprocess=preprocess,
        model=model_clip,
        tokenizer=tokenizer,
        bert_model=bert_model,
        device=device,
        transform=transform_train
    )
    valid_dataset = VQADataset(
        df_path=valid_json_path,
        image_dir=valid_image_dir,
        preprocess=preprocess,
        model=model_clip,
        tokenizer=tokenizer,
        bert_model=bert_model,
        device=device,
        transform=transform_valid,
        answer=False
    )
    valid_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)

    num_epoch = 39
    class_weights = torch.ones(len(train_dataset.answer2idx)).to(device)
    if 'unanswerable' in train_dataset.answer2idx:
        class_weights[train_dataset.answer2idx['unanswerable']] = 2.0
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    start_epoch = 38

    checkpoint_path = "checkpoint.pth.tar"
    if os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path)
        model_dict = model.state_dict()
        checkpoint_dict = checkpoint['model_state_dict']

        checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
        model_dict.update(checkpoint_dict)
        model.load_state_dict(model_dict)

        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded, resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】
"
              f"train time: {train_time:.2f} [s]
"
              f"train loss: {train_loss:.4f}
"
              f"train acc: {train_acc:.4f}
"
              f"train simple acc: {train_simple_acc:.4f}")

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, filename=checkpoint_path)

        scheduler.step()

    model.eval()
    submission = []
    for image, question in valid_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)
    np.save('/content/drive/MyDrive/Colab Notebooks/DLBasics2024_colab/最終課題/dl_lecture_competition_pub/submission.npy', submission)

if __name__ == "__main__":
    main()
