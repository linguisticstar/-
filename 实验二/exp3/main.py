import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import SentimentClassifier
import torch.nn as nn
import os

# 新增库（可视化 & 指标）
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# ============================================================
# 设置 HuggingFace 镜像
# ============================================================
def set_hf_mirrors():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = './hf_cache'

set_hf_mirrors()


# ============================================================
# 🔍 评估函数（新增：F1、AUC、Confusion Matrix）
# ============================================================
def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]  # 正类概率
            _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ===== 指标计算 =====
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    avg_loss = total_loss / len(eval_loader)
    return avg_loss, acc, f1, auc, cm


# ============================================================
# 📈 可视化: Loss / Accuracy / F1 / AUC
# ============================================================
def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # ---- Loss 曲线 ----
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.ylim(bottom=0)  # y轴从0开始
    plt.legend()
    plt.show()

    # ---- Accuracy ----
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.ylim(bottom=0)  # y轴从0开始
    plt.legend()
    plt.show()

    # ---- F1-score ----
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["val_f1"], label="Val F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.title("F1-score Curve")
    plt.ylim(bottom=0)  # y轴从0开始
    plt.legend()
    plt.show()

    # ---- AUC ----
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC Curve")
    plt.ylim(bottom=0)  # y轴从0开始
    plt.legend()
    plt.show()


    # ---- F1-score ----
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["val_f1"], label="Val F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1-score Curve")
    plt.legend()
    plt.show()

    # ---- AUC ----
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC Curve")
    plt.legend()
    plt.show()


# ============================================================
# 📊 混淆矩阵可视化
# ============================================================
def plot_confusion_matrix(cm):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# ============================================================
# 🔥 训练函数（新增记录指标 history）
# ============================================================
def train(train_texts, train_labels, val_texts=None, val_labels=None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.to(device)

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_loader = None
    if val_texts:
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.max_seq_length)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # ===== 新增：记录训练过程 =====
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": []
    }

    best_acc = 0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")

        if val_loader:
            val_loss, acc, f1, auc, cm = evaluate(model, val_loader, device)

            print(f"Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {acc:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
            print("Confusion Matrix:")
            print(cm)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(acc)
            history["val_f1"].append(f1)
            history["val_auc"].append(auc)

            if acc > best_acc:
                best_acc = acc
                model.save_model(config.model_save_path)
                print("✔ Best model updated.")

    return model, history, tokenizer, device


# ============================================================
# 🚀 主程序入口
# ============================================================
if __name__ == "__main__":
    config = Config()
    loader = DataLoaderClass(config)

    train_texts, train_labels = loader.load_csv("dataset/train_2000.csv")
    val_texts, val_labels = loader.load_csv("dataset/dev.csv")
    test_texts, test_labels = loader.load_csv("dataset/test.csv")

    # ===== 训练 =====
    model, history, tokenizer, device = train(train_texts, train_labels, val_texts, val_labels)

    # ===== 可视化 =====
    plot_training_curves(history)

    # ===== 测试集评估 =====
    print("\n========== Testing ==========")
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, config.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    model = SentimentClassifier(config.model_name, config.num_classes)
    model.load_state_dict(torch.load(config.model_save_path))
    model.to(device)

    test_loss, acc, f1, auc, cm = evaluate(model, test_loader, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print("Test Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm)
