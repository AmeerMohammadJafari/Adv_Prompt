
from transformers import AutoTokenizer, GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
import os
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers import get_scheduler
from torch.amp import autocast, GradScaler  
from tqdm.notebook import tqdm  # Progress bar for Jupyter
from IPython.display import display, clear_output
import matplotlib.pyplot as plt




class ResponseDataset(Dataset):
    def __init__(self, df, tokenizer, text_col="formatted_text", label_col="response_label"):
        self.texts = df["formatted_text"].tolist()
        self.labels = df["response_label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx], 
            padding="max_length",
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0).long()
        attention_mask = encoded["attention_mask"].squeeze(0).long()
        label = torch.tensor(self.labels[idx], dtype=torch.float)  # Binary labels
        
        return input_ids, attention_mask, label

class GPT2ForClassification(nn.Module):
    def __init__(self, gpt_model, hidden_dim=768, dropout_prob=0.2):
        super().__init__()
        self.gpt = gpt_model
        # Freeze GPT-2 parameters
        for param in self.gpt.parameters():
            param.requires_grad = False
            
        for param in self.gpt.transformer.h[-3:].parameters():  
            param.requires_grad = True  # Unfreeze last 3 transformer blocks
        
        self.dropout = nn.Dropout(dropout_prob)  
        
        self.classifier = nn.Sequential(
            nn.Linear(gpt_model.config.n_embd, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt.transformer(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        cls_representation = last_hidden_state[:, -1, :]

        cls_representation = self.dropout(cls_representation)
        logits = self.classifier(cls_representation)

        return logits
    

def train_classifier(model, model_tokenizer, df, text_col, label_col, batch_size=2, num_epochs=6, gradient_accumulation_steps=4):
    train_dataset = ResponseDataset(df, model_tokenizer, text_col=text_col, label_col=label_col)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Learning rate scheduler
    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    scaler = GradScaler(device if device == "cuda" else None)

    # For plotting loss
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)  # Progress bar
        
        for step, (input_ids, attention_mask, labels) in enumerate(progress_bar):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.float().to(device)

            optimizer.zero_grad()

            with autocast(device if device == "cuda" else None):
                logits = model(input_ids, attention_mask).squeeze(-1)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())  # Show loss dynamically

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Clear previous output and display new loss plot
        clear_output(wait=True)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linestyle="-", label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid()
        plt.show()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print("Training complete! ✅")



def get_tokenizer_and_model(name='openai-community/gpt2-large'):
    gpt_tokenizer = AutoTokenizer.from_pretrained(name)
    gpt = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large")
    return gpt_tokenizer, gpt


def download_dataset(base_path, splits, directory_name):

    os.makedirs(f"dataset/{directory_name}", exist_ok=True)


    for split_name, filename in splits.items():
        json_path = f"dataset/{directory_name}/{split_name}.json"
        if os.path.exists(json_path):
            print(f"✅ {split_name} dataset already exists. Skipping download...")
            continue
        print(f"Downloading and saving {split_name} dataset...")

        df = pd.read_json(base_path + filename)

        df.to_json(f"{json_path}", orient="records", lines=True)

    print("✅ All datasets saved successfully in 'dataset' directory!")

        
        





