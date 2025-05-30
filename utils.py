
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer




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

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        """
        Forward pass that supports both tokenized input (input_ids) and raw embeddings (inputs_embeds).
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either `input_ids` or `inputs_embeds`, but not both.")
        
        if input_ids is not None:
            outputs = self.gpt.transformer(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.gpt.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state

        # Extract CLS representation (last token's hidden state)
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
    gpt = GPT2LMHeadModel.from_pretrained(name)
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

        

class LLMProbabilisticPipeline(nn.Module):
    def __init__(self, model_name, prompt_length, num_tokens, classifier, learning_rate=1e-2, lambda_reg=1e-2, lambda_lm = 1e-2,
):
        super().__init__()
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        self.tokenizer, self.model = get_tokenizer_and_model(model_name)
        self.model.to(self.device)  # Move model to device
        self.classifier = classifier  
        self.prompt_length = prompt_length
        self.num_tokens = num_tokens  
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding_layer = self.model.get_input_embeddings()
        self.embedding_layer.weight.requires_grad = False
        self.embedding_matrix = self.embedding_layer.weight.to(self.device)
        self.prompt_logits = nn.Parameter(torch.ones(prompt_length, self.vocab_size, device=self.device))
        self.lambda_reg = lambda_reg
        self.lambda_lm = lambda_lm
        
        for param in self.model.parameters():
            param.requires_grad = False

        # Freeze classifier parameters
        for param in self.classifier.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam([self.prompt_logits], lr=learning_rate)

    def get_weighted_embedding(self, prob_dist):
        """Computes the weighted sum of embeddings based on probability distributions."""
        weighted_sum = torch.matmul(prob_dist, self.embedding_matrix)  # (seq_len, emb_dim)
        # Ensure the output shape remains (batch, seq_len, emb_dim)
        if len(weighted_sum.shape) == 2:  
            weighted_sum = weighted_sum.unsqueeze(0)  # Add batch dimension if missing

        return weighted_sum.to(self.device)  # Ensure tensor is on the correct device

    def generate_response(self, prompt_prob_dist):
        """Generates a response step by step while applying Gumbel-Softmax to logits at each step.
        Concatenates the token probabilities to the current sequence."""
        
        self.model.to(self.device)  # Ensure model is on device
        prompt_prob_dist = prompt_prob_dist.to(self.device)  # Move prompt distribution to device

        # Initialize the prompt with the weighted embedding of the initial probability distribution
        weighted_prompt = self.get_weighted_embedding(prompt_prob_dist)
        attention_mask = torch.ones(weighted_prompt.shape[:2], device=self.device)  # Mask for attention
        
        # Initial input embeddings for the first token
        input_embedding = weighted_prompt
        
        for step in range(self.num_tokens):
            # Get logits for the next token
            output = self.model(inputs_embeds=input_embedding, 
                                attention_mask=attention_mask)  # No sampling, just the logits
            
            # Extract logits for the current step (this is for the next token)
            logits = output.logits[:, -1, :]  # Get logits for the last token (current step)

            # Apply Gumbel Softmax to get the next token probabilities
            token_probs = F.gumbel_softmax(logits, tau=1.0, dim=-1)

            # Update the current sequence by appending the next token's probability distribution
            next_token_prob_dist = token_probs.unsqueeze(0)  # Add batch dimension

            # Update input embedding with the newly generated token using get_weighted_embedding
            input_embedding = torch.cat(
                (input_embedding, self.get_weighted_embedding(next_token_prob_dist)), dim=1
            )  # Concatenate the new token's weighted embedding to the sequence

            # Update the attention mask
            attention_mask = torch.cat((attention_mask, torch.ones(1, 1, device=self.device)), dim=1)  # Append 1 to attention mask

        return input_embedding[:, self.prompt_length:, :]
    
    
    def compute_lm_alignment_loss(self):
        """
        Computes -σ(x) · log(P_LM(x)) over the prompt distribution.
        Encourages soft prompts to look like natural language.
        """
        prompt_prob_dist = self.prompt_prob_dist.detach()  # Shape: (prompt_len, vocab_size)

        # Get the weighted embeddings for soft prompts
        prompt_embeds = self.get_weighted_embedding(prompt_prob_dist)

        attention_mask = torch.ones(prompt_embeds.shape[:2], device=self.device)  # (1, prompt_len)

        # Get model predictions for each prompt token
        with torch.no_grad():
            outputs = self.model(inputs_embeds=prompt_embeds, attention_mask=attention_mask)
            lm_logits = outputs.logits[:, :-1, :]  # Logits for predicting tokens 1..n-1
            log_probs = torch.log_softmax(lm_logits, dim=-1)  # Convert to log-probs

        # Shift prompt_prob_dist to align with targets (tokens at t=1 to t=n-1)
        prob_dist_shifted = prompt_prob_dist[1:]  # (prompt_len - 1, vocab_size)

        # Align dimensions for batch processing
        log_probs = log_probs[0]  # Remove batch dimension: (prompt_len - 1, vocab_size)

        # Compute cross-entropy between soft prompt distribution and LM log probs
        ce_loss = -torch.sum(prob_dist_shifted * log_probs)

        return ce_loss

    def forward(self):
        """Computes safety score based on generated responses."""
        self.prompt_prob_dist = F.gumbel_softmax(self.prompt_logits, tau=1.0, dim=-1)
        embeddings_matrix = self.generate_response(self.prompt_prob_dist)
        
        # Generate the corresponding attention mask
        attention_mask = torch.ones(embeddings_matrix.shape[:2], device=self.device)  # (1, num_tokens)
        
        safety_logit = self.classifier(attention_mask=attention_mask, inputs_embeds=embeddings_matrix).squeeze(-1)  # (1, num_tokens, 1)

        # get safety score for backpropagation from a single logit using sigmoid
        safety_score = torch.sigmoid(safety_logit)
        
        return safety_score

    def train_step(self):
        """Performs a training step to optimize the prompt distribution."""
        self.classifier.eval()  # Important: make sure classifier is deterministic
        self.optimizer.zero_grad()
        
        # Compute safety score
        safety_score = self.forward()
        
        # Compute the uniform distribution for the regularizer
        uniform_dist = torch.ones_like(self.prompt_prob_dist) / self.vocab_size  # (prompt_length, vocab_size)
        
        # Compute dot product with uniform distribution
        dot_product = torch.sum(self.prompt_prob_dist * uniform_dist)  # Scalar value representing the similarity with uniform distribution
        lm_ce_loss = self.compute_lm_alignment_loss()
        # Loss function with added regularization term
        loss = safety_score + self.lambda_reg * dot_product + self.lambda_lm * lm_ce_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    
    
    # def train_step(self):
    #     # Initial memory state
    #     print(f"Start - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
    #         f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    #     # Forward pass
    #     self.optimizer.zero_grad()
    #     print("Optimizer parameters:")
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             print(name, param.shape)
    #     safety_score = self.forward()
    #     loss = safety_score
    #     print(f"After forward - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
    #         f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    #     # Backward pass
    #     with profiler.profile(record_shapes=True, use_device='cuda') as prof:
    #         loss.backward()
    #     print(f"After backward - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
    #         f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    #     print("Backward pass profile (top 10 by CUDA memory):")
    #     print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    #     # Check gradients
    #     grad_size = (self.prompt_prob_dist.grad.element_size() * self.prompt_prob_dist.grad.nelement() / 1024**2
    #                 if self.prompt_prob_dist.grad is not None else 0)
    #     print(f"prompt_prob_dist gradient size: {grad_size:.2f} MB")

    #     # Optimizer step and explicit gradient clearing
    #     self.optimizer.step()
    #     self.prompt_prob_dist.grad = None  # Force clear gradients
    #     print(f"After optimizer.step - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
    #         f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    #     print(f"Post-step prompt_prob_dist gradient: {self.prompt_prob_dist.grad}")

    #     # Delete tensors and check references
    #     loss_value = loss.item()
    #     del safety_score, loss
    #     print(f"After del - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
    #         f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    #     # Clear cache and collect garbage
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     print(f"After empty_cache - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
    #         f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    #     # Check lingering tensors
    #     tensors = [obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda and obj.requires_grad]
    #     if tensors:
    #         print("Lingering CUDA tensors with requires_grad=True:")
    #         for t in tensors[:5]:
    #             print(f" - Size: {t.element_size() * t.nelement() / 1024**2:.2f} MB, Shape: {t.shape}")
    #             # Optional: Check if tied to prompt_prob_dist
    #             if t is self.prompt_prob_dist:
    #                 print("   - This is prompt_prob_dist!")
    #     else:
    #         print("No lingering CUDA tensors with requires_grad=True found.")

    #     return loss_value





        





