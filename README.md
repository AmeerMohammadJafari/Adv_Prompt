# Automated Gradient-Based Attack on LLMs

This repository contains the implementation and experimental setup for my B.Sc. thesis project, titled **"Automated Gradient-Based Attack on Large Language Models (LLMs)"**. The project explores how to uncover unsafe behavior in LLMs through prompt optimization using gradient-based methods.

## 🧠 Project Overview

Large Language Models are powerful tools, but they can be vulnerable to adversarial inputs that lead to unsafe or harmful outputs. This project implements a **white-box adversarial attack** that iteratively modifies prompts using gradient feedback from a differentiable classifier.

Inspired by the [GBRT paper](https://arxiv.org/pdf/2401.16656), the approach treats prompts as distributions over tokens, optimized through backpropagation to generate high-risk outputs as determined by a safety classifier.

---

## 🖼️ System Overview

### 🔹 General Flow

The image below shows the overall structure of the attack pipeline:

![image](https://github.com/user-attachments/assets/6f838e13-15dc-4dd8-b374-94748009a959)


- A soft prompt is used to generate a response from the LLM.
- The response is then passed to a safety classifier.
- The classifier returns a safety score indicating how harmful the output is.
- This score is used to compute a loss, which is backpropagated to update the prompt distribution.

---

### 🔸 Iterative Decoding Process

The following diagram details the inner loop of the decoding mechanism used to generate the response from the LLM:

![image](https://github.com/user-attachments/assets/68226590-402f-4e51-a3fc-add422713bf2)


- Starting from a soft prompt, a soft embedding is computed as a weighted average of token embeddings.
- This embedding is passed through the LLM to get output logits.
- The logits are passed through a Gumbel Softmax to obtain a new token distribution.
- This distribution is again used to compute a new embedding (weighted average).
- The new embedding is concatenated to the sequence of embeddings, and the process repeats until the full response is generated.
- Once the final output is complete, it is scored by the classifier.

---

## 📌 Objectives

- Develop a framework for white-box gradient-based prompt attacks on LLMs.
- Automate the generation of adversarial prompts that trigger unsafe model behavior.
- Experiment with different configurations to study the effectiveness and limitations of the method.

---

## 🛠️ Methodology Summary

1. **Prompt Distribution**: Initialize a learnable distribution over input tokens.
2. **Differentiable Decoding**: Use Gumbel Softmax + soft embeddings in a loop to build responses.
3. **Safety Classification**: Evaluate generated responses using a pretrained classifier.
4. **Backpropagation**: Use classifier loss to update the prompt distribution through gradients.

---
