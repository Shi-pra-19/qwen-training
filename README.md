# Qwen-2.5-Math Training with Unsloth

This repository contains code for fine-tuning the **Qwen2.5-Math-7B** model using the **Unsloth** library for faster training and lower memory usage.

## Background
This model was developed for the [Kaggle: Mapping Charting Student Math Misunderstandings](https://www.kaggle.com/c/map-charting-student-math-misunderstandings) competition. It served as one of six models in my ensemble solution that secured me a **Bronze Medal**.

## Overview
The goal of this project is to classify student math reasoning explanations into exactly one of predefined **Misconception categories**. The solution leverages LoRA (Low-Rank Adaptation) for efficient fine-tuning on this classification task.

## Features
- **Model**: `unsloth/Qwen2.5-Math-7B-bnb-4bit`
- **Optimization**: Uses Unsloth for 2x faster training and 4-bit quantization.
- **Task**: Multi-class classification (65 target classes).
- **Frameworks**: PyTorch, Transformers, TRL (`SFTTrainer`), and PEFT.

## References
- [Kaggle Notebook Link](https://www.kaggle.com/code/mehedi457/unsloth-qwen2-5-math-1-5b-training)
- [Kaggle Notebook Link](https://www.kaggle.com/code/aleaiest/lb-0-945-qwen2-5-32b-gptq)
- [Kaggle Notebook Link](https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945)
  
  
## Training Instructions
To train the model as intended for the competition solution:
1. **Data Path**: Change the dataset path in the notebook or uncomment the specified Kaggle path (`/kaggle/input/...`).
2. **Epochs & Strategy**: The finetuning solution notebook used **2 epochs**. In the `TrainingArguments` section:
   - Uncomment `num_train_epochs=2`.
   - Set `save_strategy` to `"epoch"`.
3. **Environment**: Ensure you have a GPU. You can use Colab or Kaggle and install Unsloth:
   ```bash
   pip install unsloth
