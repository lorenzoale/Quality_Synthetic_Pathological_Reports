# Quality Over Quantity: Synthetic Breast Cancer Data–Report Pairs for Clinical Information Extraction
This repository contains the code and resources used to generate, filter, fine-tune, and evaluate synthetic pathology reports from structured diagnostic variables. 

The repository supports two main experimental pipelines:

1. **Synthetic pathology report generation** from structured tabular variables.
2. **Information extraction** from generated or real pathology reports using fine-tuned language models.

## Overview

To protect patient privacy, medical models are frequently trained on synthetic datasets. 
Focusing on 13 clinical variables commonly found in breast cancer records, we propose a synthetic data generation framework in which each generated sample is a data–report pair: the structured data capture the 13 variables of interest, while the accompanying text simulates a corresponding breast cancer patient record.  Data privacy and coverage are obtained by generating the structured data using a tabular data synthesiser. We focus on enhancing the diversity and quality of the generated texts through a combination  of  data-driven knowledge distillation, data filtering and data combination. Our findings suggest that the quality of the training data used for the student model outweighs its quantity; and that a compact model (Mistral 7B) can achieve Information Extraction performance comparable to that of a significantly larger model (MedGemma 27B) when fine-tuned on high-quality synthetic data. 

## Repository structure:

- TabularDataGeneration: generation of synthetic structured data
- TextGeneration: generation of synthetic pathology reports from structured variable--value pairs;
  - NORAG: prompt-based generation using the base large language models;
  - RAG: retrieval-augmented generation with pre-instantiated pathology templates;
  - DDKD: prompt-based generation using the fine tuned models;
- FineTuneGeneration: data-driven knowledge distillation; supervised fine-tuning and LoRA fine-tuning;
- InformationExtraction: information extraction from pathology reports; evaluation of extracted variables against structured reference labels. Inference for MedGemma27B and Mistral7B (0-shot, 3-shot and sft models). Supervised fine-tuning and LoRA fine-tuning of both models. Evaluation.
- Evaluation: evaluation of generated text (for ranking, filtering and combination of generated reports). Faithfulness evaluation with LLM and lexical quality evaluation. 



