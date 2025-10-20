# Qomhrá: A Bilingual Irish-English Large Language Model
# Joseph McInerney, 2025

<img width="631" height="471" alt="Clean_Qomhrá drawio (2)" src="https://github.com/user-attachments/assets/9081fc0e-67b8-4d62-8bcd-50f40f1a6217" />

## Part 1: Pretraining
- Preprocessing: containment, CNG, , UCCIX, Bitext 
- Training Script
    - Dependencies
    - Bash Script
    - Continued Pre-training Script
    - Distributed Training Config

## Part 2: Instruction Tuning
- API Calls to Generate LLM Instruct-Response pairs
- App for Human Annotation
- LLM Automated Annotation
- Ranking and Inter-Annotator Analysis
    - Bradley Terry Ranking and Cohen's Kappa Script
    - Dependencies
- Dataset preparationg for Google Vertext AI Translation of Dolly V2
- LoRA
    - Training script
    - Dependencies
- Statistical Analysis of Open-Ended Response Lengths
- Dataset: Request Access

## Part 2: Human Feedback Data
- LIMA Translation and Preference Dataset Creation
- App for Native Annotation
- Dataset: Request Access
