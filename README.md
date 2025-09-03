# Joseph McInerney MSc AI Project

## Part 1: Pretraining
- Preprocessing: containment.py, 
- Training Script
    - Dependencies: Requirements.txt
    - Bash Script: run_cpt.sh
    - Continued Pre-training Script: cpt.py
    - Distributed Training: deepspeed_config.json

## Part 2: Instruction Tuning
- API Calls to Generate LLM Instruct-Response pairs: generate_LLM_instruct.py
- App for Human Annotation: human_annotation.py
- LLM Automated Annotation: LLM Annotation
- Ranking and Inter-Annotator Analysis: annotation_analysis.py
- Dataset preparationg for Google Vertext AI Translation of Dolly V2: dolly_gemini_trans  
- Statistical Analysis of Open-Ended Response Lengths: response_length_comparison.py
- Dataset: HF_REPO

## Part 2: Human Feedback Daat
- LIMA Translation and Preference Dataset Creation: create_hf_LIMA_ga.py
- App for Native Annotation: annotate_preferences.py
- Dataset: HF_REPO
