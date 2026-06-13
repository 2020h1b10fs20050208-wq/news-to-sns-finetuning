# 뉴스 문체 → SNS 문체 변환 LLM 파인튜닝

> 자연어정보분석 기말 프로젝트 — Unsloth LoRA Fine-tuning

---

## 프로젝트 소개

한국어 뉴스 기사 문체를 SNS(소셜미디어) 문체로 자동 변환하는 LLM을 파인튜닝한 프로젝트입니다.

- **베이스 모델**: Llama-3.2-3B-Instruct
- **파인튜닝 방법**: QLoRA (4-bit quantization + LoRA)
- **학습 프레임워크**: Unsloth + HuggingFace TRL
- **태스크**: 문체 변환 (격식체 뉴스 → 구어체 SNS)

### 변환 예시

| 뉴스 문체 | SNS 문체 |
|-----------|----------|
| 정부는 오늘 새로운 경제 활성화 정책을 발표하였습니다. 내년부터 중소기업 세금 감면 혜택이 확대될 예정입니다. | 정부가 경제 살리기 정책 발표했대요!! 내년부터 중소기업 세금 줄어든다고 😮 #경제뉴스 #중소기업 |
| 한국은행은 기준금리를 현 수준에서 동결하기로 결정하였습니다. | 한국은행 기준금리 또 동결이래요 🏦 #금리 #경제 |
| 독감 환자 수가 전주 대비 30% 증가하며 주의가 요구됩니다. | 독감 환자 30% 급증 😷🤒 손 씻기 철저히 하세요!! #독감 #건강 |

---

## 사용 기술 스택

| 항목 | 내용 |
|------|------|
| 베이스 모델 | `unsloth/Llama-3.2-3B-Instruct` |
| 파인튜닝 방법 | QLoRA (4-bit) |
| 프레임워크 | Unsloth, HuggingFace TRL, PEFT |
| 학습 환경 | Google Colab (Tesla T4 GPU, 14.5GB VRAM) |
| 데이터 출처 | 직접 구성 (뉴스-SNS 변환 페어 20건) |
| Export 형식 | GGUF (Q8_0) |

---

## 파인튜닝 설계

### 모델 선택 근거
Llama-3.2-3B-Instruct를 선택한 이유:
- 무료 Colab(T4 GPU, 15GB VRAM) 환경에서 4-bit 양자화로 실행 가능
- 3B 규모로 문체 변환 태스크에 충분한 성능
- Unsloth 공식 지원 모델로 학습 속도 2배 최적화

### 데이터셋 설계
- **형식**: ShareGPT (system / user / assistant)
- **구성**: 뉴스 문장(user) → SNS 변환문(assistant) 페어
- **규모**: 20건 (경제, 사회, 스포츠, 환경, 건강 등 다양한 주제)
- **품질 기준**: 이모지 1~3개, 해시태그 1~2개, 100자 이내

```json
{  "conversations": [
    {"role": "system", "content": "뉴스 문체를 SNS 문체로 변환해줘."},
    {"role": "user", "content": "정부는 오늘 새로운 경제 활성화 정책을 발표하였습니다."},
    {"role": "assistant", "content": "정부가 경제 살리기 정책 발표했대요!! 😮 #경제뉴스"}
  ]
}```

### 하이퍼파라미터 설정 근거

| 파라미터 | 값 | 근거 |
|----------|-----|------|
| rank (r) | 16 | 문체 변환은 중간 복잡도의 태스크 (Hu et al., 2021) |
| lora_alpha | 32 | r×2 설정이 표준적 (alpha/r = 2배 스케일링) |
| learning_rate | 2e-4 | Unsloth 권장값, AdamW 8-bit 옵티마이저와 조합 |
| epochs | 3 | 소규모 데이터셋 과적합 방지 |
| target_modules | q,k,v,o,gate,up,down proj | Attention + FFN 레이어 전체 적용 |
| load_in_4bit | True | VRAM 절약, T4 환경 최적화 |

---

## 학습 과정

### Training Loss 곡선

| Step | Training Loss |
|------|--------------|
| 1 | 3.0045 |
| 2 | 2.9301 |
| 3 | 2.9910 |
| 4 | 2.4882 |
| 5 | 2.3734 |
| 6 | 2.0670 |
| 7 | 1.7674 |
| 8 | 1.4885 |
| 9 | 1.6673 |

**판단**: Loss가 3.0 → 1.5로 전반적으로 감소하여 정상적인 학습이 이루어졌음을 확인. 데이터 20건으로 9 step 학습 완료 (40초 소요).

---

## 베이스 vs 파인튜닝 비교 결과

**입력**: `정부는 오늘 새로운 경제 활성화 정책을 발표하였습니다. 내년부터 중소기업 세금 감면 혜택이 확대될 예정입니다.`

| 항목 | 베이스 모델 | 파인튜닝 모델 |
|------|------------|--------------|
| 문체 | 설명적·격식체 유지 | SNS 구어체로 변환 |
| 이모지 | 없음 | 포함 |
| 해시태그 | 없음 | 포함 |
| 길이 | 길고 상세 | 짧고 간결 |

---

## 한계점 및 개선 방향

- **데이터 부족**: 20건으로 일반화 성능이 제한적 → AI Hub 한국어 SNS 멀티턴 대화 데이터(196,237 세션) 활용 필요
- **모델 크기**: 3B 모델의 한국어 이해도 제한 → EXAONE, HyperCLOVA 등 한국어 특화 모델 시도
- **평가 지표 부재**: BLEU, ROUGE 등 정량 평가 미적용 → 향후 자동 평가 파이프라인 구축 필요

---

## 실행 방법

### Google Colab에서 실행
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ovGMcNdydHAKQrIia2ITa1hd_jjng7db)

### 로컬에서 GGUF 실행 (Ollama)
```bash
# Ollama 설치 후
ollama create news-to-sns -f llama_finetune_gguf/Modelfile
ollama run news-to-sns "정부는 오늘 새로운 경제 정책을 발표하였습니다."
```

---

## 참고문헌

- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. https://arxiv.org/abs/2305.14314
- Zhou et al. (2023). LIMA: Less Is More for Alignment. https://arxiv.org/abs/2305.11206
- Unsloth Fine-tuning Guide: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
