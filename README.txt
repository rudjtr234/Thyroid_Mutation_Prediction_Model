# 🧬 갑상선 BRAF 변이 예측 모델 (Thyroid Mutation Prediction Model)

본 프로젝트는 **갑상선암 병리 슬라이드(WSI)** 이미지를 기반으로  
딥러닝을 활용하여 **BRAF 변이 여부를 예측**하는 인공지능 모델을 개발하기 위한 연구이자 디지털 병리 기반 정밀의료 기술 개발의 일환입니다.

---

## 🚀 프로젝트 개요
이 저장소는 **UNI2-h 비전 트랜스포머(Transformer)** 를 활용한  
WSI(Whole Slide Image) 임베딩 파이프라인과,  
**ABMIL(Attention-based Multiple Instance Learning)** 기반 BRAF 변이 예측 모델의  
전체 전처리 및 학습 코드를 포함합니다.

---

## 📊 데이터 구성

| 구분 | 설명 | 수량 |
|------|------|------|
| Meta (BRAF+) | BRAF 변이 양성 병리 슬라이드 | 약 **4,000장 중 2,000장 선택** |
| Non-meta (BRAF−) | 변이 음성 슬라이드 | **862장 전수 사용** |
| 타일 크기 | 512×512 PNG 패치 | 슬라이드당 평균 약 20,000개 |
| 임베딩 벡터 | 1타일당 1536차원 | UNI2-h 기반 feature vector |


---

## ⚙️ 임베딩 파이프라인

### 🔹 모델 백본
- **UNI2-h (MahmoodLab)**  
  - 이미지 크기: 224×224  
  - Patch size: 14  
  - Embedding dimension: 1536  
  - Transformer depth: 24  
  - Head 수: 24  
  - Activation: SiLU  
  - Layer: SwiGLU packed MLP  