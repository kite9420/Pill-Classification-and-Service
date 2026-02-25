# 💊 Pill Classification — FP32 / QAT INT8 / ONNX / ONNX-INT8

<p align="center">
  <img src="https://github.com/user-attachments/assets/d9260991-3a43-4a91-8319-2c3ed83ea546" width="35%" />
  <img src="https://github.com/user-attachments/assets/ac4f323e-bbbd-4673-bcf7-676a352bc079" width="55%" />
</p>


Bounding-box 기반 약재 이미지 분류 모델을 PyTorch로 학습하고,
FP32 → QAT INT8 → ONNX → ONNX INT8까지 전체 추론 파이프라인을 구축하여
정확도, 속도, 모델 경량화를 비교한 프로젝트입니다.

---

## 📌 Overview

### 🎯 목표

경구 약제 이미지 데이터를 활용한 분류 모델 구축 및 경량화 파이프라인 구현.

약재 이미지에서 **annotation bbox 단위로 crop된 이미지**를 입력으로 사용하여
ResNet 기반 CNN 모델을 학습하고 다음 4가지 버전을 생성·비교합니다.

* PyTorch FP32
* PyTorch INT8 (QAT eager quantization)
* ONNX FP32
* ONNX INT8 (dynamic quantization)

---

### 📊 사용 데이터

* **Train / Val**
  Codeit Sprint 제공 AI-Hub 기반 경구약제 데이터 (73 classes)

* **Test**
  AI-Hub TL_1_조합 및 TL_1조합 경구약제 데이터
  (54 classes, 그중 47 classes가 Train과 공통)

* Train에 존재하지 않는 Test 클래스는
  **모델의 순수 성능 검증을 위해 EDA 단계에서 제거**

---

## 🧠 Model Architecture

### Custom ResNet (축소형 ResNet-50)

* Stem: 7×7 Conv → **3×3 Conv ×3** 구조로 변경
* Bottleneck residual blocks 사용
* **QuantStub / DeQuantStub 포함 → QAT 지원**

파일: `model.py`

---

## 📂 Dataset Pipeline

* JSON annotation 기반 **bbox crop**
* JSON 1개 = **drug_N 단일 라벨**
* raw 이미지 탐색: **basename 매칭**
* annotation 1개 = sample 1개

파일: `dataset.py`

---

## ⚙️ Full Pipeline

### 1️⃣ FP32 Training (GPU)

```
python training.py
```

Output:

```
checkpoints/epoch_XXX.pt
artifacts/class_mapping.json
```

---

### 2️⃣ INT8 생성 (QAT 준비 → Calibration → Convert)

```
python quantize_int8_from_qat.py
```

Process:

* FP32 checkpoint load
* fuse + prepare_qat
* calibration (observer statistics 수집)
* convert → eager INT8 model

Output:

```
artifacts/model_int8.pt
```

---

### 3️⃣ Export FP32 → ONNX

```
python export_onnx.py
```

Output:

```
artifacts/model_fp32.onnx
```

---

### 4️⃣ ONNX Dynamic INT8 Quantization

```
python quantize_onnx.py
```

Output:

```
artifacts/model_int8_dynamic.onnx
```

---

### 5️⃣ 4-Way Performance Comparison

```
python compare_4ways.py
```

비교 대상:

* Torch FP32
* Torch INT8 (eager QAT)
* ONNX FP32
* ONNX INT8 dynamic

평가지표:

* Accuracy
* Cross-entropy loss
* Throughput (img/s)
* Latency (ms/img)

---

### 6️⃣ Visualization (Optional)

```
python eval_and_visualize.py
```

기능:

* bbox 표시
* prediction / ground truth 텍스트 출력
* 결과 이미지 저장

---

## ▶️ Recommended Execution Order

1. `python training.py`
2. `python quantize_int8_from_qat.py`
3. `python export_onnx.py`
4. `python quantize_onnx.py`
5. `python compare_4ways.py`

---

## 🏗️ Project Structure

```
PROJECT1/
├── Backend/
├── Frontend/
└── Model/
    ├── artifacts/
    ├── checkpoints/
    ├── checkpoints_qat/
    ├── logs/
    ├── Obsolete/
    ├── raw_data/
    ├── Test_result/
    ├── compare_4ways.py
    ├── dataset.py
    ├── eval_and_visualize.py
    ├── export_onnx.py
    ├── extract_model_only.py
    ├── model.py
    ├── qat_utils.py
    ├── quantize_int8_from_qat.py
    ├── quantize_onnx.py
    ├── training.py
    └── readme.md
```

---

## 🚀 Future Plan

### Model

* Backbone CustomResNet 앞단에 **BBox 추론 모델 추가**
  → bbox 없는 이미지도 학습 및 예측 가능하도록 확장
* 모델 성능 개선 및 추가 경량화

---

### Backend (Future)

* GCS / Google Compute Engine / Cloud Run 기반 FastAPI 서빙
* 프론트 POST 요청 시 **즉시 응답 후 비동기 처리**
* 사용자 데이터 저장 및 관리
* 업로드된 Train 데이터 기반 **재학습 기능**
* 작업 상태 반환 (Failed / Done / Pending)
* 저장 데이터 주기적 삭제 및 저장 공간 관리

---

### Frontend (Future)

* Streamlit 기반 실제 서비스 UI
* 이미지 업로드 시 약제 이름 및 정보 반환
* Session State 활용 캐시 유지
* POST 즉시 응답 후 **주기적 GET polling**
* Front를 통한 Train 데이터 업로드 지원

---
