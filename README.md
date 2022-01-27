# Dacon 농작물 병해예측 챌린지


## 코드 구조
파이토치 라이트닝 패턴을 따름

1. train.py
    - train entrypoint
2. model.py
    - 모델 및 트레이닝 trick 관련 모듈
3. data.py
    - dataset, dataloader 관련 모듈
4. inference.py
    - 제출물 만드는 용도
5. hyperparmas.py
    - 하이퍼라라미터 관리

## Train
python train.py


## Inference
python inference.py

 - tta 적용하고 싶으면
    python inference_tta.py


## Hyperparameter optimization 
python sweep_run.py