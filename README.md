### 코드 구조

```
${PROJECT}
├── config/
│   ├── train_config.yaml
│   ├── predict_config.yaml
│   └── preprocess_config.yaml
├── models/
│   ├── ML_model.py
├── modules/
│   └── utils.py
├── README.md
├── train.py
├── predict.py
└── preprocess.py
```

- config: 학습/추론에 필요한 파라미터 등을 기록하는 yaml 파일
- models:
    - ML_model.py: 모델 클래스
- modules:
    - utils.py: 여러 확장자 파일을 불러오거나 여러 확장자로 저장하는 함수 등을 포함한 파일
- train.py: 학습 시 실행하는 코드
- predict.py: 추론 시 실행하는 코드

