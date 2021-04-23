# YOLO v1

## 소개
You Only Look Once(YOLO v1): Unified, Real-Time Object Detection 논문 리뷰 및 코드 공부 (https://arxiv.org/abs/1506.02640)

논문 리뷰: https://velog.io/@skhim520/YOLO-v1-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-%EB%B0%8F-%EC%BD%94%EB%93%9C-%EA%B5%AC%ED%98%84  
참고한 코드(baseline):https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO  
참고한 유튜브: https://www.youtube.com/watch?v=n9_XyCGr-MI  

참고한 코드 및 유튜브는 aladdinpersson님의 깃허브와 유튜브 강의 입니다.

## 코드 실행

### 환경설정

```bash
matplotlib==2.2.2
numpy==1.15.4
python==3.6.7
pytorch==1.0.0
pandas==0.23.4
pillow==5.3.0
torchvision==0.2.1
tqdm==4.28.1
```


### Dataset
사용한 데이터셋은 PASCAL VOC으로 https://www.kaggle.com/dataset/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2에서 다운 가능합니다.

### Train

```bash
python train.py
```
argument는 `train.py`에서 수정 가능합니다.
```python
# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
```

## 논문 구현과 다른 점
YOLO v1 논문을 이해하기 위한 코드 구현으로 computation cost등의 문제로 논문의 세세한 부분까지는 구현하지 못 했습니다. 또한 시각화에 신경쓰지 못 했습니다.  
다른 점은 다음과 같습니다.
1. pre-train network를 사용하지 않았다.
2. hyperparmeter가 다르다.(learning rate, batch size, etc.)
3. learning rate schedule을 사용하지 않았다.
