### ResNet
  - Residual을 학습한다고 하여 ResNet
  - 기존 모델에서 layer가 깊어지면 Gradient Vanishing / exploding의 문제가 발생해 성능이 떨어지는 Degradation 발생
    - depth가 깊어질 수록 training-error가 높아짐
  - residual network 구조 (2 layer)  
![resnet](./img/resnet.png)
      - skip connection 사용
      - 입력(x)과 출력(H(x))의 차이가 0이 되도록 학습
  
  - ResNet50부터 2 layer대신 3 layer bottleneck block 사용
    - x → 1 x 1 conv → 3 x 3 conv → 1 x 1 conv → output
  
### EfficientNet
  - 2019년 발표
  - 모델 사이즈 별 B0~B7버전

- 모델 크기를 키우는 방법(CNN의 성능을 높일 수 있는 요소)
    1. width scaling
    2. depth scaling
    3. resolution scaling
  - 세 요소를 균등하게 조합한 **Compound scaling** 기법을 제안
  - 각 scale factor는 samll grid search 방법을 통해 찾음
  
- model scaling에 의한 성능 향상은 baseline network에 의존적, baseline network를 설정하는데 Neural Architecture Search(NAS)를 사용
  > NAS : 사람이 neural network 구조를 직접 정하지 않고 자동으로 찾아주는 신경망 아키텍쳐 탐색 기술  
  > 검색 공간, 검색 전략, 성능 추정 전략 세가지 측면으로 분류

### MobileNetv1
  - Depthwise separable convolution 기법을 적용
    - Depthwise convolution + Pointwise convolution 
      - Depthwise convolution : 각 채널별로 쪼개서 컨볼루션 연산 적용, 입력 채널과 출력 채널은 항상 동일
      - Pointwise convolution : 1 x 1 컨널로 컨볼루션 연산, 차원을 줄여줌
  - 기존 아키텍쳐들보다 적은 파라미터와 연산량을 가짐

### MobileNetv2


