# Pruning

#### [Pytorch 공식홈페이지 이용](https://tutorials.pytorch.kr/)
#### pruning.py

  * 모델의 정확도가 손상되지 않는 범위에서 메모리, 배터리, 하드웨어 소비량을 줄이고, 기기에 경량화된 모델을 배치하며 저전력 컴퓨터와 같은 곳에서 사용하기위해 모델에 포함된 파라미터 수를 줄여 압축하는 최적의 기법이 필요
  * Pruning, 가지치기 기법은 굉장히 많은 수의 파라미터값들로 구성된 모델과 굉장히 적은 수의 파라미터값들로 구성된 모델 간 학습 차이를 조사하는데 주로 이용
  * LeNet을 이용하여 실습 진행, torch.nn.utils.prune 내 존재하는 가지치기 기법을 이용, 해당 모듈 내에서 가지치기 기법을 적용하고자 하는 모듈과 파라미터를 지정 
  * 가지치기 기법에 적당한 키워드 인자값을 이용하여 가지치기 매개변수를 지정
  * 모델의 conv1 층의 가중치의 30%값들을 랜덤으로 가지치기 기법을 적용
  * module : 함수에 대한 첫 번째 인자값으로 전달
  * name : 문자열 식별자를 이용하여 해당 모듈 내 매개변수를 구분
  * amount : 가지치기 기법을 적용하기 위한 대상 가중치값들의 백분율 또는 가중치 값의 연결의 개수를 지정
  * 가지치기 기법은 가중치 값들을 파라미터값들로부터 제거하고 weight_orig, 즉 초기 가중치 이름에 _orig를 붙인 새로운 파라미터값으로 대체하는 것으로 실행
  * weight_orig는 텐서값에 가지치기 기법이 적용되지 않은 상태를 저장(bias는 가지치기 기법이 적용되지 않았기 때문에 그대로 남아 있음)
  * 수정이 되지 않은 상태에서 순전파를 진행ㅎ하기 위해 가중치 값 속성이 존재해야함
  * torch.nn.utils.prune 내 구현된 가지치기 기법은 가지치기 기법이 적용된 가중치 값들을 이용하여 순전파를 진행하고, weight 속성 값에 가지치기 기법이 적용된 가중치 값들을 저장
  * 가지치기 기법은 pytorch의 forward_pre_hooks를 이용하여 각 순저파가 진행되기 전에 가지치기 기법이 적용됨
  * 모듈이 가지치기 기법이 적용되었을 때, 가지치기 기법이 적용된 각 파라미터값들이 forward_pre_hook을 얻게됨, 이러한 경우, weight 이름인 기존 파라미터 값에 대해서만 가지치기 기법을 적용하였기 때문에 훅은 오직 1개만 존재

#### global_pruning.py

  * 모듈 내 같은 파라미터값에 대해 가지치기 기법이 여러번 적용될 수 있으며, 다양한 가지치기 기법의 조합이 적용된 것과 동일하게 적용될 수 있음 
  * ex ) module.weight 값에 가지치기 기법을 적용하고 싶을 때, 텐서의 0 번째 축의 L2 norm 값을 기준으로 구조화된 가지치기 기법을적용(0번째 축이 의미하는 것은 합성곱 연산을 통해 계산된 출력값에 대해 각 채널별로 적용된다는 것을 의미)
  * 마스크 버퍼들과 가지치기 기법이 적용된 텐서 계산에 사용된 기존의 파라미터를 포함하여 관련된 모든 텐서값들은 필요한 경우 모델의 state_dict에 저장되기 때문에, 쉽게 직렬화 하여 저장 할 수 있음
  * 가지치기 기법이 적용된 것을 영구적으로 만들기 위해, 재 파라미터화 관점의 weight_orig와 weight_mask, forward_pre_hook 값을 제거
  * 범용적이고 더 강력한 가지치기 방법은 각 층에서 가장 낮은 20%의 연결을 제거하는것 대신에 전체 모델에 대해 가장 낮은 20% 연결을 한 번에 제거하는 것임



# Pruning 이해
## Pruning
 * weight 및 활성화의 희소성(Sparsity)를 유도하는 일반적인 방법론 : Pruning
 * 주어진 임계값 보다 아래의 가중치에 0 값이 할당, Pruning을 통해 0으로 설정하여 Back Propagation에 참여하지 않도록 함
 * bias : 보통 Layer의 출력에 대한 기여도가 상대적으로 커 Pruning할 요소가 거의 없음
 * activation : ReLU같은경우 음의 활성화를 정확하게 0 으로 만들기 때문에 일반적으로 ReLU 계층 다음에 희소성이 유도됨
 * weight의 Sparsity는 weight가 매우 작을 수도 있지만 정확하게 0이 아니기 때문에 bias와는 다르게 일반적으로 Pruning에 사용됨

## Sparsity 정의
 * 희소성(Sparsity)는 Tensor크기에 비해 Tensor에서 얼마나 많은 요소가 정확히 0인지를 측정하는 것
 * 요소의 대부분이 0이면 Sparsity한 것으로 간주됨 
 * L1 norm을 통해 0의 요소의 수를 측정 할 수 있음 

## Weight Pruning
 * Weight Pruning 또는 Model Pruning은 가중치의 Sparsity 를 높이기 위한 방법, 즉 Tensor에서 값이 0인 요소의 양을 높이는 것
 * Pruning을 진행할시 각 요소의 절대 값을 이용, 절대 값은 일부 임계 값과 비교되며 임계값 미만인 경우 요소는 0으로 설정됨-> 절대 값이 임계값 보다 낮으면 0으로 만든다라고 생각하면 될듯
 * distiller.MagnitudeParameterPruner 클래스에 의해 구현 ->L1 Norm이 작은 가중치가 최종 결과에 거의 기여하지 않으므로 덜 중요하고 제거 할 수 있다 판단
 * over parameterized되면 중복된 특징들이 많으니 중복된 부분중 일부는 가중치를 0으로 설정하여 제거 할 수 있음 
 * 가능한 많은 0이 있는 가중치 집합을 탐색하는 것으로 생각할 수도 있음

## Pruning의 어려움
 * Pruning을 통해 Sparsity를 유도하는데 있어 어려운 부분은 각 층의 Tensor에 사용한 임계값 또는 Sparsity 수준을 결정하는 것
 * Sensitivity analysis(민감도 분석)은 Pruning에 대한 민감도에 따라 Tensor의 순위를 매기는데 도움이 되는 방법


# Paper
#### 1.[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)  
#### 2.[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
