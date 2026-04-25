- 데이콘 경진대회 데이터를 활용
- 본문 내 개념 및 주요 실험 결과는 참고자료 참고! 
- 모델링 Task

---

## [모델링] 

- 반도체 공정의 수율 예측, 개발한 신약의 효능을 높이기 위해서는 최적의 파라미터 조합이 필요

- 하지만 세분화된 변수의 조합으로 인해, 현실적으로 최적의 조합을 찾기에는 비용, 시간으로 인해 한계

- 이런 환경에서 `과거 데이터`를 기반으로 학습한 모델을 사용해 복잡한 공정의 입력 피처를 최적화하는 과정을 `오프라인 모델 기반 최적화(Offline model based Optimization)`이라고 칭함

- 이는 기존 머신러닝 모델링 과정과 달리, 최적의 입력 변수 조합을 찾을 수 있는 대리 모델(surrogate model)을 구축해야 한다.
   - `기존 모델`: 변하는 값(X)을 기반으로 Y 값을 잘 예측하는 것이 목적
   - `MBO` : Y값을 잘 예측하기 위한 입력 변수의 값을 알아내는 게 목적
   
- 또한 위에서 언급한 변수 조합이란,
   - `변수 선택(feature selection) 개념이 아님`

  |-|Feature Selection|Model-based OPtimization|
  |:---:|:---:|:---:|
  |목적|중요 피처 선별|모든 피처의 최적 값 탐색|
  |결과|피처 부분집합 선별|각 피처의 구체적인 값|
  |차원|차원 축소|차원 유지 or 증가|
  |예시|{x_1, x_4, x_8, x_10}| x_0 →1.xx, x_1 → 2.xx, … ,x_10 → 3.xx|
   -  일반적인 경우, MBO는 차원 축소하지 않음
   	   -  ex) 기존 변수 3개로 파생 변수 생성 뒤 기존 변수를 삭제하면 어떤 조합으로 생성 됐는지 파악하기 어렵기 때문!

<br>

- 앞서 알아본 모델 기반 최적화를 위해선, 우선 `대리(surrogate) 모델을 구축하는 과정이 선행`되어야 함
   - 현실 현상을 설명할 모델은 찾기도 어려울 뿐만 아니라, 찾는다해도 내부를 이해하기엔 불가능
   - 따라서 `대리 모델` 사용하여 입력 값 후보를 찾고, 이 값에 대한 예측 결과를 얻음(`=시간 및 비용 절약`)
 
> 따라서 이번 프로젝트에선 surrogate 모델 최적화를 통해 예측 성능을 높이고자 한다.


- Datasets
   - X_0, X_1, X_2, …, X_10까지 총 11개의 입력 변수와 y(종속변수)로 구성
   - 각 입력 변수가 어떤 정보인지 명시되어 있지 않음

- 평가지표
   - 예측 값의 상위 10% 이내에 속한 값이 실제 값 상위 5% 포함 비율에 대한 Recall
      - `Recall @ Top K`


</br>

---

## EDA

> 💡 우선 각 추가로 적용할 방법을 찾기 위해 EDA부터 진행!

### 1) Check Feature Skewness
![](https://velog.velcdn.com/images/yjhut/post/53c4a52f-4507-4960-b94b-fedeab4ef16b/image.png)

- 피처 대부분의 왜도(skewness)가 3 미만이라 transformation은 일단 스킵!

</br>

### 2) Check Correlation & VIF

- 각 피처 간 상관계수는 아래와 같다.
![](https://velog.velcdn.com/images/yjhut/post/03ad3221-495f-491f-a450-0a77f5fca995/image.png)

- 상관 관계가 적은 변수도 있지만 동시에 곳곳에 높은 correlation을 보임
- 피처 간 correlation이 높을 때 <u>다중공선성</u>을 확인해야 하기 때문에 <u>분산팽창지수(VIF)</u>를 사용해보자 

![](https://velog.velcdn.com/images/yjhut/post/6e35a0d0-ec52-4de0-99e5-cf44875dbd28/image.png)


- VIF가 10만 넘어도 다중공선성을 의심해야 하는데, 현재 대부분 이를 10을 넘는 상태를 보임




---

## CASE 01. Baseline 

- 전처리 없이 사용

-  LightGBM_Regressor + Random_Search

- 결과

|-|LightGBM + Random Search|
|:---:|:---:|
|estim-ators|colsample_bytree=0.7, max_depth=6, n_estimators=1000,  num_leaves=7, subsample=0.8,  min_child_samples=100, leaning_rate=0.01|
|Recall|86.07|

---

## Case 02. 상관계수 기준 + 변수 중요도 반영 파생변수 생성


### 2-1. Derived Variable

- 앞서 변수 간 상관계수를 확인했을 때, 양(+) 또는 음(-)의 상관계수. 특히 강도가 강한 변수를 선택하여 두 개의 집합 A, B로 묶어 각 집합에 맞게 파생변수를 생성하는 과정을 수행하였다.

>
- `집합 A: [x_1, x_7, x_8, x_9]`
   - x_8만 나머지 변수와 음의 상관관계, 나머지는 상호 양의 상관관계
>
>
- `집합 B: [x_4, x_5, x_10]`
   - x_5만 나머지 변수와 음의 상관관계, 나머지는 상호 양의 상관관계


- A, B에서 + 관계끼리는 sum, - 관계는 subtract를 수행
   - `신호(signal) 선명도를 높이고, 모델의 해석혁을 높이기 위한 방법`

- 파생 변수 생성 과정에서 연산 시, 동일한 가중치를 반영하기엔 문제가 있다고 판단하여, `ase01의 변수의 중요도`를 파악하였다.
![](https://velog.velcdn.com/images/yjhut/post/ae6ae30f-e7fa-412e-92f9-dfe91c0b8e8e/image.png)

- 파생 변수 생성 시 각 변수의 가중치는 아래와 같이 계산하였다.
$$
\frac{Inportance_{feature}}{Importance_{total}}
$$

<br>

- 각 집합에 해당하는 파생 변수 2개를 추가하여 모델에 적용한 결과 `(총 변수 13개)`


|-|LightGBM + Random Search|
|:---:|:---:|
|estim-ators|colsample_bytree=0.7, max_depth=4, n_estimators=500,  num_leaves=127, subsample=0.9,  min_child_samples=100, leaning_rate=0.01|
|Recall|86.56|

- case01에 비해 약간의 성능 향상`(+0.56%)`을 보이나 적절한 방법은 아닌 듯

---

## Case 03. OCAE(비선형 변환)

- case02는 선형 변환을 기반으로 변수를 추가로 생성했을 때 효과를 확인하고자 하였으나, 효과 X 

- 따라서 case03에선 `기존 변수에 비선형 변환을 적용한 결과를 lgbm 입력으로 사용`하는 방법을 적용하고자 한다.
   - `EDA`에서 확인했듯, 변수 간 강한 다중공선성이 의심되어, 비선형 변환을 통해 이를 완화하고자 시도

### 3-1. AutoEncoder 

- 비선형 변환의 여러 기법 중 `AutoEncoder`. 그 중에서도 `latent variable이 입력 차원보다 크기가 큰 OverComplete AutoEncoder`를 사용한다.
   - `차원 확장`을 통해 데이터의 복잡한 패턴을 잘 학습하고, 표현력(representation)을 높이기 위해!

>
- AutoEncoder란
   - 잠재 표현(Latent Representation)을 학습하는 모델
   - 입력 데이터를 재구성하는 과정으로 이상치 탐지하는 대표적인 모델
   - 변수 추출(feature extraction) 또는 Denoising 용도
>
</br>
>
- 오토인코더 모델의 기본 구조 아래와 같다.
![](https://velog.velcdn.com/images/yjhut/post/6f5a0720-7771-402b-810b-345c979963ff/image.png)
> - 일반적으로 `입력 데이터 차원의 크기 > 잠재 차원의 크기`이고, 이를 `Undercomplete Autoencodedr`라 하여, 차원 축소 or 압축 개념으로 사용
>
> - 반대로 `입력 데이터 차원의 크기 < 잠재 차원의 크기`인 경우, `Overcomplete Autoencoder`라 하여, 데이터의 특징 학습(Featuring Learning)하는 데 사용
>
>
- `Overcomplete`의 경우 Regularization이 없으면 모델이 입력을 복제하는 `Identify Mapping`이 발생할 위험이 높아지고, 이는 차원 확장의 의미가 없어지기 때문에 `weught_decay(L2)`, `BatchNorm`, `Sparsity Loss` 등 규제 필요

<br>


### 3-2. About latent Dimension

-  OCAE 학습 손실은 아래와 같이 산출하였다.
   - $OCAE_{Loss} = MSE_{loss} + ratio*Sparsity _{loss}$

- OCAE 학습에 사용한 파라미터는 아래와 같으며,
   - 배치 사이즈: `64`
   - 잠재 변수(z) 크기: `50`
   - OCAE 학습률: `1e-4`
   - sparsity_loss ratio: `1e-3`

- 학습 손실 그래프는 [참고자료]에서 확인 가능


### Performance

- LGBM Params `z_size=50`

|-|LightGBM + Random Search|
|:---:|:---:|
|estim-ators|colsample_bytree=1.0, max_depth=3, n_estimators=750,  num_leaves=7, subsample=0.8,  min_child_samples=100, leaning_rate=0.01|

<br>

> 💡 
- `case.01` 대비 상대 향상률 `+1.73%`, `case.02`대비 `+1.15%`개선된 결과를 보임 
>

---



## 정리

> 💡 raw_data -> OCAE -> z 추출 -> LightGBM Regressor -> Prediction의 과정을 제안
>
> 💡 치원 확장 시, 입력 데이터를 단순 복사하는 `identity mapping` 방지하기 위해 Sparsity Loss(L1)를 사용
>
> 💡 고차원 확장 시 z가 희소해지고 lgbm는 `순위 정보`를 활용하기 때문에 성능을 내는 데 무리가 없었으나, 트리 계열이 아닌 모델 사용 시 작은 값으로 인한 성능 저하를 유의해야 할 것 같다.


---


## 참고

### 1. 다중공선성(Multi Collinearity)
- 설명 변수간 강한 correlation으로 인해 설명력이 떨어지는 문제
- 예측 성능 저하, 불안정한 추정치
- 확인 방법 : 변수간 correlation 확인, VIF 확인

</br>

### 2. 분산 팽창 지수(Variance Inflation Factor)
$$ 
 \frac{1}{1-R^2_{i}} 
$$

- 각 피처의 VIF는 해당 피처를 output으로 두고, 나머지 피처의 결합으로 구성된 회귀 모델의 결정계수(R^2) 값을 기반으로 표현

- VIF 값에 따른 의미는 아래와 같다.

  |VIF|의미|
  |:---:|:---:|
  | <= 5 |다중공선성 없음|
  | 5 > and <= 10|다중공선성 주의|
  | 10 < |강한 다중공선성 의심|

</br>

### 3. 상대 향상률

$$
 \frac{|proposed_{model} - baseline|}{baseline} * 100
$$

</br>

### 4. OCAE 훈련 손실 그래프

![](https://velog.velcdn.com/images/yjhut/post/df092e9a-0e11-4578-b13f-b5c5cb8f411b/image.png)
  
