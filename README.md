- 데이콘 경진대회 데이터를 활용하였습니다.
- 본문 내 밑줄친 갠며에 대한 추가정보는 참고자료에 있습니다.
- 모델링
- 🎒GitHub: https://github.com/kimho-jun/DACON_Black-box-optimization/tree/main

---

## [모델링] 

- 반도체 공정의 수율 예측, 개발한 신약의 효능을 높이기 위해서는 최적의 파라미터 조합이 필요
- 하지만 점점 세분화된 변수의 조합으로 인해, 현실적으로 최적의 조합을 찾기에는 비용, 시간 측면에서 많은 비용이 소모
- 이런 환경에서 과거 데이터를 기반으로 학습한 모델을 사용해 복잡한 공정의 입력 피처를 최적화하는 과정을 `오프라인 모델 기반 최적화(Offline model based Optimization)`이라고 칭함
- 이는 기존 머신러닝 모델링 과정과 달리, 최적의 입력 변수 조합을 찾을 수 있는 대리 모델(surrogate model)을 구축해야 한다.
- 위에서 언급한 변수 조합이란,
-> 변수 선택(feature selection) 개념이 아님

  |-|Feature Selection|Model-based OPtimization|
  |:---:|:---:|:---:|
  |목적|중요 피처 선별|모든 피처의 최적 값 탐색|
  |결과|피처 부분집합 선별|각 피처의 구체적인 값|
  |차원|차원 축소|차원 유지|
  |예시|{x_1, x_4, x_8, x_10}| x_0 →1.xx, x_1 → 2.xx, … ,x_10 → 3.xx|


- 데이터셋
-> X_0, X_1, X_2, …, X_10까지 총 11개의 입력 변수와 y(종속변수)로 구성
-> 각 입력 변수가 어떤 정보인지 명시되어 있지 않음

- 평가지표
-> `예측 값의 상위 10% 이내에 속한 값이 실제 값 상위 5% 포함 비율에 대한 Recall`

+ 실험의 가중치 시드는 43으로 고정

</br>

---

## CASE 01. Baseline 모델
1. Random Forest_Regressor + Random_Search(cv = 5)
2. LightGBM_Regressor + Random_Search(cv = 5)

</br>

우선 두 베이스라인 모델 적용 결과는 아래와 같다.

|-|Random Forest + Random Search|LightGBM + Random Search|
|:---:|:---:|:---:|
|estim-ators|max_depth=5,max_features=0.7999999999999999,min_samples_leaf=10, n_estimators=110, n_jobs=-1,n_iter = 300|colsample_bytree=0.9,max_depth=5,n_estimators=70, n_jobs=-1,num_leaves=7,subsample=0.8,n_iter=500,min_child_samples=21,leaning_rate=0.1|
|Recall|0.8656|0.8706|

> 성능을 향상시키기 위해 데이터 EDA를 통해 적용할 방법을 살펴보자

</br>

---  
 
## 데이터 EDA
- 현재 데이터는 아래와 같이 제공되어 있다.
![](https://velog.velcdn.com/images/yjhut/post/5a7e60cb-3799-46f0-97c0-03395a528862/image.png)

</br>
  
1) Correlation + Variance_Inflation_Factor

- 각 피처 간 상관계수는 아래와 같다.
![](https://velog.velcdn.com/images/yjhut/post/cd567684-1df4-45d1-a268-95b9cb425c34/image.png) 
- 상관 관계가 적은 변수도 있지만 동시에 곳곳에 높은 correlation을 보임
- 피처 간 correlation이 높을 때 <u>다중공선성</u>을 확인해야 하기 때문에 <u>분산팽창지수(VIF)</u>를 사용해보자 

![](https://velog.velcdn.com/images/yjhut/post/10103e9f-5f7f-479f-91a7-3ba42259381f/image.png)
- VIF 결과가 비정상적으로 높아 기존 데이터를 그대로 사용하지 않고 오토인코더와 같은 비선형 변환 방법을 적용해보자

</br>

---

#### AutoEncoder
- 오토인코더란?
-> 잠재 표현(Latent Representation)을 학습하는 모델
-> 입력 데이터를 재구성하는 과정으로 이상치 탐지하는 대표적인 모델
-> 변수 추출(feature extraction) 또는 Denoising 용도

</br>

- 오토인코더 모델의 기본 구조 아래와 같다.
![](https://velog.velcdn.com/images/yjhut/post/6f5a0720-7771-402b-810b-345c979963ff/image.png)
> - 일반적으로 `입력 데이터 차원의 크기 > 잠재 차원의 크기`이고, 이를 `Undercomplete Autoencodedr`라 하고, 차원 축소 or 압축 개념으로 사용
>
> - 반대로 `입력 데이터 차원의 크기 < 잠재 차원의 크기`인 경우, `Overcomplete Autoencoder`라 하고, 데이터의 특징 학습(Featuring Learning)에 사용

</br>

- 입력 데이터 자체가 11차원으로 작기 때문에 Overcomplete 방법을 적용하려 한다.
-> 단, overcomplete의 경우 Regularization이 없으면 단순 복사인 Identify Mapping이 발생하며, 차원 확장의 의미가 없어지기 때문에 `L2 정규화`, `배치정규화` 등 규제가 필요하다.

</br>

---

## Case 02. autoencoder + lgbm
- 아래 표는 오토인코더에 사용한 파라미터(인코더 - 디코더 symmetric)

  |파라미터|값|
  |:---:|:---:|
  |Num_layer|7|
  |Activation Function|ReLU|
  |Weight_decay|1e-3|
  |Learning_rate|5e-4|
  |Gradient cliping (max_norm)|1.0|
  |Batch_Nomalization|True|
  
</br>

- 확장 크기는 후보는 30, 40, 50, 60을 사용하였으며, recall과 베이스라인 대비 <u>상대 향상률</u>은 아래와 같다.  

  |Baseline|Training Time(lgbm)|auto+lgbm(latent_size)|상대 향상률|
  |:---:|:---:|:---:|:---:|
  |0.8706|96.0|0.8905 `(30)`|+2.28%|
  |0.8706|118.0|0.8706 `(40)`|-|
  |0.8706|163.0|0.9005 `(50)`|<span style="color:yellow">`+3.43%`</span>|
  |0.8706|264.0|0.8855 `(60)`|+1.71%|
  
  </br>
  
- AE 적용 후, LightGBM에 사용한 파라미터

  |-|조합|
  |:---:|:---:|
  |estim-ators|colsample_bytree=0.9,learning_rate=0.15000000000000002,max_depth=3, min_child_samples=22, n_estimators=30, n_jobs=-1, num_leaves=5, random_state=43,n_iter=500, verbose=-1|
  
</br>

---

## Clustering

그렇다면, 기존 데이터 대비 overcomplete autoencoder(이하 OCAE)를 사용한 데이터 간 표현력(representation)을 클러스터링(KMeans) + <u>실루엣 계수</u>, <u>CH 스코어</u>로 비교해보자.

</br>

- 최적의 클러스터 개수(k)는 Elbow method를 사용
![](https://velog.velcdn.com/images/yjhut/post/4b018ac2-2e9b-4f4f-bea4-fc624d9e31d1/image.png) 이를 기반으로 `k = 3`을 사용


- k=3일때,

  |-|기존 데이터|OCAE로 확장한 데이터|
  |:---:|:---:|:---:|
  |실루엣 계수|0.5767|`0.698`|
  |CH-score|98984.48|`376258.09`|

> - OCAE로 확장한 데이터의 실루엣 계수가 약 0.111 향상된 성능을 보였으며, 마찬가지로 CH score도 기존 데이터 대비 분산 비율이 약 3.8배 상승함을 보였다.
> - 이는 OCAE 적용 데이터가 공간상에서 뚜렷한 클러스터 구조를 형성함을 확인할 수 있다.
 

</br>

---

## 후기

1. 기존 데이터에 AE를 사용, 그 중에서 차원을 확장시키는 OCAE를 처음 사용하였는데, 기존 데이터의 특징을 학습하면서 차원을 확장하는 방법에 대해 배우고 적용할 수 있었다.

2. EDA 과정에서 다중공선성과 VIF가 어떤 매커니즘으로 측정되는 지 확실히 정리하였다.

3. 클러스터링을 오랜만에 적용하여 엘보우 메서드의 의미와 세부적인 과정에 대해 확실히 정리할 수 있었고 기존에 사용하던 실루엣 계수 이외의 평가 지표인 Calinski-Harabasz score의 산출 매커니즘과 의미에 대해 정리할 수 있었다. 

</br>

---


## 참고자료

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

### 4. OCAE 훈련 손실 그래프 (epoch = 4에서 stop)
![](https://velog.velcdn.com/images/yjhut/post/8fe461aa-4c66-490b-bdc9-bc14690a39df/image.png)

  
</br>

### 5. 실루엣 계수(Silhouette Score)
$$
\frac{b(i)-a(i)}{max(b(i), a(i))}
$$
  
- b(i)는 클러스터 간 분리도(seperation), a(i)는 클러스터 내 응집도(cohension) 의미
- 분리도가 클수록, 응집도가 작을수록 좋은 구조
- 즉, 1에 가까울수록 클러스터링 품질이 좋고, 0에 가까울수록 클러스터링 품질 떨어짐
  
</br>

### 6. Calinski-Harabasz Score
$$
\frac{Tr(B_{k} / (K-1))}{Tr(W_{k} / (n-K))}
$$
  
- Tr은 행렬의 대각합(Trace)의미
- Between Cluster dispersion matrix의 Tr(B_k)는 모든 클러스터 중심이 전체 데이터의 평균으로부터 얼마나 멀리 퍼져있는 지 나타내는 분산의 총합`(= 클러스터 간 분산의 크기)`
- within cluster dispersion matrix의 Tr(W_k)는 각 클러스터 내 샘플들이 클러스터의 중심으로부터 얼마나 퍼져있는 지 나타내는 분산의 총합`(=클러스터 내 분산의 크기)`
- n은 샘플 수, K는 클러스터 개수
- 값의 범위는 [0,∞]로, Tr(B_k)가 클수록. 즉, 클러스터 간 거리가 크고 Tr(W_k)가 작을 수록. 즉, 클러스터 내 분산이 작아 데이터 간 응집도가 클 때 클러스터링이 잘 되었음을 의미  
  
  
