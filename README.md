# Heterogeneous Graph Transformer (HGT) — PyTorch 로우레벨 구현

## Overview
본 프로젝트는 **HGT 논문 구조**를 PyTorch로 **로우레벨부터 직접 구현**하고, PyG의 대표 이질그래프 모델(RGCN/HAN)과 **동일 조건**에서 비교 실험이 가능하도록 구성한다.  
- 실험 데이터는 HGT 원 논문에서 제공된 아카데믹 그래프 중 **가장 작은 서브셋인 CS**(Computer Science)만 사용한다.  
- **시간 정보(Temporal/RTE)는 고려하지 않음**: RTE 모듈 제외.  
- 우선 **공식 구현(pyHGT) 아이디어/코드 스타일을 참고**해 동작을 재현하고, 이후 **Hi-C 데이터**로 **적응(Adaptation)** 포인트를 도출한다. 

---

## Motivation
- **삼중항(메타-관계) 파라미터화**가 실제 코드에서 어떻게 분해·공유되는지 확인  
- **Attention → Message → Aggregation**으로 이어지는 **텐서 차원 흐름**(batch, heads, d_head …)을 직접 추적  
- 단순한 “재현”이 아니라 **실제 학습·추론 가능한 수준**으로 구현하고, **PyG(RGCN/HAN/GAT)**과 동일한 조건에서 성능·자원 비교

---

## Key Features
- 🧩 **논문 기반 PyTorch HGT 레이어** (Q/K/M/A를 타입별 관리, 엣지타입별 W_att/W_msg 분리)  
- 🧩 **메타-관계 삼중항 파라미터화** (⟨source type, edge type, target type⟩)  
- 🧩 **PyG HeteroData 호환 유틸 제공**  
- 🧩 **Node/Graph 분류 태스크 예제 제공**  
- 🧩 **시간 정보(RTE) 제외**: 구조 학습에 집중  

---

## Challenge & Solution

### 🔧 Challenges
1. **타입/관계별 행렬 때문에 크기 안 맞음**  
   - 노드 타입, 엣지 타입마다 다른 가중치 행렬을 쓰다 보니, 행렬 곱셈할 때 차원이 자꾸 안 맞음  

2. **autograd(자동 미분) 깨질 위험**  
   - 계산 그래프가 끊기면 역전파가 안 돼서 학습 자체가 안 됨  
   - 복잡한 연산도 전부 텐서 연산으로 짜야 안전함  

3. **메시지랑 어텐션을 따로 처리해야 함**  
   - HGT는 단순히 노드 임베딩만 전달하는 게 아니라  
   - **어텐션 경로(Q/K)**와 **메시지 경로(M)**가 따로 있어서 구현이 복잡함  

4. **희귀 관계 처리 문제**  
   - 어떤 타입-관계는 자주 등장하고, 어떤 건 거의 안 나오는데  
   - 파라미터를 “공유할지” “따로 둘지” 결정하기 어려움  

---

### ✅ Solutions
- **ModuleDict 4개로 정리**  
  - Q/K/M/A 행렬을 노드 타입별로 딱딱 구분해 저장 → 관리 쉬움  

- **엣지타입별 행렬 따로 둠**  
  - W_att(어텐션용), W_msg(메시지용)을 관계별로 따로 만들어  
  - 같은 노드 쌍이라도 관계가 다르면 다른 변환 적용  

- **scatter 연산으로 메시지 모으기**  
  - 여러 노드에서 들어온 메시지를 효율적으로 합치는 데 `torch_scatter` 사용  
  - 속도 빠르고 autograd도 안전하게 유지  

- **Shape 규칙 통일**  
  - (Batch, Head, Node, Head_dim) → (Batch, Node, Head×Head_dim)  
  - 변환 규칙을 딱 정해서 중간에 차원 꼬임 방지  

- **삼중항 분해로 파라미터 공유**  
  - (source type, edge type, target type) 세 가지로 나눠서 파라미터 정의  
  - 덕분에 자주 안 나오는 희귀 관계도 학습 가능  

---

## Experiment Plan
- **Node classification (CS)**: Micro-F1  
- **Graph classification (toy)**: AUC, F1  
- **자원 지표**: #Params, 1-epoch batch time  

**Baselines**: RGCN / HAN / GAT (동일 hidden dim, heads, layer 수, optimizer 설정)  

**Ablations**:  
- HGT(-Heter): 메타-관계 분해 제거  
- HGT(+Heter): 본 구현  


---

---

## TO DO

| 카테고리        | 작업 항목                                           | 상태    |
|-----------------|----------------------------------------------------|---------|
| **핵심 블록**   | Q/K/M/A + W_att/W_msg 구현, scatter aggregation     | ✅ |
|                 | A_Linear + residual + 활성화 체인                   | ✅ |
| **학습 루프**   | CS node classification: HGT vs RGCN/HAN/GAT         | 🔧 진행중 |
|                 | 파라미터 수/배치 타임 로깅                          | 🔧 진행중 |
| **Ablation**    | HGT(-Heter) vs HGT(+Heter)                         | 🕒 예정 |
| **그래프 분류** | toy graph AUC 실험(풀링 방식 비교)                  | 🕒 예정 |
| **도메인 연결** | bin–bin 가중치 주입 / 단계 임베딩 / 서브그래프 샘플링 | 🕒 예정 |
| **논문화 요소** | 희귀 관계 일반화, 특이도 보전, 구조-모델 정렬 근거화 | 🕒 예정 |

---


## Directory
```
.
├── data/
│ └── cs_subset/ # CS 서브셋(전처리 결과)
├── experiments/
│ ├── cs_nodecls_baselines.ipynb
│ ├── cs_nodecls_hgt.ipynb
│ └── graphclf_toy.ipynb
├── hgt/
│ ├── init.py
│ ├── layers.py # Q/K/M/A + W_att/W_msg + scatter agg
│ ├── model.py # HGTEncoder, heads(node/graph)
│ ├── sampling.py # (옵션) 타입-균형 샘플링(간단판)
│ └── utils.py # HeteroData adapter, shape checks
└── README.md
```


## Notes
- 본 구현은 **시간 정보 비사용(RTE off)** 버전  
