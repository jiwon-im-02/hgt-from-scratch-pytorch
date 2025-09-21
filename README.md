# Heterogeneous Graph Transformer (HGT) — PyTorch 로우레벨 구현

## Overview
본 프로젝트는 **HGT 논문 구조**를 PyTorch로 **로우레벨부터 직접 구현**하고, PyG의 대표 이질그래프 모델(RGCN/HAN)과 **동일 조건**에서 비교 실험이 가능하도록 구성한다.  
- 실험 데이터는 HGT 원 논문에서 제공된 아카데믹 그래프 중 **가장 작은 서브셋인 CS**(Computer Science)만 사용한다.  
- **시간 정보(Temporal/RTE)는 고려하지 않음**: RTE 모듈 제외.  
- 우선 **공식 구현(pyHGT) 아이디어/코드 스타일을 참고**해 동작을 재현하고, 이후 유동층 코팅 공정 데이터로 **적응(Adaptation)** 포인트를 도출한다.  

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
1. 타입/관계별 행렬 관리 시 shape mismatch  
2. autograd가 끊기지 않도록 모든 연산을 텐서화해야 함  
3. 메시지/어텐션 경로를 분리해 구현해야 하는 복잡성  
4. 희귀 관계 학습 시 파라미터 공유 vs 특화의 균형  

### ✅ Solutions
- **`ModuleDict` × 4**: Q/K/M/A 노드타입별 투영 일관 관리  
- **엣지타입별 행렬 분리**: W_att, W_msg를 분리해 관계별 의미 반영  
- **scatter 연산 기반 집계**: torch_scatter로 멀티헤드 메시지를 안정 집계  
- **Shape 규약·유닛테스트**: (B, H, N, d_h) → (B, N, H·d_h) 통일  
- **삼중항 분해**로 희귀 관계에도 일반화 이점 확보  

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

## Domain Adaptation (유동층 코팅 데이터에의 적용 아이디어)
1. **관계 스키마 확장**: bin–bin 관계를 공정 단계별 타입으로 분리 or 단계 one-hot을 엣지 특징으로 추가  
2. **엣지 가중치 활용**: `wt_mats1`을 메시지/어텐션 경로 모두에 반영 (특이도 보전)  
3. **그래프 분류 헤드**: bin만 풀링 vs bin+gene/pathway 멀티풀링 비교  
4. **샘플링 전략**: ID 단위 full-batch → 타입 비율 유지 HGSampling 스타일  
5. **특징 정규화/임베딩**: 연속 변수 group norm, 범주형 변수 임베딩  
6. **-Heter vs +Heter 성능 비교**: 희귀 공정 관계에서 일반화 이득 확인  

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
