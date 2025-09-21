# Heterogeneous Graph Transformer (HGT) from scratch using PyTorch

## Overview
본 프로젝트는 HGT 논문 구조를 PyTorch로 로우레벨 구현하고, PyG(RGCN/HAN)과 비교 실험이 가능하도록 구성됨.  
※ Relative Temporal Encoding(RTE)은 제외.

---

## Motivation
- 메타-관계 ⟨node_src, edge, node_tgt⟩ 기반 파라미터 공유 구조가 코드에서 어떻게 구현되는지 확인  
- Attention / Message / Aggregation 의 차원 흐름을 직접 추적  
- 단순 재현이 아닌 실제 동작 구현으로 관계와 역할 명확화  

---

## Key Features
- 🧩 **논문 기반 PyTorch HGT 레이어 구현**  
- 🧩 **노드타입별 Q/K/M/A 프로젝션** + **엣지타입별 Attention/Message 행렬**  
- 🧩 **메타-관계 삼중항 파라미터화**  
- 🧩 **PyG HeteroData 지원**  
- 🧩 **노드/그래프 분류 태스크 예제 제공**  

---

## Challenge & Solution

**Challenge**  
- 타입별/관계별 변환과 shape mismatch 문제  
- autograd가 깨지지 않도록 모든 연산을 텐서화해야 함  

**Solution**  
- `ModuleDict`를 활용한 타입별 Q/K/M/A 관리  
- 엣지타입별 `W_att`, `W_msg` 파라미터 분리  
- scatter 연산으로 효율적인 집계  

---

## Conclusion
직접 구현한 HGT와 기존 PyG 모델(RGCN / GAT)을 동일 조건에서 비교 예정.

| Dataset | Task       | Metric   | HGT-scratch | RGCN | GAT |
|---------|------------|----------|-------------|------|-----|
| MAG     | Node cls   | Micro-F1 | -           | -    | -   |
| BGP     | Graph cls  | AUC      | -           | -    | -   |

---

## TO DO

| 카테고리        | 작업 항목                          | 상태    |
|-----------------|-----------------------------------|---------|
| **핵심 블록 구현** | 전처리 모듈 (임베딩/정규화)         | ✅ |
|                 | Feature Transformer               | ✅ |
|                 | Attentive Transformer             | ✅ |
|                 | HGT Encoder 구성                  | ✅ |
|                 | Autograd 점검                     | 🔧 진행중 |
| **성능 비교 실험** | Node classification (MAG)        | 🕒 예정 |
|                 | Graph classification (BGP)       | 🕒 예정 |
| **Self-Supervised** | Decoder 블록 설계              | 🕒 예정 |
|                 | Feature masking                   | 🕒 예정 |
|                 | Pretrain → Fine-tune             | 🕒 예정 |

---

## Directory

.
├── data/
├── experiments/
│ ├── mag_small_nodecls.ipynb
│ └── bin_gene_pathway_graphclf.ipynb
├── hgt/
│ ├── init.py
│ ├── layers.py
│ ├── model.py
│ ├── sampling.py
│ └── utils.py
└── README.md
