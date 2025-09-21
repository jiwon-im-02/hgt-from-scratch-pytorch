Title

Heterogeneous Graph Transformer (HGT) from scratch using PyTorch

Overview

본 프로젝트는 HGT 논문 구조를 PyTorch로 로우레벨 구현하고, PyG(HeteroData) 및 기존 이질 그래프 모델(RGCN/HAN 등)과 비교 실험이 가능하도록 구성됨.
※ 본 구현은 Relative Temporal Encoding (RTE) 과 HGSampling 중 RTE는 제외. (필요 시 후속 TODO로 추가) 

Heterogeneous Graph Transformer

Motivation

논문에서 제안한 메타-관계(⟨node_src, edge, node_tgt⟩) 기반 삼중항 파라미터화가 실제 코드에서 어떻게 표현되고 학습 안정성/일반화에 어떤 이점을 주는지 직접 확인. 

Heterogeneous Graph Transformer

메타-관계별 Attention/Message/Aggregation이 차원과 데이터 흐름을 어떻게 바꾸는지 추적.

재현을 넘어, **우리 도메인(예: bin–gene–pathway)**에도 HGT 블록이 쉽게 이식되도록 모듈화.

Key Features

🧩 논문 기반 HGT 레이어: Q/K/M/A(Attention/Aggregation용) 노드타입별 선형변환 + 엣지타입별 행렬로 구성된 멀티헤드 주의. 

Heterogeneous Graph Transformer

🧩 메타-관계 삼중항 파라미터화로 희소 관계에도 파라미터 공유/일반화 강화. 

Heterogeneous Graph Transformer

🧩 RTE 제외 버전(시간 정보 비사용)과 후속 RTE 확장 포인트 명확화. 

Heterogeneous Graph Transformer

🧩 PyG HeteroData/NeighborLoader 기반 미니배치 학습 파이프라인.

🧩 노드타입별 입력 어댑터(feature space 정렬) 및 그래프 수준/노드 수준 태스크 예시 스크립트.

Challenge & Solution

Challenge

타입/관계마다 다른 투영행렬을 쓰되 과대매개변수화를 피하면서 모듈 간 shape 정합을 유지해야 함. 

Heterogeneous Graph Transformer

Solution

ModuleDict로 Q/K/M/A를 노드타입 키로 보관, W_att, W_msg를 엣지타입 키로 분리.

메시지패싱은 edge_type별로 일괄 계산 → scatter_add로 집계해 효율 확보.

초기엔 RTE 미적용으로 단순화하고, 필요 시 동일 인터페이스로 RTE term만 add하도록 hook 제공. 

Heterogeneous Graph Transformer

Datasets & Tasks (제안)

공개: OGB-MAG 소규모 서브셋/DBLP 변형 등 (초기 디버깅) → 필요 시 대형으로 확장.

도메인: (네 프로젝트) bin–gene–pathway 이질 그래프 생성 → 그래프 수준 이진분류 혹은 노드/링크 예측.

Directory
.
├── data/
├── experiments/
│   ├── mag_small_nodecls.ipynb
│   ├── bin_gene_pathway_graphclf.ipynb
├── hgt/
│   ├── __init__.py
│   ├── layers.py        # HGTLayer (RTE 없는 버전)
│   ├── model.py         # HGTNet (스택, readout)
│   ├── sampling.py      # (옵션) HGSampling 스텁
│   ├── utils.py         # HeteroData 빌더, metric 등
├── scripts/
│   ├── train_nodecls.py
│   ├── train_graphclf.py
└── README.md
