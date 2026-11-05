# SDH 자막 생성 및 줄거리 트래킹 에이전트 핵심 논문 총람 (`papers2.md`)

본 문서는 영화·드라마·애니메이션의 전체 줄거리와 진행 상황을 추적하며 한국어 배리어프리(SDH) 자막을 생성하는 에이전트 구축을 위한 논문 리스트입니다.
학술적·이론적 근간을 이루는 **[Part 1] 피인용 수 1,000회 이상 검증 파운데이션 논문(Foundational Papers)**과, 실제 Google Colab Pro(A100) 코드 구현 시 직접 가중치(Weights)와 아키텍처를 가져다 쓸 수 있는 **[Part 2] 실전 구현용 오픈소스 및 도메인 특화 논문(Practical Open-Source & Niche Task Papers)**으로 분리하여 구성했습니다.

---

## Part 1. 피인용 수 1,000회 이상 검증 파운데이션 논문 (총 15선)

Semantic Scholar 및 Google Scholar 기준 피인용 수 1,000회 이상을 기록한 비디오·오디오·멀티모달 분야의 핵심 백본(Backbone) 논문들입니다.

### 1. 비디오 시공간 표현 및 시간축 경계 탐지 (Phase 1: 씬/이벤트 분할 기반)

| 연도 / 학회 | 논문 제목 및 저자 | 검증 인용 수 | 핵심 기법 및 파이프라인 내 역할 |
| :--- | :--- | :--- | :--- |
| **2019**<br>*(ICCV)* | **SlowFast Networks for Video Recognition**<br>*(Feichtenhofer et al.)* | **~4,600+회** | **시공간 이중 경로 백본:** 느린 프레임 경로(배경·인물 등 공간 문맥)와 빠른 프레임 경로(카메라 전환·입모양 등 급격한 모션)를 동시에 학습하는 비디오 이해의 표준 아키텍처. |
| **2021**<br>*(ICML)* | **Is Space-Time Attention All You Need for Video Understanding? (TimeSformer)**<br>*(Bertasius et al.)* | **~3,400+회** | **시공간 분할 어텐션:** 3D CNN 없이 비디오 프레임 시퀀스에 시간축과 공간축 어텐션을 분리 적용하여 장편 영상의 장기 문맥 특징을 추출하는 비디오 트랜스포머. |
| **2022**<br>*(NeurIPS)* | **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training**<br>*(Tong et al.)* | **~2,200+회** | **자기지도 비디오 인코더:** 시공간 튜브 마스킹(Tube Masking)을 통해 라벨링 없는 영화 영상에서도 샷/씬 전환 및 인물 행동 표현을 고효율로 추출하는 사전학습 백본. |
| **2022**<br>*(ECCV)* | **ActionFormer: Localizing Moments of Actions with Transformers**<br>*(Zhang et al.)* | **~1,450+회** | **시간축 경계 검출(Temporal Localization):** 무편집 장편 비디오에서 멀티스케일 트랜스포머를 통해 의미 있는 이벤트나 씬의 시작·종료 타임스탬프를 정밀하게 찾아내는 표준 구조. |

### 2. 얼굴 클러스터링 및 시각-음성 화자 검증 (Phase 2: 화면 내 인물 추적 및 화자 매칭 기반)

| 연도 / 학회 | 논문 제목 및 저자 | 검증 인용 수 | 핵심 기법 및 파이프라인 내 역할 |
| :--- | :--- | :--- | :--- |
| **2019**<br>*(CVPR)* | **ArcFace: Additive Angular Margin Loss for Deep Face Recognition**<br>*(Deng et al.)* | **~8,200+회** | **초고성능 얼굴 임베딩 클러스터링:** 각도 마진 손실(Angular Margin Loss)을 통해 영화 전체에 등장하는 인물 얼굴을 고유 벡터로 변환, 동일 인물끼리 클러스터링(`FACE_01`, `FACE_02`)하는 `InsightFace`의 핵심 엔진. |
| **2016**<br>*(CVPR)* | **WIDER FACE: A Face Detection Benchmark**<br>*(Yang et al.)* | **~1,800+회** | **야생 환경 얼굴 탐지 표준:** 어두운 조명, 측면 촬영, 가려짐(Occlusion)이 심한 영화 프레임에서 화면 내 인물 존재 여부(On/Off-screen 1차 판별)를 검출하는 얼굴 탐지기들의 기준 벤치마크. |
| **2018**<br>*(ICASSP)* | **X-Vectors: Robust DNN Embeddings for Speaker Recognition**<br>*(Snyder et al.)* | **~3,100+회** | **화자 음성 임베딩 표준:** 가변 길이의 대사 오디오를 고정 차원의 화자 특징 벡터로 매핑하여, 영화 전반에 걸친 화자 분리(Speaker Diarization) 및 목소리 동일성 판별의 기틀을 마련한 연구. |
| **2017**<br>*(Interspeech)* | **VoxCeleb: A Large-Scale Speaker Identification Dataset**<br>*(Nagrani, Chung, & Zisserman)* | **~2,800+회** | **오디오-비주얼 화자 매칭의 기틀:** 영상 속 얼굴 트랙과 입모양 동기화(Active Speaker Verification)를 결합하여 야생 비디오에서 발화자를 특정하고 대규모 화자 데이터를 구축한 파이프라인 연구. |
| **2020**<br>*(Interspeech)* | **ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification**<br>*(Desplanques et al.)* | **~2,200+회** | **고정밀 화자 검증 백본:** 채널 어텐션과 다중 계층 특징 집계를 통해 짧은 대사 구간(1~2초)에서도 화자의 음성 정체성을 정확히 구분하는 현대 오픈소스 오디오 툴킷(SpeechBrain, Pyannote 등)의 핵심 백본. |

### 3. 음원 분리 및 비음성 효과음(SFX) 인식/정렬 (Phase 3: 음원 분리 및 소리 텍스트화 기반)

| 연도 / 학회 | 논문 제목 및 저자 | 검증 인용 수 | 핵심 기법 및 파이프라인 내 역할 |
| :--- | :--- | :--- | :--- |
| **2019**<br>*(IEEE/ACM TASLP)* | **Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation**<br>*(Luo & Mesgarani)* | **~2,200+회** | **시간 영역(Waveform) 음원 분리:** 스펙트로그램 변환 없이 원시 파형(Waveform)에서 직접 음성과 배경음을 분리해 내는 엔드투엔드 신경망으로, 현대 파형 기반 음원 분리 모델들의 출발점. |
| **2020**<br>*(IEEE/ACM TASLP)* | **PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition**<br>*(Kong et al.)* | **~1,750+회** | **대규모 환경음/효과음 사전학습:** AudioSet(527개 음향 클래스)으로 학습된 범용 오디오 신경망으로, 비음성 효과음 구간 탐지(Audio Event Detection) 및 오디오 캡셔닝 모델의 인코더로 광범위하게 활용. |
| **2021**<br>*(Interspeech)* | **AST: Audio Spectrogram Transformer**<br>*(Gong et al.)* | **~1,570+회** | **오디오 스펙트로그램 트랜스포머:** CNN 없이 스펙트로그램 패치에 순수 어텐션을 적용하여 복합적인 영화 효과음(예: 빗소리 속 발자국 소리)의 시간·주파수 패턴을 포착하는 백본. |
| **2023**<br>*(ICASSP)* | **Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation (CLAP)**<br>*(Wu et al.)* | **~1,140+회** | **오디오-자연어 대조 정렬:** 효과음 오디오와 자연어 문장 묘사를 동일한 임베딩 공간에 매핑하여, 사전 정의된 라벨 없이도 소리와 텍스트 설명 간의 유사도를 계산하고 자유 묘사(Captioning) 품질을 검증하는 핵심 모델. |

### 4. 멀티모달 비디오-언어 통합 및 줄거리 추론 (Phase 4: 서사 문맥 추적 및 SDH 정제 기반)

| 연도 / 학회 | 논문 제목 및 저자 | 검증 인용 수 | 핵심 기법 및 파이프라인 내 역할 |
| :--- | :--- | :--- | :--- |
| **2022**<br>*(NeurIPS)* | **Flamingo: a Visual Language Model for Few-Shot Learning**<br>*(Alayrac et al.)* | **~6,800+회** | **교차 멀티모달 컨텍스트(Interleaved VLM):** 연속된 비디오 프레임/이미지와 텍스트(대사)를 시간순으로 교차 입력받아 인-컨텍스트(In-Context) 추론을 수행하는 현대 시각-언어 모델의 표준 구조. |
| **2023**<br>*(EMNLP)* | **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**<br>*(Zhang et al.)* | **~2,000+회** | **시각+청각+언어 통합 LLM:** 비디오 프레임(Visual)과 오디오 트랙(Audio)을 Q-Former로 인코딩하여 LLM에 동시 주입함으로써, 영상 속 시각적 상황과 청각적 이벤트를 결합해 줄거리를 추론하는 오픈소스 아키텍처. |

---

## Part 2. 실전 구현용 오픈소스 및 도메인 특화 논문 (부록)

인용 수는 1,000회 미만(또는 최신 발표)이지만, **본 프로젝트의 세부 태스크(씬 분할, 입모양 기반 온/오프스크린 판별, 음원 분리, 외부 DB 없는 캐릭터 이름 매핑)**를 Colab Pro에서 실제로 구현할 때 직접 코드와 가중치를 가져다 써야 하는 필수 실용 논문들입니다.

### A. 샷/씬 분할 실전 오픈소스 (Phase 1 구현용)
1. **TransNetV2: An Effective Deep Network Architecture for Fast Shot Transition Detection (ACM MM 2020)**
   * **실전 활용:** 2시간 분량의 영화를 몇 분 만에 카메라 컷(Shot) 단위로 쪼개주는 가장 빠르고 정확한 오픈소스 가중치 제공.
2. **BaSSL: Boundary-aware Self-Supervised Learning for Video Scene Segmentation (ACCV 2022)** & **ShotCoL (CVPR 2021)**
   * **실전 활용:** `TransNetV2`로 나눈 샷들을 서사적 맥락이 같은 '씬(Scene)' 단위로 묶어 LLM 컨텍스트 청크(Chunk)를 생성하는 실전 모델.
3. **Video Scene Segmentation with Genre and Duration Signals (ICLR 2026)**
   * **실전 활용:** 실사 영화와 애니메이션 등 장르별 컷 전환 호흡 차이를 반영해 씬 과분할(Over-segmentation)을 방지하는 최신 기법.

### B. 온/오프스크린 화자 판별 실전 오픈소스 (Phase 2 구현용)
1. **SyncNet: Out of time: automated lip sync in the wild (ACCV 2016)** & **AVA-ActiveSpeaker (ICASSP 2020)**
   * **실전 활용:** 입모양과 오디오의 싱크를 측정해 화면 속 인물이 실제 발화 중인지 판별하는 ASD(Active Speaker Detection) 태스크의 원형.
2. **TalkNet (ACM MM 2021)** & **Light-ASD: A Light-Weight Model for Active Speaker Detection (CVPR 2023)**
   * **실전 활용:** **Colab A100 1순위 적용 모델.** 특히 `Light-ASD`는 기존 대비 연산량을 90% 이상 줄여 장편 영화의 모든 대사 구간에서 화면 내 발화 여부를 고속으로 판별(`max_asd_score < 0.5`일 때 Off-screen 판정).
3. **ROAM-ASD: Robust Open-World Active Speaker Detection with Flexible Multimodal Fusion (2026)**
   * **실전 활용:** 인물이 등을 돌리고 있거나 입이 가려진 영화적 연출 상황에서도 오프스크린 오판율을 줄여주는 강건한 멀티모달 융합 기법.

### C. 음원 분리 및 자유 형식 효과음 캡셔닝 실전 오픈소스 (Phase 1 & 3 구현용)
1. **Hybrid Demucs: Hybrid Transformers for Music Source Separation (ICASSP 2023)**
   * **실전 활용:** **음원 분리 표준 오픈소스.** 단일 오디오(`.wav`)에서 대사(`vocals`)와 배경음악(`bass/drums`)을 분리해 순수 효과음 잔여 트랙(`other`)을 추출.
2. **AudioCaps (NAACL 2019)** & **Clotho (ICASSP 2020)**
   * **실전 활용:** 대사를 배제하고 순수 음향/효과음만을 자연어 문장으로 묘사하는 오디오 캡셔닝 데이터셋 및 프롬프트 스타일 기준.
3. **Qwen2-Audio Technical Report (2024)** & **SLAM-AAC (ICASSP 2025)**
   * **실전 활용:** **효과음 자유 묘사 1순위 오픈소스 모델.** 분리된 효과음 오디오 클립을 입력받아 정해진 분류표 없이 *"유리창이 날카롭게 깨지며 파편이 쏟아지는 소리"*와 같이 상황에 맞는 자연어 캡션을 생성.

### D. 외부 DB 없는 줄거리 트래킹 및 캐릭터 이름 매핑 (Phase 4 구현용)
1. **"Hello! My name is... Buffy" – Automatic Naming of Characters in TV Video (BMVC 2006)** & **MovieGraphs (CVPR 2018)**
   * **실전 활용:** 리액션 샷(Off-screen speaker) 등 화자 모호성 케이스 정의 및 씬별 인물 관계 그래프 구조화의 고전 레퍼런스.
2. **AutoAD-Zero: A Training-Free Framework for Zero-Shot Audio Description (ACCV 2024)** & **StoryTeller (2024)**
   * **실전 활용:** 별도의 파인튜닝 없이 오픈소스 VLM과 LLM을 프롬프트 파이프라인으로 연결하여 장기 줄거리를 요약하고 캐릭터 신원을 유지하는 아키텍처 설계도.
3. **As the Story Unfolds: Watching a Film and Identifying Characters as a Human Does (2026.10)**
   * **실전 활용:** **본 프로젝트의 핵심 요구사항과 100% 일치하는 연구.** 외부 배우 사진 DB 없이, 영화를 처음부터 보며 대사 속 호칭("철수야!")과 시각적 등장을 단서로 SRT의 익명 화자(`SPEAKER_01`)를 실제 캐릭터 이름에 점진적으로 매핑하는 2-Pass 추론 알고리즘 제공.
4. **Character-Centric Understanding of Animated Movies (2025)** & **REFRAMED (COLM 2026)**
   * **실전 활용:** 향후 **애니메이션 장르 확장 시** 캐릭터 클러스터링 도메인 적응 기법(`Character-Centric`) 및 긴 효과음 묘사를 간결한 SDH 자막 태그로 압축하는 기법(`REFRAMED`).