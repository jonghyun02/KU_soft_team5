# SDH 자막 생성 에이전트를 위한 비디오·오디오 이해 핵심 논문 총람 (2020–2026)

본 문서는 영화, 드라마, 애니메이션의 전체 줄거리와 진행 상황을 추적하며 한국어 배리어프리(SDH) 자막을 자동 생성하는 AI 에이전트 구축을 위해 엄선된 핵심 논문 목록입니다. 2020년대의 기반 모델(Foundational Models)부터 **2026년 기준 최신 연구(State-of-the-Art)**까지 파이프라인의 4대 핵심 모듈별로 분류하여 정리했습니다.

---

## 1. 씬 단위 영상 분할 (Video Scene Segmentation)

단순한 카메라 컷(Shot) 변환을 넘어, 동일한 시공간과 서사적 맥락(Narrative Context)을 공유하는 '씬(Scene)' 단위로 장편 영상을 분할하기 위한 기술입니다. LLM의 줄거리 트래킹 컨텍스트 윈도우를 구성하는 기본 단위가 됩니다.

| 연도 / 학회 | 논문 제목 | 핵심 기법 및 아키텍처 | SDH 파이프라인 적용 포인트 |
| :--- | :--- | :--- | :--- |
| **2020**<br>*(ArXiv)* | **TransNetV2: An Effective Deep Network Architecture for Fast Shot Transition Detection** | 3D CNN 기반의 초고속 샷 경계 탐지(Shot Boundary Detection) 네트워크. 하드 컷(Hard Cut)과 디졸브 등 점진적 전환을 모두 탐지. | 씬 분할 모델(BaSSL 등)의 입력 단위가 되는 기본 '샷(Shot)'을 추출하는 1차 전처리 도구로 사용. |
| **2021**<br>*(CVPR)* | **ShotCoL: Shot Contrastive Self-Supervised Learning for Scene Boundary Detection** | 인접한 샷들 간의 시각적·의미적 유사성을 대조 학습(Contrastive Learning)하여 사전 라벨링 없이도 씬의 경계를 표현하는 특징 벡터를 학습. | 영화나 드라마에서 장소나 시간대가 바뀌는 지점을 시각적 특징만으로 군집화할 때 활용. |
| **2022**<br>*(ACCV)* | **BaSSL: Boundary-aware Self-Supervised Learning for Video Scene Segmentation** | 샷 시퀀스를 슬라이딩 윈도우 방식으로 탐색하며, 다중 모달리티(시각+오디오)에서 의미적 전환(Semantic Transition)이 극대화되는 경계점을 탐지. | Colab 환경에서 오픈소스로 즉시 구동 가능한 가장 안정적인 베이스라인 씬 분할 모델. |
| **2026**<br>*(ICLR)* | **Video Scene Segmentation with Genre and Duration Signals** | 영상의 장르적 특성(액션, 드라마, 애니메이션 등)과 씬의 지속 시간 분포(Duration Prior)를 트랜스포머 디코더의 조건부 신호로 주입하여 장편 영상의 과분할(Over-segmentation)을 방지. | **장르별 파라미터 분리 전략에 핵심 참고.** 실사 영화와 호흡이 빠른 애니메이션의 씬 분할 임계값을 다르게 설정할 수 있는 이론적·구조적 근거 제공. |

---

## 2. 화자 인식 및 온/오프스크린 판별 (Active Speaker Detection)

대사 타임스탬프 구간에서 화면 속 인물이 실제로 입을 움직여 발화하고 있는지(On-screen) 아니면 화면 밖에서 목소리만 들리는지(Off-screen)를 판별하기 위한 오디오-비주얼 융합 기술입니다.

| 연도 / 학회 | 논문 제목 | 핵심 기법 및 아키텍처 | SDH 파이프라인 적용 포인트 |
| :--- | :--- | :--- | :--- |
| **2021**<br>*(ACM MM)* | **TalkNet: Is Someone Speaking? Exploring Long-Term Temporal Features for Audio-Visual Active Speaker Detection** | 오디오 프레임과 크롭된 얼굴(입모양) 시퀀스를 각각 인코딩한 뒤, Cross-Attention과 Self-Attention으로 장단기 시계열 동기화 여부를 분석. | 가장 표준적인 ASD 베이스라인 모델. 오픈소스 가중치가 공개되어 있어 실사 영화에 즉시 적용 가능. |
| **2023**<br>*(CVPR)* | **Light-ASD: A Light-Weight Model for Active Speaker Detection** | 기존 3D CNN과 무거운 어텐션 구조를 경량화된 분리형 컨볼루션(Separable Convolution)과 GRU로 대체하여 연산량(FLOPs)을 90% 이상 절감. | **Colab A100 고속 처리용 핵심 모델.** 2시간 분량의 장편 영화를 오프라인 일괄 처리할 때 추론 속도를 획기적으로 단축. |
| **2024**<br>*(ICASSP)* | **TalkNCE: Improving Active Speaker Detection with Talk-Aware Contrastive Learning** | 발화 중인 얼굴-오디오 쌍과 비발화 쌍 간의 대조 손실(Contrastive Loss)을 도입하여, 여러 명이 동시에 화면에 잡힐 때 화자 구분 능력을 극대화. | 다수의 등장인물이 한 화면에서 대화하는 드라마 씬에서 정확한 발화자를 지목할 때 유용. |
| **2026**<br>*(Interspeech)* | **Revisiting Active Speaker Detection: An In-the-Wild Benchmark for Generalization and Robustness** | 기존 AVA-ActiveSpeaker 데이터셋의 편향성을 지적하고, 조명 변화, 측면 얼굴, 배경 소음이 심한 야생(In-the-Wild) 환경에서의 일반화 성능을 평가 및 개선. | 음원 분리 전 오디오 노이즈가 섞여 있거나 역광/측면 샷이 많은 최신 영화에서 오탐지율을 줄이는 후처리 기법 참고. |
| **2026**<br>*(Preprint)* | **ROAM-ASD: Robust Open-World Active Speaker Detection with Flexible Multimodal Fusion** | 인물의 입이 가려지거나(Occlusion) 뒷모습만 나오는 등 시각 정보가 결측된 상황에서, 신체의 미세한 움직임과 오디오 공간감을 유연하게 융합하여 발화 여부를 추론. | **오프스크린 판별 고도화.** 화면에 인물이 등지고 서 있을 때 이를 단순 Off-screen으로 처리할지, 화면 내 인물의 발화로 볼지 결정하는 휴리스틱 설계에 활용. |

---

## 3. 음원 분리 및 효과음 자유 묘사 (Source Separation & Audio Captioning)

단일 `.wav` 믹스 트랙에서 대사(Vocals)와 배경음악(BGM)을 제거하여 순수 효과음(SFX) 트랙을 추출하고, 이를 사전 정의된 클래스가 아닌 문맥에 맞는 자연어 문장으로 묘사하는 기술입니다.

| 연도 / 학회 | 논문 제목 | 핵심 기법 및 아키텍처 | SDH 파이프라인 적용 포인트 |
| :--- | :--- | :--- | :--- |
| **2023**<br>*(ICASSP)* | **Hybrid Demucs: Hybrid Transformers for Music Source Separation** | 시간 도메인(Waveform)과 주파수 도메인(Spectrogram)의 U-Net 구조를 결합하고 중심부에 Cross-domain Transformer를 배치해 음원 분리 성능을 극대화. | 단일 `.wav`에서 대사와 음악을 걷어내고 효과음(Others) 스템만 추출하는 Phase 1의 핵심 음원 분리 엔진. |
| **2024**<br>*(Tech Report)* | **Qwen2-Audio Technical Report** | 대규모 오디오 인코더와 Qwen LLM을 결합한 오픈소스 대형 오디오-언어 모델(LALM). 음성 인식뿐만 아니라 환경음 및 효과음에 대한 자유로운 자연어 질의응답(Audio QA) 및 캡셔닝 지원. | Phase 3에서 추출된 효과음 구간을 입력받아 1차 자연어 묘사(예: "유리창이 깨지며 파편이 튀는 소리")를 생성하는 메인 모델. |
| **2025**<br>*(ICASSP)* | **SLAM-AAC: Enhancing Audio Captioning with Paraphrasing Augmentation and CLAP-Refine through LLMs** | CLAP(Contrastive Language-Audio Pretraining) 임베딩을 활용해 오디오와 텍스트 묘사 간의 정합성을 검증하고, LLM을 통해 다양한 어휘로 캡션을 정제. | 오디오 모델이 생성한 부정확하거나 단조로운 효과음 묘사를 한국어 SDH 자막 스타일(예: `[날카로운 파열음]`)로 다듬는 프롬프트 엔지니어링에 참조. |
| **2026**<br>*(Interspeech)* | **The Interspeech 2026 Audio Encoder Capability Challenge for Large Audio Language Models (XARES-LLM)** | 대형 오디오 언어 모델(LALM)에서 중첩된 소리(Overlapping Acoustic Events)와 세밀한 시간적 이벤트(Fine-grained Temporal Events)를 구분하는 오디오 인코더 벤치마크. | 복합적인 효과음(예: 빗소리 중에 들리는 발자국 소리)을 시간순으로 분리하여 캡셔닝할 때 오디오 청크 분할 주기를 설정하는 기준으로 활용. |

---

## 4. 줄거리 트래킹 및 등장인물 이름 매핑 (Contextual Video Understanding & Character Naming)

시각 정보, 익명 대사(`Speaker_A`), 효과음 묘사를 통합하여 전체 줄거리를 추적하고, 문맥 추론을 통해 익명 화자 ID를 실제 극 중 캐릭터 이름(예: `성기훈`)으로 매핑하는 에이전트의 두뇌 역할 기술입니다.

| 연도 / 학회 | 논문 제목 | 핵심 기법 및 아키텍처 | SDH 파이프라인 적용 포인트 |
| :--- | :--- | :--- | :--- |
| **2020**<br>*(ECCV)* | **MovieNet: A Holistic Dataset for Movie Understanding** | 영화 이해를 위한 대규모 데이터셋 및 벤치마크. 씬 분할, 시놉시스 정렬, 인물 재식별(Character Re-ID), 행동 인식을 하나의 통합된 프레임워크로 정의. | 프로젝트 전체의 데이터 흐름과 영화 메타데이터 구조(JSON 스키마)를 설계하는 데 가장 근본이 되는 레퍼런스. |
| **2024**<br>*(CVPR)* | **AutoAD-III: The Prequel - Back to the Pixels for Context-Aware Audio Description** | 장편 영화에서 시각장애인용 화면해설(AD)을 생성하기 위해, 이전 씬들의 히스토리 문맥과 현재 픽셀 정보를 결합하여 인물의 행동과 상황을 서술. | 이전 씬의 줄거리 요약본을 다음 씬의 컨텍스트로 넘기는 슬라이딩 윈도우(Sliding Context) 설계의 원형. |
| **2024**<br>*(ACCV)* | **AutoAD-Zero: A Training-Free Framework for Zero-Shot Audio Description** | **별도의 모델 파인튜닝 없이(Training-Free)**, 상용/오픈소스 VLM과 LLM만을 파이프라인으로 엮어 인물을 식별하고 문맥에 맞는 화면 해설을 생성하는 제로샷 아키텍처. | **본 프로젝트 Phase 4의 핵심 설계도.** Colab 환경에서 추가 학습 없이 오픈소스 모델들의 조합만으로 캐릭터 사전을 구축하는 프롬프트 구조 제공. |
| **2025**<br>*(ArXiv)* | **Character-Centric Understanding of Animated Movies** | 실사 영화와 다른 2D/3D 애니메이션 캐릭터의 비정형적 외형과 과장된 표현을 처리하기 위해 애니메이션 도메인에 특화된 인물 클러스터링 및 감정/행동 분석 수행. | 향후 파이프라인을 **애니메이션 장르로 확장할 때**, 실사용 얼굴 인식(InsightFace)을 대체할 CLIP/DINOv2 기반 캐릭터 클러스터링 전략을 수립하는 데 필수적. |
| **2025**<br>*(ArXiv)* | **Shot-by-Shot: Film-Grammar-Aware Training-Free Audio Description Generation** | 영화 문법(클로즈업, 오버더숄더 샷, 컷어웨이 등)을 분석하여 현재 샷에서 가장 중요한 서사적 초점이 무엇인지 LLM에게 알려주는 프레임워크. | 화면 밖 화자(Off-screen)가 발생할 때, 이것이 단순한 편집 기법(예: 리액션 샷)인지 실제 공간 밖의 목소리(예: 전화 통화, 환청)인지 구분하는 단서 제공. |
| **2026**<br>*(ArXiv)* | **OmniScript: Towards Audio-Visual Script Generation for Long-Form Cinematic Video** | 장편 영화의 비디오, 오디오, 대사를 통합 분석하여 지문(Scene Heading), 인물 대사, 음향 효과가 모두 포함된 종합 시나리오 스크립트를 역생성. | 대사 JSON과 효과음 캡션을 시간축(Timeline) 상에서 충돌 없이 하나의 완결된 SDH 자막 스크립트로 병합하는 후처리 로직에 참고. |
| **2026**<br>*(COLM)* | **REFRAMED: Towards Realistic Audio Description Generation for Movies** | LLM이 영상 설명을 생성할 때 발생하는 환각(Hallucination)과 과도한 추측성 서술을 억제하고, 실제 방송 표준 가이드라인에 부합하는 간결하고 객관적인 문장을 생성하도록 제어. | 효과음 묘사나 화자 상태를 자막으로 출력할 때, 지나치게 문학적이거나 긴 문장 대신 `[문 닫히는 소리]`, `(다급하게)`처럼 직관적인 SDH 표준 포맷으로 제약하는 기법. |
| **2026.10**<br>*(ArXiv)* | **As the Story Unfolds: Watching a Film and Identifying Characters as a Human Does** | **외부 배우 데이터베이스(IMDb 등)에 전혀 의존하지 않고**, 인간이 영화를 보며 대사 속 호칭("철수야!")과 시각적 등장을 연결해 점진적으로 인물의 이름을 학습하듯 캐릭터를 식별하는 에이전트 연구. | **Phase 4 캐릭터 이름 매핑의 핵심 알고리즘.** 사용자가 제공하는 익명 화자 JSON(`Speaker_A`)을 대사 문맥만으로 실제 캐릭터 이름과 매핑하고, 초반 미식별 구간을 해결하는 2-Pass 구조의 이론적 기반. |

---

## 5. 논문 리딩 추천 순서 (구현 로드맵 기준)

1. **전체 아키텍처 및 프롬프트 설계:** `AutoAD-Zero (2024)` $\rightarrow$ `As the Story Unfolds (2026)`
2. **시각적 화자 추적 및 씬 분할 구현:** `Light-ASD (2023)` $\rightarrow$ `BaSSL (2022)` $\rightarrow$ `Video Scene Segmentation with Genre Signals (2026)`
3. **오디오 분리 및 효과음 캡셔닝 구현:** `Hybrid Demucs (2023)` $\rightarrow$ `Qwen2-Audio (2024)` $\rightarrow$ `REFRAMED (2026)`