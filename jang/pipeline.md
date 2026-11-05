## 1. 프로젝트 개요 및 확정 요구사항 (System Requirements)

본 에이전트는 영화, 드라마, 애니메이션 영상과 기본 대사 자막(SRT)을 입력받아 **전체 줄거리와 진행 상황을 내부적으로 트래킹**하면서, **화면 밖 화자(Off-screen Speaker)의 실제 정체(이름)를 표기**하고 **배경 효과음(SFX)을 문맥에 맞는 자연스러운 한국어 자막으로 묘사**하는 오프라인 일괄 처리(Batch Processing) 시스템입니다.

| 구분 | 확정된 설계 조건 | 파이프라인 아키텍처 반영 전략 | 
| ----- | ----- | ----- | 
| **1. 입력 데이터 (Case A)** | 원본 영상(`.mp4`), 단일 믹스 오디오(`.wav`), **타임스탬프와 익명 화자 ID(`SPEAKER_01` 등)가 이미 구분된 한국어 `.srt` 파일** | 별도의 음성 인식(ASR)이나 오디오 화자 분리(Speaker Diarization) 모델을 구동하지 않고, 제공된 SRT의 타임스탬프와 화자 ID를 기준점으로 삼아 연산 자원을 시각 추적과 문맥 추론에 집중. | 
| **2. 캐릭터 사전 정보** | **외부 사전 정보(배우 DB, 등장인물표) 일절 없음** | 영상 속 대사 문맥(호칭, 자기소개, 대화 흐름)과 시각적 얼굴 클러스터(`FACE_ID`)만을 결합하여 에이전트가 스스로 이름을 추론하고 매핑하는 **In-Context 2-Pass 아키텍처** 도입. | 
| **3. 하드웨어 및 모델** | **Google Colab Pro (A100 40GB VRAM)** / **100% 순수 오픈소스 모델** | 유료 외부 API 호출 배제. OOM(메모리 초과) 방지를 위해 각 단계별로 단일 모델만 로드 후 결과를 디스크에 저장하고 VRAM을 완전히 비우는 **순차 실행(Sequential Load-Unload) 프로토콜** 강제. | 
| **4. 효과음(SFX) 처리** | 음원 분리 필수 / 사전 정의된 분류가 아닌 **자유로운 문장 묘사(Audio Captioning)** | `Hybrid Demucs`로 대사와 배경음악(BGM)을 제거한 순수 효과음 트랙을 추출하고, 오픈소스 대형 오디오-언어 모델(`Qwen2-Audio`)로 1차 묘사 후 LLM이 줄거리 문맥에 맞게 다듬음. | 
| **5. 타깃 장르 및 확장성** | **1차: 실사 영화/드라마** $\rightarrow$ **2차: 애니메이션 확장** | 파이프라인의 뼈대(Phase 1, 3, 4)는 공통으로 유지하되, 도메인 차이가 발생하는 \*\*Phase 2(얼굴 탐지 및 입모양 화자 인식)\*\*만 설정 플래그(`domain_mode`)에 따라 교체 가능한 모듈형 플러그인으로 설계. | 

## 2. Colab Pro (A100 40GB) 메모리 관리 및 실행 프로토콜

장편 영화(90분\~150분)를 단일 Colab 세션에서 다운 없이 안정적으로 처리하기 위해 아래 4대 엔지니어링 규칙을 엄격히 적용합니다.

1. **Strict Sequential Execution (단계별 순차 로드 및 즉시 해제):**

   * 절대 두 개 이상의 무거운 딥러닝 모델(예: `Qwen2-Audio`와 `Qwen2.5-LLM`)을 동시에 GPU VRAM에 올리지 않습니다.

   * 각 Phase가 끝날 때마다 아래 메모리 초기화 루틴을 반드시 실행합니다.

     ```
     del model
     gc.collect()
     torch.cuda.empty_cache()
     
     ```

2. **Disk Checkpointing (디스크 기반 상태 전달):**

   * 각 Phase의 출력 결과는 파이썬 메모리(RAM) 변수로 넘기지 않고, 로컬 디스크(또는 마운트된 Google Drive)에 정형화된 `.json` 및 `.wav` 파일로 저장합니다. 런타임이 예기치 않게 재시작되더라도 완료된 Phase 다음부터 즉시 재개(Resume)할 수 있습니다.

3. **On-Demand Frame Decoding (온디맨드 프레임 추출):**

   * 2시간 분량의 비디오 프레임 전체를 System RAM에 올리면 즉시 크래시가 발생합니다. `decord` 또는 `PyAV` 라이브러리를 사용하여 SRT 대사 타임스탬프 구간(`[start_sec, end_sec]`)에 해당하는 프레임만 디스크에서 온디맨드로 시크(Seek)하여 읽어들입니다.

4. **Scene-Based Sliding Window (씬 단위 청크 처리):**

   * Phase 1에서 분할된 '씬(Scene, 평균 1\~4분 길이)'을 기본 처리 단위로 삼아 오디오 분석과 LLM 프롬프트 컨텍스트 길이를 일정하게 유지합니다.

## 3. 전체 파이프라인 아키텍처 (4-Phase Workflow)

```
[입력] movie.mp4 + movie_mix.wav + raw_anon.srt (타임스탬프 + SPEAKER_01 구분 완료)
  │
  ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ [Phase 1] 전처리: 의미 기반 씬 분할 및 오디오 음원 분리               ┃
┃  • 비전: TransNetV2 (샷 탐지) + BaSSL (의미적 씬 경계 그룹화)         ┃
┃  • 오디오: Hybrid Demucs (htdemucs_ft) -> 대사(vocals) / 효과음 분리  ┃
┃  • 출력: phase1_scenes.json, vocals.wav, sfx_residual.wav             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  │ (VRAM 초기화: del model -> empty_cache)
  ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ [Phase 2] 시각적 화자 추적 및 온/오프스크린 판별 (장르별 분기 모듈)   ┃
┃  • [실사 모드]: InsightFace (RetinaFace+ArcFace) + Light-ASD          ┃
┃  • [애니 모드]: YOLOv8-AnimeFace + DINOv2 Re-ID + 입 영역 Optical Flow┃
┃  • 핵심 로직: 각 SRT 대사 구간에서 화면 내 발화자 유무(is_offscreen)  ┃
┃               판별 및 [SPEAKER_ID ↔ FACE_ID] 동시 등장 빈도 누적      ┃
┃  • 출력: phase2_visual_tracking.json                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  │ (VRAM 초기화: del model -> empty_cache)
  ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ [Phase 3] 비음성 효과음(SFX) 이벤트 탐지 및 자유 문장 묘사            ┃
┃  • 구간 탐지: RMS 에너지 임계값 필터링 + Silero VAD (잔여 음성 배제)  ┃
┃  • 자유 묘사: Qwen2-Audio-7B-Instruct (오디오 클립 -> 자연어 서술)    ┃
┃  • 출력: phase3_sfx_captions.json                                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  │ (VRAM 초기화: del model -> empty_cache)
  ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ [Phase 4] 줄거리 트래킹, In-Context 캐릭터 매핑 및 SDH 자막 합성      ┃
┃  • 시각 요약(선택): Qwen2-VL-7B-AWQ (씬 대표 키프레임 상황 요약)      ┃
┃  • 추론 엔진: Qwen2.5-14B-Instruct-AWQ (또는 한국어 8B~14B 양자화 LLM)┃
┃  • Pass 1 (순방향): 씬별 줄거리 요약 갱신 + 대사 호칭/문맥 기반       ┃
┃                     전역 캐릭터 사전(SPEAKER_01 = 실명) 구축          ┃
┃  • Pass 2 (역방향): 초반 미식별 구간 실명 소급 적용(Backfill) +       ┃
┃                     효과음 문맥 정제(`[...]`) + 오프스크린 화자 기입  ┃
┃  • 최종 출력: final_sdh.srt, final_sdh_metadata.json                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

```

## 4. 단계별 상세 엔지니어링 설계 및 JSON 스키마

### \[Phase 0\] 초기 입력 데이터 규격 (`raw_anon.srt` $\rightarrow$ 내부 JSON 변환)

사용자가 제공하는 SRT 파일(타임스탬프 및 익명 화자 구분 완료)을 파싱하여 파이프라인의 기준 타임라인으로 삼습니다.

```
{
  "video_id": "movie_kr_01",
  "domain_mode": "live_action",
  "utterances": [
    {
      "utt_id": "UTT_0001",
      "start_sec": 12.40,
      "end_sec": 14.80,
      "speaker_id": "SPEAKER_01",
      "text": "지훈 선배! 거기 서 봐요, 할 말이 있어요."
    },
    {
      "utt_id": "UTT_0002",
      "start_sec": 15.50,
      "end_sec": 17.20,
      "speaker_id": "SPEAKER_02",
      "text": "지금은 바쁘니까 나중에 얘기하자, 수진아."
    }
  ]
}

```

### \[Phase 1\] 전처리: 씬(Scene) 분할 및 음원 분리

* **예상 VRAM 점유:** 약 6.0 GB (`Demucs` 및 `BaSSL` 순차 구동)

* **세부 동작:**

  1. **샷 및 씬 분할:** `TransNetV2`로 카메라 컷(Shot) 경계를 먼저 찾은 뒤, `BaSSL`을 통해 시각적·의미적으로 연결된 샷들을 묶어 씬(Scene) 구간(`SCENE_001`, `SCENE_002`...)을 생성합니다. 각 SRT 대사(`utt_id`)에 소속 `scene_id`를 부여합니다.

  2. **음원 분리:** `movie_mix.wav`를 `Hybrid Demucs (htdemucs_ft)`에 통과시켜 4개 스템(`vocals`, `drums`, `bass`, `other`)으로 분리합니다.

     * `vocals.wav`: Phase 2에서 입모양과 대사 음성의 동기화(ASD)를 정밀 측정할 때 사용합니다.

     * `sfx_residual.wav`: 배경음악 성분(`bass`, `melody`)과 대사(`vocals`)를 제거한 `other` 스템 중심의 트랙으로, Phase 3의 효과음 추출 소스로 사용합니다.

### \[Phase 2\] 시각적 화자 추적 및 온/오프스크린 판별 (`phase2_visual_tracking.json`)

* **예상 VRAM 점유:** 약 8.5 GB

* **핵심 과제 1 — 온/오프스크린(`is_offscreen`) 판정 규칙:**

  * 각 대사 구간 `[start_sec, end_sec]`의 비디오 프레임에서 얼굴 트랙을 검출하고 `Light-ASD`에 `(얼굴 입모양 시퀀스 + vocals.wav 해당 구간)`을 입력하여 발화 점수 `asd_score`($0.0 \sim 1.0$)를 계산합니다.

  * **판정 수식:**
    

    $$
    \text{is\_offscreen} = \begin{cases} \text{True}, & \text{if } N_{\text{faces}} = 0 \text{ or } \max_{i}(\text{asd\_score}_i) < \tau_{\text{asd}} \\ \text{False}, & \text{otherwise} \end{cases}
    $$

    
    *(실사 영화 기준 기본 임계값* $\tau_{\text{asd}} = 0.5$ *적용)*

  * 즉, 화면에 사람이 아예 없거나(배경 샷), 화면에 사람이 잡혔더라도 입을 다물고 듣기만 하는 리액션 샷(`max_asd_score < 0.5`)이면 `is_offscreen = true`가 되어 SDH 자막에 화자 이름을 표시해야 하는 대상이 됩니다.

* **핵심 과제 2 — 시각 얼굴 ID(`FACE_ID`)와 음성 화자 ID(`SPEAKER_ID`)의 교차 매핑 누적:**

  * 외부 인물 DB가 없으므로, 에이전트는 화면 안에서 입을 움직여 말할 때(`is_offscreen == false`) 해당 얼굴의 ArcFace 임베딩 클러스터(`FACE_02`)와 SRT의 화자(`SPEAKER_02`)를 연결하는 동시 발생 행렬(Co-occurrence Matrix)을 누적합니다.

  * 이를 통해 나중에 `FACE_02`가 말없이 화면에 등장하거나 `SPEAKER_02`가 화면 밖에서 말할 때 두 정체성을 하나로 묶어 추론할 수 있습니다.

* **애니메이션 확장 시 모듈 교체 (`domain_mode == "animation"`):**

  * 실사용 `Light-ASD` 대신 `YOLOv8-AnimeFace`로 캐릭터 얼굴을 잡고, 캐릭터 외형은 `DINOv2 / CLIP` 임베딩으로 클러스터링하며, 발화 여부는 대사 구간 동안 캐릭터 입 영역(Lower-face ROI)의 프레임 간 픽셀 변화량(Optical Flow / L1 Diff) 분산이 임계치를 넘는지로 판별합니다.

#### 📄 `phase2_visual_tracking.json` 출력 스키마 예시

```
{
  "scene_id": "SCENE_001",
  "utterances_tracked": [
    {
      "utt_id": "UTT_0001",
      "start_sec": 12.40,
      "end_sec": 14.80,
      "speaker_id": "SPEAKER_01",
      "text": "지훈 선배! 거기 서 봐요, 할 말이 있어요.",
      "visual_info": {
        "visible_face_ids": ["FACE_02"],
        "active_speaker_face_id": null,
        "max_asd_score": 0.11,
        "is_offscreen": true,
        "scene_visual_note": "FACE_02 sits alone looking back with mouth closed."
      }
    },
    {
      "utt_id": "UTT_0002",
      "start_sec": 15.50,
      "end_sec": 17.20,
      "speaker_id": "SPEAKER_02",
      "text": "지금은 바쁘니까 나중에 얘기하자, 수진아.",
      "visual_info": {
        "visible_face_ids": ["FACE_02"],
        "active_speaker_face_id": "FACE_02",
        "max_asd_score": 0.91,
        "is_offscreen": false,
        "scene_visual_note": "FACE_02 speaks directly on screen."
      }
    }
  ],
  "speaker_to_face_affinity": {
    "SPEAKER_02": { "primary_face_id": "FACE_02", "match_confidence": 0.94 }
  }
}

```

### \[Phase 3\] 효과음(SFX) 이벤트 탐지 및 자유 문장 묘사 (`phase3_sfx_captions.json`)

* **예상 VRAM 점유:** 약 16.5 GB (`Qwen2-Audio-7B-Instruct` BF16 구동 기준)

* **세부 동작:**

  1. **비묵음 음향 이벤트 구간 검출:** `sfx_residual.wav`에서 슬라이딩 윈도우로 RMS 에너지(dB)를 측정하여 임계값(예: `-35 dB`) 이상으로 소리가 튀는 구간(최소 0.5초\~최대 5.0초)을 잘라냅니다. 이때 `Silero VAD`를 역방향으로 적용해 미처 분리되지 않은 대사 잔여물이 섞인 구간은 배제합니다.

  2. **오디오 언어 모델을 통한 자유 묘사:** 추출된 오디오 클립을 `Qwen2-Audio-7B-Instruct`에 입력합니다.

     * **프롬프트 지시:** *"이 오디오 클립에서 발생하는 물리적 행동음, 사물 소리, 환경음을 구체적인 한국어 문장으로 묘사하시오. 배경음악이나 악기 연주에 대한 설명은 절대 출력하지 마시오."*

  3. **BGM 잔여 노이즈(Bleeding) 후처리 필터링:** 생성된 텍스트에 `'음악'`, `'멜로디'`, `'비트'`, `'악기'`, `'노래'` 등의 키워드가 포함되면 배경음악 잔향으로 간주하고 폐기합니다.

#### 📄 `phase3_sfx_captions.json` 출력 스키마 예시

```
{
  "sfx_events": [
    {
      "sfx_id": "SFX_001",
      "scene_id": "SCENE_001",
      "start_sec": 10.10,
      "end_sec": 11.80,
      "rms_db": -21.5,
      "raw_audio_caption": "복도에서 누군가 빠르게 달려오는 구두 발자국 소리가 울린다."
    },
    {
      "sfx_id": "SFX_002",
      "scene_id": "SCENE_001",
      "start_sec": 17.50,
      "end_sec": 18.90,
      "rms_db": -17.2,
      "raw_audio_caption": "무거운 철제 문이 강하게 닫히면서 둔탁한 쿵 소리가 난다."
    }
  ]
}

```

### \[Phase 4\] 줄거리 트래킹, In-Context 캐릭터 매핑 및 최종 SDH 합성

* **예상 VRAM 점유:** 약 12.0 GB \~ 19.5 GB (`Qwen2.5-14B-Instruct-AWQ` 단독 또는 `Qwen2-VL-7B-AWQ` 순차 활용)

* **외부 사전 정보 없는 2-Pass 추론 메커니즘 (*As the Story Unfolds* 기반):**

#### 🔹 Pass 1: 씬 단위 순차 줄거리 트래킹 및 전역 캐릭터 사전(`Character Registry`) 구축

Scene 1부터 마지막 Scene까지 순서대로 순회하며 LLM에 아래 컨텍스트를 주입합니다.

* **LLM 입력 컨텍스트:**

  1. `cumulative_plot_summary`: 직전 씬까지 누적된 전체 줄거리 및 인물 관계 요약본.

  2. `character_registry`: 현재까지 밝혀진 `SPEAKER_ID` $\rightarrow$ 실제 캐릭터 이름 매핑 사전.

  3. 현재 씬의 시각적 요약(필요시 `Qwen2-VL`이 추출한 씬 배경/상황 1줄 요약) + 현재 씬의 대사 목록(`phase2_visual_tracking.json`) + 현재 씬의 원시 효과음 목록(`phase3_sfx_captions.json`).

* **이름 추론(Name Resolution) 논리:**

  * 예: `UTT_0001`에서 `SPEAKER_01`이 \*"지훈 선배!"\*라고 부르고, 곧바로 `UTT_0002`에서 `SPEAKER_02`가 \*"나중에 얘기하자, 수진아"\*라고 응답함.

  * LLM은 대화의 호응 관계(Turn-taking & Vocative Cues)를 분석하여 **`SPEAKER_01 = 수진`**, `SPEAKER_02 = 지훈`임을 확정하고 `character_registry`에 신뢰도(Confidence) 및 근거 대사 ID와 함께 기록합니다.
    \*同时, 현재 씬에서 일어난 사건을 2\~3문장으로 요약하여 `cumulative_plot_summary`를 갱신합니다(이 줄거리 데이터는 자막 파일에 직접 출력되지 않고 다음 씬의 추론 정확도를 높이는 내부 상태로만 유지됨).

#### 🔹 Pass 2: 역방향 이름 소급 적용(Retroactive Backfilling) 및 한국어 SDH 포맷팅

영화 전체 씬에 대한 Pass 1 순회가 끝나면 완성된 `character_registry`를 가지고 전체 자막 리스트를 최종 정제합니다.

1. **초반 미식별 화자 소급 치환:** 영화 시작 후 10분 동안 `SPEAKER_01`의 이름이 안 나오다가 15분째에 '수진'임이 밝혀졌다면, Pass 2에서 1\~10분 구간의 오프스크린 대사에도 소급하여 `(수진)` 태그를 달아줍니다. (단, 범인의 정체 등 극적 반전을 위해 의도적으로 숨긴 목소리이거나 끝까지 이름이 안 나오는 단역은 LLM이 문맥에 맞춰 `(의문의 남자)`, `(택시 기사)` 등 역할명을 유지합니다.)

2. **온/오프스크린 화자 태그 부착:**

   * `is_offscreen == true`인 대사: 앞에 **`(화자이름) 대사`** 형태로 정체를 명시합니다.

   * `is_offscreen == false`인 대사: 화면에서 누가 말하는지 명확히 보이므로 화자 태그 없이 순수 대사만 출력합니다(단, 한 화면에 여러 명이 입을 가리고 있는 등 모호성이 높을 때만 예외적으로 표기).

3. **줄거리 문맥 기반 효과음(SFX) 자막 정제:**

   * `Qwen2-Audio`가 생성한 길고 서술적인 문장(*"무거운 철제 문이 강하게 닫히면서 둔탁한 쿵 소리가 난다"*)을 현재 줄거리 문맥(지훈이 수진의 대화를 거절하고 나가는 상황)에 맞춰 `[철문이 쿵 닫히는 소리]`처럼 15자 이내의 직관적인 명사형 대괄호 태그로 압축합니다.

#### 📄 `final_sdh_metadata.json` 및 최종 `.srt` 출력 예시

```
{
  "internal_plot_tracker_state": {
    "final_plot_summary": "수진이 복도에서 지훈을 붙잡고 대화를 요청하나 지훈은 냉정하게 거절하고 철문을 닫고 떠난다.",
    "resolved_character_registry": {
      "SPEAKER_01": {
        "resolved_name": "수진",
        "linked_face_id": "FACE_01",
        "evidence": "Called '수진아' by SPEAKER_02 in UTT_0002",
        "confidence": 0.98
      },
      "SPEAKER_02": {
        "resolved_name": "지훈",
        "linked_face_id": "FACE_02",
        "evidence": "Addressed as '지훈 선배' by SPEAKER_01 in UTT_0001",
        "confidence": 0.98
      }
    }
  },
  "final_sdh_subtitles": [
    {
      "index": 1,
      "timestamp": "00:00:10,100 --> 00:00:11,800",
      "type": "SFX",
      "sdh_text": "[다급한 구두 발자국 소리]"
    },
    {
      "index": 2,
      "timestamp": "00:00:12,400 --> 00:00:14,800",
      "type": "DIALOGUE_OFFSCREEN",
      "sdh_text": "(수진) 지훈 선배! 거기 서 봐요, 할 말이 있어요."
    },
    {
      "index": 3,
      "timestamp": "00:00:15,500 --> 00:00:17,200",
      "type": "DIALOGUE_ONSCREEN",
      "sdh_text": "지금은 바쁘니까 나중에 얘기하자, 수진아."
    },
    {
      "index": 4,
      "timestamp": "00:00:17,500 --> 00:00:18,900",
      "type": "SFX",
      "sdh_text": "[철문이 쿵 닫히는 소리]"
    }
  ]
}

```

## 5. 단계별 실험 및 구현 로드맵 (Implementation Roadmap)

1. **Step 1 (Phase 2 검증):** 짧은 영화 클립(5\~10분)과 익명 화자 SRT를 준비하여 `InsightFace` + `Light-ASD`를 구동하고, 리액션 샷과 화면 밖 목소리에서 `is_offscreen = True`가 정확히 분류되는지 임계값($\tau_{\text{asd}} = 0.5$)을 튜닝합니다.

2. **Step 2 (Phase 1 & 3 검증):** `Hybrid Demucs`로 분리한 효과음 트랙에서 에너지 임계값을 조절하여 배경음악 잔향(Bleeding)을 걸러내고, `Qwen2-Audio-7B`가 생성하는 원시 캡션의 품질을 확인합니다.

3. **Step 3 (Phase 4 통합):** 양자화 한국어 LLM(`Qwen2.5-14B-Instruct-AWQ` 등)에 씬 단위 슬라이딩 프롬프트를 적용하여 `SPEAKER_ID`가 극 중 실명으로 정확히 매핑되는지 확인하고 최종 SRT 병합 스크립트를 완성합니다.