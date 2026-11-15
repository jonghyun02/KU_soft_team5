# FigJam 붙여넣기용 MoS-SDH Studio 흐름도 모음

## 방법 1. 벡터 다이어그램 통째로 붙여넣기 (가장 깔끔함 · 추천)
1. 같은 폴더의 **`MoS-SDH_FigJam_흐름도.svg`** 파일을 열려 있는 **FigJam 보드 화면 위로 그대로 드래그 앤 드롭**하거나,
2. PowerShell에서 아래 한 줄을 실행해 클립보드에 복사한 뒤 FigJam 보드에서 **`Ctrl + V`** 를 누르면 모든 박스와 화살표, 텍스트가 깨짐 없이 편집 가능한 벡터 카드 그룹으로 즉시 붙여넣어집니다.

```powershell
Get-Content -Path ".\MoS-SDH_FigJam_흐름도.svg" -Raw -Encoding UTF8 | Set-Clipboard
```

---

## 방법 2. FigJam [Mermaid] 플러그인 / 위젯에 코드 붙여넣기
FigJam 상단/하단 툴바의 **Plugins/Widgets → `Mermaid`** 위젯을 열고 아래 코드를 그대로 붙여넣으면 FigJam 네이티브 자석 커넥터 도형으로 변환됩니다. (FigJam 파서 호환성을 위해 특수기호와 HTML 태그를 모두 안전하게 이스케이프했습니다.)

### 2-1. 전체 5-Stage 파이프라인 & 네이티브 앱 흐름도
```mermaid
flowchart TD
    EXE["MoS-SDH-Studio.exe (무콘솔 Win32 런처)"] --> QT["MoSSDHMainWindow (순수 Qt6 4분할 스튜디오)"]
    QT --> WORKER["PipelineWorker (QThread 비동기 GPU 실행)"]
    WORKER --> MODE{"기존 자막 파일(.SRT) 입력 여부"}

    MODE -- "없음 (Mode A)" --> MA["영상 단독 End-to-End 생성 모드"]
    MODE -- "있음 (Mode B)" --> MB["원음 STT 타임스탬프 정렬 및 공백정규화 sWER 진단"]

    MA --> DOC["중앙 데이터 버스: MoSDocument 초기화"]
    MB --> DOC

    DOC --> S1A["Stage 1-A/B: FFmpeg 추출 + Demucs 보컬/BGM 분리 + Silero VAD 화자분할"]
    S1A --> S1D["Stage 1-C/D: Faster-Whisper 단어 정렬 + Parselmouth 중앙값/MAD 강조 토큰화"]

    DOC --> S2A["Stage 2-A: PySceneDetect 씬 분할 + YuNet 구순 움직임 ASD + ViT 연속 표정"]
    S2A --> S2B["Stage 2-B: EasyOCR 화면 텍스트 스캔 및 하단 충돌 판정 (y_max 0.78 이상)"]
    S2B --> S3["Stage 3: 호칭 단서 휴리스틱 + 로컬 4-bit Qwen2.5-VL 인물 6화계 관계 그래프"]

    S1D --> S4A["Stage 4-A: Kiwi 3축 경어 및 6화계 파싱 (K=3000ms 담화 문맥 대조)"]
    S3 --> S4A
    S4A --> HON_DEC{"기대 화계와 불일치 발생?"}
    HON_DEC -- "일치" --> HON_OK["문맥 일치 (종결어미 유지)"]
    HON_DEC -- "불일치" --> AROUSAL_DEC{"고각성(Z 1.35 이상) 또는 강한 강조?"}
    AROUSAL_DEC -- "예" --> SHIFT["의도적 극적 전환 판정 -> (격앙되어 반말로) 지문 부여"]
    AROUSAL_DEC -- "아니오" --> CAPE["단순 오인식 판정 -> Kiwi 형태소 재결합 CAPE 어미 자동 교정"]

    S1D --> CMGI["Stage 4-B: Trimodal CmGI 음향/어휘/표정 시계열 게이트 융합"]
    S3 --> CMGI

    CMGI --> IRONY_DEC{"어휘 극성 x 시청각 극성 부호 반대 및 반어 점수 0.55 이상?"}
    IRONY_DEC -- "예 (반어)" --> IRONY_ON["반어법 감지 -> (비꼬는 어조로) 지문 강제 삽입"]
    IRONY_DEC -- "아니오 (일반)" --> IRONY_OFF["4-C: 주류 정서 및 비유창성/CPS/5구간 분포 산출"]

    HON_OK --> S5A["Stage 5-A: 위치 태그 3후보 + 로컬 4-bit LLM 중괄호 재작성 후보 생성"]
    SHIFT --> S5A
    CAPE --> S5A
    IRONY_ON --> S5A
    IRONY_OFF --> S5A

    S5A --> S5B["Stage 5-B: SentenceTransformer 의미 보존도 C_i 가드 및 조화평균 O_i 1위 채택"]
    S5B --> S5C["Stage 5-C: 인지 피로 과잉 지문 억제기 (반어/극적전환/음향 보호, 경미 지문 45% 상한)"]

    S5C --> OUT_SRT["표준 지문 자막 (.SRT / .VTT)"]
    S5C --> OUT_ASS["시각 스타일 자막 (.ASS: 화자 색상 + 강조 크기 + 하단 OCR 충돌 시 상단 배치)"]
    S5C --> OUT_REP["정량 진단 리포트 (.MD / .JSON)"]

    OUT_SRT --> HITL["Qt6 스튜디오 실시간 검수 (원클릭 후보 교체 / 화자명 일괄 변경 / 억제 슬라이더)"]
    OUT_ASS --> HITL
    OUT_REP --> HITL
    HITL -. "편집 즉시 CPS 재계산 및 5종 파일 자동 동기화" .-> S5C
```
