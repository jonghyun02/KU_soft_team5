**대본·발화·시간 정렬 형식 사전 조사 (WebSearch 9회 사용)**

결론: 대본 구조, 실제 발화, 시간을 한 형식에 모두 담는 표준은 없습니다. 연구 데이터셋들은 정렬에 실패한 부분을 대부분 기록하지 않고 버렸습니다. 대본과 받아쓰기의 불일치를 데이터로 남기는 쪽은 ASR 도구들(Kaldi ctm-edits, Gentle)이었습니다.

**1) 형식별 비교** (O 지원, △ 부분 지원, X 미지원)

| 형식 | 단위 | 시간 | 화자 | 불일치 표현 | 계층 | 도구 | 한국어 문제 |
|---|---|---|---|---|---|---|---|
| Fountain | 씬·블록 | X | O | X (`[[메모]]`만) | 씬>블록 | 편집기 다수 | 인물을 대문자로 판정해서 한국어는 `@`, 씬 제목은 `.`로 강제해야 함 |
| FDX | Paragraph Type | X | O | △ (ScriptNote·Revision) | 씬>문단 | Final Draft | 국내 대본은 FDX가 드묾 |
| ScreenPy (현재판) | Segment | X | O (`Dialogue.character`) | X | `parent/child_segment_ids` | 파이썬 | 영어 규칙 기반. 원본 위치 `start_pos/end_pos` 있음 |
| SRT | 큐 | O | X | X | X | 거의 모든 도구 | 인코딩 규정 없음. 국내는 SMI·cp949 파일이 남아 있음 |
| WebVTT | 큐, 큐 안 낱말 시각 태그 | O | O (`<v>`) | △ (NOTE) | X | 브라우저 | UTF-8 강제라 무난 |
| TTML/IMSC | div>p>span | O | O (`ttm:agent`) | △ (metadata 확장) | O | 방송·Netflix | 루비·세로쓰기 지원 |
| Netflix 대사 목록 | 대사 이벤트 | O (프레임 단위 TC) | O (Source 열, 화면 밖 목소리는 (VO)) | △ (Annotations 열, 변경 로그 탭) | X | 엑셀 | 인물 이름은 허용 글리프만 |
| CTM/STM | 낱말/발화 | O | STM만 | X | X | sclite·Kaldi | 공백으로 나누므로 단위가 어절 |
| Kaldi ctm-edits | 낱말 | O | X | **O** (`cor/sub/ins/del/sil/fix`) | X | Kaldi cleanup | 띄어쓰기 차이가 sub로 부풀려짐 |
| Gentle JSON | 낱말 | O | X | **O** (`case`: success / not-found-in-audio / not-found-in-transcript, 대본 문자 오프셋) | X | Gentle | 영어 전용 |
| TextGrid | 구간·점 tier | O | tier 이름으로만 | △ (별도 tier) | 시간 포함 관계로만 | Praat·MFA | Praat 기본 저장이 UTF-16으로 바뀌는 함정 |
| ELAN EAF | tier·annotation | O (ms) | O (PARTICIPANT) | O (참조 tier, 통제 어휘) | O (관계 4종) | ELAN·pympi | UTF-8 XML이라 무난 |
| WhisperX JSON | segments>words | O, `score` 포함 | O (화자 분리 시) | X | 2단 | WhisperX | 숫자·기호 낱말에 시각이 빠짐 |
| Qwen3-ForcedAligner | 낱말 또는 글자 (`text`, `start_time`, `end_time`) | O | X | X | X | qwen-asr | 한국어 지원. 음성 5분 한도. 낱말 분리에 soynlp 필요 (2차 출처) |
| W3C Web Annotation | 임의 범위 (TextQuote/TextPosition 선택자) | O (Media Fragment `t=`) | X | O (body로 표현) | 포인터 그래프 | JSON-LD | 오프셋이 코드포인트 기준이라 NFC/NFD 차이로 깨짐 |

**2) 연구 데이터셋의 저장 방식**

- **Everingham 2006 (Buffy):** 시간만 있는 자막과 화자만 있는 팬 대본을 문자열 DTW로 맞춘 뒤, 자막에 화자를 붙였습니다 (인라인 병합).
- **Laptev 2008, MPII-MD/LSMDC:** 대본을 자막에 동적 계획법으로 맞추고, 주변 대사의 일치 낱말 비율이 0.5 이상인 문장만 남겼습니다. 그 뒤 사람이 문장과 영상을 직접 맞췄습니다. 실패한 문장은 버렸고 기록이 없습니다. 결과는 문장과 클립 구간의 평면 표입니다. CSV 열 이름은 접근 제한 때문에 확인하지 못했습니다.
- **MovieNet:** 정규식으로 대본을 씬 셀(지문 + 대사)로 나누고, 대사를 자막과 먼저 맞춘 뒤 최적화로 타임라인에 배치했습니다. 시놉시스 문단과 샷 범위 4,208쌍은 포인터(샷 번호 범위)로 저장했습니다. JSON 필드는 확인하지 못했습니다.
- **TVQA:** 대본에서 얻은 인물 이름을 자막 앞에 붙였습니다 (인라인). 후속 연구는 Whisper/WhisperX와 DTW로 다시 맞추고 사람이 보정했습니다.
- **StoryMovie (2026):** 자막과 대본 대사를 토큰 LCS로 맞추고, 화자의 대사 블록 전체로 넓힌 뒤 블록 시각을 비례로 나눴습니다. 배포본에는 이야기 텍스트 속 인라인 태그(`<gdo char#>`)만 있고, 정렬 중간 산출물과 실패 기록은 README에 없습니다.
- **MovieSum:** Celtx로 정리한 XML(`<scene>`, `<stage_direction>`, `<scene_description>`, `<character>`, `<dialogue>`)이고 시간 정보는 없습니다. 실제 파일에서는 태그 사용이 일관되지 않습니다.
- **공통점:** 불일치는 임계값으로 거르거나 버렸습니다. 불일치를 데이터로 남긴 사례는 ASR 쪽 도구(ctm-edits, Gentle)와 Netflix 대사 목록의 변경 로그 정도입니다. Netflix 대사 목록(as-broadcast script)은 완성 영상 기준으로 받아 적은 대본이라, 업계에서 우리 최종 산출물과 가장 비슷합니다.

**3) 우리 경우에 맞는 설계 패턴**

**P1. 층을 나눈 standoff 구조와 ID 포인터 (ELAN, Lhotse 방식)**
- 영화마다 다섯 층을 둡니다.
  - `source`: 원본 파일 해시, 변환기 버전, 쪽·오프셋
  - `script`: 씬·지문·대사 블록
  - `asr_words`: id, 시작·끝 시각, 확률, 원문, 정규화문
  - `align`: 연산 종류, asr_ids, script 범위, 점수, 판정 주체
  - `final`: 발화, 낱말 id들, 화자, 씬
- 장점: 원본까지 거슬러 갈 수 있고, 한 층만 다시 계산할 수 있으며, 단계별로 버전을 나눌 수 있습니다.
- 단점: 읽으려면 층끼리 조인해야 하고, ID가 바뀌지 않게 관리해야 하며, 사람이 검토할 화면은 따로 만들어야 합니다.

**P2. 편집 연산 표 (ctm-edits, Gentle 방식)**
- 한 행이 정렬 단위 하나이고, 열은 연산·시각·확률·대본 위치입니다. 700개 문서를 Parquet 하나로 모을 수 있습니다.
- 장점: 구조가 단순하고, 불일치율 집계와 필터링이 쉽고, 대량 처리에 강합니다.
- 단점: 씬과 화자 계층을 담기 어렵고, 삭제 행에는 시각이 없으며, 사람이 읽기 불편합니다.

**P3. 인라인 병합 영상 기준 대본 (Netflix 대사 목록, TVQA 방식)**
- 발화 단위 JSON에 시작·끝 시각, 화자, 씬, 최종 텍스트(받아쓰기 기준), `script_ref`, 차이 조각을 넣습니다.
- 장점: 사람이 읽기 좋고 SRT/VTT/TTML/대사 목록으로 바로 내보낼 수 있습니다.
- 단점: 낱말 수준 근거가 빠지고, 다시 정렬하면 전체를 새로 만들어야 합니다.

**권장:** P1을 원본 데이터로 두고, P1의 `align` 층을 P2 형식으로 만들며, P3는 P1에서 만드는 파생 화면으로 둡니다.

**한국어 처리:**
- 비교는 공백을 뺀 음절 단위로 하고, 어절 경계는 따로 보존합니다.
- 오프셋은 NFC로 정규화한 텍스트 기준으로 잡고, 원문 해시와 앞뒤 문맥 인용을 함께 저장해 앵커를 이중으로 둡니다.

**출처**

이번에 확인함:
- https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
- https://friendli.ai/models/Qwen/Qwen3-ASR-0.6B-hf
- https://arxiv.org/abs/2602.21829
- https://huggingface.co/datasets/daniel3303/StoryMovieScript
- https://arxiv.org/pdf/2007.10937
- https://arxiv.org/pdf/2301.00135
- https://backlothelp.netflix.com/hc/en-us/articles/360000427707-Dialogue-List-Style-Guide
- https://backlothelp.netflix.com/hc/en-us/articles/4407306862611-As-Recorded-Dubbing-Script-Scope-of-Work
- https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/cleanup/internal/get_ctm_edits.py
- https://github.com/lowerquality/gentle/blob/master/gentle/transcription.py
- https://arxiv.org/abs/1605.03705
- https://ar5iv.labs.arxiv.org/html/1809.01696
- https://arxiv.org/pdf/2505.17536
- https://huggingface.co/datasets/rohitsaxena/MovieSum
- https://github.com/drwiner/ScreenPy/blob/master/src/screenpy/models.py

이번에 확인하지 못하고 기존 지식으로 쓴 것 (Fountain, WebVTT, TTML/IMSC, Web Annotation, ELAN, WhisperX, Lhotse, Everingham 2006, Praat 인코딩, Gentle 오프셋 단위):
- https://fountain.io/syntax
- https://www.w3.org/TR/webvtt1/
- https://www.w3.org/TR/ttml2/
- https://www.w3.org/TR/annotation-model/
- https://www.mpi.nl/tools/elan/EAF_Annotation_Format_3.0_and_ELAN.pdf
- https://github.com/m-bain/whisperX
- https://lhotse.readthedocs.io

MAD, LSMDC CSV, MovieNet JSON의 실제 필드는 접근 제한이나 문서 부족으로 확인하지 못했습니다.