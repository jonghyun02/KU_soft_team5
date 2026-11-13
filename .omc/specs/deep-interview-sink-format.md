# Deep Interview Spec: 영상 기준 대본 데이터 형식 (sink)

## Metadata
- Interview ID: sink-format-20261113
- Rounds: 11 (+ Round 0 범위 확인)
- Final Ambiguity Score: 19.4%
- Type: brownfield (원본 대본 702개 파일, 소비자 CBoE 기존 코드)
- Generated: 2026-11-13
- Threshold: 0.2
- Threshold Source: default
- Initial Context Summarized: yes (붙여넣은 이전 대화 = 불일치 해결 기법·2025~26 최신 기법 조사를 한 줄로 요약)
- Status: PASSED
- 승인 상태: approved for execution (2026-11-13, 사용자 "형식 맞추는거 저 대본 파일들에 대해서 구현 싹 해줘 ralph")

## Execution Scope (2026-11-13 승인)
- 영상 파일은 사용자가 따로 구한다. 이번 구현은 **대본 파일 쪽 전부**:
  - `raw/`(= 지금 `filmmakers_scripts`, 손대지 않음) 702개 파일 전부를 대상으로 `catalog.csv`, `docs/{doc_id}/lines.jsonl`, `docs/{doc_id}/script.json`, `qa/needs_review.csv`를 만든다(압축 해제, HWP3/HWP5/doc/rtf/docx/pdf 글자층/스캔 PDF OCR/txt 모두).
  - `schema/`에 다섯 형식(lines, script, asr, align, final)의 JSON Schema와 catalog·needs_review 헤더 규칙을 만들고, 검사 도구로 전체를 검사한다.
  - final.json·asr.json·align.json은 영상이 필요하므로 이번엔 스키마와 검사 도구(+ 손으로 만든 예시 파일이 검사를 통과하는지)까지만.
  - 카탈로그의 film_id는 KMDb/TMDB 조회가 가능하면 채우고, 불가하면 `local_` ID + 정규화 제목으로 둔다. video_* 칸은 빈칸.
  - (구현 중 수정 2026-11-13) local ID 638건은 needs_review에 올리지 않는다. 사용자가 원한 목록은 '제대로 못 읽은 대본'이라 638건이 섞이면 목록이 묻힌다. local 여부는 catalog film_id 접두어로 보인다.
- 수락 기준 중 이번 범위: A1~A3, B1~B5, E1~E2, 그리고 D·C는 '스키마가 해당 규칙을 강제하는지'로 확인.

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.84 | 0.35 | 0.293 |
| Constraint Clarity | 0.81 | 0.25 | 0.203 |
| Success Criteria | 0.78 | 0.25 | 0.195 |
| Context Clarity | 0.77 | 0.15 | 0.116 |
| **Total Clarity** | | | **0.806** |
| **Ambiguity** | | | **0.194** |

## Topology
| Component | Status | Description | Coverage |
|-----------|--------|-------------|----------|
| 카탈로그 | active | 원본 문서 ↔ 영화(영상) 짝, 판본·언어·중복 묶음 | 4.2절, 수락 기준 A1~A3 |
| 전처리 대본 | active | 형식 통일 줄(lines.jsonl) + 구조 파싱(script.json) | 4.3·4.4절, 수락 기준 B1~B5 |
| 정렬 중간 결과 | active | 받아쓰기 기록(asr.json) + 연결·판정 기록(align.json), work/에 보관 | 4.5·4.6절, 수락 기준 C1~C3 |
| 최종 영상 기준 대본 | active | 영화 1편당 final.json, CBoE가 이것만 읽음 | 4.7절, 수락 기준 D1~D9 |

미룬 덩어리 없음.

## Goal
대본 원본 702개 파일(한국영화 413·외국영화 285 글)을 전부 하나의 형식 체계로 바꾸고, 영화 1편마다 **영상에서 실제로 들린 대사를 기준으로 한 최종 대본(final.json)** 하나를 만든다. 최종본은 들린 그대로의 대사(영상 언어)와 자막용 한글을 함께 담고, 대본에서 가져온 화자·씬·지문·화면 글자·연기 지시와 대본과의 차이를 표시한다. CBoE 같은 도구는 원본 대본 없이 final.json만 읽고 동작해야 하며, 형식이 기준이고 CBoE가 이 형식에 맞춰 바뀐다.

## Constraints
- 형식이 기준이다. CBoE의 지금 형식(script.json, dialogue.json)에 맞추지 않고, CBoE가 나중에 final.json을 읽도록 바뀐다. 다만 CBoE에서 검증된 관례(초 단위 시각, 낱말 `{w,s,e,p}`, 원자적 쓰기, 원본 sha1 서명)는 가져온다.
- 영화 1편 = 최종본 1개 = 대본 1개. 같은 영화에 대본이 여러 판본이면 영상과 가장 잘 맞는 하나(`match` 최고)를 고르고 카탈로그에 기록한다.
- 대사 글자의 기준은 영상이다. `heard`는 군말·더듬기·반복까지 들린 그대로 적는다. 받아쓰기 정확도는 형식이 아니라 기법에서 올린다.
- 자막은 한국 사람용이다. 모든 대사 항목에 한글 칸 `ko`가 있다. 한국어로 말한 줄은 `heard`와 같고, 외국어로 말한 줄은 **들린 말을 번역**하며(번역 대본은 이름·용어·말투 참고), 출처를 `ko_src`로 남긴다.
- 대본에만 있는 것(씬 머리, 지문, 화면 글자, 전환, 영상에서 잘린 대사)도 최종본에 넣고 표시한다. 지문·씬은 앞뒤 대사로 추정한 시각, 잘린 대사는 시각 없음.
- 한 줄 = 대본 대사 줄 하나. 대본에 없는 말은 붙어 있는 줄에 포함하고, 떨어져 있으면 쉬는 곳에서 잘라 새 줄(`added`)로 만든다.
- 대본을 제대로 못 읽은 문서도 멈추지 않는다. 읽은 만큼만 쓰고 나머지는 '모름'으로 두며, 문제 있는 문서·영화·줄은 `qa/needs_review.csv`에 모은다.
- 영상 언어와 대본 언어가 다르면(번역 대본, 불어 대본에 영어 대사) 뜻으로 맞춰서(다국어 문장 벡터·LLM) 화자·씬·지문을 가져오고, 낱말 단위 차이 판정은 하지 않는다.
- 받아쓰기·연결 기록은 `films/{film_id}/work/`에 따로 보관한다. CBoE는 읽지 않으며, 지워도 final.json에는 영향이 없다. 정렬만 다시 돌릴 때 받아쓰기를 재사용하기 위함이다.
- 글자는 UTF-8, 유니코드 NFC. 시각은 영상 첫 프레임 기준 초, 소수 셋째 자리까지.
- 원본 `raw/`는 읽기 전용이다. 파일 이름을 바꾸지 않는다.

## Non-Goals
- 파이프라인 구현(변환기, OCR, 파서, ASR, 정렬기, 번역기)과 모델 선택. 이 명세는 결과물 형식만 정한다.
- CBoE 코드 수정. final.json을 읽게 바꾸는 일은 다음 작업이다(4.9절 대응표가 그 입력).
- 영상 파일 확보. 카탈로그에 칸만 둔다.
- 자막 칸 나누기(글자 수·시간 규칙)와 효과음 자막. CBoE의 몫이다.
- 사람 수정 보존 체계, 공식 한글 자막 수집.

## Acceptance Criteria
### 카탈로그
- [ ] A1. `catalog.csv`에 문서마다 한 행이 있고, 원본 파일 702개(압축 안 파일 포함)가 빠짐없이 어느 문서에 속하거나 `excluded`로 표시된다.
- [ ] A2. 같은 `film_id`를 가진 문서 중 `is_primary=true`는 최대 1개다.
- [ ] A3. `real_format`은 확장자가 아니라 파일 앞 바이트로 판별한 값이다.
### 전처리 대본
- [ ] B1. 모든 문서가 `lines.jsonl`과 `script.json`을 만들고 JSON Schema 검사를 통과한다. 못 읽은 부분은 `unknown` 요소로 남는다.
  (구현 중 보완 2026-11-13) 글자를 꺼내지 못한 구간(쪽 그림 실패, 망가진 파일의 뒷부분)은 줄 자체가 없으므로 요소도 없다. 그 구간은 `qa/needs_review.csv`(extract_broken, 쪽 번호·읽은 비율)에만 기록한다. 글자는 있는데 종류를 못 정한 줄이 `unknown`.
- [ ] B2. `script.json`의 모든 요소는 `lines`로 `lines.jsonl`의 줄 번호를 가리키고, 버린 줄은 `dropped`에 이유와 함께 남는다(모든 줄이 요소 또는 dropped 중 정확히 하나에 속함).
- [ ] B3. 요소 종류는 7종(front, scene, action, dialogue, onscreen, transition, unknown)만 쓴다.
- [ ] B4. 품질 지표가 기준 아래인 문서는 `qa/needs_review.csv`에 문제 코드와 함께 오른다.
- [ ] B5. 4.4.3절의 까다로운 경우 11가지가 모두 이 형식으로 표현된다(시험용 표본 문서로 확인).
### 정렬 중간 결과
- [ ] C1. `work/asr.json`의 낱말은 파일 안에서 바뀌지 않는 번호 `i`를 갖고, `align.json`의 연결은 이 번호와 대본 요소 id로만 가리킨다.
- [ ] C2. `align.json`의 연산 하나는 final.json의 대사 항목 하나와 1:1로 대응한다(`item` 칸).
- [ ] C3. 같은 asr.json으로 정렬만 다시 돌릴 수 있다(asr.json에 정렬 결과가 섞이지 않음).
### 최종본
- [ ] D1. 영상이 있는 영화마다 `final.json`이 있고 JSON Schema 검사를 통과한다.
- [ ] D2. `status`가 same·changed·added인 대사는 `start < end`, `words`가 비어 있지 않고, 모든 낱말 시각이 `[start, end]` 안에 있다.
- [ ] D3. `status=absent`는 `start=end=null`, `script_ref`가 있다. `status=added`는 `script_ref=null`.
- [ ] D4. `status=changed`면 `diffs`가 하나 이상, `same`이면 `diffs`가 비어 있다(띄어쓰기·문장부호 차이는 차이로 치지 않음).
- [ ] D5. 시각이 있는 모든 대사 항목에 `ko`가 있고, `lang=ko`면 `ko_src=same`이고 `ko == heard`다.
- [ ] D6. `match=translation`인 항목은 `diffs`가 비어 있다(언어가 달라 낱말 판정 안 함).
- [ ] D7. `items`는 시각 순서이고, 시각 없는 항목(absent, timing=none)은 대본 순서로 앞뒤 항목 사이에 놓인다.
- [ ] D8. 모든 참조(`scene`, `speakers`, `script_ref.el`, `src_el`)가 같은 파일 또는 지정한 script.json 안에 실제로 있다.
- [ ] D9. 4.9절 대응표의 CBoE 입력 칸이 모두 final.json만으로 채워진다(원본 대본·script.json 없이).
### 공통
- [ ] E1. 모든 JSON 파일에 `schema` 버전 문자열이 있고 `schema/` 폴더의 JSON Schema로 검사한다.
- [ ] E2. 모든 글자는 UTF-8·NFC이고, 파일은 임시 파일에 쓴 뒤 바꿔치기(원자적 쓰기)로 저장한다.

## 4. 형식 명세

### 4.1 폴더 구조
```
C:\projects\sink\
  raw\                              원본 그대로(filmmakers_scripts), 읽기 전용
  catalog.csv                       문서 1개 = 1행
  qa\needs_review.csv               확인 필요 목록
  docs\{doc_id}\lines.jsonl         형식 통일 줄
  docs\{doc_id}\script.json         구조 파싱
  films\{film_id}\final.json        최종 영상 기준 대본 (CBoE 입력)
  films\{film_id}\work\asr.json     받아쓰기 기록
  films\{film_id}\work\align.json   연결·판정 기록
  schema\*.schema.json              검사 규칙
```

### 4.2 ID 규칙
| 대상 | 모양 | 예 | 비고 |
|---|---|---|---|
| 문서 | `fm{글번호}` 또는 `fm{글번호}_{k}` | `fm55574`, `fm54259_2` | 한 글에 서로 다른 대본이 여럿이면(압축 묶음) `_k`. 한 대본이 여러 파일로 나뉜 경우(감시자들 1·2)는 문서 하나에 파일 여러 개 |
| 영화 | `{출처}_{번호}` | `kmdb_K12345`, `tmdb_550`, `local_0007` | KMDb(한국)·TMDB(외국) 우선, 못 찾으면 local. 폴더 이름과 같은 문자열 |
| 인물 | `P01`… | | 문서(script.json)·영화(final.json) 안에서만 유일 |
| 씬 | `SC001`… | | 대본 씬 번호(`12A`)는 `number` 칸에 문자열로 |
| 대본 요소 | `E0001`… | | script.json 안에서 유일 |
| 받아쓰기 낱말 | 정수 `i` | | asr.json 안에서 유일, 0부터 |
| 최종 항목 | `I0001`… | | final.json 안에서 유일, items 차례 |

### 4.3 catalog.csv (UTF-8 BOM, 문서 1개 = 1행)
| 칸 | 값 | 설명 |
|---|---|---|
| doc_id | `fm55574` | |
| post_id | `55574` | 필름메이커스 글 번호 |
| category | `ko_film` / `foreign_film` | 받은 게시판 |
| files | `raw/…/a.pdf\|raw/…/b.pdf` | 원본 경로, 순서대로 |
| sha1 | `…\|…` | 파일별 |
| real_format | hwp3, hwp5, doc, docx, rtf, pdf_text, pdf_scan, txt, mixed | 앞 바이트로 판별 |
| title_raw / title_norm | | 글 제목 / 군더더기 뺀 제목 |
| script_lang | ko, en, fr, es, ja, mixed … | 대본 글자의 언어 |
| is_translation | true/false | 번역 대본 여부 |
| version_label / version_raw | draft, revised, bound(제본고), shooting, final, translation, unknown / 원문 표기 | |
| dup_group | `G017` | 같은 영화 대본 묶음 |
| film_id, film_title, film_title_orig, year | | |
| video_path, video_sha1, video_duration, video_lang | | 영상이 없으면 빈칸 |
| is_primary | true/false | 이 영화 최종본에 쓰는 대본 |
| match | 0~1 | 영상과의 일치율(정렬 뒤 채움) |
| extract_quality | 0~1 | 4.4.2 품질 점수 |
| status | pending, parsed, aligned, done, excluded | |
| note | | |

### 4.4 전처리 대본
#### 4.4.1 lines.jsonl (형식 통일 줄, 물리적인 줄 1개 = 1행)
(구현 중 보완 2026-11-13: 첫 행은 머리 행 `{"schema":"sink.lines/1","doc_id":…,"files":[…],"made":{…}}`이고 둘째 행부터 줄 행이다. E1을 JSONL에도 적용하기 위함.)
```json
{"n":42,"file":0,"page":7,"x":0.10,"y":0.31,"text":"형사      너 어제 어디 있었어?","bold":false,"cell":null,"src":"pdf_text","conf":null}
```
- `n`: 문서 안 줄 번호(0부터). `file`: catalog `files` 안 순서. `page`: 1부터, 쪽 개념이 없으면 null.
- `x`,`y`: 쪽 안 왼쪽·위 위치(0~1). TXT는 앞 공백·탭을 칸 수로 바꿔 `x`=칸/80. HWP는 문단 들여쓰기·탭 위치에서 계산.
- `cell`: HWP·Word 표 안 글이면 `{"table":3,"row":5,"col":1}`.
- `src`: hwp, doc, docx, rtf, pdf_text, ocr, txt, xls(구현 중 추가: 엑셀 표 대본, 한 행 = 한 줄, `cell`에 시트·행·첫 칸). `conf`: OCR 신뢰도(0~1), 그 외 null.
- `text`: 추출한 그대로에 NFC만 적용. 머리말 지우기·줄 합치기는 여기서 하지 않는다(script.json에서 줄 번호로 처리).

#### 4.4.2 script.json (구조 파싱)
```json
{
 "schema": "sink.script/1",
 "doc_id": "fm55574",
 "parser": {"name": "sink-parse", "version": "0.1.0"},
 "lang": "ko",
 "front": {"title": "범죄의 재구성", "writer": null, "version_raw": "촬영본", "date": null},
 "people": [{"id": "P01", "name": "형사", "aliases": ["강형사"], "count": 57}],
 "scenes": [{"id": "SC012", "number": "12", "heading": "경찰서 취조실 - 밤",
             "place": "경찰서 취조실", "time": "밤", "flags": []}],
 "elements": [
  {"id": "E0120", "type": "scene", "scene": "SC012", "text": "S#12. 경찰서 취조실 - 밤", "lines": [40]},
  {"id": "E0121", "type": "action", "scene": "SC012", "text": "형사, 서류를 책상에 던진다.", "lines": [41]},
  {"id": "E0122", "type": "dialogue", "scene": "SC012",
   "speakers": ["P01"], "speaker_raw": "형사", "how": "on", "lang": "ko",
   "text": "너 어제 어디 있었어?", "raw": "형사      너 어제 어디 있었어?",
   "paren": [], "gloss": null, "sung": false, "overlap": null, "lines": [42]},
  {"id": "E0123", "type": "dialogue", "scene": "SC012",
   "speakers": ["P02"], "speaker_raw": "창혁", "how": "on", "lang": "ko",
   "text": "집에 있었는데요.", "raw": "창혁      (웃으며) 집에 있었는데요.",
   "paren": [{"text": "웃으며", "at": 0}], "gloss": null, "sung": false, "overlap": null, "lines": [43]}
 ],
 "dropped": [{"lines": [0, 1], "why": "page_header"}, {"lines": [77], "why": "cont_marker"}],
 "quality": {"lines": 3120, "typed_ratio": 0.97, "dialogue_ratio": 0.48, "unknown_ratio": 0.03,
             "speakerless_dialogue": 4, "score": 0.91}
}
```
- 요소 종류 7종: `front`(표지·인물표·연출의 말), `scene`, `action`(지문), `dialogue`, `onscreen`(자막·INSERT·화면 속 글자, `kind`: subtitle | insert | title | other), `transition`(CUT TO·F.O.·디졸브·암전), `unknown`.
- 몽타주·플래시백·회상·교차편집은 씬의 `flags`(montage, flashback, intercut, dream)로, 노래는 대사의 `sung`으로 표시한다.
- 대사 칸
  - `speakers`: 인물 id 목록(1명 이상, 화자를 못 읽으면 빈 목록). 동시에 같은 말(`일동`, `JASPER/VIRGINIA`)이면 여럿.
  - `speaker_raw`: 적힌 그대로(`천세황(무전소리)`).
  - `how`: on | off(화면 밖, E·O.S.·V.O. 중 화면 밖) | narration(NA·N·Na) | phone | radio | broadcast | video | inner(속마음) | unknown.
  - `lang`: 말하는 언어. 표기(`영어소리`, `(日)`, `en anglais`)가 있으면 그 언어, 없으면 문서 언어.
  - `text`: 말하는 글(괄호 지시 뺌). `raw`: 적힌 그대로.
  - `paren`: 연기 지시 목록. `at`은 `text` 안 글자 위치.
  - `gloss`: 대본에 병기된 번역(`Sir! … (심각한…)`의 괄호 안). 없으면 null.
  - `overlap`: 동시·겹친 대사 묶음 번호(같은 번호끼리 겹침). 없으면 null.
  - `speaker_inferred`(선택 칸, 구현 중 보완 2026-11-13): `true`면 `speakers`는 대본에 적힌 이름이 아니라 같은 씬 바로 앞 대사에서 이어받은 추정값이다(이름 줄 문서에서 지문 뒤에 이름 줄 없이 대사 칸으로 이어진 말). 추정이 아니면 칸을 쓰지 않으므로 옛 파일도 그대로 유효하다. CBoE 등 소비자는 이 칸이 참인 화자를 확정값으로 세지 않는다.
- `(N)`처럼 문서마다 뜻이 다른 표기(해운대는 밤, 다른 문서는 내레이션)는 문서별로 해석하고 그 결정을 `quality.notes`에 남긴다.
- 쪽 넘김으로 쪼개진 대사(`(MORE)`, `(CONT'D)`, `(계속)`)는 요소 하나로 합치고 표시 줄은 `dropped`(why: `cont_marker`)로 보낸다. 쪽번호·머리말·꼬리말·`(CONTINUED)`·여백 수정표시 `*`도 `dropped`.
- 줄을 합칠 때 낱말 중간 줄바꿈(`벗는\n다.`)은 공백 없이 잇는다.
- `quality.score` 초기 기준(조정 가능): typed_ratio < 0.85, dialogue_ratio < 0.15, OCR 평균 conf < 0.85, 화자 없는 대사 > 5% 중 하나라도 걸리면 `needs_review`.
  - (구현 중 보완 2026-11-13) `quality.empty_dialogue`(말 없이 이름 줄만 잡힌 대사 수)와 `quality.inferred_speaker`(speaker_inferred 대사 수)를 더 센다. 말 없는 대사 > 10%면 `parse_low_dialogue`(대사 모양을 잘못 읽어 말이 지문으로 갔을 수 있음), 추정 화자 > 10%면 `speaker_missing` 으로 올린다. 규칙 파서는 모르는 줄을 지문으로 두므로 typed_ratio 만으로는 잘못 읽은 문서를 못 잡기 때문.

#### 4.4.3 까다로운 경우 11가지와 표현 방법
| # | 경우 | 표현 |
|---|---|---|
| 1 | 한 대사에 화자 여럿(일동, 두 화자 한 줄, 좌우 두 칸 동시 대사) | `speakers` 여럿 또는 `overlap` 묶음 |
| 2 | 화자 꼬리표 조합(`천세황(무전소리)`) | `speaker_raw` + `speakers` + `how` + `lang` 분리 |
| 3 | 대본 언어 ≠ 영상 언어, 원문·번역·발음 병기 | 대사 `lang`, `gloss`; 최종본 `match=translation` |
| 4 | 소리 없는 글자(자막·INSERT)와 노래·내레이션 | `onscreen`(정렬 제외), `sung`, `how=narration` |
| 5 | 대사 안 괄호 지시, 실제 소리를 가리키는 괄호(`(buzzer goes)`) | `paren`(위치 포함). 소리 괄호도 paren에 두고 CBoE가 판단 |
| 6 | 씬 번호 변형(`12A`, `76-3`, `#100-1`, `C#3`, 번호 없음) | `number` 문자열 또는 null, 씬 안 장소 바뀜은 action |
| 7 | 쪽 넘김으로 쪼개진 대사 | 요소 하나 + `dropped(cont_marker)` |
| 8 | 깨진 글자층·OCR 품질 | lines의 `src`·`conf`, script `quality` |
| 9 | 인물 이름 변형(성 생략, 잘린 이름, 번호 엑스트라) | `people[].aliases`, `speaker_raw` |
| 10 | 앞머리(인물표 번호가 씬 머리처럼 보임) | `front` 요소 |
| 11 | 같은 영화 여러 판본 | catalog `version_label`·`dup_group`·`is_primary` |

### 4.5 work/asr.json (받아쓰기 기록)
```json
{
 "schema": "sink.asr/1",
 "video_sha1": "…", "lang": "ko",
 "passes": [
  {"id": "plain", "model": "…", "context": null,
   "words": [{"i": 0, "w": "어제", "s": 723.40, "e": 723.71, "p": 0.98, "lang": "ko"}]},
  {"id": "ctx", "model": "…", "context": "script_window",
   "words": [{"i": 0, "w": "어제", "s": 723.40, "e": 723.71, "p": 0.97, "lang": "ko"}]}
 ],
 "vad": [[721.2, 726.0]],
 "chunks": [{"start": 720.0, "end": 745.0, "lang": "ko"}]
}
```
- 받아쓰기 방식이 여럿이면(대본 없이, 대본 문맥 넣고) `passes`에 각각 둔다. 낱말 번호 `i`는 pass 안에서 유일하고, 연결에서는 `"plain:12"`처럼 pass와 함께 가리킨다.
- 정렬·판정 결과는 여기에 넣지 않는다.

### 4.6 work/align.json (연결·판정 기록, 연산 1개 = 최종 대사 1줄)
```json
{
 "schema": "sink.align/1",
 "script_doc": "fm55574", "asr_sha1": "…",
 "method": {"anchor": "difflib-blocks+LIS", "acoustic": "…", "forced": "…", "translate": "…"},
 "ops": [
  {"item": "I0452", "op": "changed", "script_el": "E0122",
   "asr": ["plain:1520", "plain:1524"], "match": "text", "text_score": 0.55,
   "candidates": [{"text": "어제 밤에 어디 있었냐고.", "src": "asr", "score": -12.3},
                  {"text": "너 어제 어디 있었어?", "src": "script", "score": -31.0}],
   "chosen": 0, "decided_by": "acoustic"},
  {"item": "I0454", "op": "added", "script_el": null, "asr": ["plain:1531", "plain:1536"],
   "match": null, "decided_by": "rule"},
  {"item": "I0455", "op": "absent", "script_el": "E0124", "asr": null, "match": "text", "decided_by": "rule"}
 ],
 "scenes": [{"scene": "SC012", "start": 721.8, "end": 790.2, "how": "anchors"}]
}
```
- `op`은 최종본 `status`와 같은 값(same, changed, added, absent). `asr`는 낱말 범위 [처음, 끝]. `decided_by`: rule | acoustic | llm | translation.

### 4.7 final.json (최종 영상 기준 대본, CBoE 입력)
```json
{
 "schema": "sink.final/1",
 "film": {"id": "kmdb_K12345", "title": "범죄의 재구성", "title_orig": null, "year": 2004,
          "video": {"sha1": "…", "duration": 7290.4, "lang": "ko"},
          "script": {"doc_id": "fm55574", "version": "shooting", "lang": "ko",
                     "translation": false, "match": 0.71}},
 "made": {"pipeline": "0.1.0", "asr": "…", "aligner": "…", "translator": null},
 "people": [{"id": "P01", "name": "형사", "aliases": ["강형사"]},
            {"id": "P02", "name": "창혁", "aliases": []}],
 "scenes": [{"id": "SC012", "number": "12", "heading": "경찰서 취조실 - 밤",
             "place": "경찰서 취조실", "time": "밤", "flags": [],
             "start": 721.8, "end": 790.2, "timing": "estimated"}],
 "items": [
  {"id": "I0450", "type": "scene", "scene": "SC012", "text": "S#12. 경찰서 취조실 - 밤",
   "start": 721.8, "end": null, "timing": "estimated", "src_el": "E0120"},
  {"id": "I0451", "type": "action", "scene": "SC012", "text": "형사, 서류를 책상에 던진다.",
   "start": 721.8, "end": null, "timing": "estimated", "src_el": "E0121"},
  {"id": "I0452", "type": "dialogue", "scene": "SC012",
   "speakers": ["P01"], "speaker_from": "script", "how": "on", "lang": "ko",
   "heard": "어제 밤에 어디 있었냐고.", "ko": "어제 밤에 어디 있었냐고.", "ko_src": "same",
   "start": 723.4, "end": 725.1, "timing": "aligned",
   "words": [{"w": "어제", "s": 723.40, "e": 723.71, "p": 0.98},
             {"w": "밤에", "s": 723.71, "e": 723.95, "p": 0.95}],
   "status": "changed", "match": "text",
   "script_ref": {"el": "E0122", "text": "너 어제 어디 있었어?"},
   "diffs": [{"kind": "word", "script": "너", "heard": ""},
             {"kind": "word", "script": "", "heard": "밤에"}],
   "paren": [], "sung": false, "overlap": null, "conf": 0.86, "src_el": "E0122"},
  {"id": "I0454", "type": "dialogue", "scene": "SC012",
   "speakers": [], "speaker_from": "unknown", "how": "unknown", "lang": "ko",
   "heard": "웃어? 지금 웃음이 나와?", "ko": "웃어? 지금 웃음이 나와?", "ko_src": "same",
   "start": 728.1, "end": 729.6, "timing": "aligned", "words": ["…"],
   "status": "added", "match": null, "script_ref": null, "diffs": [],
   "paren": [], "sung": false, "overlap": null, "conf": 0.74, "src_el": null},
  {"id": "I0455", "type": "dialogue", "scene": "SC012",
   "speakers": ["P01"], "speaker_from": "script", "how": "on", "lang": "ko",
   "heard": null, "ko": null, "ko_src": null,
   "start": null, "end": null, "timing": "none", "words": [],
   "status": "absent", "match": "text",
   "script_ref": {"el": "E0124", "text": "거짓말하지 마."}, "diffs": [],
   "paren": [], "sung": false, "overlap": null, "conf": 0.9, "src_el": "E0124"}
 ],
 "quality": {"dialogue": 1820, "same": 610, "changed": 980, "added": 150, "absent": 80,
             "speaker_unknown": 170, "mean_word_p": 0.88}
}
```
칸 규칙
- 항목 종류: `scene`, `action`, `dialogue`, `onscreen`, `transition`. 대본의 `front`·`unknown`은 최종본에 넣지 않는다(script.json에 남음).
- `timing`: aligned(강제 정렬로 확정) | estimated(앞뒤 대사로 추정) | none(시각 없음).
- `status`(대사만): same(띄어쓰기·문장부호를 빼면 대본과 같음) | changed(차이 있음, 군말 포함) | added(대본에 없는 말) | absent(영상에 없음).
- `diffs[].kind`: filler(군말) | repeat(반복·더듬기) | word(낱말 바뀜·빠짐·더함) | number(숫자 표기) | spelling(맞춤법·표기). 군말만 다른 줄도 changed이며, 걸러 보려면 kind를 본다.
- `match`: text(글자로 맞춤) | translation(뜻으로 맞춤, diffs 비움) | null(added).
- `speaker_from`: script | translation_match | unknown.
- `heard`: 영상 언어로 들린 그대로. `words`의 `w`를 이은 것과 같은 글(띄어쓰기는 받아쓰기 기준).
- `ko`: 자막용 한글. `ko_src`: same(한국어로 말함) | mt(기계 번역, 들린 말을 번역하고 대본은 참고) | human.
- 한 줄 안에 한국어와 외국어가 섞이면 `lang`은 많이 쓴 언어, 낱말에 `lang`을 붙일 수 있고, 외국어가 하나라도 있으면 `ko_src=mt`.
- `paren`, `how`, `sung`, `overlap`은 대본 요소에서 복사한다(CBoE가 script.json 없이 쓰도록).
- `conf`: 그 줄 판정의 종합 신뢰도(0~1). 계산법은 파이프라인 버전에 따라 `made`에 남긴다.
- 시각은 초(소수 셋째 자리), 영상 첫 프레임 = 0.

### 4.8 qa/needs_review.csv
| 칸 | 설명 |
|---|---|
| level | doc, film, item |
| doc_id, film_id, item_id | 해당되는 것 |
| problem | extract_broken, ocr_low, parse_low_dialogue, speaker_missing, version_mismatch(일치율 낮아 판본·영화가 틀렸을 가능성), translation_unmatched, asr_low_conf, absent_many, other |
| score | 문제를 판단한 값 |
| detail | 사람이 읽을 설명 |
| path | 원본 또는 결과 파일 경로 |

### 4.9 CBoE 입력 대응표 (CBoE 수정 작업의 입력)
| CBoE가 지금 쓰는 칸 | final.json에서 가져올 곳 |
|---|---|
| lines[].speaker / people | `items[].speakers` + `people` |
| lines[].text | 자막에는 `ko`, 소리 기준 비교에는 `heard` |
| lines[].raw | `script_ref.text`(대본 원문) |
| lines[].scene_id, scenes[].start/end | `items[].scene`, `scenes[]` |
| lines[].language, sung, source | `lang`, `sung`, `how` |
| lines[].directions | `paren` |
| lines[].start/end, words{w,s,e} | `start`, `end`, `words` |
| lines[].timing / estimated | `timing` |
| lines[].status ok/differs/absent | `status` same/changed/absent |
| lines[].diff | `diffs` |
| extras[] | `status=added` 항목 |
| elements(action, onscreen) | `items[].type` action, onscreen |
| out_of_range | 쓰지 않음(영화 전체가 범위) |

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|---|---|---|
| 파일 제목을 먼저 통일해야 한다 | 영상과 잇는 열쇠는 이름이 아니라 작품 ID | 원본 이름은 그대로, 카탈로그 ID로 통일 |
| 최종본은 CBoE 형식에 맞춘다 | 누가 기준인가 | 형식이 기준, CBoE가 맞춘다(R1) |
| 받아쓰기·연결표도 최종본에 넣어야 한다 | 최종본만 읽고 돌아야 함 | 최종본은 완결, 기록은 work/(R3) |
| 영상 기준이면 대본에만 있는 건 버린다 | 전제 뒤집기 질문 | 지문·씬·화면 글자·잘린 대사 모두 넣고 표시(R4) |
| 판본마다 결과를 만든다 | 같은 영화 29묶음 | 영화 1편 = 최종본 1개 = 대본 1개(R5) |
| 못 읽은 대본은 빼거나 보류한다 | 단순화 질문 | 읽은 만큼 쓰고 목록 파일로 알림(R6) |
| 대사는 정리해 적는다 | 자막 원천이 무엇인가 | 들린 그대로, 차이는 종류로 기록(R7) |
| 최종본 한 줄 = 소리 기준 발화 | 핵심 단위 질문 | 대본 대사 줄 기준(R8) |
| 최종본 대사는 영상 언어만 | 한국 사람용 자막 | 한글 칸 추가, 들린 말 번역 + 대본 참고(R9·R10) |

## Technical Context
- 원본 조사: 파일 702개. HWP 426(HWP3 137, HWP5 289), PDF 185(글자층 있음 112, 스캔 72, 열리지 않음 1), doc 61(RTF 위장 2, 평문 1, 미상 1), docx 2, txt 14, 압축 12(zip 8, rar 분할 2, alz 분할 2). 한국 글자층 PDF 15개 중 5개 글자층 깨짐. 제목 기준 같은 영화 29묶음 60개 글(영화 수 약 655 추정).
- 한컴오피스 2024가 설치돼 있어 HWP3·HWP5·doc·rtf를 자동화로 변환할 수 있다(파이썬 라이브러리는 HWP3 미지원).
- CBoE(C:\projects\cboe) 지금 형식: 대본 줄이 기준이고 들린 말은 `diff`·`extras`로 표시만 함, 시각은 절대 초 float, 낱말 `{w,s,e,p}`, status 5종, 원자적 JSON 쓰기, 원본 sha1 무효화. HWP3·doc·스캔·영어 대본 미지원. sink는 축을 뒤집어 들린 말이 기준.
- 외부 형식: 대본+발화+시간을 함께 담는 표준은 없음. 불일치를 데이터로 남기는 쪽은 Kaldi ctm-edits(cor/sub/ins/del), Gentle(case). 업계 최종본 유사물은 Netflix 대사 목록(as-broadcast). 이 명세는 층 분리(ELAN·Lhotse식) + 연산 표(ctm-edits식) + 인라인 최종본(대사 목록식)을 섞은 구조.
- 조사 근거 파일: 스크래치패드 `wf_landscape.md`, `wf_elements.md`, `agentC/` (세션 임시 폴더).

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|---|---|---|---|
| 문서 | core | doc_id, files, real_format, script_lang, version_label, quality | 영화에 속함, 형식 통일 줄을 가짐 |
| 영화(영상) | core | film_id, video, lang, primary 문서 | 최종본 1개, 대본 1개 |
| 형식 통일 줄 | supporting | n, page, x, y, text, src, conf | 문서에 속함, 대본 요소가 가리킴 |
| 대본 요소 | core | id, type(7종), scene, speakers, how, lang, text, raw, paren | 문서에 속함, 최종 항목이 참조 |
| 받아쓰기 낱말 | supporting | i, w, s, e, p, lang | 영화의 asr.json, 연결이 가리킴 |
| 연결 | supporting | item, op, script_el, asr 범위, match, candidates | 대본 요소와 받아쓰기 낱말을 이음, 최종 항목과 1:1 |
| 최종 항목(발화 등) | core | type, status, heard, ko, ko_src, words, speakers, start, end | 영화의 final.json, 대본 요소 0~1개 참조 |
| CBoE | external | final.json 입력 | 최종본을 읽음 |
| 확인 필요 목록 | supporting | level, problem, score, path | 문서·영화·항목을 가리킴 |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|---|---|---|---|---|---|
| 1 | 8 | 8 | - | - | - |
| 2 | 8 | 0 | 0 | 8 | 100% |
| 3 | 8 | 0 | 1 (최종 요소 → 발화) | 7 | 100% |
| 4 | 8 | 0 | 0 | 8 | 100% |
| 5 | 8 | 0 | 0 | 8 | 100% |
| 6 | 9 | 1 (확인 필요 목록) | 0 | 8 | 89% |
| 7 | 9 | 0 | 0 | 9 | 100% |
| 8 | 9 | 0 | 0 | 9 | 100% |
| 9 | 9 | 0 | 0 | 9 | 100% |
| 10 | 9 | 0 | 0 | 9 | 100% |
| 11 | 9 | 0 | 0 | 9 | 100% |

## Interview Transcript
<details>
<summary>Q&A (Round 0 + 11 rounds)</summary>

### Round 0
**Q:** 형식을 정할 덩어리 네 개(카탈로그, 전처리 대본, 정렬 중간 결과, 최종본)가 맞나?
**A:** 예시를 보고 나서 "네 개 다".

### Round 1
**Q:** 최종 영상 기준 대본은 주로 무엇에 쓰이나?
**A:** CBoE 같은 도구의 입력. 덧붙여 "CBoE에 맞춘다기보단 우리가 형식을 정해 놓고 CBoE가 이거에 맞춰야지".
**Ambiguity:** 81% → 73%

### Round 2
**Q:** 형식이 잘 정해졌다는 걸 무엇으로 확인하나?
**A:** 700개 전부 한 형식으로, CBoE가 이것만 읽고 돎.
**Ambiguity:** 62%

### Round 3
**Q:** 받아쓰기·연결표를 어디에 두나?
**A:** "어쨌든 영상 대사 기준으로 전사된 결과물이 입력으로 들어가야" → 최종본 완결, 기록은 work/에 기본값으로 보관.
**Ambiguity:** 55%

### Round 4 (전제 뒤집기)
**Q:** 대본에만 있는 것(지문, 씬, 화면 글자, 잘린 대사)도 넣나?
**A:** 다 넣고 표시.
**Ambiguity:** 52%

### Round 5
**Q:** 같은 영화에 대본이 여러 개면?
**A:** 영상 1편 = 최종본 1개, 대본 1개만(가장 잘 맞는 판본).
**Ambiguity:** 46%

### Round 6 (단순화)
**Q:** 제대로 못 읽은 대본은?
**A:** 읽은 만큼만 쓰고 계속, 제대로 못 읽은 건 따로 목록 파일로.
**Ambiguity:** 42%

### Round 7
**Q:** 대사 글자를 어떤 수준으로 적나?
**A:** 자막으로 만들 거니 들린 그대로. 정확도는 기법에서 올림.
**Ambiguity:** 38%

### Round 8 (핵심 단위)
**Q:** 최종본 대사 한 줄의 기준은?
**A:** 대본 대사 줄 기준.
**Ambiguity:** 36%

### Round 9
**Q:** 영상 언어와 대본 언어가 다르면?
**A:** 한국 사람용이라 최종 자막은 한글이어야 하고, 한국영화 속 짧은 외국어 대사도 있다 → 한글 칸 추가, 번역 대본은 뜻으로 맞춰서 씀.
**Ambiguity:** 33%

### Round 10
**Q:** 한글 칸은 어디서 채우나?
**A:** 들린 말을 번역, 대본은 참고.
**Ambiguity:** 30%

### Round 11
**Q:** 형식 초안 확정, 요소 종류 몇 개?
**A:** 이대로 확정(7종).
**Ambiguity:** 19.4%
</details>
