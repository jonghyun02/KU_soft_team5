# sinkprep 설계 (모듈 사이 약속)

형식 명세: `.omc/specs/deep-interview-sink-format.md` (4장). 이 문서는 구현 모듈끼리의 약속만 적는다.

## 실행
- 가상환경: `C:\projects\sink\.venv` (editable 설치). 실행은 항상 `.venv\Scripts\python -I -m sinkprep <명령>`.
- `-I`는 필수다. 원본 대본은 인터넷에서 받은 신뢰할 수 없는 데이터라 실행하지 않고, 원본이 있는 폴더를 현재 폴더로 두고 파이썬을 돌리지 않는다.
- 원본 `filmmakers_scripts/`(= 명세 `raw/`)에는 쓰지 않는다(`paths.assert_not_raw`, `util.write_*`가 막음).
- 모든 파일 쓰기는 `util.write_json / write_jsonl / write_text`(원자적, UTF-8)로 한다. 글자는 `util.nfc`. ffmpeg 처럼 외부 프로그램이 직접 쓰는 큰 파일은 다른 이름으로 쓴 뒤 `os.replace`.
- 코어 의존성(`pyproject.toml` 의 dependencies)에는 코어 코드가 직접 쓰는 psutil(convert·hancom_worker 의 한컴 프로세스 찾기)과
  numpy(ocr·media 의 그림·소리 배열)도 적는다(전에는 다른 패키지를 따라 딸려 왔다).
- 선택 의존성: 받아쓰기는 `pip install -e .[asr]`(faster-whisper 1.2.1 고정), GPU 로 받아쓰려면 `.[asr,asr-gpu]`(cuBLAS, cuDNN 9). ffmpeg·ffprobe 는 PATH 에, claude 는 llm 단계와 `align --llm` 에만 필요하다.

## 흐름과 파일
```
catalog   filmmakers_scripts/**  (+ 압축 해제 extracted/{post_id}/**)  -> catalog.csv
convert   hwp3/hwp5/doc/docx/rtf 파일                                 -> cache/pdf/{doc_id}/{k}.pdf, cache/convert_log.jsonl
ocr       pdf_scan 파일 + 글자층 깨진 PDF(변환 PDF 포함)              -> cache/ocr/{doc_id}/{k}.json
lines     catalog + cache                                              -> docs/{doc_id}/lines.jsonl
parse     lines.jsonl                                                  -> docs/{doc_id}/script.json
llm       parse 문제 문서의 lines + 규칙 판                              -> cache/parse_llm/{doc_id}.json (+ 뚜렷이 나쁘지 않으면 docs/{doc_id}/script.json)
video     영상 파일 1개 + --docs 의 문서                                 -> videos/{film_id}{확장자}, cache/audio/{video_sha1}.wav·.timemap.json, catalog 영상 칸
asr       catalog 영상 칸 + cache/audio                                 -> cache/asr/{video_sha1}/**, films/{film_id}/work/asr.json
align     docs/{doc_id}/script.json + work/asr.json                     -> films/{film_id}/work/align.json, catalog match·status=aligned
          (+ cache/asr 의 rescue 창 기록과 asr 지문(버릇 글귀), cache/llm(LLM 짝 판정), catalog video_duration)
final     script.json + work/asr.json + work/align.json + catalog 행    -> films/{film_id}/final.json, catalog status=done
review    cache/problems/** + 문서 품질                                 -> qa/needs_review.csv
validate  schema/*.schema.json + 의미 규칙                              -> 오류 목록(종료코드 1)
```
`k`는 catalog `files` 칸 안 파일 순서(0부터).

- `sinkprep all` 차례는 catalog → convert → ocr → lines → parse → llm → asr → align → final → review → validate(`pipeline.STAGES`). video 는 `all` 에 들지 않는다.
- llm·asr·align·final 은 예외가 나도 다음 단계로 간다(`pipeline.SOFT`). 나머지 단계는 rc≠0 이면 멈춘다.
- `all` 안의 asr 는 모듈이 `ALL_CACHE_ONLY = True` 를 밝혔을 때만 부른다(`pipeline.CACHE_ONLY_FLAG`). 그때 asr 는 모델을 열지 않는다(아래 asr 약속).
- 영상 쪽 단계(asr, align, final)의 대상은 catalog 에서 제외되지 않고, `video_path` 가 있고, `is_primary=true` 인 문서이고, 열쇠는 film_id 다. align 은 `work/asr.json`, final 은 `work/asr.json`·`work/align.json` 이 있어야 대상이 된다.
- `films/{film_id}/work/` 에는 `asr.json`·`align.json` 만 둔다. validate 가 work/ 의 `*.json` 을 모두 검사하므로 다른 중간물은 `cache/` 에 둔다.
- `videos/`, `films/` 는 git 에 넣지 않는다(영상, 전사 전문, 대본 대사). `films/` 는 영상과 전사 캐시가 있어야 다시 만들 수 있다. `cache/` 도
  git 에 들어가지 않으므로 전사 묶음 캐시(`cache/asr`)는 그 PC 에만 있다.
- 저장소에 든 실제 대본 글: `tests/gold/*.jsonl`(파서 정답 창 26개, 파일마다 40~45줄, 합 1,150줄)과 `tests/fixtures/parse/*.jsonl`(15개,
  글이 든 줄이 파일마다 15~40줄, 합 456줄)은 원본 대본 줄의 글(text 칸)을 그대로 담는다. 2026-11-13 커밋(d1955de, 정답 창은 6fa14c3 에도)부터
  들어 있다(`tests/gold/raw/` 의 a·b 표시 파일에는 text 칸이 없다). 이것을 위 방침의 예외로 둘지, text 칸을 해시·줄 번호로 바꾸고 원문은
  `docs/` 에서 읽어 채점할지는 사용자 결정 대기다.
- 반면 `catalog.csv` 의 영상 칸·match·status 는 git 에 들어간다. validate [D1] 은 catalog 에서 영상이 있는 대표 문서마다
  `films/{film_id}/final.json` 을 요구하고 `sinkprep all` 은 validate 의 종료코드를 돌려주므로(`pipeline.run`), 영상·전사 캐시·films/ 가 없는
  새 클론에서는 `all` 이 '[D1] … final.json 이 없음' 으로 1로 끝난다. 같은 까닭으로 영상을 등록한 영화는 final 까지 만들어야 validate 를
  통과한다(final 이 translation_unmatched·Stale·검사 실패로 멈춘 영화가 하나라도 있으면 `all` 이 1). 그런 영화를 영상 쪽에서 빼는 것은
  `video --unregister`(영상 등록 약속).

## catalog.csv
명세 4.3 그대로. 추가 약속:
- `files`, `sha1`은 `|`로 나눈 목록. 경로는 ROOT 기준 상대 경로(슬래시), 예: `filmmakers_scripts/한국영화 시나리오/간신 [5691744].docx`, `extracted/54292/강원도의 힘.hwp`.
- `real_format`이 `mixed`면 파일마다 형식이 다르다. 파일별 형식은 `file_formats` 칸(`|` 목록)에 둔다(명세 4.3에 칸 하나 추가).
- `status=excluded`면 `note`에 사유(예: `not_script:exe`, `archive_container`, `duplicate_file_of:fm55536`).
- 압축 파일 자체는 문서가 아니다: catalog에 `status=excluded, note=archive_container:<풀린 폴더>` 행으로 남기고, 풀린 파일들이 문서가 된다.
- 읽기: `sinkprep.catalog.load() -> list[dict]` (CSV 행 dict, `files`는 list로 풀어서 `files_list`, `file_formats_list` 추가).

- 원본 `filmmakers_scripts/manifest.csv`는 대본이 아니라 다운로드 기록이다(글 번호·제목·첨부 원래 이름). catalog가 제목과 글 번호를 얻는 데 쓰고, A1 대상 파일에서는 뺀다.
- catalog 단계는 원본만 보고 정할 수 있는 칸을 채운다. 글자를 읽어야 정해지는 칸(`script_lang`, `is_translation` 보정, `extract_quality`, `is_primary`, `status`)은 parse 뒤 `review` 단계가 `catalog.finalize()`로 갱신한다. catalog 단계에서는 `script_lang=unknown`(제목으로 알 수 있으면 그 값), `is_primary`는 임시값.

**칸을 고치는 단계.** 모두 단계 끝에 `catalog.finalize(updates=…)` 한 번으로 쓴다(병렬 일꾼에서 부르지 않는다).

| 칸 | 단계 | 값 |
|---|---|---|
| 원본으로 정해지는 칸 | catalog | 위 약속 |
| `script_lang`, `is_translation`, `extract_quality`, `is_primary`, `status`(parsed, pending), note 의 parser 표시 | review | 아래와 2026-11-13 절 |
| `video_path`, `video_sha1`, `video_duration`, `video_lang` | video | 그 영화의 제외되지 않은 모든 문서 행에 같은 값. `video_path` 는 ROOT 기준 상대 경로 `videos/{film_id}{확장자}`(catalog.csv 가 git 에 들어가므로 절대 경로를 적지 않는다). `video_duration` 은 컨테이너 길이 소수 3자리 |
| `match`, `status=aligned` | align | match 는 일치율 소수 3자리 문자열 |
| `match`, `status=done` | final | 최종본을 만들면 done. 만들지 못한 영화가 done 이면 aligned 로 내린다 |

- review 는 `status` 가 aligned·done 이면 script.json 이 있는 한 그대로 둔다(되돌리면 맞추기 결과가 catalog 에서 사라진다).
- review 의 대표 대본(`is_primary`) 고르기 순서: script 있음, `match`(빈칸이면 −1), 번역 아님, mixed 아님, 품질(0.1 단위), 판본(shooting>final>bound>revised>unknown>draft), 품질, 줄 수, doc_id 번호가 작은 것. 영상과 맞춰 본 문서가 있으면 일치율이 높은 것이 먼저다.
- catalog 를 다시 만들 때 `carry_over` 가 영상 칸·match·status 를 이어받는다(2026-11-13 절). film_id 는 차례 번호(`local_{i:04d}`, 문서 번호 순)라서 원본 문서가 늘거나 줄면 밀릴 수 있다.
  `carry_over` 는 film_id 가 달라진 행을 건너뛰므로, 번호가 밀린 문서는 영상 칸·match·status(`LATER_COLUMNS`)를 이어받지 못해 영상 등록이
  catalog 에서 소리 없이 빠지고, asr·align·final 이 그 영화를 대상에서 뺀다. 옛 `films/local_XXXX` 는 다른 영화의 ID 가 된다. 그때는 `video` 로
  다시 등록하고 옛 `videos/`, `films/`, `cache/state/*/{film_id}.json` 을 정리한다.

## lines.jsonl 머리 행
첫 행은 머리 행이다: `{"schema":"sink.lines/1","doc_id":"fm55574","files":["…"],"made":{"stage":"lines","version":"0.4.1","sinkprep":"0.1.0"}}`.
둘째 행부터 명세 4.4.1의 줄 행(`n`은 0부터). 그래서 모든 JSON/JSONL 파일이 schema 버전을 가진다(E1).
띄어쓰기를 원본 글에 맞춘 파일(`from:"source"`)과 원본 글과 견주지 못한 파일(`from:null`)은 `made.spacing` 에 파일별 기록이 붙는다
(2026-11-15 절 lines 0.4.1). 2026-11-15 기준 기록이 있는 문서 12개 가운데 맞춘 문서가 10개, 견주지 못한 문서가 2개다.

## 형식 값
`real_format`: hwp3, hwp5, hwpx, doc, docx, rtf, pdf_text, pdf_scan, pdf_broken(열리지 않음), txt, xls, archive, other.
lines의 `src`: hwp, doc, docx, rtf, pdf_text, ocr, txt, xls(엑셀 표 대본: 한 행 = 한 줄, 칸 사이 탭, `cell`={table:시트, row, col:첫 칸}, xlrd 로 읽음).

## convert 결과
- `cache/pdf/{doc_id}/{k}.pdf`
- `cache/convert_log.jsonl` 한 줄: `{"doc_id","k","src","real_format","ok","method","pages","error","seconds"}`.
  `method`: hancom | fallback_hwp3_text | fallback_hwp5_bodytext | fallback_hwp5_prvtext | fallback_hwpx | fallback_antiword | fallback_rtf | fallback_docx.
- 대체 경로로 글자만 얻은 경우: `cache/pdf/{doc_id}/{k}.txt` (UTF-8 평문, 레이아웃 없음).

## ocr 결과 `cache/ocr/{doc_id}/{k}.json`
```json
{"engine":"…","engine_version":"…","source":"cache/pdf/fm1/0.pdf|filmmakers_scripts/…pdf","dpi":300,
 "pages":[{"page":1,"width":2480,"height":3508,
   "lines":[{"text":"S#1. 빵집. 낮","box":[x0,y0,x1,y1],"conf":0.97}]}]}
```
`box`는 픽셀 좌표. lines 단계가 쪽 크기로 나눠 0~1로 바꾼다.

## llm 단계 약속 (sinkprep/llm.py 백엔드, sinkprep/llm_parse.py 단계, 설계 notes/llm_parse.md)
- 대상: 기본은 parse 단계 문제(parse_low_dialogue, speaker_missing, extract_broken = typed_ratio 낮음)가 있는 문서
  (`cache/problems/parse/{doc}.json`, 즉 규칙 판 기준) + 두 언어가 섞인 문서(catalog script_lang mixed) + 전에 llm 단계 지문이 남은
  문서(`--docs` 로 돌린 문서도 다음 기본 실행에서
  같은 판을 다시 놓도록). `--docs` 로 지정, `--limit`, `--force`(지문 무시, LLM 은 캐시 그대로),
  `--workers`(동시 claude 수, 기본 3), `--images`(OCR 줄 조각에 원본 쪽 그림을 함께). 형식 프로필이 이름 표기 없음(name_style
  none: 자막 파일·녹취·시놉시스)이면 라벨링하지 않고 규칙 판.
- 백엔드: Claude Code 구독 `claude -p`. 모델 ID 고정 `llm.MODEL`(별칭 금지), effort 고정 `llm.EFFORT`. 도구 전부 끔(`--tools ""`),
  `--safe-mode`·`--setting-sources ""`·`--strict-mcp-config`·`--disable-slash-commands`·`--no-session-persistence`, 빈 작업 폴더
  `cache/llm_work`, 부모 세션 환경 변수(CLAUDECODE·CLAUDE_CODE_*) 안 넘김. 응답 init 의 도구·MCP 와 modelUsage 의 모델을 검사
  (성공 응답에 init 이 없으면 쓰지 않음).
- 라벨 바로잡기(`fix_labels`, 결정적): 문서 글에 없는 화자 이름은 버림, 전환 말 없는 전환은 지문, INT./EXT. 꼴 문서의 영어 촬영
  머리(ANGLE ON·POV·INSERT -)는 지문, 지문으로 붙은 `이름: (행동만)` 은 그 인물의 대사. 이름에 붙은 연기 지시(`광해(버럭)`)는
  paren 에도 넣는다.
- 캐시: `cache/llm/{key}.json`, key = sha1(캐시 형식, 모델 ID, effort, 프롬프트 버전, 입력 해시(시스템·사용자 글·스키마·그림 식별자)).
  있으면 부르지 않는다. 다시 묻기는 그 파일을 지운다. 구조 검증 실패·일시 오류는 2번 더(모두 3번), 한도·과부하는 30/90/270초
  백오프 뒤 멈춤(그 실행의 남은 캐시 없는 호출은 바로 실패, 캐시 있는 것은 계속).
- 파일: 규칙 판 `cache/parse_rules/{doc}.json`(parse 가 쓴 바이트 그대로), LLM 판 `cache/parse_llm/{doc}.json`(늘),
  `docs/{doc}/script.json` = LLM 판이 validate 를 통과하고 품질 지표가 뚜렷이 나쁘지 않으면(문제 수가 늘지 않고 score 가 0.03 넘게
  떨어지지 않음) LLM 판, 아니면 규칙 판. 정답 표본에서 LLM 판이 모든 묶음에서 더 정확해 LLM 판을 기본으로 둔다(notes/llm_parse.md).
  정답 표본(tests/gold)은 Claude 서브에이전트(llm.MODEL 과 같은 claude-opus-5-5) 둘이 따로 표시하고 갈린 줄을 셋째 서브에이전트가 판정한
  것이다(사람 정답 아님).
  dropped·front 줄을 규칙 판보다 max(20, 5%) 넘게 늘린 판, 조각이 하나라도 실패한 판은 쓰지 않는다.
  parser.name `sink-parse-llm`, parser.version `{VERSION}+{PROMPT_VERSION}`.
- 지문 `cache/state/llm/{doc}.json`: 조립 버전·프롬프트 버전·모델·effort·그림 여부·parse 버전·lines sha1·규칙 판 sha1 + 결과
  sha1(out_sha1). docs 의 script.json 이 out_sha1 과 다르면(예: `parse --force` 가 규칙 판을 다시 씀) 다시 놓는다(LLM 은 캐시라
  바이트 동일). parse 는 지문이 같으면 건너뛰므로 LLM 판을 덮지 않는다.
- 문제: `cache/problems/llm/{doc}.json` 에 LLM 판을 쓸 때 그 판의 품질 문제(detail 앞 `[LLM 판]`, 칸 parser), 실패 때 problem
  other(kind llm_fail, 규칙 판 유지). review 는 문서마다 지금 쓰는 판의 품질 문제만 올린다(LLM 판이면 parse 단계 문제를 뺌).
- review: LLM 판 문서는 catalog note 에 `parser:sink-parse-llm`(규칙 판이면 뗌), extract_quality 는 지금 판의 score.
- `sinkprep all`: parse 뒤 review 앞. claude 가 없거나 실패해도 전체가 멈추지 않는다(pipeline.SOFT). all 에서는 캐시만 쓰고
  (캐시에 없는 문서는 규칙 판, 구독 사용 없음) `--llm` 을 줄 때만 부른다. `sinkprep llm` 은 늘 부른다.
- cache/llm 은 빈 캐시로 다시 부르면 결과가 조금 달라질 수 있다(같은 입력에도 모델 출력이 흔들림). 그래서 저장소에
  `data/llm_cache.zip` 으로 묶어 넣는다(`tools/llm_cache_zip.py pack` = 같은 캐시면 같은 바이트, 계정 메타데이터 session_id·uuid·
  credential_expires_at 뺌; 새 PC 에서는 `unpack` 뒤 `sinkprep all` 이 claude 를 부르지 않고 대본 쪽은 같은 결과. 다만 영상을 등록한
  영화가 catalog 에 있으면 그 PC 에 영상·전사 캐시·films/ 가 없는 한 validate [D1] 이 실패해 `all` 이 1로 끝난다. 흐름과 파일 절). 화자 이름과 형식
  프로필의 짧은 대본 인용이 들어 있다(사용자 결정 2026-11-13). `sinkprep llm` 과 `sinkprep align --llm` 으로 새로 부른 뒤에는 pack 해서 함께 커밋한다.
  큰 실행 앞뒤 사본은 `backup/llm_cache_*.zip`(gitignore).
- align 의 LLM 짝 판정(`align_llm.py`)도 같은 백엔드와 캐시를 쓴다: `llm.call`, 고정 `llm.MODEL`·`llm.EFFORT`(바꾸면 기존 캐시가
  모두 빗나가므로 건드리지 않는다), task `align_pair`, 프롬프트 버전 `align_llm.PROMPT_VERSION`. 응답은 번호와 열거값만 받게 해서
  `llm_cache.zip` 에 대사 글이 들어가지 않게 한다. 부르는 것은 `sinkprep align --llm` 뿐이고, 옵션 없는 `align` 과 `all` 은 캐시만 쓴다.
  `align --llm` 으로 새로 부른 뒤에도 `tools/llm_cache_zip.py pack` 으로 `data/llm_cache.zip` 을 갱신해 함께 커밋한다. 빠지면 다른 PC 에서
  판정이 빠져 `decisions_sha1` 이 달라지고 align.json·final.json 이 바뀐다.

## 영상 등록과 받아쓰기 약속 (video.py, media.py, asr.py)

**영상 등록(`sinkprep video <영상> --docs <doc_id> [--lang] [--force]`)**
- `--docs` 문서의 film_id 로 등록한다. 영상 줄이 없는 파일(소리만)은 받지 않는다.
- `videos/{film_id}{확장자}` 로 하드링크하고 안 되면 복사한다. 원본 영상은 고치거나 옮기지 않는다.
- 소리와 대응표를 다 만든 뒤에야 catalog 영상 칸을 쓴다(소리를 못 꺼낸 영상이 catalog 에 남지 않게).
- `--lang` 이 없으면 그 영화에 이미 적힌 `video_lang` 을 지키고, 그것도 없으면 한국영화 게시판(ko_film) 문서는 ko, 아니면 unknown.
  지역 꼬리가 붙은 값(`ko-KR`, `en-US`)은 기본 언어(`ko`, `en`)로 줄여 적고 까닭을 출력한다(`video.base_lang`). final 이 줄 언어를
  기본 언어로 견주기 때문이다. 꼴이 틀린 값은 받지 않는다. 다만 catalog 를 손으로 고쳐 넣은 `ko-KR` 은 validate 가 막지 않고, 그러면 final 의
  added 줄이 모두 한국어가 아닌 줄로 잡혀 translation_unmatched 로 멈춘다.
- 그 영화에 sha1 이 다른 영상이 이미 있으면 `--force` 가 있어야 바꾸고, 바꾸면 `videos/` 의 옛 파일을 지운다(옛 영상의 cache 는 남음).
  값이 같고 새로 만든 것이 없으면 skip.
- 영상 sha1 은 `cache/state/filesha1/` 에 경로·크기·고친 시각과 함께 캐시한다(내용이 바뀌어도 크기와 시각이 같으면 못 본다).
  그래서 asr 가 그 영상을 처음 받아쓸 때 한 번 다시 읽어 catalog 의 `video_sha1` 과 견준다.

**등록 해제(`sinkprep video --unregister --docs <doc_id> [--force]`, `video.unregister`)**
- 영상 경로와 `--lang` 없이 쓴다. 그 문서의 film_id 의 모든 문서 행에서 영상 칸 넷과 `match` 를 비우고, status 가 aligned·done 이면 parsed 로
  되돌린다(`catalog.finalize` 한 번). asr·align·final 의 문제 기록(`cache/problems/{stage}/{film_id}.json`)을 `util.set_problems(…, [])` 로 지운다.
- `videos/{film_id}.*` 는 다른 하드링크가 있는 파일(`st_nlink > 1`, 원본이 다른 링크로 남음)과 `.part` 만 지운다. 링크가 하나뿐인 파일은 링크를
  못 해 복사한 사본인지 원본 자체인지 알 수 없어 `--force` 일 때만 지운다.
- `films/{film_id}/` 결과, `cache/audio` 의 wav·대응표, `cache/asr`, `cache/state/{asr,align,final}/{film_id}.json` 은 지우지 않고 남은 것을 출력한다.
  다시 등록하면 같은 캐시를 이어 쓴다. 바뀐 것이 없으면 skip.

**시각과 소리(media)**
- 결과물(asr.json, align.json, final.json)의 모든 시각은 영상 시각 = 컨테이너 PTS − 컨테이너 시작 시각(format start_time), 0 이상이다.
  플레이어·자막·`ffmpeg -ss` 가 쓰는 시각이고 catalog `video_duration`(컨테이너 길이)과 같은 기준이다.
- 소리 시각 = `cache/audio/{video_sha1}.wav`(16kHz 모노 pcm_s16le)의 표본 번호 ÷ 16000. 소리는 틈을 메우지 않고 꺼낸다.
- 대응표 `cache/audio/{video_sha1}.timemap.json`(`media.TimeMap`): mode(map, padded, identity), frame(패킷 길이), segments
  ([소리 시각, 영상 시각] 구간 시작), audio_duration, video_duration(소리 끝의 영상 시각), gaps, gap_total, made.
  패킷 표는 정수 pts·duration × time_base 로 읽고, 패킷 길이는 디코딩한 프레임 표본 수로 잰다. (영상 시각 − 소리 시각)이
  0.002초 넘게 바뀌면 새 구간이다.
- 검증: 패킷 수 × 패킷 길이가 wav 길이와 0.1초 안, 소리 끝의 영상 시각이 소리 스트림 끝과 0.1초 안(어긋나도 컨테이너 길이와
  맞으면 받는다). 실패할 때만 틈을 무음으로 메워(`media.PAD_FILTER`) 다른 이름으로 뽑고, 길이를 확인한 뒤에만 wav 와 바꾸며
  mode 를 padded 로 적는다. 대응표와 wav 는 길이가 맞아야 짝이다. 운영 차례(video·asr)는 `build_timemap` 이 이 짝 검사(`_fits`)를 하고 짝이
  아니면 다시 만든다. 읽기만 하는 `load_timemap` 은 같은 검사를 하는 시험·조사용 함수다.
- ffprobe 정보 300초, 패킷 표 1800초, 소리 추출 7200초를 넘으면 `MediaError`.

**받아쓰기(`sinkprep asr`)**
- 출력은 `films/{film_id}/work/asr.json`(sink.asr/1) 하나다. `--docs` 로 대표가 아닌 문서를 줘도 그 영화의 대표 문서로 돈다.
- 옵션: `--passes id=모델[,id=모델]`(모델 large-v3-turbo(= turbo), large-v3), `--rescue`/`--no-rescue`, `--device auto|cpu|cuda`,
  `--threads`(기본 6). `--passes`·`--device`·`--rescue` 를 주지 않으면 그 영화의 단계 지문 값을 이어 쓰고, 처음이면 `asr.DEFAULT_*`
  (plain=large-v3-turbo, auto, rescue 없음). 기본값은 asr.py 한 곳에만 둔다.
- 장치: auto 는 CUDA 장치가 있고 여유 VRAM(nvidia-smi)이 5000MiB 이상일 때만 cuda float16, 아니면 cpu int8. nvidia-smi 가 PATH 에 없거나
  실패하면 여유를 0 으로 보아 경고 없이 cpu 다(`asr.free_vram_mib`). cuda 는 열 때
  한 번 인코딩해 보고, 실패하면 auto 는 cpu 로 가고 `--device cuda`(또는 지문의 장치가 cuda 인 영화를 옵션 없이 돌린 것)는 오류
  (`AsrUnavailable`)다. cuda 를 열기 전에 `site-packages/nvidia/*/bin`(선택 의존성 asr-gpu 의 cuBLAS·cuDNN)을 DLL 찾는 길에 더한다.
  모델은 한 번에 하나만 올리고 로컬 캐시를 먼저 찾는다.
- 장치와 asr.json: asr.json 을 어느 판의 묶음 캐시로 조립할지는 그 실행에서 풀린 장치가 정한다(꼬리표에 장치가 듦). 추격자는 CPU 판·GPU 판
  캐시가 모두 빠짐없이 있고 지문은 `device=cuda` 다. `--device cpu` 를 주거나 `--device auto` 가 cpu 로 풀리면(GPU 가 바쁨, VRAM 부족, DLL 없음)
  지문이 달라져, 다시 받아쓰지 않고 CPU 판 캐시로 asr.json 을 조용히 갈아 쓴다(경고 없음). 그 뒤 align·final 도 따라 바뀐다. GPU 판을 지키려면
  `--device` 를 주지 않거나 cuda 로 준다.
- 말 구간: faster-whisper 에 든 Silero(`asr.VAD_OPTIONS`: 문턱 0.35, 말 100ms, 쉼 300ms, 덧댐 150ms, 최대 25초)를 영화 전체에 한 번
  돌려 `cache/asr/{video_sha1}/vad.json` 에 둔다.
- 묶음: 말 구간을 25초 이하로 묶는다(앞 묶음과 틈 1초 미만이면 붙임). 앞뒤로 0.1초(이웃과의 틈의 절반까지) 덧대고 이웃과 겹치지 않는다.
- 디코딩: 묶음마다 따로(`asr.DECODE`: beam 5, 낱말 시각, `vad_filter` 없음, 앞 글 조건 없음). 온도는 faster-whisper 기본 되돌이(씨앗
  없음)라 캐시를 지우고 다시 받아쓰면 같다는 보장이 없다. 힌트는 주지 않는다.
- 묶음 캐시: `cache/asr/{video_sha1}/{꼬리표}/{a:010d}-{b:010d}.json`, 꼬리표 = `{모델}@{스냅숏 앞 8자}-{장치}-{계산형}`, a·b 는 덧대기
  전 소리 표본 번호. 기록은 key, a, b, pa, pb, lang, words(소리 시각), segs(avg_logprob, no_speech_prob, compression_ratio,
  temperature 등), seconds. 열쇠 = sha1(DECODE_VERSION, 모델 저장소·스냅숏·장치·계산형·라이브러리 판, DECODE 옵션과 힌트, 언어,
  a, b, pa, pb, 소리 종류 raw/padded). pass id 와 스레드 수는 넣지 않는다. 장치가 꼬리표와 열쇠에 들므로 CPU 판과 GPU 판 캐시는
  따로 쌓인다. 열쇠가 다르거나 망가진 기록은 그 묶음만 다시 받아쓴다. 파일 이름에는 범위만 있으므로 힌트를 준 받아쓰기는 하위 폴더를
  따로 쓴다(`chunk_path(..., sub=)`).
- 잠금: 한 영화는 한 프로세스만 받아쓴다(`cache/asr/{video_sha1}/.lock`, 죽은 프로세스의 잠금은 가져간다).
- rescue(`--rescue`): 말 구간 찾기가 놓친 대사를 되찾는다.
  - 창: 소리 전체를 빈틈없이 덮는 약 25초 창 두 벌(`cache/asr/{video_sha1}/windows.json`). 벌 A 는 25초마다 그 앞 5초 안에서 가장
    조용한 곳을 경계로, 벌 B 는 벌 A 창마다 가운데 ±2.5초 안에서 가장 조용한 곳을 경계로 삼아 두 벌의 경계가 엇갈린다.
  - 받아쓰기: plain 의 모델로 같은 방식(덧댐 없음, 같은 열쇠 규칙)으로 받아써 `{꼬리표}/full/` 에 둔다.
  - 남기는 낱말: 벌 A 낱말 가운데 (가) 가운데 시각이 plain 묶음의 덧댄 범위 밖이고, (나) plain 낱말과 같은 소리가 아니며(겹친 시간이
    짧은 쪽 길이의 절반 이상이거나, 글이 같고 사이 0.1초 이하면 같은 소리), (다) 한 창 안에서 같은 낱말이 4번 넘게 잇달지 않는 것을
    세그먼트 경계와 0.6초 넘는 쉼으로 조각낸다. 조각 앞뒤 1초 안에서 시작이 다른 벌 B 창의 이어진 낱말 범위가 그 조각과 닮고
    (rapidfuzz ratio, 글자·숫자만, 0.6 이상) 닮은 벌 B 낱말 글자의 절반 이상이 그 조각과 시간이 겹칠 때만 남긴다.
  - 버릇 글귀: 7자 이상 조각 글이 벌 A 의 서로 다른 창 3개 이상에서 똑같이 나오고 그 절반 넘게가 벌 B 와 맞지 않으면 버릇 글귀이고,
    그 글귀와 닮은(0.6 이상) 조각은 두 벌이 맞아도 버린다. 목록은 `asr.film_habits`.
  - 결과는 pass id `rescue`, context `full_window_stable`. 두 벌 자체와 버린 조각은 asr.json 에 넣지 않는다(`asr.film_rescue_fragments`
    로 읽음). 창마다 말 구간 찾기를 새로 돌려 조각을 거르지 않는다(추격자에서 진짜 대사 조각의 20%가 거기서도 말로 잡히지 않음).
    영화 맨 앞 약 12초(두 벌의 첫 창이 함께 0초에서 시작)와 plain 묶음 안의 빈 자리는 다루지 않는다.
- 조립: 캐시만으로 한다. 소리 시각을 대응표로 영상 시각으로 옮겨 소수 3자리로. 낱말 번호는 묶음 차례·묶음 안 차례(rescue 는 시각 순)로
  `words[k].i == k`. 묶음 첫 낱말의 시작이 잘라 낸 조각 시작에서 0.02초 안이면 묶음 시작 + 0.05초로 당긴다(낱말 길이 0.1초는 남김).
  글자를 못 푼 낱말(U+FFFD)은 버린다. 같은 캐시면 asr.json 바이트가 같다. `--force` 는 다시 조립만 하고, 받아쓰기부터 다시 하려면 그
  꼬리표 폴더를 지운다.
- asr.json 에는 스키마 칸만 쓴다. 모델·옵션·규칙·코드 판은 `passes[].model` 문자열에만 적는다(예: `faster-whisper 1.2.1 | …large-v3-turbo@0a363e91
  | ct2 4.8.3 cuda float16 | beam5 nocond tfallback | silero-v6 thr0.35 … | firstword +0.05 | sinkprep.asr 0.3.0 decode1`, rescue 는 그 뒤에
  rescue 규칙). 세그먼트 통계는 묶음 캐시에만 있고 `asr.film_records`, `asr.pass_word_stats` 로 읽는다.
- `all` 안(`args.cmd == "all"`): 모델을 열지 않고(cuda 도 열어 보지 않음) 말 구간 찾기도 하지 않는다. 지문의 옵션으로 캐시만으로 조립하고,
  받아쓴 적이 없거나 캐시에 없는 것(말 구간 캐시 열쇠가 다른 것 포함)이 있으면 asr.json 을 그대로 두고 `wait` 로 알린다. 다만 소리와
  대응표는 캐시만 쓰는 길에서도 `media.extract_audio`·`media.build_timemap` 을 부른다. 대응표가 낡았으면(made 가 다름: `media.VERSION` 을
  올린 경우, 또는 wav 와 길이가 안 맞음) `all` 안에서도 ffprobe 패킷 표를 다시 뽑고(최대 1800초), 검증에 실패하면 소리를 무음으로 메워 다시
  뽑아 wav 를 바꾼다. 그러면 소리 종류가 padded 가 되어 묶음 캐시 열쇠가 모두 빗나가므로 다시 받아써야 한다. `windows.json` 열쇠가
  다르면 소리 전체를 읽어 창도 다시 나눈다. 그래서 `media.VERSION` 을 올린 뒤에는 `all` 이 아니라 `sinkprep asr` 로 먼저 돌린다.
- `cache/asr/{video_sha1}` 은 다시 만들 수 없는 결과물로 다룬다(온도 되돌이에 씨앗이 없다). 큰 실행 앞뒤로 `backup/` 에 사본을 두어야 한다.
  추격자의 묶음 캐시(CPU·GPU, turbo·large-v3 네 폴더와 vad.json, windows.json, 파일 3,396개)는 `backup/asr_cache_20261115.zip` 에 사본이
  있다(2026-11-15 바이트 대조로 모두 같음). `backup/` 도 gitignore 라 같은 PC 에만 있고, PC 밖 사본은 할 일이다.
- 끝난 받아쓰기를 캐시에서 읽는 공개 함수: `film_records`, `pass_word_stats`, `speech_holes`, `film_rescue_fragments`, `film_habits`. 지금 뒤
  단계가 부르는 것은 align 의 `film_habits` 하나뿐이고, 나머지는 읽기용으로 열어 둔 함수다(`speech_holes` 는 asr 안에서 쓴다). 소리에 글을 맞춰
  점수를 내는 `encode_clip`·`force_text` 와 힌트를 주는 `transcribe_clip(hotwords=, initial_prompt=)` 도 있지만 지금 align 은 쓰지 않는다
  (ctx pass 를 만드는 코드는 없다).

## 맞추기 약속 (textnorm.py, align.py, align_llm.py)

**입출력.** `docs/{doc_id}/script.json` + `work/asr.json` → `work/align.json`(sink.align/1: script_doc, asr_sha1, method, ops, scenes).
op 하나 = 최종본 대사 항목 하나(명세 C2). `scenes` 는 대본 씬마다 `{scene, start, end, how}` 이고 그 씬의 시각 있는 대본 대사(added 는 넣지
않음)만으로 잡는다: 옮겨진 줄이 절반을 넘으면 그 줄들의 범위(how moved), 아니면 옮겨진 줄을 뺀 범위이고 닻으로 맞춘 줄이 있으면 anchors,
없으면 lines, 시각 있는 대사가 없으면 start·end null 에 none. final.json 의 씬 구간(`timeline.scene_spans`)은 그 씬으로 분류한 added 도 넣고
대사가 없는 씬은 추정 시각을 쓰므로 둘은 다를 수 있다(추격자: 구간 있는 align 씬 93개 가운데 51개). CBoE 가 읽는 것은 final 쪽이다. 입력은 이 둘 말고도 셋이 더 있다: 버릇 글귀(`align.film_habits` → `asr.film_habits`, asr 단계 지문의
`tags` 로 `cache/asr` 의 rescue 창 기록을 읽음), LLM 짝 판정 캐시(`cache/llm`), catalog `video_duration`(항목 추정 시각과 time_fix).
같은 script.json·asr.json·받아쓰기 캐시(버릇 글귀)·LLM 캐시·영상 길이면 align.json 바이트가 같다. 받아쓰기 캐시나 asr 지문이 없거나 읽히지
않으면 `align.film_habits` 가 빈 dict 를 돌려주고 조용히 넘어가므로, 버릇 글귀 거르기가 빠진 다른 align.json 이 나오고 지문의
`habits_sha1` 이 달라져 덮어쓴다(새 PC, cache/asr 를 잃은 PC 에서 `all`).

**비교 규칙은 `textnorm` 한 곳에 둔다**(align, final, validate 가 같은 함수를 쓴다).
- `key(text)`: same 판정 열쇠. NFC, 영문 소문자, 공백·문장부호·기호를 뺀 글자열. 한글 낱자모(ㅋㅋ, ㅠㅠ)와 변형 선택자는 글자로 치지
  않고 전각 영문·숫자는 반각으로 접는다. 숫자에 붙은 부호 셋만 남긴다: 소수점은 '점', 숫자 바로 뒤 % 는 그대로, 숫자 줄과 숫자 줄
  사이의 경계는 경계 글자 하나(`textnorm.SEP`, 사이에 낀 것이 공백이든 쉼표든 붙임표든 같음). 자릿수 쉼표(3,000)는 없는 것으로 친다.
  그래서 1.5 와 15, '2, 3일' 과 '23일', 100% 와 100 은 changed 이고, 016-9265-4885 와 016 9265 4885 는 same 이다.
- `loc(text, other)`: 자리 찾기 전용 글자열. 숫자와 5글자 이하 영문 약어를 읽는 꼴로 바꾸되, 읽는 꼴이 여럿이면 맞은편 글(`other`)에
  있는 것을 고르고 없으면 앞뒤 말을 보지 않는 기본 읽기다. 판정에는 쓰지 않는다.
- `verdict(script_text, heard)` → ("same", []) 또는 ("changed", diffs). same 은 `key` 가 같을 때다. 낱말별 열쇠를 이은 것은 이은 글의
  열쇠와 다를 수 있으므로 판정은 늘 이은 글로 부른다.
- `diffs`: 양쪽에서 같은 낱말을 먼저 짝짓고(숫자는 읽는 꼴로도) 그 사이만 글자로 견줘 어절 단위 조각을 만든다. 종류는 틀림없을 때만:
  number(한쪽을 읽는 꼴로 바꾸면 같아지거나 숫자 줄을 나눈 자리만 다름), spelling(소리 나는 대로 고친 꼴이 같거나 굳은 표기 흔들림:
  할께, 꺼야, 알어, 니가, 임마 등. 소리가 같은 딴 낱말도 여기), filler(그 자리에서 군말. 어, 음, 아 같은 틀림없는 군말, 줄 맨 앞의 '자',
  쉼표·말줄임표가 붙은 그·저·뭐·아이·오. 부르는 말 '야' 는 아님), repeat(남아 있는 말을 되풀이했거나 뒤 낱말의 앞머리, 끝소리
  늘임), 나머지 word.

**대본 줄.** 말 있는 대사(`key` 가 빈 글이 아닌 dialogue 요소)만 맞춘다. 말 없는 대사(문장부호·지시뿐)는 absent 이고 conf 는 null,
`method.stats.silent_turns` 로 따로 세며 일치율 분모와 absent_many 에서 뺀다. 내레이션·노래도 다른 대사와 같이 맞춘다.

**pass 구실(`pass_roles`).** asr.json pass 의 context 로 가른다: null 이면 대본 없이 받아쓴 후보이고 id 가 plain 인 것(없으면 맨 앞)이 기준,
나머지가 대안. `script_hotwords`·`script_window` 는 ctx, `full_window_stable` 은 rescue(각각 첫째만). 그 밖은 쓰지 않고 `method.roles.unused`.

**한 pass 맞추기(`match_pass`, 기준과 대안 pass 마다).**
1. 닻: 양쪽에 한 번씩만 나오는 k-gram(k = 5, 4, 3. 짧은 k 는 틈이 좁을 때만)을 씨앗으로 최대 일치 블록을 만들고, 대본 순서와 영상 순서가
   함께 늘어나는 사슬 가운데 일치 글자 합이 가장 큰 것을 뼈대로 고른다(가중 LIS). 뼈대가 덮은 글자가 5% 미만이면 8-gram 을 3번까지
   나와도 씨앗으로 쓴다. 한 줄의 닻이 시간이나 받아쓰기 글자로 너무 벌어지면 끊는다.
2. 옮겨진 덩어리: 뼈대에 못 든 블록에서 다시 사슬을 뽑아, 가까이 모인 블록의 일치 글자 합이 `MOVED_MIN`(20) 이상이면 받아들인다. 씬
   경계는 쓰지 않는다.
3. 닻 사이: 줄마다 이어진 낱말 범위 후보의 정확히 맞은 글자 수(LCS)를 재서, 순서를 지키며 (맞은 글자 − 0.5 × 안 맞은 글자) 합이 가장 큰
   짝을 고른다(DP). 받아들이는 기준은 줄 글자 수로 다르다: 3글자 이하는 글자 그대로 같아야 하고, 4~7글자는 맞은 글자 ÷ 줄 글자 0.6
   (그리고 이어진 두 글자나 자모 닮음 0.6 이 있어 맞은 글자가 흩어져 있기만 하지 않을 것), 8글자 이상은 0.4, 둘 다 글자 닮음 0.4 이상.
   8글자 이상 이어진 닻 블록이 있으면 0.2 만 넘어도 받는다(줄의 일부만 말한 경우). 짧은 줄은 맞춘 이웃 대본 줄이 20초 안에 있어야 남긴다.
4. 다시 짝짓기(`_repair`): 맞춘 두 줄 사이 창에서 못 맞춘 줄과 남은 낱말을 순서대로 다시 짝짓는다. 같은 뜻의 짧은 대답(`ANSWER_CLASSES`:
   예·네, 응·어·엉, 아니요·아뇨·아니오·아냐, 아녜요·아니에요·아니예요. 판정은 그대로 changed), 자모가 닮은 줄, 줄의 일부만 말한 줄(자모
   0.85), '요' 만 다른 되묻기를 받는다. 짧은 말은 그 창에 우연히 있을 기대 횟수가 0.1 이하일 때만, 0.1~0.2 면 이웃 맞춘 줄에서 20초 안일 때만
   받는다. 이웃 줄과 순서만 바꿔 말한 줄은 `_reclaim_moved` 가 되찾는다.

**pass 고르기(`decide`, decided_by rule).** 고르기 규칙의 글은 `align.CHOOSE_TEXT` 이고 `method.choose` 에 남는다.
- 견주기 전에 기준·대안 범위를 서로의 범위 소리 안에 있는 제 남은 낱말로 넓힌다(다른 pass 의 그쪽 글이 받쳐 줄 때만, 다른 pass 가 그 줄을
  0.25 넘게 못하게 맞추지 않았을 때만, 바깥 끝의 대답·군말과 물음표 너머 차례는 빼고).
- 기준이 대본과 같으면 기준, 아니고 대안이 같으면 대안. 둘 다 다르면 대안의 글자 닮음이 `ALT_MARGIN`(0.02) 넘게 높을 때만 대안이다. 대안을
  고르면 added 로 떨어질 기준 낱말의 글자를 대안 쪽의 안 맞은 글자로 센다. 글자 닮음이 똑같으면 자모 닮음으로 본다.
- 가려 적은 글(한글 옆 X, 자음만 적은 꼴)이 든 후보는 다른 후보가 있으면 고르지 않는다.
- 두 범위가 겹치거나 틈이 0.5초 이하일 때 같은 소리로 보고 견준다. 기준이 못 맞춘 줄을 대안만 맞췄으면 다른 줄과 겹치지 않을 때 받는다.
  소리가 겹쳐 못 고른 대안은 모든 줄을 고른 뒤 다시 본다.
- ctx pass 가 있으면 기준으로 고른 줄과 아무도 못 맞춘 줄에서만, 두 pass 가 다르게 적은 낱말의 평균 로그 확률로 견준다(decided_by acoustic,
  허용폭 tau 는 그 영화에서 잰다). 지금 ctx pass 를 만드는 코드는 없다.

**한 번 난 소리는 한 줄에만.** op 의 `asr` 는 한 pass 안의 이어진 낱말 범위(`["plain:12", "plain:15"]`, 끝 포함)다. 같은 pass 안에서는 줄
범위가 겹치지 않고, 다른 pass 의 줄끼리는 같은 소리인 낱말(`_same_word`: 겹친 시간이 짧은 쪽 낱말 길이의 절반 이상)을 함께 갖지 않는다
(`_Claims`). 다른 pass 에서 고른 줄이 덮은 기준 낱말은 added 로 세지 않는다. 한 대사가 두 pass 에 나뉘어 있으면 지금은 한 줄로 묶지 못한다.

**rescue.** 아무도 못 맞춘 줄만 찾는다. 한 줄의 범위가 다른 pass 의 말을 건너뛰어 이어지지 못한다. 줄에 안 쓰인 rescue 낱말은 다음 넷 가운데
하나라도 맞으면 added 로 만들지 않는다(`rescue_skip`): (가) 낱말 가운데 시각이 기준 pass 의 묶음(asr.json chunks) 안, (나) 기준 pass 낱말과 같은
소리(`_same_word`: 겹친 시간이 짧은 쪽 길이의 절반 이상), (다) 그 낱말 시작부터 끝 + `RESCUE_ECHO_S`(0.3초) 사이에 다른 pass 의 낱말(기준 pass
낱말과 대안 pass 에서 고른 줄의 낱말)이 같은 소리(자모 닮음 `RESCUE_ECHO_SIM` 0.8 이상)로 시작, (라) 이미 다른 줄이 가진 낱말. 다른 줄의 시간
범위와 겹치는지는 보지 않으므로 rescue added 가 다른 줄과 시간이 겹칠 수 있다(추격자: rescue added 66줄 가운데 8줄, I0484 가 I0485 를 품음).
`method.choose`(`align.CHOOSE_TEXT`)의 '어느 줄의 시간 범위와도 겹치지 않고' 는 이 구현과 다르다(글을 고치려면 align.json 이 바뀌므로 코드 쪽 할 일).

**여러 pass 다시 짝짓기(`_repair_across`).** pass 를 다 고른 뒤, 고른 줄(어느 pass 든) 사이 창에서 못 맞춘 줄과 기준·rescue 의 남은 낱말을
시각 차례로 섞어 한 번 더 다시 짝짓는다. pass 가 바뀌는 곳에서 끊어 한 줄의 범위는 한 pass 안에 있고, 다른 pass 줄과 소리가 겹치는 짝은 버린다.

**남은 낱말(`leftover`).** 옆 줄에 붙이는 것은 두 경우뿐이다.
- 근거 있는 붙이기: 줄 쪽 낱말부터 붙는 글자의 절반 이상(자모로 보면 0.7)이 대본 글자와 새로 맞을 때, 쉼 0.8초까지.
- 근거 없는 붙이기: 줄에 0.3초 안으로 이어진 낱말 전부가 한도 이하일 때 통째로. 한도 = 그 쪽(머리·꼬리)에 아직 안 맞은 대본 글자 수 + 2,
  줄 글자 × 0.75, 8글자 가운데 작은 값. 붙여서 글자 닮음이 0.25 넘게 떨어지면 붙이지 않는다. 그 쪽 대본 글자가 다 맞은 줄에는 짧은 대답 한
  낱말(`REPLY_WORDS`)을 근거 없이 붙이지 않는다.
- 나머지는 0.8초 쉼에서 잘라 added 줄로 만든다(한 줄 12낱말·6초 상한). added 는 기준 pass 와 rescue pass 의 낱말로만 만든다(대안 pass 만
  들은 대본 밖 말은 어디에도 나오지 않는다).
- align 0.4.0(`drop_noise`): added 로 남을 묶음을 말 덩이로 나눈다(앞 낱말이 문장 끝 부호로 끝나거나 낱말이 말줄임표·줄표로 시작하는 곳에서
  끊음). 말 덩이 안에서 `textnorm.NOISE_CREDIT` 이 걸린 낱말과 바로 뒤의 짝말(`CREDIT_TAIL`, 'by' 와 그 뒤 한 낱말)만 credit 으로 빼고
  (`_credit_runs`), 남은 이어진 범위 가운데 그 영화의 버릇 글귀(`asr.film_habits`)와 글자 닮음 `HABIT_SIM`(0.8) 이상인 범위를 habit 으로 뺀다.
  뺀 범위는 `method.noise_dropped`(`{"asr": [처음, 끝], "why": credit|habit}`)에 남기고 added 로 만들지 않는다. 나머지 낱말(문구 뒤에 이어
  받아써진 말 포함)은 이어진 데까지 다시 묶어 added 로 둔다. 버릇 글귀가 없으면 credit 만 본다. 묶음을 통째로 빼지 않는다.

**LLM 짝 판정(`align_llm.py`, decided_by llm).** 위 규칙이 끝난 뒤의 덧단계다.
- 창: 대본 차례로 이웃하고 시각으로도 이웃한 두 맞춘 줄 사이(틈). 틈에 못 맞춘 말 있는 줄과 added 로 남은 기준·rescue 낱말 조각이 함께
  있거나, 맞춘 줄과 같은 pass 에서 3초 안으로 바로 이어진 조각(떨어진 머리·꼬리 후보)이 있는 틈만 묻는다. 창 여럿을 요청 하나에 담는다.
- 묻는 것: 못 맞춘 줄마다 그 줄을 말한 이어진 낱말 범위와 종류(same, garbled, paraphrase, partial, short_answer, none), 머리·꼬리 후보가
  그 줄의 이어진 말인지, 헛소리인 낱말 범위. 확신 sure/unsure. 답은 번호와 열거값만.
- 받아들이기(`accept`, 결정적): sure 인 판정 가운데 한 줄의 범위가 한 조각(한 pass 안 이어진 낱말)이고 그 틈의 조각이며, 머리·꼬리는 줄 범위에
  바로 이어지고, 판정끼리 낱말이 겹치지 않고, 헛소리로 본 낱말을 쓰지 않고, 틈 안에서 대본 차례와 시각 차례가 맞고, 다른 pass 에서 고른 줄과
  같은 소리가 아닐 때만 적용한다. 아니면 버리고 까닭별로 센다(`method.llm_pair`).
- 헛소리 판정은 그 added 줄의 conf 만 낮춘다. LLM 이 정한 줄의 conf 는 0.8배다(`CONF_LLM`, 소리를 듣지 않은 판정).
- 적용한 판정들의 해시(`decisions_sha1`)가 단계 지문에 들어간다. 그래서 align 은 건너뛰기 전에 맞추기를 해 보고(캐시가 생기거나 바뀌면 다시 씀),
  `align --llm` 일 때만 캐시에 없는 요청을 부른다. 새로 부른 뒤에는 `data/llm_cache.zip` 을 pack 해 함께 커밋한다(llm 단계 약속 절).

**op 와 method.**
- op: `item`, `op`(same, changed, absent, added), `script_el`(added 는 null), `asr`(absent 는 null), `match`("text", added 는 null),
  `text_score`, `candidates`(줄을 맞춘 pass 마다 글과 평균 로그 확률 `score`, 끝에 대본 글 `src:"script"`·score null), `chosen`(고른 후보 번호),
  `decided_by`(rule, acoustic, llm). 대본 글은 후보로 기록만 하고 고르지 않는다. 들린 글자는 언제나 고른 pass 범위의 낱말 그대로다.
- method: 명세의 anchor, acoustic, forced, translate(null) 에 더해 aligner, base_pass, roles, choose, constants(상수 전부), decide(고르기 통계),
  conf(계산식 글), passes(pass 마다 role, words, matched_lines, chosen_lines, added_lines, backsteps), time_fix, noise_dropped, llm_pair, stats.
  `forced` 에는 강제 정렬을 따로 돌리지 않고 낱말 시각이 Whisper cross-attention DTW 시각이라고 적는다.
- conf 는 align.json 에 칸이 없다. `align.op_conf(script, asr, ops, method)` 로 다시 계산하고(final 이 부른다) 식은 `align.CONF_TEXT` →
  `method.conf`. same·changed 는 0.5 × 글자 닮음 + 0.5 × 낱말 확률 평균(대본 없이 받아쓴 다른 pass 가 같은 글이면 올리고 다른 글이면 × 0.85),
  LLM 이 정한 줄은 × 0.8(`CONF_LLM`). added 는 낱말 확률 평균(rescue × 0.8)에 헛소리 꼴 값을 곱한다: 자막 제작 문구 꼴(`textnorm.noise_shape`
  의 credit)은 늘 × 0.3, 인사말·되풀이 꼴은 낱말 확률 평균 0.5 이하 × 0.3 에서 0.9 이상 × 1 까지 곧은 줄(`_noise_factor`), LLM 짝 판정이
  헛소리로 본 낱말이 글자의 절반 이상이면 × 0.3(헛소리 꼴과 겹치면 둘 가운데 작은 값). absent 는 그 줄이 있을 만한 시간 창(앞뒤 맞춘 줄
  사이, 양쪽 5초 넓힘)에 설명 안 된 말이 얼마나 있는지와 그 줄의 글이 거기 들어 있는지로 0.95 에서 0.35 까지.
- 일치율(`method.stats.match`) = same·changed 줄에서 대본과 정확히 맞은 글자 수(`key` 글자열의 LCS) 합 ÷ 말 있는 대본 대사 글자 수, 소수 셋째
  자리. catalog `match`, final `film.script.match` 와 같은 값이다.
- 기준값은 가짜 영화 채점(`tools/eval_align_synth.py`)과 추격자 한 편의 줄 목록을 읽어 골랐다. 실제 정답 표본이 생기면 다시 정한다.

**쓰기와 실패.** `align.json.check` 에 쓰고 `validate_align`(asr·script 짝 포함)을 통과해야 `align.json` 으로 바꿔 놓는다. 실패하면 앞의
align.json 을 그대로 두고 지문도 남기지 않으며(다음 실행이 다시 맞춘다), 문제 기록에는 오류 코드별 수만 적는다. 예외로 끝나면 예외 이름만 남긴다.

## 최종본과 차례 약속 (timeline.py, final.py, tools/final_to_srt.py)

**차례와 id 는 `timeline.build_timeline` 만 정한다.** align 이 ops 의 `item` 을 붙이고, final 은 같은 입력(script.json, align.json ops,
asr.json, 영상 길이)으로 다시 불러 같은 항목 뼈대를 얻는다. 순수 함수이고 파일을 읽거나 쓰지 않는다.
- 항목: 대본의 scene, action, dialogue, onscreen, transition 요소가 정확히 한 번씩 나오고(front, unknown 은 넣지 않음) added 가 더해진다.
  id 는 차례대로 I0001 부터.
- 대사 시각: same·changed·added 는 낱말 범위의 가장 이른 시작과 가장 늦은 끝(`timing=aligned`), absent 는 시각 없음.
- 대사 아닌 요소: 대본 순서로 앞뒤의 시각 있는 대사 사이를 글 길이에 비례해 나눈 추정 시각(`estimated`, end null). 한 씬의 요소는 그 씬의
  대사 덩어리 곁에 둔다: 새 씬의 머리는 앞 씬에 붙은 added 뒤, 뒤 씬에 붙은 added 앞이다. 앞뒤 대사가 흐름에서 벗어난(옮겨진) 대사면 나누지
  않고 같은 씬 쪽에 붙인다.
- 차례: 시각 있는 항목을 start 순으로. 같은 시각이면 앞 대사에 딸린 요소, 딸린 데 없는 요소, 뒤 대사에 딸린 요소와 그 대사, added 순이다.
  absent 는 E 번호 오름차순으로 하나씩, 이미 놓인 항목 가운데 E 번호가 자기보다 작은 것 중 가장 큰 항목 바로 뒤에 넣는다(없으면 맨 앞, D7).
- added 의 씬: 시각으로 앞뒤에 있는 대본 대사의 씬 가운데 하나. 같은 씬이면 그 씬, 다르면 말이 없던 시간(쉼)의 합으로 잰 거리가 가까운 쪽
  (같으면 앞), 한쪽만 있으면 그쪽, 둘 다 없으면 null. 씬 머리·지문의 추정 시각은 보지 않는다.
- 씬 구간(`scene_spans`): 그 씬의 시각 있는 대본 대사와 그 씬으로 분류한 added 의 범위. 옮겨진 대사는 흐름 안 대사가 있으면 뺀다. 그런 대사가
  없는 씬은 씬 머리·지문의 추정 시각 범위다.
- 옮겨진 대사(`moved`): 대본 순서와 시각 순서가 함께 늘어나는 가장 긴 사슬 밖의 대본 대사.
- 낱말 시각 바로잡기(`op_words`, `fix_words`): 글자와 차례는 그대로 두고 시각만 고친다. 한 줄 안에서 시작이 앞 낱말의 시작보다 이르면 앞 낱말의
  시작으로 올리고, 끝이 시작보다 이르면 끝을 시작에 맞추고, 영상 길이를 넘은 시각이 허용 오차(`DURATION_TOL` 0.5초)에 닿으면 영상 끝으로
  당긴다. 줄 전체가 한 순간이면 마지막 낱말 끝을 0.02초 늘린다. 영상 길이 밖 낱말이 있었거나 시작이 1초 넘게 뒤집혀 있던 줄은 믿기 어려운
  줄이다(`time_suspect`).
- `shared_words`: 같은 pass 의 같은 낱말을 두 op 이 가진 곳(오류, validate 가 막음). `cross_pass_overlaps`: 서로 다른 pass 에서 온 두 줄의 시간
  구간이 0.6초 넘게 겹친 곳(의심, 막지 않고 기록). pass 사이 같은 낱말의 경계가 실제로 0.5초까지 어긋나기 때문이다.
- 차례·시각·씬 규칙이 바뀌면 `timeline.VERSION` 을 올린다. align 과 final 의 단계 지문에 들어간다.

**final.**
- 대사 항목: words 는 op.asr 범위의 낱말(w, s, e, p)을 바로잡은 시각으로 복사한 것이고, heard 는 w 를 공백으로 이은 글이다. 화자, how,
  paren, sung, overlap, scene, script_ref.text 는 대본 요소에서 복사하고, lang 은 대본 요소의 lang 에 아래 줄 언어 규칙을 적용한 값이다. status·diffs 는 `textnorm.verdict`, conf 는 `align.op_conf`.
  한국어 줄은 `ko = heard`, `ko_src = same`.
- 화자: 대본에 적힌 화자만 `speaker_from=script`. `speaker_inferred` 요소는 화자를 복사하되 `speaker_from=unknown` 으로 두고, added 는 화자
  없이 unknown 이다. `quality.speaker_unknown` 은 unknown 인 대사 수, `quality.speaker_inferred` 는 그 가운데 추정 화자가 붙은 수.
- 대사 아닌 항목은 대본 글 그대로에 추정 시각.
- 줄 언어(lang, final 0.3.0): 글자의 기준은 영상이다. 시각 있는 대본 대사는 `textnorm.spoken_lang(대본 lang, heard, 영상 언어)` 로 정한다.
  대본 lang 과 영상 언어의 기본 언어(지역 꼬리를 뗀 값)가 다르고 들린 글이 영상 언어 글자로 적혔으면 영상 언어, 아니면 대본 lang 그대로다.
  `textnorm.in_lang_script(글, 언어)`: NFC 뒤 문자(isalpha)만 세어 그 언어 글자가 `LANG_SCRIPT_SHARE`(0.5) 이상이면 참. ko 는 한글 음절
  (AC00–D7A3)·자모(1100–11FF)·호환 자모(3130–318F), ja 는 가나와 한자, zh 는 한자, ru 는 키릴, en·fr·es·de·it·pt·nl·la·vi 는 라틴 글자. 그 밖의
  언어이거나 글자가 없으면(숫자·문장부호뿐) 거짓. 바꾼 줄은 ko = heard, ko_src = same 이고, 수를 `quality.lang_from_heard` 에, 줄마다 항목 수준
  other(detail `대본 lang {x}, 들린 글은 영상 언어 {y} 글자: lang 을 {y} 로 둠`, 영화당 `align.ITEM_PROBLEMS_MAX` 50개까지)를 남긴다.
  absent 는 대본 lang 그대로, added 는 영상 언어다. 한계: 영어로 말했는데 받아쓰기(영상 언어로 고정)가 한글로 적은 줄도 ko 가 되어 번역 대상에서
  빠지고, 들린 글이 숫자뿐인 줄은 판정하지 못해 대본 lang 이 남는다.
- 번역은 범위 밖이다. 그래도 시각 있는 대사 가운데 lang 이 ko 가 아닌 줄이 하나라도 있으면 final.json 을 만들지 않고 영화 수준
  `translation_unmatched`(score = 그런 줄 수, detail 에 시각 있는 대사 수와 언어별 줄 수)를 남긴다. 이 영화는 영상이 등록돼 있으므로
  validate [D1] 과 `sinkprep all` 도 실패한다(흐름과 파일 절). 시각 없는 줄(absent)의 lang 은 세지 않는다.
- 낡은 align.json 거부(`Stale`): item id, script_doc, op 판정과 지금 대본 글의 판정, `method.stats.match` 가운데 하나라도 어긋나면 멈추고
  `align --force` 로 다시 맞추라고 알린다. catalog 의 video_sha1 과 asr.json 의 video_sha1 이 다르거나 video_duration 이 비면 멈춘다.
- `made`: pipeline, asr(pass 마다 모델 문자열), aligner, final, timing, scene, forced, conf, match, speaker, quality 의 뜻을 글로 남긴다.
  timing 글에는 낱말 시각 바로잡기 규칙 넷(영상 끝으로 당김, 앞 낱말 시작으로 올림, 끝을 시작에 맞춤, 한 순간인 줄의 끝을 `timeline.MIN_DUR`
  0.02초 늘림)을 모두 적는다(final 0.3.0).
  `quality`: dialogue, same, changed, added, absent, spoken, silent_turns, absent_spoken, line_coverage(same + changed ÷ spoken), match, mean_conf,
  low_conf, moved, words, by_pass, diff_kinds, speaker_unknown, speaker_inferred, mean_word_p, time_fixed, time_fixed_words, time_suspect, pass_overlaps,
  lang_from_heard.
- 쓰기 전 검사: `final.json.check` 로 쓴 뒤 `validate_final`, `validate_align`(final 과 함께), `validate_asr` 가 모두 통과해야 `final.json` 으로
  바꾼다. 실패하면 쓰지 않고, 문제 기록에는 오류 코드별 수와 항목 id 다섯 개까지만 적는다(대사 글을 넣지 않는다).
- catalog: 만들었거나(skip 포함) 일치율이 있으면 `status=done`, `match` = final 의 일치율. 실패했는데 done 이던 행은 aligned 로 내린다.
- 같은 입력이면 final.json 바이트가 같다(시각·난수·걸린 시간을 넣지 않는다).

**미리보기 자막(`tools/final_to_srt.py <film_id|final.json> [--out] [--absent] [--width 42]`).** 기본 출력 `films/{film_id}/preview.srt`.
시각 있는 대사(same, changed, added)만, 글은 heard. 화자 이름이 있으면 `이름: `(추정 화자는 `이름?: `), added 는 `[+] `, `--absent` 면 absent 를
앞 대사가 끝난 자리에 `[없음] 이름: 대본 글` 로 끼운다(말 없는 대사는 뺌). 시각 있는 두 줄 사이의 absent 는 한 자막으로 묶어 둘까지는 한 줄씩,
셋 이상이면 첫 줄 아래에 `[없음] … 외 N줄` 로 쓰고, 다음 자막 시작까지 0.5~2초(`ABSENT_MIN`·`ABSENT_SHOW`) 보인다. 그래서 다음 자막과 겹칠 수
있다. 긴 줄은 낱말 시각으로 잘라 한 자막에 두 줄(한 줄 `--width` 자 이하)까지이고, 시간으로는 나누지 않는다.
final.json 의 words 만 읽으므로 어느 pass 에서 온 줄이든 같은 꼴이다. 확인용이고 CBoE 의 자막 칸 나누기를 대신하지 않는다.

## 문제 기록
단계마다 `util.set_problems(stage, key, [...])`. 파일은 `cache/problems/{stage}/{key}.json` 이고 통째로 바꾸며 빈 목록이면 지운다.
problem dict 칸: `level, doc_id, film_id, item_id, problem, score, detail, path`(필요하면 `kind` 같은 칸을 더함).
problem 값: extract_broken, ocr_low, parse_low_dialogue, speaker_missing, version_mismatch, translation_unmatched, asr_low_conf, absent_many, other (명세 4.8).
- key: 문서 수준 문제는 doc_id, 영화 수준(영상 쪽 단계 asr·align·final, catalog 의 영화 문제)은 film_id. 영화 수준 행에도 doc_id 를 함께 넣는다
  (그 문서가 제외되면 같이 빠진다). catalog 에서 사라진 film_id 의 낡은 기록은 review 가 거르지 않으므로 `set_problems(stage, film_id, [])` 로 지운다.
- stage 폴더: catalog, convert, ocr, lines, parse, llm, asr, align, final. review 는 `cache/problems/*/*.json` 을 모두 읽는다.
- 영상 쪽 문제:

| stage | level | problem | 조건 |
|---|---|---|---|
| asr | film | asr_low_conf | rescue 를 뺀 pass 의 낱말 평균 확률 < 0.5 |
| asr | film | other(kind asr_hole) | 낱말 0개 묶음이 있거나, 말 구간 안 1초 넘는 빈 자리의 합이 말 구간의 10% 초과 |
| align, final | film | version_mismatch | 일치율 < 0.3 |
| align, final | film | absent_many | 말 있는 대사 중 absent 비율 > 0.5 |
| align, final | item | asr_low_conf | same·changed·added 줄의 conf < 0.2(영화당 50개까지) |
| align, final | item | other | 낱말 시각을 바로잡았지만 믿기 어려운 줄 |
| final | item | other | 다른 pass 에서 온 줄과 시간 구간이 0.6초 넘게 겹침 |
| final | item | other | 대본 lang 대신 들린 글자로 lang 을 영상 언어로 정한 줄(`quality.lang_from_heard`, 영화당 50개까지) |
| final | film | translation_unmatched | 들린 글자로 봐도 시각 있는 대사에 한국어가 아닌 줄이 남음(score = 줄 수) |
| align | film | other | 검사 실패(오류 코드별 수), 예외(이름만, `_fail`) |
| final | film | other | 검사 실패(오류 코드별 수), FinalError(낡은 align.json `Stale`, video 칸 문제 등) |

- final 의 예상 밖 예외(FinalError 가 아닌 것, 예: build 안의 KeyError, 검사 함수의 예외)는 `final.run` 이 잡아 상태를 error 로 두고 화면에만
  보인다. 문제 기록을 쓰지 않으므로 앞선 실행의 기록이 그대로 남고 needs_review 에 실패가 오르지 않는다(align 과 다른 점, 코드 쪽 할 일).
  catalog 의 done 은 aligned 로 내린다.
- align 과 final 은 같은 함수(`align.film_problems`)와 같은 글을 써서 review 가 한 줄로 합친다. detail 에는 수치와 id 만 적고 대사 글을 넣지 않는다
  (`qa/needs_review.csv` 는 git 에 들어간다).
- 새 단계가 문제를 쓴 뒤 `sinkprep review` 를 돌려야 `qa/needs_review.csv` 에 반영된다.

## 다시 돌리기
단계마다 `util.stage_fp(stage, key)`(`cache/state/{stage}/{key}.json`)에 입력 지문(원본 sha1, 앞 단계 출력 sha1, 단계 버전)을 남기고, 같으면
건너뛴다(`--force`면 다시). key 는 문서 단계가 doc_id, 영상 쪽 단계(asr, align, final)가 film_id 다. 실패하거나 일부만 된 결과에는 지문을 남기지
않아 다음 실행이 다시 한다. 결과 파일 sha1(`out_sha1`)이 지문과 다르면 다시 만든다.
- asr: VERSION, DECODE_VERSION, video_sha1, lang, passes, device(풀린 장치가 아니라 인자 문자열: GPU 상태 때문에 다시 만들지 않게. 다만 인자를
  바꾸면 다시 조립하고 그때 풀린 장치의 캐시를 쓰므로 asr.json 의 판이 바뀔 수 있다: 받아쓰기 약속의 '장치와 asr.json'. 지문이 cuda 인 영화에
  `--device auto` 를 줘도 글자가 달라 다시 조립하고, cuda 로 풀리면 묶음이 모두 캐시에 있어도 열어 보느라 모델을 GPU 에 올린다), rescue,
  models(저장소@스냅숏, 라이브러리 판), 대응표·묶음·창의 sha1. 묶음 캐시 폴더 꼬리표(`tags`)는 견주지 않는 칸.
- align: VERSION, timeline.VERSION, script·asr sha1, 영상 길이, 버릇 글귀 sha1, LLM 판정 해시(`llm_pair`).
- final: VERSION, align.VERSION, timeline.VERSION, 패키지 판, script·asr·align sha1, final 에 들어가는 catalog 칸(`final.ROW_FIELDS`).
- 영상 sha1 캐시는 `cache/state/filesha1/`(경로 열쇠). video 단계는 지문 없이 catalog 의 값과 견줘 건너뛴다.

| 상수 | 지금 | 올리는 때 |
|---|---|---|
| `lines.VERSION` | 0.4.1 | 줄 만들기가 바뀔 때(ocr 의 글자층 검사 캐시 열쇠에도 들어감) |
| `parse.VERSION` | 0.3.2 | 파서 규칙이 바뀔 때 |
| `media.VERSION` | 0.2.0 | 대응표 규칙(대응표의 made 가 달라져 다시 만듦) |
| `asr.VERSION` / `DECODE_VERSION` / `VAD_VERSION` / `WINDOW_VERSION` | 0.3.0 / 1 / 1 / 1 | 조립·rescue 규칙 / 받아쓰기 호출(묶음 캐시 열쇠) / 말 구간 찾기 / 창 나누기 |
| `timeline.VERSION` | 0.2.0 | 차례·시각·씬 규칙 |
| `align.VERSION` | 0.4.0 | 맞추기·고르기·conf 규칙 |
| `align_llm.PROMPT_VERSION` | ap2 | 프롬프트·스키마·창 표시(LLM 캐시 열쇠) |
| `final.VERSION` | 0.3.0 | 조립·quality·문제 기록·made 글 |

video 단계에는 VERSION 이 없다(지문 없이 catalog 값과 견줘 건너뛴다. 전에 있던 `video.VERSION` 은 아무 곳에서도 읽지 않아 지웠다).

`asr.DECODE` 표는 묶음 캐시 열쇠에 그대로 들어가므로 칸을 바꾸면 VERSION 을 올리지 않아도 캐시가 전부 빗나간다.

## 구현 중 정한 것 (2026-11-13, 모듈 작성 보고 반영)
- catalog: `file_formats`는 `real_format` 바로 뒤 칸. 제외 행 ID는 문서 다음 번호(`fm{글번호}_2`…). 원본 디스크 파일 701개(manifest 702 중 exe 1개는 내려받지 않음 → fm56878 note `missing_attachment`).
  note 값: `content:subtitle_transcript`, `content:synopsis`, `content:transcript`, `content:recording_script_xls`, `tv_episode:…`, `tv_series`, `missing_attachment:…`, `archive_container:…`, `duplicate_file_of:…`, `not_script:…`.
  `catalog.finalize(updates={doc_id:{칸:값}})` 로만 갱신(A2 어기면 ValueError).
  catalog 를 다시 만들 때 doc_id·files·sha1·film_id 가 그대로인 문서는 뒤 단계가 채운 칸(`catalog.LATER_COLUMNS`: review 칸과 video_*·match)을 이어받는다(carry_over). 이어받은 뒤 A2 를 어기는 영화는 새 임시값으로 되돌린다. 그래서 `sinkprep all` 을 다시 돌려도 catalog.csv 가 그대로다.
- review: script_lang(글자 비율·불용어로 ko/en/fr/es/mixed), is_translation(외국영화인데 ko 대본이면 true), extract_quality(script quality.score, OCR 이면 평균 conf 와 작은 값), status(script.json 있으면 parsed).
  is_primary 는 영화마다 (script 있음, 번역 아님, 품질, 판본 shooting>final>bound>revised>unknown>draft) 순으로 하나(2026-11-15 에 match 를 둘째 기준으로 더함, catalog.csv 절).
  영화 ID 가 local_ 인 것(KMDb/TMDB 키 없음)은 needs_review 에 넣지 않는다(못 읽은 대본 목록이 아니므로; catalog film_id 로 보임).
- convert: `cache/convert_log.jsonl` 칸에 `note`(rtf_sectd_removed(no_paperw), hancom_open_yes) 추가. method: hancom | fallback_hwp3_text | fallback_hwp5_bodytext | fallback_hwp5_prvtext | fallback_hwpx | fallback_antiword | fallback_rtf | fallback_docx.
  한컴 실패 뒤 대체 글만 얻은 문서는 problem `other`(score 0.5, 레이아웃 없음). HWP3 대체(`sinkprep.hwp3text`)는 문단 리스트를 직접 읽고(표·글상자 칸 포함, 머리말·꼬리말·각주·숨은 설명 제외), 구조가 망가진 곳에서 멈추면 그때까지 글을 쓰고 `error`에 `hwp3_partial:<까닭>@<위치>`를 남긴다. doc/rtf 는 한워드로 넘어가지 않게 빈 문서에 Insert. 매크로가 기본 틀이 아닌 HWP5 는 한컴으로 열지 않음.
  일부만 읽은 HWP3 은 problem `extract_broken`(score = 본문 중 읽은 비율, `error` 에 `;read=0.40`).
  교차 검사(convert.crosscheck): 한컴 변환이 성공해도 대체 경로로 읽은 원본 글과 견준다. 원본 글자(문자·숫자) 중 PDF 에 없는 비율 `loss` 를 로그 칸에 남기고,
  0.01 초과면 problem `other`(PDF 는 그대로, 레이아웃이 있어야 파싱됨), 0.03 초과면 PDF 를 `{k}.incomplete.pdf` 로 옮기고 대체 글 `{k}.txt` 를 쓴다(note `hancom_pdf_incomplete`).
  전체 481문서에서 정상 변환은 loss 0.002 미만, 한컴 2024 가 줄을 빠뜨린 문서는 fm54163 0.015, fm56063 0.045, fm55169 0.158(쪽 밖으로 밀리거나 사라진 대사). 대조 도구 `tools/check_convert.py`.
- ocr: RapidOCR(PP-OCRv5 검출 + korean 인식 / 영어는 PP-OCRv6 small), 300dpi, CPU. 결과 문서 칸 `schema:"sink.ocr/1"`, source_sha1, lang, lang_how, made, seconds, failed_pages, conf_mean, why, textlayer; 쪽 칸 render, warn, rotate, part/parts/crop(두 쪽 펼침); 줄 칸 segs(떨어진 덩어리).
  쪽 그림 실패도 extract_broken. 글자층 검사(textlayer)가 깨졌다고 해도 사유가 띄어쓰기 빠짐·이상한 문자·한 글자 줄·같은 줄·겹친 글자뿐이고,
  lines 가 글자 좌표로 다시 만든 줄이 같은 기준을 통과하면 OCR 하지 않는다(ocr._restorable_by_lines, 원본 글자층 PDF·한컴 변환 PDF 모두). 글자층 글자가 OCR 보다 정확하기 때문.
  이 기준을 넣기 전에 OCR 한 결과 중 이제 복원으로 충분한 것은 cache/ocr_superseded/ 로 옮겨 lines 가 글자층을 쓰게 한다.
- lines: OCR 줄의 segs 를 덩어리마다 넣어 칸 사이를 탭으로. PDF 는 rawdict 글자 좌표(앞 공백 빼고 첫 글자 x), 공백 4칸 이상이면 조각 분리, 글자 크기 0.25배 넘게 벌어진 곳은 공백 복원, U+F000 은 공백. 칸 사이 탭 기준은 글자 폭 1.5배.
  같은 글자가 글자 크기 0.15배 안쪽에 또 찍혀 있으면(겹쳐 찍은 굵은 글씨) 하나만 남긴다(lines._drop_overprint). 모양 같은 글자 U+2028·U+2029→공백, U+A789→:, U+2011→-, U+00AD 삭제(lines.GLYPH_MAP).
  두 쪽 펼침(lines._gutter): 가로 쪽(가로>1.15×세로)의 0.3~0.7 너비에 글자가 없는 세로 띠(쪽 너비 1.5% 이상)가 있고 양쪽에 글이 15% 이상씩이면 그 띠에서 나눈다. 가로 쪽 절반 넘게 띠가 있으면 문서 전체를 펼침으로 보고 띠 없는 쪽은 띠 위치 중앙값에서 나눈다. 이때 page 는 논리 쪽 번호(왼쪽 먼저), x 는 그 쪽 안 위치. 말뭉치 31문서(밀정·날 보러와요 등)가 이 모양이었다(전에는 왼쪽·오른쪽 같은 높이 글이 탭으로 이어졌음).
- script.json dropped.why: page_header | page_footer | page_number | cont_marker | continued | revision_mark | scene_number_margin | blank | timecode | subtitle_cue 등 snake_case. quality.notes 는 문자열 목록.
- script.json 대사 요소의 선택 칸 `speaker_inferred`(파서 0.2.0, 명세 4.4.2 보완): 화자를 앞 대사에서 이어받은 요소에만 `true` 로 쓴다(false 는 쓰지 않음). schema/script.schema.json 의 el_dialogue 에 선택 속성으로, validate 빠른 검사(_EL_DIALOGUE_KEYS_INF)에도 넣었다. 칸이 없는 옛 파일도 유효.
- E1 은 다섯 형식(lines/script/asr/align/final)에 적용. cache/ 중간 JSON 은 대상 아님(ocr 결과는 schema 칸을 갖지만 검사 대상 아님).
- 검토(architect) 반영 2026-11-13:
  - review: 번역 판정은 양방향(외국영화의 ko 대본, 한국영화의 외국어 대본·제목 '영문'). 대표 대본 고르기에서 두 언어가 섞인 판(mixed)은 한 언어 판 뒤로.
    catalog 에 없는 doc_id 의 낡은 문제 기록은 목록에서 뺌. pdf_broken 이라 catalog 가 '글자를 꺼낼 수 없음'으로 적었어도 OCR 결과가 있으면 그 판정은 뺌(OCR 단계 문제가 실제 상태를 알림).
  - parse quality: `empty_dialogue`(말 없이 이름 줄만 잡힌 대사), `inferred_speaker` 를 센다. 대사 20개 이상인 문서에서 말 없는 대사 10% 이상이면 parse_low_dialogue, 추정 화자 10% 초과면 speaker_missing. score 에도 반영.
  - ocr/lines: OCR 결과는 source 경로·source_sha1 이 지금 입력과 같을 때만 건너뛰거나 쓴다(ocr.result_is_current). 원본·변환 PDF 가 바뀌면 다시 OCR.
  - convert: antiword 는 작업 폴더(cache/convert_work)를 현재 폴더로 실행. 한컴 시간 초과·일꾼 오류는 일시적일 수 있어 예전 PDF 를 지우지 않고(note kept_previous_pdf) 지문도 남기지 않는다.
  - catalog: 압축은 풀기 전에 반디집 목록(bz l)으로 항목 이름을 보고 '..'·절대 경로·드라이브 경로가 있으면 풀지 않는다(중첩 압축도).
  - validate: D7 뒤 절반(시각 없는 항목은 대본 순서로 앞뒤 항목 사이) 검사 추가.

## 구현 중 정한 것 (2026-11-15, 영상 쪽 단계와 대본 수리)

**lines 0.4.1 (띄어쓰기 맞추기).**
- 한컴 변환 PDF 에서 줄을 만드는 파일은 원본 문서 글(`convert.fallback_text`, 교차 검사와 같은 읽기)과 견줘 공백 손실을 잰다. 공백 손실 =
  원본의 낱말 경계 가운데 PDF 줄에 공백도 탭도 없는 비율. 견줄 경계가 200개 이상이고 손실이 `RESPACE_LOSS`(0.15)를 넘는 파일만 줄 글의
  공백·탭을 원본에 맞춘다(`lines.restore_spacing`): 빠진 공백을 넣고, 이름 뒤 탭을 탭으로 올리고(이름 꼴 덩어리이고 그 이름이 두 번 이상
  나오거나 이름 칸에 놓였을 때만), 원본에 없는 PDF 공백을 지우고, 원본이 공백 하나인 자리의 탭을 공백으로 내린다. 두 글은 양쪽에 한 번씩만
  나오는 8글자 조각을 닻으로 맞추고, 닻 사이 틈(2,000자 이하)은 두 쪽이 거의 같은 글이 아니면 6글자 이상 맞은 덩어리만 받는다.
- 글자와 줄 수·n·page·x·y·src 는 바뀌지 않는다. PDF 에만 있는 글자(자동 번호, 쪽 번호, 머리말)도 그대로다.
- 머리 행 `made.spacing` 에 파일별 기록: 맞춘 파일은 `{file, from:"source", loss, matched, lines, spaces, tabs, removed, lowered}`, 견주지 못한
  파일은 `{file, from:null, skipped[, error]}`. skipped 는 source_missing(원본 파일 없음), source_unreadable(원본 글을 못 읽음), source_empty(빈 글),
  `few_bounds:N`(견줄 낱말 경계가 200개 미만), error(되살리기 예외) 가운데 하나이고, error 칸(예외 이름과 첫 토막)은 source_unreadable·error 에만
  있다. 기록이 없는 파일은 견줬고 문턱 아래다. 원본 글 읽기가 한때의 실패(시간 초과, 파일 잠김, 메모리)로
  끝난 문서는 지문을 맞추지 않아 다음 실행이 다시 만든다. lines 지문에 원본 파일 sha1(`sources`)이 들어간다.
- 명세 4.4.1 의 'text 는 추출한 그대로에 NFC 만' 은, 맞춘 파일에 한해 글자는 PDF 글자층에서, 공백·탭은 원본 문서에서 추출한 것으로 읽는다.
- 2026-11-15 기준 대상 10파일(fm54367, fm54441, fm54519, fm55036, fm55099, fm55455, fm55655, fm55664, fm55844, fm416766), 견주지 못한 파일 2개
  (fm55630, fm56026, 원본 글을 읽지 못함). 문턱 아래 파일의 이름 뒤 탭 손실(20줄 이상 48문서)과 띄어쓰기가 통째로 사라진 줄은 손대지 않는다.

**parse 0.3.1.**
- 쌍점이 든 번호 씬 머리(`망원동 골목길 : 오…` 꼴)는 받아들인 번호 씬 머리(5개 이상)와 같은 x 에 놓이고 빈 번호를 차례대로 메울 때 받는다.
  쌍점 앞이 그 문서의 화자이거나 쌍점 뒤가 말처럼 생겼으면 뺀다. 쌍점이 없으면 시간 꼬리가 있어야 한다.
- `(이름)⇥대사` 는 괄호 안이 이름표에 있는 인물일 때 그 인물의 대사이고 how=off, speaker_raw 는 괄호째다. 맨 앞 괄호가 전화·무전이면 그쪽 how 가
  먼저다. 괄호 안이 `이름 + 소리·목소리` 면 그 인물의 화면 밖 목소리다. 괄호 이름 줄의 오른쪽이 소리 묘사면 지문이다.
- 한 줄 표기가 이름 줄보다 4배 이상(20줄 이상)인 문서의 말 없는 차례(`영민⇥.......`, `미진⇥(한숨)`)는 speaker_raw 가 이름, 점은 text, 괄호는
  paren 인 한 줄 대사이고 앞 화자를 이어받지 않는다.
- 이름 칸이 뚜렷한 문서(이름표 이름의 강한 한 줄 표기 줄 50줄 이상, 95% 이상이 한 x 묶음)에서 이름 뒤 공백 하나 표기는 이름 칸에 놓인 줄이거나
  이름 바로 뒤가 들리는 방식 표시일 때만 대사 시작이다.
- 대사 이어짐(`tight_cont`): 지문 앞에 늘 빈 줄을 두는 문서에서는 대사 바로 아래 붙여 쓴 들여쓴 줄을 앞 대사의 이어짐으로 본다. 규칙이 켜진
  문서는 quality.notes 에 `대사 이어짐: …` 이 남고 46개다(지금 docs/ 의 규칙 판 44개, LLM 판으로 바뀐 2개는 cache/parse_rules 쪽). 그 가운데
  줄이 실제로 바뀐 39문서에서 2,857줄이 지문에서 대사로 간다. 정답 표본에 이 줄이 없어 결정 대기다(끄는 곳: `parse.py`
  `_cont` 의 `if tight and self.tight_cont` 한 줄).
- 씬 place·time(`parse_text.split_place_time`): time 은 씬 머리 끝의 시간 표현을 적힌 그대로 쓴다. 이어진 시간 낱말은 통째로 시간(`다음날 아침`,
  `아침-밤`), 시간 뒤에 빗금으로 덧붙은 짧은 조각(`/ 비`, `/ INT`, `/ 과거`)은 place·time 에 넣지 않고 heading 에만 남는다. 장소 낱말에 공백으로만
  이어진 내부·외부·내·외 는 place, 구분자로 떨어진 실내·실외·내부·외부는 time 쪽이다.
- 정답 표본 수치는 0.2.5 와 줄 단위까지 같다(표본이 이번 규칙이 닿는 줄을 거의 담지 않음). 추격자는 HWP 문단 구조와 견줘 말 차례 1,034개의
  화자와 씬 195개가 모두 맞지만, 그 문서를 보며 만든 규칙이라 일반화 근거는 아니다. `tools/make_types.py` 분기 표에는 0.3.x 꼬리표
  넷(scene_colon_heads, paren_names, wordless_turns, tight_cont)과 types.csv 의 spacing 칸을 더했다. `TYPES.md`·`types.csv`·`REPORT.md` 는
  2026-11-15 에 lines 0.4.1·parse 0.3.2 와 영상 쪽 결과(REPORT 6b·7절)를 반영해 다시 만들었다.

**parse 0.3.2 (언어 표시).**
- 괄호 지시·화자 꼬리표의 언어 표시(`parse_text.lang_of`, `LANG_CUES`)는 언어 이름이 낱말로 서 있을 때만 받는다(`_lang_rx`): 앞은 글자가
  아니고, 뒤에는 꼬리 하나(로·으로·로만·만·도·을·를·소리·말·말로·대사·자막)만 붙을 수 있다. 한글 이름은 글자 사이 공백을 받는다('러시아 말로',
  '한국 어'). 그래서 '(일어서며)'·'(일어나며)' 같은 동사, 'Russell'·'ENGLISHMAN' 같은 다른 낱말 속 글자는 언어 표시가 아니다.
- '일어' 뒤에 공백을 건너 서·선·나·난 등이 오면 줄바꿈으로 갈라진 동사로 본다. 이름 뒤에 accent 가 오면('(Russian accent)') 언어 표시가 아니다.
  latin·thai 는 꼬리표가 그 이름뿐이거나 'in 이름' 일 때만 받는다. '영문' 은 말뭉치에서 '(영문 모름)' 처럼 까닭이라는 뜻으로만 나와 이름에서 뺐다.
  이름이 꼬리표 열쇠의 절반 이상이어야 하는 예전 조건은 그대로다.
- 바뀐 것은 대사 요소의 lang 뿐이다(684문서 가운데 186문서 379요소: ja→ko 367 등). 요소 수·종류·id·글·화자·씬은 그대로이고, 정답 표본 채점도
  전·후가 같다. 한국어 대표 대본 368개에서 lang 이 ko 가 아닌 대사는 248문서 2,460줄에서 141문서 2,121줄이 됐다(OVERVIEW 2.6절).

**검사(validate)에 더한 규칙.**
- [D1] 인자 없이 돌릴 때: catalog 에서 video_path 가 있고 is_primary=true 인 영화마다 `films/{film_id}/final.json` 이 있고, 그 final 의
  film.script.doc_id·film.video.sha1·film.video.duration(0.0015초 안)·film.script.match 가 catalog 행과 같다. catalog.csv 는 git 에 들어가고
  films/·videos/ 는 들어가지 않으므로, 영상·전사 캐시가 없는 곳(새 클론)에서는 D1 이 실패하고 `sinkprep all` 도 1로 끝난다. 영상을 등록한
  영화는 final 까지 만들어야 통과한다(final 이 멈춰 옛 final.json 이 남아 있으면 catalog 의 새 match 와 어긋날 때 실패).
- [D2] 항목·낱말·씬의 시각이 film.video.duration + 0.5초(`DURATION_TOL`)를 넘지 않고, 한 줄 안에서 낱말 시작 시각이 줄지 않는다.
- [D4] match=text 인 줄은 same 이면 `textnorm.key(heard) == key(script_ref.text)`, changed 면 다르다.
- [D8] script.json 을 찾았을 때 그 scene·action·dialogue·onscreen·transition 요소가 src_el 로 빠짐없이 나온다. 같은 src_el 을 가리키는 항목이
  둘 이상이면 오류(script.json 없이도 본다).
- [D5] ko·ko_src·lang 이 맞물리고(lang=ko 면 ko_src=same·ko=heard), 시각 있는 대사의 들린 글이 영상 언어 글자면 lang 이 영상 언어다
  (`textnorm.spoken_lang`, final 과 같은 함수).
- [D9] script.json 을 찾았을 때 대본에서 복사한 칸이 대본과 같다: 대사의 speakers·how·lang(match=text 인 줄만. 시각 있는 줄은 대본 lang 을
  `textnorm.spoken_lang` 으로 정한 값과 견줌)·paren·sung·overlap·scene·script_ref.text, speaker_from=script 면 대본에 적힌 화자(추정 화자가 아님), 대사 아닌 항목의 text·scene·kind, 씬의 number·heading·place·time·flags,
  인물의 name·aliases. 씬 구간의 start 는 그 씬 항목들의 시각 범위 안이고, 시각 있는 항목이 있는 씬만 구간이 있다. 오류 글에는 칸 이름만 적는다.
- [Q] script.json 을 찾았을 때 `quality.lang_from_heard` 가 대본 lang 과 다른 시각 있는 대사 수와 같다.
- [C2] final 의 대사 항목 words(w, s, e, p)가 그 op 의 asr 범위 낱말과 같다(`timeline.op_words` 로 바로잡은 값). 같은 pass 의 같은 낱말을 두
  op 이 갖지 않는다. added op 에는 asr 범위가 있다. `films/{id}/final.json` 자리의 파일은 film.id 가 폴더 이름과 같다.
- 짝 파일은 자동으로 찾는다: final 은 film.script.doc_id 의 `docs/{doc}/script.json`, align 은 같은 work/ 의 asr.json 과 위 폴더의 final.json.
  영화 하나만 보려면 `sinkprep validate films/{film_id}`. `--docs` 를 주면 docs 만 본다.

**그 밖.**
- review 는 aligned·done 을 지키고 대표 대본을 match 먼저 고른다(catalog.csv 절). 영상 쪽 단계의 영화 수준 문제가 needs_review 에 오른다.
- `__main__`: 명령 video(`--lang`, `--unregister`), asr(`--passes`, `--threads`, `--device`, `--rescue`), align(`--llm`), final 을 더했다. `all --llm` 은 llm 단계에만
  쓰이고 align 의 LLM 짝 판정은 all 에서 늘 캐시만 쓴다.
- `paths`: `AUDIO`(cache/audio), `ASR_CACHE`(cache/asr), `VIDEOS`(videos). 경로는 호출 시점에 `paths.X` 로 읽는다(시험이 바꿔 끼운다).
- `.gitignore` 에 `films/`, `videos/` 를 더했다.
- 명세 해석: op 의 asr 범위는 한 pass 안이다. 그래서 한 대사가 두 pass 에 나뉘면(rescue 머리 + plain 줄) 한 줄로 묶지 못한다. 명세를 바꿀지는 결정 대기.
