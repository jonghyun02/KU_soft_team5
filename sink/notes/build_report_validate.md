## US-002 결과: 다섯 형식 스키마와 검사 도구

스키마 5개, 검사 도구, 예시 파일, 시험을 모두 만들었고 시험은 전부 통과합니다. 지금 기본 `validate`는 실패로 끝나는데, `docs/` 결과물이 아직 없어서 B1 검사가 1368건을 잡기 때문입니다. lines·parse 단계가 돌면 사라질 오류입니다.

### 만든 파일
- `C:\projects\sink\schema\lines.schema.json`, `script.schema.json`, `asr.schema.json`, `align.schema.json`, `final.schema.json`
  - draft 2020-12로 썼고 다섯 개 모두 `check_schema`를 통과합니다. 모든 형식에 `schema` 칸이 필수입니다(E1).
  - lines는 파일 전체를 행 배열로 보고, 0번 행은 머리 행, 나머지는 줄 행 규칙으로 봅니다.
  - D3·D4·D6, timing, `ko_src=same`이면 `lang=ko` 같은 조건 규칙은 스키마 안에도 넣었습니다.
- `C:\projects\sink\sinkprep\validate.py`
  - 함수: `validate_lines`, `validate_script`, `validate_asr`, `validate_final`, `validate_align`, `validate_catalog(check_files=True, path=None, check_sha1=False)`, `validate_needs_review`, `check_file`, `run(args)`.
  - 오류마다 앞에 규칙 코드가 붙습니다: [schema] [E1] [E2] [B1] [B2] [ref] [C1] [C2] [D2]~[D8] [T] [Q] [A1] [A2] [catalog] [review].
  - 줄 행, script 요소, final 항목, 낱말, align 연산처럼 개수가 많은 것은 빠른 검사를 먼저 합니다. 빠른 검사에서 걸린 것만 jsonschema로 다시 봅니다. 빠른 검사를 통과한 데이터는 스키마도 반드시 통과하는지 시험에서 무작위로 바꾼 사본으로 확인했습니다.
  - `run`은 실패한 파일마다 오류 수와 앞 20개를 찍고, 오류가 하나라도 있으면 1을 돌려줍니다. 파일이 16개 이상이면 프로세스를 여러 개 띄워 나눠 검사합니다.
- 예시 파일: `C:\projects\sink\examples\asr.json`, `align.json`, `final.json`, `examples\fm_example\lines.jsonl`, `script.json`
  - 범죄의 재구성(fm55574)을 바탕으로 줄 18개, 요소 14개, 항목 12개를 만들었습니다.
  - same·changed·added·absent, 외국어 줄(`ko_src=mt`), 무전(radio), 노래, INSERT, 쪽 넘김(cont_marker), `12A`·flashback 씬을 담았습니다.
  - 명세의 "…" 자리는 실제 값으로 채웠고, `asr_sha1`은 `asr.json` 파일의 실제 sha1입니다.
- `C:\projects\sink\tests\test_validate.py`

### 확인한 방법과 숫자
- `.venv/Scripts/python -I -m pytest -q tests/test_validate.py`: **122 passed**. test_util까지 같이 돌리면 127 passed입니다.
  - 예시 파일 5개는 오류 0개입니다.
  - D2~D8, C1, C2, B2, B3, E1, E2 규칙마다 일부러 어긴 사본을 만들어 거부되는지 확인했습니다.
  - catalog와 needs_review는 tmp 폴더에 만든 가짜 원본과 catalog로 시험했습니다(paths를 monkeypatch).
- `python -I -m sinkprep validate examples`: 검사 5개, 실패 0개.
- 다른 에이전트가 만든 실제 `catalog.csv`도 통과합니다.
  - 712행(pending 684, excluded 28)입니다.
  - 행에 적힌 파일은 714개이고, A1 대상 파일(원본에서 manifest를 뺀 것 + extracted/)도 714개로 맞습니다. 오류 0개, 0.1초.
  - sha1을 다시 계산해 비교해도 오류 0개, 0.59초.
- 속도는 가짜 문서로 쟀습니다. 문서마다 줄 3500개와 요소 약 1794개, 모두 700개(854MB)입니다.
  - 1400개 파일 검사에 8.1초, 다시 돌리니 5.3초였습니다(12개 동시).
  - 일부러 넣은 오류 3개(모르는 요소 종류, 덮이지 않은 줄, NFD 글자)를 모두 잡고 1을 돌려줬습니다.
  - 문서 하나를 혼자 돌리면 lines 0.04초, script 0.06초입니다. script를 jsonschema로만 보면 0.30초였습니다.
- 가짜 영화 하나(대사 1800줄, 낱말 14,400개 × 2회 받아쓰기): asr 0.29초, final 0.31초, align 0.15초.
- lines.py의 `lines_from_text`와 `merge_segments`(OCR 포함)로 만든 행이 줄 스키마를 통과하는 것도 확인했습니다.

### 남은 문제
- **기본 `validate`는 지금 실패로 끝납니다.** catalog의 문서 684개마다 `docs/{id}/lines.jsonl`과 `script.json`이 없어서 B1 오류가 1368개 나옵니다. lines와 parse가 아직 돌지 않았기 때문입니다.
- parse.py와 review.py가 아직 없어서, 실제 `script.json`과 `needs_review.csv`가 이 스키마와 맞는지는 아직 확인하지 못했습니다.
- `-I` 규칙을 한 번 어겼습니다. validate.py 일부를 고칠 때 stdin 파이썬을 `-I` 없이 실행했습니다(현재 폴더는 C:\projects\sink).

### DESIGN 변경 제안
1. 명세에 정해진 값이 없거나 뜻이 갈리는 칸을 이렇게 정했습니다. DESIGN에 적어 두는 게 좋겠습니다.
   - `dropped.why`: 정해진 목록이 없어 snake_case 모양만 검사합니다. 목록 예: page_header, page_footer, page_number, cont_marker, continued, revision_mark, blank.
   - `quality.notes`: 문자열 목록으로 정했습니다.
   - final의 onscreen 항목: 대본 요소의 `kind`를 선택 칸으로 복사할 수 있게 했습니다.
   - `align.asr_sha1`: `work/asr.json` 파일 바이트의 sha1로 해석했습니다.
2. D5와 명세 4.7("외국어가 하나라도 있으면 ko_src=mt")이 부딪힙니다. 낱말에 `lang`이 붙은 외국어가 섞인 줄은 `ko_src`가 same이 아니어야 하고, 섞이지 않은 `lang=ko` 줄은 same이면서 `ko==heard`여야 하도록 맞췄습니다.
3. OCR 줄의 `conf`는 숫자나 null을 허용했습니다. lines.py가 신뢰도를 못 얻으면 None을 쓰기 때문입니다.
4. 명세에 없던 규칙을 추가했습니다.
   - [T] 시각은 소수 셋째 자리까지.
   - [Q] final의 quality 개수가 items로 센 값과 같음.
   - D2 보강: `heard`가 낱말 `w`를 이은 글과 같음(공백 무시).
   - D8 보강: 항목 종류가 가리키는 요소 종류와 같음.
   - asr와 final의 영상 sha1이 같음.
5. catalog 칸 목록 `validate.CATALOG_COLUMNS`를 catalog 모듈과 같이 쓰는 상수로 두는 게 좋겠습니다. 지금 catalog.csv와 순서까지 같지만, 검사는 칸 이름이 다 있는지만 봅니다.
6. catalog에 정한 규칙들:
   - catalog는 BOM이 필수이고, needs_review는 BOM이 있어도 없어도 됩니다.
   - excluded 행은 film_id, title_raw, version_label 같은 칸을 비워도 되지만 `note`는 꼭 있어야 합니다.
   - doc_id 모양 `fm{post_id}(_k)`는 excluded가 아닌 행에만 요구합니다.
7. E1은 다섯 형식에만 적용했습니다. `cache/` 아래 중간 JSON(ocr, convert_log, problems, state)에는 schema 칸이 없어서 검사 대상에서 뺐습니다. E1을 "모든 JSON"으로 읽는다면 DESIGN에 이 파일들의 schema 값을 정해야 합니다.