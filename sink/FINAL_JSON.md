# final.json 에 들어가는 것

`films/{film_id}/final.json` 은 영화 한 편의 영상 기준 대본이다. 영상에서 실제로 들린 말을 기준으로 삼고, 대본에서 화자·씬·지문을 가져와 시간 순서로 늘어놓았다. 이 메모는 그 파일에 무엇이 들어 있는지를 한곳에 정리한 것이다.

- 스키마: `schema/final.schema.json` (`sink.final/1`)
- 만드는 코드: `sinkprep/final.py`, `sinkprep/timeline.py`
- 수치 예시는 추격자(`local_0270`)와 써니(`local_0556`)의 실제 결과다. 대사 글 예시는 지어낸 것이다.
- 사용법은 `README.md`, 만드는 방법의 자세한 내용은 `OVERVIEW.md` 3장과 `DESIGN.md` 에 있다.

## 1. 한눈에

```
final.json
├─ schema    "sink.final/1"
├─ film      어떤 영화를 어떤 영상과 어떤 대본으로 만들었나
├─ made      무엇으로 어떻게 만들었나 (모델, 규칙, 계산식을 적은 글)
├─ people    등장인물 목록
├─ scenes    씬 목록
├─ items     본문. 시간 순서로 늘어놓은 씬 머리, 지문, 대사, 화면 글자, 전환
└─ quality   영화 전체 집계
```

본문은 `items` 다. 위에서부터 읽으면 영상 순서의 대본이 된다.

| 영화 | 항목 | 씬 머리 | 지문 | 대사 | 화면 글자 | 전환 |
|---|---|---|---|---|---|---|
| 추격자 | 2,096 | 195 | 468 | 1,423 | 8 | 2 |
| 써니 | 1,935 | 94 | 380 | 1,461 | 0 | 0 |

## 2. 먼저 알아둘 규칙

- **시각은 초 단위 소수이고 영상 시작이 0이다.** 영상 플레이어에 보이는 시각과 같다.
- **글자의 기준은 영상이다.** 대사의 글(`heard`)은 받아쓰기가 적은 그대로다. 대본 글은 `script_ref.text` 에 따로 남는다.
- **대본의 요소는 정확히 한 번씩 나온다.** 씬 머리, 지문, 대사, 화면 글자, 전환이 대상이다. 영상에서 못 찾은 대본 대사도 지우지 않고 `absent` 로 남긴다. 대본에 없는데 들린 말은 `added` 로 더한다. 대본의 앞머리(표지, 인물표)와 종류를 못 정한 줄은 넣지 않아서 대본 요소 id 가 건너뛸 수 있다(써니의 첫 항목은 `E0004` 다).
- **id 꼴**: 항목 `I0001`, 씬 `SC001`, 인물 `P01`, 대본 요소 `E0001`. 대본 요소 id 는 `docs/{doc_id}/script.json` 의 id 다.
- **`timing` 은 그 시각을 어떻게 얻었는지다.**

| timing | 뜻 | 쓰이는 곳 |
|---|---|---|
| `aligned` | 받아쓰기 낱말의 시각 | 들린 대사 |
| `estimated` | 앞뒤 대사 시각으로 어림한 값 | 씬 머리, 지문, 화면 글자, 전환, 씬 구간 |
| `none` | 시각 없음(`start`, `end` 가 null) | 영상에서 못 찾은 대사, 시각을 하나도 못 얻은 씬 구간 |

- 같은 입력이면 파일이 바이트까지 같다. 만든 날짜나 걸린 시간은 넣지 않는다.

## 3. film

어떤 영화를 무엇으로 만들었는지다.

```json
{"id": "local_0270", "title": "추격자", "title_orig": null, "year": null,
 "video": {"sha1": "faa323e9…", "duration": 7548.093, "lang": "ko"},
 "script": {"doc_id": "fm55664", "version": "unknown", "lang": "ko",
            "translation": false, "match": 0.613}}
```

| 칸 | 뜻 |
|---|---|
| `id` | 영화 ID. 지금은 `local_번호` 뿐이다(KMDb, TMDB 연결은 하지 않았다) |
| `title` | 제목 |
| `title_orig`, `year` | 원제와 연도. 지금은 null |
| `video.sha1` | 영상 파일의 sha1. 어느 영상 파일로 만들었는지 가린다 |
| `video.duration` | 영상 길이(초) |
| `video.lang` | 영상 말소리 언어 |
| `script.doc_id` | 쓴 대본 문서 (`catalog.csv` 의 doc_id) |
| `script.version` | 대본 판(draft, revised, bound, shooting, final, translation, unknown) |
| `script.lang`, `script.translation` | 대본 언어, 번역 대본인지 |
| `script.match` | 일치율. 아래 8절 |

## 4. made

무엇으로 어떻게 만들었는지를 글로 적은 칸이다. 파일만 받은 사람도 값의 뜻을 알 수 있게 넣었다.

| 칸 | 내용 |
|---|---|
| `pipeline` | sinkprep 버전 |
| `asr` | 받아쓰기 pass 마다의 모델과 옵션. pass 는 같은 소리를 설정을 바꿔 받아쓴 한 벌이고, 대사 한 줄은 한 pass 의 낱말만 쓴다. 지금은 `plain`(large-v3-turbo), `lv3`(large-v3), `rescue`(놓친 대사 되찾기) 셋 |
| `aligner` | 맞추기 코드 버전과 방법 한 줄 |
| `translator` | 번역기. 번역을 하지 않아 null |
| `final` | 최종본 코드 버전 |
| `timing` | 세 가지 timing 의 뜻과 낱말 시각을 바로잡는 규칙 |
| `scene` | 대본에 없는 말의 씬을 정하는 규칙 |
| `forced` | 강제 정렬을 따로 하지 않았다는 기록 |
| `conf` | conf 계산식 전문 |
| `match` | 일치율 계산식 |
| `speaker` | `speaker_from` 값의 뜻 |
| `quality` | quality 칸마다의 뜻 |

## 5. people

대본에 나온 화자 목록이다. 대사의 `speakers` 가 이 id 를 가리킨다.

```json
{"id": "P01", "name": "중호", "aliases": []}
```

`aliases` 는 같은 인물의 다른 표기다. 두 영화에서는 모두 빈 목록이다. 추격자 55명, 써니 41명.

## 6. scenes

대본의 씬 목록이다.

```json
{"id": "SC004", "number": "4", "heading": "4. 초소 앞 / 밤", "place": "초소 앞", "time": "밤",
 "flags": [], "start": 161.465, "end": 167.485, "timing": "estimated"}
```

| 칸 | 뜻 |
|---|---|
| `number` | 대본에 적힌 씬 번호(글자) |
| `heading` | 씬 머리 줄 전체 |
| `place`, `time` | 장소와 때. 대본에 없으면 null |
| `flags` | montage, flashback, intercut, dream 가운데 해당하는 것 |
| `start`, `end` | 그 씬에서 들린 대사들의 범위 |
| `timing` | 두 영화에서는 모두 `estimated`. 시각을 하나도 못 얻은 씬만 `none` |

씬 구간은 실제 컷 경계가 아니다. 그 씬의 시각 있는 대본 대사와 그 씬으로 붙인 `added` 가 놓인 범위다. 대본 순서와 다른 자리에서 들린 대사는 뺀다. 들린 대사가 없는 씬은 씬 머리와 지문의 어림 시각으로 잡아서 `start` 와 `end` 가 같을 수 있다(추격자 13개, 써니 12개).

## 7. items

### 7.1 대사가 아닌 항목

`type` 이 `scene`(씬 머리), `action`(지문), `onscreen`(화면 글자), `transition`(전환)인 항목이다.

```json
{"id": "I0002", "type": "action", "scene": "SC001", "text": "어두운 골목. 비가 내린다.",
 "start": 3.101, "end": null, "timing": "estimated", "src_el": "E0002"}
```

| 칸 | 뜻 |
|---|---|
| `scene` | 속한 씬 |
| `text` | 대본 글 그대로 |
| `start` | 어림 시각. 대본 순서로 앞뒤 대사의 시각 사이를 글 길이에 비례해 나눈 값이다 |
| `end` | 늘 null |
| `src_el` | 대본 요소 id |
| `kind` | `onscreen` 에만 있다. subtitle, insert, title, other |

`type` 이 `scene` 인 항목은 씬 머리 줄이고, 그 씬의 정보는 `scenes` 에 있다.

### 7.2 대사 항목

`type` 이 `dialogue` 인 항목이고 칸이 23개다. 네 묶음으로 나눠 보면 쉽다.

**누가, 어떻게 말했나 (대본에서 가져온 것)**

| 칸 | 뜻 |
|---|---|
| `scene` | 속한 씬. 대본 대사는 대본 그대로다. `added` 는 시각으로 앞뒤에 있는 대본 대사의 씬 가운데 가까운 쪽을 붙인 것이고, 앞뒤에 대본 대사가 없으면 null 이다 |
| `speakers` | 화자의 인물 id 목록. 보통 한 명, 같이 말한 줄은 둘 이상, 모르면 빈 목록 |
| `speaker_from` | `script`: 대본에 적힌 화자. `unknown`: 대본에 적힌 화자가 아니다. 이때 `speakers` 가 빈 목록이면 화자를 모르는 것이고(대본에 없는 말, 화자 없는 대본 줄), 목록이 있으면 앞 대사에서 이어받은 추정 화자다. 두 영화에는 추정 화자가 없다. `translation_match` 는 번역 대본용 값이고 지금은 쓰지 않는다 |
| `how` | 말하는 방식. on, off(화면 밖), narration, phone, radio, broadcast, video, inner(속마음), unknown |
| `paren` | 대사 안 괄호 지시 목록. `{"text": "웃으며", "at": 0}` 의 `at` 은 `script_ref.text`(괄호 지시를 뺀 대본 글) 안에서 그 지시가 있던 글자 위치다. `heard` 의 위치가 아니다 |
| `sung` | 노래로 부르는 줄이면 true |
| `overlap` | 겹쳐 말하는 대사 묶음 번호. 같은 번호끼리 겹친다. 없으면 null |

**무엇이 들렸나 (영상에서 얻은 것)**

| 칸 | 뜻 |
|---|---|
| `heard` | 들린 글. 받아쓰기 낱말을 공백으로 이은 것 |
| `words` | 낱말 목록. `{"w": 글자, "s": 시작, "e": 끝, "p": 확률}` |
| `start`, `end` | 낱말 가운데 가장 이른 시작과 가장 늦은 끝 |
| `timing` | 들린 줄은 `aligned`, 못 찾은 줄은 `none` |
| `lang` | 그 줄을 말한 언어. 두 영화는 모두 ko |
| `ko` | 자막용 한글. 한국어로 말한 줄은 `heard` 와 같다 |
| `ko_src` | `ko` 의 출처. same(들린 그대로), mt(기계 번역), human. 번역을 하지 않아 지금은 들린 줄은 모두 same 이고 absent 는 null 이다 |

**대본과 견준 결과**

| 칸 | 뜻 |
|---|---|
| `status` | same, changed, added, absent. 아래 7.3 |
| `match` | 무엇과 맞췄나. `text`: 대본 글. `translation`: 번역 대본(지금은 쓰지 않음). 대본에 없는 말은 null |
| `script_ref` | 짝이 된 대본 대사. `{"el": 대본 요소 id, "text": 대본 글}` |
| `diffs` | 대본과 다른 조각 목록. `{"kind": 종류, "script": 대본 쪽 글, "heard": 들린 쪽 글}` |
| `conf` | 이 줄을 얼마나 믿을 수 있나. 0~1. 아래 7.4 |

**출처**

| 칸 | 뜻 |
|---|---|
| `id`, `type` | 항목 id, `dialogue` |
| `src_el` | 대본 요소 id. `script_ref.el` 과 같다. 대본에 없는 말은 null |

### 7.3 status 네 가지

| status | 뜻 | 시각, 낱말, heard | script_ref | diffs |
|---|---|---|---|---|
| `same` | 대본 줄이 그대로 들렸다. 띄어쓰기, 문장부호, 기호, 영문 대소문자 차이는 보지 않는다 | 있음 | 있음 | 없음 |
| `changed` | 대본 줄이 들렸지만 글이 다르다 | 있음 | 있음 | 하나 이상 |
| `added` | 대본에 없는데 들린 말이다. 화자가 없다 | 있음 | null | 없음 |
| `absent` | 대본에 있는데 영상에서 못 찾았다 | 없음(null, 빈 목록) | 있음 | 없음 |

| 영화 | same | changed | added | absent |
|---|---|---|---|---|
| 추격자 | 228 | 490 | 389 | 316 |
| 써니 | 158 | 391 | 640 | 272 |

`diffs` 의 종류는 다섯이다. 틀림없을 때만 앞의 넷을 붙이고 나머지는 word 다.

| kind | 뜻 |
|---|---|
| `filler` | 군말이 들어가거나 빠졌다(어, 음) |
| `repeat` | 말을 되풀이하거나 더듬었다. 끝소리를 늘인 것(야, 야아)도 |
| `number` | 같은 수를 달리 적었다(12시, 열두 시) |
| `spelling` | 같은 말의 다른 표기다(할게, 할께) |
| `word` | 낱말이 다르다 |

`changed` 에는 차이가 군말, 표기, 숫자 표기뿐인 줄도 든다(추격자 36줄, 써니 14줄). 낱말이 실제로 바뀐 줄만 보려면 `diffs` 에 `word` 가 있는지 본다.

### 7.4 conf

0~1 이고 높을수록 믿을 만하다. status 마다 뜻이 다르다. 계산식 전문은 파일의 `made.conf` 에 있다.

| status | conf 가 말하는 것 |
|---|---|
| same, changed | 이 대본 줄과 이 소리가 맞는 짝인가. 글자가 닮은 정도와 낱말 확률의 평균이다 |
| added | 실제로 한 말인가. 낱말 확률의 평균이다. `rescue` pass 에서 온 줄은 0.8 을 곱하고, 받아쓰기가 지어냈을 법한 꼴이면 더 낮춘다 |
| absent | 정말 영상에 없었나. 그 자리에 설명 안 된 말이 많을수록 낮다. 범위는 0.35~0.95 |

글자 없이 문장부호나 지시만 있는 대본 대사는 `absent` 이고 conf 가 null 이다.

### 7.5 항목 예시

글은 지어낸 것이고 칸의 모양만 실제와 같다.

```json
{"id": "I0020", "type": "dialogue", "scene": "SC005",
 "speakers": ["P03"], "speaker_from": "script", "how": "on", "lang": "ko",
 "heard": "어제 어디 있었어", "ko": "어제 어디 있었어", "ko_src": "same",
 "start": 228.083, "end": 229.423, "timing": "aligned",
 "words": [{"w": "어제", "s": 228.083, "e": 228.503, "p": 0.91},
           {"w": "어디", "s": 228.503, "e": 228.863, "p": 0.88},
           {"w": "있었어", "s": 228.863, "e": 229.423, "p": 0.79}],
 "status": "changed", "match": "text",
 "script_ref": {"el": "E0019", "text": "어제 밤에 어디 있었어?"},
 "diffs": [{"kind": "word", "script": "밤에", "heard": ""}],
 "paren": [{"text": "낮게", "at": 0}], "sung": false, "overlap": null,
 "conf": 0.867, "src_el": "E0019"}
```

대본에 없는 말(`added`)은 화자와 대본 참조가 비어 있다.

```json
{"id": "I0013", "type": "dialogue", "scene": "SC004",
 "speakers": [], "speaker_from": "unknown", "how": "unknown", "lang": "ko",
 "heard": "잠깐만요", "ko": "잠깐만요", "ko_src": "same",
 "start": 166.565, "end": 167.485, "timing": "aligned",
 "words": [{"w": "잠깐만요", "s": 166.565, "e": 167.485, "p": 0.657}],
 "status": "added", "match": null, "script_ref": null, "diffs": [],
 "paren": [], "sung": false, "overlap": null, "conf": 0.526, "src_el": null}
```

영상에서 못 찾은 대본 대사(`absent`)는 들린 쪽이 비어 있다.

```json
{"id": "I0032", "type": "dialogue", "scene": "SC009",
 "speakers": ["P05"], "speaker_from": "script", "how": "on", "lang": "ko",
 "heard": null, "ko": null, "ko_src": null,
 "start": null, "end": null, "timing": "none", "words": [],
 "status": "absent", "match": "text",
 "script_ref": {"el": "E0031", "text": "그만 가자."},
 "diffs": [], "paren": [], "sung": false, "overlap": null,
 "conf": 0.35, "src_el": "E0031"}
```

### 7.6 차례

- 시각 있는 항목은 `start` 순이다.
- `absent` 대사는 시각이 없어서 대본 순서로 끼운다. 대본에서 자기보다 앞선 항목 가운데 가장 가까운 것 바로 뒤에 놓인다.

## 8. quality

영화 전체 집계다. 값은 추격자의 것이다.

| 칸 | 값 | 뜻 |
|---|---|---|
| `dialogue` | 1,423 | 대사 항목 수. same + changed + added + absent |
| `same`, `changed`, `added`, `absent` | 228, 490, 389, 316 | status 별 수 |
| `spoken` | 975 | 말이 있는 대본 대사 수 |
| `silent_turns` | 59 | 말이 없는 대본 대사 수(문장부호나 지시뿐). absent 로 두고 일치율에서 뺀다 |
| `absent_spoken` | 257 | 말이 있는 대본 대사 가운데 absent |
| `line_coverage` | 0.736 | (same + changed) ÷ spoken. 대본 대사 가운데 영상에서 찾은 줄의 몫 |
| `match` | 0.613 | 일치율. 대본 글자 가운데 영상에서 그대로 들린 글자의 몫. `film.script.match`, `catalog.csv` 의 match 와 같다 |
| `mean_conf` | 0.695 | 들린 줄의 conf 평균 |
| `low_conf` | 27 | 들린 줄 가운데 conf 가 0.2 미만인 수 |
| `speaker_unknown` | 392 | `speaker_from` 이 unknown 인 대사 수 |
| `speaker_inferred` | 0 | 그 가운데 추정 화자가 붙은 수 |
| `moved` | 7 | 대본 순서와 다른 자리에서 들린 대본 대사 수. 어느 줄인지는 항목에 적지 않는다 |
| `words` | 4,273 | 낱말 수 |
| `mean_word_p` | 0.77 | 낱말 확률 평균 |
| `by_pass` | plain 835, lv3 160, rescue 112 | 어느 받아쓰기 pass 에서 온 줄인지 |
| `diff_kinds` | word 980, spelling 56, filler 53, number 18, repeat 12 | 차이 조각 종류별 수 |
| `time_fixed`, `time_fixed_words` | 1, 1 | 낱말 시각을 바로잡은 줄 수와 낱말 수 |
| `time_suspect` | 빈 목록 | 바로잡았지만 시각을 믿기 어려운 항목 id |
| `pass_overlaps` | 빈 목록 | 서로 다른 pass 에서 온 두 줄의 시간이 0.6초 넘게 겹친 곳. 비어 있어도 겹침이 없다는 뜻은 아니다(10절) |
| `lang_from_heard` | 0 | 대본의 언어 표시와 달리 들린 글자를 보고 영상 언어로 둔 줄 수 |

써니는 일치율 0.565, line_coverage 0.713 이다.

## 9. 읽는 법

들린 대사를 시각, 화자, 대본과 견준 결과, 글로 찍는 예다.

```python
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")  # 출력을 파일로 받을 때 한글이 깨지지 않게

with open("films/local_0270/final.json", encoding="utf-8") as f:
    d = json.load(f)

names = {p["id"]: p["name"] for p in d["people"]}
for it in d["items"]:
    if it["type"] == "dialogue" and it["timing"] == "aligned":
        who = ", ".join(names[s] for s in it["speakers"])
        if not who:
            who = "(화자 없음)"
        elif it["speaker_from"] != "script":
            who += "?"  # 추정 화자. preview.srt 와 같은 표시
        print(f'{it["start"]:8.2f}  {it["status"]:7}  {who}: {it["heard"]}')
```

눈으로 보려면 `films/{film_id}/preview.srt` 를 영상 플레이어에서 자막으로 띄운다. 이 자막은 `final.json` 으로 만든 것이다.

## 10. 쓸 때 조심할 것

- **정답과 견준 적이 없다.** 사람이 소리를 듣고 만든 정답 표본이 없다. 언어 모델이 글로만 대조한 점검에서는 추격자의 짝 718줄 가운데 틀린 짝이 0이었다(판단 보류 2줄). 써니는 이 점검을 하지 않았다.
- **absent 와 added 가 다 맞지는 않다.** 같은 점검에서 absent 257줄 가운데 30줄은 그 말이 근처 added 줄에 들어 있는 것으로 보였고, added 389줄 가운데 92줄은 대본 줄의 말로 보였다. conf 가 낮은 줄부터 의심한다.
- **지어낸 말이 added 에 남을 수 있다.** 받아쓰기는 말이 없는 곳에 짧은 말을 지어내기도 한다. 같은 점검에서 added 389줄 가운데 36줄이 그렇게 보였고, 대부분 conf 가 낮은 한두 낱말이었다.
- **화자는 대본에 적힌 대로다.** 화면에서 누가 말하는지 확인한 것이 아니다. `added` 는 화자가 없다.
- **대사가 아닌 항목의 시각과 씬 구간은 어림이다.** 지문이 그 시각에 화면에 나온다는 뜻이 아니다.
- **낱말 시각은 받아쓰기 모델이 낸 값이다.** 소리와 글을 다시 맞춰 시각을 재는 일(강제 정렬)은 따로 하지 않았다. 줄의 첫 낱말이 앞의 조용한 구간 쪽으로 늘어나 시작이 실제보다 이를 수 있다. 첫 낱말 길이가 1초를 넘는 줄이 추격자 73줄, 써니 83줄이고 `rescue` pass 에서 온 줄에 많다.
- **한 대사 항목이 긴 침묵을 건너뛸 수 있다.** 줄의 범위는 이어진 받아쓰기 낱말로만 정해서, 낱말 사이에 긴 틈이 낀 줄이 있다. 10초 넘는 줄이 추격자 21줄, 써니 43줄이고, 써니의 `I1922` 는 낱말 3개가 203초에 걸친다. 자막을 나눌 때는 `start`, `end` 대신 `words` 의 시각을 쓴다.
- **대사 항목끼리 시간이 조금 겹칠 수 있다.** 앞 줄이 끝나기 전에 시작하는 줄이 추격자 42곳, 써니 28곳이고 모두 0.6초 이하다. 시작과 끝이 같은 낱말도 있다(추격자 20개, 써니 27개).
- **어느 pass 에서 온 줄인지는 항목에 없다.** `work\align.json` 의 `ops` 에서 `item` 이 그 항목 id 인 것을 찾아 `asr` 값의 앞부분(`plain:`, `lv3:`, `rescue:`)을 본다.
- **외국어 대사는 번역하지 않는다.** 외국어로 들린 줄이 있는 영화는 `final.json` 을 만들지 않고 멈춘다. 그래서 지금은 외국어 대사가 없는 한국어 영화만 된다.
- **`final.json` 은 저장소에 없다.** 대사 글이 들어 있어서 `films/` 를 git 에 넣지 않는다. 영상과 대본이 있는 PC 에서 만들어야 한다.
