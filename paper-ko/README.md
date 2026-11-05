# paper-ko — 논문 PDF 자리 그대로 한국어 번역

영어 논문 PDF 를 넣으면 **글자가 있던 자리에 한국어를 그대로 넣은 PDF** 를 만든다.
그림·표·수식·쪽 배치는 원본 그대로 두고 문단 글자만 바꾼다.

## 쓰는 법

- **창 프로그램**: `dist\논문번역기.exe` 를 실행하고 PDF 를 창에 끌어다 놓은 뒤 [번역 시작].
  exe 아이콘 위에 PDF 를 바로 끌어다 놓아도 목록에 들어간다. 다른 PC 로 exe 하나만 복사하면 된다.
- **끌어다 놓기(스크립트)**: PDF 를 `논문번역.bat` 아이콘 위로 끌어다 놓는다 (여러 개 가능).
- **명령줄**:
  ```
  python paper_ko.py 논문.pdf                  # Claude 로 번역 (기본, 품질 좋음)
  python paper_ko.py 논문.pdf --pages 1-5      # 일부 쪽만
  python paper_ko.py 논문.pdf --engine google  # 무료 구글 번역 (빠르지만 품질 낮고, 많이 보내면 막힘)
  python paper_ko.py 논문.pdf --model opus     # 더 좋은 모델 (느림)
  python paper_ko.py 논문.pdf --debug          # 문단 상자와 건너뛴 이유 표시 (점검용)
  ```

결과는 원본 옆에 생긴다.

| 파일 | 내용 |
|---|---|
| `논문_ko.pdf` | 번역본 |
| `논문_ko_dual.pdf` | 왼쪽 원문, 오른쪽 번역 나란히 (`--no-dual` 이면 안 만듦) |

## 번역 엔진

- `claude` (기본): 이 PC 에 로그인된 Claude Code(`claude -p`)를 부른다. **API 키 환경변수를 지우고 부르므로 구독 사용량으로만 돌고 API 요금은 나가지 않는다.** 15쪽 논문 약 1분.
- `google`: 키 없는 무료 주소. 짧은 시간에 많이 보내면 구글이 막는다(그때 남은 문단은 영어로 남김).

### Claude 연결 (exe 창의 "Claude 연결" 칸)

exe 는 Claude 를 직접 부르지 않고, 그 PC 에 설치된 **Claude Code(`claude.exe`)** 를 불러 번역한다.
그래서 쓰는 PC 에 Claude Code 가 있어야 한다 (창의 [Claude Code 설치] = 공식 명령 `irm https://claude.ai/install.ps1 | iex`).

인증은 아래 순서로 처음 있는 것을 쓴다 (기록 칸에 `Claude 인증: …` 으로 어느 것을 썼는지 나온다).

| 순서 | 방법 | 하는 일 |
|---|---|---|
| 1 | 창의 '구독 토큰' 칸 | 이번 실행에만 씀, 저장 안 함 |
| 2 | **`.env` 파일** | exe(스크립트는 `paper_ko.py`) 와 같은 폴더의 `.env` 에서 `CLAUDE_CODE_OAUTH_TOKEN=` 한 줄만 읽음 |
| 3 | 환경변수 `CLAUDE_CODE_OAUTH_TOKEN` | Windows 사용자 환경변수에 넣어 둔 경우 |
| 4 | 이 PC 로그인 | [Claude 로그인] → 브라우저에서 로그인 |

`.env` 쓰는 법: [토큰 받기] (`claude setup-token`, 1년짜리) → 창의 [.env 열기] (없으면 양식을 만들어 메모장으로 엶) →
`CLAUDE_CODE_OAUTH_TOKEN=` 뒤에 토큰을 붙여넣고 저장.

```
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

- `.env` 에서는 이 한 줄만 읽는다. 같은 파일에 `ANTHROPIC_API_KEY` 가 있어도 **읽지 않고**, claude 를 부를 때 환경에서도 지운다 → API 요금이 나갈 수 없다.
- 토큰은 Claude Code 공식 변수로 `claude.exe` 에만 넘긴다. 프로그램이 토큰을 따로 저장하거나 Anthropic 에 직접 보내지 않는다 (정책상 금지).
- 토큰은 계정 비밀번호와 같다. exe 를 남에게 줄 때 `.env` 는 빼고 준다 (구독은 개인용).
- 토큰이 틀리거나 만료돼서 번역을 하나도 못 하면, 결과 파일을 만들지 않는다 (예전 번역본을 영어로 덮어쓰지 않음).
- [연결 확인] 은 Haiku 로 아주 짧은 요청 한 번을 보내 본다.

번역 결과는 `cache\` 에 저장된다 (exe 는 `%LOCALAPPDATA%\paper-ko\cache`, 마지막 기록은 같은 폴더 `last_log.txt`). 같은 논문을 다시 돌리면 저장된 번역을 쓰고, 실패한 문단만 다시 번역한다.

## 무엇을 번역하고 무엇을 두나

| 번역함 | 그대로 둠 |
|---|---|
| 제목, 초록, 본문 문단, 소제목, 그림·표 캡션, 각주 | 수식 위주 줄, 숫자 위주 표 칸, 그림 속 글자, 저자·소속·이메일, 참고문헌 목록, 세로 글자(arXiv 도장) |

문단 안의 짧은 수식(예: d_k, √d_k)은 `{1}` 자리표시로 바꿔 번역한 뒤 첨자·기울임을 살려 다시 넣는다.

## 한계

- 스캔한 PDF(글자가 그림)는 안 된다.
- 한국어가 원문보다 길면 그 문단만 글자가 작아진다.
- 문단 안 수식의 분수·큰 기호는 한 줄 글자로 단순화된다.
- 문단마다 따로 번역하므로 같은 용어가 문단에 따라 조금 다르게 옮겨질 수 있다.

## 필요한 것

- exe: Windows 10/11, Claude Code (Claude 엔진을 쓸 때). 글꼴은 나눔명조·나눔고딕이 있으면 쓰고, 없으면 맑은 고딕.
- 스크립트: Python 3, `pip install pymupdf tkinterdnd2` (창은 `python gui.py`).
- exe 다시 만들기: `build_exe.bat` (깨끗한 가상환경에서 빌드. 전역 Python 으로 빌드하면 torch 등이 딸려 들어가 오래 걸리고 커진다).
