# Research RAG Assistant

개인 연구 문서를 기반으로 질문에 답하고, 필요할 때 외부 검색을 결합하며, 결과를 이메일 draft 또는 실제 전송 액션까지 확장할 수 있는 `agentic RAG assistant PoC`입니다.

## 프로젝트 소개

이 프로젝트는 연구 업무에서 자주 발생하는 흐름을 하나의 assistant로 묶는 것을 목표로 합니다.

1. 개인 연구 노트와 PDF 문서를 검색하고
2. 필요할 때 외부 웹 검색으로 관련 정보를 보강한 뒤
3. 결과를 요약하고
4. 협업 상대에게 보낼 이메일 draft를 만들고
5. 실제 전송은 사람 승인 후에만 진행합니다

즉 단순 문서 QA를 넘어서, `검색 -> 정리 -> 공유/전송`까지 이어지는 실무형 PoC입니다.

## 핵심 기능

### 로컬 문서 기반 RAG

- Markdown, PDF 문서를 로드하고
- 문서를 청크로 분할한 뒤
- Chroma 벡터 저장소와 BM25 검색기를 함께 사용합니다.

### 질문 유형별 검색 방식 분기

질문 특성에 따라 다음 전략 중 하나를 선택합니다.

- `bm25`
- `vector`
- `hybrid`

정확한 논문명/약어 검색과 의미 기반 요약/비교 질문을 분리해 처리합니다.

### 외부 검색 결합

내부 문서만으로 부족한 경우 `DuckDuckGoSearchResults` 기반 웹 검색을 함께 사용합니다.

### 연락처 장기 메모리

`memory/contacts.json`에 이름-이메일 매핑을 저장해, 이후 이름만으로도 이메일 draft 생성이나 전송에 활용할 수 있습니다.

### 이메일 draft 생성

조사 결과를 바탕으로 이메일 초안을 생성하고:

- 로컬 draft 파일 저장
- Gmail draft 생성

을 지원합니다.

### Human-in-the-loop 전송 승인

실제 이메일 전송은 `HumanInTheLoopMiddleware`를 통해 사용자 승인 후에만 실행됩니다.

## 프로젝트 구조

```text
.
├── data/
│   └── research_notes/
├── drafts/
├── memory/
│   └── contacts.json
├── secrets/
│   └── google/
│       ├── credentials.json
│       └── token.json
├── main.py
├── requirements.txt
├── README.md
└── README.study.md
```

## 동작 흐름

### 단일 질문 모드

```text
사용자 질문
-> agent가 필요한 tool 선택
-> 로컬 문서 검색 / 웹 검색 / 연락처 조회 / draft 생성 / 전송
-> 최종 응답 반환
```

### 채팅 모드

```text
사용자 입력
-> 같은 thread_id로 agent 실행
-> 필요 시 tool 사용
-> send_email이면 interrupt 발생
-> 사용자 승인/거절
-> Command(resume=...)로 재개
-> 최종 응답 출력
```

## 사용 기술

- Python
- LangChain `create_agent`
- LangGraph `InMemorySaver`
- Chroma
- BM25Retriever
- OpenAI Chat / Embeddings
- Gmail API
- DuckDuckGo Search

## 실행 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env`에 최소한 아래 값을 설정하세요.

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## 실행 예시

### 단일 질문

```bash
python main.py --question "내 연구 노트 기준으로 ADWQ를 설명해줘"
```

### 채팅 모드

```bash
python main.py --chat
```

### 예시 요청

- `내 연구 노트에서 ADWQ를 설명해줘`
- `ADWQ와 비슷한 유명한 논문 2개를 찾아 비교해줘`
- `위 내용을 예은이한테 공유할 이메일 draft를 만들어줘`
- `이제 실제로 보내줘`

## Gmail OAuth 설정

Gmail OAuth 클라이언트 파일은 아래 경로에 둡니다.

- `secrets/google/credentials.json`

최초 실행 시 브라우저 인증을 거치면 아래 파일이 자동 생성됩니다.

- `secrets/google/token.json`

권한 범위를 바꾸면 `token.json`을 삭제하고 다시 인증해야 할 수 있습니다.

## 저장 위치

- 연구 문서: `data/research_notes/`
- 생성된 draft: `drafts/`
- 연락처 장기 메모리: `memory/contacts.json`
- Gmail OAuth 파일: `secrets/google/`
- 벡터 저장소: `.chroma/`

## 이 프로젝트가 다루는 것

이 프로젝트는 다음 주제를 함께 다룹니다.

- 문서 기반 RAG
- 검색 전략 분기
- 외부 검색 tool 결합
- short-term memory
- long-term memory
- 이메일 draft/action tool
- human-in-the-loop middleware

즉 단순 튜토리얼형 RAG가 아니라, action과 memory를 포함한 `agentic workflow`를 다루는 PoC입니다.

## 한계

- production 서비스가 아니라 PoC 중심 프로젝트입니다
- 연락처 장기 메모리는 파일 기반 저장소를 사용합니다
- 검색 전략 분기는 규칙 기반 로직입니다
- UI는 터미널 중심이며 별도 웹 인터페이스는 제공하지 않습니다
- 평가 지표와 자동 테스트는 아직 제한적입니다

