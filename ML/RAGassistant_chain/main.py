from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import os
import re
from contextlib import redirect_stderr
from email.message import EmailMessage
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, after_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.graph.message import add_messages
from openai import BadRequestError
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "research_notes"
CHROMA_DIR = BASE_DIR / ".chroma"
DRAFTS_DIR = BASE_DIR / "drafts"
MEMORY_DIR = BASE_DIR / "memory"
CONTACTS_PATH = MEMORY_DIR / "contacts.json"
GOOGLE_SECRETS_DIR = BASE_DIR / "secrets" / "google"
GMAIL_CREDENTIALS_PATH = GOOGLE_SECRETS_DIR / "credentials.json"
GMAIL_TOKEN_PATH = GOOGLE_SECRETS_DIR / "token.json"
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]
_RETRIEVER_CACHE: tuple[BM25Retriever, Chroma] | None = None
_CHECKPOINTER = InMemorySaver()


class EmailContent(BaseModel):
    subject: str = Field(description="Clean email subject line only")
    body: str = Field(description="Actual email body only, without assistant notes")
    notes_to_user: str = Field(
        default="",
        description="Optional side note for the user, not for the email body",
    )


class AssistantState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    final_response: str


def load_research_documents() -> list[Document]:
    """
    5장 Load 단계.
    연구 노트(.md)와 논문/자료 PDF(.pdf)를 읽어 온다.
    """
    markdown_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    pdf_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=False,
    )

    documents: list[Document] = []
    documents.extend(markdown_loader.load())

    # 일부 PDF는 pypdf가 구조 경고를 stderr로 출력한다.
    # 데모용 CLI에서는 해당 경고보다 검색 결과가 중요하므로 로그는 숨긴다.
    with redirect_stderr(io.StringIO()):
        documents.extend(pdf_loader.load())

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    5장 Split 단계.
    긴 문서를 적절한 크기의 청크로 나눠 검색 정확도를 높인다.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    return splitter.split_documents(documents)


def build_local_retrievers() -> tuple[BM25Retriever, Chroma]:
    """
    5장 Embed / Store / Retrieve 단계.
    - BM25: 정확한 용어, 논문명, 약어 검색에 유리
    - Vector store: 의미 기반 질문에 유리
    """
    global _RETRIEVER_CACHE

    if _RETRIEVER_CACHE is not None:
        return _RETRIEVER_CACHE

    documents = load_research_documents()
    split_docs = split_documents(documents)

    bm25 = BM25Retriever.from_documents(split_docs)
    bm25.k = 4

    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    # 벡터 DB
    # 실무에서는 Pinecone, FAISS, Milvus 등 다양한 벡터 DB를 사용하지만, 로컬 환경에서 가장 빠르고 직관적으로 확인하기 좋은 Chroma를 사용.
    vectorstore = Chroma.from_documents(
        documents=split_docs, # split한 문서 리스트
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR), # 데이터 저장 폴더
        collection_name="research_notes", # db내의 이름
    )

    _RETRIEVER_CACHE = (bm25, vectorstore)
    return _RETRIEVER_CACHE


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0,
    )


def slugify_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", value).strip("_")
    return cleaned or "draft"


def load_contacts() -> dict:
    """
    이름-이메일 매핑을 저장하는 간단한 장기 메모리 저장소를 읽는다.
    """
    if not CONTACTS_PATH.exists():
        return {}
    return json.loads(CONTACTS_PATH.read_text(encoding="utf-8"))


def save_contacts(contacts: dict) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    CONTACTS_PATH.write_text(
        json.dumps(contacts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_contact_memory(
    recipient_name: str,
    recipient_email: str,
    recipient_role: str = "",
    notes: str = "",
) -> None:
    """
    이름과 이메일을 장기 메모리처럼 저장한다.
    """
    if not recipient_name or not recipient_email:
        return

    contacts = load_contacts()
    key = recipient_name.strip().lower()
    contacts[key] = {
        "name": recipient_name.strip(),
        "email": recipient_email.strip(),
        "role": recipient_role.strip(),
        "notes": notes.strip(),
        "updated_at": dt.datetime.now().isoformat(),
    }
    save_contacts(contacts)


def lookup_contact_memory(recipient_name: str) -> dict | None:
    """
    저장된 연락처에서 이름으로 사람 정보를 찾는다.
    """
    if not recipient_name:
        return None
    contacts = load_contacts()
    return contacts.get(recipient_name.strip().lower())


def get_gmail_service():
    """
    Gmail Draft API 클라이언트를 준비한다.
    Google 공식 Python quickstart 흐름에 맞춰 credentials.json + token.json을 사용한다.
    """
    if not GMAIL_CREDENTIALS_PATH.exists():
        return None

    creds = None

    if GMAIL_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(GMAIL_TOKEN_PATH), GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(GMAIL_CREDENTIALS_PATH),
                GMAIL_SCOPES,
            )
            creds = flow.run_local_server(port=0)

        GOOGLE_SECRETS_DIR.mkdir(parents=True, exist_ok=True)
        GMAIL_TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    return build("gmail", "v1", credentials=creds)


def write_local_draft_file(
    recipient_name: str,
    recipient_role: str,
    subject: str,
    body: str,
    recipient_email: str = "",
) -> Path:
    """
    로컬 draft 파일을 기록하고 경로를 반환한다.
    """
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{slugify_filename(recipient_name)}.md"
    draft_path = DRAFTS_DIR / filename

    draft_text = "\n".join(
        [
            f"To: {recipient_name}",
            f"Role: {recipient_role}",
            f"Email: {recipient_email or '(not provided)'}",
            f"Subject: {subject}",
            "",
            body.strip(),
        ]
    )
    draft_path.write_text(draft_text + "\n", encoding="utf-8")
    return draft_path


def sanitize_email_body(body: str) -> str:
    """
    이메일 본문에 섞여 들어온 assistant 메타 코멘트를 제거한다.
    """
    cleaned = body.strip()

    stop_markers = [
        "내가 해줄 수 있는 다음 작업",
        "필요하면 이 메일을 바로 보낼 수 있게 수정해서 보낼게",
        "필요하면 이 메일을 바로 보낼 수 있게",
        "더 추가할 내용",
        "초안 내용 수정",
        "메일 내용 수정",
        "원하면 내가",
    ]

    for marker in stop_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].rstrip()

    lines = cleaned.splitlines()
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") and any(
            keyword in stripped for keyword in ["초안", "수정", "추가", "보내기"]
        ):
            continue
        filtered_lines.append(line)

    cleaned = "\n".join(filtered_lines).strip()
    return cleaned or body.strip()


def structure_email_content(
    subject: str,
    body: str,
    recipient_name: str,
    recipient_role: str,
) -> EmailContent:
    """
    이메일 초안 생성 전 subject/body를 구조화해 메타 코멘트를 분리한다.
    Gmail 본문에는 body만 들어가고, 사용자 안내는 notes_to_user로 분리된다.
    """
    llm = get_llm().with_structured_output(EmailContent)
    prompt = f"""
    You are cleaning an email draft before it is saved or sent.

    Recipient name: {recipient_name}
    Recipient role: {recipient_role}

    Rules:
    - Keep only the real email content in `body`.
    - Remove assistant meta-comments such as:
      "원하시면", "첨부해 드릴게요", "추가할 내용", "내가 해줄 수 있는 다음 작업"
    - If there are follow-up suggestions for the user, put them in `notes_to_user`.
    - Keep the tone appropriate for the recipient role.
    - `subject` must contain only the actual email subject.
    - `body` must be ready to paste into Gmail as-is.

    Input subject:
    {subject}

    Input body:
    {body}
    """
    try:
        return llm.invoke(prompt)
    except Exception:
        return EmailContent(
            subject=subject.strip(),
            body=sanitize_email_body(body),
            notes_to_user="",
        )


def parse_draft_file(draft_path: Path) -> dict[str, str]:
    """
    로컬 draft 파일에서 이메일 메타데이터와 본문을 파싱한다.
    """
    raw = draft_path.read_text(encoding="utf-8")
    header_part, _, body_part = raw.partition("\n\n")
    headers: dict[str, str] = {}
    for line in header_part.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    headers["body"] = body_part.strip()
    return headers


def format_draft_preview(draft_path: Path) -> str:
    """
    채팅에서 바로 볼 수 있는 draft 미리보기를 만든다.
    """
    draft = parse_draft_file(draft_path)
    preview_lines = [
        "Draft Preview",
        f"To: {draft.get('to', '')}",
        f"Role: {draft.get('role', '')}",
        f"Email: {draft.get('email', '')}",
        f"Subject: {draft.get('subject', '')}",
        "",
        draft.get("body", ""),
        "",
        f"Saved at: {draft_path}",
    ]
    return "\n".join(preview_lines)


def print_assistant_message(message: str) -> None:
    """
    터미널 채팅 출력을 조금 더 읽기 쉽게 정리한다.
    """
    print("\n" + "=" * 60)
    print("Assistant")
    print("-" * 60)
    print(message.strip())
    print("=" * 60)


def clean_agent_response_text(text: str) -> str:
    """
    최종 사용자 응답에서 불필요한 공백과 과도한 줄바꿈을 정리한다.
    """
    cleaned = text.replace("\r\n", "\n").strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned


def send_email_via_gmail(
    recipient_email: str,
    subject: str,
    body: str,
) -> str:
    """
    Gmail API로 실제 메일을 전송한다.
    이 함수는 human-in-the-loop 승인 뒤에만 호출되도록 설계한다.
    """
    service = get_gmail_service()
    if service is None:
        raise ValueError("Gmail OAuth is not configured. Cannot send email.")

    message = EmailMessage()
    message["To"] = recipient_email
    message["Subject"] = subject
    message.set_content(body.strip())

    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    send_body = {"raw": encoded_message}
    response = service.users().messages().send(userId="me", body=send_body).execute()
    message_id = response.get("id", "(unknown)")
    return f"Email sent successfully. Message id: {message_id}"


def choose_local_search_strategy(query: str) -> str:
    """
    질문 유형에 따라 로컬 검색 방식을 고르는 간단한 규칙 기반 로직.
        - BM25: 논문명, 약어, 정확한 표현 검색에 유리
        - Vector: 설명, 의미, 개념, 차이점 같은 의미 기반 질문에 유리
        - Hybrid: 둘 다 필요한 복합 질문 (예: "논문 제목이랑 핵심 아이디어 설명해줘")
    """
    lowered = query.lower()

    exact_keywords = [
        "논문",
        "약어",
        "정확한 표현",
        "원문",
        "제목",
        "pdf",
    ]
    semantic_keywords = [
        "설명",
        "요약",
        "핵심",
        "의미",
        "개념",
        "차이",
        "비교",
        "정리",
    ]

    has_exact_pattern = bool(re.search(r"[A-Z]{2,}", query))
    has_exact_keyword = any(keyword in lowered for keyword in exact_keywords)
    has_semantic_keyword = any(keyword in lowered for keyword in semantic_keywords)

    if (has_exact_pattern or has_exact_keyword) and has_semantic_keyword:
        return "hybrid"
    if has_exact_pattern or has_exact_keyword:
        return "bm25"
    return "vector"


def retrieve_local_context(query: str) -> tuple[str, str]:
    """
    개인 연구 문서를 검색하고, 어떤 검색 전략을 썼는지도 함께 반환한다.
    """
    strategy = choose_local_search_strategy(query)
    bm25, vectorstore = build_local_retrievers()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    if strategy == "bm25":
        docs = bm25.invoke(query)
    elif strategy == "hybrid":
        bm25_docs = bm25.invoke(query)
        vector_docs = vector_retriever.invoke(query)

        seen = set()
        docs = []
        for doc in bm25_docs + vector_docs:
            key = (doc.page_content, doc.metadata.get("source"))
            if key not in seen:
                seen.add(key)
                docs.append(doc)
    else:
        docs = vector_retriever.invoke(query)

    context = "\n\n".join(
        f"[source={Path(doc.metadata.get('source', 'unknown')).name}]\n{doc.page_content}"
        for doc in docs
    )
    return strategy, context or "관련 문서를 찾지 못했습니다."


@tool
def search_research_documents(query: str) -> str:
    """
    개인 연구 문서와 PDF에서 정보를 검색합니다.
    연구 노트, 논문, 내부 정리 문서처럼 로컬 자료를 기반으로 답해야 할 때 사용합니다.
    질문 유형에 따라 vector, bm25, hybrid 중 하나를 자동 선택합니다.
    """
    strategy, context = retrieve_local_context(query)
    return f"[local_search_strategy={strategy}]\n{context}"


@tool
def search_web(query: str) -> str:
    """
    외부 웹 검색이 필요할 때 사용합니다.
    최신 정보, 공개 자료, 최근 동향, 외부 사실 확인 질문에 사용합니다.
    """
    search_tool = DuckDuckGoSearchResults(max_results=5)
    return search_tool.run(query)


@tool
def save_contact(
    recipient_name: str,
    recipient_email: str,
    recipient_role: str = "",
    notes: str = "",
) -> str:
    """
    사람 이름과 이메일을 장기 메모리처럼 저장합니다.
    이후에는 이름만으로도 이메일 초안 생성에 활용할 수 있습니다.
    """
    save_contact_memory(
        recipient_name=recipient_name,
        recipient_email=recipient_email,
        recipient_role=recipient_role,
        notes=notes,
    )
    return f"Saved contact memory for {recipient_name} <{recipient_email}>."


@tool
def lookup_contact(recipient_name: str) -> str:
    """
    저장된 연락처 장기 메모리에서 이름으로 정보를 조회합니다.
    """
    contact = lookup_contact_memory(recipient_name)
    if not contact:
        return f"No saved contact found for {recipient_name}."
    return json.dumps(contact, ensure_ascii=False)


@tool
def create_email_draft(
    recipient_name: str,
    recipient_role: str,
    subject: str,
    body: str,
    recipient_email: str = "",
) -> str:
    """
    이메일 초안을 로컬 파일로 저장합니다.
    본문에는 이메일 내용만 넣어야 하며, 설명문이나 메타 코멘트는 넣지 않습니다.
    실제 전송은 하지 않고 draft만 만듭니다.

    recipient_role 예시:
    - friend
    - colleague
    - junior
    - professor
    - first_author
    """
    if not recipient_email:
        contact = lookup_contact_memory(recipient_name)
        if contact:
            recipient_email = contact.get("email", "")
            recipient_role = recipient_role or contact.get("role", "")

    structured_email = structure_email_content(
        subject=subject,
        body=body,
        recipient_name=recipient_name,
        recipient_role=recipient_role,
    )
    subject = structured_email.subject.strip() or subject.strip()
    body = structured_email.body.strip() or sanitize_email_body(body)

    draft_path = write_local_draft_file(
        recipient_name=recipient_name,
        recipient_role=recipient_role,
        subject=subject,
        body=body,
        recipient_email=recipient_email,
    )

    if not recipient_email:
        preview = format_draft_preview(draft_path)
        return (
            f"Recipient email was missing, so a local draft was saved to {draft_path}. "
            f"\n\n{preview}\n"
            f"{f' Note: {structured_email.notes_to_user}' if structured_email.notes_to_user else ''} "
            "No email was sent."
        )

    save_contact_memory(
        recipient_name=recipient_name,
        recipient_email=recipient_email,
        recipient_role=recipient_role,
    )

    service = get_gmail_service()
    if service is None:
        preview = format_draft_preview(draft_path)
        return (
            f"Gmail credentials were not configured, so a local draft was saved to {draft_path}. "
            f"\n\n{preview}\n"
            f"{f' Note: {structured_email.notes_to_user}' if structured_email.notes_to_user else ''} "
            "No email was sent."
        )

    try:
        message = EmailMessage()
        message["To"] = recipient_email
        message["Subject"] = subject
        message.set_content(body.strip())

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"message": {"raw": encoded_message}}
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body=create_message)
            .execute()
        )
        draft_id = draft.get("id", "(unknown)")
        preview = format_draft_preview(draft_path)
        return (
            f"Gmail draft created successfully for {recipient_name} <{recipient_email}>. "
            f"Draft id: {draft_id}. Local draft copy: {draft_path}."
            f"\n\n{preview}\n"
            f"{f' Note: {structured_email.notes_to_user}' if structured_email.notes_to_user else ''} "
            "No email was sent."
        )
    except HttpError as error:
        preview = format_draft_preview(draft_path)
        return (
            f"Gmail draft creation failed with {error}. "
            f"A local draft was saved to {draft_path}."
            f"\n\n{preview}\n"
            f"{f' Note: {structured_email.notes_to_user}' if structured_email.notes_to_user else ''} "
            "No email was sent."
        )


@tool
def send_email(
    recipient_name: str,
    recipient_role: str,
    subject: str,
    body: str,
    recipient_email: str = "",
) -> str:
    """
    이메일을 실제로 전송합니다.
    이 tool은 human-in-the-loop 승인 후에만 실행되어야 합니다.
    """
    if not recipient_email:
        contact = lookup_contact_memory(recipient_name)
        if contact:
            recipient_email = contact.get("email", "")
            recipient_role = recipient_role or contact.get("role", "")

    if not recipient_email:
        raise ValueError("Recipient email is required to send an email.")

    structured_email = structure_email_content(
        subject=subject,
        body=body,
        recipient_name=recipient_name,
        recipient_role=recipient_role,
    )
    subject = structured_email.subject.strip() or subject.strip()
    body = structured_email.body.strip() or sanitize_email_body(body)

    draft_path = write_local_draft_file(
        recipient_name=recipient_name,
        recipient_role=recipient_role,
        subject=subject,
        body=body,
        recipient_email=recipient_email,
    )

    save_contact_memory(
        recipient_name=recipient_name,
        recipient_email=recipient_email,
        recipient_role=recipient_role,
    )

    send_result = send_email_via_gmail(
        recipient_email=recipient_email,
        subject=subject,
        body=body,
    )
    return (
        f"{send_result} Local copy: {draft_path}."
        f"{f' Note: {structured_email.notes_to_user}' if structured_email.notes_to_user else ''}"
    )


@after_agent(state_schema=AssistantState, name="FinalizeResponseMiddleware")
def finalize_response(state: AssistantState, _runtime) -> dict[str, str] | None:
    """
    에이전트가 끝난 뒤 사용자에게 보여줄 최종 응답 문자열을 한 번 더 정리한다.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    final_text = clean_agent_response_text(extract_final_answer(messages))
    if not final_text:
        return None
    return {"final_response": final_text}


def build_agent():
    """
    현재 LangChain 버전에 맞춰 `create_agent + @tool + InMemorySaver` 조합으로 구성한다.
    """
    tools = [
        search_research_documents,
        search_web,
        save_contact,
        lookup_contact,
        create_email_draft,
        send_email,
    ]
    system_prompt = """
    너는 개인 연구 문서 기반 RAG 어시스턴트다.

    동작 원칙:
    1. 연구 노트, PDF, 내부 문서 기반 질문이면 `search_research_documents`를 먼저 사용한다.
    2. 최신 정보, 공개 자료, 외부 사실 확인이 필요하면 `search_web`를 사용한다.
    3. 질문이 내부 문서와 외부 정보를 함께 요구하면 두 도구를 모두 사용한다.
    4. 연락처 정보가 주어지면 `save_contact`를 사용해 이름-이메일을 장기 메모리처럼 저장할 수 있다.
    5. 사용자가 누군가에게 공유/전달/이메일 초안 생성을 원하면 조사 후 `create_email_draft`를 사용한다.
    6. 사용자가 실제 발송을 요청하면 `send_email`을 사용하되, 이 tool은 반드시 human approval 뒤에만 실행된다.
    7. 이메일 주소가 없고 이름만 있으면 `lookup_contact`로 저장된 연락처를 조회해 활용한다.
    8. 후속 질문이 오면 이전 대화 맥락을 참고해서 이어서 답한다.

    답변 원칙:
    - 근거가 불충분하면 모른다고 솔직히 말한다.
    - 마지막에 "검색 경로 요약"을 2~3줄로 덧붙인다.
    - 로컬 문서 검색을 썼다면 어떤 검색 전략(vector/bm25/hybrid)이 선택됐는지 반영한다.
    - 이메일 초안을 만들 때는 수신자 관계에 맞춰 톤을 조절한다.
    - `create_email_draft`의 body에는 실제 이메일 본문만 넣는다. "필요하면 수정할게" 같은 assistant 설명 문구는 넣지 않는다.
    - 이메일 초안을 만든 경우, 최종 답변에 draft 파일 위치 또는 Gmail draft 생성 결과를 함께 알려준다.
    """

    hitl_middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "send_email": True,
        }
    )

    return create_agent(
        model=get_llm(),
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=_CHECKPOINTER,
        middleware=[hitl_middleware, finalize_response],
        state_schema=AssistantState,
    )


def make_thread_id(prefix: str) -> str:
    return f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def invoke_with_recovery(agent, payload, thread_id: str):
    """
    HITL 이후 남은 tool 메시지 때문에 thread 상태가 깨지면 새 thread로 한 번 복구한다.
    """
    try:
        result = agent.invoke(
            payload,
            config={"configurable": {"thread_id": thread_id}},
            version="v2",
        )
        return result, thread_id
    except BadRequestError as error:
        message = str(error)
        if "messages with role 'tool'" not in message:
            raise
        fresh_thread_id = make_thread_id("chat")
        print("\nSession state was reset after a previous tool action. Retrying with a fresh chat session.")
        result = agent.invoke(
            payload,
            config={"configurable": {"thread_id": fresh_thread_id}},
            version="v2",
        )
        return result, fresh_thread_id


def ask_once(question: str) -> str:
    """
    단일 질의 모드.
    매 실행마다 별도 thread_id를 만들어 한 번 질문하고 끝낸다.
    """
    agent = build_agent()
    thread_id = make_thread_id("single")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
        version="v2",
    )
    return get_final_response(result)


def extract_final_answer(messages: list[BaseMessage]) -> str:
    """
    create_agent 결과의 messages에서 마지막 assistant 텍스트를 추출한다.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                joined = "\n".join(text for text in texts if text)
                if joined.strip():
                    return joined
    return "모델 응답을 찾지 못했습니다."


def get_result_value(result, key: str, default=None):
    """
    GraphOutput과 dict 둘 다 안전하게 다루기 위한 헬퍼.
    """
    if isinstance(result, dict):
        return result.get(key, default)
    if key == "__interrupt__":
        return getattr(result, "interrupts", default)

    value = getattr(result, "value", None)
    if isinstance(value, dict):
        return value.get(key, default)

    return getattr(result, key, default)


def get_final_response(result) -> str:
    """
    after_agent middleware가 만든 final_response를 우선 사용하고, 없으면 messages에서 추출한다.
    """
    final_response = get_result_value(result, "final_response", "")
    if isinstance(final_response, str) and final_response.strip():
        return final_response
    return extract_final_answer(get_result_value(result, "messages", []))


def run_chat_mode():
    """
    대화형 모드.
    InMemorySaver 기반 checkpointer로 현재 세션의 단기 메모리를 유지한다.
    """
    agent = build_agent()
    thread_id = make_thread_id("chat")
    print("Chat mode started. Type 'exit' to finish.")

    while True:
        question = input("\nYou> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Chat ended.")
            break

        result, thread_id = invoke_with_recovery(
            agent,
            {"messages": [{"role": "user", "content": question}]},
            thread_id,
        )
        had_interrupt = False

        while get_result_value(result, "__interrupt__", None):
            had_interrupt = True
            interrupt_payload = get_result_value(result, "__interrupt__")[0].value
            print("\nApproval required before sending.")
            for idx, action in enumerate(interrupt_payload.get("action_requests", []), start=1):
                args = action.get("args", {})
                print(
                    f"{idx}. to={args.get('recipient_email', '(lookup needed)')} "
                    f"subject={args.get('subject', '')}"
                )
                if args.get("body"):
                    print("Body preview:")
                    print(sanitize_email_body(args["body"]))

            decision = input("Approve send? (yes/no): ").strip().lower()
            if decision in {"yes", "y", "approve"}:
                resume_payload = {"decisions": [{"type": "approve"}]}
            else:
                reason = input("Reason for rejection (optional): ").strip()
                reject_decision = {"type": "reject"}
                if reason:
                    reject_decision["message"] = reason
                resume_payload = {"decisions": [reject_decision]}

            result, thread_id = invoke_with_recovery(
                agent,
                Command(resume=resume_payload),
                thread_id,
            )

        print_assistant_message(get_final_response(result))

        if had_interrupt:
            thread_id = make_thread_id("chat")


def main():
    # 환경 변수 로드 (.env 파일에서 OPENAI_API_KEY 등)
    load_dotenv()

    # argparse로 CLI 옵션 파싱 (터미널 명령어 옵션 받기)
    parser = argparse.ArgumentParser(
        description="Interview-friendly RAG assistant built in a LangChain style."
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive chat session with short-term memory.",
    )
    parser.add_argument(
        "--question",
        help="Ask a single question to the assistant.",
    )
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment variables or .env file.")

    if args.chat:
        run_chat_mode()
        return

    if not args.question:
        raise ValueError("Use --question or --chat.")

    print(ask_once(args.question))


if __name__ == "__main__":
    main()
