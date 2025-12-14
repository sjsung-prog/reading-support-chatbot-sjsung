import os
import zipfile
import gdown

import streamlit as st

from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ✅ Streamlit은 set_page_config가 "첫 Streamlit 호출"이어야 경고가 안 남
st.set_page_config(page_title="학교도서관 독서활동 지원 RAG 챗봇", page_icon="📚")


# 🔑 Streamlit Cloud의 secrets.toml 에서 UPSTAGE_API_KEY를 가져와서 환경변수로 설정
if "UPSTAGE_API_KEY" in st.secrets:
    os.environ["UPSTAGE_API_KEY"] = st.secrets["UPSTAGE_API_KEY"]


def guess_source_name(metadata: dict) -> str:
    """메타데이터에서 문서명/URL처럼 보이는 값을 최대한 찾아서 반환(페이지 없이 문서명만)."""
    if not metadata:
        return "unknown"

    # 흔히 들어오는 키들 우선 탐색
    for k in ["source", "file_name", "filename", "file_path", "path", "url", "link", "title", "name"]:
        v = metadata.get(k)
        if isinstance(v, str) and v.strip():
            s = v.strip()
            # 경로면 파일명만
            if "/" in s or "\\" in s:
                s = os.path.basename(s)
            return s

    # 그래도 없으면 metadata 값 중 문자열 하나라도
    for _, v in metadata.items():
        if isinstance(v, str) and v.strip():
            return v.strip()

    return "unknown"


# ✅ Google Drive 에서 chroma_db.zip 내려받아서 풀기
def download_and_unpack_chroma_db():
    # ⚠️ 여기에 네 Google Drive 파일 ID 넣기!
    file_id = "1XXyTjn8-yxa795E3k4stplJfNdFDyro2"
    url = f"https://drive.google.com/uc?id={file_id}"

    # 이미 폴더가 있고 안에 파일이 있으면 재다운로드 안 함
    if os.path.exists("chroma_db") and os.listdir("chroma_db"):
        # st.write 대신 print: (캐시/재실행 시 UI가 지저분해질 수 있어서)
        print("✅ chroma_db 폴더 이미 존재 → 다운로드 생략")
        return

    # 혹시 이전에 깨진 zip이 남아 있을 수도 있으니 삭제
    if os.path.exists("chroma_db.zip"):
        os.remove("chroma_db.zip")

    st.write("⬇ Google Drive에서 벡터 DB(chroma_db.zip)를 불러오는 중입니다...")

    # 🔽 gdown이 구글 드라이브의 각종 확인/토큰 처리를 알아서 해줌
    gdown.download(url, "chroma_db.zip", quiet=False)

    # 다운이 너무 작으면 (HTML 페이지만 받아온 경우 대비)
    size = os.path.getsize("chroma_db.zip")
    if size < 1000:  # 1KB도 안 된다? → 거의 HTML 에러 페이지
        st.error(
            "❌ chroma_db.zip 파일 크기가 비정상적으로 작습니다. "
            "구글 드라이브 공유 설정(링크가 있는 모든 사용자 보기)을 다시 확인해 주세요."
        )
        return

    try:
        with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
            zip_ref.extractall(".")
    except zipfile.BadZipFile:
        st.error(
            "❌ ZIP 파일을 열 수 없습니다. 구글 드라이브에 올라간 파일이 "
            "정상적인 chroma_db.zip인지 다시 확인해 주세요."
        )
        return

    st.success("✅ chroma_db 준비 완료!")


@st.cache_resource
def load_rag_chain():
    """Google Drive에서 chroma_db를 내려받고, Chroma + Upstage LLM으로 RAG 체인 구성"""

    # 1) chroma_db 없으면 Google Drive에서 받아오기
    download_and_unpack_chroma_db()

    # 2) 임베딩 + 벡터스토어 로드
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 3) 프롬프트: 학교도서관 독서지원 사서 역할
    prompt = ChatPromptTemplate.from_template(
        """
너는 학교도서관에서 학생들의 독서활동을 도와주는 도우미야.
아래 '참고 문서(context)' 내용을 바탕으로, 학생의 질문에 대해
친절하고 구체적인 답변을 한국어로 작성해줘.

가능하면:
- 도서관 이용 규정, 대출/반납/연장 방법
- 책 고르는 방법, 독후감 작성법, 독서 토론 방법
등을 중심으로 설명해 줘.

만약 문서에 정보가 없으면 모르는 부분은 솔직하게 모른다고 말해.

[참고 문서]
{context}

[학생의 질문]
{question}
        """
    )

    # 4) Upstage LLM
    llm = ChatUpstage()  # 기본 solar-1-mini 사용 (secrets의 키 필요)

    # 5) RAG 체인 (기존 그대로)
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # ✅ retriever를 함께 반환 (문서명 표시용)
    return rag_chain, retriever


# -------------------------
# Streamlit 챗봇 UI 부분
# -------------------------
st.title("📚 학교도서관 독서활동 지원 RAG 챗봇")
st.caption("도서관 소장자료와 독서교육 자료를 기반으로 독서 관련 질문에 답해주는 챗봇입니다.")

with st.sidebar:
    st.subheader("ℹ️ 사용 안내")
    st.markdown(
        """
**예시 질문**

- 독서모임 진행 방법이 궁금해.
- 독후감 작성 팁 알려줄 수 있어?
- 독서토론 기법의 종류에 대해 설명해줘.
- 중학생이 읽기 좋은 글쓰기 관련 책 추천해줘.

답변은 미리 인덱싱해 둔 도서관 소장자료와 독서교육 자료를 우선적으로 활용해서 생성됩니다.
        """
    )

# 실제 RAG 체인 + retriever 준비
rag_chain, retriever = load_rag_chain()

# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 지금까지의 대화 보여주기
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
user_input = st.chat_input("독서방법, 독서활동, 도서관 이용 등에 대해 궁금한 것을 물어보세요.")

if user_input:
    # 사용자 메시지 화면에 추가
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("생각 중입니다..."):
            # ✅ (추가) 문서명 표시를 위해 docs를 먼저 얻어둠 (코랩 재실행 필요 없음)
            docs = retriever.get_relevant_documents(user_input)

            # 기존 RAG 호출
            answer = rag_chain.invoke(user_input)
            st.markdown(answer)

        # ✅ (추가) 문서명/URL만 간단 표시 (페이지 없음)
        with st.expander("📌 참고 자료(문서명) 보기"):
            if not docs:
                st.info("검색된 참고 자료가 없습니다.")
            else:
                # 중복 제거해서 깔끔하게
                names = []
                for d in docs:
                    name = guess_source_name(d.metadata or {})
                    if name not in names:
                        names.append(name)

                for n in names:
                    st.markdown(f"- {n}")

    # 어시스턴트 응답도 히스토리에 저장
    st.session_state["messages"].append({"role": "assistant", "content": answer})
