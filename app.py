import os
import zipfile
import gdown

import streamlit as st

from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ✅ set_page_config는 가능한 한 위에서 1번만 (Streamlit 경고 방지)
st.set_page_config(page_title="학교도서관 독서활동 지원 챗봇", page_icon="📚")


# 🔑 Streamlit Cloud의 secrets.toml 에서 UPSTAGE_API_KEY를 가져와서 환경변수로 설정
if "UPSTAGE_API_KEY" in st.secrets:
    os.environ["UPSTAGE_API_KEY"] = st.secrets["UPSTAGE_API_KEY"]


# ✅ Google Drive 에서 chroma_db.zip 내려받아서 풀기
def download_and_unpack_chroma_db():
    # ⚠️ 여기에 네 Google Drive 파일 ID 넣기!
    file_id = "1XXyTjn8-yxa795E3k4stplJfNdFDyro2"
    url = f"https://drive.google.com/uc?id={file_id}"

    # 이미 폴더가 있고 안에 파일이 있으면 재다운로드 안 함
    if os.path.exists("chroma_db") and os.listdir("chroma_db"):
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

    # 5) RAG 체인
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# 실제 RAG 체인 준비
rag_chain = load_rag_chain()


# -------------------------
# Streamlit 챗봇 UI 부분
# -------------------------
st.title("📚 학교도서관 독서활동 지원 챗봇")
st.caption("도서관 소장자료와 독서교육 자료를 참고하여 독서 관련 질문에 답해주는 챗봇입니다.")


# ✅ 사이드바 탭(분류) + 독서활동 예시(회색)
with st.sidebar:
    st.subheader("📌 메뉴")
    menu = st.radio(
        "기능 선택",
        ["도서관 이용 안내", "책 추천", "독서활동"],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    if menu == "도서관 이용 안내":
        st.markdown("**도서관 이용 방법에 대해 답해드립니다.**")
        st.caption("예: 대출/반납 방법, 이용시간, 도서 검색 안내 등")

        st.markdown("**예시 질문**")
        st.caption("• 대출 권수 및 기간이 어떻게 돼?")
        st.caption("• 도서관 홈페이지 이용법 알려줘")
        st.caption("• 신간도서 신청하려면 어떻게 해?")

    elif menu == "책 추천":
        st.markdown("**학생 상황에 맞는 책 추천을 도와드려요.**")
        st.caption("예: 학년/관심 주제/분량/장르에 맞춘 추천")

        st.markdown("**예시 질문**")
        st.caption("• 중학생이 읽기 좋은 과학 책 추천해줘.")
        st.caption("• 우정/관계 주제 소설 3권 추천해줘.")
        st.caption("• 짧고 재미있는 추리소설 있어?")

    else:  # 독서활동
        st.markdown("**독서활동 관련 도움을 드려요.**")

        # 회색 글씨로 연하게
        st.caption("📖 읽기 활동 ex) 올바른 독서법")
        st.caption("✍️ 쓰기 활동 ex) 독서감상문, 서평, 독서논술 등")
        st.caption("👥 그룹 활동 ex) 독서토론, 독서동아리 등")

        st.markdown("**예시 질문**")
        st.caption("• 독후감 서론을 어떻게 시작하면 좋을까?")
        st.caption("• 독서토론 질문을 잘 만드는 방법은?")
        st.caption("• 서평과 독후감 차이가 뭐야?")


# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 지금까지의 대화 보여주기
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ✅ menu(탭) 선택에 맞춰 질문을 조금 더 명확하게 전달 (탭이 ‘진짜 기능’처럼 보이게)
MODE_PREFIX = {
    "도서관 이용 안내": "[도서관 이용 안내] ",
    "책 추천": "[책 추천] ",
    "독서활동": "[독서활동] "
}

user_input = st.chat_input("궁금한 것을 입력하세요. (예: 대출 연장 방법 / 책 추천 / 독후감 팁)")

if user_input:
    # 사용자 메시지 화면에 추가
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 탭 선택을 반영한 질의
    query_for_chain = MODE_PREFIX.get(menu, "") + user_input

    # RAG 호출
    with st.chat_message("assistant"):
        with st.spinner("생각 중입니다..."):
            answer = rag_chain.invoke(query_for_chain)
            st.markdown(answer)

    # 어시스턴트 응답도 히스토리에 저장
    st.session_state["messages"].append({"role": "assistant", "content": answer})

