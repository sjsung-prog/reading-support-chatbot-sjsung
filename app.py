import os
import zipfile
import gdown

import streamlit as st

from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ✅ set_page_config는 가능한 한 위에서 1번만
st.set_page_config(page_title="학교도서관 독서활동 지원 챗봇", page_icon="📚")


# 🔑 API KEY
if "UPSTAGE_API_KEY" in st.secrets:
    os.environ["UPSTAGE_API_KEY"] = st.secrets["UPSTAGE_API_KEY"]


def download_and_unpack_chroma_db():
    file_id = "1XXyTjn8-yxa795E3k4stplJfNdFDyro2"
    url = f"https://drive.google.com/uc?id={file_id}"

    if os.path.exists("chroma_db") and os.listdir("chroma_db"):
        return

    if os.path.exists("chroma_db.zip"):
        os.remove("chroma_db.zip")

    st.write("⬇ 벡터 DB(chroma_db.zip)를 불러오는 중입니다...")
    gdown.download(url, "chroma_db.zip", quiet=False)

    if os.path.getsize("chroma_db.zip") < 1000:
        st.error("❌ chroma_db.zip 파일 오류")
        return

    with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
        zip_ref.extractall(".")


@st.cache_resource
def load_rag_chain():
    download_and_unpack_chroma_db()

    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ✅ 학생 정보(profile) + 메뉴(menu) 반영
    prompt = ChatPromptTemplate.from_template(
        """
너는 학교도서관에서 학생들의 독서활동을 도와주는 도우미야.
아래 참고 문서를 바탕으로 질문에 답해줘.

[현재 기능]
{menu}

[학생 정보]
{profile}

지침:
- '책 추천' 질문이면 학생 정보(학년/관심/읽기수준)를 반영해 추천
- 정보가 없으면 일반적인 기준으로 안내
- 문서에 없으면 모른다고 솔직히 말해

[참고 문서]
{context}

[질문]
{question}
        """
    )

    llm = ChatUpstage()

    rag_chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "profile": lambda x: x["profile"],
            "menu": lambda x: x["menu"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = load_rag_chain()


# -------------------------
# UI
# -------------------------
st.title("📚 학교도서관 독서활동 지원 챗봇")
st.caption("도서관 소장자료와 독서교육 자료를 참고하여 독서 관련 질문에 답해주는 챗봇입니다.")


with st.sidebar:
    st.subheader("📌 메뉴")

    menu = st.radio(
        "기능 선택",
        ["도서관 이용 안내", "책 추천", "독서활동"],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    # 기본값
    grade = "없음"
    interest = "없음"
    level = "없음"

    if menu == "도서관 이용 안내":
        st.markdown("**도서관 이용 방법 안내**")
        st.caption("대출·반납·연장·이용 규정 등")

    elif menu == "책 추천":
        st.markdown("**학생 프로필 기반 책 추천**")

        grade = st.selectbox("학년", ["초등", "중등", "고등"])
        interest = st.text_input("관심 주제 (예: 우정, 추리, 과학)", "")
        level = st.select_slider(
            "읽기 수준",
            options=["쉬움", "보통", "어려움"],
            value="보통"
        )

        st.caption("※ 입력할수록 추천 정확도가 높아집니다.")

    else:
        st.markdown("**독서활동 지원**")
        st.caption("📖 읽기 활동 ex) 올바른 독서법")
        st.caption("✍️ 쓰기 활동 ex) 독서감상문, 서평")
        st.caption("👥 그룹 활동 ex) 독서토론, 독서동아리")


# 채팅 히스토리
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("궁금한 것을 입력하세요.")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    profile = f"학년:{grade}, 관심:{interest or '없음'}, 읽기수준:{level}"

    # ✅ (2) 프로필을 질문 텍스트에 섞어서 retriever에도 영향
    # ✅ (3) 단, '책 추천' 탭에서만 적용
    if menu == "책 추천":
        question_for_rag = f"{user_input}\n\n[학생 정보] {profile}"
    else:
        question_for_rag = user_input

    with st.chat_message("assistant"):
        with st.spinner("생각 중입니다..."):
            answer = rag_chain.invoke({
                "question": question_for_rag,   # ✅ 여기만 변경
                "profile": profile,
                "menu": menu
            })
            st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})

