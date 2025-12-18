import os
import zipfile
import gdown

import streamlit as st

from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ✅ PDF 리포트용 import (추가)
from io import BytesIO
from datetime import datetime
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ✅ set_page_config는 가능한 한 위에서 1번만
st.set_page_config(page_title="학교도서관 독서활동 지원 챗봇", page_icon="📚")


# 🔑 API KEY
if "UPSTAGE_API_KEY" in st.secrets:
    os.environ["UPSTAGE_API_KEY"] = st.secrets["UPSTAGE_API_KEY"]


# -------------------------
# PROMPT
# -------------------------
BASE_PROMPT = """
너는 학교도서관에서 학생들의 독서활동을 도와주는 도우미야.
아래 '참고 문서(context)' 내용을 바탕으로, 학생의 질문에 대해
친절하고 구체적인 답변을 한국어로 작성해줘.

가능하면:
- 도서관 이용 규정, 대출/반납/연장 방법
- 책 고르는 방법, 독후감 작성법, 독서 토론 방법
등을 중심으로 설명해 줘.

만약 문서에 정보가 없으면 모르는 부분은 솔직하게 모른다고 말해.
""".strip()

MODE_PROMPT = {
    "도서관 이용 안내": """
[도서관 이용 안내 모드 지침]
- 도서관 이용 규정, 대출/반납/연장, 운영시간, 검색/신청 방법을 우선적으로 안내해줘.
- 절차가 있으면 단계별로 설명해줘.
""".strip(),
    "책 추천": """
[책 추천 모드 지침]
- 학생 정보(학년/관심 주제/읽기 수준)를 반영해서 추천해줘.
- 가능하면 추천 이유(왜 이 책이 맞는지)를 짧게 덧붙여줘.
- 문서 근거가 부족하면 일반적인 추천 기준으로 안내해줘.
""".strip(),
    "독서활동": """
[독서활동 모드 지침]
- 읽기/쓰기/토론 중 어떤 활동인지 구분해서 안내해줘.
- 학생이 바로 써먹을 수 있는 문장 예시나 질문 예시를 제시해줘.
""".strip(),
}


# -------------------------
# PDF: 대화 리포트 생성 함수 (✅ 호출 전에 반드시 정의돼야 함)
# -------------------------
def build_chat_pdf(messages, title="학교도서관 독서활동 지원 챗봇 리포트", meta=None):
    """
    messages: [{"role":"user"/"assistant", "content":"..."}]
    meta: {"menu":..., "profile":...} 같은 부가정보
    """
    buf = BytesIO()

    # ✅ 한글 폰트 등록 (레포에 fonts/NotoSansKR-Regular.ttf 필요)
    font_path = "fonts/NotoSansKR-Regular.ttf"
    pdfmetrics.registerFont(TTFont("NotoSansKR", font_path))

    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    left = 40
    top = height - 50
    y = top

    c.setFont("NotoSansKR", 16)
    c.drawString(left, y, title)
    y -= 24

    c.setFont("NotoSansKR", 10)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.drawString(left, y, f"생성 시각: {now_str}")
    y -= 16

    if meta:
        if meta.get("menu"):
            c.drawString(left, y, f"탭: {meta['menu']}")
            y -= 14
        if meta.get("profile"):
            c.drawString(left, y, f"학생 정보: {meta['profile']}")
            y -= 14

    y -= 6
    c.line(left, y, width - left, y)
    y -= 18

    def wrap_lines(text, max_chars=65):
        lines = []
        for paragraph in str(text).split("\n"):
            while len(paragraph) > max_chars:
                lines.append(paragraph[:max_chars])
                paragraph = paragraph[max_chars:]
            lines.append(paragraph)
        return lines

    c.setFont("NotoSansKR", 11)

    for m in messages:
        role = "학생" if m.get("role") == "user" else "챗봇"
        lines = [f"[{role}]"] + wrap_lines(m.get("content", "")) + [""]

        for line in lines:
            if y < 60:
                c.showPage()
                c.setFont("NotoSansKR", 11)
                y = top
            c.drawString(left, y, line)
            y -= 14

    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# -------------------------
# DB 다운로드/언팩
# -------------------------
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

    try:
        with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
            zip_ref.extractall(".")
    except zipfile.BadZipFile:
        st.error("❌ ZIP 파일을 열 수 없습니다. chroma_db.zip 파일을 확인해 주세요.")
        return


@st.cache_resource
def load_rag_chain():
    download_and_unpack_chroma_db()

    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ✅ BASE_PROMPT 유지 + mode_guide/메뉴/학생정보만 추가로 붙임 (기존 의도 유지)
    prompt = ChatPromptTemplate.from_template(
        BASE_PROMPT
        + """
{mode_guide}

[현재 기능]
{menu}

[학생 정보]
{profile}

[참고 문서]
{context}

[학생의 질문]
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
            "mode_guide": lambda x: x["mode_guide"],
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
    interest = ""
    level = "없음"

    if menu == "도서관 이용 안내":
        st.markdown("**ℹ️ 도서관 이용 방법**")
        st.caption("예: 대출/반납 방법, 이용시간, 도서 검색 안내 등")

        st.write("")

        st.markdown("**예시 질문**")
        st.caption("• 도서관 이용규칙에 대해 알려줘.")
        st.caption("• 대출 권수와 기간이 어떻게 돼?")
        st.caption("• 신간도서 신청 방법이 궁금해.")

    elif menu == "책 추천":
        st.markdown("**🎯 맞춤형 도서 추천**")

        grade = st.selectbox("학년", ["초등", "중등", "고등"])
        interest = st.text_input("관심 주제 (예: 우정, 인권, 경제)", "")
        level = st.select_slider(
            "읽기 수준",
            options=["쉬움", "보통", "어려움"],
            value="보통"
        )

        st.caption("※ 입력할수록 추천 정확도가 높아집니다.")

    else:  # 독서활동
        st.markdown("**독서활동 관련 도움을 드려요**")

        # 회색 글씨로 연하게
        st.caption("📖 읽기 활동 ex) 올바른 독서법")
        st.caption("✍️ 쓰기 활동 ex) 독서감상문, 서평, 독서논술 등")
        st.caption("👥 그룹 활동 ex) 독서토론, 독서동아리 등")

        st.write("")

        st.markdown("**예시 질문**")
        st.caption("• 독후감 서론을 어떻게 시작하면 좋을까?")
        st.caption("• 독서토론 질문을 잘 만드는 방법은?")
        st.caption("• 서평과 독후감 차이가 뭐야?")

    # ✅ (요청대로) 사이드바 맨 아래: PDF 다운로드 버튼
    st.divider()
    st.subheader("📄 리포트")

    # 프로필 문자열(기존 너가 쓰던 형식 유지)
    profile_for_pdf = f"학년:{grade}, 관심:{interest or '없음'}, 읽기수준:{level}"

    pdf_data = build_chat_pdf(
        st.session_state.get("messages", []),
        meta={"menu": menu, "profile": profile_for_pdf}
    )

    st.download_button(
        label="대화 리포트 PDF 다운로드",
        data=pdf_data,
        file_name="chat_report.pdf",
        mime="application/pdf",
    )


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

    # ✅ (2)+(3) 반영: 책 추천일 때만 question에 프로필을 붙여 retriever에도 영향 주기 (기존 로직 유지)
    if menu == "책 추천":
        question_for_rag = f"{user_input}\n\n[학생 정보] {profile}"
    else:
        question_for_rag = user_input

    with st.chat_message("assistant"):
        with st.spinner("생각 중입니다."):
            answer = rag_chain.invoke({
                "question": question_for_rag,
                "profile": profile,
                "menu": menu,
                "mode_guide": MODE_PROMPT.get(menu, "")
            })
            st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
