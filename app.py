"""
University-FineTuned-Assistant Web Interface.
Streamlit application for interactive comparison of Base Model,
Fine-Tuned LoRA Assistant, and RAG Lecture Assistant.
"""

import os
import streamlit as st
from src.inference.model_loader import UniversityAssistant
from src.rag.retriever import LectureRetriever
from src.training.utils import get_hardware_info

# Page Configuration
st.set_page_config(
    page_title="Apex National University | AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .chat-bubble-user {
        background-color: #EFF6FF;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .chat-bubble-assistant {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .source-box {
        font-size: 0.85rem;
        color: #4B5563;
        background-color: #F3F4F6;
        border-left: 3px solid #3B82F6;
        padding: 8px 12px;
        margin-top: 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_assistant(model_type: str):
    return UniversityAssistant(model_type=model_type)


@st.cache_resource(show_spinner=False)
def get_retriever():
    return LectureRetriever()


def main():
    # Sidebar
    with st.sidebar:
        st.title("🎓 ANU Assistant")
        st.markdown("---")
        
        # Model Configuration
        st.subheader("⚙️ Model Configuration")
        model_choice = st.radio(
            "Select Operating Mode:",
            options=["Fine-Tuned LoRA Model", "Base Model (Zero-Shot)", "RAG Lecture Assistant"],
            index=0
        )

        st.markdown("---")
        st.subheader("🎛️ Inference Settings")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        max_tokens = st.slider("Max New Tokens", min_value=50, max_value=500, value=200, step=25)

        st.markdown("---")
        st.subheader("💻 System Information")
        hw = get_hardware_info()
        st.caption(f"**Device:** {hw['device'].upper()} ({hw['device_name']})")
        if hw['cuda_available']:
            st.caption(f"**VRAM:** {hw['vram_gb']} GB")
        st.caption("**Base Model:** `Qwen/Qwen2.5-0.5B-Instruct`")
        adapter_status = "✅ Available" if os.path.exists("models/university-lora") else "⚠️ Not Found (Run train.py)"
        st.caption(f"**LoRA Adapter:** {adapter_status}")

        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main Area Header
    st.markdown('<div class="main-header">Apex National University Information Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Answering queries on degree programs, campus facilities, academic policies, and course lectures.</div>', unsafe_allow_html=True)

    # Example Prompts
    st.markdown("##### 💡 Example Questions")
    col1, col2, col3, col4 = st.columns(4)
    example_query = None

    if col1.button("Which faculty offers Software Engineering?", use_container_width=True):
        example_query = "Which faculty offers Software Engineering?"
    if col2.button("What are Ada Lovelace Library hours?", use_container_width=True):
        example_query = "What is the purpose and operating hours of the Ada Lovelace Memorial Library?"
    if col3.button("Explain SOLID principles (SE204)", use_container_width=True):
        example_query = "Explain the SOLID design principles covered in SE204."
    if col4.button("What is the attendance policy?", use_container_width=True):
        example_query = "What minimum lecture attendance is required to sit for exams?"

    # Session State for Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                st.markdown("**Referenced Sources:**")
                for src in set(msg["sources"]):
                    st.markdown(f"- `{src}`")

    # Handle input (either from chat box or example buttons)
    user_query = st.chat_input("Ask a question about the university or lecture notes...")
    prompt_to_process = example_query or user_query

    if prompt_to_process:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    if model_choice == "RAG Lecture Assistant":
                        retriever = get_retriever()
                        assistant = get_assistant("lora")
                        retrieved_docs = retriever.retrieve(prompt_to_process, top_k=3)

                        if retrieved_docs:
                            context_str = "\n\n".join([
                                f"[Source: {d['source']} | Course: {d['course']}]\n{d['text']}"
                                for d in retrieved_docs
                            ])
                            sys_prompt = (
                                "You are the official University Academic & Lecture Assistant for Apex National University. "
                                "Use the provided lecture notes and course excerpts below to answer the student's question accurately.\n\n"
                                f"--- RELEVANT LECTURE EXCERPTS ---\n{context_str}\n----------------------------------"
                            )
                            aug_query = f"Question: {prompt_to_process}\n\nPlease provide a clear answer based on the lecture material."
                            response = assistant.generate_response(
                                user_message=aug_query,
                                system_prompt=sys_prompt,
                                max_new_tokens=max_tokens,
                                temperature=temperature
                            )
                            sources = [f"{d['course']} ({d['source']})" for d in retrieved_docs]
                        else:
                            response = assistant.generate_response(
                                user_message=prompt_to_process,
                                max_new_tokens=max_tokens,
                                temperature=temperature
                            )
                            sources = []
                    else:
                        m_type = "lora" if "Fine-Tuned" in model_choice else "base"
                        assistant = get_assistant(m_type)
                        response = assistant.generate_response(
                            user_message=prompt_to_process,
                            max_new_tokens=max_tokens,
                            temperature=temperature
                        )
                        sources = []

                    st.markdown(response)
                    if sources:
                        st.markdown("**Referenced Sources:**")
                        for src in set(sources):
                            st.markdown(f"- `{src}`")

                    # Save Assistant Message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error during inference: {e}")


if __name__ == "__main__":
    main()
