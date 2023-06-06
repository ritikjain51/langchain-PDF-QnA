import base64

import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header

from backend import QnASystem
from schema import TransformType, EmbeddingTypes, IndexerType, BotType

kwargs = {}
source_docs = []
st.set_page_config(page_title="PDFChat - An LLM-powered experimentation app")

if "qna_system" not in st.session_state:
    st.session_state.qna_system = QnASystem()


def show_pdf(f):
    f.seek(0)
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" ' \
                  f'type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def model_settings():
    kwargs["temperature"] = st.slider("Temperature", max_value=1.0, min_value=0.0)
    kwargs["max_tokens"] = st.number_input("Max Token", min_value=0, value=512)


st.title("PDF Question and Answering")

tab1, tab2, tab3 = st.tabs(["Upload and Ingest PDF", "Ask", "Show PDF"])

with st.sidebar:
    st.header("Advance Setting ⚙️")
    require_pdf = st.checkbox("Show PDF", value=1)
    st.markdown('---')
    kwargs["bot_type"] = st.selectbox("Bot Type", options=BotType)
    st.markdown("---")
    st.text("Model Parameters")
    kwargs["return_documents"] = st.checkbox("Require Source Documents", value=True)
    text_transform = st.selectbox("Text Transformer", options=TransformType)
    st.markdown("---")
    selected_model = st.selectbox("Select Model", options=EmbeddingTypes)
    match selected_model:
        case EmbeddingTypes.OPENAI:
            api_key = st.text_input("OpenAI API Key", placeholder="sk-...", type="password")
            if not api_key.startswith('sk-'):
                st.warning('Please enter your OpenAI API key!', icon='⚠')
            model_settings()
        case EmbeddingTypes.HUGGING_FACE:
            api_key = st.text_input("Hugging Face API Key", placeholder="hg-...", type="password")
            if not api_key.startswith('hg-'):
                st.warning('Please enter your HuggingFace API key!', icon='⚠')
            huggingface_model = st.selectbox("Choose Model", options=["google/flan-t5-xl"])
            model_settings()
        case EmbeddingTypes.COHERE:
            api_key = st.text_input("Cohere API Key", placeholder="...", type="password")
            if not api_key:
                st.warning('Please enter your Cohere API key!', icon='⚠')
            model_settings()
        case _:
            api_key = None
    kwargs["api_key"] = api_key
    st.markdown("---")

    vector_indexer = st.selectbox("Vector Indexer", options=IndexerType)
    match vector_indexer:
        case IndexerType.ELASTICSEARCH:
            kwargs["elasticsearch_url"] = st.text_input("Elastic Search URL: ")
            if not kwargs.get("elasticsearch_url"):
                st.warning("Please enter your elastic search url", icon='⚠')
            kwargs["elasticsearch_index"] = st.text_input("Elastic Search Index: ")
            if not kwargs.get("elasticsearch_index"):
                st.warning("Please enter your elastic search index", icon='⚠')

    st.markdown("---")
    st.text("Chain Settings")
    kwargs["chain_type"] = st.selectbox("Chain Type", options=["stuff", "map_reduce"])
    kwargs["search_type"] = st.selectbox("Search Type", options=["similarity"])
    st.markdown("---")

with tab1:
    uploaded_file = st.file_uploader("Upload and Ingest PDF 🚀", type="pdf")
    if uploaded_file:
        with st.spinner("Uploading and Ingesting"):
            documents = st.session_state.qna_system.read_and_load_pdf(uploaded_file)
            if selected_model == EmbeddingTypes.NA:
                st.warning("Please select the model", icon='⚠')
            else:
                st.session_state.qna_system.build_chain(transform_type=text_transform, embedding_type=selected_model,
                                                        indexer_type=vector_indexer, **kwargs)


def generate_response(prompt):
    if prompt and uploaded_file:
        response = st.session_state.qna_system.ask_question(prompt)
        return response.get("answer", response.get("result", "")), response.get("source_documents")
    return "", []


with tab2:
    if not uploaded_file:
        st.warning("Please upload PDF", icon='⚠')
    else:
        match kwargs["bot_type"]:
            case BotType.qna:
                with st.container():
                    with st.form('my_form'):
                        text = st.text_area("", placeholder='Ask me...')
                        submitted = st.form_submit_button('Submit')
                        if text:
                            st.write(f"Question:\n{text}")
                            response, source_docs = generate_response(text)
                            st.write(response)
            case BotType.conversational:
                # Generate empty lists for generated and past.
                ## generated stores AI generated responses
                if 'generated' not in st.session_state:
                    st.session_state['generated'] = ["Hi! I'm PDF Assistant 🤖, How may I help you?"]
                ## past stores User's questions
                if 'past' not in st.session_state:
                    st.session_state['past'] = ['Hi!']

                input_container = st.container()
                colored_header(label='', description='', color_name='blue-30')
                response_container = st.container()
                response = ""


                def get_text():
                    input_text = st.text_input("You: ", "", key="input")
                    return input_text


                with input_container:
                    user_input = get_text()
                    if st.button("Clear"):
                        st.session_state.generated.clear()
                        st.session_state.past.clear()

                with response_container:
                    if user_input:
                        response, source_docs = generate_response(user_input)
                        st.session_state.past.append(user_input)
                        st.session_state.generated.append(response)

                    if st.session_state['generated']:
                        for i in range(len(st.session_state['generated'])):
                            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                            message(st.session_state["generated"][i], key=str(i))

        require_document = st.container()
        if kwargs["return_documents"]:
            with require_document:
                with st.expander("Related Documents", expanded=False):
                    for source in source_docs:
                        metadata = source.metadata
                        st.write("{source} - {page_no}".format(source=metadata.get("source"),
                                                               page_no=metadata.get("page_no")))
                        st.write(source.page_content)
                        st.markdown("---")

with tab3:
    if require_pdf and uploaded_file:
        show_pdf(uploaded_file)
    elif uploaded_file:
        st.warning("Feature not enabled.", icon='⚠')
    else:
        st.warning("Please upload PDF", icon='⚠')
