import logging
import os

import gradio as gr
from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

documents = []
qa = None


def get_file(file):
    try:
        global documents
        data = PyPDFLoader(file.name)
        documents = data.load_and_split(CharacterTextSplitter(chunk_size=2000, chunk_overlap=0))
    except Exception as e:
        logging.error(e, exc_info=True)
        return "Failed to upload."
    return "File Uploaded."


def model_configuration(model_name, api_key=None, hug_model=None, hug_token=None):
    try:
        embeddings, llm = None, None
        if not documents:
            return gr.update(value="Please upload correct PDF!", visible=True)
        global qa
        if model_name == "OpenAI":
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", api_key)
            embeddings = OpenAIEmbeddings()
            llm = OpenAI(temperature=1)
        elif model_name == "HuggingFace":
            return gr.update(value="Hugging Face is not yet supported!", visible=True)
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("", hug_token)
            # embeddings = HuggingFaceEmbeddings(model_name=hug_model, model_kwargs={'device': 'cpu'})
            # llm = HuggingFaceHub(repo_id=hug_model)

        if embeddings:
            db = Chroma.from_documents(documents, embeddings)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            qa = ConversationalRetrievalChain.from_llm(llm, chain_type="map_reduce", retriever=retriever,
                                                       return_source_documents=True, verbose=False)
    except Exception as e:
        logging.error(e, exc_info=True)
        return gr.update(value="Error occurred!", visible=True)
    return gr.update(value="Model Built", visible=True)


def response(msg, chat_history):
    global qa
    result = qa({"question": msg, "chat_history": map(tuple, chat_history)})
    final_resp = result.get("answer", "").strip()
    chat_history.append((msg, final_resp))
    return "", chat_history


with gr.Blocks() as demo:
    with gr.Tab("PDF Ingestion") as pdf_input:
        file = None
        with gr.Column() as r1:
            file = gr.File(file_types=[".pdf"])
            op_txt = gr.Label(value="", label="")
            fl_btn = gr.Button("Upload & Ingest 🚀")
            fl_btn.click(get_file, inputs=[file], outputs=op_txt)

    with gr.Tab("Select Model") as model:
        model_name = gr.Dropdown(
            ["NA", "OpenAI", "HuggingFace"],
            show_label=True,
            label="Model Name",
            multiselect=False,
            value="NA"
        )
        with gr.Column(visible=False) as openai_config:
            api_key = gr.Textbox(value="", label="OPENAI API KEY", placeholder="sk-...", visible=True, interactive=True)

        with gr.Column(visible=False) as huggy_config:
            hug_model = gr.Dropdown(["distilbert-base-uncased"],
                                    value="distilbert-base-uncased", multiselect=False)
            hug_token = gr.Textbox(value="", placeholder="hf-...", interactive=True)


        def show_configuration(model_name):
            if model_name == "OpenAI":
                return {
                    openai_config: gr.update(visible=True),
                    huggy_config: gr.update(visible=False)
                }
            elif model_name == "HuggingFace":
                return {
                    openai_config: gr.update(visible=False),
                    huggy_config: gr.update(visible=True)
                }
            return {
                openai_config: gr.update(visible=False),
                huggy_config: gr.update(visible=False)
            }


        model_name.change(show_configuration, inputs=[model_name], outputs=[openai_config, huggy_config])
        model_updated = gr.Label("", visible=False)
        btn = gr.Button("Configure Model 🤖")
        btn.click(model_configuration, inputs=[model_name, api_key, hug_model, hug_token], outputs=model_updated)

    with gr.Tab("Q&A") as qna:
        with gr.Column() as r:
            chatbot = gr.Chatbot(show_label=True)
            msg = gr.Textbox(placeholder="Ask Something")
            clear = gr.Button("Clear")
            msg.submit(response, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
