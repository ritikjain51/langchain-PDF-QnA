import os

from langchain import FAISS, OpenAI, HuggingFaceHub, Cohere, PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, CohereEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, NLTKTextSplitter, \
    SpacyTextSplitter
from langchain.vectorstores import Chroma, ElasticVectorSearch
from pypdf import PdfReader

from schema import EmbeddingTypes, IndexerType, TransformType, BotType


class QnASystem:

    def read_and_load_pdf(self, f_data):
        pdf_data = PdfReader(f_data)
        documents = []
        for idx, page in enumerate(pdf_data.pages):
            documents.append(Document(page_content=page.extract_text(),
                                      metadata={"page_no": idx, "source": f_data.name}))

        self.documents = documents

    def document_transformer(self, transform_type: TransformType):
        match transform_type:
            case TransformType.CharacterTransform:
                t_type = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            case TransformType.RecursiveTransform:
                t_type = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            case TransformType.NLTKTransform:
                t_type = NLTKTextSplitter()
            case TransformType.SpacyTransform:
                t_type = SpacyTextSplitter()

            case _:
                raise IndexError("Invalid Transformer Type")

        self.transformed_documents = t_type.split_documents(documents=self.documents)

    def generate_embeddings(self, embedding_type: EmbeddingTypes = EmbeddingTypes.OPENAI,
                            indexer_type: IndexerType = IndexerType.FAISS, **kwargs):
        temperature = kwargs.get("temperature", 0)
        max_tokens = kwargs.get("max_tokens", 512)
        match embedding_type:
            case EmbeddingTypes.OPENAI:
                os.environ["OPENAI_API_KEY"] = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
                embeddings = OpenAIEmbeddings()
                llm = OpenAI(temperature=temperature, max_tokens=max_tokens)
            case EmbeddingTypes.HUGGING_FACE:
                embeddings = HuggingFaceEmbeddings(model_name=kwargs.get("model_name"))
                llm = HuggingFaceHub(repo_id=kwargs.get("model_name"),
                                     model_kwargs={"temperature": temperature, "max_tokens": max_tokens})
            case EmbeddingTypes.COHERE:
                embeddings = CohereEmbeddings(model=kwargs.get("model_name"), cohere_api_key=kwargs.get("api_key"))
                llm = Cohere(model=kwargs.get("model_name"), cohere_api_key=kwargs.get("api_key"),
                             model_kwargs={"temperature": temperature,
                                           "max_tokens": max_tokens})
            case _:
                raise IndexError("Invalid Embedding Type")

        match indexer_type:
            case IndexerType.FAISS:
                indexer = FAISS
            case IndexerType.CHROMA:
                indexer = Chroma()

            case IndexerType.ELASTICSEARCH:
                indexer = ElasticVectorSearch(elasticsearch_url=kwargs.get("elasticsearch_url"))
            case _:
                raise IndexError("Invalid Indexer Function")

        self.llm = llm
        self.indexer = indexer
        self.vector_store = indexer.from_documents(documents=self.transformed_documents, embedding=embeddings)

    def get_retriever(self, search_type="similarity", top_k=5, **kwargs):
        retriever = self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": top_k})
        self.retriever = retriever

    def get_prompt(self, bot_type: BotType, **kwargs):
        match bot_type:
            case BotType.qna:
                prompt = """
                You are a smart and helpful AI assistant, who answer the question given context
                {context}
                Question: {question}
                """
            case BotType.conversational:
                prompt = """
                Given the following conversation and a follow up question, 
                rephrase the follow up question to be a standalone question, in its original language.
                \nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:
                """
        return PromptTemplate(input_variables=["context", "question", "chat_history"], template=prompt)

    def build_qa(self, qa_type: BotType, chain_type="stuff",
                 return_documents: bool = True, **kwargs):
        match qa_type:
            case BotType.qna:
                self.chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever, chain_type=chain_type,
                                                         return_source_documents=return_documents, verbose=True)

            case BotType.conversational:
                self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,
                                                       output_key="answer")
                self.chain = ConversationalRetrievalChain.from_llm(llm=self.llm, retriever=self.retriever,
                                                                   chain_type=chain_type,
                                                                   return_source_documents=return_documents,
                                                                   memory=self.memory, verbose=True)

            case _:
                raise IndexError("Invalid QA Type")

    def ask_question(self, query):
        if type(self.chain) == RetrievalQA:
            data = {"query": query}
        else:
            data = {"question": query}
        return self.chain(data)

    def build_chain(self, transform_type, embedding_type, indexer_type, **kwargs):
        if hasattr(self, "llm"):
            return self.chain
        self.document_transformer(transform_type)
        self.generate_embeddings(embedding_type=embedding_type,
                                 indexer_type=indexer_type, **kwargs)
        self.get_retriever(**kwargs)
        qa = self.build_qa(qa_type=kwargs.get("bot_type"), **kwargs)
        return qa


if __name__ == "__main__":
    qna = QnASystem()
    with open("../docs/Doc A.pdf", "rb") as f:
        qna.read_and_load_pdf(f)
        chain = qna.build_chain(
            transform_type=TransformType.RecursiveTransform,
            embedding_type=EmbeddingTypes.OPENAI, indexer_type=IndexerType.FAISS,
            chain_type="map_reduce", bot_type=BotType.conversational, return_documents=True
        )
        question = qna.ask_question(query="Hi! Summarize the document.")
        question = qna.ask_question(query="What happened from June 1984 to September 1996")
        print(question)
