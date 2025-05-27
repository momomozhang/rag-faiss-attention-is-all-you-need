import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
pdf_path = "/Users/zhangmengni/Desktop/coding/rag-faiss-attention-is-all-you-need/NIPS-2017-attention-is-all-you-need-Paper.pdf"

def format_docs(docs):
    """
    Formats a list of document objects into a single string, separating each document's content with two newlines.

    Args:
        docs (list): A list of document objects, each expected to have a 'page_content' attribute.

    Returns:
        str: A single string containing the concatenated 'page_content' of all documents, separated by two newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("hi")
    loader = PyPDFLoader(file_path=pdf_path)
    doc = loader.load()
    text_splitter = CharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=30,
                        separator="\n",
                        )
    doc_chunked = text_splitter.split_documents(documents=doc)
    
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    vectorstore = FAISS.from_documents(
                    documents=doc_chunked,
                    embedding=embeddings,
                    )
    
    vectorstore.save_local("faiss_index_attention")
    
    new_vectorstore = FAISS.load_local(
        "faiss_index_attention",
        embeddings,
        allow_dangerous_deserialization=True,
        )
    
    template = """use the following peices of contet to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up answer.
    Keep the answer as concise as possible.
    
    {context}
    
    Question: {question}
    
    Helpful Answer: 
    
    """
    
    custom_rag_prompt = PromptTemplate.from_template(template)
    
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4.1-nano",
        temperature=0.2,
        )
    
    rag_chain = (
        {"question": RunnablePassthrough(), "context": new_vectorstore.as_retriever() | format_docs}
        | custom_rag_prompt
        | llm
    )
    
    query = "What are the tasks are best suited for Transformers?"
    
    rag_result = rag_chain.invoke(query)
    
    print(rag_result.content)