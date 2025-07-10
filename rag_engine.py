import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
# CHANGE: Use HuggingFace embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

def ingest_news_to_pinecone(news_articles: list, ticker: str, index_name: str):
    """
    Takes a list of news articles, chunks them, and stores them in Pinecone.
    """
    if not news_articles:
        st.warning("No news articles provided to ingest.")
        return

    st.info(f"Preparing to index {len(news_articles)} news articles for {ticker}...")

    documents = []
    for article in news_articles:
        combined_content = f"Headline: {article['headline']}\n\nSummary: {article['summary']}"
        doc = Document(
            page_content=combined_content,
            metadata={
                "headline": article['headline'],
                "url": article['url'],
                "source": article['source']
            }
        )
        documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunked_docs = text_splitter.split_documents(documents)
    st.write(f"Split {len(documents)} articles into {len(chunked_docs)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    try:
        st.write("Connecting to Pinecone and indexing documents...")
        PineconeVectorStore.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            index_name=index_name,
            namespace=ticker
        )
        st.success(f"Successfully indexed news for {ticker} in Pinecone!")
        st.balloons()
    except Exception as e:
        st.error(f"Failed to ingest data into Pinecone: {e}")