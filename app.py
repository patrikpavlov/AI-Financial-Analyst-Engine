import streamlit as st
import yfinance as yf
import finnhub
import json
from datetime import datetime, timedelta
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore

from rag_engine import ingest_news_to_pinecone

PINECONE_INDEX_NAME = "financial-reports"

st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def init_connections_and_models():
    """Initializes and caches connections and models."""
    finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])
    
    # CHANGE: Ensure embeddings are from Google, matching the index dimension
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    
    return finnhub_client, embeddings, llm

def call_analyst_model(news_context: str):
    """
    Calls the custom fine-tuned model endpoint to get a sentiment analysis.
    """
    api_url = st.secrets["HF_API_URL"]
    headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}
    
    payload = {
        "inputs": news_context,
        "parameters": {"max_new_tokens": 50}
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        try:
            # The response is a list containing one dictionary
            return response.json()[0]
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            return {"error": f"Failed to parse model response: {e}", "raw_output": response.text}
    else:
        return {"error": f"API call failed with status {response.status_code}", "details": response.text}


# --- Main App Logic ---
try:
    finnhub_client, embeddings, llm = init_connections_and_models()
except Exception as e:
    st.error(f"Failed to initialize APIs or models: {e}. Please check your secret keys.")
    st.stop()

st.title("The AI Financial Analyst Engine 📈")
ticker = st.text_input("Enter a US Stock Ticker Symbol (e.g., AAPL, GOOGL)", "NVDA").upper()

if ticker:
    st.write("---")

    today = datetime.now()
    one_week_ago = today - timedelta(days=7)
    start_date, end_date = one_week_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    news_list = finnhub_client.company_news(ticker, _from=start_date, to=end_date)[:10]

    tab1, tab2, tab3 = st.tabs(["📊 Price & News Headlines", "🤖 Chat with Recent News","Fine-Tuned Sentiment Analysis"])


    # --- Tab 1: Price, Info, and News Headlines ---
    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader(f"Historical Price Chart for {ticker}")
            stock_data = yf.Ticker(ticker).history(period="5y")
            st.line_chart(stock_data['Close'])

        with col2:
            st.subheader(f"Recent News Headlines for {ticker}")
            if not news_list:
                st.info("No recent news found.")
            else:
                for item in news_list:
                    st.markdown(f"**[{item['headline']}]({item['url']})**")
                    st.write(f"_{datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d')}_ - {item['source']}")
                    st.divider()

    # --- Tab 2: RAG Engine for News ---
    with tab2:
        st.header(f"Analyze Recent News for {ticker}")
        st.write("This tool lets you chat with the 10 most recent news articles.")

        if st.button("Analyze and Index News"):
            with st.spinner("Processing news, creating embeddings, and indexing in Pinecone..."):
                ingest_news_to_pinecone(news_list, ticker, PINECONE_INDEX_NAME)

        st.subheader("Ask a Question About the News")

        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=ticker
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        query = st.text_input("Your Question:", placeholder="e.g., What were the key points from the latest earnings call news?")


        if query:
                st.write("➡️ Step 1: Retrieving relevant news chunks from Pinecone...")
                retriever = vectorstore.as_retriever()
                retrieved_docs = retriever.invoke(query,k=5)
                with st.expander("See documents retrieved from vector store"):
                    st.json([doc.to_json() for doc in retrieved_docs])

                # --- Step 2: Combine documents and create a prompt for the LLM ---
                st.write("📝 Step 2: Combining documents and creating a prompt for the LLM...")
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                prompt_template = f"""
                You are a helpful financial analyst assistant. Answer the following question based ONLY on the context provided below.
                If the answer is not in the context, say "I don't have enough information from the news articles to answer that."

                CONTEXT:
                {context}

                QUESTION:
                {query}

                ANSWER:
                """
                with st.expander("See the final prompt sent to the LLM"):
                    st.text(prompt_template)

                # --- Step 3: Call the LLM to get the final answer ---
                st.write("🤖 Step 3: Getting final answer from Google's Gemini model...")
                try:
                    # The response from the LLM is an AIMessage object
                    response = llm.invoke(prompt_template)
                    st.success("**Answer:**")
                    # We need to access the 'content' attribute of the response
                    st.write(response.content)
                except Exception as e:
                    st.error(f"An error occurred during question answering: {e}")
    with tab3:
        st.header(f"Generate Analyst Briefing for {ticker}")
        st.write("This feature uses a custom-trained Llama 3 model to provide a sentiment analysis based on the latest news headlines.")

        if not news_list:
            st.info("No news headlines available to analyze.")
        else:
            if st.button("✨ Generate Analyst Briefing"):
                with st.spinner("Calling custom model... Please wait, this can take a moment."):
                    # Combine headlines into a single string for analysis
                    news_context = "\n".join([item['headline'] for item in news_list])
                    
                    # Call the model
                    result = call_analyst_model(news_context)
                    
                    if "error" in result:
                        st.error(f"Could not generate briefing: {result['error']}")
                        st.json(result)
                    else:
                        st.subheader("AI-Generated Sentiment")
                        sentiment = result.get("sentiment", "N/A")
                        
                        if sentiment == "Positive":
                            st.success(f"**{sentiment}** 👍")
                        elif sentiment == "Negative":
                            st.error(f"**{sentiment}** 👎")
                        else:
                            st.warning(f"**{sentiment}** 😐")
                        
                        st.write("---")
                        st.write("**News Headlines Analyzed:**")
                        st.text(news_context)
