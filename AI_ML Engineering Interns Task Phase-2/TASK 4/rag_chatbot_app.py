"""
RAG Chatbot - Context-Aware with Memory
========================================
Run:  streamlit run rag_chatbot_app.py
Free - no API key needed, uses built-in AI via the app

API is not integerated

Requirements:
    pip install streamlit scikit-learn numpy anthropic
"""

import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────
# KNOWLEDGE BASE
# 13 documents about AI/ML topics
# In a real project this would be your own PDFs or docs
# ─────────────────────────────────────────────────────
DOCS = [
    {
        "id": "ai_001",
        "title": "What is Artificial Intelligence?",
        "category": "AI Basics",
        "content": """Artificial Intelligence (AI) refers to computer systems that can perform tasks
        that normally require human intelligence. These tasks include learning, reasoning,
        problem-solving, understanding language, and recognising patterns. AI is not a single
        technology but a broad field that includes many approaches and techniques.
        Modern AI systems are powered by machine learning, which allows computers to learn
        from data rather than being explicitly programmed for every situation. AI is divided
        into narrow AI which is designed for specific tasks like playing chess or recognising
        faces, and general AI which would be capable of performing any intellectual task a
        human can do. Currently, all practical AI systems are narrow AI."""
    },
    {
        "id": "ml_001",
        "title": "How Machine Learning Works",
        "category": "Machine Learning",
        "content": """Machine learning is a branch of AI where systems learn patterns from data.
        Instead of a programmer writing rules manually, the machine figures out the rules
        itself by studying examples. There are three main types: supervised learning which
        involves learning from labelled examples like training a spam filter on emails marked
        spam or not spam, unsupervised learning which finds hidden patterns in unlabelled data
        like customer segmentation, and reinforcement learning which learns by trial and error
        with rewards and penalties like training a game-playing agent. Machine learning models
        include decision trees, neural networks, support vector machines, and random forests.
        The quality of the data used for training is the single most important factor."""
    },
    {
        "id": "dl_001",
        "title": "Deep Learning and Neural Networks",
        "category": "Deep Learning",
        "content": """Deep learning is a type of machine learning that uses neural networks with
        many layers. These layers process data hierarchically where each layer learns more
        abstract features from the one before it. A neural network has an input layer, hidden
        layers, and an output layer. Each node in the network is connected to adjacent nodes
        with adjustable weights. Training involves feeding data through the network, calculating
        the error in predictions, and using backpropagation to adjust weights to reduce that
        error. Deep learning powers most modern AI breakthroughs including image recognition,
        natural language processing, and generative AI. The downside is that it requires large
        amounts of data and significant computing power to train."""
    },
    {
        "id": "llm_001",
        "title": "Large Language Models Explained",
        "category": "LLMs",
        "content": """Large Language Models are a type of deep learning model trained on massive
        amounts of text data. They learn statistical patterns of language and can generate
        human-like text, answer questions, translate languages, and write code. LLMs use a
        transformer architecture which processes all words in a sentence simultaneously using
        attention mechanisms rather than one word at a time. The most well-known LLMs include
        GPT-4, Claude, Gemini, and Llama. LLMs work by predicting the next most likely token
        given previous tokens. They are fine-tuned using RLHF which stands for Reinforcement
        Learning from Human Feedback to align their outputs with human preferences. A key
        limitation is that LLMs can hallucinate which means generating confident-sounding but
        incorrect information."""
    },
    {
        "id": "rag_001",
        "title": "Retrieval-Augmented Generation (RAG)",
        "category": "RAG",
        "content": """RAG is a technique that improves LLM responses by retrieving relevant
        documents before generating an answer. Instead of relying solely on knowledge baked
        into the model during training, a RAG system searches a document store for relevant
        content and gives it to the LLM as context. This solves two key problems with plain
        LLMs: they cannot access up-to-date information, and they sometimes hallucinate facts.
        The RAG pipeline works in steps. Documents are split into chunks and converted to
        vector embeddings. These vectors are stored in a vector database. When a user asks a
        question, the question is also converted to an embedding. The most similar document
        chunks are retrieved. The LLM generates an answer using the retrieved chunks as context.
        RAG is widely used in enterprise chatbots, customer support systems, and document Q&A."""
    },
    {
        "id": "emb_001",
        "title": "Vector Embeddings Explained",
        "category": "Embeddings",
        "content": """A vector embedding is a way of representing text as a list of numbers, a
        vector in high-dimensional space. The key property is that semantically similar content
        gets similar vectors. For example, the embeddings for car and automobile would be very
        close together while car and pizza would be far apart. Embeddings are created by neural
        networks trained on large datasets. Models like Word2Vec, GloVe, and sentence
        transformers generate these representations. In a RAG system, you convert both your
        documents and the user question into embeddings, then find documents whose embeddings
        are closest to the question embedding. These are the most semantically relevant chunks.
        Cosine similarity is the most common distance metric used to compare embeddings."""
    },
    {
        "id": "lc_001",
        "title": "LangChain Framework",
        "category": "LangChain",
        "content": """LangChain is a Python framework for building applications with Large Language
        Models. It provides ready-made components for common LLM tasks such as connecting to
        different LLM providers like OpenAI and Anthropic, building retrieval pipelines,
        managing conversation memory, creating chains of LLM calls, and building agents that
        use tools. The core abstractions are Chains which are sequences of operations,
        Retrievers which fetch relevant documents, Memory which stores and recalls conversation
        history, and Agents which are LLMs that can use tools like web search or calculators.
        LangChain significantly reduces the boilerplate code needed to build production-ready
        LLM applications and is widely used in the industry for chatbots and document Q&A."""
    },
    {
        "id": "nlp_001",
        "title": "Natural Language Processing",
        "category": "NLP",
        "content": """Natural Language Processing or NLP is the branch of AI focused on
        understanding and generating human language. Key NLP tasks include text classification,
        named entity recognition, sentiment analysis, machine translation, summarisation, and
        question answering. Before transformers, NLP relied on techniques like bag-of-words,
        TF-IDF, and RNNs. The 2017 Attention Is All You Need paper introduced the transformer
        architecture which revolutionised NLP. BERT showed that pretraining on large text
        corpora and then fine-tuning on specific tasks dramatically outperforms training from
        scratch. GPT models showed that autoregressive language models can generalise
        surprisingly well to many tasks with just prompting."""
    },
    {
        "id": "eval_001",
        "title": "Model Evaluation Metrics",
        "category": "Model Evaluation",
        "content": """Choosing the right evaluation metric is critical in machine learning.
        For classification: accuracy is the proportion of correct predictions but is misleading
        for imbalanced datasets, precision measures of predicted positives how many are actually
        correct, recall measures of actual positives how many the model caught, F1 score is the
        harmonic mean of precision and recall and is good for imbalanced problems, and ROC-AUC
        measures the ability to distinguish between classes. For regression: MAE is mean
        absolute error giving average error in original units, RMSE is root mean squared error
        which penalises large errors more heavily, and R-squared shows proportion of variance
        explained. In medical diagnosis recall matters more because missing a disease is worse
        than a false alarm."""
    },
    {
        "id": "data_001",
        "title": "Data Preprocessing in Machine Learning",
        "category": "Data Science",
        "content": """Data preprocessing is the process of cleaning and transforming raw data
        before feeding it to a machine learning model. It is often the most time-consuming
        part of any ML project, typically taking 60 to 80 percent of the total time. Key steps
        include handling missing values through imputation with mean median or mode or by
        dropping rows, removing duplicates, handling outliers by capping removing or
        transforming, encoding categorical variables using one-hot encoding label encoding or
        ordinal encoding, feature scaling using normalisation to 0-1 range or standardisation
        to mean 0 and standard deviation 1, and feature engineering to create new features from
        existing ones. Poor data quality is the most common reason ML models fail in production.
        The principle garbage in garbage out is fundamental."""
    },
    {
        "id": "cv_001",
        "title": "Computer Vision and CNNs",
        "category": "Computer Vision",
        "content": """Computer vision is the field of AI that deals with enabling computers to
        interpret and understand visual information from images and videos. Key tasks include
        image classification to identify what object is in an image, object detection to find
        where objects are located, semantic segmentation to classify every pixel, and image
        generation. CNNs or Convolutional Neural Networks were the dominant architecture for
        years using convolutional layers to detect spatial patterns. More recently Vision
        Transformers have matched or exceeded CNN performance on many tasks. Computer vision
        is used in self-driving cars, medical imaging diagnosis, facial recognition, quality
        control in manufacturing, and augmented reality. Transfer learning is especially powerful
        in computer vision."""
    },
    {
        "id": "eth_001",
        "title": "AI Ethics and Bias",
        "category": "AI Ethics",
        "content": """AI systems can perpetuate and amplify existing biases in society if not
        carefully designed. Bias enters through training data which reflects historical
        inequalities, model architecture choices, and deployment contexts. Famous examples
        include facial recognition systems that perform worse on darker skin tones, hiring
        algorithms that disadvantaged women, and recidivism prediction tools that were biased
        against certain demographic groups. Fairness in ML is not a single concept as there are
        multiple mathematical definitions that are often mutually incompatible. Key practices
        include using diverse training data, conducting bias audits, using explainability tools
        like LIME and SHAP, and ongoing monitoring. Regulations like GDPR and emerging AI Acts
        are making ethical AI a legal requirement."""
    },
    {
        "id": "prod_001",
        "title": "Deploying ML Models to Production",
        "category": "MLOps",
        "content": """Getting a model from a notebook to production involves several steps.
        First the model is serialised and saved to disk using joblib pickle or a framework
        specific format. Then it is wrapped in an API using Flask FastAPI or Django so other
        services can call it over HTTP. The API is containerised using Docker for consistent
        deployment. It is then deployed to cloud infrastructure such as AWS GCP or Azure and
        load balanced for scalability. Model monitoring tracks data drift where input
        distribution shifts over time, concept drift where the relationship between inputs and
        outputs changes, and performance degradation. MLflow and Weights and Biases help manage
        experiments and deployments. Continuous retraining pipelines ensure the model stays
        accurate as the world changes."""
    },
]


# ─────────────────────────────────────────────────────
# VECTOR STORE — built once and cached in memory
# ─────────────────────────────────────────────────────
@st.cache_resource
def build_vector_store():
    """
    Chunk all documents and build a TF-IDF vector store.
    This is cached so it only runs once when the app starts.

    In a real project you'd use:
    - sentence-transformers for better semantic search
    - ChromaDB or FAISS as the vector database
    Here TF-IDF shows the same concept without extra dependencies.
    """
    chunks = []
    metadata = []

    for doc in DOCS:
        # Split into ~120-word chunks with 20-word overlap
        words = doc["content"].split()
        i = 0
        while i < len(words):
            end = min(i + 120, len(words))
            chunk_text = " ".join(words[i:end])
            chunks.append(chunk_text)
            metadata.append({
                "title":    doc["title"],
                "category": doc["category"],
                "doc_id":   doc["id"],
            })
            i += 100  # 20-word overlap

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
    vectors = vectorizer.fit_transform(chunks)

    return vectorizer, vectors, chunks, metadata


def retrieve(query, vectorizer, vectors, chunks, metadata, top_k=3):
    """
    Find the most relevant document chunks for a given query.

    Steps:
    1. Convert the query to the same vector format as stored chunks
    2. Calculate cosine similarity between query and all chunk vectors
    3. Return the top_k most similar chunks
    """
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, vectors)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    seen_titles = set()
    for idx in top_idx:
        if sims[idx] > 0.005:
            title = metadata[idx]["title"]
            # avoid showing duplicate chunks from the same doc
            if title not in seen_titles:
                seen_titles.add(title)
                results.append({
                    "text":     chunks[idx],
                    "score":    round(float(sims[idx]), 4),
                    "title":    title,
                    "category": metadata[idx]["category"],
                })
    return results


# ─────────────────────────────────────────────────────
# RESPONSE GENERATOR
# Uses the Anthropic API connected to this app
# ─────────────────────────────────────────────────────
def generate_answer(question, retrieved_chunks, conversation_history, api_key=None):
    """
    Build the full prompt with context and history, call the LLM.

    The prompt has three parts:
    1. Retrieved document chunks (the RAG part)
    2. Past conversation (the memory part)
    3. The current question
    """
    try:
        import anthropic
        if api_key:
            client = anthropic.Anthropic(api_key=api_key)
        else:
            client = anthropic.Anthropic()

        # Format retrieved context
        context = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            context += f"\n[Document {i}: {chunk['title']}]\n{chunk['text']}\n"

        system = """You are a helpful AI and machine learning knowledge assistant.
Answer questions based on the provided context documents. Be clear, friendly, and use plain English.
If the answer is not in the context, say so honestly — don't make things up.
Keep answers to 2-4 paragraphs. For follow-up questions, use the conversation history."""

        # Build message list including history (this is the memory)
        messages = []
        for turn in conversation_history[-6:]:  # keep last 6 turns to avoid token limits
            messages.append({"role": "user",      "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        # Current question with context
        messages.append({
            "role": "user",
            "content": f"Relevant documents from knowledge base:\n{context}\n---\nQuestion: {question}"
        })

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=system,
            messages=messages
        )
        return response.content[0].text, None

    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Context-Aware AI with Document Retrieval")
    st.divider()

    # API key input — paste your key here
    st.subheader("API Key")
    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get a free key at console.anthropic.com → API Keys"
    )
    if api_key_input:
        st.session_state["api_key"] = api_key_input
        st.success("Key saved!", icon="✅")
    elif "api_key" not in st.session_state:
        st.info("Paste your API key above to start chatting.")

    st.divider()

    st.subheader("Knowledge Base")
    st.caption(f"{len(DOCS)} documents loaded")

    categories = sorted(set(d["category"] for d in DOCS))
    for cat in categories:
        docs_in_cat = [d for d in DOCS if d["category"] == cat]
        with st.expander(f"📂 {cat} ({len(docs_in_cat)})"):
            for d in docs_in_cat:
                st.markdown(f"• {d['title']}")

    st.divider()

    st.subheader("Settings")
    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1, max_value=5, value=3,
        help="How many document chunks to retrieve per question"
    )
    show_sources = st.checkbox("Show sources", value=True)
    show_scores  = st.checkbox("Show relevance scores", value=False)

    st.divider()

    if st.button("Clear conversation", use_container_width=True, type="secondary"):
        st.session_state.messages  = []
        st.session_state.history   = []
        st.rerun()

    st.caption("Built with Streamlit · TF-IDF Retrieval · Claude LLM")


# ─────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────
st.title("Context-Aware RAG Chatbot")
st.markdown(
    "Ask anything about **AI, ML, Deep Learning, RAG, NLP, and more**. "
    "I search my knowledge base first, then answer — so you know exactly where the info comes from."
)

# Init session state
if "messages"  not in st.session_state: st.session_state.messages  = []
if "history"   not in st.session_state: st.session_state.history   = []

# Build vector store (runs once, cached)
vectorizer, vectors, chunks, meta = build_vector_store()

# ── Example question buttons ──────────────────────────
st.markdown("**Try asking:**")
examples = [
    "What is RAG and why is it useful?",
    "How do neural networks learn?",
    "What is the difference between precision and recall?",
    "How do vector embeddings work?",
    "What is LangChain used for?",
    "What are the types of machine learning?",
]

cols = st.columns(3)
for i, (col, q) in enumerate(zip(cols * 2, examples)):
    with col:
        if st.button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state["pending"] = q

st.divider()

# ── Chat history display ──────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources under each assistant message
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander("📚 Sources retrieved", expanded=False):
                for src in msg["sources"]:
                    score_text = f" · score: `{src['score']}`" if show_scores else ""
                    st.markdown(f"**[{src['category']}]** {src['title']}{score_text}")

# ── Handle pending (example button click) ────────────
user_input = None
if "pending" in st.session_state:
    user_input = st.session_state.pop("pending")
else:
    user_input = st.chat_input("Ask a question about AI/ML...")

# ── Process question ──────────────────────────────────
if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):

        # Step 1: Retrieve relevant chunks
        with st.spinner("Searching knowledge base..."):
            retrieved = retrieve(user_input, vectorizer, vectors, chunks, meta, top_k)

        # Step 2: Generate answer using LLM + retrieved context + history
        with st.spinner("Writing answer..."):
            answer, error = generate_answer(
                user_input,
                retrieved,
                st.session_state.history,
                api_key=st.session_state.get("api_key")
            )

        if error:
            st.error(f"Something went wrong: {error}")
            answer = "Sorry, I ran into an error. Please try again."

        st.markdown(answer)

        # Show sources inline
        if show_sources and retrieved:
            with st.expander("📚 Sources retrieved", expanded=True):
                for src in retrieved:
                    score_text = f" · relevance: `{src['score']}`" if show_scores else ""
                    st.markdown(f"**[{src['category']}]** {src['title']}{score_text}")
                    st.caption(src["text"][:180] + "...")

    # Save to history (memory)
    st.session_state.history.append({
        "user":      user_input,
        "assistant": answer or ""
    })
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer or "",
        "sources": [{"title": r["title"], "score": r["score"],
                     "category": r["category"]} for r in retrieved]
    })

# ── Footer ─────────────────────────────────────────────
if st.session_state.messages:
    n_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.caption(
        f"Conversation: {n_turns} turn{'s' if n_turns != 1 else ''} · "
        f"Memory: last {min(len(st.session_state.history), 6)} turns · "
        f"Knowledge base: {len(DOCS)} docs · "
        f"Retrieving top {top_k} chunks"
    )