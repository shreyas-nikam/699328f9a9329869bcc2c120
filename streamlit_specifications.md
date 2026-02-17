
# Streamlit Application Specification: Topic Modeling 10-K Filings for Risk Surveillance

## 1. Application Overview

**Purpose**: This Streamlit application serves as a practical tool for **CFA Charterholders and Investment Professionals** to automate and scale the analysis of "Risk Factors" sections from 10-K filings. It aims to transform vast amounts of unstructured text into actionable investment intelligence by identifying latent risk themes, benchmarking companies, and detecting year-over-year risk profile shifts.

**High-level Story Flow**:
The application guides an equity analyst at Alpha Capital Management through a structured workflow to leverage advanced NLP for risk surveillance.
1.  **Data Ingestion & Preprocessing**: The analyst begins by loading raw 10-K risk factor texts and preparing them through a meticulous preprocessing pipeline, including custom financial stop-word removal. This ensures only meaningful terms are considered.
2.  **Topic Discovery (LDA)**: Next, the analyst employs Latent Dirichlet Allocation (LDA) to uncover classical, keyword-driven risk themes, evaluating the optimal number of topics using coherence scores and interactively visualizing the results to assign human-readable labels.
3.  **Semantic Topic Discovery (Embeddings)**: As a modern alternative, embedding-based clustering is used to capture more nuanced semantic relationships in the text. This involves generating dense vector representations of paragraphs and clustering them, with cluster quality assessed by silhouette scores.
4.  **Portfolio Risk Mapping**: With topics established, the analyst quantifies each company's exposure to these themes, visualizing this through a company-topic heatmap. A company similarity network is then constructed to identify non-obvious peer groupings based on shared risk profiles.
5.  **Dynamic Risk Monitoring (Drift Detection)**: Crucially, the app detects significant year-over-year shifts in a company's risk profile using Jensen-Shannon Divergence, flagging potential early warning signals for strategic changes or emerging threats.
6.  **Actionable Intelligence**: Finally, the analyst synthesizes insights from both LDA and embedding models, comparing their strengths and weaknesses, and summarizing key findings to inform investment decisions and portfolio risk management. The application facilitates a data-driven approach to scaling risk coverage and making better-informed investment decisions.

## 2. Code Requirements

### Import Statement

```python
from source import *
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import pyLDAvis.display
import base64
import nltk # Explicitly import for NLTK downloads
```

### `st.session_state` Initialization, Update, and Read

`st.session_state` keys are used to preserve application state and intermediate results across page transitions and user interactions.

**Initialization (at the top of `app.py`):**
```python
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'
if 'corpus_df' not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if 'preprocessed_done' not in st.session_state:
    st.session_state.preprocessed_done = False
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = None
if 'bow_corpus' not in st.session_state:
    st.session_state.bow_corpus = None
if 'lda_done' not in st.session_state:
    st.session_state.lda_done = False
if 'lda_coherence_scores' not in st.session_state:
    st.session_state.lda_coherence_scores = {}
if 'lda_models_dict' not in st.session_state:
    st.session_state.lda_models_dict = {}
if 'best_k_lda' not in st.session_state:
    st.session_state.best_k_lda = None
if 'best_lda_model' not in st.session_state:
    st.session_state.best_lda_model = None
if 'pyldavis_html_path' not in st.session_state:
    st.session_state.pyldavis_html_path = 'lda_10k_topics.html' # Default path
if 'embedding_done' not in st.session_state:
    st.session_state.embedding_done = False
if 'para_df' not in st.session_state:
    st.session_state.para_df = pd.DataFrame()
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'sbert_model' not in st.session_state:
    st.session_state.sbert_model = None # Store the SentenceTransformer model
if 'best_k_emb' not in st.session_state:
    st.session_state.best_k_emb = None
if 'best_km_model' not in st.session_state:
    st.session_state.best_km_model = None
if 'embedding_topic_labels' not in st.session_state:
    # Placeholder labels, can be refined based on actual clusters
    st.session_state.embedding_topic_labels = {
        0: "Cybersecurity & Data Privacy (Embedding)",
        1: "Regulatory & Legal Compliance (Embedding)",
        2: "Economic & Market Volatility (Embedding)",
        3: "Supply Chain & Operations (Embedding)",
        4: "Technological & Innovation Risks (Embedding)",
        5: "ESG & Climate Change (Embedding)",
        6: "Talent & Workforce Issues (Embedding)",
        7: "M&A & Integration (Embedding)",
        8: "Competition & IP (Embedding)",
        9: "Inflation & Interest Rate (Embedding)",
        10: "Geopolitical & Trade (Embedding)",
        11: "Credit & Liquidity Risk (Embedding)",
        12: "Market Access & Growth (Embedding)",
        13: "Product Innovation & Obsolescence (Embedding)",
        14: "Operational Efficiency (Embedding)"
    }
if 'mapping_done' not in st.session_state:
    st.session_state.mapping_done = False
if 'topic_df_for_drift' not in st.session_state:
    st.session_state.topic_df_for_drift = pd.DataFrame() # To store topic_df for later use
if 'full_topic_labels_map' not in st.session_state:
    st.session_state.full_topic_labels_map = {}
if 'company_topics_avg_labeled' not in st.session_state:
    st.session_state.company_topics_avg_labeled = pd.DataFrame()
if 'sim_df' not in st.session_state:
    st.session_state.sim_df = pd.DataFrame()
if 'G_network' not in st.session_state:
    st.session_state.G_network = None
if 'company_names' not in st.session_state:
    st.session_state.company_names = []
if 'drift_done' not in st.session_state:
    st.session_state.drift_done = False
if 'drift_df' not in st.session_state:
    st.session_state.drift_df = pd.DataFrame()
if 'FILING_DIR' not in st.session_state:
    st.session_state.FILING_DIR = 'filings/risk_factors'
```

**Helper for displaying local HTML (at the top of `app.py`):**
```python
def render_html_file_in_streamlit(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=800, scrolling=True)
    else:
        st.error(f"HTML file not found: {filepath}")
```

### UI Interactions and `source.py` Function Call Points

**Constants (global in `app.py`):**
```python
PAGE_NAMES = [
    'Home / Introduction',
    '1. Data Loading & Preprocessing',
    '2. LDA Topic Discovery',
    '3. Embedding-Based Topic Clustering',
    '4. Portfolio Risk Mapping',
    '5. Year-over-Year Topic Drift',
    '6. Synthesize & Compare'
]
YEAR_1 = 2023
YEAR_2 = 2024
```

**Sidebar Navigation:**
```python
st.sidebar.title("Navigation")
st.session_state.current_page = st.sidebar.selectbox('Go to', PAGE_NAMES)
```

**Page: Home / Introduction**
```python
if st.session_state.current_page == 'Home / Introduction':
    st.title("Topic Modeling 10-K Filings for Risk Surveillance")
    st.markdown(f"## An Investment Professional's Workflow at Alpha Capital Management")

    st.markdown(f"As a **CFA Charterholder** and **Equity Analyst** at **Alpha Capital Management**, your core responsibility is to identify and assess material risks affecting the companies in your investment portfolio. Each year, thousands of pages of 'Risk Factors' disclosures in 10-K filings contain crucial insights, but the sheer volume makes manual review impossible. You need a scalable, automated solution to uncover latent risk themes, benchmark companies against their true risk peers, and detect emerging threats or strategic shifts early.")

    st.markdown(f"This application guides you through a real-world workflow to leverage advanced Natural Language Processing (NLP) techniques, specifically topic modeling, to transform unstructured 10-K text into actionable investment intelligence. You will:")
    st.markdown(f"- **Preprocess** a corpus of 10-K Risk Factor sections, including domain-specific stop words to filter boilerplate language.")
    st.markdown(f"- **Discover topics** using both classical Latent Dirichlet Allocation (LDA) and a modern embedding-based clustering approach.")
    st.markdown(f"- **Interpret and label** these topics (e.g., 'Regulatory/Legal,' 'Cybersecurity,' 'Supply Chain').")
    st.markdown(f"- **Visualize** topic prevalence and relationships.")
    st.markdown(f"- **Construct a company-topic heatmap** to assess risk exposure across your portfolio.")
    st.markdown(f"- **Build a company similarity network** to identify non-obvious peer groupings.")
    st.markdown(f"- **Detect year-over-year topic drift** for individual companies, flagging significant changes in their risk profiles.")

    st.markdown(f"This automated approach allows you to direct your valuable human analytical capacity to areas of highest impact, providing a transparent, data-driven methodology for scaling risk coverage and making better-informed investment decisions.")

```

**Page: 1. Data Loading & Preprocessing**
```python
elif st.session_state.current_page == '1. Data Loading & Preprocessing':
    st.title("1. Data Loading & Preprocessing")
    st.markdown(f"## Cleaning the Unstructured Gold: Preparing for Analysis")

    st.markdown(f"As an equity analyst, the first step is always to gather your raw data. For this analysis, you'll be working with the 'Risk Factors' sections from 10-K filings of 20-50 S&P 500 companies over two consecutive years (FY{YEAR_1} and FY{YEAR_2}). This corpus, though a fraction of the total filings, represents a manageable yet substantial dataset for initial exploration and methodology validation. Our goal is to ingest this data efficiently for subsequent processing.")

    st.markdown(f"### 1.1 Load 10-K Risk Factors")
    if st.button("Load 10-K Risk Factors"):
        with st.spinner("Loading data..."):
            st.session_state.corpus_df = load_risk_factors(filing_dir=st.session_state.FILING_DIR)
        if not st.session_state.corpus_df.empty:
            st.success(f"Loaded {len(st.session_state.corpus_df)} documents from {st.session_state.corpus_df['ticker'].nunique()} companies over {st.session_state.corpus_df['year'].nunique()} years.")
            st.dataframe(st.session_state.corpus_df.head())
            st.markdown(f"Distribution of documents per year:")
            st.dataframe(st.session_state.corpus_df.groupby('year')['ticker'].count())
        else:
            st.error(f"Failed to load data. Please ensure the directory '{st.session_state.FILING_DIR}' exists and contains the required files.")
    
    st.markdown(f"<div class='alert alert-info'><strong>Explanation of Execution:</strong> This initial step loads all risk factor text files into a pandas DataFrame. For you, the analyst, seeing the `head()` of the DataFrame and the document distribution by year confirms that the raw data is correctly structured and ready for the next stage of analysis. This is crucial for verifying the integrity of your input data before significant computational tasks.</div>", unsafe_allow_html=True)


    st.markdown(f"### 1.2 Preprocess Text")
    st.markdown(f"10-K risk factor sections contain a lot of noise: boilerplate legal language, common words, and punctuation. As an analyst, you know that irrelevant words can obscure the true underlying risk themes. To ensure our topic models extract meaningful, discriminative insights, we must rigorously preprocess the text. This involves tokenization, lemmatization, and critically, the removal of custom financial stop words that might be common in 10-K filings but carry no specific topical meaning.")

    if st.button("Preprocess Text", disabled=st.session_state.corpus_df.empty):
        with st.spinner("Downloading NLTK data and preprocessing text..."):
            # NLTK data downloads are part of the initial setup in source.py,
            # ensuring they are available for preprocessing.
            # In a Streamlit app, we'd ensure these run once or handle gracefully.
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)

            st.session_state.corpus_df['tokens'] = st.session_state.corpus_df['text'].apply(preprocess_for_topics)
            st.session_state.corpus_df['n_tokens'] = st.session_state.corpus_df['tokens'].apply(len)
        st.session_state.preprocessed_done = True
        st.success("Text preprocessing complete!")
        st.markdown(f"Average tokens per document after preprocessing: {st.session_state.corpus_df['n_tokens'].mean():.0f}")
        st.markdown(f"Sample of preprocessed tokens for the first document:")
        st.write(st.session_state.corpus_df['tokens'].iloc[0][:20])

    if st.session_state.preprocessed_done:
        st.markdown(f"<div class='alert alert-warning'><strong>Practitioner Warning: Financial Stop Words are Critical.</strong> Standard NLP stop-word lists don't include words like \"material,\" \"adverse,\" or \"significant.\" These terms, while common, appear in virtually every 10-K risk factor and carry no discriminative power between topics. Without removing them, LDA produces topics dominated by generic filing words rather than substantive risk themes. The `FINANCIAL_STOPWORDS` list (from `source.py`) is tuned for SEC filings; different document types (e.g., analyst reports, news) would require different domain-specific stop words. For you, the analyst, this custom list is key to extracting genuinely novel insights.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='alert alert-info'><strong>Explanation of Execution:</strong> The code applies a robust preprocessing pipeline: lowercasing, punctuation removal, tokenization, lemmatization, and filtering against both generic English and specific financial stop words. The average token count provides a quick check on the effectiveness of the filtering. This meticulous cleaning ensures that the subsequent topic models are fed with text that is rich in meaningful vocabulary, enabling you to discover more precise and actionable risk themes.</div>", unsafe_allow_html=True)

```

**Page: 2. LDA Topic Discovery**
```python
elif st.session_state.current_page == '2. LDA Topic Discovery':
    st.title("2. LDA Topic Discovery")
    st.markdown(f"## Unveiling Latent Risk Themes: Classical Approach")

    st.markdown(f"With a clean corpus, your next objective is to discover the latent risk themes. Latent Dirichlet Allocation (LDA) is a powerful, classical probabilistic model that assumes each document is a mixture of 'K' topics, and each topic is a distribution over words. Your task is to apply LDA, determine the optimal number of topics (K) for interpretability, and then visualize these topics.")

    st.markdown(f"The core idea of LDA is that a document can be represented as a probabilistic distribution over topics, and a topic as a probabilistic distribution over words. Mathematically, the probability of a word $w$ occurring in a document $d$ can be expressed as:")
    st.markdown(r"$$P(w | d) = \sum_{k=1}^{K} P(w | z = k) \cdot P(z = k | d) = \sum_{k=1}^{K} \phi_{kw} \cdot \theta_{dk}$$")
    st.markdown(r"where $P(w | z = k)$ is the probability of word $w$ given topic $k$. This is also denoted as $\phi_{kw}$, representing the **topic-word distribution** ($\phi_k \sim \text{Dir}(\eta)$).")
    st.markdown(r"$P(z = k | d)$ is the probability of topic $k$ given document $d$. This is also denoted as $\theta_{dk}$, representing the **document-topic distribution** ($\theta_d \sim \text{Dir}(\alpha)$).")
    st.markdown(r"$\alpha$ and $\eta$ are Dirichlet priors that control the sparsity of topic distributions in documents and word distributions in topics, respectively. A lower $\alpha$ means documents tend to concentrate on fewer topics.")
    st.markdown(f"For an investment professional, the $\theta_d$ vectors are critical; they represent a 'risk profile fingerprint' for each company, showing its exposure to various risk themes.")

    st.markdown(f"To select the optimal number of topics $K$, we will use the **Coherence Score ($C_v$)**, which measures how semantically meaningful and interpretable the topics are. A higher coherence score indicates that the top words in a topic tend to co-occur together in the corpus.")
    st.markdown(f"The $C_v$ coherence score is calculated as:")
    st.markdown(r"$$C_v = \frac{1}{\binom{N}{2}} \sum_{i<j} \text{NPMI}(w_i, w_j)$$")
    st.markdown(r"where $\text{NPMI}$ is the Normalized Pointwise Mutual Information of co-occurring word pairs $(w_i, w_j)$ within a topic, and $N$ is the number of top words considered per topic. Our target is a $C_v > 0.40$ for well-formed financial topics.")

    min_topics, max_topics = 5, 15
    selected_k_range = st.slider("Select range for Number of Topics (K) to evaluate (LDA)", min_value=min_topics, max_value=max_topics, value=(min_topics, max_topics))

    if st.button("Run LDA Topic Modeling", disabled=not st.session_state.preprocessed_done):
        with st.spinner("Building dictionary and training LDA models..."):
            st.session_state.dictionary = corpora.Dictionary(st.session_state.corpus_df['tokens'])
            st.session_state.dictionary.filter_extremes(no_below=3, no_above=0.7)
            st.session_state.bow_corpus = [st.session_state.dictionary.doc2bow(tokens) for tokens in st.session_state.corpus_df['tokens']]

            st.session_state.lda_coherence_scores = {}
            st.session_state.lda_models_dict = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, k in enumerate(range(selected_k_range[0], selected_k_range[1] + 1)):
                status_text.text(f"Training LDA for K={k} ({i+1}/{selected_k_range[1]-selected_k_range[0]+1})...")
                lda_model = LdaMulticore(
                    st.session_state.bow_corpus,
                    num_topics=k,
                    id2word=st.session_state.dictionary,
                    passes=15,
                    workers=os.cpu_count() or 1, # Use available CPU cores
                    random_state=42,
                    alpha='asymmetric',
                    eta='auto'
                )
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=st.session_state.corpus_df['tokens'],
                    dictionary=st.session_state.dictionary,
                    coherence='c_v'
                )
                coherence_score = coherence_model.get_coherence()
                st.session_state.lda_coherence_scores[k] = coherence_score
                st.session_state.lda_models_dict[k] = lda_model
                st.write(f"K={k}: Coherence = {coherence_score:.4f}")
                progress_bar.progress((i + 1) / (selected_k_range[1]-selected_k_range[0]+1))

            st.session_state.best_k_lda = max(st.session_state.lda_coherence_scores, key=st.session_state.lda_coherence_scores.get)
            st.session_state.best_lda_model = st.session_state.lda_models_dict[st.session_state.best_k_lda]
            st.success(f"Best K = {st.session_state.best_k_lda} (Coherence = {st.session_state.lda_coherence_scores[st.session_state.best_k_lda]:.4f})")
            st.session_state.lda_done = True

            # Plot coherence scores
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(list(st.session_state.lda_coherence_scores.keys()), list(st.session_state.lda_coherence_scores.values()), marker='o')
            ax.set_xlabel("Number of Topics (K)")
            ax.set_ylabel("Coherence Score ($C_v$)")
            ax.set_title("LDA Coherence Score ($C_v$) vs. Number of Topics (K)")
            ax.set_xticks(list(st.session_state.lda_coherence_scores.keys()))
            ax.grid(True)
            st.pyplot(fig)

            st.markdown(f"\nTop 10 keywords for each of the {st.session_state.best_k_lda} discovered topics (LDA):")
            for idx, topic in st.session_state.best_lda_model.print_topics(num_words=10):
                st.write(f"Topic {idx}: {topic}")

            # Visualize the topics interactively using pyLDAvis
            vis = gensimvis.prepare(st.session_state.best_lda_model, st.session_state.bow_corpus, st.session_state.dictionary)
            gensimvis.save_html(vis, st.session_state.pyldavis_html_path)
            st.markdown(f"Interactive LDA visualization saved to `{st.session_state.pyldavis_html_path}`.")
            st.markdown(f"**Open this HTML file in your browser to explore topics interactively.**")
            st.markdown(f"*(For security reasons, Streamlit does not directly embed local HTML files by default. You can download and open it.)*")
            # Provide a download button for the HTML file
            with open(st.session_state.pyldavis_html_path, "rb") as file:
                btn = st.download_button(
                    label="Download pyLDAvis HTML",
                    data=file,
                    file_name=st.session_state.pyldavis_html_path,
                    mime="text/html"
                )

    if st.session_state.lda_done:
        st.markdown(f"<div class='alert alert-info'><strong>Explanation of Execution:</strong> This section first builds a vocabulary (`dictionary`) and a Bag-of-Words (`bow_corpus`) representation of your preprocessed text. Then, it iteratively trains LDA models for a range of topic numbers ($K$) and calculates the $C_v$ coherence score for each. As an analyst, you'd look for an \"elbow\" in the coherence plot (though often it's less clear-cut) or simply choose the $K$ with the highest coherence, as we do here. The generated HTML file (`lda_10k_topics.html`) provides an an interactive visualization. This tool lets you visually inspect topic distances, prevalences, and the relevance of words to each topic, which is critical for **interpreting and assigning human-readable labels** to these machine-discovered themes (e.g., \"Topic 0: 'Regulatory/Legal', 'Topic 1: 'Cybersecurity'\"). This forms the foundation of your risk surveillance.</div>", unsafe_allow_html=True)
```

**Page: 3. Embedding-Based Topic Clustering**
```python
elif st.session_state.current_page == '3. Embedding-Based Topic Clustering':
    st.title("3. Embedding-Based Topic Clustering")
    st.markdown(f"## Semantic Nuances: Modern Approach")

    st.markdown(f"While LDA is powerful, it's a bag-of-words model, meaning it doesn't inherently understand semantic similarity. For instance, 'cyber attack' and 'data breach' are semantically close but might be treated as distinct if they don't co-occur frequently. As an analyst, you want to ensure the most nuanced semantic relationships are captured. This section introduces a modern alternative: embedding-based topic clustering using Sentence-BERT to generate dense semantic vector representations for text, followed by K-Means clustering.")

    st.markdown(f"Sentence-BERT maps text to a dense vector space, where semantically similar sentences are close together. The similarity between two paragraph vectors, $\mathbf{{v}}_i$ and $\mathbf{{v}}_j$, can be computed using **cosine similarity**:")
    st.markdown(r"$$\text{sim}(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{||\mathbf{v}_i|| \cdot ||\mathbf{v}_j||}$$")
    st.markdown(r"where $||\mathbf{v}||$ denotes the Euclidean norm of vector $\mathbf{v}$. This metric ranges from -1 (opposite) to 1 (identical), with 0 indicating orthogonality.")

    st.markdown(f"After generating embeddings, K-Means clustering groups these vectors into 'K' clusters, representing our topics. We'll assess the quality of these clusters using the **Silhouette Score**, which measures how similar an object is to its own cluster compared to other clusters. A higher silhouette score ($> 0.20$ is a reasonable target) indicates well-separated and dense clusters.")

    min_clusters_emb, max_clusters_emb = 6, 15
    selected_k_emb_range = st.slider("Select range for Number of Clusters (K) to evaluate (Embeddings)", min_value=min_clusters_emb, max_value=max_clusters_emb, value=(min_clusters_emb, max_clusters_emb))

    if st.button("Run Embedding-Based Clustering", disabled=not st.session_state.preprocessed_done):
        with st.spinner("Splitting into paragraphs, generating embeddings, and clustering..."):
            all_paragraphs = []
            for idx, row in st.session_state.corpus_df.iterrows():
                paras = [p.strip() for p in row['text'].split('\n\n') if len(p.strip()) > 100]
                for p in paras:
                    all_paragraphs.append({
                        'ticker': row['ticker'],
                        'year': row['year'],
                        'paragraph': p,
                        'doc_idx': idx
                    })
            st.session_state.para_df = pd.DataFrame(all_paragraphs)
            st.write(f"Total paragraphs extracted for embedding: {len(st.session_state.para_df)}")

            sbert_model_name = 'all-MiniLM-L6-v2'
            if st.session_state.sbert_model is None:
                st.session_state.sbert_model = SentenceTransformer(sbert_model_name)
            st.markdown(f"\nGenerating embeddings using Sentence-BERT model: {sbert_model_name}...")
            st.session_state.embeddings = st.session_state.sbert_model.encode(
                st.session_state.para_df['paragraph'].tolist(),
                show_progress_bar=False, # Streamlit handles progress differently
                batch_size=32,
                normalize_embeddings=True
            )
            st.write(f"Embedding shape: {st.session_state.embeddings.shape}")

            best_sil_score = -1
            best_k_emb = -1
            best_km_model = None
            kmeans_labels = None

            st.markdown(f"\nSearching for optimal number of clusters (K) using Silhouette Score:")
            progress_bar_emb = st.progress(0)
            status_text_emb = st.empty()
            
            for i, k in enumerate(range(selected_k_emb_range[0], selected_k_emb_range[1] + 1)):
                status_text_emb.text(f"Running K-Means for K={k} ({i+1}/{selected_k_emb_range[1]-selected_k_emb_range[0]+1})...")
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(st.session_state.embeddings)
                silhouette_avg = silhouette_score(st.session_state.embeddings, labels, sample_size=min(5000, len(st.session_state.embeddings)), random_state=42)
                
                if silhouette_avg > best_sil_score:
                    best_sil_score = silhouette_avg
                    best_k_emb = k
                    best_km_model = km
                    kmeans_labels = labels
                st.write(f"K={k}: Silhouette Score = {silhouette_avg:.4f}")
                progress_bar_emb.progress((i + 1) / (selected_k_emb_range[1]-selected_k_emb_range[0]+1))

            st.session_state.para_df['cluster'] = kmeans_labels
            st.session_state.best_k_emb = best_k_emb
            st.session_state.best_km_model = best_km_model
            st.success(f"Best K for embedding-based clustering = {st.session_state.best_k_emb} (Silhouette Score = {best_sil_score:.4f})")
            st.session_state.embedding_done = True

            st.markdown(f"\nRepresentative paragraphs for each embedding-based cluster (topic):")
            for c in range(st.session_state.best_k_emb):
                mask = st.session_state.para_df['cluster'] == c
                if mask.sum() == 0:
                    continue
                centroid = st.session_state.best_km_model.cluster_centers_[c]
                dists = np.linalg.norm(st.session_state.embeddings[mask] - centroid, axis=1)
                nearest_idx_in_cluster = np.argmin(dists)
                global_idx_for_para = st.session_state.para_df[mask].index[nearest_idx_in_cluster]
                
                interpreted_label = st.session_state.embedding_topic_labels.get(c, f'Unnamed Embedding Topic {c}')
                st.markdown(f"\n**Cluster {c} ({interpreted_label}, {mask.sum()} paragraphs):**")
                st.write(f"  Representative (first 200 chars): {st.session_state.para_df.loc[global_idx_for_para, 'paragraph'][:200]}...")

    if st.session_state.embedding_done:
        st.markdown(f"<div class='alert alert-info'><strong>Key Insight: LDA vs. Embedding Clusters: Complementary Strengths.</strong><br><br>For an analyst, understanding both methods is key.<br><br>**LDA** excels at producing **interpretable word lists** per topic (e.g., \"regulation, compliance, SEC, enforcement, penalty\") and document-topic distributions ideal for human labeling. Its transparency supports regulatory and compliance applications.<br><br>**Embedding-based clustering** captures **semantic meaning** more effectively (\"cybersecurity risk from nation-state actors\" might cluster with \"data breach liability\" even without shared keywords). This reveals subtler connections but topics can be harder to label automatically as they don't explicitly provide keywords, relying instead on reviewing representative paragraphs.<br><br>**Best practice:** Use both. LDA gives you the topic vocabulary, while embeddings give you the semantic groupings. When they agree, you have high confidence. When they disagree, you've found interesting edge cases or nuances that warrant deeper investigation.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='alert alert-info'><strong>Explanation of Execution:</strong> This section first breaks down full documents into paragraphs, as embeddings perform better on shorter, more semantically focused text. It then uses a pre-trained Sentence-BERT model to convert each paragraph into a numerical vector. K-Means clustering then groups these vectors into `best_k_emb` clusters. The Silhouette Score helps determine optimal `K`, and displaying representative paragraphs for each cluster is crucial for manual interpretation. You, the analyst, would review these to understand the semantic themes captured by each cluster and compare them qualitatively to your LDA topics, seeking areas of agreement and unique insights.</div>", unsafe_allow_html=True)

```

**Page: 4. Portfolio Risk Mapping**
```python
elif st.session_state.current_page == '4. Portfolio Risk Mapping':
    st.title("4. Portfolio Risk Mapping")
    st.markdown(f"## Company-Topic Exposure Heatmap & Similarity Network: Cross-Company Insights")

    st.markdown(f"Now that you have identified risk themes using LDA, the next step for an investment professional is to quantify each company's exposure to these themes. This is crucial for portfolio risk management and peer analysis. You'll achieve this in two ways:")
    st.markdown(f"1.  **Company-Topic Exposure Heatmap:** Visualizing the average topic distribution for each company. This immediately shows which companies concentrate on which risks, enabling cross-portfolio risk surveillance.")
    st.markdown(f"2.  **Company Similarity Network:** Building a network graph where companies are nodes and edges represent their textual risk profile similarity. This can reveal non-obvious peer groupings that transcend traditional sector classifications, a valuable input for relative valuation and risk benchmarking.")

    if st.button("Generate Heatmap & Similarity Network", disabled=not st.session_state.lda_done):
        with st.spinner("Aggregating topic distributions and building network..."):
            topic_matrix = []
            for i, bow in enumerate(st.session_state.bow_corpus):
                topic_dist = dict(st.session_state.best_lda_model.get_document_topics(bow, minimum_probability=0.0))
                row_data = {f'Topic_{k}': topic_dist.get(k, 0.0) for k in range(st.session_state.best_k_lda)}
                row_data['ticker'] = st.session_state.corpus_df.iloc[i]['ticker']
                row_data['year'] = st.session_state.corpus_df.iloc[i]['year']
                topic_matrix.append(row_data)

            st.session_state.topic_df_for_drift = pd.DataFrame(topic_matrix) # Store for drift calculation
            company_topics_avg = st.session_state.topic_df_for_drift.groupby('ticker').mean().reset_index()
            company_topics_avg = company_topics_avg.drop(columns=['year'], errors='ignore')

            # Define illustrative topic labels based on LDA results (replace with actual interpretation)
            # This mapping should ideally come from user interpretation or a saved artifact after LDA.
            # For specification, using placeholders.
            example_labels = [
                'Regulatory & Legal Compliance', 'Market & Economic Volatility', 'Cybersecurity & Data Privacy',
                'Competition & IP Protection', 'Supply Chain & Operations', 'ESG & Climate Risks',
                'Credit & Liquidity Risk', 'Talent & Workforce Issues',
                'Inflation & Interest Rate', 'Geopolitical & Trade', 'M&A & Integration' # Max 11 labels
            ]
            st.session_state.full_topic_labels_map = {f'Topic_{k}': f'Topic_{k}: {label}' 
                                     for k, label in zip(range(st.session_state.best_k_lda), example_labels[:st.session_state.best_k_lda])}

            current_topic_cols = [col for col in company_topics_avg.columns if col.startswith('Topic_')]
            st.session_state.company_topics_avg_labeled = company_topics_avg[current_topic_cols].rename(columns=st.session_state.full_topic_labels_map)
            st.session_state.company_topics_avg_labeled.index = company_topics_avg['ticker']
            
            # Heatmap Visualization
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, 10))
            sns.heatmap(st.session_state.company_topics_avg_labeled, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'Topic Exposure'}, ax=ax_heatmap)
            ax_heatmap.set_title('Company Risk Topic Exposure (from 10-K Risk Factors)', fontsize=16)
            ax_heatmap.set_ylabel('Company', fontsize=12)
            ax_heatmap.set_xlabel('Risk Theme', fontsize=12)
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig_heatmap)

            # Company Similarity Network
            company_topic_vectors = company_topics_avg[current_topic_cols].values
            st.session_state.company_names = company_topics_avg['ticker'].tolist()

            sim_matrix = cosine_similarity(company_topic_vectors)
            st.session_state.sim_df = pd.DataFrame(sim_matrix, index=st.session_state.company_names, columns=st.session_state.company_names)

            G = nx.Graph()
            threshold = st.slider("Select similarity threshold for network edges", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

            for company in st.session_state.company_names:
                G.add_node(company)

            for i in range(len(st.session_state.company_names)):
                for j in range(i + 1, len(st.session_state.company_names)):
                    c1 = st.session_state.company_names[i]
                    c2 = st.session_state.company_names[j]
                    similarity = st.session_state.sim_df.loc[c1, c2]
                    if similarity > threshold:
                        G.add_edge(c1, c2, weight=similarity)
            st.session_state.G_network = G

            # Visualize the network
            fig_network, ax_network = plt.subplots(figsize=(12, 10))
            pos = nx.spring_layout(st.session_state.G_network, k=0.8, iterations=50, seed=42)
            node_size = 800
            font_size = 8

            nx.draw_networkx_nodes(st.session_state.G_network, pos, node_size=node_size, node_color='skyblue', edgecolors='black', ax=ax_network)
            nx.draw_networkx_labels(st.session_state.G_network, pos, font_size=font_size, font_weight='bold', ax=ax_network)
            
            edges = st.session_state.G_network.edges(data=True)
            widths = [d['weight'] * 4 for u, v, d in edges]
            nx.draw_networkx_edges(st.session_state.G_network, pos, width=widths, alpha=0.5, edge_color='gray', ax=ax_network)

            ax_network.set_title(f'Company Risk Similarity Network (Cosine Similarity > {threshold})', fontsize=16)
            ax_network.axis('off')
            plt.tight_layout()
            st.pyplot(fig_network)

            st.markdown(f"\nNon-obvious peer clusters (companies with similar risks, potentially different sectors):")
            for i, component in enumerate(nx.connected_components(st.session_state.G_network)):
                if len(component) > 1:
                    st.write(f"Cluster {i+1}: {sorted(component)}")
            st.session_state.mapping_done = True

    if st.session_state.mapping_done:
        st.markdown(f"<div class='alert alert-warning'><strong>Practitioner Warning: 10-K Risk Factor Language is Heavily Boilerplate.</strong> Many risk factors are near-identical across companies because corporate attorneys use template language. Before concluding that two companies have similar risk profiles, verify that the similarity reflects genuine shared risks rather than shared legal templates. The domain-specific stop words (Step 1) and the year-over-year drift analysis (Step 6) help mitigate this: boilerplate is stable across years, while genuine risk changes show up as drift. As an analyst, you should use these tools to discern true signals from noise.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='alert alert-info'><strong>Explanation of Execution:</strong> This section first averages LDA topic probabilities for each company across both years to get a consolidated risk profile. The **heatmap** then visually represents this exposure, allowing you, the analyst, to quickly identify which companies are most sensitive to specific risk categories (e.g., \"Tech Co. A has high Cybersecurity exposure\"). The **similarity network** takes this further by computing cosine similarity between these risk profiles. By setting a similarity threshold, you can identify \"risk-based\" peer groups (connected components) that might not align with traditional GICS sectors. This is invaluable for finding overlooked peers for valuation, or identifying systemic vulnerabilities across your portfolio. For example, two companies from different sectors showing high similarity in \"Supply Chain\" risk might warrant a deeper dive into their shared global sourcing strategies.</div>", unsafe_allow_html=True)

```

**Page: 5. Year-over-Year Topic Drift**
```python
elif st.session_state.current_page == '5. Year-over-Year Topic Drift':
    st.title("5. Year-over-Year Topic Drift")
    st.markdown(f"## Early Warning Signals: Dynamic Risk Monitoring")

    st.markdown(f"A static view of risk is insufficient. For an equity analyst, detecting changes in a company's risk profile year-over-year can be an early warning sign of strategic shifts, emerging threats, or significant events (e.g., M&A, new regulations). You will quantify this shift using **Jensen-Shannon Divergence (JSD)**, which measures the dissimilarity between two probability distributions.")

    st.markdown(f"The JSD for two probability distributions $P$ and $Q$ is defined as:")
    st.markdown(r"$$\text{JSD}(P||Q) = \sqrt{\frac{D_{KL}(P||M) + D_{KL}(Q||M)}{2}}$$")
    st.markdown(r"where $M = \frac{1}{2}(P+Q)$ is the midpoint distribution, and $D_{KL}$ is the Kullback-Leibler (KL) divergence.")
    st.markdown(f"The JSD value ranges from 0 to 1, where 0 indicates identical distributions and 1 indicates completely different distributions. For company topic vectors, a JSD $> 0.25$ is a useful threshold for flagging a significant thematic shift warranting analyst review.")

    if st.button(f"Compute Topic Drift ({YEAR_1} vs. {YEAR_2})", disabled=not st.session_state.mapping_done):
        with st.spinner(f"Computing topic drift for all companies between FY{YEAR_1} and FY{YEAR_2}..."):
            drift_results = []
            for ticker in st.session_state.corpus_df['ticker'].unique():
                result = compute_topic_drift(st.session_state.topic_df_for_drift, ticker, YEAR_1, YEAR_2, st.session_state.full_topic_labels_map)
                if result:
                    drift_results.append(result)

            st.session_state.drift_df = pd.DataFrame(drift_results).sort_values('jsd', ascending=False).round(4)
            st.session_state.drift_done = True
        st.success("Topic drift computation complete!")

    if st.session_state.drift_done:
        st.markdown(f"Companies with highest topic drift (FY{YEAR_1} -> FY{YEAR_2}):")
        st.dataframe(st.session_state.drift_df[['ticker', 'jsd']].head(10).reset_index(drop=True))

        if not st.session_state.drift_df.empty:
            top_drifter_ticker = st.session_state.drift_df.iloc[0]['ticker']
            st.markdown(f"**Visualize topic drift for the highest-drifting company: {top_drifter_ticker}**")

            top_drifter_data = st.session_state.drift_df[st.session_state.drift_df['ticker'] == top_drifter_ticker].iloc[0]
            delta_cols = [col for col in st.session_state.drift_df.columns if col.startswith('Topic_') and col.endswith(':')] # Filter for labeled topic columns
            topic_deltas = top_drifter_data[delta_cols].sort_values(ascending=False)

            fig_drift, ax_drift = plt.subplots(figsize=(12, 7))
            topic_deltas.plot(kind='bar', color=['red' if x < 0 else 'green' for x in topic_deltas.values], ax=ax_drift)
            ax_drift.set_title(f'Topic Prominence Change for {top_drifter_ticker} (FY{YEAR_1} vs. FY{YEAR_2})', fontsize=16)
            ax_drift.set_xlabel('Risk Topic', fontsize=12)
            ax_drift.set_ylabel('Change in Prominence (Year2 - Year1)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            ax_drift.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_drift)

        st.markdown(f"<div class='alert alert-info'><strong>Financial Interpretation: Jensen-Shannon Divergence (JSD).</strong> A high JSD value between a company's topic distributions across consecutive years means its risk landscape has changed materially. This can signal:<ul><li>**Strategic Pivots:** The company is shifting its business focus, leading to new categories of risk.</li><li>**Emerging Threats:** New industry-wide or company-specific risks are gaining prominence (e.g., \"AI regulation\" in 2023).</li><li>**Regulatory/Legal Changes:** New compliance burdens or ongoing litigation.</li><li>**M&A Activity:** Integration or divestiture risks.</li></ul>Conversely, a low JSD indicates stable risk disclosures. This often correlates with stable business performance, a finding highlighted by S&P Global research (\"No News Is Good News,\" Zhao, 2021). As an analyst, you would flag companies with JSD $> 0.25$ for deeper qualitative due diligence, as these are your potential early warning signals.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='alert alert-info'><strong>Explanation of Execution:</strong> The `compute_topic_drift` function calculates the JSD between a company's topic distribution vectors for two specified years. It also computes the individual change in prominence for each topic. By running this for all companies and sorting by JSD, you can quickly identify the companies undergoing the most significant shifts in their disclosed risk profiles. The bar chart for the top-drifting company visually highlights *which specific topics* are increasing or decreasing in prominence. This allows you, the analyst, to pinpoint the exact areas of change that require your investigative attention, providing critical dynamic risk monitoring.</div>", unsafe_allow_html=True)
```

**Page: 6. Synthesize & Compare**
```python
elif st.session_state.current_page == '6. Synthesize & Compare':
    st.title("6. Synthesize & Compare")
    st.markdown(f"## Actionable Intelligence: Comparing Models & Summarizing Insights")

    st.markdown(f"As a CFA Charterholder, your role culminates in synthesizing these quantitative insights into actionable intelligence for your portfolio managers. This final section encourages a critical review of the models, a summary of key findings, and reflections on how this workflow empowers better investment decisions.")

    st.markdown(f"### 6.1 Qualitative Comparison: LDA vs. Embedding-Based Topics")
    st.markdown(f"Review the top keywords from your LDA model and the representative paragraphs from your embedding-based clustering.")
    st.markdown(f"- **LDA Topics:** Focus on keyword lists, their coherence, and your assigned human-readable labels. How intuitive are they?")
    st.markdown(f"- **Embedding-Based Topics:** Consider the semantic nuances captured by representative paragraphs. Do they reveal subtler relationships or different facets of risk compared to LDA?")
    st.markdown(f"This comparison helps you understand the strengths and weaknesses of each approach and provides a more robust, multi-faceted understanding of the risk landscape.")

    if st.session_state.lda_done:
        st.markdown(f"\n--- **LDA Topics (Top words for Best K)** ---")
        if st.session_state.best_lda_model:
            for idx, topic in st.session_state.best_lda_model.print_topics(num_words=10):
                st.write(f"Topic {idx}: {topic}")
        else:
            st.info("LDA model not yet run or loaded.")
    else:
        st.info("Please complete LDA Topic Discovery to see results.")

    if st.session_state.embedding_done:
        st.markdown(f"\n--- **Embedding-Based Clusters (Representative Paragraphs & Interpreted Themes)** ---")
        if not st.session_state.para_df.empty and st.session_state.best_km_model and st.session_state.embeddings is not None:
            for c in range(st.session_state.best_k_emb):
                mask = st.session_state.para_df['cluster'] == c
                if mask.sum() == 0:
                    continue
                centroid = st.session_state.best_km_model.cluster_centers_[c]
                dists = np.linalg.norm(st.session_state.embeddings[mask] - centroid, axis=1)
                nearest_idx_in_cluster = np.argmin(dists)
                global_idx_for_para = st.session_state.para_df[mask].index[nearest_idx_in_cluster]
                
                interpreted_label = st.session_state.embedding_topic_labels.get(c, f'Unnamed Embedding Topic {c}')
                st.markdown(f"\n**Cluster {c} ({interpreted_label}):**")
                st.write(f"  Representative (first 200 chars): {st.session_state.para_df.loc[global_idx_for_para, 'paragraph'][:200]}...")
        else:
            st.info("Embedding-based clustering not yet run or loaded.")
    else:
        st.info("Please complete Embedding-Based Topic Clustering to see results.")

    st.markdown(f"\n**Discussion points:**")
    st.markdown(f"- Are there common themes identified by both models?")
    st.markdown(f"- Do the embedding clusters reveal more nuanced or semantically similar risks that LDA might miss?")
    st.markdown(f"- Which approach provides more easily interpretable topics for your stakeholders?")
    
    st.markdown(f"<div class='alert alert-info'><strong>Explanation of Execution:</strong> This final code cell simply re-displays the key outputs from the LDA and embedding models side-by-side. The value here is not in the code itself, but in your expert analysis. As a CFA Charterholder, you are trained to synthesize information. You would reflect on how LDA provides clear keyword-based topics, while embedding-based clustering surfaces semantic groupings. For example, if LDA identifies a 'Cybersecurity' topic, embedding clusters might further differentiate between 'Cybersecurity Incident Response' and 'Data Privacy Regulation', showing a more granular view. This comparison allows you to present a comprehensive and robust view of risk to your portfolio managers, acknowledging both traditional and modern NLP strengths.</div>", unsafe_allow_html=True)


    st.markdown(f"### 6.2 Synthesizing Key Insights for Investment Decisions")
    st.markdown(f"Based on the entire workflow, you can now provide a concise summary of actionable intelligence:")

    st.markdown(f"- **Most Prevalent Risks:** Identify the dominant risk themes across your portfolio, perhaps using the aggregated company-topic heatmap. Are there any systemic risks emerging?")
    if st.session_state.mapping_done and not st.session_state.company_topics_avg_labeled.empty:
        st.dataframe(st.session_state.company_topics_avg_labeled.sum(axis=0).sort_values(ascending=False).to_frame(name="Total Exposure"))
    
    st.markdown(f"- **Cross-Sector Peer Groupings:** Highlight companies that, despite traditional sector differences, share similar risk profiles as revealed by the network graph. This can inform new research avenues or challenge existing assumptions about peer groups.")
    if st.session_state.mapping_done and st.session_state.G_network:
        components_found = []
        for i, component in enumerate(nx.connected_components(st.session_state.G_network)):
            if len(component) > 1:
                components_found.append(f"Cluster {i+1}: {sorted(component)}")
        if components_found:
            st.markdown("\n".join(components_found))
        else:
            st.info("No significant peer groupings found with the current similarity threshold.")

    st.markdown(f"- **Companies with Significant Risk Profile Drift:** Emphasize the companies flagged by JSD. These are prime candidates for immediate qualitative review, as their evolving risk narratives could signal impending operational or strategic changes relevant to your investment thesis.")
    if st.session_state.drift_done and not st.session_state.drift_df.empty:
        st.dataframe(st.session_state.drift_df[['ticker', 'jsd']].head(5).reset_index(drop=True))

    st.markdown(f"- **Emerging Themes:** Use the topic lists (especially from the pyLDAvis visualization) to spot new risk categories that might be gaining traction across filings, signaling future market trends.")

    st.markdown(f"This automated topic modeling pipeline empowers you, the investment professional, to efficiently monitor and analyze vast amounts of unstructured text data, transforming it into clear, data-driven insights that inform critical investment and risk management decisions for Alpha Capital Management.")

```
