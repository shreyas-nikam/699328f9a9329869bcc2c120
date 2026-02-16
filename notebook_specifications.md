
# Topic Modeling 10-K Filings for Risk Surveillance: An Investment Professional's Workflow

## Introduction to the Case Study

As a **CFA Charterholder** and **Equity Analyst** at **Alpha Capital Management**, your core responsibility is to identify and assess material risks affecting the companies in your investment portfolio. Each year, thousands of pages of "Risk Factors" disclosures in 10-K filings contain crucial insights, but the sheer volume makes manual review impossible. You need a scalable, automated solution to uncover latent risk themes, benchmark companies against their true risk peers, and detect emerging threats or strategic shifts early.

This notebook guides you through a real-world workflow to leverage advanced Natural Language Processing (NLP) techniques, specifically topic modeling, to transform unstructured 10-K text into actionable investment intelligence. You will:
*   **Preprocess** a corpus of 10-K Risk Factor sections, including domain-specific stop words to filter boilerplate language.
*   **Discover topics** using both classical Latent Dirichlet Allocation (LDA) and a modern embedding-based clustering approach.
*   **Interpret and label** these topics (e.g., "Regulatory/Legal," "Cybersecurity," "Supply Chain").
*   **Visualize** topic prevalence and relationships.
*   **Construct a company-topic heatmap** to assess risk exposure across your portfolio.
*   **Build a company similarity network** to identify non-obvious peer groupings.
*   **Detect year-over-year topic drift** for individual companies, flagging significant changes in their risk profiles.

This automated approach allows you to direct your valuable human analytical capacity to areas of highest impact, providing a transparent, data-driven methodology for scaling risk coverage and making better-informed investment decisions.

## 1. Environment Setup and Data Loading

Before diving into the analysis, we need to set up our Python environment by installing the necessary libraries and then load our raw 10-K risk factor data.

### 1.1 Install Required Libraries

```python
!pip install pandas numpy nltk gensim scikit-learn sentence-transformers pyldavis matplotlib seaborn networkx scipy
```

### 1.2 Import Required Dependencies

```python
import pandas as pd
import numpy as np
import os
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, cosine_similarity

import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from scipy.spatial.distance import jensenshannon
```

### 1.3 Setting the Stage: Loading 10-K Risk Factor Data (Initial Due Diligence)

As an equity analyst, the first step is always to gather your raw data. For this analysis, you'll be working with the "Risk Factors" sections from 10-K filings of 20-50 S&P 500 companies over two consecutive years (FY2023 and FY2024). This corpus, though a fraction of the total filings, represents a manageable yet substantial dataset for initial exploration and methodology validation. Our goal is to ingest this data efficiently for subsequent processing.

```python
# Function Definition
def load_risk_factors(filing_dir='filings/risk_factors'):
    """
    Loads pre-downloaded 10-K risk factor sections.
    Each file should be named as: {TICKER}_{YEAR}_risk_factors.txt
    
    Args:
        filing_dir (str): Directory containing the 10-K risk factor text files.
        
    Returns:
        pandas.DataFrame: A DataFrame with 'ticker', 'year', 'text', and 'word_count' columns.
    """
    documents = []
    # Ensure the directory exists
    if not os.path.exists(filing_dir):
        print(f"Error: Directory '{filing_dir}' not found. Please ensure your 10-K text files are placed here.")
        return pd.DataFrame()

    for filename in os.listdir(filing_dir):
        if filename.endswith('.txt'):
            try:
                parts = filename.replace('.txt', '').split('_')
                ticker = parts[0]
                year = int(parts[1])
                
                filepath = os.path.join(filing_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                documents.append({
                    'ticker': ticker,
                    'year': year,
                    'text': text,
                    'word_count': len(text.split()) # Simple word count for initial assessment
                })
            except (IndexError, ValueError) as e:
                print(f"Skipping malformed filename: {filename} due to error: {e}")
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                
    corpus_df = pd.DataFrame(documents)
    print(f"Loaded {len(corpus_df)} documents from {corpus_df['ticker'].nunique()} companies over {corpus_df['year'].nunique()} years.")
    return corpus_df

# Function Execution
corpus_df = load_risk_factors()
print("\nSample of loaded data:")
print(corpus_df.head())
print("\nDistribution of documents per year:")
print(corpus_df.groupby('year')['ticker'].count())
```

<div class="alert alert-info">
    <strong>Explanation of Execution:</strong>
    This initial step loads all risk factor text files into a pandas DataFrame. For you, the analyst, seeing the `head()` of the DataFrame and the document distribution by year confirms that the raw data is correctly structured and ready for the next stage of analysis. This is crucial for verifying the integrity of your input data before significant computational tasks.
</div>

## 2. Cleaning the Unstructured Gold: Preprocessing 10-K Text (Preparing for Analysis)

10-K risk factor sections contain a lot of noise: boilerplate legal language, common words, and punctuation. As an analyst, you know that irrelevant words can obscure the true underlying risk themes. To ensure our topic models extract meaningful, discriminative insights, we must rigorously preprocess the text. This involves tokenization, lemmatization, and critically, the removal of custom financial stop words that might be common in 10-K filings but carry no specific topical meaning.

```python
# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Define custom financial stop words
# These are common in 10-Ks but not informative for topics
FINANCIAL_STOPWORDS = {
    'company', 'companies', 'may', 'could', 'would', 'also', 'including', 'result',
    'results', 'business', 'operations', 'material', 'materially', 'adverse',
    'adversely', 'effect', 'affect', 'significant', 'significantly', 'certain',
    'subject', 'financial', 'statements', 'fiscal', 'year', 'quarter', 'risks', 'risk',
    'factors', 'form', 'annual', 'report', 'future', 'management', 'securities',
    'investors', 'ability', 'information', 'accordance', 'generally', 'assets',
    'liabilities', 'employees', 'product', 'services', 'provide', 'markets', 'market',
    'industry', 'demand', 'changes', 'new', 'related', 'require', 'requirements',
    'government', 'laws', 'regulations', 'environmental', 'economic', 'conditions',
    'performance', 'operating', 'revenue', 'expenses', 'growth', 'share', 'stock',
    'value', 'exchange', 'price', 'dividend', 'debt', 'capital', 'funds', 'cash',
    'flow', 'customers', 'suppliers', 'partners', 'compete', 'competition', 'intellectual',
    'property', 'patent', 'trademark', 'copyright', 'litigation', 'legal', 'proceedings',
    'cybersecurity', 'data', 'breach', 'security', 'incident', 'privacy', 'personal',
    'technology', 'systems', 'network', 'software', 'hardware', 'disruption',
    'failure', 'reliance', 'third', 'party', 'providers', 'vendors', 'contractors',
    'supply', 'chain', 'global', 'international', 'trade', 'tariff', 'political',
    'geopolitical', 'inflation', 'interest', 'rates', 'currency', 'exchange', 'rates',
    'foreign', 'country', 'tax', 'taxation', 'accounting', 'auditing', 'internal',
    'controls', 'compliance', 'regulatory', 'environment', 'health', 'safety',
    'disaster', 'natural', 'catastrophe', 'climate', 'change', 'sustainability', 'ESG',
    'social', 'governance', 'reputation', 'brand', 'goodwill', 'impairment', 'valuation',
    'acquisition', 'merger', 'divestiture', 'integration', 'transition', 'synergies',
    'synergy', 'innovation', 'research', 'development', 'product', 'lifecycle',
    'obsolescence', 'litigation', 'adverse', 'outcome', 'reputational', 'damage',
    'disputes', 'claims', 'settlement', 'judgment', 'enforcement', 'penalties', 'fines',
    'sanctions', 'investigations', 'scrutiny', 'oversight', 'disclosure', 'materiality',
    'statements', 'cautionary', 'note', 'forward', 'looking', 'uncertainties', 'assumptions',
    'estimates', 'judgments', 'could', 'would', 'should', 'might', 'will', 'expect',
    'anticipate', 'believe', 'intend', 'plan', 'seek', 'project', 'may', 'cause',
    'actual', 'differ', 'materially', 'those', 'expressed', 'implied', 'herein',
    'future', 'events', 'circumstances', 'developments', 'economic', 'political',
    'social', 'technological', 'legal', 'regulatory', 'competitive', 'factors',
    'including', 'without', 'limitation', 'pandemic', 'epidemic', 'disease', 'outbreak',
    'public', 'health', 'crisis', 'remote', 'workforce', 'work', 'place', 'travel',
    'restrictions', 'government', 'responses', 'stimulus', 'packages', 'relief',
    'programs', 'lockdowns', 'quarantines', 'vaccine', 'testing', 'contact', 'tracing',
    'supply', 'chain', 'disruptions', 'logistics', 'transportation', 'shipping', 'ports',
    'factories', 'manufacturing', 'production', 'inventories', 'shortages', 'delays',
    'backlogs', 'demand', 'fluctuations', 'consumer', 'spending', 'business', 'investment',
    'credit', 'availability', 'financing', 'liquidity', 'solvency', 'defaults', 'bankruptcies',
    'restructuring', 'financial', 'instruments', 'derivatives', 'hedging', 'foreign',
    'currency', 'interest', 'rate', 'commodity', 'prices', 'energy', 'oil', 'gas',
    'metals', 'minerals', 'agriculture', 'food', 'water', 'natural', 'resources',
    'climate', 'change', 'carbon', 'emissions', 'renewable', 'energy', 'sustainability',
    'ESG', 'environmental', 'social', 'governance', 'reporting', 'standards',
    'disclosures', 'metrics', 'goals', 'targets', 'initiatives', 'investments', 'funds',
    'portfolios', 'strategies', 'allocations', 'returns', 'risks', 'opportunities',
    'valuation', 'models', 'assumptions', 'inputs', 'outputs', 'scenarios', 'stress',
    'testing', 'sensitivity', 'analysis', 'simulations', 'forecasts', 'predictions',
    'guidance', 'estimates', 'projections', 'outlook', 'expectations', 'forward',
    'looking', 'statements', 'cautionary', 'language', 'disclaimer', 'uncertainties',
    'variables', 'factors', 'events', 'circumstances', 'developments', 'trends',
    'patterns', 'shifts', 'changes', 'evolution', 'dynamics', 'drivers', 'influences',
    'impacts', 'consequences', 'effects', 'outcomes', 'results', 'benefits', 'costs',
    'expenses', 'revenues', 'earnings', 'profitability', 'margins', 'growth',
    'expansion', 'contraction', 'cycles', 'volatility', 'instability', 'uncertainty',
    'disruption', 'innovation', 'technology', 'digital', 'transformation', 'automation',
    'artificial', 'intelligence', 'machine', 'learning', 'data', 'analytics',
    'cloud', 'computing', 'cyber', 'security', 'privacy', 'breach', 'incident',
    'attack', 'threat', 'vulnerability', 'exposure', 'protection', 'resilience',
    'governance', 'compliance', 'regulatory', 'legal', 'ethical', 'social',
    'responsibility', 'reputation', 'brand', 'trust', 'confidence', 'stakeholders',
    'customers', 'employees', 'investors', 'communities', 'partners', 'suppliers',
    'vendors', 'government', 'agencies', 'public', 'opinion', 'media', 'coverage',
    'activism', 'scrutiny', 'oversight', 'investigations', 'enforcement', 'penalties',
    'fines', 'sanctions', 'litigation', 'disputes', 'claims', 'settlements', 'judgments',
    'adverse', 'outcomes', 'remedies', 'damages', 'indemnification', 'insurance',
    'coverage', 'claims', 'policies', 'premiums', 'underwriting', 'reserves',
    'actuarial', 'calculations', 'modeling', 'risk', 'management', 'mitigation',
    'strategies', 'frameworks', 'policies', 'procedures', 'controls', 'audits',
    'assessments', 'reporting', 'disclosures', 'transparency', 'accountability',
    'governance', 'board', 'directors', 'committees', 'executive', 'management',
    'officers', 'employees', 'workforce', 'talent', 'acquisition', 'retention',
    'development', 'diversity', 'inclusion', 'compensation', 'incentives',
    'culture', 'values', 'ethics', 'conduct', 'harassment', 'discrimination',
    'workplace', 'safety', 'health', 'wellness', 'benefits', 'retirement', 'pension',
    'plans', 'labor', 'relations', 'unions', 'collective', 'bargaining', 'disputes',
    'strikes', 'work', 'stoppages', 'human', 'rights', 'social', 'impact',
    'community', 'engagement', 'philanthropy', 'corporate', 'citizenship',
    'sustainability', 'environmental', 'stewardship', 'resource', 'management',
    'waste', 'reduction', 'pollution', 'prevention', 'energy', 'efficiency',
    'renewable', 'energy', 'carbon', 'footprint', 'climate', 'change',
    'adaptation', 'mitigation', 'resilience', 'biodiversity', 'ecosystems',
    'natural', 'capital', 'water', 'scarcity', 'deforestation', 'land', 'use',
    'conservation', 'habitat', 'loss', 'extinction', 'species', 'ecosystem',
    'services', 'pollution', 'air', 'water', 'soil', 'noise', 'light', 'waste',
    'hazardous', 'materials', 'chemicals', 'pesticides', 'herbicides',
    'fertilizers', 'mining', 'drilling', 'extraction', 'refining', 'processing',
    'manufacturing', 'industrial', 'operations', 'transportation', 'logistics',
    'supply', 'chains', 'sourcing', 'procurement', 'distribution', 'sales',
    'marketing', 'advertising', 'branding', 'public', 'relations', 'reputation',
    'crisis', 'management', 'media', 'relations', 'stakeholder', 'engagement',
    'investor', 'relations', 'shareholder', 'activism', 'proxy', 'voting',
    'corporate', 'governance', 'board', 'diversity', 'independence', 'structure',
    'committees', 'executive', 'compensation', 'say', 'pay', 'clawbacks',
    'succession', 'planning', 'auditor', 'independence', 'internal', 'controls',
    'risk', 'oversight', 'ethics', 'compliance', 'whistleblower', 'policies',
    'conflicts', 'interest', 'anti', 'bribery', 'corruption', 'sanctions',
    'export', 'controls', 'trade', 'compliance', 'money', 'laundering',
    'terrorist', 'financing', 'data', 'privacy', 'cybersecurity', 'breach',
    'incident', 'response', 'notification', 'data', 'protection', 'regulations',
    'GDPR', 'CCPA', 'HIPAA', 'PII', 'PHI', 'encryption', 'tokenization',
    'anonymization', 'pseudonymization', 'access', 'control', 'authentication',
    'authorization', 'security', 'audits', 'vulnerability', 'assessments',
    'penetration', 'testing', 'incident', 'response', 'plans', 'disaster',
    'recovery', 'business', 'continuity', 'planning', 'crisis', 'communication',
    'public', 'relations', 'legal', 'counsel', 'forensic', 'investigations',
    'insurance', 'coverage', 'cyber', 'insurance', 'professional',
    'liability', 'errors', 'omissions', 'directors', 'officers', 'insurance',
    'employment', 'practices', 'liability', 'product', 'liability', 'general',
    'liability', 'property', 'insurance', 'business', 'interruption',
    'coverage', 'reinsurance', 'captives', 'self', 'insurance'
}


# Initialize lemmatizer and combined stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) | FINANCIAL_STOPWORDS

# Function Definition
def preprocess_for_topics(text):
    """
    Tokenizes, cleans, and lemmatizes text for topic modeling.
    Removes short words and custom financial stop words.
    
    Args:
        text (str): Input text.
        
    Returns:
        list: List of cleaned tokens.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize, remove stopwords, and filter short words
    tokens = [
        lemmatizer.lemmatize(t) for t in tokens
        if t not in stop_words and len(t) > 3
    ]
    return tokens

# Function Execution
corpus_df['tokens'] = corpus_df['text'].apply(preprocess_for_topics)
corpus_df['n_tokens'] = corpus_df['tokens'].apply(len)

print(f"\nAvg tokens per document after preprocessing: {corpus_df['n_tokens'].mean():.0f}")
print("\nSample of preprocessed tokens for the first document:")
print(corpus_df['tokens'].iloc[0][:20]) # Display first 20 tokens
```

<div class="alert alert-warning">
    <strong>Practitioner Warning: Financial Stop Words are Critical.</strong>
    Standard NLP stop-word lists don't include words like "material," "adverse," or "significant." These terms, while common, appear in virtually every 10-K risk factor and carry no discriminative power between topics. Without removing them, LDA produces topics dominated by generic filing words rather than substantive risk themes. The `FINANCIAL_STOPWORDS` list above is tuned for SEC filings; different document types (e.g., analyst reports, news) would require different domain-specific stop words. For you, the analyst, this custom list is key to extracting genuinely novel insights.
</div>

<div class="alert alert-info">
    <strong>Explanation of Execution:</strong>
    The code applies a robust preprocessing pipeline: lowercasing, punctuation removal, tokenization, lemmatization, and filtering against both generic English and specific financial stop words. The average token count provides a quick check on the effectiveness of the filtering. This meticulous cleaning ensures that the subsequent topic models are fed with text that is rich in meaningful vocabulary, enabling you to discover more precise and actionable risk themes.
</div>

## 3. Unveiling Latent Risk Themes: LDA Topic Modeling (Classical Approach)

With a clean corpus, your next objective is to discover the latent risk themes. Latent Dirichlet Allocation (LDA) is a powerful, classical probabilistic model that assumes each document is a mixture of 'K' topics, and each topic is a distribution over words. Your task is to apply LDA, determine the optimal number of topics (K) for interpretability, and then visualize these topics.

The core idea of LDA is that a document can be represented as a probabilistic distribution over topics, and a topic as a probabilistic distribution over words. Mathematically, the probability of a word $w$ occurring in a document $d$ can be expressed as:

$$P(w | d) = \sum_{k=1}^{K} P(w | z = k) \cdot P(z = k | d) = \sum_{k=1}^{K} \phi_{kw} \cdot \theta_{dk}$$

Where:
*   $P(w | z = k)$ is the probability of word $w$ given topic $k$. This is also denoted as $\phi_{kw}$, representing the **topic-word distribution** ($\phi_k \sim \text{Dir}(\eta)$).
*   $P(z = k | d)$ is the probability of topic $k$ given document $d$. This is also denoted as $\theta_{dk}$, representing the **document-topic distribution** ($\theta_d \sim \text{Dir}(\alpha)$).
*   $\alpha$ and $\eta$ are Dirichlet priors that control the sparsity of topic distributions in documents and word distributions in topics, respectively. A lower $\alpha$ means documents tend to concentrate on fewer topics.

For an investment professional, the $\theta_d$ vectors are critical; they represent a "risk profile fingerprint" for each company, showing its exposure to various risk themes.

To select the optimal number of topics $K$, we will use the **Coherence Score ($C_v$)**, which measures how semantically meaningful and interpretable the topics are. A higher coherence score indicates that the top words in a topic tend to co-occur together in the corpus.

The $C_v$ coherence score is calculated as:
$$C_v = \frac{1}{\binom{N}{2}} \sum_{i<j} \text{NPMI}(w_i, w_j)$$
Where NPMI is the Normalized Pointwise Mutual Information of co-occurring word pairs $(w_i, w_j)$ within a topic, and $N$ is the number of top words considered per topic. Our target is a $C_v > 0.40$ for well-formed financial topics.

```python
# Build dictionary and bag-of-words corpus
dictionary = corpora.Dictionary(corpus_df['tokens'])
# Filter out words that appear in less than 3 documents or more than 70% of documents
dictionary.filter_extremes(no_below=3, no_above=0.7)
bow_corpus = [dictionary.doc2bow(tokens) for tokens in corpus_df['tokens']]

# Train LDA for multiple K values and compute coherence
coherence_scores = {}
lda_models = {}
min_topics, max_topics = 5, 15 # Explore K from 5 to 15

print("Searching for optimal number of topics (K) using Coherence Score:")
for k in range(min_topics, max_topics + 1):
    lda_model = LdaMulticore(
        bow_corpus,
        num_topics=k,
        id2word=dictionary,
        passes=15, # Number of passes through the corpus during training
        workers=4, # Number of CPU cores to use
        random_state=42,
        alpha='asymmetric', # Emphasize some topics more than others
        eta='auto' # Automatically learn the eta parameter
    )
    
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=corpus_df['tokens'],
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    coherence_scores[k] = coherence_score
    lda_models[k] = lda_model
    print(f"K={k}: Coherence = {coherence_score:.4f}")

# Select best K (highest coherence)
best_k = max(coherence_scores, key=coherence_scores.get)
best_lda = lda_models[best_k]

print(f"\nBest K = {best_k} (Coherence = {coherence_scores[best_k]:.4f})")

# Plot coherence scores to visualize the elbow
plt.figure(figsize=(10, 6))
plt.plot(list(coherence_scores.keys()), list(coherence_scores.values()), marker='o')
plt.xlabel("Number of Topics (K)")
plt.ylabel("Coherence Score ($C_v$)")
plt.title("LDA Coherence Score ($C_v$) vs. Number of Topics (K)")
plt.xticks(list(coherence_scores.keys()))
plt.grid(True)
plt.savefig('lda_coherence_elbow_plot.png', dpi=150)
plt.show()

# Display top words for the selected best_lda model
print(f"\nTop 10 keywords for each of the {best_k} discovered topics (LDA):")
for idx, topic in best_lda.print_topics(num_words=10):
    print(f"Topic {idx}: {topic}")

# Visualize the topics interactively using pyLDAvis
# Note: pyLDAvis requires the model, corpus, and dictionary
vis = gensimvis.prepare(best_lda, bow_corpus, dictionary)
gensimvis.save_html(vis, 'lda_10k_topics.html')
print("\nInteractive LDA visualization saved to lda_10k_topics.html")
print("Open this HTML file in your browser to explore topics interactively.")
```

<div class="alert alert-info">
    <strong>Explanation of Execution:</strong>
    This section first builds a vocabulary (`dictionary`) and a Bag-of-Words (`bow_corpus`) representation of your preprocessed text. Then, it iteratively trains LDA models for a range of topic numbers ($K$) and calculates the $C_v$ coherence score for each. As an analyst, you'd look for an "elbow" in the coherence plot (though often it's less clear-cut) or simply choose the $K$ with the highest coherence, as we do here. The generated HTML file (`lda_10k_topics.html`) provides an interactive visualization. This tool lets you visually inspect topic distances, prevalences, and the relevance of words to each topic, which is critical for **interpreting and assigning human-readable labels** to these machine-discovered themes (e.g., "Topic 0: 'Regulatory/Legal', 'Topic 1: 'Cybersecurity'"). This forms the foundation of your risk surveillance.
</div>

## 4. Semantic Nuances: Embedding-Based Topic Clustering (Modern Approach)

While LDA is powerful, it's a bag-of-words model, meaning it doesn't inherently understand semantic similarity. For instance, "cyber attack" and "data breach" are semantically close but might be treated as distinct if they don't co-occur frequently. As an analyst, you want to ensure the most nuanced semantic relationships are captured. This section introduces a modern alternative: embedding-based topic clustering using Sentence-BERT to generate dense semantic vector representations for text, followed by K-Means clustering.

Sentence-BERT maps text to a dense vector space, where semantically similar sentences are close together. The similarity between two paragraph vectors, $\mathbf{v}_i$ and $\mathbf{v}_j$, can be computed using **cosine similarity**:

$$\text{sim}(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{||\mathbf{v}_i|| \cdot ||\mathbf{v}_j||}$$

where $||\mathbf{v}||$ denotes the Euclidean norm of vector $\mathbf{v}$. This metric ranges from -1 (opposite) to 1 (identical), with 0 indicating orthogonality.

After generating embeddings, K-Means clustering groups these vectors into 'K' clusters, representing our topics. We'll assess the quality of these clusters using the **Silhouette Score**, which measures how similar an object is to its own cluster compared to other clusters. A higher silhouette score ($> 0.20$ is a reasonable target) indicates well-separated and dense clusters.

```python
# Split documents into paragraphs for finer-grained topic discovery
# Paragraphs provide more specific context for embeddings
all_paragraphs = []
for idx, row in corpus_df.iterrows():
    # Split by double newline, filter for non-empty and reasonably long paragraphs
    paras = [p.strip() for p in row['text'].split('\n\n') if len(p.strip()) > 100]
    for p in paras:
        all_paragraphs.append({
            'ticker': row['ticker'],
            'year': row['year'],
            'paragraph': p,
            'doc_idx': idx # Keep track of original document for later aggregation
        })

para_df = pd.DataFrame(all_paragraphs)
print(f"Total paragraphs extracted for embedding: {len(para_df)}")

# Embed paragraphs with Sentence-BERT
sbert_model_name = 'all-MiniLM-L6-v2' # A good balance of speed and performance
sbert = SentenceTransformer(sbert_model_name)
print(f"\nGenerating embeddings using Sentence-BERT model: {sbert_model_name}...")
embeddings = sbert.encode(
    para_df['paragraph'].tolist(),
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True # Normalize embeddings for cosine similarity
)
print(f"Embedding shape: {embeddings.shape}")

# K-Means clustering on embeddings
best_sil_score = -1
best_k_emb = -1
best_km_model = None
kmeans_labels = None

min_clusters, max_clusters = 6, 15 # Explore K from 6 to 15 for embeddings
print(f"\nSearching for optimal number of clusters (K) using Silhouette Score:")
for k in range(min_clusters, max_clusters + 1):
    km = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init=10 to run KMeans 10 times with different centroids
    labels = km.fit_predict(embeddings)
    # Using a subset for silhouette_score calculation speeds it up
    silhouette_avg = silhouette_score(embeddings, labels, sample_size=5000, random_state=42)
    
    if silhouette_avg > best_sil_score:
        best_sil_score = silhouette_avg
        best_k_emb = k
        best_km_model = km
        kmeans_labels = labels
    print(f"K={k}: Silhouette Score = {silhouette_avg:.4f}")

para_df['cluster'] = kmeans_labels
print(f"\nBest K for embedding-based clustering = {best_k_emb} (Silhouette Score = {best_sil_score:.4f})")

# Display representative paragraphs for each cluster
print("\nRepresentative paragraphs for each embedding-based cluster (topic):")
for c in range(best_k_emb):
    mask = para_df['cluster'] == c
    if mask.sum() == 0:
        continue
    # Find paragraph nearest to centroid for interpretability
    centroid = best_km_model.cluster_centers_[c]
    dists = np.linalg.norm(embeddings[mask] - centroid, axis=1)
    nearest_idx_in_cluster = np.argmin(dists)
    
    # Get the global index for the paragraph
    global_idx_for_para = para_df[mask].index[nearest_idx_in_cluster]
    
    print(f"\nCluster {c} ({mask.sum()} paragraphs):")
    print(f"  Representative (first 200 chars): {para_df.loc[global_idx_for_para, 'paragraph'][:200]}...")

# For comparison, let's just show top words from a 'manual' interpretation
# This is a qualitative step based on reviewing representative paragraphs
print("\nQualitative comparison: Top words/themes from Embedding-based clusters (manual interpretation):")
# In a real-world scenario, you would manually review the representative paragraphs
# and assign human-readable labels to these clusters similar to LDA topics.
# For this specification, we'll illustrate with placeholder labels based on common themes.
embedding_topic_labels = {
    0: "Cybersecurity & Data Privacy (Embedding)",
    1: "Regulatory & Legal Compliance (Embedding)",
    2: "Economic & Market Volatility (Embedding)",
    3: "Supply Chain & Operations (Embedding)",
    4: "Technological & Innovation Risks (Embedding)",
    5: "ESG & Climate Change (Embedding)",
    # ... add more based on best_k_emb
}
for i in range(min(best_k_emb, len(embedding_topic_labels))): # Use min for safety
    print(f"Cluster {i}: {embedding_topic_labels.get(i, f'Unnamed Embedding Topic {i}')}")
```

<div class="alert alert-info">
    <strong>Key Insight: LDA vs. Embedding Clusters: Complementary Strengths.</strong>
    For an analyst, understanding both methods is key.
    <br><br>
    **LDA** excels at producing **interpretable word lists** per topic (e.g., "regulation, compliance, SEC, enforcement, penalty") and document-topic distributions ideal for human labeling. Its transparency supports regulatory and compliance applications.
    <br><br>
    **Embedding-based clustering** captures **semantic meaning** more effectively ("cybersecurity risk from nation-state actors" might cluster with "data breach liability" even without shared keywords). This reveals subtler connections but topics can be harder to label automatically as they don't explicitly provide keywords, relying instead on reviewing representative paragraphs.
    <br><br>
    **Best practice:** Use both. LDA gives you the topic vocabulary, while embeddings give you the semantic groupings. When they agree, you have high confidence. When they disagree, you've found interesting edge cases or nuances that warrant deeper investigation.
</div>

<div class="alert alert-info">
    <strong>Explanation of Execution:</strong>
    This section first breaks down full documents into paragraphs, as embeddings perform better on shorter, more semantically focused text. It then uses a pre-trained Sentence-BERT model to convert each paragraph into a numerical vector. K-Means clustering then groups these vectors into `best_k_emb` clusters. The Silhouette Score helps determine optimal `K`, and displaying representative paragraphs for each cluster is crucial for manual interpretation. You, the analyst, would review these to understand the semantic themes captured by each cluster and compare them qualitatively to your LDA topics, seeking areas of agreement and unique insights.
</div>

## 5. Portfolio Risk Mapping: Company-Topic Exposure Heatmap & Similarity Network (Cross-Company Insights)

Now that you have identified risk themes using LDA, the next step for an investment professional is to quantify each company's exposure to these themes. This is crucial for portfolio risk management and peer analysis. You'll achieve this in two ways:
1.  **Company-Topic Exposure Heatmap:** Visualizing the average topic distribution for each company. This immediately shows which companies concentrate on which risks, enabling cross-portfolio risk surveillance.
2.  **Company Similarity Network:** Building a network graph where companies are nodes and edges represent their textual risk profile similarity. This can reveal non-obvious peer groupings that transcend traditional sector classifications, a valuable input for relative valuation and risk benchmarking.

```python
# Aggregate LDA topic distributions to company level (average across years)
# First, get the topic distribution for each document
topic_matrix = []
for i, bow in enumerate(bow_corpus):
    # Get document-topic probabilities from the best LDA model
    topic_dist = dict(best_lda.get_document_topics(bow, minimum_probability=0.0))
    row_data = {f'Topic_{k}': topic_dist.get(k, 0.0) for k in range(best_k)}
    row_data['ticker'] = corpus_df.iloc[i]['ticker']
    row_data['year'] = corpus_df.iloc[i]['year']
    topic_matrix.append(row_data)

topic_df = pd.DataFrame(topic_matrix)

# Average topic exposure per company across years
company_topics_avg = topic_df.groupby('ticker').mean().reset_index()
# Drop 'year' column as it's been averaged
company_topics_avg = company_topics_avg.drop(columns=['year'], errors='ignore')


# Rename topics with human labels (after examining top words from LDA and pyLDAvis)
# These are illustrative labels; in practice, you'd carefully craft them.
topic_labels_map = {
    f'Topic_{i}': f'Topic_{i}: {label}' for i, label in enumerate([
        'Regulatory & Legal Compliance', 'Market & Economic Volatility', 'Cybersecurity & Data Privacy',
        'Competition & IP Protection', 'Supply Chain & Operations', 'ESG & Climate Risks',
        'Credit & Liquidity Risk', 'Talent & Workforce Issues', 'Inflation & Interest Rate', 'Geopolitical & Trade'
    ]) # Adjust the list length to match best_k
}
# Only map labels up to best_k
topic_labels_map = {f'Topic_{k}': f'Topic_{k}: {label}' 
                    for k, label in zip(range(best_k), [
                        'Regulatory & Legal Compliance', 'Market & Economic Volatility', 'Cybersecurity & Data Privacy',
                        'Competition & IP Protection', 'Supply Chain & Operations', 'ESG & Climate Risks',
                        'Credit & Liquidity Risk', 'Talent & Workforce Issues',
                        'Inflation & Interest Rate', 'Geopolitical & Trade', 'M&A & Integration' # Add more as needed up to best_k
                    ][:best_k])}

# Ensure all original topic columns exist in the map to avoid KeyErrors
# If best_k is different from the number of example labels, adjust dynamically
current_topic_cols = [col for col in company_topics_avg.columns if col.startswith('Topic_')]
company_topics_avg_labeled = company_topics_avg[current_topic_cols].rename(columns=topic_labels_map)
company_topics_avg_labeled.index = company_topics_avg['ticker'] # Set ticker as index for heatmap

# Heatmap Visualization
plt.figure(figsize=(14, 10))
sns.heatmap(company_topics_avg_labeled, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'Topic Exposure'})
plt.title('Company Risk Topic Exposure (from 10-K Risk Factors)', fontsize=16)
plt.ylabel('Company', fontsize=12)
plt.xlabel('Risk Theme', fontsize=12)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('company_topic_heatmap.png', dpi=150)
plt.show()

# Company Similarity Network
# Calculate cosine similarity between company topic vectors
# Use the un-renamed topic columns for similarity calculation for consistency
company_topic_vectors = company_topics_avg[current_topic_cols].values
company_names = company_topics_avg['ticker'].tolist()

sim_matrix = cosine_similarity(company_topic_vectors)
sim_df = pd.DataFrame(sim_matrix, index=company_names, columns=company_names)

# Build network graph (edges where similarity > threshold)
G = nx.Graph()
threshold = 0.75 # Only connect highly similar companies (adjust as needed)

for company in company_names:
    G.add_node(company)

for i in range(len(company_names)):
    for j in range(i + 1, len(company_names)): # Avoid self-loops and duplicate edges
        c1 = company_names[i]
        c2 = company_names[j]
        similarity = sim_df.loc[c1, c2]
        if similarity > threshold:
            G.add_edge(c1, c2, weight=similarity)

# Visualize the network
plt.figure(figsize=(12, 10))
# Using spring_layout for force-directed graph
pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42) # k adjusts optimal distance between nodes
node_size = 800
font_size = 8

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', edgecolors='black')
# Draw labels
nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold')
# Draw edges, scale width by similarity
edges = G.edges(data=True)
widths = [d['weight'] * 4 for u, v, d in edges] # Scale width for better visibility
nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color='gray')

plt.title(f'Company Risk Similarity Network (Cosine Similarity > {threshold})', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig('risk_similarity_network.png', dpi=150)
plt.show()

print("\nNon-obvious peer clusters (companies with similar risks, potentially different sectors):")
# Identify connected components (clusters) in the graph
for i, component in enumerate(nx.connected_components(G)):
    if len(component) > 1:
        print(f"Cluster {i+1}: {sorted(component)}")
```

<div class="alert alert-warning">
    <strong>Practitioner Warning: 10-K Risk Factor Language is Heavily Boilerplate.</strong>
    Many risk factors are near-identical across companies because corporate attorneys use template language. Before concluding that two companies have similar risk profiles, verify that the similarity reflects genuine shared risks rather than shared legal templates. The domain-specific stop words (Step 1) and the year-over-year drift analysis (Step 6) help mitigate this: boilerplate is stable across years, while genuine risk changes show up as drift. As an analyst, you should use these tools to discern true signals from noise.
</div>

<div class="alert alert-info">
    <strong>Explanation of Execution:</strong>
    This section first averages LDA topic probabilities for each company across both years to get a consolidated risk profile. The **heatmap** then visually represents this exposure, allowing you, the analyst, to quickly identify which companies are most sensitive to specific risk categories (e.g., "Tech Co. A has high Cybersecurity exposure"). The **similarity network** takes this further by computing cosine similarity between these risk profiles. By setting a similarity threshold, you can identify "risk-based" peer groups (connected components) that might not align with traditional GICS sectors. This is invaluable for finding overlooked peers for valuation, or identifying systemic vulnerabilities across your portfolio. For example, two companies from different sectors showing high similarity in "Supply Chain" risk might warrant a deeper dive into their shared global sourcing strategies.
</div>

## 6. Early Warning Signals: Detecting Year-over-Year Topic Drift (Dynamic Risk Monitoring)

A static view of risk is insufficient. For an equity analyst, detecting changes in a company's risk profile year-over-year can be an early warning sign of strategic shifts, emerging threats, or significant events (e.g., M&A, new regulations). You will quantify this shift using **Jensen-Shannon Divergence (JSD)**, which measures the dissimilarity between two probability distributions.

The JSD for two probability distributions $P$ and $Q$ is defined as:
$$\text{JSD}(P||Q) = \sqrt{\frac{D_{KL}(P||M) + D_{KL}(Q||M)}{2}}$$
Where $M = \frac{1}{2}(P+Q)$ is the midpoint distribution, and $D_{KL}$ is the Kullback-Leibler (KL) divergence.

The JSD value ranges from 0 to 1, where 0 indicates identical distributions and 1 indicates completely different distributions. For company topic vectors, a JSD $> 0.25$ is a useful threshold for flagging a significant thematic shift warranting analyst review.

```python
# Function Definition
def compute_topic_drift(topic_df, ticker, year1, year2, topic_labels_map):
    """
    Compute Jensen-Shannon divergence between a company's
    topic distribution in two consecutive years, and the per-topic changes.
    
    Args:
        topic_df (pd.DataFrame): DataFrame with 'ticker', 'year', and 'Topic_X' columns.
        ticker (str): The company ticker.
        year1 (int): The first year for comparison (e.g., 2023).
        year2 (int): The second year for comparison (e.g., 2024).
        topic_labels_map (dict): Mapping from 'Topic_X' to human-readable labels.
        
    Returns:
        dict: A dictionary containing ticker, JSD, and per-topic deltas, or None if data is missing.
    """
    # Get topic columns, mapping them to human-readable labels for output
    topic_cols = [col for col in topic_df.columns if col.startswith('Topic_')]
    
    t1 = topic_df[(topic_df['ticker'] == ticker) & (topic_df['year'] == year1)]
    t2 = topic_df[(topic_df['ticker'] == ticker) & (topic_df['year'] == year2)]

    if len(t1) == 0 or len(t2) == 0:
        return None

    # Extract topic distributions (ensure they are 1D arrays)
    p = t1[topic_cols].values[0]
    q = t2[topic_cols].values[0]

    # Ensure valid probability distributions (sum to 1, no zeros)
    p = np.maximum(p, 1e-10) # Avoid log(0)
    q = np.maximum(q, 1e-10) # Avoid log(0)
    p = p / p.sum()
    q = q / q.sum()
    
    # Compute JSD
    jsd = jensenshannon(p, q)

    # Compute per-topic changes (delta = Year2 - Year1 prominence)
    deltas = {topic_labels_map.get(col, col): (t2[col].values[0] - t1[col].values[0]) for col in topic_cols}
    
    return {'ticker': ticker, 'jsd': jsd, **deltas}

# Function Execution: Compute drift for all companies
drift_results = []
year_1 = 2023
year_2 = 2024

# Re-create the full topic_labels_map for consistent reporting
full_topic_labels_map = {
    f'Topic_{i}': f'Topic_{i}: {label}' for i, label in enumerate([
        'Regulatory & Legal Compliance', 'Market & Economic Volatility', 'Cybersecurity & Data Privacy',
        'Competition & IP Protection', 'Supply Chain & Operations', 'ESG & Climate Risks',
        'Credit & Liquidity Risk', 'Talent & Workforce Issues',
        'Inflation & Interest Rate', 'Geopolitical & Trade', 'M&A & Integration'
    ])
}
# Only include labels up to best_k
full_topic_labels_map = {f'Topic_{k}': f'Topic_{k}: {label}' 
                         for k, label in zip(range(best_k), [
                            'Regulatory & Legal Compliance', 'Market & Economic Volatility', 'Cybersecurity & Data Privacy',
                            'Competition & IP Protection', 'Supply Chain & Operations', 'ESG & Climate Risks',
                            'Credit & Liquidity Risk', 'Talent & Workforce Issues',
                            'Inflation & Interest Rate', 'Geopolitical & Trade', 'M&A & Integration' # Max 11 labels for example
                         ][:best_k])}


for ticker in corpus_df['ticker'].unique():
    result = compute_topic_drift(topic_df, ticker, year_1, year_2, full_topic_labels_map)
    if result:
        drift_results.append(result)

drift_df = pd.DataFrame(drift_results).sort_values('jsd', ascending=False).round(4)

print(f"Companies with highest topic drift (FY{year_1} -> FY{year_2}):")
print(drift_df[['ticker', 'jsd']].head(10).to_string(index=False))

# Visualize topic drift for the top-drifting company
if not drift_df.empty:
    top_drifter_ticker = drift_df.iloc[0]['ticker']
    top_drifter_data = drift_df[drift_df['ticker'] == top_drifter_ticker].iloc[0]
    
    # Filter for topic delta columns
    delta_cols = [col for col in drift_df.columns if '_delta' not in col and col != 'ticker' and col != 'jsd']
    topic_deltas = top_drifter_data[delta_cols].sort_values(ascending=False)

    plt.figure(figsize=(12, 7))
    topic_deltas.plot(kind='bar', color=['red' if x < 0 else 'green' for x in topic_deltas.values])
    plt.title(f'Topic Prominence Change for {top_drifter_ticker} (FY{year_1} vs. FY{year_2})', fontsize=16)
    plt.xlabel('Risk Topic', fontsize=12)
    plt.ylabel('Change in Prominence (Year2 - Year1)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{top_drifter_ticker}_topic_drift_bar_chart.png', dpi=150)
    plt.show()
```

<div class="alert alert-info">
    <strong>Financial Interpretation: Jensen-Shannon Divergence (JSD).</strong>
    A high JSD value between a company's topic distributions across consecutive years means its risk landscape has changed materially. This can signal:
    <ul>
        <li>**Strategic Pivots:** The company is shifting its business focus, leading to new categories of risk.</li>
        <li>**Emerging Threats:** New industry-wide or company-specific risks are gaining prominence (e.g., "AI regulation" in 2023).</li>
        <li>**Regulatory/Legal Changes:** New compliance burdens or ongoing litigation.</li>
        <li>**M&A Activity:** Integration or divestiture risks.</li>
    </ul>
    Conversely, a low JSD indicates stable risk disclosures. This often correlates with stable business performance, a finding highlighted by S&P Global research ("No News Is Good News," Zhao, 2021). As an analyst, you would flag companies with JSD $> 0.25$ for deeper qualitative due diligence, as these are your potential early warning signals.
</div>

<div class="alert alert-info">
    <strong>Explanation of Execution:</strong>
    The `compute_topic_drift` function calculates the JSD between a company's topic distribution vectors for two specified years. It also computes the individual change in prominence for each topic. By running this for all companies and sorting by JSD, you can quickly identify the companies undergoing the most significant shifts in their disclosed risk profiles. The bar chart for the top-drifting company visually highlights *which specific topics* are increasing or decreasing in prominence. This allows you, the analyst, to pinpoint the exact areas of change that require your investigative attention, providing critical dynamic risk monitoring.
</div>

## 7. Actionable Intelligence: Comparing Models & Summarizing Insights (Strategic Decision Support)

As a CFA Charterholder, your role culminates in synthesizing these quantitative insights into actionable intelligence for your portfolio managers. This final section encourages a critical review of the models, a summary of key findings, and reflections on how this workflow empowers better investment decisions.

### 7.1 Qualitative Comparison: LDA vs. Embedding-Based Topics

Review the top keywords from your LDA model and the representative paragraphs from your embedding-based clustering.
*   **LDA Topics:** Focus on keyword lists, their coherence, and your assigned human-readable labels. How intuitive are they?
*   **Embedding-Based Topics:** Consider the semantic nuances captured by representative paragraphs. Do they reveal subtler relationships or different facets of risk compared to LDA?

This comparison helps you understand the strengths and weaknesses of each approach and provides a more robust, multi-faceted understanding of the risk landscape.

```python
print("--- Qualitative Comparison: LDA vs. Embedding-Based Topics ---")

print("\nLDA Topics (Top words for Best K):")
for idx, topic in best_lda.print_topics(num_words=10):
    print(f"Topic {idx}: {topic}")

print("\nEmbedding-Based Clusters (Representative Paragraphs & Interpreted Themes):")
# Re-iterating representative paragraphs for direct comparison
for c in range(best_k_emb):
    mask = para_df['cluster'] == c
    if mask.sum() == 0:
        continue
    centroid = best_km_model.cluster_centers_[c]
    dists = np.linalg.norm(embeddings[mask] - centroid, axis=1)
    nearest_idx_in_cluster = np.argmin(dists)
    global_idx_for_para = para_df[mask].index[nearest_idx_in_cluster]
    
    # Use the example labels if available
    interpreted_label = embedding_topic_labels.get(c, f'Unnamed Embedding Topic {c}')
    print(f"\nCluster {c} ({interpreted_label}):")
    print(f"  Representative (first 200 chars): {para_df.loc[global_idx_for_para, 'paragraph'][:200]}...")

print("\nDiscussion points:")
print("- Are there common themes identified by both models?")
print("- Do the embedding clusters reveal more nuanced or semantically similar risks that LDA might miss?")
print("- Which approach provides more easily interpretable topics for your stakeholders?")
```

<div class="alert alert-info">
    <strong>Explanation of Execution:</strong>
    This final code cell simply re-displays the key outputs from the LDA and embedding models side-by-side. The value here is not in the code itself, but in your expert analysis. As a CFA Charterholder, you are trained to synthesize information. You would reflect on how LDA provides clear keyword-based topics, while embedding-based clustering surfaces semantic groupings. For example, if LDA identifies a 'Cybersecurity' topic, embedding clusters might further differentiate between 'Cybersecurity Incident Response' and 'Data Privacy Regulation', showing a more granular view. This comparison allows you to present a comprehensive and robust view of risk to your portfolio managers, acknowledging both traditional and modern NLP strengths.
</div>

### 7.2 Synthesizing Key Insights for Investment Decisions

Based on the entire workflow, you can now provide a concise summary of actionable intelligence:

*   **Most Prevalent Risks:** Identify the dominant risk themes across your portfolio, perhaps using the aggregated company-topic heatmap. Are there any systemic risks emerging?
*   **Cross-Sector Peer Groupings:** Highlight companies that, despite traditional sector differences, share similar risk profiles as revealed by the network graph. This can inform new research avenues or challenge existing assumptions about peer groups.
*   **Companies with Significant Risk Profile Drift:** Emphasize the companies flagged by JSD. These are prime candidates for immediate qualitative review, as their evolving risk narratives could signal impending operational or strategic changes relevant to your investment thesis.
*   **Emerging Themes:** Use the topic lists (especially from the pyLDAvis visualization) to spot new risk categories that might be gaining traction across filings, signaling future market trends.

This automated topic modeling pipeline empowers you, the investment professional, to efficiently monitor and analyze vast amounts of unstructured text data, transforming it into clear, data-driven insights that inform critical investment and risk management decisions for Alpha Capital Management.
