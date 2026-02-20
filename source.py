import pandas as pd
import numpy as np
import os
import re
import warnings
import shutil # For cleanup in example

# NLTK imports and downloads
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Gensim imports
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel

# Sentence-Transformers and Sklearn for embeddings and clustering
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Visualization libraries
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Scipy for divergence calculation
from scipy.spatial.distance import jensenshannon

# SEC Edgar Downloader
from sec_edgar_downloader import Downloader

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Global Configuration / Constants ---
# Define custom financial stop words - these are common in 10-Ks but not informative for topics
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

# --- NLTK Setup ---
def _setup_nltk_resources():
    """Downloads necessary NLTK data."""
    nltk_packages = ['punkt', 'stopwords', 'wordnet']
    for package in nltk_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except nltk.downloader.DownloadError:
            print(f"Downloading NLTK package: {package}...")
            nltk.download(package, quiet=True)
    
    # Original notebook had 'punkt_tab', which is not a standard NLTK download.
    # Keeping the check but commenting out download as it will likely fail.
    # If this is a custom resource, it needs to be made available externally.
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except (nltk.downloader.DownloadError, LookupError):
        print("Warning: 'punkt_tab' NLTK resource not found. This might be a non-standard or custom resource not available via `nltk.download`.")

_setup_nltk_resources() # Call once at module load time

# Initialize lemmatizer and combined stopwords globally after downloads
lemmatizer = WordNetLemmatizer()
combined_stopwords = set(stopwords.words('english')) | FINANCIAL_STOPWORDS

# --- Data Acquisition Functions ---
def initialize_sec_downloader(company_name: str, email: str) -> Downloader:
    """
    Initializes the SEC Edgar Downloader.
    """
    if not company_name or not email:
        raise ValueError("Company name and email are required for SEC Downloader as per SEC Fair Access Policy.")
    return Downloader(company_name, email)

def download_10k_filings(
    downloader: Downloader,
    tickers: list[str],
    start_year: int,
    end_year: int,
    output_dir: str = 'sec-edgar-filings'
):
    """
    Downloads 10-K filings for specified tickers and year range.
    Filings are saved to output_dir/{ticker}/10-K/{accession_number}/full-submission.txt
    """
    print(f"Starting 10-K download for tickers: {tickers} from {start_year} to {end_year}...")
    for ticker in tickers:
        print(f"  Downloading filings for {ticker}...")
        try:
            downloader.get(
                "10-K",
                ticker,
                after=f"{start_year}-01-01",
                before=f"{end_year}-12-31",
                download_folder=output_dir
            )
            print(f"  Finished downloading for {ticker}.")
        except Exception as e:
            print(f"  Error downloading for {ticker}: {e}")
    print("10-K download complete.")

def load_and_extract_risk_factors(base_download_dir: str = 'sec-edgar-filings') -> pd.DataFrame:
    """
    Loads 10-K filings from the sec-edgar-downloader directory structure,
    extracts 'Item 1A. Risk Factors', cleans HTML, and returns a DataFrame.

    Structure assumed: base_download_dir/{TICKER}/10-K/{ACCESSION_NUMBER}/full-submission.txt
    """
    documents = []
    if not os.path.exists(base_download_dir):
        print(f"Error: Download directory '{base_download_dir}' not found. Please run download_10k_filings first.")
        return pd.DataFrame()

    print(f"Loading and extracting risk factors from '{base_download_dir}'...")
    # Walk through the downloader's directory tree
    for root, _, files in os.walk(base_download_dir):
        for filename in files:
            if filename == 'full-submission.txt':
                try:
                    filepath = os.path.join(root, filename)

                    # Extract Ticker from the path structure: base_download_dir/TICKER/10-K/...
                    path_parts = root.split(os.sep)
                    try:
                        idx_10k = path_parts.index('10-K')
                        ticker = path_parts[idx_10k - 1]
                    except ValueError:
                        print(f"Warning: '10-K' not found in path for {filepath}. Skipping.")
                        continue

                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_content = f.read()

                    # Extract Year from the filing header (CONFORMED PERIOD OF REPORT)
                    year_match = re.search(r"CONFORMED PERIOD OF REPORT:\s+(\d{4})", raw_content)
                    year = int(year_match.group(1)) if year_match else "Unknown"

                    # BASIC EXTRACTION: Item 1A (Risk Factors)
                    pattern = re.compile(r"Item\s+1A\.?\s+Risk\s+Factors(.*?)(?=Item\s+(?:1B|2)\.?|\Z)", re.DOTALL | re.IGNORECASE)
                    match = pattern.search(raw_content)

                    if match:
                        risk_text = match.group(1).strip()
                        # Clean up HTML tags and other common text artifacts
                        risk_text = re.sub('<[^<]+?>', '', risk_text) # Remove HTML tags
                        risk_text = re.sub(r'&#\d+;', '', risk_text) # Remove HTML entities like &#160;
                        risk_text = re.sub(r'&nbsp;', ' ', risk_text, flags=re.IGNORECASE) # Replace common HTML entities
                        risk_text = re.sub(r' +', ' ', risk_text).strip() # Consolidate multiple spaces
                        risk_text = re.sub(r'\n\s*\n', '\n\n', risk_text) # Consolidate multiple newlines
                        
                        documents.append({
                            'ticker': ticker,
                            'year': year,
                            'text': risk_text,
                            'word_count': len(risk_text.split())
                        })
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    corpus_df = pd.DataFrame(documents)
    corpus_df = corpus_df[corpus_df['year'] != "Unknown"].reset_index(drop=True)
    corpus_df['year'] = corpus_df['year'].astype(int)

    print(f"Loaded {len(corpus_df)} documents from {corpus_df['ticker'].nunique()} companies after extraction.")
    return corpus_df

# --- Text Preprocessing Functions ---
def preprocess_text_for_topic_modeling(text: str) -> list[str]:
    """
    Tokenizes, cleans, and lemmatizes text for topic modeling.
    Removes short words and custom financial stop words.

    Args:
        text (str): Input text.

    Returns:
        list: List of cleaned tokens.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t) for t in tokens
        if t not in combined_stopwords and len(t) > 3
    ]
    return tokens

def add_preprocessed_tokens(corpus_df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Applies text preprocessing to a DataFrame column and adds tokenized column.
    """
    print("Preprocessing text for topic modeling...")
    corpus_df['tokens'] = corpus_df[text_column].apply(preprocess_text_for_topic_modeling)
    corpus_df['n_tokens'] = corpus_df['tokens'].apply(len)
    print(f"Avg tokens per document after preprocessing: {corpus_df['n_tokens'].mean():.0f}")
    return corpus_df

# --- LDA Topic Modeling Functions ---
def train_lda_model(
    corpus_df: pd.DataFrame,
    min_topics: int = 2,
    max_topics: int = 7,
    text_token_column: str = 'tokens',
    num_passes: int = 15,
    num_workers: int = 4,
    random_seed: int = 42
) -> tuple[LdaMulticore, corpora.Dictionary, list, dict, int]:
    """
    Builds dictionary, BOW corpus, trains LDA models for various K,
    computes coherence scores, and returns the best model.

    Returns:
        tuple: (best_lda_model, dictionary, bow_corpus, coherence_scores, best_k)
    """
    print(f"\nBuilding dictionary and bag-of-words corpus...")
    dictionary = corpora.Dictionary(corpus_df[text_token_column])
    dictionary.filter_extremes(no_below=3, no_above=0.7)
    bow_corpus = [dictionary.doc2bow(tokens) for tokens in corpus_df[text_token_column]]
    print(f"Dictionary size: {len(dictionary)} unique tokens.")
    print(f"BOW corpus created for {len(bow_corpus)} documents.")

    print(f"\nSearching for optimal number of topics (K) using Coherence Score (K from {min_topics} to {max_topics}):")
    coherence_scores = {}
    lda_models = {}

    for k in range(min_topics, max_topics + 1):
        print(f"  Training LDA model for K={k} topics...")
        lda_model = LdaMulticore(
            bow_corpus,
            num_topics=k,
            id2word=dictionary,
            passes=num_passes,
            workers=num_workers,
            random_state=random_seed,
            alpha='asymmetric',
            eta='auto'
        )

        coherence_model = CoherenceModel(
            model=lda_model,
            texts=corpus_df[text_token_column],
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        coherence_scores[k] = coherence_score
        lda_models[k] = lda_model
        print(f"  K={k}: Coherence = {coherence_score:.4f}")

    best_k = max(coherence_scores, key=coherence_scores.get)
    best_lda = lda_models[best_k]
    print(f"\nBest K = {best_k} (Coherence = {coherence_scores[best_k]:.4f})")

    return best_lda, dictionary, bow_corpus, coherence_scores, best_k

def visualize_lda_topics(
    lda_model: LdaMulticore,
    bow_corpus: list,
    dictionary: corpora.Dictionary,
    output_html_path: str = 'lda_10k_topics.html',
    plot_coherence_path: str = 'lda_coherence_elbow_plot.png',
    coherence_scores: dict = None
):
    """
    Generates interactive LDA visualization and plots coherence scores.
    """
    print("\nDisplaying top keywords for discovered LDA topics:")
    for idx, topic in lda_model.print_topics(num_words=10):
        print(f"Topic {idx}: {topic}")

    if coherence_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(list(coherence_scores.keys()), list(coherence_scores.values()), marker='o')
        plt.xlabel("Number of Topics (K)")
        plt.ylabel("Coherence Score ($C_v$)")
        plt.title("LDA Coherence Score ($C_v$) vs. Number of Topics (K)")
        plt.xticks(list(coherence_scores.keys()))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_coherence_path, dpi=150)
        plt.show()
        print(f"Coherence plot saved to {plot_coherence_path}")

    print(f"Generating interactive LDA visualization to {output_html_path}...")
    vis = gensimvis.prepare(lda_model, bow_corpus, dictionary)
    pyLDAvis.save_html(vis, output_html_path)
    print(f"Interactive LDA visualization saved. Open '{output_html_path}' in your browser.")


# --- Sentence Embedding & Clustering Functions ---
def split_into_paragraphs(corpus_df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Splits documents in corpus_df into paragraphs and creates a new DataFrame.
    """
    print("\nSplitting documents into paragraphs for embedding...")
    all_paragraphs = []
    for idx, row in corpus_df.iterrows():
        # Split by double newline, filter for non-empty and reasonably long paragraphs
        # Also, clean up multiple spaces/newlines within paragraphs before splitting
        cleaned_text = re.sub(r' +', ' ', row[text_column]).strip()
        paras = [p.strip() for p in cleaned_text.split('\n\n') if len(p.strip()) > 100]
        for p in paras:
            all_paragraphs.append({
                'ticker': row['ticker'],
                'year': row['year'],
                'paragraph': p,
                'doc_idx': idx # Keep track of original document for later aggregation
            })
    para_df = pd.DataFrame(all_paragraphs)
    print(f"Total paragraphs extracted for embedding: {len(para_df)}")
    return para_df

def embed_paragraphs(
    para_df: pd.DataFrame,
    paragraph_column: str = 'paragraph',
    sbert_model_name: str = 'all-MiniLM-L6-v2'
) -> tuple[np.ndarray, SentenceTransformer]:
    """
    Embeds paragraphs using Sentence-BERT.

    Returns:
        tuple: (embeddings_array, sbert_model_instance)
    """
    if para_df.empty:
        print("No paragraphs to embed. Returning empty array.")
        return np.array([]), None

    print(f"\nGenerating embeddings using Sentence-BERT model: {sbert_model_name}...")
    sbert_model = SentenceTransformer(sbert_model_name)
    embeddings = sbert_model.encode(
        para_df[paragraph_column].tolist(),
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True
    )
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings, sbert_model

def cluster_embeddings(
    embeddings: np.ndarray,
    min_clusters: int = 6,
    max_clusters: int = 15,
    random_seed: int = 42
) -> tuple[np.ndarray, KMeans, int, float]:
    """
    Performs K-Means clustering on embeddings and selects the best K
    based on Silhouette Score.

    Returns:
        tuple: (cluster_labels, best_kmeans_model, best_k, best_silhouette_score)
    """
    if embeddings.shape[0] < max_clusters:
        print(f"Warning: Not enough samples ({embeddings.shape[0]}) for max_clusters ({max_clusters}). Adjusting max_clusters.")
        max_clusters = embeddings.shape[0] // 2 if embeddings.shape[0] > 1 else 1 # Simple heuristic
        if max_clusters < min_clusters:
            min_clusters = max_clusters

    if embeddings.shape[0] < 2 or min_clusters < 1:
        print("Not enough data points or invalid cluster range for clustering. Returning dummy values.")
        return np.array([0]*embeddings.shape[0]), None, 1, -1.0

    print(f"\nSearching for optimal number of clusters (K) using Silhouette Score (K from {min_clusters} to {max_clusters}):")
    best_sil_score = -1
    best_k_emb = min_clusters # Initialize with min_clusters
    best_km_model = None
    kmeans_labels = None

    for k in range(min_clusters, max_clusters + 1):
        if k < 2 or k >= embeddings.shape[0]: # Silhouette score requires at least 2 clusters and k < n_samples
            continue
        
        km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = km.fit_predict(embeddings)
        
        # Use a subset for silhouette_score calculation to speed it up on large datasets
        sample_size = min(len(embeddings), 5000)
        silhouette_avg = silhouette_score(embeddings, labels, sample_size=sample_size, random_state=random_seed)

        if silhouette_avg > best_sil_score:
            best_sil_score = silhouette_avg
            best_k_emb = k
            best_km_model = km
            kmeans_labels = labels
        print(f"  K={k}: Silhouette Score = {silhouette_avg:.4f}")

    # Fallback if no valid silhouette scores were found (e.g., all clusters were too small or k=1)
    if best_km_model is None:
        best_k_emb = min_clusters if min_clusters > 1 else 2
        best_km_model = KMeans(n_clusters=best_k_emb, random_state=random_seed, n_init=10)
        kmeans_labels = best_km_model.fit_predict(embeddings)
        print(f"  No optimal K found with Silhouette. Defaulting to K={best_k_emb}.")
        
    print(f"\nBest K for embedding-based clustering = {best_k_emb} (Silhouette Score = {best_sil_score:.4f})")
    return kmeans_labels, best_km_model, best_k_emb, best_sil_score

def display_embedding_clusters(
    para_df: pd.DataFrame,
    embeddings: np.ndarray,
    km_model: KMeans,
    best_k_emb: int,
    num_samples_per_cluster: int = 1
):
    """
    Displays representative paragraphs for each embedding-based cluster.
    """
    if para_df.empty or km_model is None or embeddings.shape[0] == 0:
        print("No embedding clusters to display.")
        return

    print("\nRepresentative paragraphs for each embedding-based cluster (topic):")
    para_df_with_clusters = para_df.copy()
    para_df_with_clusters['cluster'] = km_model.labels_ # Assign labels to the df

    for c in range(best_k_emb):
        mask = para_df_with_clusters['cluster'] == c
        cluster_paragraphs = para_df_with_clusters[mask]
        
        if cluster_paragraphs.empty:
            print(f"  Cluster {c} is empty.")
            continue

        # Find paragraph(s) nearest to centroid for interpretability
        centroid = km_model.cluster_centers_[c]
        cluster_embeddings_subset = embeddings[mask]
        
        dists = np.linalg.norm(cluster_embeddings_subset - centroid, axis=1)
        
        # Get indices of paragraphs sorted by distance to centroid
        sorted_indices_in_cluster = np.argsort(dists)

        print(f"\nCluster {c} ({len(cluster_paragraphs)} paragraphs):")
        for i in range(min(num_samples_per_cluster, len(cluster_paragraphs))):
            nearest_idx_in_cluster = sorted_indices_in_cluster[i]
            global_idx_for_para = cluster_paragraphs.index[nearest_idx_in_cluster]
            print(f"  Sample {i+1} (first 200 chars): {para_df_with_clusters.loc[global_idx_for_para, 'paragraph'][:200]}...")


# --- Analysis & Visualization Functions ---
def calculate_company_topic_exposure(
    corpus_df: pd.DataFrame,
    best_lda_model: LdaMulticore,
    best_k: int,
    bow_corpus: list,
    topic_labels_map: dict = None
) -> pd.DataFrame:
    """
    Aggregates LDA topic distributions to company level (average across years).
    """
    print("\nCalculating company-level topic exposure...")
    topic_matrix = []
    for i, bow in enumerate(bow_corpus):
        topic_dist = dict(best_lda_model.get_document_topics(bow, minimum_probability=0.0))
        row_data = {f'Topic_{k}': topic_dist.get(k, 0.0) for k in range(best_k)}
        row_data['ticker'] = corpus_df.iloc[i]['ticker']
        row_data['year'] = corpus_df.iloc[i]['year']
        topic_matrix.append(row_data)

    topic_df = pd.DataFrame(topic_matrix)
    company_topics_avg = topic_df.groupby('ticker').mean().reset_index()
    company_topics_avg = company_topics_avg.drop(columns=['year'], errors='ignore')

    if topic_labels_map:
        # Create a mapping that only applies to existing topic columns
        current_topic_cols = [col for col in company_topics_avg.columns if col.startswith('Topic_')]
        filtered_topic_labels_map = {k: v for k, v in topic_labels_map.items() if k in current_topic_cols}
        company_topics_avg_labeled = company_topics_avg[current_topic_cols].rename(columns=filtered_topic_labels_map)
        company_topics_avg_labeled.index = company_topics_avg['ticker']
        return company_topics_avg_labeled
    else:
        company_topics_avg.index = company_topics_avg['ticker']
        return company_topics_avg.drop(columns=['ticker'])


def plot_company_topic_heatmap(
    company_topic_exposure_df: pd.DataFrame,
    output_path: str = 'company_topic_heatmap.png'
):
    """
    Plots a heatmap of company topic exposure.
    """
    if company_topic_exposure_df.empty:
        print("No company topic exposure data to plot heatmap.")
        return

    print(f"\nGenerating company topic exposure heatmap to {output_path}...")
    plt.figure(figsize=(max(10, len(company_topic_exposure_df.columns) * 1.2), max(8, len(company_topic_exposure_df) * 0.7)))
    sns.heatmap(company_topic_exposure_df, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'Topic Exposure'})
    plt.title('Company Risk Topic Exposure (from 10-K Risk Factors)', fontsize=16)
    plt.ylabel('Company', fontsize=12)
    plt.xlabel('Risk Theme', fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

def plot_company_similarity_network(
    company_topic_exposure_df: pd.DataFrame,
    threshold: float = 0.75,
    output_path: str = 'risk_similarity_network.png'
):
    """
    Builds and visualizes a network graph of company risk similarities.
    """
    company_topic_vectors = company_topic_exposure_df.values
    company_names = company_topic_exposure_df.index.tolist()

    if len(company_names) < 2:
        print("Not enough companies to build a similarity network.")
        return

    print(f"\nGenerating company risk similarity network (threshold > {threshold}) to {output_path}...")
    sim_matrix = cosine_similarity(company_topic_vectors)
    sim_df = pd.DataFrame(sim_matrix, index=company_names, columns=company_names)

    G = nx.Graph()
    for company in company_names:
        G.add_node(company)

    for i in range(len(company_names)):
        for j in range(i + 1, len(company_names)):
            c1 = company_names[i]
            c2 = company_names[j]
            similarity = sim_df.loc[c1, c2]
            if similarity > threshold:
                G.add_edge(c1, c2, weight=similarity)

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    node_size = 800
    font_size = 8

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold')
    
    edges = G.edges(data=True)
    if edges:
        widths = [d['weight'] * 4 for u, v, d in edges]
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color='gray')

    plt.title(f'Company Risk Similarity Network (Cosine Similarity > {threshold})', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    print("\nNon-obvious peer clusters (companies with similar risks, potentially different sectors):")
    components_found = False
    for i, component in enumerate(nx.connected_components(G)):
        if len(component) > 1:
            print(f"  Cluster {i+1}: {sorted(component)}")
            components_found = True
    if not components_found:
        print("  No significant clusters found above the specified similarity threshold.")


def compute_topic_drift(
    topic_df: pd.DataFrame,
    ticker: str,
    year1: int,
    year2: int,
    topic_labels_map: dict
) -> dict | None:
    """
    Compute Jensen-Shannon divergence between a company's
    topic distribution in two consecutive years, and the per-topic changes.
    """
    topic_cols = [col for col in topic_df.columns if col.startswith('Topic_')]

    t1 = topic_df[(topic_df['ticker'] == ticker) & (topic_df['year'] == year1)]
    t2 = topic_df[(topic_df['ticker'] == ticker) & (topic_df['year'] == year2)]

    if t1.empty or t2.empty:
        # print(f"  Warning: No data for {ticker} in both {year1} and {year2}. Skipping drift calculation.")
        return None

    # Extract topic distributions (ensure they are 1D arrays)
    p = t1[topic_cols].values[0]
    q = t2[topic_cols].values[0]

    # Ensure valid probability distributions (sum to 1, no zeros)
    p = np.maximum(p, 1e-10) # Avoid log(0)
    q = np.maximum(q, 1e-10) # Avoid log(0)
    p = p / p.sum()
    q = q / q.sum()

    jsd = jensenshannon(p, q)

    # Compute per-topic changes (delta = Year2 - Year1 prominence)
    deltas = {f"{topic_labels_map.get(col, col)}_delta": (t2[col].values[0] - t1[col].values[0]) for col in topic_cols}

    return {'ticker': ticker, 'jsd': jsd, **deltas}

def compute_and_report_topic_drift_for_all_companies(
    corpus_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    year1: int,
    year2: int,
    topic_labels_map: dict
) -> pd.DataFrame:
    """
    Computes topic drift for all unique tickers in the corpus_df.
    """
    print(f"\nComputing topic drift for all companies (FY{year1} -> FY{year2}):")
    drift_results = []
    
    # Ensure topic_df has 'ticker' and 'year'
    if 'ticker' not in topic_df.columns or 'year' not in topic_df.columns:
        print("Error: topic_df must contain 'ticker' and 'year' columns for drift calculation.")
        return pd.DataFrame()

    for ticker in corpus_df['ticker'].unique():
        result = compute_topic_drift(topic_df, ticker, year1, year2, topic_labels_map)
        if result:
            drift_results.append(result)

    drift_df = pd.DataFrame(drift_results).sort_values('jsd', ascending=False).round(4)
    if not drift_df.empty:
        print(f"Companies with highest topic drift (FY{year1} -> FY{year2}):")
        print(drift_df[['ticker', 'jsd']].head(10).to_string(index=False))
    else:
        print("No topic drift data to report.")
    return drift_df

def plot_topic_drift(
    drift_df: pd.DataFrame,
    year1: int,
    year2: int,
    output_path_prefix: str = 'topic_drift_bar_chart'
):
    """
    Visualizes topic drift for the top-drifting company.
    """
    if drift_df.empty:
        print("No topic drift data to plot.")
        return

    top_drifter_ticker = drift_df.iloc[0]['ticker']
    top_drifter_data = drift_df[drift_df['ticker'] == top_drifter_ticker].iloc[0]

    # Filter for topic delta columns
    delta_cols = [col for col in drift_df.columns if col.endswith('_delta')]
    topic_deltas = top_drifter_data[delta_cols]
    
    # Remove '_delta' suffix from column names for cleaner plot labels
    topic_deltas.index = topic_deltas.index.str.replace('_delta', '')
    topic_deltas = topic_deltas.sort_values(ascending=False)

    print(f"\nGenerating topic drift chart for top drifter {top_drifter_ticker} to {top_drifter_ticker}_{output_path_prefix}.png...")
    plt.figure(figsize=(12, 7))
    topic_deltas.plot(kind='bar', color=['red' if x < 0 else 'green' for x in topic_deltas.values])
    plt.title(f'Topic Prominence Change for {top_drifter_ticker} (FY{year1} vs. FY{year2})', fontsize=16)
    plt.xlabel('Risk Topic', fontsize=12)
    plt.ylabel('Change in Prominence (Year2 - Year1)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{top_drifter_ticker}_{output_path_prefix}.png', dpi=150)
    plt.show()

# --- Qualitative Comparison ---
def qualitative_comparison_report(
    best_lda_model: LdaMulticore,
    para_df_with_clusters: pd.DataFrame,
    best_km_model: KMeans,
    embeddings: np.ndarray,
    best_k_emb: int,
    embedding_topic_labels: dict = None,
    num_samples_per_cluster: int = 1
):
    """
    Provides a qualitative comparison between LDA and embedding-based topics.
    """
    print("\n" + "="*80)
    print("--- Qualitative Comparison: LDA vs. Embedding-Based Topics ---")
    print("="*80)

    print("\nLDA Topics (Top words for Best K):")
    for idx, topic in best_lda_model.print_topics(num_words=10):
        print(f"  Topic {idx}: {topic}")

    print("\nEmbedding-Based Clusters (Representative Paragraphs & Interpreted Themes):")
    if para_df_with_clusters.empty or best_km_model is None or embeddings.shape[0] == 0:
        print("  No embedding clusters to compare.")
    else:
        for c in range(best_k_emb):
            mask = para_df_with_clusters['cluster'] == c
            cluster_paragraphs = para_df_with_clusters[mask]

            if cluster_paragraphs.empty:
                print(f"  Cluster {c} is empty.")
                continue

            centroid = best_km_model.cluster_centers_[c]
            cluster_embeddings_subset = embeddings[mask]
            dists = np.linalg.norm(cluster_embeddings_subset - centroid, axis=1)
            sorted_indices_in_cluster = np.argsort(dists)

            interpreted_label = embedding_topic_labels.get(c, f'Unnamed Embedding Topic {c}') if embedding_topic_labels else f'Unnamed Embedding Topic {c}'
            print(f"\n  Cluster {c} ({interpreted_label}):")
            for i in range(min(num_samples_per_cluster, len(cluster_paragraphs))):
                nearest_idx_in_cluster = sorted_indices_in_cluster[i]
                global_idx_for_para = para_df_with_clusters.index[nearest_idx_in_cluster]
                print(f"    Representative (first 200 chars): {para_df_with_clusters.loc[global_idx_for_para, 'paragraph'][:200]}...")

    print("\nDiscussion points:")
    print("- Are there common themes identified by both models?")
    print("- Do the embedding clusters reveal more nuanced or semantically similar risks that LDA might miss?")
    print("- Which approach provides more easily interpretable topics for your stakeholders?")
    print("="*80)


# --- Main Orchestration Function ---
def analyze_risk_factors(
    company_name: str,
    email: str,
    tickers: list[str],
    start_year: int,
    end_year: int,
    sec_download_dir: str = 'sec-edgar-filings',
    lda_min_topics: int = 2,
    lda_max_topics: int = 7,
    sbert_model_name: str = 'all-MiniLM-L6-v2',
    embedding_min_clusters: int = 6,
    embedding_max_clusters: int = 15,
    similarity_threshold: float = 0.75,
    random_seed: int = 42
):
    """
    Orchestrates the entire risk factor analysis pipeline.
    """
    # 1. Setup and Data Acquisition
    downloader = initialize_sec_downloader(company_name, email)
    download_10k_filings(downloader, tickers, start_year, end_year, sec_download_dir)
    corpus_df = load_and_extract_risk_factors(sec_download_dir)

    if corpus_df.empty:
        print("No documents loaded. Exiting analysis.")
        return

    # 2. Text Preprocessing
    corpus_df = add_preprocessed_tokens(corpus_df)

    # 3. LDA Topic Modeling
    best_lda, dictionary, bow_corpus, coherence_scores, best_k_lda = train_lda_model(
        corpus_df,
        min_topics=lda_min_topics,
        max_topics=lda_max_topics,
        random_seed=random_seed
    )
    visualize_lda_topics(best_lda, bow_corpus, dictionary, coherence_scores=coherence_scores)
    
    # Dynamically create LDA topic labels for reporting.
    # Extend this list or provide a more robust mapping if 'best_k_lda' can be very high.
    lda_topic_base_labels = [
        'Regulatory & Legal Compliance', 'Market & Economic Volatility', 'Cybersecurity & Data Privacy',
        'Competition & IP Protection', 'Supply Chain & Operations', 'ESG & Climate Risks',
        'Credit & Liquidity Risk', 'Talent & Workforce Issues', 'Inflation & Interest Rate',
        'Geopolitical & Trade', 'M&A & Integration', 'Operational Efficiency', 'Technological Innovation',
        'Data Security & Privacy', 'Product Liability'
    ]
    topic_labels_map = {
        f'Topic_{k}': f'Topic_{k}: {label}'
        for k, label in zip(range(best_k_lda), lda_topic_base_labels[:best_k_lda])
    }
    # If best_k_lda exceeds base labels, use generic labels
    if best_k_lda > len(lda_topic_base_labels):
        for k in range(len(lda_topic_base_labels), best_k_lda):
            topic_labels_map[f'Topic_{k}'] = f'Topic_{k}: Generic Risk {k}'


    # 4. Sentence Embedding & Clustering
    para_df = split_into_paragraphs(corpus_df)
    embeddings, sbert_model = embed_paragraphs(para_df, sbert_model_name=sbert_model_name)
    
    kmeans_labels, best_km_model, best_k_emb, best_sil_score = cluster_embeddings(
        embeddings,
        min_clusters=embedding_min_clusters,
        max_clusters=embedding_max_clusters,
        random_seed=random_seed
    )

    if best_km_model is None: # Handle cases where clustering failed (e.g., too few data points)
        print("Embedding clustering failed or not enough data. Skipping related steps.")
        para_df_with_clusters = para_df.copy() # Just keep the original para_df
    else:
        para_df_with_clusters = para_df.copy()
        para_df_with_clusters['cluster'] = kmeans_labels
        display_embedding_clusters(para_df_with_clusters, embeddings, best_km_model, best_k_emb)

    # 5. Analysis & Visualization
    company_topics_avg_labeled = calculate_company_topic_exposure(
        corpus_df, best_lda, best_k_lda, bow_corpus, topic_labels_map
    )
    plot_company_topic_heatmap(company_topics_avg_labeled)
    plot_company_similarity_network(company_topics_avg_labeled, threshold=similarity_threshold)

    # Prepare topic_df for drift calculation (contains 'ticker', 'year', 'Topic_X' columns)
    topic_matrix_for_drift = []
    for i, bow in enumerate(bow_corpus):
        topic_dist = dict(best_lda.get_document_topics(bow, minimum_probability=0.0))
        row_data = {f'Topic_{k}': topic_dist.get(k, 0.0) for k in range(best_k_lda)}
        row_data['ticker'] = corpus_df.iloc[i]['ticker']
        row_data['year'] = corpus_df.iloc[i]['year']
        topic_matrix_for_drift.append(row_data)
    topic_df_for_drift = pd.DataFrame(topic_matrix_for_drift)

    drift_df = compute_and_report_topic_drift_for_all_companies(
        corpus_df,
        topic_df_for_drift,
        start_year,
        end_year,
        topic_labels_map
    )
    plot_topic_drift(drift_df, start_year, end_year)

    # 6. Qualitative Comparison
    # Example labels for embedding clusters - these would typically be assigned manually
    # after reviewing the representative paragraphs. Dynamic adjustment for best_k_emb.
    embedding_topic_base_labels = {
        0: "Cybersecurity & Data Privacy (Embedding)", 1: "Regulatory & Legal Compliance (Embedding)",
        2: "Economic & Market Volatility (Embedding)", 3: "Supply Chain & Operations (Embedding)",
        4: "Technological & Innovation Risks (Embedding)", 5: "ESG & Climate Change (Embedding)",
        6: "Product & Market Fit (Embedding)", 7: "M&A & Integration (Embedding)",
        8: "Talent & Workforce (Embedding)", 9: "Intellectual Property (Embedding)",
        10: "Geopolitical & Trade (Embedding)", 11: "Litigation & Legal Disputes (Embedding)",
        12: "Inflation & Interest Rates (Embedding)", 13: "Competition & Pricing (Embedding)",
        14: "Capital Allocation & Debt (Embedding)"
    }
    embedding_topic_labels_actual = {
        i: embedding_topic_base_labels.get(i, f'Unnamed Embedding Topic {i}') for i in range(best_k_emb)
    }

    qualitative_comparison_report(
        best_lda, para_df_with_clusters, best_km_model, embeddings, best_k_emb, embedding_topic_labels_actual
    )

# --- Entry Point for the Application ---
if __name__ == "__main__":
    # Example Configuration for demonstration
    MY_COMPANY_NAME = "My Test Company" # Replace with your company name
    MY_EMAIL = "my.email@example.com"    # Replace with your email (SEC requirement)
    TARGET_TICKERS = ["AAPL", "MSFT", "GOOGL"]
    REPORTING_START_YEAR = 2023 # For historical context
    REPORTING_END_YEAR = 2024   # For most recent filings

    # Clean up old files if they exist for a fresh run
    print("Performing cleanup of previous run artifacts...")
    if os.path.exists('sec-edgar-filings'):
        shutil.rmtree('sec-edgar-filings')
    for f in ['lda_coherence_elbow_plot.png', 'lda_10k_topics.html',
              'company_topic_heatmap.png', 'risk_similarity_network.png']:
        if os.path.exists(f):
            os.remove(f)
    for ticker in TARGET_TICKERS:
        drift_file = f'{ticker}_topic_drift_bar_chart.png'
        if os.path.exists(drift_file):
            os.remove(drift_file)
    print("Cleanup complete.")

    analyze_risk_factors(
        company_name=MY_COMPANY_NAME,
        email=MY_EMAIL,
        tickers=TARGET_TICKERS,
        start_year=REPORTING_START_YEAR,
        end_year=REPORTING_END_YEAR,
        lda_min_topics=2,
        lda_max_topics=7,
        embedding_min_clusters=6,
        embedding_max_clusters=15,
        similarity_threshold=0.75,
        random_seed=42
    )
