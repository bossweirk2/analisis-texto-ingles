import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer
from PIL import Image

# ========================
# CONFIGURACIÓN DE PÁGINA
# ========================
st.set_page_config(
    page_title="Linguistic Matrix – Semantic Intelligence System",
    layout="centered"
)

# ========================
# ESTILO MATRIX
# ========================
st.markdown("""
<style>
.stApp {
    background-color: #000000; /* Fondo negro */
    color: #00FF41; /* Verde neón */
    font-family: 'Courier New', monospace;
}

/* Botón principal */
.stButton>button {
    background-color: #003B00;
    color: #00FF41;
    border-radius: 8px;
    border: 1px solid #00FF41;
    padding: 0.5em 1em;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #00FF41;
    color: #000000;
    transform: scale(1.05);
}

/* Títulos */
h1, h2, h3, h4 {
    color: #00FF41;
    text-align: center;
    text-shadow: 0 0 10px #00FF41;
}

/* Texto de ayuda y subtítulos */
p, label, div, span {
    color: #00FF41 !important;
}

/* Cuadros de texto */
textarea, input {
    background-color: #001a00 !important;
    color: #00FF41 !important;
    border: 1px solid #00FF41 !important;
    border-radius: 6px !important;
}

/* Tablas */
.dataframe {
    color: #00FF41 !important;
    background-color: #000000 !important;
}

/* Divisor */
hr {
    border: 1px solid #00FF41;
    opacity: 0.3;
}
</style>
""", unsafe_allow_html=True)

# ========================
# CABECERA MATRIX
# ========================
st.markdown("""
<div style='text-align:center;'>
    <h1>Linguistic Matrix</h1>
    <h3>Semantic Intelligence System</h3>
    <p>
        You are entering the linguistic mainframe.<br>
        Each document will be decoded into its semantic essence.<br>
        The system will reveal which one aligns most with your query.
    </p>
</div>
""", unsafe_allow_html=True)

# Imagen decorativa (opcional)
try:
    image = Image.open("matrix_banner.jpg")
    st.image(image, caption="Decoding textual reality...", use_container_width=True)

# ========================
# ENTRADA DE DATOS
# ========================
st.markdown("#### Enter the textual data to analyze:")
text_input = st.text_area(
    "Each line represents a separate data stream:",
    "The system detects anomalies.\nLanguage is a code.\nMachines understand patterns.",
    height=150
)

st.markdown("#### Input your semantic query:")
question = st.text_input("Query:", "What do machines understand?")

# Inicializar stemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# ========================
# ANÁLISIS
# ========================
if st.button("Run Semantic Scan"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.warning("⚠️ Input at least one document for analysis.")
    else:
        with st.spinner("Analyzing data streams..."):
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                stop_words="english",
                token_pattern=None
            )

            X = vectorizer.fit_transform(documents)

            df_tfidf = pd.DataFrame(
                X.toarray(),
                columns=vectorizer.get_feature_names_out(),
                index=[f"Stream {i+1}" for i in range(len(documents))]
            )

            st.markdown("### TF-IDF Matrix")
            st.dataframe(df_tfidf.round(3))

            question_vec = vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, X).flatten()

            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]

            st.markdown("---")
            st.markdown("### Scan Results")

            st.success(f"""
            **Query:** {question}  
            **Most Relevant Stream:** Stream {best_idx+1}  
            **Text:** "{best_doc}"  
            **Semantic Correlation:** {best_score:.3f}
            """)

            sim_df = pd.DataFrame({
                "Stream": [f"Stream {i+1}" for i in range(len(documents))],
                "Text": documents,
                "Correlation": similarities
            }).sort_values("Correlation", ascending=False)

            st.markdown("### Correlation Ranking")
            st.dataframe(sim_df)

            vocab = vectorizer.get_feature_names_out()
            q_stems = tokenize_and_stem(question)
            matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]

            if matched:
                st.markdown("### Matched Lexical Patterns")
                st.write(", ".join(matched))
            else:
                st.info("No direct lexical matches found.")

# ========================
# PIE DE PÁGINA
# ========================
st.markdown("""
<hr>
<div style='text-align:center; color:#00FF41; font-size:14px;'>
Linguistic Matrix · Semantic Intelligence System v1.0  
Reality is just text waiting to be decoded.
</div>
""", unsafe_allow_html=True)
