import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import re
from datetime import datetime
import nltk

# Configuration de la page
st.set_page_config(
    page_title="Chatbot Restaurant Africain 🍲",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Liste des mots vides français adaptés au domaine restaurant africain
FRENCH_STOPWORDS = {
    'le', 'la', 'les', 'de', 'des', 'du', 'un', 'une', 'et', 'à', 'en', 'pour', 'avec', 'sur', 'au', 'aux',
    'ce', 'ces', 'dans', 'par', 'plus', 'ne', 'pas', 'que', 'qui', 'se', 'sa', 'son', 'ses', 'nous', 'vous',
    'ils', 'elles', 'être', 'avoir', 'faire', 'comme', 'mais', 'ou', 'donc', 'or', 'ni', 'car',
    'plat', 'plats', 'ingrédient', 'ingrédients', 'cuisson', 'réservation', 'menu',
    'commande', 'serveur', 'serveuse', 'client', 'clients', 'restaurant', 'table',
    'boisson', 'boissons', 'heure', 'heures', 'jour', 'jours', 'prix', 'tarif',
    'spécialité', 'spécialités', 'africain', 'africaine', 'afrique', 'nourriture',
    'repas', 'déjeuner', 'dîner', 'goûter', 'service', 'commande', 'livraison',
    'allergie', 'allergies', 'intolérance', 'intolérances', 'végétarien', 'végétarienne',
    'vegan', 'sans', 'gluten', 'piment', 'épicé', 'douceur',
    'foutou', 'fufu', 'attiéké', 'alloco', 'ceebu', 'jën', 'thiebou', 'dieune',
    'tô', 'soupe', 'egusi', 'kedjenou', 'ayimolou', 'poulet', 'yassa', 'mafé',
    'arachide', 'couscous', 'thiéboudiène', 'moambe', 'jollof', 'rice', 'bananes', 'plantains'
}

# Dictionnaire des descriptions des plats
DISH_DESCRIPTIONS = {
    'foutou': "👨‍🍳 Pâte de banane plantain, manioc ou igname, accompagnant les sauces.",
    'attiéké': "👨‍🍳 Semoule de manioc fermentée, servie avec poisson grillé.",
    'alloco': "👨‍🍳 Bananes plantains frites, croustillantes et moelleuses.",
    'ceebu jën': "👨‍🍳 Riz au poisson sénégalais, avec légumes et sauce tomate.",
    'kedjenou': "👨‍🍳 Ragoût de poulet ivoirien cuit à l'étouffée avec légumes et épices."
}

# Base de connaissances enrichie
KNOWLEDGE_TEXT = """
Foutou est une pâte traditionnelle africaine faite à base de manioc, d'igname ou de banane plantain.
Attiéké est un plat ivoirien à base de manioc fermenté, souvent accompagné de poisson.
Alloco sont des bananes plantains frites, croustillantes et moelleuses.
Ceebu Jën est un riz au poisson sénégalais, souvent avec sauce tomate et légumes.
Kedjenou est un poulet épicé cuit à l'étouffée avec des légumes.
Nos horaires sont de 11h00 à 22h00 tous les jours.
Réservez une table en appelant le 01 23 45 67 89.
Nous proposons du foutou, attiéké, mafé, yassa, alloco, etc.
La livraison est possible dans un rayon de 10 km.
Nous acceptons les paiements en espèces, carte, Orange Money et Wave.
Merci pour votre visite et bon appétit !
"""

# Nettoyage du texte

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = [word for word in text.split() if word not in FRENCH_STOPWORDS]
    return ' '.join(words)

# Construction du corpus
@st.cache_data(show_spinner=False)
def build_corpus(text: str):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    processed = [preprocess(s) for s in sentences]
    vec = TfidfVectorizer()
    matrix = vec.fit_transform(processed)
    return sentences, matrix, vec

# Récupérer la meilleure réponse

def get_response(sentences: list[str], matrix: np.ndarray, vec: TfidfVectorizer, query: str, threshold: float = 0.1):
    q_processed = preprocess(query)
    q_vec = vec.transform([q_processed])
    sims = cosine_similarity(q_vec, matrix).flatten()
    idx = int(np.argmax(sims))
    score = sims[idx]
    if score < threshold:
        return ("Désolé, je n'ai pas compris votre demande.", score)
    return (sentences[idx], score)

# Application principale

def main():
    st.title("🤖 Chatbot Restaurant Africain")
    st.write("Posez vos questions sur notre menu, horaires, plats, etc.")

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1046/1046784.png", width=120)
        st.write("Bienvenue dans notre restaurant africain 🇨🇮")
        st.write(f"🗓️ Aujourd'hui : {datetime.now().strftime('%d/%m/%Y')}")

    sentences, matrix, vec = build_corpus(KNOWLEDGE_TEXT)

    question = st.text_input("💬 Posez votre question :")

    if st.button("Envoyer") and question:
        with st.spinner("Recherche de réponse..."):
            time.sleep(1)
            response, score = get_response(sentences, matrix, vec, question)

        st.write(f"**Question :** {question}")
        st.success(f"**Réponse :** {response}")
        st.caption(f"🔎 Score de similarité : {round(score, 2)}")

        for dish, description in DISH_DESCRIPTIONS.items():
            if dish in response.lower():
                st.info(description)
                break

if __name__ == '__main__':
    main()
