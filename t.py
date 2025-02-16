import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import LabelEncoder
from io import StringIO

# Charger le modèle et le tokenizer BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # 2 labels: toxique, non toxique

# Charger le modèle (si disponible localement ou téléchargeable)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fonction pour prédire la toxicité d'un commentaire
def predict_toxicity(comment):
    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()  # 0 -> non toxique, 1 -> toxique
    return pred

# Fonction pour charger et afficher les commentaires depuis la base de données (CSV)
def load_comments_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return pd.DataFrame(columns=["Commentaire", "Toxicité"])

# Fonction pour filtrer les données
def filter_comments(df, toxicity_level):
    if toxicity_level == "Toxique":
        return df[df["Toxicité"] == 1]
    elif toxicity_level == "Non Toxique":
        return df[df["Toxicité"] == 0]
    return df

# Interface Streamlit
st.title("Modération de Commentaires - BERT")

st.sidebar.header("Téléchargez votre base de données")
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type=["csv"])

# Charger les données
df = load_comments_data(uploaded_file)

# Afficher un message si aucune donnée n'est chargée
if df.empty:
    st.sidebar.warning("Veuillez télécharger un fichier CSV contenant des commentaires.")
else:
    # Appliquer un filtre sur les niveaux de toxicité
    toxicity_level = st.sidebar.radio("Filtrer par toxicité", ("Tous", "Toxique", "Non Toxique"))
    df_filtered = filter_comments(df, toxicity_level)

    # Afficher la table filtrée
    st.subheader("Commentaires filtrés")
    st.write(df_filtered)

    # Prédiction de toxicité pour un commentaire
    st.subheader("Prédiction de Toxicité")
    comment_input = st.text_area("Entrez un commentaire à analyser", "")
    if st.button("Analyser"):
        if comment_input:
            prediction = predict_toxicity(comment_input)
            toxic_label = "Toxique" if prediction == 1 else "Non Toxique"
            st.write(f"Le commentaire est : {toxic_label}")
        else:
            st.warning("Veuillez entrer un commentaire à analyser.")

    # Option pour ajouter des commentaires et obtenir des prédictions
    st.subheader("Ajouter un nouveau commentaire")
    new_comment = st.text_area("Entrez un commentaire à ajouter", "")
    if st.button("Ajouter le commentaire"):
        if new_comment:
            prediction = predict_toxicity(new_comment)
            toxic_label = "Toxique" if prediction == 1 else "Non Toxique"
            new_data = pd.DataFrame([[new_comment, toxic_label]], columns=["Commentaire", "Toxicité"])
            df = pd.concat([df, new_data], ignore_index=True)
            st.success("Commentaire ajouté avec succès !")
        else:
            st.warning("Veuillez entrer un commentaire.")

    # Option pour sauvegarder les commentaires modifiés
    st.sidebar.subheader("Sauvegarder les commentaires")
    if st.sidebar.button("Télécharger les données modifiées"):
        csv_data = df.to_csv(index=False)
        st.sidebar.download_button("Télécharger CSV", csv_data, file_name="comments_moderation.csv", mime="text/csv")
