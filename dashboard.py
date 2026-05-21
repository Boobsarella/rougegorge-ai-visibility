"""
dashboard.py — Dashboard RougeGorge AI Visibility (version partageable)

Inclut un bouton pour lancer le benchmark directement depuis l'interface.
Déployable sur Streamlit Community Cloud.

Usage local  : streamlit run dashboard.py
Usage cloud  : déployer sur share.streamlit.io
"""

import os
import json
import pandas as pd
import streamlit as st
import anthropic
from datetime import datetime

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="RougeGorge — AI Visibility",
    page_icon="🌹",
    layout="wide"
)

st.title("🌹 RougeGorge — Dashboard Visibilité IA")
st.caption("Mesure la présence de RougeGorge dans les réponses des assistants IA")


# ── Chargement de la clé API ──────────────────────────────────────────────────
# Sur Streamlit Cloud, la clé vient des Secrets.
# En local, elle vient du fichier .env via les variables d'environnement.
def get_api_key():
    # Vérifie d'abord les secrets Streamlit (pour le déploiement cloud)
    if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
        return st.secrets["ANTHROPIC_API_KEY"]
    # Sinon utilise la variable d'environnement locale
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("ANTHROPIC_API_KEY")


# ── Fonctions du benchmark ────────────────────────────────────────────────────
MODEL = "claude-haiku-4-5-20251001"
BRAND = "RougeGorge"


def get_ai_answer(client, prompt: str) -> str:
    """Envoie un prompt à Claude et retourne sa réponse naturelle."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def analyze_response(client, prompt: str, response: str, competitors: list) -> dict:
    """Analyse une réponse Claude pour mesurer la visibilité de RougeGorge."""
    competitors_str = ", ".join(competitors)
    system_prompt = f"""Tu es un expert en analyse de visibilité de marque dans les réponses IA.
Tu analyses des réponses d'assistants IA pour mesurer la présence de la marque {BRAND}.
Tu réponds TOUJOURS avec un JSON valide, sans texte avant ou après.
Concurrents à surveiller : {competitors_str}"""

    analysis_request = f"""Analyse cette réponse d'un assistant IA.

Question posée : {prompt}

Réponse analysée : {response}

Réponds UNIQUEMENT avec ce JSON :
{{
  "rougegorge_mentionnee": true ou false,
  "concurrents_mentionnes": ["marque1", "marque2"],
  "tonalite": "positive" ou "neutre" ou "negative",
  "score_visibilite": 0 à 100,
  "recommandation": "Une recommandation courte pour améliorer la visibilité de RougeGorge"
}}"""

    result = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": analysis_request}]
    )

    raw = result.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"rougegorge_mentionnee": False, "concurrents_mentionnes": [],
                "tonalite": "neutre", "score_visibilite": 0, "recommandation": "Erreur d'analyse"}


def run_benchmark_streamlit(client, prompts_df, competitors):
    """Lance le benchmark avec une barre de progression Streamlit."""
    total = len(prompts_df)
    results = []

    progress_bar = st.progress(0, text="Démarrage...")
    status = st.empty()

    for i, row in prompts_df.iterrows():
        prompt = row["prompt"]
        category = row.get("category", "général")

        progress_bar.progress((i) / total, text=f"Prompt {i + 1}/{total} — {category}")
        status.markdown(f"**Interrogation de Claude :** _{prompt[:80]}..._")

        answer = get_ai_answer(client, prompt)
        analysis = analyze_response(client, prompt, answer, competitors)

        results.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "category": category,
            "prompt": prompt,
            "response": answer,
            "rougegorge_mentionnee": analysis.get("rougegorge_mentionnee", False),
            "concurrents_mentionnes": ", ".join(analysis.get("concurrents_mentionnes", [])),
            "tonalite": analysis.get("tonalite", "neutre"),
            "score_visibilite": analysis.get("score_visibilite", 0),
            "recommandation": analysis.get("recommandation", "")
        })

    progress_bar.progress(1.0, text="Analyse terminée !")
    status.empty()
    return pd.DataFrame(results)


# ── Sidebar : lancer le benchmark ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Benchmark")

    api_key = get_api_key()
    if not api_key:
        st.error("Clé API manquante. Vérifie ton fichier .env ou les secrets Streamlit.")
        st.stop()

    if st.button("🚀 Lancer le benchmark", type="primary", use_container_width=True):
        try:
            client = anthropic.Anthropic(api_key=api_key)
            prompts_df = pd.read_csv("prompts.csv")
            competitors = pd.read_csv("competitors.csv")["competitor"].tolist()

            with st.spinner("Benchmark en cours..."):
                df_new = run_benchmark_streamlit(client, prompts_df, competitors)

            # Sauvegarde les résultats
            os.makedirs("data", exist_ok=True)
            df_new.to_csv("data/analyzed_results.csv", index=False, encoding="utf-8")
            st.session_state["df"] = df_new
            st.success(f"✅ Terminé ! Score moyen : {df_new['score_visibilite'].mean():.1f}/100")
            st.rerun()

        except Exception as e:
            st.error(f"Erreur : {e}")

    st.divider()
    st.caption("RougeGorge AI Visibility Monitor")


# ── Chargement des données ────────────────────────────────────────────────────
DATA_FILE = "data/analyzed_results.csv"

# Priorité : données fraîches en session, sinon fichier CSV
if "df" in st.session_state:
    df = st.session_state["df"]
elif os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    st.info("Aucune donnée disponible. Clique sur **Lancer le benchmark** dans le menu à gauche.")
    st.stop()

if df.empty:
    st.warning("Les données sont vides. Lance le benchmark.")
    st.stop()


# ── Section 1 : Métriques principales ────────────────────────────────────────
st.subheader("📊 Vue d'ensemble")

col1, col2, col3, col4 = st.columns(4)
score_moyen = df["score_visibilite"].mean()
mentions = int(df["rougegorge_mentionnee"].sum())
pct_mentions = (mentions / len(df)) * 100
positives = int((df["tonalite"] == "positive").sum())

with col1:
    st.metric("Prompts testés", len(df))
with col2:
    st.metric("Score moyen de visibilité", f"{score_moyen:.1f} / 100")
with col3:
    st.metric("RougeGorge mentionnée", f"{mentions} / {len(df)}", f"{pct_mentions:.0f}%")
with col4:
    st.metric("Tonalité positive", f"{positives} fois")

st.divider()

# ── Section 2 : Graphiques ────────────────────────────────────────────────────
col_g, col_d = st.columns(2)

with col_g:
    st.subheader("📈 Score par catégorie")
    if "category" in df.columns:
        score_cat = df.groupby("category")["score_visibilite"].mean().sort_values(ascending=False)
        st.bar_chart(score_cat)

with col_d:
    st.subheader("😊 Répartition des tonalités")
    st.bar_chart(df["tonalite"].value_counts())

st.divider()

# ── Section 3 : Concurrents ───────────────────────────────────────────────────
st.subheader("🎯 Concurrents les plus cités par l'IA")
all_competitors = []
for entry in df["concurrents_mentionnes"].dropna():
    if entry and str(entry).strip():
        for name in str(entry).split(","):
            name = name.strip()
            if name:
                all_competitors.append(name)

if all_competitors:
    st.bar_chart(pd.Series(all_competitors).value_counts())
else:
    st.info("Aucun concurrent mentionné.")

st.divider()

# ── Section 4 : Tableau détaillé ─────────────────────────────────────────────
st.subheader("📋 Détail des résultats")

col_f1, col_f2 = st.columns(2)
with col_f1:
    only_mentioned = st.checkbox("Seulement où RougeGorge est citée")
with col_f2:
    filtre_ton = st.selectbox("Filtrer par tonalité", ["Toutes", "positive", "neutre", "negative"])

filtered = df.copy()
if only_mentioned:
    filtered = filtered[filtered["rougegorge_mentionnee"] == True]
if filtre_ton != "Toutes":
    filtered = filtered[filtered["tonalite"] == filtre_ton]

st.dataframe(
    filtered[["prompt", "score_visibilite", "rougegorge_mentionnee", "tonalite",
              "concurrents_mentionnes", "recommandation"]].rename(columns={
        "prompt": "Prompt", "score_visibilite": "Score",
        "rougegorge_mentionnee": "RG citée ?", "tonalite": "Tonalité",
        "concurrents_mentionnes": "Concurrents cités", "recommandation": "Recommandation"
    }),
    use_container_width=True, hide_index=True
)

# Bouton de téléchargement
st.download_button(
    label="⬇️ Télécharger les résultats (CSV)",
    data=df.to_csv(index=False, encoding="utf-8"),
    file_name=f"rougegorge_visibility_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

st.divider()

# ── Section 5 : Recommandations ───────────────────────────────────────────────
st.subheader("💡 Recommandations SEO/GEO")
recs = [r for r in df["recommandation"].dropna().unique() if r and r != "Erreur d'analyse"]
for i, rec in enumerate(recs[:8], 1):
    st.markdown(f"**{i}.** {rec}")

if "date" in df.columns:
    st.divider()
    st.caption(f"Dernière mise à jour : {df['date'].max()}")
