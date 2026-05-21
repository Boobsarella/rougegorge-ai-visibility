"""
dashboard.py — Dashboard RougeGorge AI Visibility (v2 — design moderne)

Usage local : streamlit run dashboard.py
Usage cloud : déployer sur share.streamlit.io
"""

import os
import json
import pandas as pd
import streamlit as st
import anthropic
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Config page ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RougeGorge · AI Visibility",
    page_icon="🌹",
    layout="wide"
)

BRAND_COLOR = "#DC3545"

# ── CSS moderne ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.9rem !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .rec-card {
        background: #fff7f7;
        border-left: 4px solid #DC3545;
        border-radius: 6px;
        padding: 10px 16px;
        margin: 6px 0;
        font-size: 0.92em;
        line-height: 1.5;
    }
    .suggest-card {
        background: #f8faff;
        border: 1px solid #dbe8ff;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 6px 0;
    }
    .suggest-title { font-weight: 600; font-size: 0.95em; margin-bottom: 4px; }
    .suggest-desc  { color: #555; font-size: 0.85em; line-height: 1.4; }
    .page-title    { font-size: 1.8rem; font-weight: 700; margin-bottom: 0; }
    .page-sub      { color: #888; font-size: 0.88em; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Clé API (local .env ou Streamlit secrets) ──────────────────────────────────
def get_api_key():
    if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
        return st.secrets["ANTHROPIC_API_KEY"]
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("ANTHROPIC_API_KEY")


# ── Fonctions benchmark ────────────────────────────────────────────────────────
MODEL = "claude-haiku-4-5-20251001"
BRAND = "RougeGorge"


def get_ai_answer(client, prompt):
    r = client.messages.create(
        model=MODEL, max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text


def analyze_response(client, prompt, response, competitors):
    competitors_str = ", ".join(competitors)
    system = f"""Tu es un expert en visibilité de marque dans les réponses IA.
Tu analyses des réponses pour mesurer la présence de {BRAND}.
Tu réponds UNIQUEMENT en JSON valide. Concurrents surveillés : {competitors_str}"""

    request = f"""Analyse cette réponse d'un assistant IA.

Question : {prompt}
Réponse : {response}

JSON attendu :
{{
  "rougegorge_mentionnee": true ou false,
  "position_citation": "première" ou "parmi d'autres" ou "en dernier" ou "absente",
  "concurrents_mentionnes": ["marque1"],
  "tonalite": "positive" ou "neutre" ou "negative",
  "score_visibilite": 0 à 100,
  "recommandation": "action concrète pour améliorer la visibilité GEO de RougeGorge sur ce type de requête"
}}
Score : 0=absente, 25=brièvement citée, 50=clairement citée, 75=bonne option, 100=premier choix"""

    r = client.messages.create(
        model=MODEL, max_tokens=500,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": request}]
    )
    raw = r.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"rougegorge_mentionnee": False, "position_citation": "absente",
                "concurrents_mentionnes": [], "tonalite": "neutre",
                "score_visibilite": 0, "recommandation": "Erreur d'analyse"}


def run_benchmark_streamlit(client, prompts_df, competitors):
    total = len(prompts_df)
    results = []
    bar = st.progress(0, text="Démarrage...")
    status = st.empty()

    for i, row in prompts_df.iterrows():
        prompt, cat = row["prompt"], row.get("category", "général")
        bar.progress(i / total, text=f"Prompt {i + 1}/{total} — {cat}")
        status.markdown(f"*{prompt[:90]}...*")

        answer = get_ai_answer(client, prompt)
        analysis = analyze_response(client, prompt, answer, competitors)

        results.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "category": cat,
            "prompt": prompt,
            "response": answer,
            "rougegorge_mentionnee": analysis.get("rougegorge_mentionnee", False),
            "position_citation": analysis.get("position_citation", "absente"),
            "concurrents_mentionnes": ", ".join(analysis.get("concurrents_mentionnes", [])),
            "tonalite": analysis.get("tonalite", "neutre"),
            "score_visibilite": analysis.get("score_visibilite", 0),
            "recommandation": analysis.get("recommandation", "")
        })

    bar.progress(1.0, text="✅ Analyse terminée !")
    status.empty()
    return pd.DataFrame(results)


# ── Helpers graphiques ─────────────────────────────────────────────────────────
def make_gauge(score):
    color = "#22c55e" if score >= 60 else "#f59e0b" if score >= 30 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 1),
        number={"suffix": " / 100", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc"},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 30],  "color": "#fee2e2"},
                {"range": [30, 60], "color": "#fef3c7"},
                {"range": [60, 100],"color": "#dcfce7"},
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Lancer une analyse")
    api_key = get_api_key()

    if not api_key:
        st.error("Clé API manquante. Vérifie ton fichier .env.")
        st.stop()

    if st.button("🚀 Lancer le benchmark", type="primary", use_container_width=True):
        try:
            client_api = anthropic.Anthropic(api_key=api_key)
            prompts_df = pd.read_csv("prompts.csv")
            competitors = pd.read_csv("competitors.csv")["competitor"].tolist()
            with st.spinner("Interrogation de Claude en cours..."):
                df_new = run_benchmark_streamlit(client_api, prompts_df, competitors)
            os.makedirs("data", exist_ok=True)
            df_new.to_csv("data/analyzed_results.csv", index=False, encoding="utf-8")
            st.session_state["df"] = df_new
            st.success(f"Score moyen : {df_new['score_visibilite'].mean():.1f}/100")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur : {e}")

    st.divider()
    st.markdown("**À propos**")
    st.caption("Mesure la visibilité de RougeGorge dans les réponses des assistants IA (Claude Haiku).")
    st.caption("🌹 RougeGorge AI Visibility Monitor · 2026")


# ── Chargement des données ─────────────────────────────────────────────────────
DATA_FILE = "data/analyzed_results.csv"
if "df" in st.session_state:
    df = st.session_state["df"]
elif os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    st.markdown('<p class="page-title">🌹 RougeGorge — AI Visibility Monitor</p>', unsafe_allow_html=True)
    st.info("Clique sur **Lancer le benchmark** dans le menu à gauche pour démarrer la première analyse.")
    st.stop()

if df.empty:
    st.stop()


# ── TITRE ──────────────────────────────────────────────────────────────────────
last_date = df["date"].max() if "date" in df.columns else ""
st.markdown('<p class="page-title">🌹 RougeGorge — AI Visibility Monitor</p>', unsafe_allow_html=True)
st.markdown(f'<p class="page-sub">Dernière analyse : {last_date} · {len(df)} prompts · Modèle : Claude Haiku</p>',
            unsafe_allow_html=True)
st.divider()


# ── KPIs ───────────────────────────────────────────────────────────────────────
score_moyen = df["score_visibilite"].mean()
mentions    = int(df["rougegorge_mentionnee"].sum())
pct         = (mentions / len(df)) * 100
positives   = int((df["tonalite"] == "positive").sum())

all_comp = []
for e in df["concurrents_mentionnes"].dropna():
    for c in str(e).split(","):
        c = c.strip()
        if c:
            all_comp.append(c)

total_mentions = mentions + len(all_comp)
sov = round((mentions / total_mentions * 100) if total_mentions > 0 else 0, 1)
top_comp = pd.Series(all_comp).value_counts().index[0] if all_comp else "—"

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Score global", f"{score_moyen:.1f} / 100")
with c2: st.metric("Taux de citation", f"{pct:.0f}%", f"{mentions} / {len(df)} prompts")
with c3: st.metric("Share of voice", f"{sov}%", "face aux concurrents")
with c4: st.metric("Tonalité positive", f"{positives} réponses")
with c5: st.metric("Concurrent n°1", top_comp, "le plus cité par l'IA")

st.divider()


# ── JAUGE + TONALITÉ ───────────────────────────────────────────────────────────
col_gauge, col_ton = st.columns([1, 1])

with col_gauge:
    st.markdown("**Score de visibilité global**")
    st.plotly_chart(make_gauge(score_moyen), use_container_width=True)

with col_ton:
    st.markdown("**Répartition des tonalités**")
    ton = df["tonalite"].value_counts().reset_index()
    ton.columns = ["Tonalité", "Nb"]
    fig_ton = px.pie(
        ton, values="Nb", names="Tonalité", hole=0.55,
        color="Tonalité",
        color_discrete_map={"positive": "#22c55e", "neutre": "#94a3b8", "negative": "#ef4444"}
    )
    fig_ton.update_layout(
        height=200, margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", y=-0.15),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_ton, use_container_width=True)

st.divider()


# ── SCORES PAR CATÉGORIE ───────────────────────────────────────────────────────
st.markdown("### 📊 Score de visibilité par catégorie")
score_cat = (df.groupby("category")["score_visibilite"]
             .mean().sort_values(ascending=True).reset_index())
score_cat.columns = ["Catégorie", "Score"]

fig_cat = px.bar(
    score_cat, x="Score", y="Catégorie", orientation="h",
    text=score_cat["Score"].apply(lambda s: f"{s:.0f}"),
    color="Score",
    color_continuous_scale=[[0, "#ef4444"], [0.3, "#f59e0b"], [0.6, "#22c55e"], [1, "#16a34a"]],
    range_color=[0, 100]
)
fig_cat.update_traces(textposition="outside")
fig_cat.update_layout(
    height=max(280, len(score_cat) * 38),
    coloraxis_showscale=False,
    xaxis=dict(range=[0, 115], title="Score / 100"),
    yaxis_title=None,
    margin=dict(t=10, b=10, l=10, r=60),
    paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_cat, use_container_width=True)

st.divider()


# ── PART DE VOIX CONCURRENTS ───────────────────────────────────────────────────
if all_comp:
    st.markdown("### 🎯 Part de voix — RougeGorge vs concurrents")
    comp_counts = pd.Series(all_comp).value_counts().reset_index()
    comp_counts.columns = ["Marque", "Citations"]

    rg_row = pd.DataFrame([{"Marque": "🌹 RougeGorge", "Citations": mentions}])
    comp_chart = pd.concat([rg_row, comp_counts], ignore_index=True).sort_values("Citations")
    comp_chart["type"] = comp_chart["Marque"].apply(
        lambda x: "RougeGorge" if "RougeGorge" in x else "Concurrent"
    )

    fig_comp = px.bar(
        comp_chart, x="Citations", y="Marque", orientation="h",
        text="Citations", color="type",
        color_discrete_map={"RougeGorge": BRAND_COLOR, "Concurrent": "#cbd5e1"}
    )
    fig_comp.update_traces(textposition="outside")
    fig_comp.update_layout(
        height=max(300, len(comp_chart) * 34),
        showlegend=False,
        xaxis_title="Nombre de citations",
        yaxis_title=None,
        margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.divider()


# ── TABLEAU DÉTAILLÉ ───────────────────────────────────────────────────────────
st.markdown("### 📋 Résultats détaillés")

cf1, cf2, cf3 = st.columns(3)
with cf1:
    only_rg = st.checkbox("Seulement où RougeGorge est citée")
with cf2:
    f_ton = st.selectbox("Tonalité", ["Toutes", "positive", "neutre", "negative"])
with cf3:
    cats = ["Toutes"] + sorted(df["category"].dropna().unique().tolist())
    f_cat = st.selectbox("Catégorie", cats)

filtered = df.copy()
if only_rg:
    filtered = filtered[filtered["rougegorge_mentionnee"] == True]
if f_ton != "Toutes":
    filtered = filtered[filtered["tonalite"] == f_ton]
if f_cat != "Toutes":
    filtered = filtered[filtered["category"] == f_cat]

cols_show = ["category", "prompt", "score_visibilite", "position_citation",
             "rougegorge_mentionnee", "tonalite", "concurrents_mentionnes", "recommandation"]
cols_show = [c for c in cols_show if c in filtered.columns]

rename_map = {
    "category": "Catégorie", "prompt": "Prompt", "score_visibilite": "Score",
    "position_citation": "Position RG", "rougegorge_mentionnee": "RG citée ?",
    "tonalite": "Tonalité", "concurrents_mentionnes": "Concurrents cités",
    "recommandation": "Recommandation GEO"
}

display = filtered[cols_show].rename(columns=rename_map)


def color_score(val):
    try:
        v = float(val)
        if v >= 60: return "background-color:#dcfce7"
        if v >= 30: return "background-color:#fef3c7"
        return "background-color:#fee2e2"
    except Exception:
        return ""


st.dataframe(
    display.style.map(color_score, subset=["Score"]) if "Score" in display.columns else display,
    use_container_width=True,
    hide_index=True,
    height=380
)

st.download_button(
    "⬇️ Télécharger les résultats (CSV)",
    data=df.to_csv(index=False, encoding="utf-8"),
    file_name=f"rougegorge_visibility_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

st.divider()


# ── RECOMMANDATIONS ────────────────────────────────────────────────────────────
st.markdown("### 💡 Recommandations SEO / GEO prioritaires")
recs = [r for r in df["recommandation"].dropna().unique() if r and r != "Erreur d'analyse"]
for i, rec in enumerate(recs[:6], 1):
    st.markdown(f'<div class="rec-card"><strong>{i}.</strong> {rec}</div>', unsafe_allow_html=True)

st.divider()


# ── SUGGESTIONS D'ANALYSES ────────────────────────────────────────────────────
st.markdown("### 🔬 Idées pour aller plus loin")

suggestions = [
    ("🤖 Comparaison multi-LLM",
     "Tester les mêmes prompts sur ChatGPT-4o, Gemini 2.0 et Mistral pour comparer la visibilité selon les modèles IA."),
    ("📅 Suivi de l'évolution dans le temps",
     "Relancer ce benchmark chaque semaine ou mois pour mesurer l'impact des actions SEO/GEO sur le score de visibilité."),
    ("🌍 Analyse multi-langue",
     "Tester en anglais, néerlandais et espagnol pour mesurer si RougeGorge est connue hors de France."),
    ("📍 Prompts géolocalisés",
     "Ajouter 'à Paris', 'en Belgique', 'à Lyon', 'à Bruxelles' aux prompts pour mesurer la visibilité locale."),
    ("🎯 Requêtes hyper-spécifiques",
     "Tester des niches : lingerie de mariage, post-mastectomie, grandes tailles XXL, maternité, slow fashion éco-responsable..."),
    ("🗣️ Prompts conversationnels",
     "Tester des formulations naturelles : 'j'ai besoin d'un soutien-gorge confortable', 'aide-moi à choisir une lingerie pour ma copine'..."),
    ("⭐ Analyse de la réputation",
     "Demander 'quelle marque a les meilleures avis clients ?' et mesurer si RougeGorge est associée à la satisfaction."),
    ("🛍️ Analyse par type de produit",
     "Segmenter les prompts par produit : soutiens-gorge, culottes, nuisettes, bain, sport, maternité — pour identifier les lacunes."),
]

col_s1, col_s2 = st.columns(2)
for idx, (titre, desc) in enumerate(suggestions):
    with (col_s1 if idx % 2 == 0 else col_s2):
        st.markdown(
            f'<div class="suggest-card">'
            f'<div class="suggest-title">{titre}</div>'
            f'<div class="suggest-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
