"""
dashboard.py — Dashboard RougeGorge AI Visibility (v3 — multi-LLM)

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

st.set_page_config(
    page_title="RougeGorge · AI Visibility",
    page_icon="🌹",
    layout="wide"
)

BRAND_COLOR = "#DC3545"

LLM_COLORS = {
    "Claude Haiku":     "#DC3545",
    "GPT-4o":           "#10a37f",
    "Gemini 2.0 Flash": "#4285f4",
    "Perplexity Sonar": "#5436DA",
}

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.9rem !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #888;
                                    text-transform: uppercase; letter-spacing: 0.05em; }
    .rec-card { background:#fff7f7; border-left:4px solid #DC3545; border-radius:6px;
                padding:10px 16px; margin:6px 0; font-size:0.92em; line-height:1.5; }
    .suggest-card { background:#f8faff; border:1px solid #dbe8ff; border-radius:10px;
                    padding:14px 16px; margin:6px 0; }
    .suggest-title { font-weight:600; font-size:0.95em; margin-bottom:4px; }
    .suggest-desc  { color:#555; font-size:0.85em; line-height:1.4; }
</style>
""", unsafe_allow_html=True)


# ── Clé API ────────────────────────────────────────────────────────────────────
def get_secret(key):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    from dotenv import load_dotenv
    load_dotenv()
    val = os.getenv(key, "")
    return val if val and "REMPLACE" not in val else None


# ── Fonctions LLM ──────────────────────────────────────────────────────────────
def query_claude(prompt, api_key):
    import anthropic as ant
    c = ant.Anthropic(api_key=api_key)
    r = c.messages.create(model="claude-haiku-4-5-20251001", max_tokens=800,
                          messages=[{"role": "user", "content": prompt}])
    return r.content[0].text

def query_gpt4o(prompt, api_key):
    from openai import OpenAI
    c = OpenAI(api_key=api_key)
    r = c.chat.completions.create(model="gpt-4o", max_tokens=800,
                                   messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content

def query_gemini(prompt, api_key):
    from google import genai
    c = genai.Client(api_key=api_key)
    r = c.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return r.text

def query_perplexity(prompt, api_key):
    from openai import OpenAI
    c = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    r = c.chat.completions.create(model="sonar", max_tokens=800,
                                   messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


LLM_CONFIG = {
    "Claude Haiku":     {"fn": query_claude,     "env": "ANTHROPIC_API_KEY"},
    "GPT-4o":           {"fn": query_gpt4o,       "env": "OPENAI_API_KEY"},
    "Gemini 2.0 Flash": {"fn": query_gemini,      "env": "GOOGLE_API_KEY"},
    "Perplexity Sonar": {"fn": query_perplexity,  "env": "PERPLEXITY_API_KEY"},
}

def get_active_llms():
    return {name: {"fn": cfg["fn"], "key": get_secret(cfg["env"])}
            for name, cfg in LLM_CONFIG.items() if get_secret(cfg["env"])}


def analyze_response(prompt, response, competitors, claude_client):
    competitors_str = ", ".join(competitors)
    system = f"""Tu es un expert en visibilité de marque dans les réponses IA.
Tu analyses des réponses pour mesurer la présence de RougeGorge.
Tu réponds UNIQUEMENT en JSON valide. Concurrents surveillés : {competitors_str}"""
    request = f"""Question : {prompt}\nRéponse : {response}

JSON :
{{
  "rougegorge_mentionnee": true/false,
  "position_citation": "première"/"parmi d'autres"/"en dernier"/"absente",
  "concurrents_mentionnes": ["marque"],
  "tonalite": "positive"/"neutre"/"negative",
  "score_visibilite": 0-100,
  "recommandation": "action GEO concrète"
}}"""
    r = claude_client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=500,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": request}]
    )
    raw = r.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"rougegorge_mentionnee": False, "position_citation": "absente",
                "concurrents_mentionnes": [], "tonalite": "neutre",
                "score_visibilite": 0, "recommandation": "Erreur d'analyse"}


def run_benchmark_streamlit(active_llms, claude_client, prompts_df, competitors):
    total = len(prompts_df) * len(active_llms)
    results = []
    bar = st.progress(0)
    status = st.empty()
    count = 0

    for llm_name, llm in active_llms.items():
        for _, row in prompts_df.iterrows():
            prompt, cat = row["prompt"], row.get("category", "général")
            count += 1
            bar.progress(count / total, text=f"{llm_name} — {cat} ({count}/{total})")
            status.markdown(f"*{prompt[:90]}...*")

            try:
                answer = llm["fn"](prompt, llm["key"])
            except Exception as e:
                answer = ""
                st.warning(f"{llm_name} : erreur — {e}")

            analysis = analyze_response(prompt, answer, competitors, claude_client) if answer else \
                {"rougegorge_mentionnee": False, "position_citation": "absente",
                 "concurrents_mentionnes": [], "tonalite": "neutre",
                 "score_visibilite": 0, "recommandation": "Erreur de requête"}

            results.append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "llm": llm_name,
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

    bar.progress(1.0, text="✅ Terminé !")
    status.empty()
    return pd.DataFrame(results)


# ── Graphiques ─────────────────────────────────────────────────────────────────
def make_gauge(score):
    color = "#22c55e" if score >= 60 else "#f59e0b" if score >= 30 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score, 1),
        number={"suffix": " / 100", "font": {"size": 28, "color": color}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.3},
               "steps": [{"range": [0, 30], "color": "#fee2e2"},
                         {"range": [30, 60], "color": "#fef3c7"},
                         {"range": [60, 100], "color": "#dcfce7"}]}
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Lancer une analyse")

    anthropic_key = get_secret("ANTHROPIC_API_KEY")
    if not anthropic_key:
        st.error("Clé Anthropic manquante (nécessaire pour l'analyse).")
        st.stop()

    active_llms = get_active_llms()
    if active_llms:
        st.markdown("**LLMs détectés :**")
        for name in active_llms:
            color = LLM_COLORS.get(name, "#888")
            st.markdown(f"<span style='color:{color}'>●</span> {name}", unsafe_allow_html=True)
    else:
        st.warning("Aucune clé API configurée.")

    st.divider()

    if st.button("🚀 Lancer le benchmark", type="primary", use_container_width=True,
                 disabled=not active_llms):
        try:
            claude_client = anthropic.Anthropic(api_key=anthropic_key)
            prompts_df = pd.read_csv("prompts.csv")
            competitors = pd.read_csv("competitors.csv")["competitor"].tolist()
            df_new = run_benchmark_streamlit(active_llms, claude_client, prompts_df, competitors)
            os.makedirs("data", exist_ok=True)
            df_new.to_csv("data/analyzed_results.csv", index=False, encoding="utf-8")
            st.session_state["df"] = df_new
            avg = df_new["score_visibilite"].mean()
            st.success(f"Score moyen toutes IA : {avg:.1f}/100")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur : {e}")

    st.divider()
    st.caption("🌹 RougeGorge AI Visibility · 2026")


# ── Chargement données ─────────────────────────────────────────────────────────
DATA_FILE = "data/analyzed_results.csv"
if "df" in st.session_state:
    df = st.session_state["df"]
elif os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    st.title("🌹 RougeGorge — AI Visibility Monitor")
    st.info("Configure tes clés API dans `.env` puis clique sur **Lancer le benchmark**.")
    st.stop()

if df.empty:
    st.stop()

available_llms = sorted(df["llm"].unique().tolist()) if "llm" in df.columns else []

# ── TITRE ──────────────────────────────────────────────────────────────────────
st.markdown("## 🌹 RougeGorge — AI Visibility Monitor")
last_date = df["date"].max() if "date" in df.columns else ""
llm_list = " · ".join(available_llms)
st.caption(f"Dernière analyse : {last_date} · {len(df)} réponses analysées · {llm_list}")
st.divider()


# ── KPIs GLOBAUX ───────────────────────────────────────────────────────────────
score_moyen = df["score_visibilite"].mean()
mentions    = int(df["rougegorge_mentionnee"].sum())
pct         = (mentions / len(df)) * 100
positives   = int((df["tonalite"] == "positive").sum())

all_comp = []
for e in df["concurrents_mentionnes"].dropna():
    for c in str(e).split(","):
        c = c.strip()
        if c: all_comp.append(c)

total_mentions = mentions + len(all_comp)
sov      = round((mentions / total_mentions * 100) if total_mentions > 0 else 0, 1)
top_comp = pd.Series(all_comp).value_counts().index[0] if all_comp else "—"

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Score global", f"{score_moyen:.1f} / 100")
with c2: st.metric("Taux de citation", f"{pct:.0f}%", f"{mentions} / {len(df)}")
with c3: st.metric("Share of voice", f"{sov}%", "vs concurrents")
with c4: st.metric("Tonalité positive", f"{positives} réponses")
with c5: st.metric("Concurrent n°1", top_comp)

st.divider()


# ── COMPARAISON ENTRE LLMs ────────────────────────────────────────────────────
if len(available_llms) > 1:
    st.markdown("### 🤖 Comparaison entre les IA")

    llm_stats = (df.groupby("llm")
                   .agg(score_moyen=("score_visibilite", "mean"),
                        taux_citation=("rougegorge_mentionnee", lambda x: round(x.mean() * 100, 1)))
                   .reset_index()
                   .sort_values("score_moyen", ascending=False))

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Score moyen de visibilité par IA**")
        color_map = {row["llm"]: LLM_COLORS.get(row["llm"], "#888") for _, row in llm_stats.iterrows()}
        fig_llm = px.bar(llm_stats, x="llm", y="score_moyen", text="score_moyen",
                         color="llm", color_discrete_map=color_map)
        fig_llm.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_llm.update_layout(height=280, showlegend=False, yaxis_range=[0, 110],
                               xaxis_title=None, yaxis_title="Score / 100",
                               margin=dict(t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_llm, use_container_width=True)

    with col_b:
        st.markdown("**Taux de citation de RougeGorge par IA (%)**")
        fig_cit = px.bar(llm_stats, x="llm", y="taux_citation", text="taux_citation",
                         color="llm", color_discrete_map=color_map)
        fig_cit.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig_cit.update_layout(height=280, showlegend=False, yaxis_range=[0, 110],
                               xaxis_title=None, yaxis_title="% de prompts où RG est citée",
                               margin=dict(t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cit, use_container_width=True)

    # Heatmap score par LLM × catégorie
    st.markdown("**Heatmap : score par IA et par catégorie**")
    heat = df.groupby(["category", "llm"])["score_visibilite"].mean().reset_index()
    heat_pivot = heat.pivot(index="category", columns="llm", values="score_visibilite").fillna(0)
    fig_heat = px.imshow(heat_pivot, color_continuous_scale=["#fee2e2", "#fef3c7", "#dcfce7"],
                         zmin=0, zmax=100, text_auto=".0f", aspect="auto")
    fig_heat.update_layout(height=max(250, len(heat_pivot) * 35),
                            margin=dict(t=20, b=20), paper_bgcolor="rgba(0,0,0,0)",
                            coloraxis_showscale=False)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()


# ── JAUGE + TONALITÉ ───────────────────────────────────────────────────────────
col_g, col_t = st.columns([1, 1])
with col_g:
    st.markdown("**Score global (toutes IA confondues)**")
    st.plotly_chart(make_gauge(score_moyen), use_container_width=True)
with col_t:
    st.markdown("**Tonalité des réponses**")
    ton = df["tonalite"].value_counts().reset_index()
    ton.columns = ["Tonalité", "Nb"]
    fig_ton = px.pie(ton, values="Nb", names="Tonalité", hole=0.55,
                     color="Tonalité",
                     color_discrete_map={"positive": "#22c55e", "neutre": "#94a3b8", "negative": "#ef4444"})
    fig_ton.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=-0.15), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ton, use_container_width=True)

st.divider()


# ── SCORES PAR CATÉGORIE ───────────────────────────────────────────────────────
st.markdown("### 📊 Score par catégorie")
score_cat = (df.groupby("category")["score_visibilite"].mean()
               .sort_values(ascending=True).reset_index())
score_cat.columns = ["Catégorie", "Score"]
fig_cat = px.bar(score_cat, x="Score", y="Catégorie", orientation="h",
                  text=score_cat["Score"].apply(lambda s: f"{s:.0f}"),
                  color="Score", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                  range_color=[0, 100])
fig_cat.update_traces(textposition="outside")
fig_cat.update_layout(height=max(280, len(score_cat) * 38), coloraxis_showscale=False,
                       xaxis=dict(range=[0, 115]), yaxis_title=None,
                       margin=dict(t=10, b=10, l=10, r=60), paper_bgcolor="rgba(0,0,0,0)")
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
        lambda x: "RougeGorge" if "RougeGorge" in x else "Concurrent")
    fig_comp = px.bar(comp_chart, x="Citations", y="Marque", orientation="h", text="Citations",
                       color="type", color_discrete_map={"RougeGorge": BRAND_COLOR, "Concurrent": "#cbd5e1"})
    fig_comp.update_traces(textposition="outside")
    fig_comp.update_layout(height=max(300, len(comp_chart) * 34), showlegend=False,
                            xaxis_title="Citations", yaxis_title=None,
                            margin=dict(t=10, b=10, l=10, r=60), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_comp, use_container_width=True)
    st.divider()


# ── TABLEAU DÉTAILLÉ ───────────────────────────────────────────────────────────
st.markdown("### 📋 Résultats détaillés")

cf1, cf2, cf3, cf4 = st.columns(4)
with cf1:
    llm_opts = ["Tous"] + available_llms
    f_llm = st.selectbox("IA", llm_opts)
with cf2:
    cats = ["Toutes"] + sorted(df["category"].dropna().unique().tolist())
    f_cat = st.selectbox("Catégorie", cats)
with cf3:
    f_ton = st.selectbox("Tonalité", ["Toutes", "positive", "neutre", "negative"])
with cf4:
    only_rg = st.checkbox("Seulement où RG est citée")

filtered = df.copy()
if f_llm   != "Tous":    filtered = filtered[filtered["llm"]      == f_llm]
if f_cat   != "Toutes":  filtered = filtered[filtered["category"] == f_cat]
if f_ton   != "Toutes":  filtered = filtered[filtered["tonalite"] == f_ton]
if only_rg:              filtered = filtered[filtered["rougegorge_mentionnee"] == True]

cols = [c for c in ["llm", "category", "prompt", "score_visibilite", "position_citation",
                     "rougegorge_mentionnee", "tonalite", "concurrents_mentionnes", "recommandation"]
        if c in filtered.columns]
rename = {"llm": "IA", "category": "Catégorie", "prompt": "Prompt",
          "score_visibilite": "Score", "position_citation": "Position RG",
          "rougegorge_mentionnee": "RG citée ?", "tonalite": "Tonalité",
          "concurrents_mentionnes": "Concurrents cités", "recommandation": "Recommandation GEO"}

display = filtered[cols].rename(columns=rename)

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
    use_container_width=True, hide_index=True, height=380
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


# ── SUGGESTIONS D'ANALYSES ─────────────────────────────────────────────────────
st.markdown("### 🔬 Idées pour aller plus loin")
suggestions = [
    ("📅 Suivi hebdomadaire", "Relancer le benchmark chaque semaine pour mesurer l'impact de tes actions SEO/GEO sur le score de visibilité."),
    ("🌍 Multi-langue", "Tester en anglais, néerlandais et espagnol pour mesurer la visibilité internationale."),
    ("📍 Prompts géolocalisés", "Ajouter 'à Paris', 'en Belgique', 'à Lyon' aux prompts pour mesurer la visibilité locale."),
    ("🎯 Requêtes de niche", "Tester : lingerie de mariage, post-mastectomie, grandes tailles XXL, maternité, slow fashion..."),
    ("🗣️ Prompts conversationnels", "Formulations naturelles : 'j'ai besoin d'un soutien-gorge confortable', 'aide-moi à choisir...'"),
    ("⭐ Réputation & avis", "Demander 'quelle marque a les meilleurs avis clients ?' pour mesurer l'association RG ↔ satisfaction."),
    ("🛍️ Par type de produit", "Segmenter par produit : soutiens-gorge, culottes, nuisettes, bain, sport, maternité."),
    ("📰 Contenu GEO", "Créer des pages de contenu optimisées pour les requêtes où RG est absente des réponses IA."),
]
col_s1, col_s2 = st.columns(2)
for i, (titre, desc) in enumerate(suggestions):
    with (col_s1 if i % 2 == 0 else col_s2):
        st.markdown(
            f'<div class="suggest-card"><div class="suggest-title">{titre}</div>'
            f'<div class="suggest-desc">{desc}</div></div>',
            unsafe_allow_html=True
        )
