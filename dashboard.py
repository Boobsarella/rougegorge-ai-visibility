"""
dashboard.py — Dashboard RougeGorge AI Visibility
"""

import os
import re
import json
import pandas as pd
import streamlit as st
import anthropic
from openai import OpenAI
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="RougeGorge · AI Visibility", page_icon="🌹", layout="wide")

BRAND_COLOR = "#DC3545"
CATEGORIES  = [
    "decouverte", "cadeau", "confort", "taille_plus", "sport", "nuit",
    "comparaison", "prix", "occasion", "boutique", "mariage", "tendance",
    "durabilite", "autre",
]
LLM_COLORS = {
    "Claude Sonnet 4.6":      "#DC3545",
    "Claude Opus 4.7":        "#a61c2e",
    "Claude Sonnet 4.6 Web":  "#ff6b6b",
    "GPT-5.5":                "#10a37f",
    "GPT-5.4":                "#1a7a5e",
    "GPT-5.4-mini":           "#34d399",
    "GPT-4o Web":             "#0ea5e9",
    "GPT-5 Search":           "#0369a1",
    "Perplexity Sonar Pro":   "#7c3aed",
    "Perplexity Sonar":       "#a78bfa",
}

# ── Modèles ────────────────────────────────────────────────────────────────────
LLMS_CLAUDE = [
    {"name": "Claude Sonnet 4.6", "model": "claude-sonnet-4-6", "source": "anthropic"},
    {"name": "Claude Opus 4.7",   "model": "claude-opus-4-7",   "source": "anthropic"},
]
LLMS_CLAUDE_WEB = [
    {"name": "Claude Sonnet 4.6 Web", "model": "claude-sonnet-4-6", "source": "anthropic_web"},
]
LLMS_OPENAI = [
    {"name": "GPT-5.5",      "model": "gpt-5.5",      "source": "openai"},
    {"name": "GPT-5.4",      "model": "gpt-5.4",      "source": "openai"},
    {"name": "GPT-5.4-mini", "model": "gpt-5.4-mini", "source": "openai"},
]
LLMS_OPENAI_SEARCH = [
    {"name": "GPT-4o Web",   "model": "gpt-4o-search-preview", "source": "openai"},
    {"name": "GPT-5 Search", "model": "gpt-5-search-api",      "source": "openai"},
]
LLMS_PERPLEXITY = [
    {"name": "Perplexity Sonar Pro", "model": "sonar-pro", "source": "perplexity"},
    {"name": "Perplexity Sonar",     "model": "sonar",     "source": "perplexity"},
]


# ── CSS + Poppins ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* Poppins sur le texte — exclut les polices d'icônes Streamlit (Material Icons) */
body, .stApp,
p, h1, h2, h3, h4, h5, h6,
.stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown li, .stMarkdown td, .stMarkdown th,
.stButton > button,
label,
input[type="text"], input[type="number"], input[type="search"], textarea,
.stSelectbox [data-baseweb="select"] *,
.stTextInput input,
.stNumberInput input,
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stCaption"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-baseweb="tab"] > div,
[data-baseweb="menu"] li {
    font-family: 'Poppins', sans-serif !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.85rem !important; font-weight: 700;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; color: #888;
    text-transform: uppercase; letter-spacing: 0.07em;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500; font-size: 0.92em; padding: 8px 20px;
}
.rec-card {
    background: #fff7f7; border-left: 4px solid #DC3545;
    border-radius: 8px; padding: 10px 16px; margin: 6px 0;
    font-size: 0.9em; line-height: 1.6;
}
.suggest-card {
    background: #f8faff; border: 1px solid #dbe8ff;
    border-radius: 10px; padding: 14px 16px; margin: 6px 0;
}
.suggest-title { font-weight: 600; font-size: 0.95em; margin-bottom: 4px; }
.suggest-desc  { color: #555; font-size: 0.85em; line-height: 1.4; }
.trend-card {
    background: #fafafa; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 12px 14px; margin-bottom: 4px;
}
.trend-cat {
    display: inline-block; background: #fee2e2; color: #991b1b;
    font-size: 0.68em; font-weight: 600; padding: 2px 9px;
    border-radius: 20px; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 5px;
}
.q-count { font-size: 0.88em; color: #6b7280; margin: 4px 0 12px 0; }
.result-card {
    background: #f8faff; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 14px 18px; margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Clés & clients ─────────────────────────────────────────────────────────────
def get_secret(key):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    from dotenv import load_dotenv
    load_dotenv()
    val = os.getenv(key, "")
    return val if val and "REMPLACE" not in val else None


anthropic_key    = get_secret("ANTHROPIC_API_KEY")
openai_key       = get_secret("OPENAI_API_KEY")
perplexity_key   = get_secret("PERPLEXITY_API_KEY")

claude_client     = anthropic.Anthropic(api_key=anthropic_key) if anthropic_key  else None
openai_client     = OpenAI(api_key=openai_key)                 if openai_key     else None
perplexity_client = OpenAI(
    api_key=perplexity_key,
    base_url="https://api.perplexity.ai"
) if perplexity_key else None


# ── Concurrents ────────────────────────────────────────────────────────────────
competitors_list = (
    pd.read_csv("competitors.csv")["competitor"].tolist()
    if os.path.exists("competitors.csv") else []
)


# ── Session state : pool de questions ─────────────────────────────────────────
if "prompts_pool" not in st.session_state:
    if os.path.exists("prompts.csv"):
        _df = pd.read_csv("prompts.csv")
        _df["selected"] = True
        st.session_state["prompts_pool"] = _df[["prompt", "category", "selected"]].copy()
    else:
        st.session_state["prompts_pool"] = pd.DataFrame(
            columns=["prompt", "category", "selected"])


# ── Fonctions de requête ───────────────────────────────────────────────────────
def query_claude(prompt, model_id, client):
    r = client.messages.create(
        model=model_id, max_tokens=800,
        messages=[{"role": "user", "content": prompt}])
    return r.content[0].text

def query_openai(prompt, model_id, client):
    kwargs = dict(model=model_id, messages=[{"role": "user", "content": prompt}])
    try:
        r = client.chat.completions.create(**kwargs, max_completion_tokens=800)
    except Exception:
        r = client.chat.completions.create(**kwargs, max_tokens=800)
    return r.choices[0].message.content

def query_perplexity(prompt, model_id, client):
    r = client.chat.completions.create(
        model=model_id, max_tokens=800,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content

def query_claude_web(prompt, model_id, client):
    """Claude avec recherche web (tool built-in Anthropic web_search_20250305)."""
    messages = [{"role": "user", "content": prompt}]
    for _ in range(10):
        r = client.messages.create(
            model=model_id,
            max_tokens=1500,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages,
        )
        texts = [b.text for b in r.content
                 if getattr(b, "type", "") == "text" and hasattr(b, "text")]
        if r.stop_reason == "end_turn":
            return "\n".join(texts)
        if r.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": r.content})
            tool_results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": []}
                for b in r.content if getattr(b, "type", "") == "tool_use"
            ]
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        else:
            return "\n".join(texts)
    return "\n".join(texts) if texts else ""

def run_llm(prompt, llm):
    """Route vers le bon client selon la source du LLM."""
    if llm["source"] == "anthropic":
        return query_claude(prompt, llm["model"], claude_client)
    elif llm["source"] == "anthropic_web":
        return query_claude_web(prompt, llm["model"], claude_client)
    elif llm["source"] == "perplexity":
        return query_perplexity(prompt, llm["model"], perplexity_client)
    else:
        return query_openai(prompt, llm["model"], openai_client)


# ── Détection Python de RougeGorge (gratuit, avant tout appel API) ────────────
def rg_in_text(text):
    return bool(re.search(r"rouge[\s\-]?gorge", text, re.IGNORECASE))


# ── Analyse avec Claude Sonnet ────────────────────────────────────────────────
def analyze_response(prompt, response, competitors, client):
    # Pré-check Python : RG trouvée ?
    rg_found_python = rg_in_text(response)

    # Envoie la réponse complète si RG détectée, sinon cap à 8000 chars
    full_text = response if rg_found_python else response[:8000]

    system = f"""Tu es un expert en visibilité de marque dans les réponses IA.
Tu analyses des réponses pour mesurer la présence de RougeGorge.
Tu réponds UNIQUEMENT en JSON valide. Concurrents surveillés : {', '.join(competitors)}"""

    hint = ("⚠️ NOTE : le texte Python contient 'RougeGorge' — cherche-la attentivement."
            if rg_found_python else "")

    request = f"""Question : {prompt}
{hint}
Réponse COMPLÈTE à analyser :
{full_text}

JSON :
{{
  "rougegorge_mentionnee": true/false,
  "position_citation": "première"/"parmi d'autres"/"en dernier"/"absente",
  "concurrents_mentionnes": ["marque"],
  "tonalite": "positive"/"neutre"/"negative",
  "score_visibilite": 0-100,
  "recommandation": "action GEO concrète"
}}
Score : 0=absente, 25=brièvement citée, 50=clairement citée, 75=bonne option, 100=premier choix"""

    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=600,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": request}])
    raw = r.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
    try:
        result = json.loads(raw)
        # Filet de sécurité : si Python trouve RG mais Sonnet dit absente → forcer
        if rg_found_python and not result.get("rougegorge_mentionnee"):
            result["rougegorge_mentionnee"] = True
            if result.get("position_citation") == "absente":
                result["position_citation"] = "parmi d'autres"
            if result.get("score_visibilite", 0) == 0:
                result["score_visibilite"] = 25
        return result
    except Exception:
        return {"rougegorge_mentionnee": rg_found_python,
                "position_citation": "parmi d'autres" if rg_found_python else "absente",
                "concurrents_mentionnes": [], "tonalite": "neutre",
                "score_visibilite": 25 if rg_found_python else 0,
                "recommandation": "Erreur de parsing JSON"}


# ── Génération de questions tendance ──────────────────────────────────────────
def generate_trending_questions(client, n=12):
    r = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=1200,
        messages=[{"role": "user", "content": f"""Tu es expert en comportement consommateur lingerie, Europe 2026.
Génère {n} questions authentiques que des gens posent aux IA pour trouver de la lingerie.
Sois varié : cadeaux, occasions, tailles, confort, sport, prix, tendances, durabilité...
Formule comme un vrai utilisateur — naturel, varié, pas toutes sur le même modèle.

Réponds UNIQUEMENT en JSON :
{{"questions": [{{"prompt": "...", "category": "..."}}]}}

Catégories possibles : decouverte | cadeau | confort | taille_plus | sport | nuit | comparaison | prix | occasion | boutique | mariage | tendance | durabilite"""}])
    raw = r.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)["questions"]


# ── Benchmark complet ──────────────────────────────────────────────────────────
def run_benchmark_streamlit(selected_prompts_df, selected_llms):
    total, results, count = len(selected_prompts_df) * len(selected_llms), [], 0
    bar    = st.progress(0)
    status = st.empty()

    for llm in selected_llms:
        for _, row in selected_prompts_df.iterrows():
            prompt, cat = row["prompt"], row.get("category", "général")
            count += 1
            bar.progress(count / total, text=f"{llm['name']} — {cat} ({count}/{total})")
            status.markdown(f"*{str(prompt)[:90]}...*")

            try:
                answer = run_llm(prompt, llm)
            except Exception as e:
                answer = ""
                st.warning(f"{llm['name']} : {e}")

            analysis = (analyze_response(prompt, answer, competitors_list, claude_client)
                        if answer else
                        {"rougegorge_mentionnee": False, "position_citation": "absente",
                         "concurrents_mentionnes": [], "tonalite": "neutre",
                         "score_visibilite": 0, "recommandation": "Erreur de requête"})

            results.append({
                "date":                   datetime.now().strftime("%Y-%m-%d %H:%M"),
                "llm":                    llm["name"],
                "category":               cat,
                "prompt":                 prompt,
                "response":               answer,
                "rougegorge_mentionnee":  analysis.get("rougegorge_mentionnee", False),
                "position_citation":      analysis.get("position_citation", "absente"),
                "concurrents_mentionnes": ", ".join(analysis.get("concurrents_mentionnes", [])),
                "tonalite":               analysis.get("tonalite", "neutre"),
                "score_visibilite":       analysis.get("score_visibilite", 0),
                "recommandation":         analysis.get("recommandation", ""),
            })

    bar.progress(1.0, text="✅ Terminé !")
    status.empty()
    return pd.DataFrame(results)


# ── Suggestions d'articles de blog ───────────────────────────────────────────
def generate_blog_suggestions(prompt, results, client):
    valid = [r for r in results if not r.get("error") and r.get("response")]
    if not valid:
        return []
    summary = "\n".join(
        f"- {r['llm']}: {r['score']}/100 — RG {'citée' if r['cited'] else 'absente'} "
        f"({r['position']})"
        for r in valid)
    cited_anywhere = any(r["cited"] for r in valid)
    r = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=1200,
        messages=[{"role": "user", "content": f"""Tu es expert GEO (Generative Engine Optimization) pour RougeGorge, marque de lingerie belge/française.

Question testée : "{prompt}"
Visibilité RougeGorge par IA :
{summary}

RougeGorge est {"parfois citée mais" if cited_anywhere else "peu ou pas citée —"} il faut améliorer sa présence sur ce type de requête.

Propose 4 articles de blog que RougeGorge devrait produire pour apparaître dans les réponses IA sur cette requête.
Chaque article doit être : informatif, factuel, citable par les IA, avec des données concrètes.

Réponds UNIQUEMENT en JSON :
{{"articles": [
  {{
    "titre": "...",
    "angle": "en 1 phrase : le positionnement éditorial de l'article",
    "mots_cles": ["...", "...", "..."],
    "impact": "fort" ou "moyen" ou "faible",
    "pourquoi": "en 1 phrase : pourquoi cet article ferait citer RougeGorge par les IA"
  }}
]}}"""}])
    raw = r.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)["articles"]
    except Exception:
        return []


# ── Test d'une seule question ──────────────────────────────────────────────────
def run_single_question(prompt, selected_llms):
    results, prog = [], st.progress(0)
    for i, llm in enumerate(selected_llms):
        prog.progress((i + 1) / len(selected_llms), text=f"Test sur {llm['name']}...")
        error_msg = ""
        try:
            answer = run_llm(prompt, llm)
        except Exception as e:
            answer = ""
            error_msg = str(e)
        analysis = (analyze_response(prompt, answer, competitors_list, claude_client)
                    if answer else
                    {"rougegorge_mentionnee": False, "position_citation": "absente",
                     "concurrents_mentionnes": [], "tonalite": "neutre",
                     "score_visibilite": 0, "recommandation": ""})
        results.append({
            "llm":            llm["name"],
            "score":          analysis.get("score_visibilite", 0),
            "cited":          analysis.get("rougegorge_mentionnee", False),
            "position":       analysis.get("position_citation", "absente"),
            "tonalite":       analysis.get("tonalite", "neutre"),
            "recommandation": analysis.get("recommandation", ""),
            "response":       answer,
            "error":          error_msg,
        })
    prog.empty()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")

    if not anthropic_key:
        st.error("ANTHROPIC_API_KEY manquante dans .env")
        st.stop()

    selected_llms: list = []

    st.markdown("**Claude — base :**")
    for llm in LLMS_CLAUDE:
        if st.checkbox(llm["name"], value=True, key=f"chk_{llm['name']}"):
            selected_llms.append(llm)
    st.markdown("**Claude — recherche web 🔍 :**")
    for llm in LLMS_CLAUDE_WEB:
        if st.checkbox(llm["name"], value=True, key=f"chk_{llm['name']}"):
            selected_llms.append(llm)

    if openai_key:
        st.markdown("**OpenAI — base :**")
        for llm in LLMS_OPENAI:
            if st.checkbox(llm["name"], value=True, key=f"chk_{llm['name']}"):
                selected_llms.append(llm)
        st.markdown("**OpenAI — recherche web 🔍 :**")
        for llm in LLMS_OPENAI_SEARCH:
            if st.checkbox(llm["name"], value=True, key=f"chk_{llm['name']}"):
                selected_llms.append(llm)
        with st.expander("Diagnostic OpenAI"):
            if st.button("Lister mes modèles disponibles", use_container_width=True):
                try:
                    models = openai_client.models.list()
                    gpt_ids = sorted(
                        m.id for m in models.data
                        if any(x in m.id for x in ("gpt", "o1", "o3", "o4")))
                    if gpt_ids:
                        st.code("\n".join(gpt_ids))
                    else:
                        st.info("Aucun modèle GPT trouvé.")
                except Exception as e:
                    st.error(str(e))
    else:
        st.caption("_Ajoute OPENAI_API_KEY pour tester GPT-5_")

    if perplexity_key:
        st.markdown("**Perplexity (web natif) 🔍 :**")
        for llm in LLMS_PERPLEXITY:
            if st.checkbox(llm["name"], value=True, key=f"chk_{llm['name']}"):
                selected_llms.append(llm)
    else:
        st.caption("_Ajoute PERPLEXITY_API_KEY pour tester Perplexity_")

    st.divider()

    pool     = st.session_state["prompts_pool"]
    n_sel    = int(pool["selected"].sum()) if not pool.empty else 0
    n_llms   = len(selected_llms)
    n_req    = n_sel * n_llms

    st.caption(
        f"**{n_sel}** questions sélectionnées  \n"
        f"**{n_llms}** LLMs  \n"
        f"→ **{n_req} requêtes** au total"
    )

    if st.button("🚀 Lancer le benchmark", type="primary",
                 use_container_width=True, disabled=not selected_llms or n_sel == 0):
        try:
            sel_df = pool[pool["selected"]].copy()
            df_new = run_benchmark_streamlit(sel_df, selected_llms)
            os.makedirs("data", exist_ok=True)
            df_new.to_csv("data/analyzed_results.csv", index=False, encoding="utf-8")
            st.session_state["df"] = df_new
            st.success(f"Score moyen : {df_new['score_visibilite'].mean():.1f}/100")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur : {e}")

    st.divider()
    st.caption("🌹 RougeGorge AI Visibility · 2026")


# ══════════════════════════════════════════════════════════════════════════════
# TITRE + ONGLETS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🌹 RougeGorge — AI Visibility Monitor")

tab_dash, tab_questions = st.tabs(["📊 Résultats & Analyses", "❓ Questions"])


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_questions:

    # ── 1. Questions tendance ──────────────────────────────────────────────────
    st.markdown("### 🔥 Questions tendance")
    st.caption("Génère les questions les plus posées aux IA sur la lingerie en 2026")

    cg1, cg2, cg3 = st.columns([1, 1, 3])
    with cg1:
        n_gen = st.selectbox("Nombre", [8, 12, 16, 20], index=1, key="n_gen")
    with cg2:
        st.write("")
        st.write("")
        gen_btn = st.button("✨ Générer", use_container_width=True, key="btn_generate")

    if gen_btn:
        with st.spinner("Génération en cours..."):
            try:
                st.session_state["trending"] = generate_trending_questions(
                    claude_client, n_gen)
            except Exception as e:
                st.error(f"Erreur : {e}")

    if st.session_state.get("trending"):
        trending = st.session_state["trending"]
        pool_prompts = set(st.session_state["prompts_pool"]["prompt"].tolist())
        cols_t = st.columns(2)
        for i, q in enumerate(trending):
            with cols_t[i % 2]:
                already = q["prompt"] in pool_prompts
                st.markdown(
                    f'<div class="trend-card">'
                    f'<span class="trend-cat">{q.get("category","autre")}</span><br>'
                    f'<span style="font-size:0.88em;line-height:1.5">{q["prompt"]}</span>'
                    f'</div>', unsafe_allow_html=True)
                if already:
                    st.caption("✓ Déjà dans le pool")
                else:
                    if st.button("＋ Ajouter au pool", key=f"add_t_{i}",
                                 use_container_width=True):
                        new_row = pd.DataFrame([{
                            "prompt": q["prompt"],
                            "category": q.get("category", "autre"),
                            "selected": True,
                        }])
                        st.session_state["prompts_pool"] = pd.concat(
                            [st.session_state["prompts_pool"], new_row],
                            ignore_index=True)
                        st.rerun()

    st.divider()

    # ── 2. Question personnalisée ──────────────────────────────────────────────
    st.markdown("### ✏️ Tester une question")
    st.caption("La question est envoyée telle quelle aux IA — aucun filtre n'est appliqué")

    custom_q = st.text_input(
        "Ta question", key="custom_q",
        placeholder="Ex : lingerie grande taille bonnet G",
        label_visibility="collapsed")

    cb1, cb2, cb3 = st.columns([1, 1, 3])
    with cb1:
        test_btn = st.button("▶ Tester maintenant", use_container_width=True,
                             type="primary",
                             disabled=not (custom_q or "").strip() or not selected_llms)
    with cb2:
        add_btn = st.button("＋ Ajouter au pool", use_container_width=True,
                            disabled=not (custom_q or "").strip())

    if add_btn and custom_q.strip():
        # Demander la catégorie seulement au moment d'ajouter
        pool_prompts = set(st.session_state["prompts_pool"]["prompt"].tolist())
        if custom_q.strip() in pool_prompts:
            st.warning("Cette question est déjà dans le pool.")
        else:
            add_cat = st.selectbox(
                "Catégorie (pour organiser dans le pool)",
                CATEGORIES, key="add_cat_select")
            if st.button("Confirmer l'ajout", key="confirm_add", type="primary"):
                new_row = pd.DataFrame([{
                    "prompt": custom_q.strip(), "category": add_cat, "selected": True}])
                st.session_state["prompts_pool"] = pd.concat(
                    [st.session_state["prompts_pool"], new_row], ignore_index=True)
                st.success(f"Question ajoutée dans «{add_cat}»")
                st.rerun()

    if test_btn and custom_q.strip() and selected_llms:
        with st.spinner(f"Test sur {len(selected_llms)} LLMs..."):
            try:
                st.session_state["single_results"] = run_single_question(
                    custom_q.strip(), selected_llms)
                st.session_state["single_prompt"] = custom_q.strip()
            except Exception as e:
                st.error(f"Erreur : {e}")

    if st.session_state.get("single_results"):
        st.markdown(
            f"**Résultats pour :** *{st.session_state.get('single_prompt', '')}*")
        for r in st.session_state["single_results"]:
            score_color = ("#22c55e" if r["score"] >= 60
                           else "#f59e0b" if r["score"] >= 30 else "#ef4444")
            cited_label = "Citée" if r["cited"] else "Absente"
            cited_color = "#16a34a" if r["cited"] else "#dc2626"

            with st.container():
                c_llm, c_score, c_status, c_pos, c_ton = st.columns([2.5, 1.2, 1.2, 1.5, 1.2])
                with c_llm:
                    st.markdown(f"**{r['llm']}**")
                with c_score:
                    st.markdown(
                        f'<span style="font-size:1.25em;font-weight:700;color:{score_color}">'
                        f'{r["score"]}/100</span>', unsafe_allow_html=True)
                with c_status:
                    st.markdown(
                        f'<span style="color:{cited_color};font-weight:600">'
                        f'{cited_label}</span>', unsafe_allow_html=True)
                with c_pos:
                    st.caption(r["position"])
                with c_ton:
                    st.caption(r["tonalite"])

                if r.get("error"):
                    st.error(f"Erreur API : {r['error']}")
                else:
                    if r["recommandation"]:
                        st.markdown(
                            f'<div class="rec-card">💡 {r["recommandation"]}</div>',
                            unsafe_allow_html=True)
                    if r["response"]:
                        with st.expander("Voir la réponse complète"):
                            display_text = r["response"][:2500]
                            if len(r["response"]) > 2500:
                                display_text += "\n\n*[réponse tronquée à 2500 caractères]*"
                            highlighted = re.sub(
                                r"(rougegorge|rouge[\s\-]gorge)",
                                r'<mark style="background:#fee2e2;border-radius:3px;'
                                r'padding:1px 4px;font-weight:600">\1</mark>',
                                display_text, flags=re.IGNORECASE)
                            st.markdown(highlighted, unsafe_allow_html=True)
                st.divider()

    # ── Articles de blog suggérés ──────────────────────────────────────────────
    if st.session_state.get("single_results"):
        valid_r = [r for r in st.session_state["single_results"]
                   if not r.get("error") and r.get("response")]
        if valid_r:
            st.markdown("---")
            st.markdown("### 📝 Articles de blog pour améliorer cette visibilité")
            st.caption("Contenus GEO à produire pour que les IA citent RougeGorge sur cette requête")

            if st.button("Générer les suggestions d'articles", key="btn_blog",
                         use_container_width=False):
                with st.spinner("Génération en cours..."):
                    try:
                        arts = generate_blog_suggestions(
                            st.session_state.get("single_prompt", ""),
                            st.session_state["single_results"], claude_client)
                        st.session_state["blog_articles"] = arts
                        st.session_state["blog_for"] = st.session_state.get("single_prompt")
                    except Exception as e:
                        st.error(f"Erreur : {e}")

            if (st.session_state.get("blog_articles") and
                    st.session_state.get("blog_for") == st.session_state.get("single_prompt")):
                impact_color = {"fort": "#16a34a", "moyen": "#d97706", "faible": "#94a3b8"}
                cols_b = st.columns(2)
                for i, art in enumerate(st.session_state["blog_articles"]):
                    col = cols_b[i % 2]
                    ic  = impact_color.get(art.get("impact", "moyen"), "#888")
                    with col:
                        st.markdown(f"""<div class="suggest-card">
<div class="suggest-title">{art.get("titre","")}</div>
<div class="suggest-desc" style="margin-top:4px">{art.get("angle","")}</div>
<div style="margin-top:8px;font-size:0.8em;color:#555">
  🔑 {' · '.join(art.get("mots_cles",[]))}
</div>
<div style="margin-top:6px;font-size:0.8em;font-weight:600;color:{ic}">
  Impact estimé : {art.get("impact","?")}
</div>
<div style="margin-top:4px;font-size:0.78em;color:#6b7280;font-style:italic">
  {art.get("pourquoi","")}
</div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── 3. Sélection des questions ─────────────────────────────────────────────
    st.markdown("### ✅ Sélection des questions")
    st.caption("Choisis les questions à inclure dans le prochain benchmark")

    pool = st.session_state["prompts_pool"]
    n_total = len(pool)
    n_selected_q = int(pool["selected"].sum()) if not pool.empty else 0

    st.markdown(
        f'<p class="q-count">'
        f'<strong>{n_selected_q}</strong> / {n_total} questions sélectionnées'
        f'</p>', unsafe_allow_html=True)

    cs1, cs2, cs3, cs4 = st.columns([1, 1, 2, 1])
    with cs1:
        if st.button("☑ Tout sélectionner", use_container_width=True):
            st.session_state["prompts_pool"]["selected"] = True
            st.rerun()
    with cs2:
        if st.button("☐ Tout désélectionner", use_container_width=True):
            st.session_state["prompts_pool"]["selected"] = False
            st.rerun()
    with cs4:
        if st.button("💾 Sauvegarder CSV", use_container_width=True):
            save_df = pool[["prompt", "category"]].copy()
            save_df.to_csv("prompts.csv", index=False, encoding="utf-8")
            st.success("prompts.csv mis à jour !")

    if not pool.empty:
        edited = st.data_editor(
            pool[["selected", "category", "prompt"]].reset_index(drop=True),
            column_config={
                "selected": st.column_config.CheckboxColumn(
                    "✓", width="small", default=True),
                "category": st.column_config.SelectboxColumn(
                    "Catégorie", options=CATEGORIES, width="medium"),
                "prompt": st.column_config.TextColumn(
                    "Question", width="large"),
            },
            hide_index=True,
            use_container_width=True,
            height=min(560, max(200, len(pool) * 40 + 60)),
            num_rows="dynamic",
            key="pool_editor",
        )
        if edited is not None:
            clean = edited.dropna(subset=["prompt"])
            clean = clean[clean["prompt"].str.strip() != ""]
            clean["selected"] = clean["selected"].fillna(True)
            clean["category"] = clean["category"].fillna("autre")
            st.session_state["prompts_pool"] = clean.reset_index(drop=True)
    else:
        st.info("Pool vide — génère des questions tendance ou ajoute-en manuellement.")


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    DATA_FILE = "data/analyzed_results.csv"
    if "df" in st.session_state:
        df = st.session_state["df"]
    elif os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        st.info(
            "Aucun résultat pour l'instant.  \n"
            "➡️ Configure tes questions dans l'onglet **❓ Questions**, "
            "puis clique sur **🚀 Lancer le benchmark** dans le menu à gauche.")
        st.stop()

    if df.empty:
        st.warning("Le fichier de résultats est vide.")
        st.stop()

    available_llms = sorted(df["llm"].unique().tolist()) if "llm" in df.columns else []
    last_date = df["date"].max() if "date" in df.columns else ""
    st.caption(
        f"Dernière analyse : {last_date} · "
        f"{len(df)} réponses · {' · '.join(available_llms)}")
    st.divider()

    # ── KPIs ──────────────────────────────────────────────────────────────────
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

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Score global",       f"{score_moyen:.1f} / 100")
    with k2: st.metric("Taux de citation",   f"{pct:.0f}%", f"{mentions} / {len(df)}")
    with k3: st.metric("Share of voice",     f"{sov}%", "vs concurrents")
    with k4: st.metric("Tonalité positive",  f"{positives} réponses")
    with k5: st.metric("Concurrent n°1",     top_comp)
    st.divider()

    # ── Comparaison LLMs ──────────────────────────────────────────────────────
    if len(available_llms) > 1:
        st.markdown("### 🤖 Comparaison entre les IA")
        llm_stats = (
            df.groupby("llm")
              .agg(
                  score_moyen=("score_visibilite", "mean"),
                  taux_citation=("rougegorge_mentionnee",
                                 lambda x: round(x.mean() * 100, 1)))
              .reset_index()
              .sort_values("score_moyen", ascending=False))
        cmap = {r["llm"]: LLM_COLORS.get(r["llm"], "#888") for _, r in llm_stats.iterrows()}

        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Score moyen de visibilité**")
            fig = px.bar(llm_stats, x="llm", y="score_moyen", text="score_moyen",
                         color="llm", color_discrete_map=cmap)
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(height=280, showlegend=False, yaxis_range=[0, 115],
                              xaxis_title=None, yaxis_title="Score / 100",
                              margin=dict(t=20, b=10),
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Poppins, sans-serif"))
            st.plotly_chart(fig, use_container_width=True)
        with cb:
            st.markdown("**Taux de citation de RougeGorge**")
            fig2 = px.bar(llm_stats, x="llm", y="taux_citation", text="taux_citation",
                          color="llm", color_discrete_map=cmap)
            fig2.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig2.update_layout(height=280, showlegend=False, yaxis_range=[0, 115],
                               xaxis_title=None, yaxis_title="% prompts",
                               margin=dict(t=20, b=10),
                               paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Poppins, sans-serif"))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Heatmap — score par IA et par catégorie**")
        heat  = df.groupby(["category", "llm"])["score_visibilite"].mean().reset_index()
        pivot = heat.pivot(index="category", columns="llm",
                           values="score_visibilite").fillna(0)
        fig_h = px.imshow(
            pivot, color_continuous_scale=["#fee2e2", "#fef3c7", "#dcfce7"],
            zmin=0, zmax=100, text_auto=".0f", aspect="auto")
        fig_h.update_layout(height=max(250, len(pivot) * 36),
                             margin=dict(t=20, b=20),
                             paper_bgcolor="rgba(0,0,0,0)",
                             coloraxis_showscale=False,
                             font=dict(family="Poppins, sans-serif"))
        st.plotly_chart(fig_h, use_container_width=True)
        st.divider()

    # ── Jauge + Tonalité ───────────────────────────────────────────────────────
    cg, ct = st.columns(2)
    with cg:
        st.markdown("**Score global**")
        color = ("#22c55e" if score_moyen >= 60
                 else "#f59e0b" if score_moyen >= 30 else "#ef4444")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=round(score_moyen, 1),
            number={"suffix": " / 100", "font": {"size": 28, "color": color,
                                                  "family": "Poppins, sans-serif"}},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": color, "thickness": 0.3},
                   "steps": [{"range": [0, 30],  "color": "#fee2e2"},
                              {"range": [30, 60], "color": "#fef3c7"},
                              {"range": [60, 100],"color": "#dcfce7"}]}))
        fig_gauge.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20),
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 font=dict(family="Poppins, sans-serif"))
        st.plotly_chart(fig_gauge, use_container_width=True)
    with ct:
        st.markdown("**Tonalité des réponses**")
        ton = df["tonalite"].value_counts().reset_index()
        ton.columns = ["Tonalité", "Nb"]
        fig_ton = px.pie(ton, values="Nb", names="Tonalité", hole=0.55,
                         color="Tonalité",
                         color_discrete_map={"positive": "#22c55e",
                                             "neutre": "#94a3b8",
                                             "negative": "#ef4444"})
        fig_ton.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10),
                               legend=dict(orientation="h", y=-0.15),
                               paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Poppins, sans-serif"))
        st.plotly_chart(fig_ton, use_container_width=True)
    st.divider()

    # ── Scores par catégorie ───────────────────────────────────────────────────
    st.markdown("### 📊 Score par catégorie")
    score_cat = (df.groupby("category")["score_visibilite"]
                   .mean().sort_values(ascending=True).reset_index())
    score_cat.columns = ["Catégorie", "Score"]
    fig_cat = px.bar(score_cat, x="Score", y="Catégorie", orientation="h",
                     text=score_cat["Score"].apply(lambda s: f"{s:.0f}"),
                     color="Score",
                     color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                     range_color=[0, 100])
    fig_cat.update_traces(textposition="outside")
    fig_cat.update_layout(
        height=max(280, len(score_cat) * 40), coloraxis_showscale=False,
        xaxis=dict(range=[0, 120]), yaxis_title=None,
        margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif"))
    st.plotly_chart(fig_cat, use_container_width=True)
    st.divider()

    # ── Part de voix ───────────────────────────────────────────────────────────
    if all_comp:
        st.markdown("### 🎯 Part de voix — RougeGorge vs concurrents")
        comp_counts = pd.Series(all_comp).value_counts().reset_index()
        comp_counts.columns = ["Marque", "Citations"]
        rg_row = pd.DataFrame([{"Marque": "🌹 RougeGorge", "Citations": mentions}])
        comp_chart = (pd.concat([rg_row, comp_counts], ignore_index=True)
                        .sort_values("Citations"))
        comp_chart["type"] = comp_chart["Marque"].apply(
            lambda x: "RougeGorge" if "RougeGorge" in x else "Concurrent")
        fig_comp = px.bar(comp_chart, x="Citations", y="Marque", orientation="h",
                          text="Citations", color="type",
                          color_discrete_map={"RougeGorge": BRAND_COLOR,
                                              "Concurrent": "#cbd5e1"})
        fig_comp.update_traces(textposition="outside")
        fig_comp.update_layout(
            height=max(300, len(comp_chart) * 36), showlegend=False,
            xaxis_title="Citations", yaxis_title=None,
            margin=dict(t=10, b=10, l=10, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Poppins, sans-serif"))
        st.plotly_chart(fig_comp, use_container_width=True)
        st.divider()

    # ── Tableau détaillé ───────────────────────────────────────────────────────
    st.markdown("### 📋 Résultats détaillés")
    tf1, tf2, tf3, tf4 = st.columns(4)
    with tf1: f_llm  = st.selectbox("IA",        ["Tous"] + available_llms)
    with tf2: f_cat  = st.selectbox("Catégorie", ["Toutes"] + sorted(
        df["category"].dropna().unique().tolist()))
    with tf3: f_ton  = st.selectbox("Tonalité",  ["Toutes", "positive", "neutre", "negative"])
    with tf4: only_rg = st.checkbox("Seulement où RG est citée")

    filtered = df.copy()
    if f_llm   != "Tous":   filtered = filtered[filtered["llm"]      == f_llm]
    if f_cat   != "Toutes": filtered = filtered[filtered["category"] == f_cat]
    if f_ton   != "Toutes": filtered = filtered[filtered["tonalite"] == f_ton]
    if only_rg:             filtered = filtered[filtered["rougegorge_mentionnee"] == True]

    cols_show = [c for c in [
        "llm", "category", "prompt", "score_visibilite", "position_citation",
        "rougegorge_mentionnee", "tonalite", "concurrents_mentionnes", "recommandation"]
        if c in filtered.columns]
    rename = {
        "llm": "IA", "category": "Catégorie", "prompt": "Prompt",
        "score_visibilite": "Score", "position_citation": "Position RG",
        "rougegorge_mentionnee": "RG citée ?", "tonalite": "Tonalité",
        "concurrents_mentionnes": "Concurrents", "recommandation": "Recommandation GEO",
    }
    display = filtered[cols_show].rename(columns=rename)

    def color_score(val):
        try:
            v = float(val)
            if v >= 60: return "background-color:#dcfce7"
            if v >= 30: return "background-color:#fef3c7"
            return "background-color:#fee2e2"
        except Exception:
            return ""

    st.dataframe(
        display.style.map(color_score, subset=["Score"]) if "Score" in display.columns
        else display,
        use_container_width=True, hide_index=True, height=380)
    st.download_button(
        "⬇️ Télécharger CSV",
        data=df.to_csv(index=False, encoding="utf-8"),
        file_name=f"rougegorge_visibility_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv")
    st.divider()

    # ── Recommandations ────────────────────────────────────────────────────────
    st.markdown("### 💡 Recommandations GEO prioritaires")
    recs = [r for r in df["recommandation"].dropna().unique()
            if r and r != "Erreur d'analyse"]
    for i, rec in enumerate(recs[:6], 1):
        st.markdown(
            f'<div class="rec-card"><strong>{i}.</strong> {rec}</div>',
            unsafe_allow_html=True)
    st.divider()

    # ── Suggestions ────────────────────────────────────────────────────────────
    st.markdown("### 🔬 Idées pour aller plus loin")
    suggestions = [
        ("📅 Suivi hebdomadaire",
         "Relancer chaque semaine pour mesurer l'impact de tes actions GEO dans le temps."),
        ("🌍 Multi-langue",
         "Tester en anglais, néerlandais, espagnol pour mesurer la visibilité internationale."),
        ("📍 Prompts géolocalisés",
         "Ajouter 'à Paris', 'en Belgique', 'à Lyon' pour mesurer la visibilité locale."),
        ("🎯 Requêtes de niche",
         "Lingerie de mariage, post-mastectomie, grandes tailles, slow fashion..."),
        ("🗣️ Prompts conversationnels",
         "Formulations naturelles : 'j'ai besoin d'un soutien-gorge confortable'..."),
        ("⭐ Réputation & avis",
         "Demander 'quelle marque a les meilleurs avis ?' pour mesurer l'association RG ↔ satisfaction."),
        ("🛍️ Par type de produit",
         "Segmenter : soutiens-gorge, culottes, nuisettes, bain, sport, maternité."),
        ("📰 Contenu GEO ciblé",
         "Créer des pages optimisées pour les requêtes où RG est absente des réponses IA."),
    ]
    cs1, cs2 = st.columns(2)
    for i, (titre, desc) in enumerate(suggestions):
        with (cs1 if i % 2 == 0 else cs2):
            st.markdown(
                f'<div class="suggest-card">'
                f'<div class="suggest-title">{titre}</div>'
                f'<div class="suggest-desc">{desc}</div>'
                f'</div>', unsafe_allow_html=True)
