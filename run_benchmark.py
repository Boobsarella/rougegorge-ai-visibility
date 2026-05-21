"""
run_benchmark.py — Benchmark multi-LLM RougeGorge AI Visibility

Interroge tous les LLMs configurés (Claude, GPT-4o, Gemini, Perplexity)
et analyse les réponses avec Claude pour mesurer la visibilité de RougeGorge.

Usage : python run_benchmark.py
"""

import os
import json
import pandas as pd
import anthropic
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── Modèle d'analyse (toujours Claude Haiku, économique) ──────────────────────
ANALYSIS_MODEL = "claude-haiku-4-5-20251001"
BRAND = "RougeGorge"


# ── Fonctions de requête par LLM ──────────────────────────────────────────────

def query_claude(prompt, api_key):
    import anthropic as ant
    client = ant.Anthropic(api_key=api_key)
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text


def query_gpt4o(prompt, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content


def query_gemini(prompt, api_key):
    from google import genai
    client = genai.Client(api_key=api_key)
    r = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return r.text


def query_perplexity(prompt, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    r = client.chat.completions.create(
        model="sonar",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content


# ── Configuration des LLMs disponibles ────────────────────────────────────────
LLM_CONFIG = {
    "Claude Haiku":       {"fn": query_claude,      "env": "ANTHROPIC_API_KEY"},
    "GPT-4o":             {"fn": query_gpt4o,        "env": "OPENAI_API_KEY"},
    "Gemini 2.0 Flash":   {"fn": query_gemini,       "env": "GOOGLE_API_KEY"},
    "Perplexity Sonar":   {"fn": query_perplexity,   "env": "PERPLEXITY_API_KEY"},
}


def get_active_llms():
    """Retourne uniquement les LLMs dont la clé API est configurée."""
    active = {}
    for name, config in LLM_CONFIG.items():
        key = os.getenv(config["env"])
        if key and not key.startswith("sk-ant-REMPLACE") and not key.startswith("sk-REMPLACE") \
                and not key.startswith("AI-REMPLACE") and not key.startswith("pplx-REMPLACE"):
            active[name] = {"fn": config["fn"], "key": key}
    return active


# ── Analyse avec Claude ────────────────────────────────────────────────────────

def analyze_response(prompt, response, competitors, claude_client):
    """Analyse une réponse LLM pour mesurer la visibilité de RougeGorge."""
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
  "concurrents_mentionnes": ["marque1", "marque2"],
  "tonalite": "positive" ou "neutre" ou "negative",
  "score_visibilite": 0 à 100,
  "recommandation": "action GEO concrète pour améliorer la visibilité de RougeGorge sur ce type de requête"
}}
Score : 0=absente, 25=brièvement citée, 50=clairement citée, 75=bonne option, 100=premier choix"""

    r = claude_client.messages.create(
        model=ANALYSIS_MODEL,
        max_tokens=500,
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


# ── Benchmark principal ────────────────────────────────────────────────────────

def run_benchmark():
    print("=" * 65)
    print("🌹 RougeGorge — Benchmark AI Visibility (multi-LLM)")
    print(f"   Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 65)

    os.makedirs("data", exist_ok=True)

    # Charge les données
    prompts_df = pd.read_csv("prompts.csv")
    competitors = pd.read_csv("competitors.csv")["competitor"].tolist()
    active_llms = get_active_llms()

    if not active_llms:
        print("❌ Aucune clé API configurée. Vérifie ton fichier .env")
        return

    print(f"\n✓ {len(prompts_df)} prompts | {len(competitors)} concurrents")
    print(f"✓ LLMs actifs : {', '.join(active_llms.keys())}\n")

    # Client Claude pour l'analyse (toujours requis)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY manquante (nécessaire pour l'analyse)")
        return
    claude_client = anthropic.Anthropic(api_key=anthropic_key)

    results = []
    total = len(prompts_df) * len(active_llms)
    count = 0

    for llm_name, llm in active_llms.items():
        print(f"\n{'─' * 50}")
        print(f"🤖 {llm_name}")
        print(f"{'─' * 50}")

        for _, row in prompts_df.iterrows():
            prompt = row["prompt"]
            category = row.get("category", "général")
            count += 1

            print(f"[{count}/{total}] {category} — {prompt[:60]}...")

            # Interroge le LLM
            try:
                answer = llm["fn"](prompt, llm["key"])
            except Exception as e:
                print(f"  ⚠️  Erreur {llm_name} : {e}")
                answer = ""

            # Analyse avec Claude
            analysis = analyze_response(prompt, answer, competitors, claude_client) if answer else \
                {"rougegorge_mentionnee": False, "position_citation": "absente",
                 "concurrents_mentionnes": [], "tonalite": "neutre",
                 "score_visibilite": 0, "recommandation": "Erreur de requête"}

            results.append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "llm": llm_name,
                "category": category,
                "prompt": prompt,
                "response": answer,
                "rougegorge_mentionnee": analysis.get("rougegorge_mentionnee", False),
                "position_citation": analysis.get("position_citation", "absente"),
                "concurrents_mentionnes": ", ".join(analysis.get("concurrents_mentionnes", [])),
                "tonalite": analysis.get("tonalite", "neutre"),
                "score_visibilite": analysis.get("score_visibilite", 0),
                "recommandation": analysis.get("recommandation", "")
            })

            score = analysis.get("score_visibilite", 0)
            cited = "✓" if analysis.get("rougegorge_mentionnee") else "✗"
            print(f"  → Score : {score}/100 | RG : {cited}")

    # Sauvegarde
    df = pd.DataFrame(results)
    df.to_csv("data/raw_results.csv", index=False, encoding="utf-8")
    df.to_csv("data/analyzed_results.csv", index=False, encoding="utf-8")

    # Résumé par LLM
    print("\n" + "=" * 65)
    print("✅ Benchmark terminé ! Résumé par LLM :")
    for llm_name, grp in df.groupby("llm"):
        score = grp["score_visibilite"].mean()
        mentions = grp["rougegorge_mentionnee"].sum()
        print(f"   {llm_name:<22} Score : {score:5.1f}/100  |  RG citée : {mentions}/{len(grp)}")
    print(f"\n   Lance le dashboard : streamlit run dashboard.py")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
