"""
run_benchmark.py — Benchmark multi-LLM RougeGorge AI Visibility

Interroge Claude et GPT directement via leurs APIs respectives
et analyse toutes les réponses avec Claude pour mesurer la visibilité de RougeGorge.

Clés nécessaires :
  - ANTHROPIC_API_KEY → Claude Sonnet 4.6, Claude Opus 4.7 + analyse des réponses
  - OPENAI_API_KEY    → GPT-5.5, GPT-5.4, GPT-5.4-mini (optionnel)

Usage : python run_benchmark.py
"""

import os
import json
import pandas as pd
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── LLMs Claude via Anthropic ──────────────────────────────────────────────────
LLMS_CLAUDE = [
    {"name": "Claude Sonnet 4.6", "model": "claude-sonnet-4-6", "source": "anthropic"},
    {"name": "Claude Opus 4.7",   "model": "claude-opus-4-7",   "source": "anthropic"},
]

# ── LLMs OpenAI via clé directe ───────────────────────────────────────────────
LLMS_OPENAI = [
    {"name": "GPT-5.5",      "model": "gpt-5.5",      "source": "openai"},
    {"name": "GPT-5.4",      "model": "gpt-5.4",      "source": "openai"},
    {"name": "GPT-5.4-mini", "model": "gpt-5.4-mini", "source": "openai"},
]

ANALYSIS_MODEL = "claude-haiku-4-5-20251001"
BRAND = "RougeGorge"


def query_claude(prompt, model_id, claude_client):
    """Interroge Claude via l'API Anthropic."""
    r = claude_client.messages.create(
        model=model_id, max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text

def query_openai(prompt, model_id, openai_client):
    """Interroge GPT via l'API OpenAI (max_completion_tokens requis pour GPT-5.x)."""
    r = openai_client.chat.completions.create(
        model=model_id,
        max_completion_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content


def analyze_response(prompt, response, competitors, claude_client):
    """Analyse une réponse LLM avec Claude pour mesurer la visibilité de RougeGorge."""
    competitors_str = ", ".join(competitors)

    system = f"""Tu es un expert en visibilité de marque dans les réponses IA.
Tu analyses des réponses pour mesurer la présence de {BRAND}.
Tu réponds UNIQUEMENT en JSON valide. Concurrents surveillés : {competitors_str}"""

    request = f"""Question : {prompt}
Réponse à analyser : {response}

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


def run_benchmark():
    print("=" * 65)
    print("🌹 RougeGorge — Benchmark AI Visibility")
    print(f"   Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 65)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key    = os.getenv("OPENAI_API_KEY")

    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY manquante dans .env")
        return

    claude_client  = anthropic.Anthropic(api_key=anthropic_key)
    openai_client  = OpenAI(api_key=openai_key) if openai_key else None

    all_llms = LLMS_CLAUDE + (LLMS_OPENAI if openai_client else [])

    if not openai_client:
        print("ℹ️  OPENAI_API_KEY absente — seuls les modèles Claude seront testés.")

    os.makedirs("data", exist_ok=True)
    prompts_df  = pd.read_csv("prompts.csv")
    competitors = pd.read_csv("competitors.csv")["competitor"].tolist()

    total = len(prompts_df) * len(all_llms)
    print(f"\n✓ {len(prompts_df)} prompts · {len(all_llms)} LLMs · {total} requêtes au total")
    print(f"✓ LLMs : {', '.join(l['name'] for l in all_llms)}\n")

    results = []
    count = 0

    for llm in all_llms:
        print(f"\n{'─' * 55}")
        print(f"🤖 {llm['name']}  ({llm['model']})")
        print(f"{'─' * 55}")

        for _, row in prompts_df.iterrows():
            prompt   = row["prompt"]
            category = row.get("category", "général")
            count   += 1

            print(f"[{count}/{total}] {category} — {prompt[:60]}...")

            try:
                if llm["source"] == "anthropic":
                    answer = query_claude(prompt, llm["model"], claude_client)
                else:
                    answer = query_openai(prompt, llm["model"], openai_client)
            except Exception as e:
                print(f"  ⚠️  Erreur : {e}")
                answer = ""

            if answer:
                analysis = analyze_response(prompt, answer, competitors, claude_client)
            else:
                analysis = {"rougegorge_mentionnee": False, "position_citation": "absente",
                            "concurrents_mentionnes": [], "tonalite": "neutre",
                            "score_visibilite": 0, "recommandation": "Erreur de requête"}

            results.append({
                "date":                   datetime.now().strftime("%Y-%m-%d %H:%M"),
                "llm":                    llm["name"],
                "category":               category,
                "prompt":                 prompt,
                "response":               answer,
                "rougegorge_mentionnee":  analysis.get("rougegorge_mentionnee", False),
                "position_citation":      analysis.get("position_citation", "absente"),
                "concurrents_mentionnes": ", ".join(analysis.get("concurrents_mentionnes", [])),
                "tonalite":               analysis.get("tonalite", "neutre"),
                "score_visibilite":       analysis.get("score_visibilite", 0),
                "recommandation":         analysis.get("recommandation", "")
            })

            score = analysis.get("score_visibilite", 0)
            cited = "✓ Citée" if analysis.get("rougegorge_mentionnee") else "✗ Absente"
            print(f"  → Score : {score}/100 | RG : {cited}")

    df = pd.DataFrame(results)
    df.to_csv("data/raw_results.csv",      index=False, encoding="utf-8")
    df.to_csv("data/analyzed_results.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print("✅ Benchmark terminé ! Résumé :")
    for llm_name, grp in df.groupby("llm"):
        score    = grp["score_visibilite"].mean()
        mentions = grp["rougegorge_mentionnee"].sum()
        print(f"   {llm_name:<28} {score:5.1f}/100  |  RG citée : {mentions}/{len(grp)}")
    print(f"\n   Dashboard : streamlit run dashboard.py")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
