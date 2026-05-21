"""
run_benchmark.py — Script principal du benchmark RougeGorge AI Visibility

Usage : python run_benchmark.py
"""

import os
import json
import pandas as pd
import anthropic
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-haiku-4-5-20251001"
BRAND = "RougeGorge"


def load_data():
    prompts_df = pd.read_csv("prompts.csv")
    competitors_df = pd.read_csv("competitors.csv")
    competitors = competitors_df["competitor"].tolist()
    print(f"✓ {len(prompts_df)} prompts chargés")
    print(f"✓ {len(competitors)} concurrents à surveiller")
    return prompts_df, competitors


def get_ai_answer(prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def analyze_response(prompt: str, response: str, competitors: list) -> dict:
    competitors_str = ", ".join(competitors)

    system_prompt = f"""Tu es un expert en analyse de visibilité de marque dans les réponses IA.
Tu analyses des réponses pour mesurer la présence de {BRAND}.
Tu réponds TOUJOURS avec un JSON valide, sans texte avant ou après.
Concurrents à surveiller : {competitors_str}"""

    analysis_request = f"""Analyse cette réponse d'un assistant IA.

Question : {prompt}
Réponse : {response}

JSON attendu :
{{
  "rougegorge_mentionnee": true ou false,
  "position_citation": "première" ou "parmi d'autres" ou "en dernier" ou "absente",
  "concurrents_mentionnes": ["marque1", "marque2"],
  "tonalite": "positive" ou "neutre" ou "negative",
  "score_visibilite": 0 à 100,
  "recommandation": "recommandation courte et actionnable pour améliorer la visibilité GEO de RougeGorge"
}}

Score : 0=absente, 25=brièvement citée, 50=clairement citée, 75=bonne option, 100=premier choix"""

    result = client.messages.create(
        model=MODEL,
        max_tokens=500,
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
        print("    ⚠️  Erreur parsing JSON, valeurs par défaut utilisées")
        return {
            "rougegorge_mentionnee": False,
            "position_citation": "absente",
            "concurrents_mentionnes": [],
            "tonalite": "neutre",
            "score_visibilite": 0,
            "recommandation": "Erreur d'analyse"
        }


def run_benchmark():
    print("=" * 60)
    print("🌹 RougeGorge — Benchmark AI Visibility")
    print(f"   Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)
    prompts_df, competitors = load_data()

    raw_results = []
    analyzed_results = []
    total = len(prompts_df)

    for i, row in prompts_df.iterrows():
        prompt = row["prompt"]
        category = row.get("category", "général")

        print(f"\n[{i + 1}/{total}] {category} — {prompt[:70]}...")
        print("  → Interrogation de Claude...")
        answer = get_ai_answer(prompt)

        raw_results.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "category": category,
            "prompt": prompt,
            "response": answer
        })

        print("  → Analyse...")
        analysis = analyze_response(prompt, answer, competitors)

        analyzed_results.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
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
        cited = "✓ Citée" if analysis.get("rougegorge_mentionnee") else "✗ Absente"
        print(f"  ✅ Score : {score}/100 | {cited}")

    pd.DataFrame(raw_results).to_csv("data/raw_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(analyzed_results).to_csv("data/analyzed_results.csv", index=False, encoding="utf-8")

    score_moyen = sum(r["score_visibilite"] for r in analyzed_results) / total
    mentions = sum(1 for r in analyzed_results if r["rougegorge_mentionnee"])

    print("\n" + "=" * 60)
    print("✅ Benchmark terminé !")
    print(f"   Score moyen : {score_moyen:.1f}/100")
    print(f"   RougeGorge citée dans : {mentions}/{total} réponses")
    print(f"\n   Lance le dashboard : streamlit run dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
