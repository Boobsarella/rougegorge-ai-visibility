"""
run_benchmark.py — Script principal du benchmark RougeGorge AI Visibility

Ce script :
1. Lit les prompts depuis prompts.csv
2. Envoie chaque prompt à Claude (API Anthropic)
3. Analyse chaque réponse (mentions, tonalité, score, recommandation)
4. Sauvegarde tout dans data/raw_results.csv et data/analyzed_results.csv

Usage : python run_benchmark.py
"""

import os
import json
import pandas as pd
import anthropic
from dotenv import load_dotenv
from datetime import datetime

# Charge les variables d'environnement depuis le fichier .env
load_dotenv()

# Initialise le client Anthropic avec la clé API
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Modèle utilisé (haiku = rapide et économique)
MODEL = "claude-haiku-4-5-20251001"

# Nom de la marque à surveiller
BRAND = "RougeGorge"


def load_data():
    """Charge les prompts et la liste des concurrents depuis les CSV."""
    prompts_df = pd.read_csv("prompts.csv")
    competitors_df = pd.read_csv("competitors.csv")
    competitors = competitors_df["competitor"].tolist()
    print(f"✓ {len(prompts_df)} prompts chargés")
    print(f"✓ {len(competitors)} concurrents à surveiller")
    return prompts_df, competitors


def get_ai_answer(prompt: str) -> str:
    """
    Envoie un prompt à Claude et retourne sa réponse.
    C'est la réponse "naturelle" que Claude donnerait à un vrai utilisateur.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text


def analyze_response(prompt: str, response: str, competitors: list) -> dict:
    """
    Demande à Claude d'analyser une réponse pour mesurer la visibilité de RougeGorge.
    Utilise le prompt caching pour économiser des tokens (le contexte système est réutilisé).
    Retourne un dictionnaire avec le score, la tonalité, etc.
    """
    competitors_str = ", ".join(competitors)

    # Ce message système est mis en cache pour éviter de le renvoyer à chaque fois
    system_prompt = f"""Tu es un expert en analyse de visibilité de marque dans les réponses IA.
Tu analyses des réponses d'assistants IA pour mesurer la présence de la marque {BRAND}.
Tu réponds TOUJOURS avec un JSON valide, sans texte avant ou après.
Concurrents à surveiller : {competitors_str}"""

    analysis_request = f"""Analyse cette réponse d'un assistant IA.

Question posée : {prompt}

Réponse analysée : {response}

Réponds UNIQUEMENT avec ce JSON (remplace les valeurs) :
{{
  "rougegorge_mentionnee": true ou false,
  "concurrents_mentionnes": ["marque1", "marque2"],
  "tonalite": "positive" ou "neutre" ou "negative",
  "score_visibilite": 0 à 100,
  "recommandation": "Une recommandation courte pour améliorer la visibilité de RougeGorge"
}}

Règles pour score_visibilite :
- 0  = RougeGorge absente
- 25 = mentionnée brièvement parmi d'autres
- 50 = mentionnée clairement
- 75 = présentée comme bonne option
- 100 = recommandée en premier choix

Pour la tonalite (uniquement si RougeGorge est mentionnée) :
- positive = associée à des termes positifs
- neutre = mentionnée sans jugement
- negative = associée à des termes négatifs"""

    response_analysis = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {"role": "user", "content": analysis_request}
        ]
    )

    raw_text = response_analysis.content[0].text.strip()

    # Nettoie le texte si Claude a ajouté des balises markdown (```json ... ```)
    if "```" in raw_text:
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        print(f"    ⚠️  Erreur parsing JSON, valeurs par défaut utilisées")
        return {
            "rougegorge_mentionnee": False,
            "concurrents_mentionnes": [],
            "tonalite": "neutre",
            "score_visibilite": 0,
            "recommandation": "Erreur d'analyse"
        }


def run_benchmark():
    """Fonction principale : exécute tout le benchmark."""
    print("=" * 60)
    print("🌹 RougeGorge — Benchmark AI Visibility")
    print(f"   Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 60)

    # Crée le dossier data/ s'il n'existe pas
    os.makedirs("data", exist_ok=True)

    # Charge les données
    prompts_df, competitors = load_data()

    raw_results = []
    analyzed_results = []
    total = len(prompts_df)

    for i, row in prompts_df.iterrows():
        prompt = row["prompt"]
        category = row.get("category", "général")

        print(f"\n[{i + 1}/{total}] Catégorie : {category}")
        print(f"  Prompt : {prompt[:70]}...")

        # Étape 1 : obtenir la réponse de Claude
        print("  → Interrogation de Claude...")
        answer = get_ai_answer(prompt)

        # Sauvegarde le résultat brut
        raw_results.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "category": category,
            "prompt": prompt,
            "response": answer
        })

        # Étape 2 : analyser la réponse
        print("  → Analyse de la réponse...")
        analysis = analyze_response(prompt, answer, competitors)

        # Sauvegarde le résultat analysé
        analyzed_results.append({
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

        score = analysis.get("score_visibilite", 0)
        mentioned = "✓ Citée" if analysis.get("rougegorge_mentionnee") else "✗ Absente"
        print(f"  ✅ Score : {score}/100 | RougeGorge : {mentioned}")

    # Sauvegarde les deux fichiers CSV
    pd.DataFrame(raw_results).to_csv("data/raw_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(analyzed_results).to_csv("data/analyzed_results.csv", index=False, encoding="utf-8")

    # Affiche le résumé final
    score_moyen = sum(r["score_visibilite"] for r in analyzed_results) / total
    mentions = sum(1 for r in analyzed_results if r["rougegorge_mentionnee"])

    print("\n" + "=" * 60)
    print("✅ Benchmark terminé !")
    print(f"   Score moyen de visibilité : {score_moyen:.1f}/100")
    print(f"   RougeGorge citée dans : {mentions}/{total} réponses")
    print(f"   Résultats bruts    → data/raw_results.csv")
    print(f"   Résultats analysés → data/analyzed_results.csv")
    print(f"\n   Lance le dashboard : streamlit run dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
