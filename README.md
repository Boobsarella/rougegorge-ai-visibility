# 🌹 RougeGorge — AI Visibility Monitor

Mesure si RougeGorge est citée dans les réponses des assistants IA (comme Claude).

---

## C'est quoi ce projet ?

Ce projet envoie des questions à Claude (via l'API Anthropic) et analyse les réponses pour savoir :
- Est-ce que RougeGorge est mentionnée ?
- Quels concurrents sont cités à la place ?
- La tonalité est-elle positive, neutre ou négative ?
- Quel est le score de visibilité sur 100 ?
- Quelle recommandation SEO/GEO appliquer ?

---

## Structure du projet

```
.
├── prompts.csv              ← Les questions envoyées à l'IA
├── competitors.csv          ← Les concurrents à surveiller
├── run_benchmark.py         ← Script principal (lance le benchmark)
├── dashboard.py             ← Dashboard visuel avec Streamlit
├── requirements.txt         ← Dépendances Python
├── .env.example             ← Modèle pour les variables d'environnement
├── .env                     ← Ton fichier clé API (à créer, ne pas partager !)
└── data/
    ├── raw_results.csv      ← Réponses brutes de Claude (créé automatiquement)
    └── analyzed_results.csv ← Résultats analysés (créé automatiquement)
```

---

## Installation (à faire une seule fois)

### Étape 1 — Prérequis
Assure-toi d'avoir Python 3.10+ installé.
Vérifie avec : `python3 --version`

### Étape 2 — Crée un environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate       # Mac / Linux
# ou sur Windows : venv\Scripts\activate
```

### Étape 3 — Installe les dépendances
```bash
pip install -r requirements.txt
```

### Étape 4 — Configure ta clé API Anthropic

1. Va sur [console.anthropic.com](https://console.anthropic.com) et crée un compte
2. Crée une clé API dans la section "API Keys"
3. Copie le fichier `.env.example` en `.env` :
   ```bash
   cp .env.example .env
   ```
4. Ouvre `.env` et remplace `sk-ant-REMPLACE_PAR_TA_VRAIE_CLE` par ta vraie clé

⚠️ Ne partage jamais ton fichier `.env` et ne le mets pas sur GitHub !

---

## Utilisation

### Lance le benchmark
```bash
python run_benchmark.py
```

Le script va :
1. Envoyer chaque prompt de `prompts.csv` à Claude
2. Analyser chaque réponse
3. Afficher la progression dans le terminal
4. Sauvegarder les résultats dans `data/`

Durée estimée : ~2 à 4 minutes pour 14 prompts.

### Lance le dashboard
```bash
streamlit run dashboard.py
```

Ouvre ton navigateur sur `http://localhost:8501` pour voir le dashboard.

---

## Personnalisation

### Ajouter des prompts
Ouvre `prompts.csv` et ajoute des lignes en suivant le même format :
```
category,prompt
nouvelle_categorie,"Ta nouvelle question ici ?"
```

### Ajouter des concurrents
Ouvre `competitors.csv` et ajoute des lignes :
```
competitor,country
NouvelleMarque,France
```

---

## Coût estimé

Ce projet utilise le modèle Claude Haiku (le plus économique).
Pour 14 prompts : environ **0,05 à 0,10 €** par exécution.

---

## En cas de problème

**Erreur `ANTHROPIC_API_KEY`** → Vérifie que ton fichier `.env` existe et contient la bonne clé.

**Erreur `ModuleNotFoundError`** → Vérifie que ton environnement virtuel est activé (`source venv/bin/activate`) et que tu as lancé `pip install -r requirements.txt`.

**Le dashboard est vide** → Lance d'abord `python run_benchmark.py` pour créer les données.
