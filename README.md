# Language-Detection-Translation
<div align="center">

# 🌐 LinguaDetect AI

### Detect the language of any text and translate it — instantly.

Custom-trained Naive Bayes classifier · FastText cross-verification · Helsinki-NLP MarianMT translation

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.61-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit--learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.13%2Bcpu-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](#)

</div>

---

## ✨ What it does

```
"Bonjour, comment allez-vous?"  →  🇫🇷 French (100%)  →  translate to  →  🇬🇧 English
```

LinguaDetect AI takes any text, figures out what language it's in using **two independent
models that cross-check each other**, shows you *how confident* it actually is (not just a
raw number), and then translates it into any target language you pick.

| | |
|---|---|
| 🧠 **Custom NB detector** | Character-level TF-IDF + Multinomial Naive Bayes, trained on 22 languages — **97.57%** held-out accuracy |
| 🌍 **FastText backup** | 176-language identification for anything outside the 22 trained classes |
| 🔀 **Confidence fusion** | Combines both models' agreement, margin, and input length into one honest score |
| 🔤 **Script-aware** | Detects the dominant Unicode script (Devanagari, Arabic, CJK, Cyrillic…) and flags mixed-language input |
| 🌐 **3-level translation fallback** | Direct MarianMT model → `tc-big` model → pivot through English |
| 📋 **Working copy button** | Real clipboard copy, character/word counters, stable session state |

---

## 🏗️ Architecture

<div align="center">
<img width="1024" height="1536" alt="ChatGPT Image Aug 11, 2026, 04_07_13 PM" src="https://github.com/user-attachments/assets/7f77a680-dbe1-4f81-ae1a-e50f22e62db0" />


</div>

> 🎨 **Tip:** ask Claude to show you the colorful interactive versions of the **detection**
> and **translation** flowcharts inline — they break this down step-by-step with clickable
> nodes.

---

## 🧰 Tech stack

| Layer | Technology | Why |
|---|---|---|
| Language | 🐍 **Python 3.10** | Runtime — see [Environment](#-environment) for why 3.10 specifically |
| UI | 🎈 **Streamlit** | Fast Python-native web app, easy to deploy |
| Data | 🐼 **Pandas** | Dataset loading and exploration |
| ML pipeline | 🔬 **Scikit-learn** | `TextCleaner` → `TF-IDF` → `MultinomialNB`, wrapped in one `Pipeline` |
| Vectorization | 🔡 **TF-IDF (char n-grams 2–4)** | Captures language-specific spelling/character patterns |
| Classifier | 📊 **MultinomialNB** | Fast, accurate for 22-language text classification |
| Fallback detector | 🌍 **FastText** (`lid.176.ftz`) | 170+ languages, catches what NB can't |
| Translation | 🗣️ **Helsinki-NLP MarianMT** | Pretrained transformer models, many language pairs |
| Model I/O | 💾 **Joblib** | Train once, save once, load & reuse forever |
| Persistence | 🔥 **PyTorch (CPU)** + 🤗 **Transformers** + **SentencePiece** | Runs MarianMT inference |

---

## 📦 Environment

> **Python 3.10 is required — and it's easy to add alongside a newer Python you already have.**
> Old Python versions are *not* removed from python.org after a newer release; 3.10 installers
> stay downloadable and 3.10 is still officially supported (EOL Oct 2026). You do **not** need
> to uninstall a newer Python to use this project.

This is the **verified working stack** on Windows:

| Package | Version | Note |
|---|---|---|
| Python | `3.10.0` | via `py -3.10` |
| NumPy | `1.26.4` | **must stay < 2.0** — fastText's C++ extension breaks under numpy 2.x |
| Scikit-learn | `1.7.2` | pipeline was pickled under 1.6.1; 1.3–1.8 load it cleanly |
| Joblib | `1.5.3` | |
| FastText | `fasttext-community 0.11.7` | see [why not `fasttext==0.9.3`](#-why-fasttext-community-on-windows) |
| Streamlit | `1.61.1` | |
| PyTorch | `2.13.0+cpu` | CPU build — no CUDA needed |
| Transformers | `4.57.6` | |
| SentencePiece | `0.2.2` | |

<details>
<summary><b>🩹 Why <code>fasttext-community</code> instead of <code>fasttext==0.9.3</code>?</b></summary>
<br>

The original `requirements.txt` pins `fasttext==0.9.3`, which needs a C++ compiler to
build from source. On Windows this build fails by default (no compiler toolchain).
`fasttext-community` ships a prebuilt wheel with the **same `import fasttext` API**, so
no application code changes are needed — just swap the install command.

</details>

---

## 🚀 Quick start (Windows)

```powershell
cd C:\Users\User\Desktop\langDEUI

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install numpy==1.26.4
python -m pip install "scikit-learn>=1.3,<1.9"
python -m pip install "joblib>=1.3,<2.0"
python -m pip install "streamlit>=1.38,<2.0"
python -m pip install fasttext-community
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "transformers>=4.30,<5.0"
python -m pip install "sentencepiece>=0.1.99"

streamlit run app.py
```

Then open **http://localhost:8501** 🎉

<details>
<summary>📋 Step-by-step (with explanations)</summary>

1. **Open the project folder** and confirm `app.py`, `requirements.txt`,
   `NB_language_pipeline.pkl`, and `lid.176.ftz` are all present.
2. **Create the venv**: `py -3.10 -m venv .venv` — an isolated Python 3.10 environment,
   scoped only to this project.
3. **Activate it**: `.\.venv\Scripts\Activate.ps1` — your prompt should now start with
   `(.venv)`.
4. **Verify the interpreter**: `python --version` and `where.exe python` — the first
   result should point inside `...\langDEUI\.venv\Scripts\python.exe`.
5. **Install dependencies** one by one (see command block above) — installing them
   individually makes it obvious exactly which one fails, if any does.
6. **Verify the environment** (optional but recommended — see below).
7. **Run**: `streamlit run app.py`.

</details>

---

## ✅ Verify the environment

```powershell
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
python -c "import joblib; print('Joblib:', joblib.__version__)"
python -c "import fasttext; print('FastText import: OK')"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import sentencepiece; print('SentencePiece: OK')"
```

`CUDA available: False` is **expected and fine** — this project runs entirely on CPU.

Sanity-check the FastText model itself:

```powershell
python -c "import fasttext; m=fasttext.load_model('lid.176.ftz'); print(m.predict('Hello, how are you?', k=3))"
```

---

## 🔍 Detection pipeline

```
Input → Validation → Script Analysis → NB (22 classes) → FastText (176 classes)
      → Confidence Fusion → Detected Language + Confidence
```

| Confidence | Behavior |
|---|---|
| 🟢 High / ≥60% | NB result used directly |
| 🟡 Medium/Low | Cross-checked against FastText; agreement boosts confidence |
| 🔴 Below threshold | Falls back to FastText outright, labeled *"FastText Fallback"* |

## 🌐 Translation pipeline

```
Detected Language + Target Language
   ↓
Level 1 → Direct opus-mt-{src}-{tgt} model
   ↓ (if unavailable)
Level 2 → tc-big model
   ↓ (if unavailable)
Level 3 → Pivot: src → English → target
   ↓
Translated Text
```

Models load **lazily** (only when a translation is actually requested) and are cached per
`(source, target)` pair, so the same pair is never re-downloaded or re-instantiated twice
in one session.

---

## 📂 Project structure

```
LinguaDetect-AI/
│
├── app.py                     ← full Streamlit application
├── NB_language_pipeline.pkl   ← pre-trained NB pipeline (do not delete/replace)
├── lid.176.ftz                ← FastText language ID model
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/                     ← local virtual environment (not committed)
```

---

## 🧪 Suggested test cases

| Input | Expected |
|---|---|
| `Hello, how are you today?` | 🇬🇧 English |
| `नमस्ते, आप कैसे हैं?` | 🇮🇳 Hindi |
| `Bonjour, comment allez-vous?` | 🇫🇷 French |
| `Guten Morgen, wie geht es Ihnen?` | 🇩🇪 German — good fallback test, since German isn't one of NB's 22 trained classes and only FastText catches it correctly |

---

## 🛠️ Troubleshooting

<details>
<summary><code>ModuleNotFoundError: No module named 'fasttext'</code></summary>

```powershell
python -m pip install fasttext-community
```
</details>

<details>
<summary>NumPy is 2.x and fastText breaks</summary>

```powershell
python -m pip install numpy==1.26.4
```
</details>

<details>
<summary><code>ModuleNotFoundError: No module named 'torch'</code></summary>

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```
</details>

<details>
<summary><code>ModuleNotFoundError: No module named 'transformers'</code></summary>

```powershell
python -m pip install "transformers>=4.30,<5.0"
```
</details>

<details>
<summary><code>ModuleNotFoundError: No module named 'sentencepiece'</code></summary>

```powershell
python -m pip install "sentencepiece>=0.1.99"
```
</details>

<details>
<summary><code>AttributeError: Can't get attribute 'TextCleaner'</code></summary>

The pickled pipeline expects the custom `TextCleaner` class to exist in the loading
context. This is **not** a sign of a corrupted model — always load it through `app.py`,
where the class is defined, not a bare `joblib.load()` script.
</details>

<details>
<summary><code>FileNotFoundError</code> for the model files</summary>

Keep `NB_language_pipeline.pkl` and `lid.176.ftz` in the same directory as `app.py`, and
run Streamlit from that directory.
</details>

---

## 🔁 Running it again later

```powershell
cd C:\Users\User\Desktop\langDEUI
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Stop with `Ctrl + C`. Leave the venv with `deactivate`.

---

## ⚠️ Limitations

- The custom NB model only natively knows **22 languages** — everything else routes
  through FastText, which is general-purpose but not tuned on this project's data.
- Romanized/code-switched text (e.g. English + romanized Hindi, no Devanagari script) is
  genuinely hard for both models and is disclosed via confidence status rather than
  hidden behind a falsely-confident answer.
- Confidence scores are a **heuristic fusion**, not calibrated probabilities — labeled
  as such in the UI.
- Translation quality depends entirely on the underlying Helsinki-NLP model for that
  language pair; some pairs may not exist, in which case the app says so rather than
  guessing.
- The **first** translation for any new language pair downloads model weights from
  Hugging Face — needs an internet connection once per pair, per environment.

---

<div align="center">

**LinguaDetect AI** — multilingual language detection + translation, built with
Streamlit · Scikit-learn · FastText · MarianMT · Transformers · PyTorch

</div>
