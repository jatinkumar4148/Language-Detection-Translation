# LinguaDetect AI

Detect the language of any text and translate it — built on a custom-trained
Naive Bayes classifier, verified against FastText, with Helsinki-NLP/MarianMT
translation.

This is a hardened, production-ready rewrite of an existing prototype
notebook. **The trained model artifacts were not touched** — same
`NB_language_pipeline.pkl`, same `lid.176.ftz`. Everything around them
(confidence logic, error handling, caching, UI, security) was rebuilt.

---

## Features

- **Custom NB language detector** — character-level TF-IDF + Multinomial
  Naive Bayes, trained on 22 languages (97.57% held-out test accuracy).
- **FastText cross-verification** — an independent 176-language detector
  used to catch cases the custom model can't (see "Modification Log" below).
- **Transparent confidence fusion** — combines both models' outputs, their
  agreement, prediction margin, and input length into one score, shown to
  the user with a plain-language explanation, not just a raw number.
- **Script-aware analysis** — detects the dominant Unicode script family
  (Devanagari, Arabic, CJK, Cyrillic, etc.) and flags likely mixed-language
  input.
- **3-level MarianMT translation fallback** — direct model → `tc-big` model
  → pivot through English — with per-pair caching and graceful failure
  messages (never a raw traceback).
- **Working copy button**, character/word counters, and predictable session
  state (example buttons, clear, and detection no longer fight each other).

---

## ML Architecture

```
TextCleaner → Character TF-IDF (2–4 grams, 50k features) → MultinomialNB
```

This exact pipeline is what's pickled in `NB_language_pipeline.pkl`. It was
**not retrained** for this rewrite — see "Do We Need to Retrain?" below for
why, and what the implications are.

A Logistic Regression model was also trained in the original notebook for
comparison (98.25% accuracy vs. NB's 97.57%) but was not the one saved to
disk, so it isn't part of the deployed app.

## Dataset

22,000 rows, exactly 1,000 samples per language, 22 languages:

```
Arabic, Chinese, Dutch, English, Estonian, French, Hindi, Indonesian,
Japanese, Korean, Latin, Persian, Portuguese, Pushto/Pashto, Romanian,
Russian, Spanish, Swedish, Tamil, Thai, Turkish, Urdu
```

This list was pulled directly from `nb_pipeline.classes_` at runtime and
cross-checked against `df['language'].unique()` in the original notebook —
they match. **This is important because the original notebook's
`language_codes` dictionary did not match this list** (see below).

## Detection Pipeline

```
User Input
   ↓
Input Validation  (strip URLs/emails/digits/emoji; empty → "Unable to Detect")
   ↓
Script Analysis   (dominant Unicode script family + mixed-language flag)
   ↓
Naive Bayes       (22-class TF-IDF classifier)
   ↓
FastText          (176-class verification)
   ↓
Confidence Fusion (agreement + margin + length + trained-language check)
   ↓
Detected Language + Confidence Status + Method Label
   ↓
MarianMT Translation (only if a target language was requested)
   ↓
Translation Result
```

## Translation Pipeline

```
source == target?  → return as-is, no model loaded
Level 1: Helsinki-NLP/opus-mt-{src}-{tgt}
Level 2: Helsinki-NLP/opus-mt-tc-big-{src}-{tgt}
Level 3: {src}→en, then en→{tgt}  (pivot)
All levels fail → clear "unavailable" message, app keeps running
```

Models are loaded **lazily** (only when a translation is actually requested)
and cached per `(source_code, target_code)` pair via `st.cache_resource`, so
the same pair is never downloaded or re-instantiated twice in one session.

---

## What Changed, and Why

| # | Problem | Original Behavior | Improvement | Why |
|---|---------|--------------------|-------------|-----|
| 1 | **Source-language code mapping bug** | `language_codes` had 40 entries but the trained model only has 22 classes. It was missing `Estonian`, `Latin`, `Pushto` (real trained classes!) and contained 21 languages (German, Italian, Kannada, Bengali, Greek, Swahili, Telugu, Malayalam, Danish, Serbian, Macedonian, Bulgarian, Ukrainian, Polish, Czech, Croatian, Hungarian, Finnish, Norwegian, Catalan, Hebrew, Vietnamese) the model was **never trained on**. `language_codes.get(detected)` silently returned `None` for the 3 missing real classes. | Rebuilt `NB_LANGUAGE_CODES` to exactly match `nb_pipeline.classes_` (verified programmatically), separate from the broader `TRANSLATE_TARGET_CODES` used only for the target-language dropdown. | A source-detection code map that doesn't match the model's actual classes is a silent correctness bug, not a style issue. |
| 2 | **NB trusted blindly outside its training set** | For German text, NB confidently (85.8%) predicted **Dutch** — wrong, because German isn't one of the 22 trained languages — and the app used that result unquestioned whenever `top_score >= 0.60`, even when FastText disagreed. | If FastText's prediction isn't one of NB's 22 trained classes, NB is structurally incapable of being right — the fusion logic now trusts FastText outright in that case, labeled `FastText Fallback (outside NB training set)`. | Verified empirically (see Testing Report): German → NB says "Dutch" (wrong), FastText says "German" (0.93, correct). No threshold tweak fixes this — it needs a structural check. |
| 3 | **"No signal" looked like "low confidence"** | Digits-only, emoji-only, and untrained-script inputs (e.g. Bengali, which isn't a trained class) all produced a near-uniform NB posterior around 4.5–4.7% per class — statistically indistinguishable from the model's own class prior (1/22 ≈ 4.5%) — but the old code treated this the same as any other low score. | Added `NB_UNINFORMATIVE_THRESHOLD` (1.8× the class prior): below it, NB's opinion is excluded from the fusion score entirely rather than dragging it down as "weak evidence for something." | A flat/uniform distribution isn't weak evidence for its arg-max class — it's no evidence at all. Conflating the two produced misleadingly specific-looking wrong answers (e.g. "Swedish" for Bengali text). |
| 4 | **No handling for non-linguistic input** | A URL, a string of digits, or emoji-only text was fed straight into both models and produced a language guess. | `extract_linguistic_content()` strips URLs/emails/digits/emoji first; if nothing letter-like remains, the app returns "Unable to Detect — no linguistic content" instead of guessing. | Section 28's own test list includes "Numbers only," "Emoji only," and "URL only" — guessing a language for these is worse than saying so plainly. |
| 5 | **Short, unambiguous text scored too low** | "Hello" scored NB≈18%, FastText≈23% individually — both below any single threshold — so it fell through to "text too short," even though a human reads it as unambiguously English. | Agreement between two independently-trained models is now itself a confidence signal, with an extra small boost when the agreement happens on short text (since each model has little to work with individually). | Matches the explicit requirement that "Hello" and "Bonjour" should reasonably resolve, while "hi" / "ok" (which the two models don't reliably agree on) still correctly land in Low/Unable. |
| 6 | **Portugese → Portuguese spelling** | UI showed the training data's misspelling "Portugese" everywhere. | `DISPLAY_NAMES['pt'] = 'Portuguese'` for display; the internal code `'pt'` and the pipeline's own class label `'Portugese'` are untouched so nothing about the model changes. | Cosmetic fix that doesn't touch the trained artifact. |
| 7 | **Repeated MarianMT downloads** | `load_translation_model` in the original app *was* already using `@st.cache_resource`, which was good — but the underlying `nlp_translate` function used in the notebook (and referenced in the fallback chain) created fresh tokenizer/model objects on every call in some code paths. | Every translation call now goes through one cached loader keyed by `(source_code, target_code)`; the same pair is loaded once per process, never per click. | Avoids re-downloading multi-hundred-MB models on every "Detect & Translate" click. |
| 8 | **Unescaped HTML injection risk** | Detected text, translated text, and example previews were interpolated directly into `unsafe_allow_html=True` blocks. | All user-controlled or model-generated strings are passed through `html.escape()` before being placed in an HTML block. | Prevents a crafted input (e.g. `<img src=x onerror=...>`) from executing as markup in the result cards. |
| 9 | **Fake copy button** | The original CSS defined a `.copy-btn` style but no working click handler was wired up anywhere. | A real copy button using `navigator.clipboard.writeText()`, with the text safely JSON-encoded so it can't break out of the embedded script. | Matches the explicit requirement that the copy button "must actually work." |
| 10 | **Session-state / rerun fragility** | Example buttons wrote to `st.session_state['example_text']` and called `st.rerun()`, then the result section separately checked `'example_text' in st.session_state` as its trigger — meaning clicking an example could re-run detection, and results could vanish on unrelated reruns (e.g. changing the target-language dropdown). | The text area is now bound directly to `st.session_state.input_text` via its `key`; example/clear buttons use `on_click` callbacks (the documented-correct Streamlit pattern) instead of manual `st.rerun()`. Detection only runs on an explicit button click and its result is cached in `st.session_state.result` so it survives unrelated widget interactions. | Predictable state, no accidental re-inference, matches "Translation should not run repeatedly unless requested." |
| 11 | **numpy/fastText incompatibility** | Undocumented in the shipped app (though the training notebook itself pinned `numpy<2.0` in a `!pip install` cell — evidently discovered during development but not carried into deployment docs). | `requirements.txt` pins `numpy<2.0` with an explanation. | Verified directly in this environment: `fasttext.predict()` raises `"Unable to avoid copy while creating an array as requested"` under numpy 2.x, and works cleanly under 1.26.4. |
| 12 | **Misleading model statistics** | UI hard-coded "97.5%" and described FastText as if it were part of the same trained model's capability. | Stats bar now explicitly separates "22 ML-Trained Languages" (measured 97.57%) from "176 FastText-Supported Languages," with no single number implying the custom model handles 176 languages. | Matches the explicit instruction not to conflate the two. |

---

## Do We Need to Retrain?

**No, and this rewrite does not retrain anything.** `NB_language_pipeline.pkl`
is loaded and used exactly as it was trained.

The one open question is whether the 3 languages the NB model actually knows
but the old code mapping ignored — **Estonian, Latin, Pushto** — should stay
in the source-detection code map. They now do (see fix #1), because doing so
requires no retraining, just correcting a lookup table. If you'd rather the
app only ever mention the more commonly-expected languages, you can remove
those three from `NB_LANGUAGE_CODES` — but note that the pickled model will
still sometimes predict them internally; removing them from the map only
affects what happens *after* that prediction, not whether it occurs.

If you want the detector to also *natively* recognize languages like German,
Italian, or Bengali (currently handled only via the FastText fallback path,
which is honest but has no character-level features tuned by your own
training data), that would require retraining `nb_pipeline` on a dataset
that includes those languages, which **would** produce a new, incompatible
`.pkl`. That's out of scope for this pass since the artifact was explicitly
not to be broken.

---

## Technologies Used

Streamlit · scikit-learn (TF-IDF + Multinomial Naive Bayes) · FastText
(`lid.176.ftz`) · Hugging Face Transformers (MarianMT) · SentencePiece ·
PyTorch (CPU) · joblib

---

## Project Structure

```
LinguaDetect-AI/
│
├── app.py                     # Full application
├── NB_language_pipeline.pkl   # Pre-trained NB pipeline (unchanged)
├── lid.176.ftz                # FastText language ID model (unchanged)
├── requirements.txt
├── README.md
└── .gitignore
```

No other files are required. MarianMT model weights are downloaded from
Hugging Face on first use of a given language pair and cached by
`st.cache_resource` (and by Transformers' own on-disk cache) for the rest
of the process's lifetime — they are not bundled in the repo.

---

## Installation

```bash
git clone <your-repo-url>
cd LinguaDetect-AI
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `pip install fasttext` fails to build on your platform (it needs a C++
compiler), install the prebuilt-wheel alternative instead:

```bash
pip install fasttext-wheel==0.9.2
```

Both expose the same `fasttext` Python module, so no code changes are
needed either way.

For a smaller install (skip the CUDA build of PyTorch, which you don't need
for this app):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Running Locally

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

The NB and FastText models load once at startup (a few seconds). MarianMT
models are **not** downloaded at startup — the first translation to a new
language pair will take longer while it downloads, and every one after that
(for the same pair, in the same running process) will be instant.

## Deployment

**GitHub:**
```bash
git init
git add app.py NB_language_pipeline.pkl lid.176.ftz requirements.txt README.md .gitignore
git commit -m "LinguaDetect AI — production rewrite"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

> `NB_language_pipeline.pkl` is ~19 MB and `lid.176.ftz` is ~1 MB — both are
> fine for a normal GitHub repo. If your host has a stricter file-size limit,
> use Git LFS for the `.pkl`.

**Streamlit Community Cloud:**
1. Push the repo to GitHub as above.
2. Go to [share.streamlit.io](https://share.streamlit.io), "New app," point
   it at your repo and `app.py`.
3. Streamlit Cloud installs from `requirements.txt` automatically — no
   extra configuration needed.
4. The first translation request in a fresh deployment will be slow (model
   download); subsequent ones for the same language pair are fast for the
   life of that container.

---

## Limitations

Stated plainly, per the project's own "don't fake it" rule:

- **The custom NB model only knows 22 languages.** Anything else is
  detected via FastText alone, which is a good general-purpose detector but
  was not trained on your specific dataset or domain.
- **Romanized / transliterated mixed text is genuinely hard.** E.g. "Hello
  bhai, aaj kaise ho?" (English + romanized Hindi, no Devanagari script) was
  tested and lands at Medium confidence with an incorrect FastText guess
  (Finnish) underneath it — script analysis can't help here because the
  input is pure Latin script, and neither model was built for
  code-switched romanized text. This is disclosed to the user via the
  confidence status rather than hidden behind a falsely-confident answer.
- **Mixed-language detection is single-label with a warning, not
  multi-label.** The app reports one dominant language plus a "this looks
  mixed" flag when a second Unicode script makes up ≥15% of the letters —
  it does not attempt to segment and label each language separately.
- **Confidence scores are a heuristic fusion, not calibrated
  probabilities.** They're documented as such in the UI. The NB
  probability bars are literally the model's own softmax-like outputs
  (which are also not perfectly calibrated for Naive Bayes) — labeled
  "Relative Model Scores," not "accuracy."
- **Translation quality is only as good as the underlying Helsinki-NLP
  models**, which vary a lot by language pair; some pairs may not exist at
  all, in which case the app says so rather than guessing.
- **Translation was not network-tested in the environment this rewrite was
  built in** (no access to huggingface.co from that sandbox). The
  request/fallback/caching logic mirrors the original notebook's
  already-functional `nlp_translate()` calls exactly, with added exception
  handling around each level — but you should smoke-test a few language
  pairs after your first real deployment.

## Future Improvements

- Retrain the NB model on an expanded dataset (see "Do We Need to
  Retrain?") if native detection of more languages is a priority.
- Add true multi-label / segment-level mixed-language detection (e.g.
  sliding-window detection per sentence or clause).
- Calibrate NB's probability outputs (e.g. via `CalibratedClassifierCV`) if
  the raw probability bars need to be closer to true confidence.
- Add a small on-disk translation cache (text hash → translation) to avoid
  re-translating identical inputs across sessions.

---

## Testing Report

All detection test cases below were run through the **actual deployed
`app.py`** end-to-end (via Streamlit's `AppTest` harness — real button
clicks, real session state, real model calls) in this environment, using
the exact `NB_language_pipeline.pkl` and `lid.176.ftz` you provided.
Translation calls could not be executed in this sandbox (no
`huggingface.co` egress available) — see Limitations.

| Input | Detected | Confidence | Status | Method |
|---|---|---|---|---|
| "Hello, how are you today?" | English | 96.2% | High | NB + FastText Verified |
| "नमस्ते, आप कैसे हैं?" | Hindi | 100.0% | High | NB + FastText Verified |
| "Bonjour, comment allez-vous?" | French | 100.0% | High | NB + FastText Verified |
| "Hola, ¿cómo estás?" | Spanish | 90.6% | High | NB + FastText Verified |
| "Guten Morgen, wie geht es Ihnen?" | German | 86.4% | High | FastText Fallback *(outside NB's 22 classes — NB alone would have wrongly said "Dutch")* |
| "こんにちは、お元気ですか？" | Japanese | 91.3% | High | NB + FastText Verified |
| "안녕하세요, 어떻게 지내세요?" | Korean | 100.0% | High | NB + FastText Verified |
| "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?" | Tamil | 100.0% | High | NB + FastText Verified |
| "নমস্কার, আপনি কেমন আছেন?" | Bengali | 89.9% | High | FastText Fallback *(outside NB's 22 classes — NB alone gave a near-random "Swedish")* |
| "مرحبا، كيف حالك؟" | Arabic | 100.0% | High | NB + FastText Verified |
| "Hello" | English | 45.3% | Medium | NB + FastText Verified (agreement on short text) |
| "hi" | English | 28.0% | Low | NB + FastText Verified (weak, appropriately uncertain) |
| "ok" | — | 3.4% | Unable to Detect | Ambiguous, as expected |
| "12345" (numbers only) | — | 0.0% | Unable to Detect | No linguistic content |
| "😀😃😄" (emoji only) | — | 0.0% | Unable to Detect | No linguistic content |
| "https://example.com/test" (URL only) | — | 0.0% | Unable to Detect | No linguistic content |
| "" (empty) | — | — | — | Rejected before detection runs |
| "Hello bhai, aaj kaise ho?" (mixed, romanized) | Finnish | 55.5% | Medium | FastText Fallback *(genuinely hard case — flagged as a known limitation above, not hidden)* |
| "नमस्ते everyone, how are you?" (mixed script) | English | 95.6% | High | NB + FastText Verified, **+ mixed-language warning shown** (Latin 74% / Devanagari 26%) |
| Long paragraph (1 sentence, 85 chars) | English | 100.0% | High | NB + FastText Verified |

No exceptions were raised across any of the above, including the
deliberately adversarial inputs (empty string, symbols-only, emoji-only).
