"""
NLP Playground – Week 1 methods (NLP, MS AI, UCP)
Type a string, click a method, see the input and the result.
Methods follow the Week 1 slides: tokenization, stop words,
stemming, lemmatization and POS tagging (NLTK and spaCy).
"""

import html

import nltk
import streamlit as st

st.set_page_config(page_title="NLP Playground", page_icon="🧠", layout="centered")


# ---------------------------------------------------------------
# 1. Load the NLP resources once (cached so it is fast)
# ---------------------------------------------------------------
@st.cache_resource
def load_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        nltk.download(pkg, quiet=True)
    return True


@st.cache_resource
def load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None  # app still works with NLTK methods only


load_nltk()
nlp = load_spacy()

from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# Penn Treebank tag meanings (from the POS slides)
POS_MEANING = {
    "CC": "Coordinating conjunction", "CD": "Cardinal number", "DT": "Determiner",
    "EX": "Existential there", "FW": "Foreign word", "IN": "Preposition / subordinating conjunction",
    "JJ": "Adjective", "JJR": "Adjective, comparative", "JJS": "Adjective, superlative",
    "LS": "List item marker", "MD": "Modal", "NN": "Noun, singular", "NNS": "Noun, plural",
    "NNP": "Proper noun, singular", "NNPS": "Proper noun, plural", "PDT": "Predeterminer",
    "POS": "Possessive ending", "PRP": "Personal pronoun", "PRP$": "Possessive pronoun",
    "RB": "Adverb", "RBR": "Adverb, comparative", "RBS": "Adverb, superlative",
    "RP": "Particle", "SYM": "Symbol", "TO": "to", "UH": "Interjection",
    "VB": "Verb, base form", "VBD": "Verb, past tense", "VBG": "Verb, gerund / present participle",
    "VBN": "Verb, past participle", "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present", "WDT": "Wh-determiner",
    "WP": "Wh-pronoun", "WP$": "Possessive wh-pronoun", "WRB": "Wh-adverb",
}


# ---------------------------------------------------------------
# 2. The NLP methods. Each returns (result, display_type)
#    display_type: "list" -> shown as a list, "table" -> shown as a table
# ---------------------------------------------------------------
def nltk_word_tokenize(text):
    return word_tokenize(text), "list"


def nltk_sent_tokenize(text):
    return sent_tokenize(text), "list"


def spacy_word_tokenize(text):
    return [t.text for t in nlp(text)], "list"


def spacy_sent_tokenize(text):
    return [s.text for s in nlp(text).sents], "list"


def nltk_remove_stopwords(text):
    stop = set(stopwords.words("english"))
    return [w for w in word_tokenize(text) if w.lower() not in stop], "list"


def nltk_show_stopwords(text):
    stop = set(stopwords.words("english"))
    return [w for w in word_tokenize(text) if w.lower() in stop], "list"


def spacy_remove_stopwords(text):
    stop = nlp.Defaults.stop_words
    return [t.text for t in nlp(text) if t.text.lower() not in stop], "list"


def porter_stem(text):
    ps = PorterStemmer()
    return [{"Word": w, "Stem": ps.stem(w)} for w in word_tokenize(text)], "table"


def snowball_stem(text):
    ss = SnowballStemmer(language="english")
    return [{"Word": w, "Stem": ss.stem(w)} for w in word_tokenize(text)], "table"


def nltk_lemmatize(text):
    lem = WordNetLemmatizer()
    return [{"Word": w, "Lemma": lem.lemmatize(w)} for w in word_tokenize(text)], "table"


def spacy_lemmatize(text):
    return [{"Word": t.text, "Lemma": t.lemma_} for t in nlp(text)], "table"


def nltk_pos_tag(text):
    return [{"Word": w, "Tag": tag, "Meaning": POS_MEANING.get(tag, "-")}
            for w, tag in pos_tag(word_tokenize(text))], "table"


def spacy_pos_tag(text):
    return [{"Word": t.text, "Tag": t.tag_, "POS": t.pos_, "Meaning": POS_MEANING.get(t.tag_, "-")}
            for t in nlp(text)], "table"


def count_tokens(text):
    words = word_tokenize(text)
    sents = sent_tokenize(text)
    return [{"Measure": "Characters", "Count": len(text)},
            {"Measure": "Word tokens", "Count": len(words)},
            {"Measure": "Sentences", "Count": len(sents)}], "metrics"


# Grouped like the slides. Methods marked spaCy are hidden if spaCy is not installed.
METHODS = {
    "Tokenization": [
        ("Word Tokenize (NLTK)", nltk_word_tokenize, False),
        ("Sentence Tokenize (NLTK)", nltk_sent_tokenize, False),
        ("Word Tokenize (spaCy)", spacy_word_tokenize, True),
        ("Sentence Tokenize (spaCy)", spacy_sent_tokenize, True),
        ("Count Tokens", count_tokens, False),
    ],
    "Stop Words": [
        ("Remove Stop Words (NLTK)", nltk_remove_stopwords, False),
        ("Show Stop Words (NLTK)", nltk_show_stopwords, False),
        ("Remove Stop Words (spaCy)", spacy_remove_stopwords, True),
    ],
    "Stemming": [
        ("Porter Stemmer", porter_stem, False),
        ("Snowball Stemmer", snowball_stem, False),
    ],
    "Lemmatization": [
        ("Lemmatize (NLTK WordNet)", nltk_lemmatize, False),
        ("Lemmatize (spaCy)", spacy_lemmatize, True),
    ],
    "Parts of Speech": [
        ("POS Tagging (NLTK)", nltk_pos_tag, False),
        ("POS Tagging (spaCy)", spacy_pos_tag, True),
    ],
}

# Short explanations shown above each group's buttons (from the Week 1 slides)
GROUP_INFO = {
    "Tokenization": "Splits text into words, symbols and punctuation (tokens) or into sentences. It is the first step of any NLP task.",
    "Stop Words": "Very common words (a, the, in, of, to) that carry little meaning. They are often removed before search or classification.",
    "Stemming": "Cuts words down to a root form by chopping endings. Fast, but the stem may not be a real word (computing → comput).",
    "Lemmatization": "Turns each word into its real dictionary form, the lemma (mice → mouse, was → be).",
    "Parts of Speech": "Labels each word with its grammatical role, such as noun (NN), verb (VB) or adjective (JJ).",
}
GROUP_ICONS = {"Tokenization": "✂️", "Stop Words": "🚫", "Stemming": "🌱",
               "Lemmatization": "📖", "Parts of Speech": "🏷️"}

EXAMPLES = {
    "Slide example": "At eight o'clock on Thursday morning Raza didn't feel very good. "
                     "Hello Mr. Raza, how are you doing today? The weather is nice. The sky is pinkish-blue.",
    "Stemming words": "compute computer computed computing caresses running runs",
    "Lemma words": "cats cacti geese rock better best running runs mice was",
}

# ---------------------------------------------------------------
# 3. Styling (works in light and dark mode, adapts to phone screens)
# ---------------------------------------------------------------
st.markdown("""
<style>
.block-container {max-width: 920px; padding-top: 3.5rem; padding-bottom: 3rem;}
.hero {
  padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%); color: #fff;
}
.hero h1 {color: #fff; font-size: 2.1rem; margin: 0 0 .3rem 0; padding: 0;}
.hero p {margin: 0; opacity: .92; font-size: 1rem;}
.hero .tags {margin-top: .8rem; display: flex; flex-wrap: wrap; gap: .4rem;}
.hero .tags span {background: rgba(255,255,255,.18); border-radius: 999px; padding: .15rem .7rem; font-size: .8rem;}
.section-label {font-weight: 700; font-size: 1.05rem; margin: .4rem 0 .2rem 0;}
.group-info {font-size: .88rem; opacity: .75; margin: .2rem 0 .6rem 0;}
.chips {display: flex; flex-wrap: wrap; gap: .4rem; margin: .3rem 0 .4rem 0;}
.chip {
  background: rgba(124, 58, 237, .12); border: 1px solid rgba(124, 58, 237, .35);
  border-radius: 8px; padding: .2rem .55rem; font-family: ui-monospace, monospace; font-size: .85rem;
  word-break: break-word;
}
.chip.sent {font-family: inherit; border-radius: 10px; display: block; width: 100%;}
.chip small {opacity: .6; margin-right: .35rem;}
.badge {
  display: inline-block; background: #7c3aed; color: #fff; border-radius: 999px;
  padding: .15rem .75rem; font-size: .8rem; font-weight: 600;
}
.meta {font-size: .8rem; opacity: .6; margin-left: .4rem;}
.input-box {
  border-left: 3px solid #7c3aed; padding: .45rem .8rem; margin: .5rem 0 .7rem 0;
  background: rgba(127,127,127,.08); border-radius: 0 8px 8px 0; font-size: .92rem; word-break: break-word;
}
.label {font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; opacity: .6; font-weight: 600;}
div.stButton > button {border-radius: 10px; min-height: 2.6rem; white-space: normal; line-height: 1.2;}
div[data-baseweb="tab-list"] {flex-wrap: wrap; gap: .25rem;}
textarea {font-size: 1rem !important;}
@media (max-width: 640px) {
  .block-container {padding: 3rem .8rem 2rem .8rem;}
  .hero {padding: 1.1rem 1.1rem; border-radius: 14px;}
  .hero h1 {font-size: 1.5rem;}
  .hero p {font-size: .9rem;}
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 4. Session state and button actions
# ---------------------------------------------------------------
if "text" not in st.session_state:
    st.session_state.text = ""
if "history" not in st.session_state:
    st.session_state.history = []


def run_method(name, func, group):
    text = st.session_state.text.strip()
    if not text:
        st.session_state.warning = True
        return
    result, kind = func(text)
    st.session_state.history.insert(0, {"method": name, "group": group, "input": text,
                                        "result": result, "kind": kind})


def set_example(sample):
    st.session_state.text = sample


def clear_input():
    st.session_state.text = ""


def clear_all():
    st.session_state.text = ""
    st.session_state.history = []


def remove_result(index):
    st.session_state.history.pop(index)


# ---------------------------------------------------------------
# 5. Page layout
# ---------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🧠 NLP Playground</h1>
  <p>Type any text, pick an NLP method and see the result instantly.</p>
  <div class="tags"><span>Tokenization</span><span>Stop words</span><span>Stemming</span>
  <span>Lemmatization</span><span>POS tagging</span></div>
</div>
""", unsafe_allow_html=True)

# --- Step 1: input ---
with st.container(border=True):
    st.markdown('<div class="section-label">1 · Enter your text</div>', unsafe_allow_html=True)
    st.text_area("Your text", key="text", height=130, label_visibility="collapsed",
                 placeholder="Type or paste a sentence or paragraph here…")
    st.caption("Or try an example:")
    ex_cols = st.columns(len(EXAMPLES))
    for col, (label, sample) in zip(ex_cols, EXAMPLES.items()):
        col.button(label, key=f"ex_{label}", on_click=set_example, args=(sample,), width="stretch")
    words = len(st.session_state.text.split())
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="center")
    c1.caption(f"{len(st.session_state.text)} characters · {words} words")
    c2.button("Clear text", icon=":material/backspace:", on_click=clear_input, width="stretch")
    c3.button("Clear all", icon=":material/delete:", on_click=clear_all, width="stretch", type="primary")

# --- Step 2: methods ---
with st.container(border=True):
    st.markdown('<div class="section-label">2 · Choose a method</div>', unsafe_allow_html=True)
    if nlp is None:
        st.caption("ℹ️ spaCy is not installed, so only the NLTK methods are shown.")
    tabs = st.tabs([f"{GROUP_ICONS[g]} {g}" for g in METHODS])
    for tab, (group, items) in zip(tabs, METHODS.items()):
        with tab:
            st.markdown(f'<div class="group-info">{GROUP_INFO[group]}</div>', unsafe_allow_html=True)
            items = [i for i in items if not (i[2] and nlp is None)]
            for start in range(0, len(items), 3):  # max 3 per row; stacks on phones
                cols = st.columns(3)
                for col, (name, func, _) in zip(cols, items[start:start + 3]):
                    col.button(name, key=name, on_click=run_method, args=(name, func, group),
                               width="stretch")

if st.session_state.pop("warning", None):
    st.warning("Please type some text first.", icon="✍️")


# --- Step 3: results ---
def render_result(item):
    result, kind = item["result"], item["kind"]
    if kind == "list":
        is_sent = "Sentence" in item["method"]
        chips = "".join(
            f'<span class="chip{" sent" if is_sent else ""}"><small>{n}</small>{html.escape(str(t))}</span>'
            if is_sent else f'<span class="chip">{html.escape(str(t))}</span>'
            for n, t in enumerate(result, 1))
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
        st.caption(f"{len(result)} {'sentences' if is_sent else 'tokens'}")
    elif kind == "metrics":
        cols = st.columns(len(result))
        for col, row in zip(cols, result):
            col.metric(row["Measure"], row["Count"])
    else:
        st.dataframe(result, width="stretch", hide_index=True)


st.markdown("")
st.markdown(f'<div class="section-label">3 · Results <span class="meta">{len(st.session_state.history)} so far, newest first</span></div>',
                unsafe_allow_html=True)

if not st.session_state.history:
    st.info("No results yet. Enter some text above and click a method.", icon="👆")

for i, item in enumerate(st.session_state.history):
    with st.container(border=True):
        st.markdown(f'<span class="badge">{GROUP_ICONS.get(item["group"], "")} {html.escape(item["method"])}</span>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="label">Input</div><div class="input-box">{html.escape(item["input"])}</div>'
                    f'<div class="label">Result</div>', unsafe_allow_html=True)
        render_result(item)
        st.button("Remove", icon=":material/close:", key=f"rm_{i}_{id(item)}", type="tertiary",
                  help="Remove this result", on_click=remove_result, args=(i,))

st.markdown("")
st.caption("Built with Streamlit, NLTK and spaCy · MS AI, University of Central Punjab")