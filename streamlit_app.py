"""
NLP Playground – Week 1 methods (NLP, MS AI, UCP)
Type a string, click a method, see the input and the result.
Methods follow the Week 1 slides: tokenization, stop words,
stemming, lemmatization and POS tagging (NLTK and spaCy).
"""

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
            {"Measure": "Sentences", "Count": len(sents)}], "table"


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

# ---------------------------------------------------------------
# 3. Session state: input text + history of results
# ---------------------------------------------------------------
if "text" not in st.session_state:
    st.session_state.text = ""
if "history" not in st.session_state:
    st.session_state.history = []


def run_method(name, func):
    text = st.session_state.text.strip()
    if not text:
        st.session_state.warning = "Please type a string first."
        return
    result, kind = func(text)
    st.session_state.history.insert(0, {"method": name, "input": text, "result": result, "kind": kind})


def clear_all():
    st.session_state.text = ""
    st.session_state.history = []


def clear_input():
    st.session_state.text = ""


# ---------------------------------------------------------------
# 4. Page layout
# ---------------------------------------------------------------
st.title("🧠 NLP Playground")
st.caption("Week 1 methods: tokenization, stop words, stemming, lemmatization and POS tagging")

st.text_area("Enter your text", key="text", height=120,
             placeholder="e.g. At eight o'clock on Thursday morning Raza didn't feel very good. "
                         "The weather is nice.")

c1, c2 = st.columns(2)
c1.button("Clear text", on_click=clear_input, width="stretch")
c2.button("Clear all results", on_click=clear_all, width="stretch", type="primary")

if nlp is None:
    st.info("spaCy model not found, so spaCy methods are hidden. "
            "Run: python -m spacy download en_core_web_sm")

st.subheader("Choose a method")
for group, items in METHODS.items():
    st.markdown(f"**{group}**")
    items = [i for i in items if not (i[2] and nlp is None)]
    for start in range(0, len(items), 3):  # max 3 buttons per row so labels fit
        row = items[start:start + 3]
        cols = st.columns(3)
        for col, (name, func, _) in zip(cols, row):
            col.button(name, key=name, on_click=run_method, args=(name, func), width="stretch")

if st.session_state.pop("warning", None):
    st.warning("Please type a string first.")

# ---------------------------------------------------------------
# 5. Results (newest first)
# ---------------------------------------------------------------
st.divider()
st.subheader("Results")
if not st.session_state.history:
    st.write("No results yet. Type a string and click a method.")

for i, item in enumerate(st.session_state.history):
    with st.container(border=True):
        st.markdown(f"**{item['method']}**")
        st.markdown("**Input:**")
        st.code(item["input"], language=None, wrap_lines=True)
        st.markdown("**Result:**")
        if item["kind"] == "list":
            st.write(item["result"])
            st.caption(f"{len(item['result'])} items")
        else:
            st.dataframe(item["result"], width="stretch", hide_index=True)