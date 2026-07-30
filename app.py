# --- STREAMLIT CLOUD SQLITE FIX ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ----------------------------------

import os
import time
import csv
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ------------------------------------------------------------------
# Alapbeállítások
# ------------------------------------------------------------------
st.set_page_config(page_title="BGE HKR Asszisztens", page_icon="🎓", layout="centered")

TOP_K = 5
LOG_FAJL = "hasznalati_naplo.csv"


@st.cache_resource
def betoltes():
    """Egyszer töltődik be a vektortár és a modell, nem minden kérdésnél újra."""
    embedding_modell = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    vektortár = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_modell,
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    return vektortár, llm


vektortár, llm = betoltes()

PROMPT_SABLON = ChatPromptTemplate.from_template("""
Te a Budapesti Gazdasági Egyetem (BGE) hallgatói asszisztense vagy.
Kizárólag az alábbi, a Hallgatói Követelményrendszerből (HKR) származó
szövegrészletek alapján válaszolj a hallgató kérdésére.

Szabályok:
- Ha a válasz egyértelműen megtalálható a szövegrészletekben, válaszolj
  pontosan és tömören, magyar nyelven.
- Ha a szövegrészletek nem tartalmaznak elegendő információt a válaszhoz,
  ezt egyértelműen jelezd, és javasold, hogy a hallgató forduljon a
  Tanulmányi Hivatalhoz. NE találj ki választ.
- A válasz végén röviden jelezd, mely szövegrészlet alapján válaszoltál.

Szövegrészletek a HKR-ből:
---
{kontextus}
---

A hallgató kérdése: {kerdes}

Válasz:
""")


def naplozas(kerdes, valaszido_masodperc, tesztalany_azonosito):
    uj_fajl = not os.path.exists(LOG_FAJL)
    with open(LOG_FAJL, "a", newline="", encoding="utf-8") as f:
        iro = csv.writer(f)
        if uj_fajl:
            iro.writerow(["idobelyeg", "tesztalany_azonosito", "kerdes", "valaszido_masodperc"])
        iro.writerow([datetime.now().isoformat(), tesztalany_azonosito, kerdes, round(valaszido_masodperc, 2)])


# ------------------------------------------------------------------
# Felület
# ------------------------------------------------------------------
st.title("🎓 BGE Hallgatói Követelményrendszer - Asszisztens")
st.caption("Szakdolgozati kutatási prototípus - kérdezz a BGE HKR-ről!")

with st.sidebar:
    st.subheader("Tesztelési adatok")
    tesztalany_azonosito = st.text_input(
        "Add meg az azonosítódat (amit a konzulensedtől/kutatótól kaptál)",
        value="",
        help="Ez segít összekötni a válaszaidat a UX-kérdőíveddel, anonim módon.",
    )
    st.markdown("---")
    st.markdown(
        "Ez egy kutatási célú prototípus. A rendszer kizárólag a BGE "
        "nyilvános szabályzatai alapján válaszol, és nem helyettesíti "
        "a Tanulmányi Hivatal hivatalos tájékoztatását."
    )

if "elozmenyek" not in st.session_state:
    st.session_state.elozmenyek = []

for uzenet in st.session_state.elozmenyek:
    with st.chat_message(uzenet["szerep"]):
        st.markdown(uzenet["tartalom"])

kerdes = st.chat_input("Írd be a kérdésed a BGE HKR-ről...")

if kerdes:
    if not tesztalany_azonosito:
        st.warning("Kérlek, add meg az azonosítódat a bal oldali sávban, mielőtt kérdezel!")
        st.stop()

    st.session_state.elozmenyek.append({"szerep": "user", "tartalom": kerdes})
    with st.chat_message("user"):
        st.markdown(kerdes)

    with st.chat_message("assistant"):
        with st.spinner("Keresek a HKR-ben..."):
            kezdo_ido = time.time()

            talalt_szegmensek = vektortár.similarity_search(kerdes, k=TOP_K)
            kontextus = "\n\n".join(sz.page_content for sz in talalt_szegmensek)
            prompt = PROMPT_SABLON.format(kontextus=kontextus, kerdes=kerdes)
            valasz = llm.invoke(prompt)

            if isinstance(valasz.content, str):
                valasz_szoveg = valasz.content
            else:
                valasz_szoveg = "".join(
                    resz.get("text", "") for resz in valasz.content if isinstance(resz, dict)
                )

            valaszido = time.time() - kezdo_ido

        st.markdown(valasz_szoveg)
        st.caption(f"Válaszidő: {valaszido:.1f} másodperc")

    st.session_state.elozmenyek.append({"szerep": "assistant", "tartalom": valasz_szoveg})
    naplozas(kerdes, valaszido, tesztalany_azonosito)
