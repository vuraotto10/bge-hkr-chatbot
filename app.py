"""
BGE HKR RAG-chatbot - Streamlit felület (Gemini-verzió)

Ez a fájl a szakdolgozat prototípusának éles, hallgatók által
tesztelhető felülete. A vektortárat (chroma_db mappa) előre el kell
készíteni a 01_index_epites_es_teszt.ipynb notebook futtatásával,
mielőtt ezt az appot elindítanád.

Futtatás: streamlit run app.py
"""

# ------------------------------------------------------------------
# FONTOS: ennek a blokknak a fájl LEGELEJÉN, minden más import előtt
# kell lennie! A Streamlit Cloud alap sqlite3-verziója túl régi a
# ChromaDB-hez, ezért lecseréljük egy újabb, csomagolt verzióra
# (pysqlite3-binary), mielőtt bármi más importálná a sqlite3-at.
# ------------------------------------------------------------------
import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

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
# LangSmith - explicit, kódból létrehozott kliens (EU-régió)
# Nem környezeti változókra hagyatkozunk, hanem közvetlenül adjuk
# meg az adatokat, hogy elkerüljük az elnevezés-inkonzisztenciákat.
# ------------------------------------------------------------------
_LANGSMITH_AKTIV = False
_langsmith_tracer = None

if "LANGCHAIN_API_KEY" in st.secrets:
    try:
        from langsmith import Client as LangSmithClient
        from langchain_core.tracers.langchain import LangChainTracer

        _langsmith_client = LangSmithClient(
            api_url="https://eu.api.smith.langchain.com",
            api_key=str(st.secrets["LANGCHAIN_API_KEY"]).strip(),
        )
        _langsmith_tracer = LangChainTracer(
            project_name=str(st.secrets.get("LANGCHAIN_PROJECT", "bge-hkr-chatbot")).strip(),
            client=_langsmith_client,
        )
        _LANGSMITH_AKTIV = True
    except Exception as e:
        print(f"LangSmith inicializálási hiba: {e}")

# ------------------------------------------------------------------
# Alapbeállítások
# ------------------------------------------------------------------
st.set_page_config(page_title="BGE HKR Asszisztens", page_icon="🎓", layout="centered")

TOP_K = 5  # ugyanaz az érték, mint a notebookban - tartsd konzisztensen
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
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
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
    """
    Minden kérdés-válasz párost elment egy CSV-fájlba, hogy a
    2.3-as fejezetben leírt időmérési metrikát utólag ki tudd
    értékelni (nem kell manuálisan stoppert használnod).
    """
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

# Korábbi üzenetek megjelenítése
for uzenet in st.session_state.elozmenyek:
    with st.chat_message(uzenet["szerep"]):
        st.markdown(uzenet["tartalom"])

# Új kérdés bekérése
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

            if _LANGSMITH_AKTIV:
                valasz = llm.invoke(
                    prompt,
                    config={
                        "callbacks": [_langsmith_tracer],
                        "metadata": {"tesztalany_azonosito": tesztalany_azonosito},
                        "tags": [tesztalany_azonosito],
                    },
                )
            else:
                valasz = llm.invoke(prompt)

            # Gemini válasz kinyerése (kezeli ha szöveg vagy szótáras/listás formátum)
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

    # Automatikus naplózás a 2.3-as fejezet időmérési metrikájához
    naplozas(kerdes, valaszido, tesztalany_azonosito)


# ------------------------------------------------------------------
# Admin panel - ide te tudsz csak belépni, hogy letöltsd az adatokat
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    with st.expander("🔒 Admin"):
        admin_jelszo = st.text_input("Jelszó", type="password", key="admin_jelszo")
        elvart_jelszo = os.getenv("ADMIN_JELSZO", "") or st.secrets.get("ADMIN_JELSZO", "")

        if admin_jelszo and elvart_jelszo and admin_jelszo == elvart_jelszo:
            if os.path.exists(LOG_FAJL):
                with open(LOG_FAJL, "rb") as f:
                    st.download_button(
                        label="📥 Napló letöltése (CSV)",
                        data=f,
                        file_name="hasznalati_naplo.csv",
                        mime="text/csv",
                    )
                import pandas as pd
                df = pd.read_csv(LOG_FAJL)
                st.caption(f"Eddig {len(df)} kérdés-válasz páros érkezett.")
                st.dataframe(df.tail(10))
            else:
                st.info("Még nincs naplózott adat.")
        elif admin_jelszo:
            st.error("Hibás jelszó.")
