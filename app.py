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
import warnings
import logging

# Csak a zavaró, ártalmatlan figyelmeztető/telemetria-üzeneteket némítjuk el
# a naplóban - ez nem befolyásolja a rendszer tényleges működését.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

import time
import csv
import uuid
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

TOP_K = 7
LOG_FAJL = "hasznalati_naplo.csv"


@st.cache_resource
def betoltes():
    """Egyszer töltődik be a vektortár és a modell, nem minden kérdésnél újra."""
    if not os.path.exists("chroma_db"):
        st.error(
            "HIBA: a 'chroma_db' mappa nem található. Előbb futtasd le a "
            "01_index_epites_es_teszt.ipynb notebookot, és told fel a "
            "létrejött 'chroma_db' mappát is a GitHub-repóba."
        )
        st.stop()

    import torch
    torch.set_num_threads(1)  # könnyebb erőforrás-terhelés a felhős szerveren

    embedding_modell = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
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
szövegrészletek alapján válaszolj a hallgató kérdésére. Minden szövegrészlet
elején zárójelben szerepel, melyik paragrafusból származik.

Szabályok:
- Figyelmesen olvasd át MINDEGYIK megadott szövegrészletet a válaszadás
  előtt - a releváns információ bármelyikben, akár a lista végén is
  szerepelhet.
- Ha a válasz egyértelműen megtalálható a szövegrészletekben, válaszolj
  pontosan és tömören, magyar nyelven.
- Ha a szövegrészletek nem tartalmaznak elegendő információt a válaszhoz,
  ezt egyértelműen jelezd, és javasold, hogy a hallgató forduljon a
  Tanulmányi Hivatalhoz. NE találj ki választ.
- NE zárd a válaszod forrásmegjelöléssel vagy paragrafushivatkozással -
  ezt a rendszer automatikusan, külön hozzáfűzi a válaszodhoz.

Szövegrészletek a HKR-ből:
---
{kontextus}
---

A hallgató kérdése: {kerdes}

Válasz:
""")


def forras_lista_epitese(talalt_szegmensek):
    """
    A ténylegesen visszaadott (top-k) szegmensek paragrafus-metaadatából
    PROGRAMBÓL, nem a modell által generálva állítja össze a forráslistát.
    """
    paragrafusok = sorted(
        {sz.metadata.get("paragrafus", "ismeretlen") for sz in talalt_szegmensek},
        key=lambda p: (p == "ismeretlen", int(p) if p.isdigit() else 0),
    )
    szamozott = [f"{p}. §" for p in paragrafusok if p != "ismeretlen"]
    if not szamozott:
        return "a rendszer nem tudott konkrét paragrafust azonosítani"
    return ", ".join(szamozott)


def naplozas(kerdes, valaszido_masodperc, munkamenet_azonosito):
    """
    Minden kérdés-válasz párost elment egy CSV-fájlba, hogy a
    2.3-as fejezetben leírt időmérési metrikát utólag ki tudd
    értékelni. A munkamenet-azonosító a felhasználó bevonása
    nélkül, automatikusan generálódik - nem kell tőle semmilyen
    személyes vagy azonosító adatot bekérni.
    """
    uj_fajl = not os.path.exists(LOG_FAJL)
    with open(LOG_FAJL, "a", newline="", encoding="utf-8") as f:
        iro = csv.writer(f)
        if uj_fajl:
            iro.writerow(["idobelyeg", "munkamenet_azonosito", "kerdes", "valaszido_masodperc"])
        iro.writerow([datetime.now().isoformat(), munkamenet_azonosito, kerdes, round(valaszido_masodperc, 2)])


# ------------------------------------------------------------------
# Munkamenet-azonosító - automatikusan generálódik, nem kérünk be
# semmilyen adatot a felhasználótól (a témavezető kifejezett kérése).
# ------------------------------------------------------------------
if "munkamenet_azonosito" not in st.session_state:
    st.session_state.munkamenet_azonosito = str(uuid.uuid4())[:8]

munkamenet_azonosito = st.session_state.munkamenet_azonosito

# ------------------------------------------------------------------
# Felület
# ------------------------------------------------------------------
st.title("🎓 BGE Hallgatói Követelményrendszer - Asszisztens")
st.caption("Szakdolgozati kutatási prototípus - kérdezz a BGE HKR-ről!")

with st.sidebar:
    st.markdown("---")
    st.markdown(
        "Ez egy kutatási célú prototípus. A kitöltés és a tesztelés önkéntes "
        "és anonim. A rendszer a kutatás céljából naplózza a feltett "
        "kérdéseket és a válaszidőt, de semmilyen személyes vagy azonosító "
        "adatot nem kér be és nem rögzít. A válaszok nem helyettesítik a "
        "Tanulmányi Hivatal hivatalos tájékoztatását."
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
    st.session_state.elozmenyek.append({"szerep": "user", "tartalom": kerdes})
    with st.chat_message("user"):
        st.markdown(kerdes)

    with st.chat_message("assistant"):
        with st.spinner("Keresek a HKR-ben..."):
            kezdo_ido = time.time()

            talalt_szegmensek = vektortár.similarity_search(kerdes, k=TOP_K)
            kontextus = "\n\n".join(
                f"[{sz.metadata.get('paragrafus', 'ismeretlen')}. §]\n{sz.page_content}"
                for sz in talalt_szegmensek
            )
            prompt = PROMPT_SABLON.format(kontextus=kontextus, kerdes=kerdes)

            if _LANGSMITH_AKTIV:
                valasz = llm.invoke(
                    prompt,
                    config={
                        "callbacks": [_langsmith_tracer],
                        "metadata": {"munkamenet_azonosito": munkamenet_azonosito},
                        "tags": [munkamenet_azonosito],
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

            # A forráshivatkozást MI építjük fel a metaadatból, nem a modell.
            forras = forras_lista_epitese(talalt_szegmensek)
            valasz_szoveg = f"{valasz_szoveg}\n\n---\n*Forrás (a rendszer által azonosítva): {forras}*"

            valaszido = time.time() - kezdo_ido

        st.markdown(valasz_szoveg)
        st.caption(f"Válaszidő: {valaszido:.1f} másodperc")

    st.session_state.elozmenyek.append({"szerep": "assistant", "tartalom": valasz_szoveg})

    # Automatikus naplózás a 2.3-as fejezet időmérési metrikájához
    naplozas(kerdes, valaszido, munkamenet_azonosito)


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
