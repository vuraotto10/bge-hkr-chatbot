"""
BGE HKR RAG-chatbot - Streamlit felület (Gemini-verzió)

Ez a fájl a szakdolgozat prototípusának éles, hallgatók által
tesztelhető felülete. A vektortárat (chroma_db mappa) előre el kell
készíteni a 01_index_epites_es_teszt.ipynb notebook futtatásával,
mielőtt ezt az appot elindítanád.

Futtatás:
streamlit run app.py
"""

# ------------------------------------------------------------------
# FONTOS:
# A Streamlit file watcher + PyTorch torch.classes együttese
# "Tried to instantiate class '__path__._path'" hibát okozhat.
#
# A file watcher kikapcsolása ezt a problémát megszünteti.
# Ennek a fájl LEGELEJÉN kell lennie, minden más import előtt.
# ------------------------------------------------------------------
import os

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# ------------------------------------------------------------------
# ChromaDB / SQLite kompatibilitás
#
# Streamlit Cloud környezetben a beépített sqlite3 verziója
# bizonyos esetekben túl régi lehet a ChromaDB számára.
# Ezért pysqlite3-binary használata esetén lecseréljük az sqlite3-at.
# ------------------------------------------------------------------
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# ------------------------------------------------------------------
# Alap importok
# ------------------------------------------------------------------
import warnings
import logging
import time
import csv
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ------------------------------------------------------------------
# Környezeti változók
# ------------------------------------------------------------------
load_dotenv()


# ------------------------------------------------------------------
# Telemetria és felesleges figyelmeztetések kikapcsolása
# ------------------------------------------------------------------
os.environ["ANONYMIZED_TELEMETRY"] = "False"

warnings.filterwarnings("ignore")

logging.getLogger("chromadb").setLevel(logging.ERROR)


# ------------------------------------------------------------------
# Streamlit oldal konfiguráció
# ------------------------------------------------------------------
st.set_page_config(
    page_title="BGE HKR Asszisztens",
    page_icon="🎓",
    layout="centered"
)


# ------------------------------------------------------------------
# Alapbeállítások
# ------------------------------------------------------------------
TOP_K = 5

LOG_FAJL = "hasznalati_naplo.csv"


# ------------------------------------------------------------------
# LangSmith
#
# Ha a Streamlit Secrets között megtalálható a
# LANGCHAIN_API_KEY, akkor bekapcsoljuk a LangSmith követést.
# Ha nincs beállítva, az alkalmazás ettől még működik.
# ------------------------------------------------------------------
_LANGSMITH_AKTIV = False
_langsmith_tracer = None

if "LANGCHAIN_API_KEY" in st.secrets:

    try:

        from langsmith import Client as LangSmithClient
        from langchain_core.tracers.langchain import LangChainTracer

        _langsmith_client = LangSmithClient(
            api_url="https://eu.api.smith.langchain.com",
            api_key=str(
                st.secrets["LANGCHAIN_API_KEY"]
            ).strip(),
        )

        _langsmith_tracer = LangChainTracer(
            project_name=str(
                st.secrets.get(
                    "LANGCHAIN_PROJECT",
                    "bge-hkr-chatbot"
                )
            ).strip(),
            client=_langsmith_client,
        )

        _LANGSMITH_AKTIV = True

    except Exception as e:

        print(
            f"LangSmith inicializálási hiba: {e}"
        )


# ------------------------------------------------------------------
# Vektortár és Gemini modell betöltése
#
# @st.cache_resource miatt a modellek csak egyszer töltődnek be,
# nem minden kérdés után újra.
# ------------------------------------------------------------------
@st.cache_resource
def betoltes():

    # --------------------------------------------------------------
    # Embedding modell
    # --------------------------------------------------------------
    embedding_modell = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-mpnet-base-v2"
        )
    )

    # --------------------------------------------------------------
    # ChromaDB vektortár
    # --------------------------------------------------------------
    vektortár = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_modell,
    )

    # --------------------------------------------------------------
    # Google Gemini
    # --------------------------------------------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0
    )

    return vektortár, llm


# ------------------------------------------------------------------
# Modellek betöltése
# ------------------------------------------------------------------
vektortár, llm = betoltes()


# ------------------------------------------------------------------
# RAG prompt
# ------------------------------------------------------------------
PROMPT_SABLON = ChatPromptTemplate.from_template(
    """
Te a Budapesti Gazdasági Egyetem (BGE) hallgatói asszisztense vagy.

Kizárólag az alábbi, a Hallgatói Követelményrendszerből (HKR)
származó szövegrészletek alapján válaszolj a hallgató kérdésére.

SZABÁLYOK:

- Ha a válasz egyértelműen megtalálható a szövegrészletekben,
  válaszolj pontosan és tömören, magyar nyelven.

- Ha a szövegrészletek nem tartalmaznak elegendő információt
  a válaszhoz, ezt egyértelműen jelezd.

- Ne találj ki információt.

- Ne használj külső tudást olyan információ megadására,
  amely nem található meg a megadott szövegrészletekben.

- Ha a kérdésre nem adható biztos válasz a megadott
  kontextus alapján, mondd azt, hogy:
  "A rendelkezésre álló HKR-részletek alapján erre nem tudok
  biztos választ adni."

- Ha lehetséges, hivatkozz a releváns HKR-részletre.

- Ne állíts olyat, ami ellentmond a megadott szövegrészleteknek.

- A választ magyar nyelven add meg.

KONTEXTUS:

{context}

HALLGATÓ KÉRDÉSE:

{question}

VÁLASZ:
"""
)


# ------------------------------------------------------------------
# Használati napló
# ------------------------------------------------------------------
def naplozas(kerdes, valasz):

    try:

        fajl_lett_letrehozva = os.path.exists(LOG_FAJL)

        with open(
            LOG_FAJL,
            "a",
            newline="",
            encoding="utf-8"
        ) as fajl:

            iro = csv.writer(fajl)

            if not fajl_lett_letrehozva:

                iro.writerow(
                    [
                        "idopont",
                        "kerdes",
                        "valasz"
                    ]
                )

            iro.writerow(
                [
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    kerdes,
                    valasz
                ]
            )

    except Exception as e:

        print(
            f"Naplózási hiba: {e}"
        )


# ------------------------------------------------------------------
# Oldal címe
# ------------------------------------------------------------------
st.title("🎓 BGE HKR Asszisztens")

st.markdown(
    """
Ez az alkalmazás a Budapesti Gazdasági Egyetem
Hallgatói Követelményrendszerének (HKR) dokumentumai
alapján válaszol a hallgatói kérdésekre.

A válaszok RAG-alapú keresés és Gemini nyelvi modell
segítségével készülnek.
"""
)


# ------------------------------------------------------------------
# Kérdés bevitele
# ------------------------------------------------------------------
kerdes = st.chat_input(
    "Írd be a kérdésedet..."
)


# ------------------------------------------------------------------
# Kérdés feldolgozása
# ------------------------------------------------------------------
if kerdes:

    # --------------------------------------------------------------
    # Felhasználói kérdés megjelenítése
    # --------------------------------------------------------------
    with st.chat_message("user"):

        st.markdown(kerdes)


    # --------------------------------------------------------------
    # Asszisztens válasza
    # --------------------------------------------------------------
    with st.chat_message("assistant"):

        with st.spinner(
            "Keresés a HKR dokumentumban..."
        ):

            try:

                # --------------------------------------------------
                # Hasonló dokumentumrészek keresése
                # --------------------------------------------------
                dokumentumok = vektortár.similarity_search(
                    kerdes,
                    k=TOP_K
                )


                # --------------------------------------------------
                # Ellenőrzés
                # --------------------------------------------------
                if not dokumentumok:

                    valasz = (
                        "A rendelkezésre álló HKR-dokumentumok "
                        "alapján nem találtam releváns információt."
                    )

                    st.warning(valasz)

                    naplozas(
                        kerdes,
                        valasz
                    )

                else:

                    # ----------------------------------------------
                    # Kontextus összeállítása
                    # ----------------------------------------------
                    kontextus_reszek = []

                    for dokumentum in dokumentumok:

                        szoveg = dokumentum.page_content

                        metadata = dokumentum.metadata

                        forras = metadata.get(
                            "source",
                            "HKR dokumentum"
                        )

                        oldal = metadata.get(
                            "page",
                            None
                        )

                        if oldal is not None:

                            forras_info = (
                                f"{forras}, oldal: {oldal}"
                            )

                        else:

                            forras_info = str(
                                forras
                            )

                        kontextus_reszek.append(
                            f"[Forrás: {forras_info}]\n"
                            f"{szoveg}"
                        )


                    context = "\n\n---\n\n".join(
                        kontextus_reszek
                    )


                    # ----------------------------------------------
                    # Prompt elkészítése
                    # ----------------------------------------------
                    prompt = PROMPT_SABLON.invoke(
                        {
                            "context": context,
                            "question": kerdes
                        }
                    )


                    # ----------------------------------------------
                    # Gemini meghívása
                    # ----------------------------------------------
                    if _LANGSMITH_AKTIV:

                        valasz_obj = llm.invoke(
                            prompt,
                            config={
                                "callbacks": [
                                    _langsmith_tracer
                                ]
                            }
                        )

                    else:

                        valasz_obj = llm.invoke(
                            prompt
                        )


                    # ----------------------------------------------
                    # Válasz szövegének kinyerése
                    # ----------------------------------------------
                    valasz = valasz_obj.content


                    # ----------------------------------------------
                    # Válasz megjelenítése
                    # ----------------------------------------------
                    st.markdown(valasz)


                    # ----------------------------------------------
                    # Források megjelenítése
                    # ----------------------------------------------
                    with st.expander(
                        "📚 Felhasznált HKR-részletek"
                    ):

                        for i, dokumentum in enumerate(
                            dokumentumok,
                            start=1
                        ):

                            metadata = dokumentum.metadata

                            forras = metadata.get(
                                "source",
                                "HKR dokumentum"
                            )

                            oldal = metadata.get(
                                "page",
                                None
                            )

                            st.markdown(
                                f"**{i}. forrás:** "
                                f"{forras}"
                            )

                            if oldal is not None:

                                st.caption(
                                    f"Oldal: {oldal}"
                                )

                            st.text(
                                dokumentum.page_content
                            )

                            if i < len(dokumentumok):

                                st.divider()


                    # ----------------------------------------------
                    # Naplózás
                    # ----------------------------------------------
                    naplozas(
                        kerdes,
                        valasz
                    )


            except Exception as e:

                # --------------------------------------------------
                # Hiba kezelése
                # --------------------------------------------------
                hiba_szoveg = str(e)

                st.error(
                    "Hiba történt a kérdés feldolgozása közben."
                )

                with st.expander(
                    "Technikai részletek"
                ):

                    st.code(
                        hiba_szoveg
                    )

                naplozas(
                    kerdes,
                    f"HIBA: {hiba_szoveg}"
                )


# ------------------------------------------------------------------
# Oldalsáv
# ------------------------------------------------------------------
with st.sidebar:

    st.header("ℹ️ Az alkalmazásról")

    st.write(
        """
Ez a prototípus a BGE Hallgatói
Követelményrendszerét használja
tudásbázisként.

A rendszer működése:

1. A hallgató kérdést tesz fel.
2. A rendszer releváns HKR-részleteket keres.
3. A megtalált szövegrészeket a Gemini modell kapja meg.
4. A modell ezek alapján készíti el a választ.
        """
    )

    st.divider()

    st.caption(
        f"Top-K keresési érték: {TOP_K}"
    )

    if _LANGSMITH_AKTIV:

        st.success(
            "LangSmith: aktív"
        )

    else:

        st.info(
            "LangSmith: nincs bekapcsolva"
        )
