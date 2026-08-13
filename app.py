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

# Telemetria kikapcsolása és alapvető figyelmeztetések elrejtése
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

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
  ezt egyértelműen jelezd, és j
