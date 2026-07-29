# BGE HKR RAG-chatbot - szakdolgozati prototípus

Ez a projekt a szakdolgozat 2.2-es fejezetében leírt RAG-architektúrát
valósítja meg: a BGE Hallgatói Követelményrendszerére (HKR) épülő,
zárt tudásbázisú chatbot.

## Fájlok

- `01_index_epites_es_teszt.ipynb` - Jupyter notebook, ami lépésről
  lépésre bemutatja és felépíti a vektortárat (indexelés). Ezt futtasd
  le ELŐSZÖR, mielőtt az app.py-t elindítanád.
- `app.py` - a Streamlit chat-felület, amit a hallgatók fognak
  használni a teszteléshez.
- `requirements.txt` - a szükséges Python-csomagok listája.
- `.env.example` - sablon az API-kulcsokhoz.
- `data/` - ide másold be a BGE HKR fájlt (PDF vagy DOCX formátumban).
- `chroma_db/` - ide kerül automatikusan a felépített vektortár
  (ezt a notebook hozza létre, neked nincs vele teendőd).

## Beüzemelés lépései (Anaconda környezetben)

### 1. Környezet létrehozása

```bash
conda create -n rag_szakdolgozat python=3.11
conda activate rag_szakdolgozat
```

### 2. Csomagok telepítése

```bash
pip install -r requirements.txt
```

### 3. API-kulcsok beállítása

Másold le a `.env.example` fájlt `.env` néven, és írd bele a saját
Anthropic és OpenAI API-kulcsaidat:

```bash
cp .env.example .env
```

Majd nyisd meg a `.env` fájlt egy szövegszerkesztőben, és cseréld ki
a helyőrző szövegeket a saját kulcsaidra.

- Anthropic API-kulcs: https://console.anthropic.com/settings/keys
- OpenAI API-kulcs: https://platform.openai.com/api-keys

### 4. A dokumentum elhelyezése

Másold a BGE HKR fájlját a `data/` mappába, majd a notebookban írd át
a fájl nevét a `DOKUMENTUM_UTVONAL` változóban, ha nem `bge_hkr.pdf`
a neve.

### 5. Az index felépítése

Nyisd meg a notebookot:

```bash
jupyter notebook
```

Futtasd le sorban az összes cellát a `01_index_epites_es_teszt.ipynb`
fájlban. Ez létrehozza a `chroma_db` mappát a kész vektortárral, és
egyben leteszteli, hogy a rendszer helyesen válaszol-e néhány
mintakérdésre.

### 6. A Streamlit app helyi tesztelése

```bash
streamlit run app.py
```

Ez megnyit egy böngészőablakot (általában `localhost:8501`), ahol
kipróbálhatod a chatbotot, mielőtt élesbe (Streamlit Community Cloud)
telepítenéd.

### 7. Élesítés (hallgatói teszteléshez)

1. Töltsd fel ezt a projektet (a `.env` fájl NÉLKÜL - azt soha ne
   oszd meg nyilvánosan!) egy GitHub-repóba.
2. Regisztrálj a https://share.streamlit.io oldalon.
3. Kösd össze a GitHub-repódat, és add meg az API-kulcsaidat a
   Streamlit Cloud "Secrets" beállításai között (nem a `.env`
   fájlban - a felhős verzióban ez máshogy működik, a Streamlit
   Cloud dokumentációja pontosan leírja, hogyan).
4. Kapsz egy állandó linket, amit el tudsz küldeni a
   tesztalanyoknak.

## A hasznalati_naplo.csv fájl

Az `app.py` minden kérdés-válasz párost automatikusan naplóz egy
`hasznalati_naplo.csv` fájlba (időbélyeg, tesztalany azonosítója,
kérdés, válaszidő másodpercben). Ezt az adatot tudod felhasználni
a szakdolgozat 3.2-es fejezetében az információkeresési idő
elemzéséhez - nem kell manuálisan stopperolnod a teszt közben.

## Fontos megjegyzés

Ez egy kutatási célú prototípus, nem éles, produkciós rendszer.
A `.env` fájlt és az API-kulcsaidat soha ne töltsd fel nyilvános
GitHub-repóba!
