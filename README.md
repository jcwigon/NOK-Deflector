# NOK Detector

## Overview
Aplikacja Streamlit do kontroli jakości (NOK) na podstawie zdjęć.  
W tym repo aktualnie dostępny jest moduł **Side Check (RIGHT vs LEFT)**, który wykrywa stronę montażu przez **template matching (OpenCV)** i oznacza wynik jako:
- `OK`
- `WRONG_SIDE`
- `UNCERTAIN` (niepewne)

Wyniki można eksportować do **CSV** oraz **XLSX**.

---

## Features
- Web UI w **Streamlit**
- Batch upload (wiele zdjęć naraz)
- Predykcja strony: `RIGHT` / `LEFT` / `UNCERTAIN`
- Status względem oczekiwania: `OK` / `WRONG_SIDE` / `UNCERTAIN`
- Podgląd zdjęć z overlayem + kolorami:
  - OK → zielony
  - WRONG_SIDE → czerwony
  - UNCERTAIN → pomarańczowy
- Tabela wyników (AgGrid):
  - klik w checkbox w tabeli podświetla zdjęcie
  - przycisk **Wybierz** przy zdjęciu zaznacza wiersz w tabeli
  - scroll (bez stronicowania)
- Eksport wyników: **CSV** i **XLSX**

---

## Requirements
- Python 3.10+ (zalecane)
- Windows 10/11 (testowane)
- Zależności z `requirements.txt`

---

## Installation (Windows)
```cmd
git clone https://github.com/jcwigon/nok-detector.git
cd nok-detector

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Run
```cmd
streamlit run app.py
```

Aplikacja odpali się domyślnie pod adresem:
- http://localhost:8501

---

## Model / konfiguracja
Aplikacja używa katalogu modelu zawierającego:
- `template.png`
- `config.json`

Ścieżkę ustawiasz w sidebarze jako **Model dir** (domyślna wartość jest w `app.py`).

---

## Usage
1. Uruchom aplikację.
2. W sidebarze ustaw:
   - `Model dir` (katalog z `template.png` i `config.json`)
   - `Oczekiwana strona (kafel)` → RIGHT/LEFT
   - (opcjonalnie) `Pokazuj też OK`
3. Wgraj zdjęcia (multi-upload).
4. Tabela:
   - zaznacz checkbox wiersza → podświetli zdjęcie niżej
5. Podgląd:
   - kliknij **Wybierz** przy zdjęciu → zaznaczy odpowiadający wiersz w tabeli
6. Pobierz wyniki jako CSV lub XLSX.

---

## Files
- `app.py` – Streamlit UI + tabela + eksport (CSV/XLSX)
- `side_model.py` – logika klasyfikacji strony + rysowanie overlay
- `requirements.txt` – zależności

---

## Dev notes
Aktualizacja `requirements.txt` po zmianach w zależnościach:
```cmd
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
```