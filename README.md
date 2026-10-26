# Campus Facility Reservation Analysis — Streamlit App

## Files
- `app.py` — Streamlit application converted from `TORQUITEDAISYMAE (2).ipynb`.
- `requirements.txt` — Python packages needed to run the app.
- `README.md` — setup and usage guide.

## Features
- Automatically creates the 15-record sample dataset; no CSV upload is required.
- Dataset preview, data-quality checks, cleaning, and feature engineering.
- Exploratory data analysis with charts and summary tables.
- Decision Tree classification with evaluation metrics and feature importance.
- Functional requirements table.
- Administrator dashboard with predicted outcomes, risk scores, and HIGH/MEDIUM/LOW ratings.
- Download buttons for cleaned data, original data, requirements, and predictions.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal/command prompt in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Streamlit:
   ```bash
   streamlit run app.py
   ```
5. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Deploy / use in Colab
Streamlit is normally launched from a terminal. In Google Colab, install the packages with `!pip install -r requirements.txt`, then run Streamlit in the background and expose it with a tunnel tool approved by your instructor. Do not publish private data.

## Important note
The notebook uses a small, automatically generated sample dataset. Model predictions and risk ratings are demonstrations, not reliable decisions for real reservations. A HIGH risk rating must not automatically reject a reservation; an administrator should review the details.
