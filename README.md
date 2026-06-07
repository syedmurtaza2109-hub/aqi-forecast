# 🌫️ AQI Forecast — 3-Day Air Quality Prediction

A fully serverless ML pipeline that predicts the Air Quality Index (AQI) for
your city 3 days ahead, using real-time data and multiple ML models.

---

## 📁 Project Structure

```
aqi_project/
├── config.py                        ← shared settings (reads from .env)
├── requirements.txt                 ← all Python packages
├── .env.example                     ← template for your API keys
├── eda.ipynb                        ← exploratory data analysis notebook
│
├── feature_pipeline/
│   ├── fetch_and_store.py           ← runs hourly: fetch API → store features
│   └── backfill.py                  ← one-time: fill historical data for training
│
├── training_pipeline/
│   └── train.py                     ← daily: train models, save best to registry
│
├── inference_pipeline/
│   ├── predict.py                   ← generate 72-hour forecast
│   └── explain.py                   ← SHAP feature importance
│
├── dashboard/
│   ├── app.py                       ← Streamlit web dashboard
│   └── api.py                       ← Flask REST API backend
│
└── .github/workflows/
    ├── feature_pipeline.yml         ← hourly GitHub Actions job
    └── training_pipeline.yml        ← daily GitHub Actions job
```

---

## 🚀 Setup Guide (Step by Step)

### Step 1 — Get your free API keys

**A. AQICN (air quality data)**
1. Go to https://aqicn.org/data-platform/token/
2. Enter your email and click "Send me my token"
3. Check your email and copy the token

**B. OpenWeatherMap (weather data)**
1. Go to https://home.openweathermap.org/users/sign_up
2. Create a free account
3. Go to https://home.openweathermap.org/api_keys
4. Copy the "Default" API key

**C. Hopsworks (feature store + model registry)**
1. Go to https://app.hopsworks.ai and sign up for free
2. Create a new project called `aqi_forecast`
3. Go to Settings (top right) → API Keys → Create new key
4. Copy the key

---

### Step 2 — Set up your local environment

```bash
# Clone the project
git clone https://github.com/YOUR_USERNAME/aqi-forecast.git
cd aqi-forecast

# Create a virtual environment
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate
# Activate it (Windows)
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

---

### Step 3 — Configure your API keys

```bash
# Copy the example file
cp .env.example .env

# Open .env in any text editor and fill in your keys:
AQICN_TOKEN=paste_your_aqicn_token
OPENWEATHER_API_KEY=paste_your_openweather_key
HOPSWORKS_API_KEY=paste_your_hopsworks_key
HOPSWORKS_PROJECT_NAME=aqi_forecast
```

> ⚠️ Never commit `.env` to GitHub — it's already in `.gitignore`

---

### Step 4 — Find your city's AQICN station ID

1. Go to https://aqicn.org/map/
2. Search for your city and click on the nearest station marker
3. The URL will look like: `https://aqicn.org/city/pakistan/karachi/`
4. You can also use `@` + the station number (Karachi = `@9530`)
5. Update `AQICN_STATION` in your `.env`

---

### Step 5 — Run the historical backfill (one time only)

This collects past data so the model has enough to train on.

```bash
python feature_pipeline/backfill.py --days 180 --hours-per-day 4
```

This will take ~20 minutes. It's only run once.

---

### Step 6 — Run the feature pipeline once to test it

```bash
python feature_pipeline/fetch_and_store.py
```

You should see something like:
```
[2026-06-03 10:00:00] Starting feature pipeline for Karachi…
  Fetching AQI data…
  AQI: 87.0  PM2.5: 45.0  PM10: 62.0
  Fetching weather data…
  Temp: 33.2°C  Humidity: 68%  Wind: 4.1 m/s
  Storing in Feature Store…
✓ Features stored successfully.
```

---

### Step 7 — Train the models

```bash
python training_pipeline/train.py
```

This will:
- Load all features from Hopsworks
- Train Ridge Regression, Random Forest, Gradient Boosting, and TensorFlow LSTM
- Print RMSE / MAE / R² for each
- Save the best model back to Hopsworks

---

### Step 8 — Run the dashboard locally

Open **two terminals**:

**Terminal 1 — Flask API:**
```bash
python dashboard/api.py
```

**Terminal 2 — Streamlit:**
```bash
streamlit run dashboard/app.py
```

Then open http://localhost:8501 in your browser.

---

### Step 9 — Set up automation (GitHub Actions)

1. Push your code to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/aqi-forecast.git
   git push -u origin main
   ```

2. Go to your GitHub repo → **Settings → Secrets and variables → Actions**

3. Add these **Secrets** (sensitive values):
   | Name | Value |
   |------|-------|
   | `AQICN_TOKEN` | your AQICN token |
   | `OPENWEATHER_API_KEY` | your OpenWeather key |
   | `HOPSWORKS_API_KEY` | your Hopsworks key |
   | `HOPSWORKS_PROJECT_NAME` | `aqi_forecast` |

4. Add these **Variables** (non-sensitive):
   | Name | Value |
   |------|-------|
   | `CITY_NAME` | `Karachi` |
   | `CITY_LAT` | `24.8607` |
   | `CITY_LON` | `67.0011` |
   | `AQICN_STATION` | `@9530` |

5. Go to **Actions** tab → you should see both workflows listed.
   Click "Run workflow" to test them manually.

After this:
- The **feature pipeline** runs automatically every hour ✅
- The **training pipeline** retrains every day at 02:00 UTC ✅

---

### Step 10 — Deploy dashboard to Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub
2. Click **New app**
3. Select your repository
4. Set **Main file path** to `dashboard/app.py`
5. Click **Advanced settings → Secrets** and paste:
   ```toml
   AQICN_TOKEN = "your_token"
   OPENWEATHER_API_KEY = "your_key"
   HOPSWORKS_API_KEY = "your_key"
   HOPSWORKS_PROJECT_NAME = "aqi_forecast"
   CITY_NAME = "Karachi"
   CITY_LAT = "24.8607"
   CITY_LON = "67.0011"
   AQICN_STATION = "@9530"
   API_BASE_URL = "http://localhost:5000"
   ```
6. Click **Deploy** — your app will be live in ~2 minutes!

---

## 📊 Dashboard Features

| Feature | Description |
|---------|-------------|
| Current AQI badge | Live colour-coded AQI with level label |
| 3-day hourly forecast | Bar chart coloured by AQI level |
| Daily summary table | Min/max/avg per day |
| Historical trend | 7-day AQI + temperature/humidity charts |
| Pollutant breakdown | PM2.5 and PM10 over time |
| SHAP explanations | Which features drive predictions most |
| Hazard alerts | Automatic warnings for unhealthy air |
| AQI scale legend | Good → Hazardous reference |

---

## 🤖 Models Trained

| Model | Description |
|-------|-------------|
| Ridge Regression | Fast linear baseline |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Stronger ensemble, handles non-linearity |
| TensorFlow LSTM | Deep learning, captures time sequences |

The best model (lowest RMSE) is automatically selected and saved.

---

## 🔁 How the Pipeline Works

```
Every hour:
  GitHub Actions → fetch_and_store.py → Hopsworks Feature Store

Every day at 2am:
  GitHub Actions → train.py → Hopsworks Model Registry

Dashboard:
  Streamlit ←→ Flask API → Hopsworks → predictions
```

---

## ❓ Common Issues

**"AQICN returned status: Unknown station"**
→ Double-check your `AQICN_STATION` value. Try using the city name: `karachi` instead of `@9530`

**"OpenWeather 401 Unauthorized"**
→ New API keys take up to 2 hours to activate. Wait and retry.

**"Hopsworks login failed"**
→ Check your `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT_NAME` match exactly.

**"Not enough data to train"**
→ Run the backfill script first: `python feature_pipeline/backfill.py --days 180`

**Dashboard shows no data**
→ Make sure Flask is running (`python dashboard/api.py`) before starting Streamlit.
