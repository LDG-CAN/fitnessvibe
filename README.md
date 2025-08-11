
# Daily Fitness Log with AI Estimation

Streamlit app with SQLite persistence, calendar logging, targets, health score, and AI estimation.

AI modes:
- Local estimator: heuristic parser and fuzzy match against local foods.
- OpenAI API: enter your API key in the sidebar, the app calls `gpt-4o-mini` and expects strict JSON.

## Run
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

## Persistence
Data saved to `data/app.sqlite`. Use backup and restore in the sidebar.

## Notes
- The OpenAI mode is optional. If you provide a key, items are created automatically with the model’s per-serving macros.
- You can add your own foods in-app. The AI will use them on the next estimate.
