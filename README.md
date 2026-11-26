# Connected Karak News Classifier

Classifies news headlines in **English, Urdu and Pashto** into 7 categories:
Crime, Education, Health, Politics, Sports, Weather, Other.

Built by fine-tuning XLM-RoBERTa with Hugging Face Transformers, served with FastAPI.

## Features
- Single headline and batch (up to 200) classification
- Confidence scores for all categories, with an "unsure" warning
- Language guess, light/dark theme, history, CSV download
- "Send fix" button that saves corrections for retraining

## Results
About 86% accuracy on a small test set (22 headlines). The dataset is
small (86 headlines), so treat this as a learning project.

## Files
- `train.ipynb` - training and evaluation
- `app.py` + `index.html` - FastAPI app and UI
- `news_data.csv` - training data

## Model
The trained model (about 1.1 GB) is not stored here because it is too
large for GitHub. Run `train.ipynb` to create it.

## Screenshot
(add a screenshot of the app here)
