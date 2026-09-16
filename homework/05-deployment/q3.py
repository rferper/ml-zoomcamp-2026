import pickle

with open('pipeline.bin', 'rb') as f_in:
    artifact = pickle.load(f_in)

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

# pipeline.bin may be a scikit-learn Pipeline or a (dv, model) tuple
if isinstance(artifact, tuple):
    dv, model = artifact
    proba = model.predict_proba(dv.transform([lead]))[0, 1]
else:
    proba = artifact.predict_proba([lead])[0, 1]

print(round(float(proba), 3))
