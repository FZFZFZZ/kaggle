from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def train_model(df):
    X = df.drop("Survived", axis = 1)
    y = df["Survived"]
    model = RandomForestClassifier(n_estimators=500, 
                                   random_state=1,
                                   max_depth=5,
                                   min_samples_leaf=6,
                                   max_features="sqrt",
                                   oob_score=True,
                                   n_jobs=-1)
    model.fit(X, y)
    print("OOB Score:", model.oob_score_)
    return model

def evaluate_model(model, df):
    X = df.drop("Survived", axis = 1)
    y = df["Survived"]
    y_pred = model.predict(X)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    return None