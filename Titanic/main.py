import train, process
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

def main():
    file_path = "data/train.csv"
    raw_data = process.load_data(file_path)

    #process.analyse(raw_data)
    cleaned_data = process.clean_data(raw_data)

    model = train.train_model(cleaned_data)
    train.evaluate_model(model, cleaned_data)

    prep_submit(model)


def prep_submit(model):
    file_path = "data/test.csv"
    raw_data = process.load_data(file_path)
    id = raw_data["PassengerId"]
    raw_data['FamilySize'] = raw_data['SibSp'] + raw_data['Parch'] + 1
    raw_data['IsAlone'] = 1
    raw_data['IsAlone'].loc[raw_data['FamilySize'] > 1] = 0
    data = raw_data[["Sex", "Pclass", "Fare", "Age", "Embarked", "IsAlone"]]
    data['AgeGroup'] = data['Age'].apply(lambda x: 0 if x < 16 else 1 if x < 64 else 2)
    data['FareAbove20'] = (data['Fare'] > 15).astype(int)

    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 0})
    data.drop("Age", axis=1, inplace=True)
    data.drop("Fare", axis=1, inplace=True)
    categorical_features = ['Pclass']
    data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    pred = model.predict(data_encoded)
    submission = pd.DataFrame({
        'PassengerId': id,
        'Survived': pred
    })
    print(submission.head())
    submission.to_csv("submission.csv", index=False)
    return None

if __name__ == "__main__":
    main()