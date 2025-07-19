import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("adult 3.csv")
df.replace("?", "Unknown", inplace=True)


label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


df['past_income'] = 0  
df['present_income'] = df['income']
df['future_income'] = df.apply(
    lambda row: 1 if row['educational-num'] >= 13 or row['hours-per-week'] >= 45 else row['income'], axis=1
)


X = df.drop(['income', 'past_income', 'present_income', 'future_income'], axis=1)


def train_and_predict(label_column):
    y = df[label_column]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    decoded_preds = label_encoders['income'].inverse_transform(preds)
    
    
    acc = accuracy_score(y, preds)
    print(f"\nAccuracy for {label_column}: {acc:.4f}")
    
    if len(set(y)) > 1:
        print(classification_report(y, preds, target_names=['<=50K', '>50K']))
    else:
        label = label_encoders['income'].inverse_transform([y.iloc[0]])[0]
        print(f"Only one class '{label}' in {label_column}. Skipping detailed report.")
    
    return decoded_preds


df['Predicted_Past_Income'] = train_and_predict('past_income')
df['Predicted_Present_Income'] = train_and_predict('present_income')
df['Predicted_Future_Income'] = train_and_predict('future_income')


print("\n All Predictions:")
print(df[['age', 'education', 'hours-per-week', 
          'Predicted_Past_Income', 'Predicted_Present_Income', 'Predicted_Future_Income']].to_string(index=False))


df.to_csv("predicted_salaries_full.csv", index=False)
print("\n All predictions saved to 'predicted_salaries_full.csv'")
