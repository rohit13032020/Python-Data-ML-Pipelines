import pandas as pd

df = pd.read_csv("student-data.csv")

def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df

def transform_data(df):
    df["absence_category"] = df["absences"].apply(
        lambda x : "No Absence" if x == 0
        else "Low" if x<=5
        else "High"
    )
    return df

df = clean_data(df)
df = transform_data(df)

df.to_csv("student-data-cleaned.csv", index=False)

print("Pipeline completed successfully. Cleaned data saved to student-data-cleaned.csv")