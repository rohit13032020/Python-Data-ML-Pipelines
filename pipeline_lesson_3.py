import pandas as pd

df = pd.read_csv("student-data.csv")

def remove_duplicates(df):
    return df.drop_duplicates()

def clear_data(df):
    df = df.fillna(0)
    return df

#Features Engineering
def transform_data(df):
    df["age_category"] = df["age"].apply(
        lambda x: "Young" if x < 18 else "Adult"
    )
    return df

def data_pipeline(data):
    df_remove_duplicates = remove_duplicates(data)
    df_clear_data = clear_data(df_remove_duplicates)
    df_transform_data = transform_data(df_clear_data)
    df_transform_data.to_csv(
        "student-data-cleaned.csv",
        index=False
    )

    return df_transform_data


# Execute pipeline
cleaned_df = data_pipeline(df)

print(cleaned_df.head())
print(cleaned_df.shape)

print("Pipeline completed successfully!")
