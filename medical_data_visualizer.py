import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("medical_examination.csv")

# Create a BMI Series ; use BMI to calculate df["overweight"]
BMI = pd.Series(df["weight"]/(df["height"]/100)**2)
df['overweight'] = BMI.apply(lambda x: 1 if x > 25 else 0)

# Normalise "gluc" and "cholesterol"
# Make 0 always good, make 1 always bad 
df["gluc"] = df["gluc"].apply(lambda x: 0 if x == 1 else 1)
df["cholesterol"] = df["cholesterol"].apply(lambda x: 0 if x == 1 else 1)


def draw_cat_plot():
    # Create a DataFrame with identifier column "cardio"
    # "variable" column is some columns of df , "values" columns is their associated values
    df_cat = df.melt(id_vars=["cardio"], value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])

    # Group by "cardio" AND "variable", count for each feature
    df_cat = df_cat.groupby(["cardio", "variable"]).value_counts()
    

    # Create a new DataFrame from df_cat, assigning value_counts() to the column "total"
    # Reset the index to use "cardio", "variable", "value" as columns
    df_cat = pd.DataFrame({"total": df_cat}).reset_index()

    # Plot a barchart from df_cat ; total is function of variable
    fig = sns.catplot(data=df_cat, x="variable", y="total", col="cardio", hue="value", kind="bar")
   

    # Save figure and return
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    # Filter out some data for the heatmap
    df_heat = df.loc[
        (df["ap_lo"] <= df["ap_hi"]) 
        & (df["height"] >= df["height"].quantile(0.025)) 
        & (df["height"] <= df["height"].quantile(0.975)) 
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw the heatmap
    sns.heatmap(
        corr,
        center= 0,
        annot= True,
        mask= mask,
        fmt=".1f",
        square=True,
        vmin=-.1,
        vmax=.7
    )

    # Save figure and return
    fig.savefig('heatmap.png')
    return fig
