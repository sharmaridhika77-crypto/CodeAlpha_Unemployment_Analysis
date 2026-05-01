import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Improved dataset (more realistic)
data = {
    "State": ["Delhi", "Maharashtra", "UP", "Bihar", "Punjab", "Haryana", "Rajasthan", "Gujarat"],
    "Unemployment_Rate": [8.5, 7.2, 9.1, 10.4, 6.3, 7.8, 9.5, 6.9],
    "Year": [2023]*8
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

# ---------------- Graph 1 ----------------
plt.figure(figsize=(10,5))
sns.barplot(x="State", y="Unemployment_Rate", data=df)
plt.title("Unemployment Rate by State")
plt.xticks(rotation=45)
plt.show()

# ---------------- Graph 2 ----------------
plt.figure(figsize=(6,6))
plt.pie(df["Unemployment_Rate"], labels=df["State"], autopct='%1.1f%%')
plt.title("Unemployment Distribution")
plt.show()

# ---------------- Graph 3 ----------------
plt.figure(figsize=(8,5))
sns.histplot(df["Unemployment_Rate"], kde=True)
plt.title("Unemployment Rate Distribution")
plt.show()

# Analysis
highest = df.loc[df["Unemployment_Rate"].idxmax()]
lowest = df.loc[df["Unemployment_Rate"].idxmin()]

print("\nHighest Unemployment State:", highest["State"])
print("Rate:", highest["Unemployment_Rate"])

print("\nLowest Unemployment State:", lowest["State"])
print("Rate:", lowest["Unemployment_Rate"])