1.
summary = df.describe(include="all")
print("Summary Statistics:\n", summary)

extra_stats = df.select_dtypes(include='number').agg(['mean', 'median', 'std'])
print("\nExtra Stats (Mean, Median, Std):\n", extra_stats)   

2.
df[num_features].hist(figsize=(10, 6), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=14)
plt.show()

for col in num_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()
  features = ['Parch', 'SibSp', 'Fare', 'Age', 'Pclass', 'Sex', 'Embarked']
3.
plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Titanic Features")
plt.show()

titanic_df = sns.load_dataset('titanic')
numeric_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']
plot_data = titanic_df[numeric_features + ['survived']].dropna()
sns.pairplot(
    data=plot_data,
    hue="survived", 
    diag_kind="hist",
    palette="Set1"
)
plt.suptitle("Pairplot of Selected Features (Colored by Survival)", y=1.02, fontsize=16)
plt.show()

4.
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Sex")
plt.show()

plt.figure(figsize=(8,6))
sns.kdeplot(df [df ["Survived"]==1]["Age"], shade=True,
label="Survived")
sns.kdeplot(df [df ["Survived"]==0] ["Age"], shade=True,
label="Not Survived")
plt.title("Age Distributi Survival")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x="Survived", y="Fare", data=df)
plt.title("Fare vs Survival")
plt.show()

df["FamilySize"] = df ["SibSp"] + df ["Parch"] + 1
sns.countplot(x="FamilySize", hue="Survived", data=df)
plt.title("Survival by Family Size")
plt.show()
