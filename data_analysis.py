import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import shapiro, levene, f_oneway, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.oneway import anova_oneway

"""
Read Data
"""

# Reading Data
path = 'data/australian_vehicle_prices_cleaned.csv'
df = pd.read_csv(path)

print("\nOriginal Data Frame; features frequencies")
for c in df:
    numbers = df[c].unique()
    print(f"Numbers for {c} : {len(numbers)}")




"""
Numerical features
"""

# Pearson Correlation
# numeric columns (based on this dataset)
numeric_cols = ['Year', 'Engine', 'FuelConsumption' , 'Kilometres', 'CylindersinEngine' , 'Doors', 'Seats', 'Price']

# compute Pearson correlation between Price and each other numeric feature
results = []
for col in numeric_cols:
    if col != 'Price':
        r, p = pearsonr(df['Price'], df[col])
        results.append({'Feature': col, 'Pearson_r': r, 'p_value': p})

corr_df = pd.DataFrame(results).sort_values(by='Pearson_r', ascending=False)
print(corr_df)

corr_df.sort_values(by='Pearson_r', ascending=True).plot(
    kind='barh', x='Feature', y='Pearson_r', legend=False, figsize=(6,4)
)
plt.title('Pearson Correlation with Price')
plt.xlabel('Correlation Coefficient (r)')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.savefig('result/pearson.png', dpi=300, bbox_inches='tight')

# create one subplot per numeric column
plt.figure(figsize=(12, 6))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 4, i)  # adjust grid size if you have more or fewer columns
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('result/num_distribution.png', dpi=300, bbox_inches='tight')




"""
Categorical features
"""

# categorical columns (based on this dataset)
cat_cols = ['Brand', 'Model', 'Car/Suv' , 'UsedOrNew', 'Transmission' , 'DriveType', 'FuelType', 'ColourExtInt' , 'Location', 'BodyType']

plt.figure(figsize=(12, 6))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(3, 4, i)
    sns.histplot(df[col], kde=True)  
    plt.title(f'{col} Distribution')
    plt.xlabel('')  # remove x-axis label
    plt.xticks([])  # hide x-axis ticks/labels
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('result/cat_distribution.png', dpi=300, bbox_inches='tight')

#

top_n = 5  # top most frequent values

plt.figure(figsize=(12, 8))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(3, 4, i)
    top_vals = df[col].value_counts().nlargest(top_n)
    sns.barplot(x=top_vals.index, y=top_vals.values)
    plt.title(f'{col} (Top {top_n})')
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.ylabel('Count')

plt.tight_layout()
plt.savefig('result/top_cat_frequency.png', dpi=300, bbox_inches='tight')

#

plt.figure(figsize=(15,6))
sns.boxplot(x='Brand', y='Price', data=df)
plt.xticks(rotation=90)
plt.savefig('result/brand_boxplot.png', dpi=300, bbox_inches='tight')

#

# comparing used/new prices

df_usednew = df[(df['UsedOrNew'] == 'USED') | (df['UsedOrNew'] == 'NEW')]
print(f'Average price: {df.groupby('UsedOrNew')['Price'].mean().round()}')

plt.figure(figsize=(10, 5))
# box plot
plt.subplot(1, 2, 1)
sns.boxplot(x='UsedOrNew', y='Price', data=df_usednew, showmeans=True)
plt.title('Price Distribution: Used vs New Cars')
plt.xlabel('Car Condition')
plt.ylabel('Price')


# scatter plot
plt.subplot(1, 2, 2)
sns.stripplot(x='UsedOrNew', y='Price', data=df_usednew, jitter=True, alpha=0.6)
plt.title('Price Distribution: Used vs New Cars')
plt.xlabel('Car Condition')
plt.ylabel('Price')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('result/used_new_price_distribution.png', dpi=300, bbox_inches='tight')

#

# function for calculating Normality, variance homogeneity, ANOVA
def compare_test_summary(df, cat_cols):
    results = []

    for column in cat_cols:
        print(f"\n===== Analyzing {column} =====")

        num_unique = df[column].nunique()
        print(f"Number of unique {column}: {num_unique}")

        # Keep only the Column with at least 3 valid price entries
        valid_groups = [
            var for var, group in df.groupby(column)
            if group["Price"].notna().sum() >= 3
        ]
        df_sub = df[df[column].isin(valid_groups)]
        num_valid = len(valid_groups)
        print(f"Remaining {column} with ≥3 samples: {len(valid_groups)}")
        

        if num_valid < 2:
            print(f"⚠️ Skipping {column} — less than 2 groups with ≥3 samples.")
            results.append([column, num_unique, num_valid, np.nan, np.nan, np.nan])
            continue

        '''
        assumption : 
            1- normality per features (column) (Shapiro–Wilk)
            2- variance homogeneity (Levene test)
        test:
            ANOVA if both assumptions pass
            Welch’s ANOVA if normal but variances unequal
            Kruskal–Wallis if non-normal
        '''

        # === Step 1. Normality check ===
        normality_results = {}
        for variable, group in df_sub.groupby(column):
            if len(group["Price"]) >= 3:
                stat, p = shapiro(group["Price"])
                normality_results[variable] = p
            else:
                normality_results[variable] = np.nan

        # Determine if normality holds (at least 80% of groups normal)
        normal_groups = [p for p in normality_results.values() if not np.isnan(p)]
        normality_ok = np.mean(np.array(normal_groups) > 0.05) > 0.8
        print(f"✅ Normality check passed for {np.mean(np.array(normal_groups) > 0.05)*100:.1f}% of {column}")

        # === Step 2. Homogeneity of variances ===
        groups = [group["Price"].values for _, group in df_sub.groupby(column) if len(group) > 1]
        stat_lev, p_lev = levene(*groups)
        equal_var = p_lev > 0.05
        print(f"✅ Levene’s test p-value: {p_lev:.4f} → {'Equal variances' if equal_var else 'Unequal variances'}")

        # === Step 3. Choose the right test ===
        F_stat, p_val, eta2 = np.nan, np.nan, np.nan

        try:
            if normality_ok:
                if equal_var:
                    # Standard one-way ANOVA
                    model = ols(f'Price ~ C({column})', data=df_sub).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    F_stat = anova_table["F"][0]
                    p_val = anova_table["PR(>F)"][0]
                    # Eta squared = SS_between / SS_total
                    eta2 = anova_table["sum_sq"][0] / anova_table["sum_sq"].sum()
                else:
                    # Welch’s ANOVA
                    result = anova_oneway(df_sub["Price"], df_sub[column], use_var='unequal')
                    F_stat = result.statistic
                    p_val = result.pvalue
                    eta2 = np.nan  # Welch ANOVA doesn't easily give eta² directly
            else:
                # Kruskal–Wallis test
                stat_kw, p_kw = kruskal(*groups)
                F_stat, p_val = stat_kw, p_kw
                k = len(groups)
                N = sum(len(g) for g in groups)
                eta2 = (stat_kw - k + 1) / (N - k)
        except Exception as e:
            print(f"❌ Error in {column}: {e}")

        results.append([column, num_unique, num_valid, F_stat, p_val, eta2])

    # === Combine into DataFrame ===
    results_df = pd.DataFrame(results, columns=[
        "Feature", "Unique Values", "Groups ≥3", "F-statistic", "p-value", "Eta²"
    ])

    return results_df


at_cols = ['Brand', 'Model', 'Car/Suv', 'UsedOrNew', 'Transmission',
            'DriveType', 'FuelType', 'ColourExtInt', 'Location', 'BodyType']

summary_table = compare_test_summary(df, cat_cols)
#print(summary_table)

# 

def interpret_eta2(eta):
    if pd.isna(eta):
        return np.nan
    elif eta < 0.01:
        return "Negligible"
    elif eta < 0.06:
        return "Small"
    elif eta < 0.14:
        return "Medium"
    else:
        return "Large"

summary_table["Effect Strength"] = summary_table["Eta²"].apply(interpret_eta2)
print(summary_table.sort_values("Eta²", ascending=False))

