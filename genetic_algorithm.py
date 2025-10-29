import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import random
from deap import base, creator, tools
import matplotlib.pyplot as plt
import seaborn as sns
import argparse   # for reading arguments in a regular way


"""
Read Inputs
"""

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("--n_generations", type=int, help="Number of the exploring generations", default= 5)
parser.add_argument("--pop_size", type=int, help="Population Size", default= 10)
args = parser.parse_args()

args_dict = vars(args)
print(args_dict)

df = pd.read_csv(args_dict["data"])



"""
PREPROCESS: choose candidate columns & reducing diversity
"""

col_len = len(df.columns)
X = df.iloc[:, :col_len-1]
y = df.iloc[:, -1] / 1000

HIGH_CARDINALITY_THRESHOLD = 3  # still optional if you want to skip some columns

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical Columns:\n {categorical_cols}")
numeric_cols = X.select_dtypes(include=['number', 'bool']).columns.tolist()
print(f"Numerical Columns:\n {numeric_cols}")

# Replace categories with counts <= 2 as '__OTHER__'
def reduce_rare_categories(series, min_count=3):
    """Replace categories with frequency less than min_count with '__OTHER__'."""
    value_counts = series.value_counts()
    rare_categories = value_counts[value_counts < min_count].index
    return series.where(~series.isin(rare_categories), other='__OTHER__')

X_reduced = X.copy()
for c in categorical_cols:
    # Optional: only apply if the column has enough unique categories
    if X_reduced[c].nunique() > HIGH_CARDINALITY_THRESHOLD:
        X_reduced[c] = reduce_rare_categories(X_reduced[c], min_count=3)

# recompute categorical list (unchanged except rarified)
categorical_cols = X_reduced.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X_reduced.select_dtypes(include=['number', 'bool']).columns.tolist()

# Candidate feature names (we'll treat each original column as a single "feature")
feature_names = list(X_reduced.columns)
n_features = len(feature_names)
print(f"Using {n_features} candidate features: {feature_names}")


print("\nOriginal Data Frame; features frequencies")
for c in X:
    numbers = X[c].unique()
    print(f"Numbers for {c} : {len(numbers)}")

print("\nReduced Data Frame; features frequencies")
for c in X_reduced:
    numbers = X_reduced[c].unique()
    print(f"Numbers for {c} : {len(numbers)}")
    

"""
Build a sklearn preprocessing transformer that we will use inside fitness
"""

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='__NA__')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor_all = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])


"""
Preprocess on selected columns & Define model
"""
def make_pipeline_for_columns(selected_columns):
    # Identify which selected columns are numeric/categorical
    sel_num = [c for c in selected_columns if c in numeric_cols]
    sel_cat = [c for c in selected_columns if c in categorical_cols]
    
    # Build a column transformer using only the chosen features
    transformers = []
    if sel_num:
        transformers.append(('num', numeric_transformer, sel_num))
    if sel_cat:
        transformers.append(('cat', categorical_transformer, sel_cat))

    # ColumnTransformer 
    # for num -> numeric_transformer (StandardScaler)
    # for cat -> numeric_transformer (OneHotEncoding)
    ct = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # Define a simple model (e.g., Random Forest)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    model = RandomForestRegressor(n_estimators=20, max_depth=6, random_state=42, n_jobs=-1)
    #model = LinearRegression()
    
    # Combine preprocessing (on ct) + model (LR, RF,..) into one pipeline
    pipeline = Pipeline(steps=[('pre', ct), ('model', model)])
    return pipeline




"""
Evaluate every individual == Fintness Function
"""

def evaluate_individual(individual, X, y):
    # Define the selected features first
    selected = [col for col, bit in zip(X.columns, individual) if bit == 1]

    # Handle the case where no features are selected
    if not selected:
        return (-999999999,)

    try:
        # Build the preprocessing + model pipeline for the chosen columns
        pipe = make_pipeline_for_columns(selected)

        # Evaluate using cross-validation (negative MSE)
        scores = cross_val_score(pipe, X[selected], y, cv=5, scoring='neg_mean_squared_error')

        # Return the *average* score across folds
        mean_score = np.mean(scores)

        # Debug info
        print(f"Selected: {selected} | Fitness (neg MSE): {mean_score:.4f}")

        # Important: return a single-element tuple
        return (mean_score,)

    except Exception as e:
        # General debug print
        print(f"Error during evaluation for {selected if 'selected' in locals() else 'unknown'}: {e}")
        return (-999999999,)


"""
Define the optimaization goal
"""

# We want to maximize fitness (since we use negative MSE, higher = better)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


"""
Register GA components
"""

toolbox = base.Toolbox()

# Assume you have n_features defined
n_features = len(X_reduced.columns)

# Individual: binary list of 0/1 for each feature
# Each gene → random 0 or 1
toolbox.register("attr_bool", random.randint, 0, 1)
# Each individual → list of n_features genes
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
# The initial population → list of many individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


"""
Register the fitness function and genetic operators
"""

# Fitness evaluation function (defined earlier)
toolbox.register("evaluate", evaluate_individual, X=X_reduced, y=y)

# Crossover: two-point crossover (swap feature segments)
toolbox.register("mate", tools.cxTwoPoint)

# Mutation: flip bits with probability indpb
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# Selection: tournament (pick best among random subsets)
#toolbox.register("select", tools.selTournament, tournsize=3)   # stochastic selection (tournament selection).
toolbox.register("select", tools.selBest)                       # deterministic (pure elitist) selection.


"""
Run the GA loop
"""

def run_ga(n_generations, pop_size):
    random.seed(42)
    
    best_per_gen = []     # track best fitness per generation
    best_subsets = []     # track best subset (selected features) per generation

    # Initialize population
    pop = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Evolution loop
    for gen in range(n_generations):
        print(f"\n=== Generation {gen+1} ===")
        
        # Selection and reproduction
        offspring = toolbox.select(pop, 10)
        #offspring = toolbox.selBest(pop, k=3)
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < 0.3:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate new individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        pop[:] = offspring
        
        # Track best of this generation
        best_ind = tools.selBest(pop, 1)[0]
        best_per_gen.append(best_ind.fitness.values[0])
        selected = [col for col, bit in zip(X_reduced.columns, best_ind) if bit == 1]
        best_subsets.append(selected)
        
        print(f"Best fitness this gen: {best_ind.fitness.values[0]:.4f}")

    # Final best individual
    # Final best individual - based on the ALL generation
    # Find index of the generation with the best fitness
    best_index = np.argmax(best_per_gen)
        
    # Get best fitness value and its corresponding subset
    best_fitness = best_per_gen[best_index]
    selected_features = best_subsets[best_index]
    
    print("\n🎯Best overall fitness:", best_fitness)
    print("Best feature subset:", selected_features)

    # --------------------
    # For defining the feature importance
    final_model = make_pipeline_for_columns(selected_features)
    final_model.fit(X_reduced[selected_features], y)

    # Extract feature importances
    rf_model = final_model.named_steps['model']
    feature_names = final_model.named_steps['pre'].get_feature_names_out()
    importances = rf_model.feature_importances_
    # --------------------
    
    return best_ind, selected_features, best_per_gen, best_subsets, final_model, feature_names, importances


"""
Run the GA
"""

best_ind, selected_features, best_per_gen, best_subsets, final_model, feature_names, importances = run_ga(args.n_generations , args.pop_size)


"""
Compare to ALL features
"""
# Score for all featues selected
li = [1] * (col_len-1)
print("\nSCORE for ALL FEATURES - Comparing to the best result")
score_all_selected = evaluate_individual(li, X_reduced, y)


"""
Visualization
"""

# GA Convergence Curve

plt.figure(figsize=(8,4))
plt.plot(best_per_gen, marker='o', linestyle='-', color='b')
plt.title('GA Convergence Curve')
plt.xlabel('Generation')
plt.ylabel('Best Fitness (−MSE, higher = better)')
plt.grid(True)
plt.savefig("result/1_ga_convergence.png", dpi=300, bbox_inches='tight')

# ---------------------------------------------------------------------------------
# Feature Frequency Barplot

# Suppose you stored all best subsets per generation in `best_subsets`
flat = [f for subset in best_subsets for f in subset]
freq = pd.Series(flat).value_counts(normalize=True)

plt.figure(figsize=(8,4))
sns.barplot(x=freq.index, y=freq.values)
plt.title('Feature Selection Frequency Across Generations')
plt.ylabel('Selection Frequency')
plt.xticks(rotation=45)
plt.savefig("result/2_featue_selection_frequency.png", dpi=300, bbox_inches='tight')

# ---------------------------------------------------------------------------------
# Performance Comparison (All vs GA), (MSE, R2)

from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(X, y, selected_features, model=None):
    """Train and evaluate a model on the given feature subset."""
    if model is None:
        model = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=42)

    # Build preprocessing + model pipeline for the chosen columns
    pipe = make_pipeline_for_columns(selected_features)
    pipe.fit(X[selected_features], y)
    y_pred = pipe.predict(X[selected_features])

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2

# Best features from GA

# Evaluate both
mse_all, r2_all = evaluate_model(X_reduced, y, X_reduced.columns)
mse_best, r2_best = evaluate_model(X_reduced, y, selected_features)

print(f"📊 All Features — MSE: {mse_all:.2f}, R²: {r2_all:.3f}")
print(f"🏆 GA Selected — MSE: {mse_best:.2f}, R²: {r2_best:.3f}")

# Data
models = ['All Features', 'GA Selected']
mse_values = [mse_all, mse_best]
r2_values = [r2_all, r2_best]

# --- MSE Plot ---
plt.figure(figsize=(8,4))
plt.bar(models, mse_values, color=['gray', 'green'])
plt.title('Model Comparison - Mean Squared Error (Lower is Better)')
plt.ylabel('MSE')
plt.savefig("result/3_comparison_performance_mse.png", dpi=300, bbox_inches='tight')

# --- R² Plot ---
plt.figure(figsize=(8,4))
plt.bar(models, r2_values, color=['gray', 'green'])
plt.title('Model Comparison - R² Score (Higher is Better)')
plt.ylabel('R²')
plt.savefig("result/4_comparison_performance_r2.png", dpi=300, bbox_inches='tight')

# ---------------------------------------------------------------------------------
# Comparing
# --- All features ---
pipe_all = make_pipeline_for_columns(X_reduced.columns)
pipe_all.fit(X_reduced, y)
y_pred_all = pipe_all.predict(X_reduced)

# --- GA-selected subset ---
ga_features = selected_features

pipe_best = make_pipeline_for_columns(ga_features)
pipe_best.fit(X_reduced, y)
y_pred_best = pipe_best.predict(X_reduced)

# --- Side-by-side plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# All features
axes[0].scatter(y, y_pred_all, alpha=0.25, color='gray')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_title('All Features Model')
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')

# GA-selected features
axes[1].scatter(y, y_pred_best, alpha=0.25, color='green')
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[1].set_title('GA-Selected Features Model')
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')

plt.suptitle('Predicted vs Actual Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("result/5_comparison_predicted_vs_actual.png", dpi=300, bbox_inches='tight')


# ---------------------------------------------------------------------------------
# Feature Importances (from final model)
# Aggregate feature importance for original features (handling categorical encoding)
def get_original_feature_name(encoded_name):
    # Remove prefix ('num__' or 'cat__')
    name = encoded_name.split('__', 1)[-1]
    # Everything before the first underscore in the value (handles empty/underscores safely)
    parts = name.split('_', 1)
    return parts[0] if parts[0] else name  # fallback if first part is empty

base_feature_names = [get_original_feature_name(f) for f in feature_names]

importance_df = pd.DataFrame({'Feature': base_feature_names, 'Importance': importances})
importance_summary = (
    importance_df.groupby('Feature')['Importance']
    .sum()
    .sort_values(ascending=False)
)

print("\nFeature importance summary (aggregated by original feature):")
print(importance_summary)
    

# Plot
plt.figure(figsize=(8, 5))
importance_summary.plot(kind='barh', title="Feature Importance (from Final Model)")
plt.gca().invert_yaxis()
plt.xlabel("Total Importance")
plt.tight_layout()
plt.savefig("result/6_feature_importance.png", dpi=300, bbox_inches='tight')