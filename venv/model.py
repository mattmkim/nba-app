import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
import statsmodels.api as sm
from sklearn.utils.extmath import cartesian
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

stats = pd.read_csv("Seasons_Stats.csv")
all_nba = pd.read_csv("All.NBA.1984-2018.csv")

# Create new column Year

all_nba = all_nba.iloc[1:]
all_nba["Unnamed: 2"] = all_nba["Unnamed: 2"].str[0:4]
all_nba["Year"] = all_nba["Unnamed: 2"].astype(int) + 1

# Filter both dataframes to only contain data from 1998 onwards

modern_stats = stats[stats["Year"] >= 1998.0]
modern_allnba = all_nba[all_nba["Year"] >= 1998]

# Remove players with NaN values in categories except 'blank' and 'blank2'
# Remove columns with all NaN values, then remove rows with NaN values

modern_stats = modern_stats.dropna(axis=1, how="all")

# Find location where players are double counted due to trades

idx_stats = modern_stats.index[modern_stats["Tm"] == "TOT"]
idx_allnba = modern_allnba.index[modern_allnba["Unnamed: 4"] == "TOT"]

# Remove instances in 'modern_stats' dataframe with "TOT" as team

modern_stats = modern_stats[modern_stats.Tm != "TOT"]

# Change "TOT" in 'modern_allnba' dataframe to team that player played majority of games with

modern_allnba.at[315, "Unnamed: 4"] = "DEN"
modern_allnba.at[413, "Unnamed: 4"] = "ATL"

# Need to get per game stats for each player in 'modern_stats'
# Divide specific categories by number of games a player played in a certain season

columns = ["FG", "FGA", "3P", "3PA", "2P", "2PA", "FT", "FTA", "ORB", "DRB", "TRB", "AST",
           "STL", "BLK", "TOV", "PF", "PTS"]
modern_stats[columns] = modern_stats[columns].div(modern_stats["G"].values, axis=0)

# Remove outliers in the modern_stats dataset - do not want players who have played less than 15 minutes per game
# or less than 10 games

modern_stats = modern_stats[modern_stats.MP > 15]
modern_stats = modern_stats[modern_stats.G > 10]

# Create new column in 'modern_stats' dataframe with values 0 or 1 to indicate whether player was an All NBA player
# Create new column in both datasets that combines name and year to determine which players in 'modern_stats' dataframe
# were All NBA players

modern_allnba["Identifier"] = modern_allnba["Unnamed: 1"] + modern_allnba["Year"].astype(str) \
                              + modern_allnba["Unnamed: 4"].astype(str) \
                              + modern_allnba["Unnamed: 3"].astype(int).astype(str)
modern_stats["Identifier"] = modern_stats["Player"] + modern_stats["Year"].astype(int).astype(str) \
                             + modern_stats["Tm"].astype(str) + modern_stats["Age"].astype(int).astype(str)


def all_nba_label(row):
    if modern_allnba["Identifier"].str.contains(row["Identifier"]).any():
        return 1
    else:
        return 0


modern_stats["AllNBA"] = modern_stats.apply(all_nba_label, axis=1)

# Check if there are only 15 All NBA players for each year

modern_stats["Year"] = modern_stats["Year"].astype(int)

count = modern_stats[modern_stats.AllNBA == 1].groupby("Year").size()
print(count)

# Only 14 players in 2004 had the All NBA identifier -- Metta World Peace was part of the 2003-2004 All NBA team, but
# is not included in the 'modern_stats' dataframe

print(modern_stats[modern_stats["Year"] == 2004][modern_stats["AllNBA"] == 1])
print(modern_allnba[modern_allnba["Year"] == 2004])
print(modern_stats[modern_stats["Player"] == "Metta World Peace"])

modern_stats = modern_stats.fillna(0)

# Use support vector machines to predict All NBA selections; normalize data

training = modern_stats[modern_stats["Year"] < 2012]
test = modern_stats[modern_stats["Year"] >= 2013]

y_train = np.array(training["AllNBA"])
x_train = training.drop(["Unnamed: 0", "Year", "Player", "Pos", "Age", "Tm", "G", "GS", "MP", "Identifier", "AllNBA"], axis=1)

x_test = test.drop(["Unnamed: 0", "Year", "Player", "Pos", "Age", "Tm", "G", "GS", "MP", "Identifier", "AllNBA"], axis=1).values
y_test = test["AllNBA"].values

# use grid search to determine parameters
parameters = [{'kernel': ['linear', 'rbf', 'poly'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
scores = ['precision', 'recall']

for score in scores:
    clf = GridSearchCV(SVC(C=1), parameters, cv=5, scoring=score)
    clf.fit(x_train, y_train)

    print("Best parameters set:")
    print(clf.best_estimator_)
    print("Grid Scores:")
    print(clf.cv_results_)
    print("Classification Report: ")
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))


sv_classifier = SVC(kernel="linear", C=0.01)
sv_classifier.fit(x_train, y_train)

y_predicted = sv_classifier.predict(x_test)

target_names = ["Not All NBA", "All NBA"]

print(accuracy_score(y_test, y_predicted))
print(classification_report(y_test, y_predicted, target_names=target_names))
print(confusion_matrix(y_test, y_predicted))

print(sv_classifier.coef_.ravel())


def plot_feature_weights(svm, names):
    coef = svm.coef_.ravel()[0:42]
    pos = np.argsort(coef)[-18:]
    neg = np.argsort(coef)[:24]
    coefficients = np.hstack([neg, pos])
    plt.figure(figsize=(15, 5))
    colors = ['blue' if c < 0 else 'red' for c in coef[coefficients]]
    plt.bar(np.arange(42), coef[coefficients], color=colors, width=0.6, align='center')
    feature_names = np.array(names)
    plt.title("Feature Weights")
    plt.xticks(range(42), feature_names[coefficients], rotation=60, ha='right', fontsize=9)
    plt.show(block=False)


features = ['PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS',
            'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
            'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

plot_feature_weights(sv_classifier, features)

# some stats had high magnitude of feature weight that were unexpected

stat_list = ['FTA', 'STL%', 'TOV%', '3PA', 'BPM', 'WS']

dummy_list = []
for stat in stat_list:
    stat_values = np.linspace(modern_stats[stat].min(), modern_stats[stat].max(), 20)
    dummy_list.append(stat_values)

predictions = pd.DataFrame(cartesian(dummy_list))
predictions.columns = stat_list

for stat in stat_list:
    logit = sm.Logit(modern_stats['AllNBA'], modern_stats[stat])
    result = logit.fit()
    print(np.exp(result.params))