# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Detecting Invalid Text Responses
# %% [markdown]
# This script requires installation of the following:
# * spaCy
# * pandas
# * numpy
# * seaborn
# * sklearn
# * imblearn
# * statsmodels
# * matplotlib

# %%
import spacy 
nlp = spacy.load('en_core_web_sm')

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from numpy.random import seed
seed(317)
from collections import namedtuple
from datetime import datetime

import seaborn as sns 
sns.set_style('whitegrid')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, classification_report, make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN
from imblearn.pipeline import make_pipeline

import statsmodels.api as stmod
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# %% [markdown]
# # Preparing Text Data for Analysis 
# 
# 

# %%
### import data
raw_data = pd.read_csv('data/text_data.csv')
print('Imported {} texts.'.format(len(raw_data['text'])))

### remove rows with no text before pre-processing 
text_data = raw_data.dropna(subset=['text'])
print('Dropped {} cases without any text before pre-processing.'.format(len(raw_data['text'])-len(text_data['text'])))
print('{} texts remaining before pre-processing.'.format(len(text_data['text'])))
preprocessed_texts_noblanks = len(text_data['text']) # store number of texts after dropping blanks before pre-processing


# %%
### tokenize raw texts using nlp()
text_data['tokenized_text'] = [nlp(string) for string in text_data['text']]
text_data.sample(10, random_state=317)


# %%
### process tokenized text by removing stop words and punctuation, then lemmatizing tokens
def process_text(string):
    processed = []
    for token in string:
        if token.is_stop == False and token.is_punct == False and token.is_alpha == True:
            processed.append(token.lemma_.lower())
    return processed

text_data['processed_tokens'] = [process_text(tokens) for tokens in text_data['tokenized_text']]
print('Processed {} texts. Here are the first five: \n'.format(len(text_data['processed_tokens'])))
for each in text_data['processed_tokens'][:5]:
    print(each)


# %%
### join processed tokens into strings
text_data['processed_text'] = [' '.join(processed_tokens) for processed_tokens in text_data['processed_tokens']]
text_data.sample(5, random_state=317)


# %%
### remove rows with no text after pre-processing
text_data['processed_text'].replace('', np.nan, inplace=True) # replace empty cells with NaN
text_data = text_data.dropna(subset=['processed_text']) # remove rows with NaN values
print('Dropped {} cases without any text after pre-processing.'.format(preprocessed_texts_noblanks-len(text_data['processed_text'])))
print('{} texts remaining after pre-processing.'.format(len(text_data['processed_text'])))


# %%
### filter labelled and unlabelled subsets based on whether cases have received labels (either 0 or 1; labelled) or have no labels (blanks; unlabelled)
labelled_data = text_data[(text_data['human_labelled'] == 0) | (text_data['human_labelled'] == 1)]
unlabelled_data = text_data[text_data['human_labelled'].isnull()]


# %%
### view a random sample of cases from the labelled subset
labelled_data.sample(5, random_state=317)


# %%
### count number of each class (0 = valid, 1 = invalid) in the labelled subset
print('Number of labelled texts:', labelled_data['doc_id'].count())
print(labelled_data['human_labelled'].value_counts())


# %%
### view a random sample of cases from the unlabelled subset
unlabelled_data.sample(5, random_state=317)


# %%
### count number cases in the unlabelled subset
print('Number of unlabelled texts:', unlabelled_data['doc_id'].count())

# %% [markdown]
# ## Split into Training, Validation, and Test Data
# 
# Partition annotated subset into further subsets: one for training the model, one for testing the resulting model's performance.
# 

# %%
vect = CountVectorizer()
t_vect = TfidfVectorizer()


# %%
### select columns from the labelled data, where X is the predictor variable (the processed texts) and y is the outcome variable (whether it is valid or invalid)
X = labelled_data['processed_text']
y = labelled_data['human_labelled']


# %%
### in the labelled data, split off 20% of the cases as the test set (held out for final evaluation)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=317)
print('Train + Validation Set:\n',y_train_val.value_counts())
print('Test Set:\n',y_test.value_counts())


# %%
### set k = 10 for stratified K fold cross-validation
kfolds = 10
skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=317)


# %%
### count the number of cases (and their classes) in each of the stratified k-fold splits (k = 10) of train and validation sets
fold_no = 1
train_size_list = []
val_size_list = []

for train_index, val_index in skf.split(X_train_val, y_train_val):
	### select rows
	X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
	y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
	### summarize train and validation sets
	train_0, train_1 = len(y_train[y_train==0]), len(y_train[y_train==1])
	val_0, val_1 = len(y_val[y_val==0]), len(y_val[y_val==1])
	train_percent = (train_1/(train_1+train_0))*100
	train_size_list.append(len(y_train))
	val_percent = (val_1/(val_1+val_0))*100
	val_size_list.append(len(y_val))
	print('Fold',str(fold_no),'Train: 0=%d, 1=%d, %.1f%%, total=%d // Validation: 0=%d, 1=%d, %.1f%%, total=%d' % (train_0, train_1, train_percent, len(y_train), val_0, val_1, val_percent, len(y_val)))
	fold_no += 1


# %%
print('Train + Validation Set Size: %d texts.' % (len(y_train_val)))
print('Mean Train Set Size Across Folds: %.1f texts.' % (np.mean(train_size_list)))
print('Mean Validation Set Size Across Folds: %.1f texts.' % (np.mean(val_size_list)))
print('Test Set Size: %d texts.' % (len(y_test)))
print('Unlabelled Set Size: %d texts.' % (len(unlabelled_data)))

# %% [markdown]
# # Helper Functions - Model Validation

# %%
def train_val_model_labelled(classifier, resampler=None, count=False, print_precision=False, print_recall=False):
    """Train and cross-validate a model, as specified by the arguments provided.

    Args:
        classifier (str): Name of classifier to be used in model.
        resampler (str, optional): Name of resampler to be used in model. Defaults to None.
        count (bool, optional): Set whether the vectorizer uses count (True) or TF-IDF (False). Defaults to False.
        print_precision (bool, optional): Set whether the output prints precision metrics. Defaults to False.
        print_recall (bool, optional): Set whether the output prints recall metrics. Defaults to False.

    Returns:
        cv_results (dict): Training and cross-validation metrics for the specified model.
    """
    if count == True:
        vectorizer = vect
    else:
        vectorizer = t_vect
        
    print('Classifier:', classifier, '/', 'Resampler:', resampler, '/', 'Vectorizer:', vectorizer)

    model = make_pipeline(vectorizer, resampler, classifier)

    cv_results = cross_validate(
        model, X_train_val, y_train_val,
        scoring={'f1': 'f1_macro',
                'mcc': make_scorer(matthews_corrcoef),
                'precision': 'precision_macro',
                'recall': 'recall_macro'},
        return_train_score=True, return_estimator=True,
        n_jobs=-1, cv=skf
    )

    print(f"Mean macro F1: "
        f"{cv_results['test_f1'].mean():.3f} / SD: {cv_results['test_f1'].std():.3f}")
    print(f"Macro F1 by fold: {cv_results['test_f1']}")

    print(f"Mean MCC: "
        f"{cv_results['test_mcc'].mean():.3f} / SD: {cv_results['test_mcc'].std():.3f}")
    print(f"MCC by fold: {cv_results['test_mcc']}")

    if print_precision == True:
        print(f"Mean macro precision: "
            f"{cv_results['test_precision'].mean():.3f} / SD: {cv_results['test_precision'].std():.3f}")
        print(f"Macro precision by fold: {cv_results['test_precision']}")
    else:
        pass
        
    if print_recall == True:
        print(f"Mean macro recall: "
            f"{cv_results['test_recall'].mean():.3f} / SD: {cv_results['test_recall'].std():.3f}")
        print(f"Macro recall by fold: {cv_results['test_recall']}")
    else:
        pass

    return cv_results

# %% [markdown]
# # SMOTE
# Synthetic Minority Oversampling Technique (SMOTE; Chawla et al., 2002) is a resampling method, which mitigates the effects of class imbalance in our data (~7.5% were flagged as invalid in the labelled subset). SMOTE upsamples the minority class with synthetic, plausible cases. "The minority class is over-sampled by taking each minority class sample and introducing synthetic examples along the line segments joining any/all of the *k* minority class nearest neighbors" (Chawla et al., 2002, p. 328).

# %%
sm = SMOTE(k_neighbors=5, random_state=317)

# %% [markdown]
# ## SMOTE - Naive Bayes

# %%
nb = MultinomialNB()


# %%
nb_sm = train_val_model_labelled(nb, sm)

# %% [markdown]
# ## SMOTE - Logistic Regression

# %%
logreg = LogisticRegression(random_state=317)


# %%
logreg_sm = train_val_model_labelled(logreg, sm)

# %% [markdown]
# ## SMOTE - Decision Tree

# %%
dt = DecisionTreeClassifier(random_state=317)


# %%
dt_sm = train_val_model_labelled(dt, sm)

# %% [markdown]
# ## SMOTE - Random Forest

# %%
rf = RandomForestClassifier(random_state=317)


# %%
rf_sm = train_val_model_labelled(rf, sm)

# %% [markdown]
# ## SMOTE - Gradient Boost

# %%
gb = GradientBoostingClassifier(random_state=317)


# %%
gb_sm = train_val_model_labelled(gb, sm)

# %% [markdown]
# ## SMOTE - LinearSVC

# %%
lsvc = LinearSVC(random_state=317)


# %%
lsvc_sm = train_val_model_labelled(lsvc, sm)

# %% [markdown]
# # SVM-SMOTE
# %% [markdown]
# Support Vector Machine-SMOTE (SVM-SMOTE) is an extension of SMOTE that "focuses only on the minority class instances lying around the borderline due to the fact that this area is most crucial for establishing the decision boundary" (Nguyen et al., 2011, p. 24). After oversampling at the borderline between the minority class and the majority class, Support Vector Machine (SVM) classifier then is trained to predict new unknown instances.

# %%
svmsm = SVMSMOTE(k_neighbors=5, random_state=317)

# %% [markdown]
# ## SVMSMOTE - Naive Bayes

# %%
nb_svmsm = train_val_model_labelled(nb, svmsm, count=True) # using count here because TF-IDF results in values that are incompatible with NB classifier using our dataset

# %% [markdown]
# ## SVMSMOTE - Logistic Regression

# %%
logreg_svmsm = train_val_model_labelled(logreg, svmsm)

# %% [markdown]
# ## SVMSMOTE - Decision Tree

# %%
dt_svmsm = train_val_model_labelled(dt, svmsm)

# %% [markdown]
# ## SVMSMOTE - Random Forest

# %%
rf_svmsm = train_val_model_labelled(rf, svmsm)

# %% [markdown]
# ## SVMSMOTE - Gradient Boost

# %%
gb_svmsm = train_val_model_labelled(gb, svmsm)

# %% [markdown]
# ## SVMSMOTE - LinearSVC

# %%
lsvc_svmsm = train_val_model_labelled(lsvc, svmsm)

# %% [markdown]
# # ADASYN
# Adaptive synthetic sampling (ADASYN; He et al., 2008) "is based on the idea of adaptively generating minority data samples according to their distributions: more synthetic data is generated for minority class samples that are harder to learn compared to those minority samples that are easier to learn" (p. 1323). 

# %%
ad = ADASYN(n_neighbors=5, random_state=317)

# %% [markdown]
# ## ADASYN - Naive Bayes

# %%
nb_ad = train_val_model_labelled(nb, ad)

# %% [markdown]
# ## ADASYN - Logistic Regression

# %%
logreg_ad = train_val_model_labelled(logreg, ad)

# %% [markdown]
# ## ADASYN - Decision Tree

# %%
dt_ad = train_val_model_labelled(dt, ad)

# %% [markdown]
# ## ADASYN - Random Forest

# %%
rf_ad = train_val_model_labelled(rf, ad)

# %% [markdown]
# ## ADASYN - Gradient Boost

# %%
gb_ad = train_val_model_labelled(gb, ad)

# %% [markdown]
# ## ADASYN - LinearSVC

# %%
lsvc_ad = train_val_model_labelled(lsvc, ad)

# %% [markdown]
# # Model Validation

# %%
### store names of classifiers and resamplers in lists
classifier_names = ['nb', 'logreg', 'dt', 'rf', 'gb', 'lsvc']
resampler_names = ['sm', 'svmsm', 'ad']


# %%
### create lists of macro F1 and MCC scores for each classifier by resampler (SMOTE)
sm_f1 = [nb_sm['test_f1'], logreg_sm['test_f1'], dt_sm['test_f1'], rf_sm['test_f1'], gb_sm['test_f1'], lsvc_sm['test_f1']]
sm_mcc = [nb_sm['test_mcc'], logreg_sm['test_mcc'], dt_sm['test_mcc'], rf_sm['test_mcc'], gb_sm['test_mcc'], lsvc_sm['test_mcc']]


# %%
### create lists of macro F1 and MCC scores for each classifier by resampler (SVM-SMOTE)
svmsm_f1 = [nb_svmsm['test_f1'], logreg_svmsm['test_f1'], dt_svmsm['test_f1'], rf_svmsm['test_f1'], gb_svmsm['test_f1'], lsvc_svmsm['test_f1']]
svmsm_mcc = [nb_svmsm['test_mcc'], logreg_svmsm['test_mcc'], dt_svmsm['test_mcc'], rf_svmsm['test_mcc'], gb_svmsm['test_mcc'], lsvc_svmsm['test_mcc']]


# %%
### create lists of macro F1 and MCC scores for each classifier by resampler (ADASYN)
ad_f1 = [nb_ad['test_f1'], logreg_ad['test_f1'], dt_ad['test_f1'], rf_ad['test_f1'], gb_ad['test_f1'], lsvc_ad['test_f1']]
ad_mcc = [nb_ad['test_mcc'], logreg_ad['test_mcc'], dt_ad['test_mcc'], rf_ad['test_mcc'], gb_ad['test_mcc'], lsvc_ad['test_mcc']]

# %% [markdown]
# ## Model Validation - Macro F1

# %%
f1_by_resampler = [sm_f1, svmsm_f1, ad_f1] # create list of macro F1 scores across all classifiers and resamplers
dfs = {} # initialize empty dictionary

for each, df_name in zip(f1_by_resampler, resampler_names): # store macro F1 scores by classifier in dictionary
    dfs[df_name] = pd.DataFrame(each, index=classifier_names) # can access dfs as: dfs['sm'], dfs['svmsm'], dfs['ad']


# %%
### combine macro F1 dataframes, grouped by classifier and resampler
df_f1_all = pd.concat(dfs, keys=resampler_names, names=['resampler', 'classifier']).reset_index()
df_f1_all


# %%
### pivot dataframe from wide to long format (all macro F1s in the same column, identified by new fold variable)
df_f1_all_melted = pd.melt(df_f1_all, id_vars=['resampler', 'classifier'], var_name='fold', value_name='macro_f1')
df_f1_all_melted = df_f1_all_melted.sort_values(by=['resampler', 'classifier', 'fold'], ignore_index=True)
df_f1_all_melted


# %%
### get current date and time in string format, to be added to file names of written outputs
date_string = datetime.now().strftime('%Y-%m-%d_%I-%M-%S-%p')


# %%
### write F1 data to CSV in output folder
output_name = 'f1_all_models'
file_name = str('output/' + output_name + '_' + date_string + '.csv')
df_f1_all_melted.to_csv(file_name, index=False)


# %%
### get macro F1 scores, averaged across folds, and grouped by classifier and resampler
grouped_data = df_f1_all_melted.groupby(['resampler', 'classifier'])
grouped_data_agg_mean = grouped_data['macro_f1'].aggregate(np.mean).reset_index().sort_values(by=['macro_f1'], ascending=False)

grouped_data_agg_std = grouped_data['macro_f1'].aggregate(np.std).reset_index().sort_values(by=['macro_f1'], ascending=False).rename(columns={'macro_f1': 'std'})
grouped_data_agg = pd.merge(grouped_data_agg_mean, grouped_data_agg_std, how='outer', on=['resampler', 'classifier'])
grouped_data_agg


# %%
### alternatively, reorder by classifier and resampler
grouped_data_agg.sort_values(by=['classifier','resampler'], ascending=True)


# %%
### visualize macro F1 scores by resampler
plot_f1_by_resampler = sns.boxplot(x='resampler', y='macro_f1', hue='classifier', data=df_f1_all_melted, palette='Set3')
#plot_f1_by_resampler
output_name = 'f1_by_resampler'
file_name = str('output/' + output_name + '_' + date_string + '.png')
plt.savefig(file_name, bbox_inches='tight')
plt.close()


# %%
### visualize macro F1 scores by classifier
plot_f1_by_classifier = sns.boxplot(x='classifier', y='macro_f1', hue='resampler', data=df_f1_all_melted, palette='Set3') 
#plot_f1_by_classifier
output_name = 'f1_by_classifier'
file_name = str('output/' + output_name + '_' + date_string + '.png')
plt.savefig(file_name, bbox_inches='tight')
plt.close()

# %%
### run two-way ANOVA where DV = macro F1 score and IVs = classifier, resampler
anova_model = ols('macro_f1 ~ C(resampler) + C(classifier) + C(resampler):C(classifier)', data=df_f1_all_melted).fit()
stmod.stats.anova_lm(anova_model, typ=3)


# %%
### run post hoc comparisons by resampler if appropriate
anova_posthoc = pairwise_tukeyhsd(df_f1_all_melted['macro_f1'], df_f1_all_melted['resampler'])
print(anova_posthoc)


# %%
### run post hoc comparisons by classifier if appropriate
anova_posthoc = pairwise_tukeyhsd(df_f1_all_melted['macro_f1'], df_f1_all_melted['classifier'])
print(anova_posthoc)

# %% [markdown]
# ## Model Validation - MCC

# %%
mcc_by_resampler = [sm_mcc, svmsm_mcc, ad_mcc] # create list of MCC scores across all classifiers and resamplers
dfs = {} # initialize empty dictionary

for each, df_name in zip(mcc_by_resampler, resampler_names): # store MCC scores by classifier in dictionary
    dfs[df_name] = pd.DataFrame(each, index=classifier_names) # can access dfs as: dfs['sm'], dfs['svmsm'], dfs['ad']


# %%
### combine MCC dataframes, grouped by classifier and resampler
df_mcc_all = pd.concat(dfs, keys=resampler_names, names=['resampler', 'classifier']).reset_index()
df_mcc_all


# %%
### pivot dataframe from wide to long format (all MCCs in the same column, identified by new fold variable)
df_mcc_all_melted = pd.melt(df_mcc_all, id_vars=['resampler', 'classifier'], var_name='fold', value_name='mcc')
df_mcc_all_melted = df_mcc_all_melted.sort_values(by=['resampler', 'classifier', 'fold'], ignore_index=True)
df_mcc_all_melted


# %%
### write MCC data to CSV in output folder
output_name = 'mcc_all_models'
file_name = str('output/' + output_name + '_' + date_string + '.csv')
df_mcc_all_melted.to_csv(file_name, index=False) 


# %%
### get MCC scores, averaged across folds, and grouped by classifier and resampler
grouped_data = df_mcc_all_melted.groupby(['resampler', 'classifier'])
grouped_data_agg_mean = grouped_data['mcc'].aggregate(np.mean).reset_index().sort_values(by=['mcc'], ascending=False)

grouped_data_agg_std = grouped_data['mcc'].aggregate(np.std).reset_index().sort_values(by=['mcc'], ascending=False).rename(columns={"mcc": "std"})
grouped_data_agg = pd.merge(grouped_data_agg_mean, grouped_data_agg_std, how='outer', on=['resampler', 'classifier'])
grouped_data_agg


# %%
### alternatively, reorder by classifier and resampler
grouped_data_agg.sort_values(by=['classifier','resampler'], ascending=True)


# %%
### visualize MCC scores by resampler
plot_mcc_by_resampler = sns.boxplot(x='resampler', y='mcc', hue='classifier', data=df_mcc_all_melted, palette='Set3')
#plot_mcc_by_resampler
output_name = 'mcc_by_resampler'
file_name = str('output/' + output_name + '_' + date_string + '.png')
plt.savefig(file_name, bbox_inches='tight')
plt.close()

# %%
### visualize MCC scores by classifier
plot_mcc_by_classifier = sns.boxplot(x='classifier', y='mcc', hue='resampler', data=df_mcc_all_melted, palette='Set3')
#plot_mcc_by_classifier
output_name = 'mcc_by_classifier'
file_name = str('output/' + output_name + '_' + date_string + '.png')
plt.savefig(file_name, bbox_inches='tight')
plt.close()

# %%
### run two-way ANOVA where DV = MCC score and IVs = classifier, resampler
anova_model = ols('mcc ~ C(resampler) + C(classifier) + C(resampler):C(classifier)', data=df_mcc_all_melted).fit()
stmod.stats.anova_lm(anova_model, typ=3)


# %%
### run post hoc comparisons by resampler if appropriate
anova_posthoc = pairwise_tukeyhsd(df_mcc_all_melted['mcc'], df_mcc_all_melted['resampler'])
print(anova_posthoc)


# %%
### run post hoc comparisons by classifier if appropriate
anova_posthoc = pairwise_tukeyhsd(df_mcc_all_melted['mcc'], df_mcc_all_melted['classifier'])
print(anova_posthoc)

# %% [markdown]
# # Final Evaluation on Test Set

# %%
def eval_test_model_labelled(cv_results):
    """ Evaluate performance of an already trained and cross-validated model on the unseen test set.
    
    Args:
        cv_results (string): Name of model that has been already trained and cross-validated.
    
    Returns:
        f1_scores (list): List of F1 scores by fold.
        mcc_scores (list): List of MCC scores by fold.
    """
    
    f1_scores = []
    mcc_scores = []
    for fold_id, cv_model in enumerate(cv_results["estimator"]):
        f1_scores.append(
            f1_score(y_test, cv_model.predict(X_test), average = 'macro')
        )
        mcc_scores.append(
            matthews_corrcoef(y_test, cv_model.predict(X_test))
        )

    print(f"Mean macro F1: "
        f"{np.mean(f1_scores):.3f} / SD: {np.std(f1_scores):.3f}")
    print(f"Macro F1 by fold: {f1_scores}")

    print(f"Mean MCC: "
        f"{np.mean(mcc_scores):.3f} / SD: {np.std(mcc_scores):.3f}")
    print(f"MCC by fold: {mcc_scores}")

    return f1_scores, mcc_scores


# %%
### apply already trained model from cross-validation to the unseen test set
### if performance metrics are similar here as they were during cross-validation above, provides some evidence that the model can successfully generalize to unseen data
eval_test_model = eval_test_model_labelled(nb_svmsm)

# %% [markdown]
# # Final Predictions on Test Set

# %%
### create dataframe with test set data and document IDs
X_test.rename('doc_id').reset_index() # add index as column, rename that column to 'doc_id'
X_test_df = pd.DataFrame(X_test).reset_index() # convert it to a dataframe
X_test_df.columns = ['doc_id', 'processed_text'] # name that dataframe's columns
X_test_df['doc_id'] += 1 # doc_id is zero-indexed, so add one to match with original dataset


# %%
def pred_test_model_labelled(classifier, resampler=None, count=False):
    """ Re-train a model on combined train and validation sets, and then make predictions on the unseen test set.
    
    Args:
        classifier (str): Name of classifier to be used in model.
        resampler (str, optional): Name of resampler to be used in model. Defaults to None.
        count (bool, optional): Set whether the vectorizer uses count (True) or TF-IDF (False). Defaults to False.
    
    Returns:
        test_metrics (tuple): Named tuple 'metrics', with fields 'f1_list', 'precision_list', 'recall_list', and 'accuracy_list'
            containing classification metrics by each fold.
        y_pred (array): Model's predictions made on unseen test set.
    """
    if count == True:
        vectorizer = vect
    else:
        vectorizer = t_vect
        
    print('Classifier:', classifier, '/', 'Resampler:', resampler, '/', 'Vectorizer:', vectorizer)

    model = make_pipeline(vectorizer, resampler, classifier)

    model.fit(X_train_val, y_train_val)
    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    print(f'> Macro F1: {macro_f1:.3f}')
    print(f'> MCC: {mcc:.3f}')
    print(f'> Macro Precision: {macro_precision:.3f}')
    print(f'> Macro Recall: {macro_recall:.3f}')
    print(classification_report(y_test, y_pred))
    print(pd.crosstab(y_test, y_pred, rownames=['human_labelled'], colnames=['model_predicted'], margins=True))
    
    test_metrics = namedtuple('test_metrics', ['macro_f1', 'mcc', 'macro_precision', 'macro_recall'])
    return test_metrics(macro_f1, mcc, macro_precision, macro_recall), y_pred


# %%
### re-train model on full train and validation set, then make predictions on test set of labelled data
### arguments supplied here (classifier, resampler) should be chosen based on your cross-validation and final evaluation results above
pred_test_model = pred_test_model_labelled(nb, svmsm, count=True)


# %%
### add model's predictions as new column to dataframe of test set with document IDs and processed texts
X_test_df['model_predicted'] = pred_test_model[1]
X_test_df


# %%
### merge in original data linked to each document (e.g., human labels, raw texts)
test_df = pd.merge(X_test_df, labelled_data, how='outer', on=['doc_id', 'processed_text'])
test_df


# %%
### get a random sample of cases in the test set based on whether the model predicted them to be valid (0) or invalid (1)
test_df.groupby('model_predicted').apply(lambda x: x.sample(n=10, random_state=317))


# %%
### write data for labelled subset (including predictions on test set) to CSV in output folder
output_name = 'test_model_predictions'
file_name = str('output/' + output_name + '_' + date_string + '.csv')
test_df.to_csv(file_name, index=False) 

# %% [markdown]
# # Predict on Unlabelled Data

# %%
### store data from labelled subset
X_labelled = labelled_data['processed_text']
y_labelled = labelled_data['human_labelled']

### store data from unlabelled subset
X_unlabelled = unlabelled_data['processed_text']
unlabelled_doc_id = unlabelled_data['doc_id']


# %%
def pred_model_unlabelled(classifier, resampler=None, count=False):
    """ Re-train a model on the labelled subset (i.e., combined train, validation, and test sets), and then make predictions on the unlabelled subset.
    
    Args:
        classifier (str): Name of classifier to be used in model.
        resampler (str, optional): Name of resampler to be used in model. Defaults to None.
        count (bool, optional): Set whether the vectorizer uses count (True) or TF-IDF (False). Defaults to False.
    
    Returns:
        y_pred_df (dataframe): Dataframe of model's predictions on unlabelled subset and document IDs.
    """
    if count == True:
        vectorizer = vect
    else:
        vectorizer = t_vect

    print('Classifier:', classifier, '/', 'Resampler:', resampler, '/', 'Vectorizer:', vectorizer)

    model = make_pipeline(vectorizer, resampler, classifier)

    model.fit(X_labelled, y_labelled)
    y_pred = model.predict(X_unlabelled)

    y_pred_df = pd.DataFrame(y_pred, unlabelled_doc_id, columns=['model_predicted'])
    print(y_pred_df)
    return y_pred_df


# %%
### re-train model on full labelled subset (train, validation, and test set), then make predictions on unlabelled data
### arguments supplied here (classifier, resampler) should be chosen based on cross-validation and final evaluation results above
unlabelled_model = pred_model_unlabelled(nb, svmsm, count=True)


# %%
### count how many valid (0) and invalid (1) cases were predicted to be in the unlabelled subset by the model
print(unlabelled_model.value_counts())
print("Predicted invalid cases: " f"{(len(unlabelled_model[(unlabelled_model['model_predicted'] == 1)]) / len(unlabelled_model))*100:.2f}%")


# %%
### merge in original data linked to each document (e.g., processed texts, raw texts)
unlabelled_predictions_df = pd.merge(unlabelled_data, unlabelled_model, how='outer', on=['doc_id'])
unlabelled_predictions_df


# %%
### get a random sample of cases in the test set based on whether the model predicted them to be valid (0) or invalid (1)
unlabelled_predictions_df.groupby('model_predicted').apply(lambda x: x.sample(n=10, random_state=317))


# %%
### write data for unlabelled subset (including predictions on unlabelled subset) to CSV in output folder
output_name = 'unlabelled_model_predictions'
file_name = str('output/' + output_name + '_' + date_string + '.csv')
unlabelled_predictions_df.to_csv(file_name, index=False)


