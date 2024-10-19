#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy.stats import skew

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import seaborn as sns

import random

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


course_data = pd.read_csv('course_data_final.csv')
module_data = pd.read_csv('module_data_final.csv')
all_data_final = pd.read_csv('all_data_final.csv')
pisa_data = pd.read_csv('pisa-scores-by-country-2024.csv')


# In[ ]:


module_data.head()


# In[ ]:


course_data.head()


# In[ ]:


all_data_final.head()


# <hr>

# In[ ]:


# Create a new dataframe with ModuleName, CourseLevel, and average of ModuleTotal
module_avg_total = module_data.groupby(['ModuleName', 'CourseLevel']).agg(
    AvgModuleTotal=('ModuleTotal', 'mean')
).reset_index()

# Add AvgModuleTotal column into module_data dataframe
module_data_2 = pd.merge(module_data, module_avg_total, on=['ModuleName', 'CourseLevel'], how='left')

# Calculate the percentile for each LearnerCode in each Module
module_data_2['Percentile'] = module_data_2.groupby('ModuleName')['ModuleTotal'].rank(pct=True)

# Filter out ModuleCreditEquivalence == 0 
module_data_2 = module_data_2[module_data_2['ModuleCreditEquivalence'] != 0]

# Keep only the required columns
module_data_2 = module_data_2[['ModuleName', 'ModulePass', 'ModuleCode', 'LearnerCode', 'ModuleTotal' , 'AvgModuleTotal', 'Percentile']]

# Add pass rate into module_data
module_pass_rate = module_data_2.groupby(['ModuleCode']).apply(lambda x: (x['ModulePass'] == 1).mean()).reset_index(name='PassRate')

# Merge the pass rate with module_data_2
module_data_2 = pd.merge(module_data_2, module_pass_rate, on='ModuleCode', how='left')

module_data_2


# In[ ]:


# Calculate skewness for each course based on the 'ModuleTotal' scores
# Group the data by 'ModuleCode' and calculate skewness for 'ModuleTotal' in each group
skewness_data = module_data_2.groupby('ModuleCode')['ModuleTotal'].apply(lambda x: skew(x)).reset_index()

skewness_data.columns = ['ModuleCode', 'Skewness']

# Filter the skewness_data for ModuleCode 'P_AES', to check for skewness individually
selected_skewness_data = skewness_data[skewness_data['ModuleCode'] == 'P_AES']
print(selected_skewness_data)

skewness_data.sort_values(by='Skewness', ascending=False)


# In[ ]:


# Merge skewness_data into module_data_2
module_data_3 = pd.merge(module_data_2, skewness_data, on='ModuleCode', how='left')
module_data_3 = module_data_3.groupby(['ModuleName', 'ModuleCode','PassRate','Skewness'])['LearnerCode'].nunique().reset_index()

#Filter out small sample size modules
module_data_3 = module_data_3[module_data_3['LearnerCode'] > 50]
module_data_3


# In[ ]:


# K-means clustering for better grouping
features = module_data_3[['PassRate', 'Skewness', 'LearnerCode']]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0)
module_data_3['RiskGroup'] = kmeans.fit_predict(features_scaled)

clustered_data = module_data_3[['ModuleName', 'ModuleCode', 'PassRate', 'Skewness', 'LearnerCode', 'RiskGroup']]
clustered_data


# In[ ]:


# Set up the figure
plt.figure(figsize=(13, 6))

# Scatter plot with pass rate on the x-axis and skewness on the y-axis, colored by risk group
sns.scatterplot(
    data=clustered_data,
    x='PassRate',
    y='Skewness',
    hue='RiskGroup',
    palette='rocket',
    s=100,
    legend='full'
)

plt.xlabel('Pass Rate')
plt.ylabel('Skewness')
plt.title('Pass Rate vs Skewness by Risk Group')
plt.legend(title='Risk Group', loc='upper right', bbox_to_anchor=(1.07, 1))

plt.grid(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show


# Risk Group= 0: Very high pass rate with the least negative skewness, indicating generally good performance among students. (bottom right)

# Risk Group= 1: Modules with a very high pass rate and the most negative skewness, indicating many students pass, but there is a significant tail of lower scores. This group has a moderate number of students. (top right) 

# Risk Group = 2: Lower pass rate and moderate negative skewness, suggesting these modules may be more challenging for students. (top left)

# Risk Group= 3: High pass rate with moderate negative skewness, indicating a balanced performance distribution but with some lower scores. (rest)

# In[ ]:


# Define the risk score mapping based on the given criteria
def assign_risk_score(row):
    if row['RiskGroup'] == 0:
        return 1
    elif row['RiskGroup'] == 1:
        return 4
    elif row['RiskGroup'] == 2:
        return 8
    elif row['RiskGroup'] == 3:
        return 6
    else:
        return 10  # Default case, should not occur

# Apply the risk score assignment
clustered_data['RiskScore'] = clustered_data.apply(assign_risk_score, axis=1)

clustered_data


# In[ ]:


# Load the course and module data
course_data = pd.read_csv('course_data_final.csv')
module_data = pd.read_csv('module_data_final.csv')

# Merging course_data and module_data, and including the RiskGroup from clustered_data
merged_course_module = pd.merge(course_data, module_data, on=['AcademicYear', 'LearnerCode'], how='inner')
merged_course_module = pd.merge(merged_course_module, clustered_data[['ModuleCode', 'RiskScore']], on='ModuleCode', how='left')

# Selecting the relevant columns
merged_course_module = merged_course_module[['CourseName', 'ModuleName', 'ModuleCode', 'RiskScore']]

merged_course_module.dropna(subset=['RiskScore'], inplace=True)

merged_course_module


# In[ ]:


# Group by CourseName and calculate the average RiskScore
grouped_course_risk = merged_course_module.groupby('CourseName').agg(
    AvgRiskScore=('RiskScore', 'mean')
).reset_index()

grouped_course_risk


# In[ ]:


# Merging all_data_final and course risk scores
all_data_final_2 = pd.merge(all_data_final, grouped_course_risk, on='CourseName', how='left')

all_data_final_2.head()


# In[ ]:


# Mapping for Nationality to Nation
nationality_to_nation = {
    "Chinese": "China",
    "Iraqi": "Iraq",
    "Kenyan": "Kenya",
    "British": "United Kingdom",
    "Kazakhstani": "Kazakhstan",
    "Taiwanese": "Taiwan",
    "Lebanese": "Lebanon",
    "Kuwaiti": "Kuwait",
    "Qatari": "Qatar",
    "Vietnamese": "Vietnam",
    "Egyptian": "Egypt",
    "Emirati": "United Arab Emirates",
    "Hong Kong Chinese": "Hong Kong",
    "Nigerian": "Nigeria",
    "Indian": "India",
    "Myanmarian": "Myanmar",
    "Singaporean": "Singapore",
    "Bahraini": "Bahrain",
    "Japanese": "Japan",
    "Bangladeshi": "Bangladesh",
    "Libyan": "Libya",
    "Jordanian": "Jordan",
    "Saudi": "Saudi Arabia",
    "Angolan": "Angola",
    "Namibian": "Namibia",
    "Zimbabwean": "Zimbabwe",
    "Ghanaian": "Ghana",
    "American": "United States",
    "Omani": "Oman",
    "Senegalese": "Senegal",
    "Brazilian": "Brazil",
    "Ugandan": "Uganda",
    "Turkish": "Turkey",
    "Mexican": "Mexico",
    "Peruvian": "Peru",
    "Mozambican": "Mozambique",
    "Syrian": "Syria",
    "Pakistani": "Pakistan",
    "Mauritanian": "Mauritania",
    "German": "Germany",
    "Afghan": "Afghanistan",
    "Antiguan": "Antigua and Barbuda",
    "Russian": "Russia",
    "Algerian": "Algeria",
    "Guyanese": "Guyana",
    "Malaysian": "Malaysia",
    "Spanish": "Spain",
    "South Sudanese": "South Sudan",
    "Thai": "Thailand",
    "Sri Lankan": "Sri Lanka",
    "Grenadian": "Grenada",
    "Korean": "South Korea",
    "Guinean": "Guinea",
    "Slovak": "Slovakia",
    "Belgian": "Belgium",
    "Sudanese": "Sudan",
    "Indonesian": "Indonesia",
    "Moroccan": "Morocco",
    "Venezuelan": "Venezuela",
    "Tanzanian": "Tanzania",
    "Gabonese": "Gabon",
    "Iranian": "Iran",
    "South African": "South Africa",
    "Albanian": "Albania",
    "Irish": "Ireland",
    "Nepalese": "Nepal",
    "Armenian": "Armenia",
    "Ukrainian": "Ukraine",
    "Eritrean": "Eritrea",
    "Botswanan": "Botswana",
    "Italian": "Italy",
    "Azerbaijani": "Azerbaijan",
    "Uzbek": "Uzbekistan",
    "Dominica (Commonwealth)": "Dominica",
    "Belarusian": "Belarus",
    "Canadian": "Canada",
    "Dominican Republic": "Dominican Republic",
    "Bolivian": "Bolivia",
    "Philippine": "Philippines",
    "Dutch": "Netherlands",
    "Israeli": "Israel",
    "Portuguese": "Portugal",
    "Norwegian": "Norway",
    "Macau Chinese": "Macau",
    "Romanian": "Romania",
    "Zambian": "Zambia",
    "Bruneian": "Brunei",
    "Seychelles": "Seychelles",
    "French": "France",
    "Swiss": "Switzerland",
    "Ecuadorian": "Ecuador",
    "Bahamian": "Bahamas",
    "British National (Overseas)": "United Kingdom",
    "Honduran": "Honduras",
    "Cameroonian": "Cameroon",
    "Argentine": "Argentina",
    "Yemeni": "Yemen",
    "Icelandic": "Iceland",
    "Liberian": "Liberia",
    "Tunisian": "Tunisia",
    "Georgian": "Georgia",
    "Swedish": "Sweden",
    "Guatemalan": "Guatemala",
    "Chilean": "Chile",
    "Mongolian": "Mongolia",
    "Polish": "Poland",
    "Kyrgyzstani": "Kyrgyzstan",
    "Latvian": "Latvia",
    "Colombian": "Colombia",
    "Czech": "Czech Republic",
    "Greek": "Greece",
    "Cypriot": "Cyprus",
    "(Not known)": "Unknown",
    "Paraguayan": "Paraguay",
    "Ethiopian": "Ethiopia",
    "Cambodian": "Cambodia",
    "Palestinian": "Palestine",
    "Rwandan": "Rwanda",
    "Ivorian": "Ivory Coast",
    "Tajikistani": "Tajikistan",
    "Mauritian": "Mauritius",
    "Kittitian": "Saint Kitts and Nevis",
    "Sierra Leonean": "Sierra Leone",
    "Moldovan": "Moldova",
    "Australian": "Australia",
    "Bermudian": "Bermuda",
    "Panamanian": "Panama",
    "Malawian": "Malawi",
    "Equatoguinean": "Equatorial Guinea",
    "Maltese": "Malta",
    "Bulgarian": "Bulgaria",
    "Malian": "Mali",
    "Hungarian": "Hungary",
    "Austrian": "Austria",
    "Swazi": "Eswatini",
    "Croatian": "Croatia",
    "New Zealand": "New Zealand",
    "Uruguayan": "Uruguay",
    "Turkmen": "Turkmenistan",
    "Danish": "Denmark",
    "Burundi": "Burundi",
    "Fijian": "Fiji",
    "Trinidadian": "Trinidad and Tobago",
    "Nigerien": "Niger",
    "Maldivian": "Maldives",
    "Serbian": "Serbia",
    "Nicaraguan": "Nicaragua",
    "Laotian": "Laos",
    "Bosnian": "Bosnia and Herzegovina",
    "Barbadian": "Barbados",
    "Jamaican": "Jamaica"
}

# Apply the mapping
all_data_final_2['Nation'] = all_data_final_2['Nationality'].map(nationality_to_nation)

# Merge with pisa_data on 'Nation' and 'country' columns
all_data_final_2 = pd.merge(all_data_final_2, pisa_data, left_on='Nation', right_on='country', how='left')

# Display the merged data
all_data_final_2.head()


# In[ ]:


# Define the columns of interest, including LearnerCode
columns_of_interest_with_learner = ['LearnerCode', 'UniversityRanking', 'country', 'OverallPisaScore2022', 'Gender', 'Age', 'LP_Level_Rank', 'AvgRiskScore', 'EligibleToProgress']

# Filter the dataset to include only these columns and drop missing values
df_phase1_with_learner = all_data_final_2[columns_of_interest_with_learner].dropna()

# Encode categorical variables
label_encoders = {}
for column in ['country', 'Gender']:
    le = LabelEncoder()
    df_phase1_with_learner[column] = le.fit_transform(df_phase1_with_learner[column])
    label_encoders[column] = le


# In[ ]:


# Define the features and target variable, excluding LearnerCode
X_all_with_learner = df_phase1_with_learner.drop(columns=['EligibleToProgress', 'LearnerCode'])
y_all_with_learner = df_phase1_with_learner['EligibleToProgress']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all_with_learner, y_all_with_learner, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a basic Random Forest model with default parameters
basic_rf = RandomForestClassifier(random_state=42)
basic_rf.fit(X_train_scaled, y_train)

# Calculate the probabilities and ROC AUC score for the untuned model
y_pred_prob_basic = basic_rf.predict_proba(X_test_scaled)[:, 1]
roc_auc_basic = roc_auc_score(y_test, y_pred_prob_basic)

# Generate the ROC curve for the untuned model
fpr_basic, tpr_basic, _ = roc_curve(y_test, y_pred_prob_basic)

# Print the ROC AUC score for the untuned model
print("ROC AUC for untuned model: {:.3f}".format(roc_auc_basic))


# In[ ]:


# Make predictions on the test set
y_pred_basic = basic_rf.predict(X_test_scaled)

# Generate a classification report for the untuned model
classification_report_basic = classification_report(y_test, y_pred_basic)

print("Classification Report for Untuned Model:")
print(classification_report_basic)


# In[ ]:


# Apply scaling to the entire dataset before training the optimized model
X_all_with_learner_scaled = scaler.fit_transform(X_all_with_learner)

# Train the Random Forest model with the best parameters
best_params = {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
optimized_rf = RandomForestClassifier(**best_params, random_state=42)
optimized_rf.fit(X_train_scaled, y_train)


# In[ ]:


# Calculate the ROC AUC for the tuned model on the test set
y_pred_prob_optimized_rf = optimized_rf.predict_proba(X_test_scaled)[:, 1]
roc_auc_optimized = roc_auc_score(y_test, y_pred_prob_optimized_rf)

# Generate the ROC curve for the tuned model
fpr_optimized, tpr_optimized, _ = roc_curve(y_test, y_pred_prob_optimized_rf)

# Print the ROC AUC score for the tuned model
print("ROC AUC for tuned model: {:.3f}".format(roc_auc_optimized))


# In[ ]:


# Make predictions on the test set
y_pred_optimized_rf = optimized_rf.predict(X_test_scaled)

# Generate a classification report for the untuned model
classification_report_optimized = classification_report(y_test, y_pred_optimized_rf)

print("Classification Report for Optimized Model:")
print(classification_report_optimized)


# In[ ]:


# Plot the ROC curves for both models
plt.figure(figsize=(10, 6))
plt.plot(fpr_basic, tpr_basic, label='Untuned Model (AUC = {:.3f})'.format(roc_auc_basic))
plt.plot(fpr_optimized, tpr_optimized, label='Tuned Model (AUC = {:.3f})'.format(roc_auc_optimized))
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.5)')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: Untuned vs. Tuned Model')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[ ]:


# Calculate the probability of failure for all students
#y_pred_prob_optimized_rf = optimized_rf.predict_proba(X_all_scaled_with_learner)[:, 1]
#probability_of_failure_optimized_rf = 1 - y_pred_prob_optimized_rf

y_pred_prob_basic = basic_rf.predict_proba(X_all_with_learner_scaled)[:, 1]
probability_of_failure_basic = 1 - y_pred_prob_basic

# Combine LearnerCode, EligibleToProgress, and Probability_of_Failure
#results_df_optimized_rf = df_phase1_with_learner[['LearnerCode', 'EligibleToProgress']].copy()
#results_df_optimized_rf['Probability_of_Failure'] = probability_of_failure_optimized_rf

# Combine LearnerCode, EligibleToProgress, and Probability_of_Failure
results_df_basic = df_phase1_with_learner[['LearnerCode', 'EligibleToProgress']].copy()
results_df_basic['Probability_of_Failure'] = probability_of_failure_basic

#results_df_optimized_rf = results_df_optimized_rf.drop_duplicates(subset='LearnerCode')
#results_df_optimized_rf

results_df_basic = results_df_basic.drop_duplicates(subset='LearnerCode')
results_df_basic.head()


# In[ ]:


# Export the results to a CSV file
export_file_path_basic = 'students_failure_probabilities_basic.csv'
results_df_basic.to_csv(export_file_path_basic, index=False)


# In[ ]:


import matplotlib.pyplot as plt

# Plot the scatter graph
plt.figure(figsize=(14, 6))
colors = results_df_basic['EligibleToProgress'].map({True: 'lightgreen', False: 'lightcoral'})

plt.scatter(results_df_basic['LearnerCode'], results_df_basic['Probability_of_Failure'], c=colors, alpha=0.8, edgecolors='w', linewidth=0.5)

plt.xlabel('Learner Code')
plt.ylabel('Probability of Failure')
plt.title('Probability of Failure by Learner Code')
plt.grid(True, linestyle='--', alpha=0.1)

# Create a legend
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Eligible',
                          markerfacecolor='lightgreen', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Not Eligible',
                          markerfacecolor='lightcoral', markersize=10)]
plt.legend(handles=legend_elements, loc='upper right')

plt.show()


# In[ ]:


# Merge with custom suffixes
all_data_final_3 = pd.merge(all_data_final_2, results_df_basic, on='LearnerCode', how='inner', suffixes=('', '_rf'))

# Filter the dataframe based on 'Probability_of_Failure'
all_data_final_3_filtered = all_data_final_3[all_data_final_3['Probability_of_Failure'] <= 0.1]

# Filter the dataframe based on 'EligibleToProgress'
all_data_final_3_filtered = all_data_final_3_filtered[all_data_final_3_filtered['EligibleToProgress'] == False]

# Select the specified columns
all_data_final_3_filtered = all_data_final_3_filtered[['LearnerCode', 'Age', 'Gender', 'country', 'OverallPisaScore2022', 
                                     'LP_Level_Rank', 'ProgressionUniversity', 'UniversityRanking', 
                                     'CourseCategory', 'CourseName', 'CourseLevel_x', 'AvgRiskScore', 
                                     'Probability_of_Failure', 'CourseAttendancePercentage', 'EligibleToProgress']]

# Display the final dataframe
all_data_final_3_filtered = all_data_final_3_filtered.drop_duplicates(subset='LearnerCode').reset_index(drop=True)
all_data_final_3_filtered.head()


# In[ ]:


grouped_course_data = all_data_final_3.groupby(['ProgressionUniversity','CourseName']).agg({
    'LearnerCode': 'count',
    'Age': 'mean',
    'OverallPisaScore2022': 'mean',
    'LP_Level_Rank': 'mean',
    'UniversityRanking': 'mean',
    'CourseCategory': 'first',
    'CourseLevel_x': 'first',
    'AvgRiskScore': 'mean',
    'Probability_of_Failure': 'mean',
    'CourseAttendancePercentage': 'mean',
    'EligibleToProgress': 'mean'}).reset_index()

# Display the result
grouped_course_data.head()


# In[ ]:


university_data = grouped_course_data.groupby('ProgressionUniversity').agg({
    'LearnerCode': 'sum',
    'Age': 'mean',
    'OverallPisaScore2022': 'mean',
    'LP_Level_Rank': 'mean',
    'UniversityRanking': 'mean',
    'AvgRiskScore': 'mean',
    'Probability_of_Failure': 'mean',
    'CourseAttendancePercentage': 'mean',
    'EligibleToProgress': 'mean'}).reset_index()

university_data


# <hr>

# ## Phase 1: Randomizer for student data

# <hr>

# !!!! Randomizer for both phases here

# In[ ]:


# Filtering nationalities with available Pisa scores
nationalities_with_pisa = all_data_final_2.dropna(subset=['OverallPisaScore2022'])['Nationality'].unique()

# Define possible values for each field, excluding NaN values
possible_universities = university_data['ProgressionUniversity'].dropna().unique()
possible_nationalities = [nat for nat in all_data_final_2['Nationality'].dropna().unique() if nat in nationalities_with_pisa]
possible_genders = all_data_final_2['Gender'].dropna().unique()
possible_courses = grouped_course_data['CourseName'].dropna().unique()
possible_universities = all_data_final_2['ProgressionUniversity'].dropna().unique()
possible_degrees= all_data_final_2['ProgressionDegree'].dropna().unique()
possible_terms = all_data_final_2['Term'].dropna().unique()

# Function to generate random student data
def generate_random_student_data():
    return {
        'LearnerCode': random.randint(100000, 999999),
        'ProgressionUniversity': random.choice(possible_universities),
        'Nationality': random.choice(possible_nationalities),
        'Gender': random.choice(possible_genders),
        'Age': random.randint(18, 50),
        'LP_Level_Rank': random.randint(1, 5),
        'CourseName': random.choice(possible_courses),
		'ProgressionDegree': random.choice(possible_degrees), 
        'ExpectedtHours': random.randint(15, 60),
        'Term': random.choice(possible_terms),
        'CourseAttendancePercentage': random.randint(0, 100),
        'Percentile': random.uniform(0, 1)
    }

# Generate random student data
new_student_data = generate_random_student_data()
new_student_data


# <hr>

# In[ ]:


# Lookup and calculate additional features
new_student_data['UniversityRanking'] = all_data_final_3.loc[all_data_final_3['ProgressionUniversity'] == new_student_data['ProgressionUniversity'], 'UniversityRanking'].values[0]
new_student_data['OverallPisaScore2022'] = all_data_final_3.loc[all_data_final_3['Nationality'] == new_student_data['Nationality'], 'OverallPisaScore2022'].values[0]
new_student_data['AvgRiskScore'] = all_data_final_3.loc[all_data_final_3['CourseName'] == new_student_data['CourseName'], 'AvgRiskScore'].values[0]

# Update 'country' and fetch PISA score
new_student_data['country'] = all_data_final_3.loc[all_data_final_3['Nationality'] == new_student_data['Nationality'], 'country'].values[0]

# Convert to DataFrame for processing
new_student_df = pd.DataFrame([new_student_data])

# Reorder the columns to match the desired order so it fits the original RF fit
new_student_df = new_student_df[['LearnerCode', 'ProgressionUniversity', 'CourseName','Nationality', 'UniversityRanking', 'country', 'OverallPisaScore2022', 'Gender', 'Age', 'LP_Level_Rank', 'AvgRiskScore']]
#print(new_student_df)

# Encode categorical variables using the same encoders as the training data
for column in ['country', 'Gender']:
    new_student_df[column] = label_encoders[column].transform(new_student_df[column])

# Standardize the features using the same scaler as the training data
new_student_scaled = scaler.transform(new_student_df.drop(columns=['LearnerCode', 'ProgressionUniversity', 'CourseName','Nationality']))

# Calculate the probability of the student being eligible to progress
prob_eligible = basic_rf.predict_proba(new_student_scaled)[:, 1]

# Calculate the probability of the student failing
prob_failure = 1 - prob_eligible

plt.rcParams['font.family'] = 'STIXGeneral'

# Prepare the data for display
display_data = {
    'Learner Code': new_student_data['LearnerCode'],
    'University': new_student_data['ProgressionUniversity'],
    'Country': new_student_data['country'],
    'PISA Score': new_student_data['OverallPisaScore2022'],
    'Gender': new_student_data['Gender'],
    'Age': new_student_data['Age'],
    'Course': new_student_data['CourseName'],
    'LP Level Rank': new_student_data['LP_Level_Rank'],
    'Avg Risk Score': new_student_data['AvgRiskScore']
}

# Calculate the eligibility probability as a percentage
prob_eligible_percent = prob_eligible[0] * 100

# Determine the color based on the probability
if prob_eligible_percent > 75:
    prob_color = 'darkgreen'
elif prob_eligible_percent > 25:
    prob_color = 'orange'
else:
    prob_color = 'darkred'

# Create the figure and the gridspec layout
fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

# Create a subplot for the student information
ax0 = plt.subplot(gs[0])
ax0.axis('off')

# Display student information with bold headings and dark grey text
text = "\n".join([f"{key}: {value}" for key, value in display_data.items()])
formatted_text = "\n".join([f"{key}: " + f"{value}" for key, value in display_data.items()])

# Add each line of text separately to apply formatting
for idx, (key, value) in enumerate(display_data.items()):
    ax0.text(0, 1 - (idx * 0.07), f"{key}: ", va='center', ha='left', fontsize=12,
             fontweight='bold', color='grey', transform=ax0.transAxes)
    ax0.text(0.35, 1 - (idx * 0.07), f"{value}", va='center', ha='left', fontsize=10,
             color='darkgrey', transform=ax0.transAxes)

# Create a subplot for the eligibility probability
ax1 = plt.subplot(gs[1])
ax1.axis('off')

# Display the probability as a big number
ax1.text(1, 0.9, f"{prob_eligible_percent:.1f}%", color=prob_color,
         fontsize=50, fontweight='bold', ha='center', va='center')

# Set the title or additional information if needed
fig.suptitle('Student Eligibility Prediction', fontsize=16, fontweight='bold')

plt.show()


# In[ ]:


# Feature importance for eligibilitytoprogress
# Extract feature importances from the Random Forest model
feature_importances = basic_rf.feature_importances_

# Create a DataFrame to organize feature names and their importance
features_df = pd.DataFrame({
    'Feature': X_all_with_learner.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
features_df = features_df.sort_values(by='Importance', ascending=False)

# Plotting a bar chart instead of a barplot
plt.figure(figsize=(14, 6))
plt.bar(features_df['Feature'], features_df['Importance'], color='#013989ff')
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Random Forest Model - Phase1')
plt.show()


# In[ ]:


# Assign a risk group to all students based on their Probability_of_Failure
all_data_final_3['StudentFailRiskGroup'] = all_data_final_3['Probability_of_Failure'].apply(
    lambda prob: 5 if prob >= 0.75 else (3 if prob >= 0.25 else 1)
)

# Display the updated DataFrame
all_data_final_3


# In[ ]:


# Create a new dataframe with ModuleName, CourseLevel, and average of ModuleTotal
module_avg_total = module_data.groupby(['ModuleName', 'CourseLevel']).agg(
    AvgModuleTotal=('ModuleTotal', 'mean')
).reset_index()

# Add AvgModuleTotal column into module_data dataframe
module_data_2 = pd.merge(module_data, module_avg_total, on=['ModuleName', 'CourseLevel'], how='left')

# Calculate the percentile for each LearnerCode in each Module
module_data_2['Percentile'] = module_data_2.groupby('ModuleName')['ModuleTotal'].rank(pct=True)

# Filter out ModuleCreditEquivalence == 0
module_data_2 = module_data_2[module_data_2['ModuleCreditEquivalence'] != 0]

# Keep only the required columns
module_data_2 = module_data_2[['ModuleName', 'ModulePass', 'ModuleCode', 'LearnerCode', 'ModuleTotal' , 'AvgModuleTotal', 'Percentile']]

# Group by LearnerCode and calculate the mean of Percentile
student_percentile = module_data_2.groupby('LearnerCode')['Percentile'].mean().reset_index()

# Ensure 'LearnerCode' column is retained from 'all_data_final_3' during merge
student_phase2 = pd.merge(student_percentile, all_data_final_3[['LearnerCode', 'EligibleToProgress', 'StudentFailRiskGroup']], on='LearnerCode', how='left')

student_phase2 = student_phase2.dropna(subset=['EligibleToProgress'])

# Get attandance percentage from course data
student_phase2 = pd.merge(student_phase2, course_data[['LearnerCode', 'CourseAttendancePercentage']], on='LearnerCode', how='left')

student_phase2 = student_phase2.drop_duplicates(subset=['LearnerCode'])
student_phase2


# In[ ]:


# Define thresholds for attendance and percentile
attendance_thresholds = {
    "high": 85,    # I need to figure out these numbers better!!!
    "low": 60      
}

# Define percentiles thresholds (for splitting into thirds: Above average, Average, Below average)
percentile_thresholds = student_phase2["Percentile"].quantile([1/3, 2/3])

# Add a new column to classify attendance
def classify_attendance(attendance):
    if attendance >= attendance_thresholds['high']:
        return 'High'
    elif attendance >= attendance_thresholds['low']:
        return 'Normal'
    else:
        return 'Low'

# Add a new column to classify percentile
def classify_percentile(percentile):
    if percentile >= percentile_thresholds[2/3]:
        return 'Above average'
    elif percentile >= percentile_thresholds[1/3]:
        return 'Average'
    else:
        return 'Below average'

# Apply the classification to the dataset
student_phase2['AttendanceCategory'] = student_phase2['CourseAttendancePercentage'].apply(classify_attendance)
student_phase2['PercentileCategory'] = student_phase2['Percentile'].apply(classify_percentile)

# Define the risk levels based on the provided mapping
risk_mapping = {
    ('High', 'Above average'): 0,
    ('High', 'Average'): 1,
    ('High', 'Below average'): 2,
    ('Normal', 'Above average'): 1,
    ('Normal', 'Average'): 2,
    ('Normal', 'Below average'): 3,
    ('Low', 'Above average'): 2,
    ('Low', 'Average'): 3,
    ('Low', 'Below average'): 4,
}

# Create the risk group column based on the attendance and percentile categories
student_phase2['Phase2RiskGroup'] = student_phase2.apply(lambda row: risk_mapping.get((row['AttendanceCategory'], row['PercentileCategory']), None), axis=1)

student_phase2


# In[ ]:


# Merge the dataframes to add phase2 risk group into the main dataframe
all_data_final_3 = pd.merge(all_data_final_3, student_phase2[['LearnerCode', 'Phase2RiskGroup']], on='LearnerCode', how='left')

all_data_final_3.columns


# In[ ]:


# Drop the columns with the '_y' suffix
columns_to_drop = [col for col in all_data_final_3.columns if col.endswith('_y')]
all_data_final_3_cleaned = all_data_final_3.drop(columns=columns_to_drop)

# Rename columns by removing the '_x' suffix
all_data_final_3_cleaned = all_data_final_3_cleaned.rename(columns=lambda x: x.rstrip('_x'))

all_data_final_4 = all_data_final_3_cleaned.drop(columns=['EligibleToProgress_rf'])

# Check the renamed columns
all_data_final_4.columns


# In[ ]:


# Detailed analysis columns
analysis_columns = ['CentreName', 'AcademicYear', 'LearnerCode', 'Gender', 'Nationality',
       'CourseLevel', 'CourseName', 'CourseFirstIntakeDate', 'IsFirstCohort', 'CompletedCourse', 'ProgressionDegree', 'ProgressionUniversity',
       'EligibleToProgress', 'ExpectedtHours', 'CourseCategory', 'Age',
       'UniversityRanking', 'Term', 'ContactHours', 'LP_Level', 'LP_Level_Rank', 'AvgRiskScore',
       'OverallPisaScore2022', 'StudentFailRiskGroup', 'Phase2RiskGroup']

# Basic analysis columns
basic_analysis_columns = ['ProgressionDegree','LearnerCode',
       'EligibleToProgress', 'ExpectedtHours', 'Age',
       'UniversityRanking', 'Term', 'AvgRiskScore',
       'OverallPisaScore2022', 'StudentFailRiskGroup', 'Phase2RiskGroup']

# Basic analysis columns v2
basic_analysis_columns_v2 = ['ProgressionDegree','LearnerCode',
       'EligibleToProgress', 'ExpectedtHours', 
       'Term', 'StudentFailRiskGroup', 'Phase2RiskGroup']

# Select only the columns that are in the analysis_columns list
phase2_df = all_data_final_4[basic_analysis_columns_v2]

# List of columns to convert to object type
columns_to_convert = ['LearnerCode', 'StudentFailRiskGroup', 'Phase2RiskGroup']

# Convert the columns to object type
phase2_df.loc[:, columns_to_convert] = phase2_df[columns_to_convert].astype('object')

# Drop rows with missing values
phase2_df = phase2_df.dropna()

# Rename StudentFailRiskGroup to Phase1RiskGroup
phase2_df = phase2_df.rename(columns={'StudentFailRiskGroup': 'Phase1RiskGroup'})

# Verify the change
phase2_df.head()


# In[ ]:


# Handle categorical variables by label encoding (for simplicity)
df_encoded = phase2_df.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)

# Split the data into features and target
X = df_encoded.drop(['EligibleToProgress', 'LearnerCode'], axis=1)
y = df_encoded['EligibleToProgress']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print the ROC AUC score
print(f"ROC AUC Score: {roc_auc:.4f}")


# In[ ]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation on the model
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

# Print the cross-validation scores and the mean score
cv_scores, cv_scores.mean()


# In[ ]:


# Out-of-bag validation
model_oob = RandomForestClassifier(random_state=42, oob_score=True)

# Fit the model on the training data
model_oob.fit(X_train, y_train)

# Get the OOB score
oob_score = model_oob.oob_score_

# Print the OOB score
print(f"OOB Score: {oob_score:.4f}")


# In[ ]:


# Get feature importance from the trained model
feature_importances = model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(14, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Predicting Eligible to Progress')
plt.gca().invert_yaxis()
plt.show()


# In[ ]:


all_data_final_4_cleaned = all_data_final_4.dropna(subset=['AvgRiskScore'])

# Calculate the Phase 1 Risk Group for the randomized student
if prob_failure >= 0.75:
    p1_risk_group = 5
elif prob_failure >= 0.25:
    p1_risk_group = 3
else:
    p1_risk_group = 1

print(p1_risk_group)

# Manual config for attandance and percentile for Phase 2 Calculations
manual_attendance = 55  # Select something between 0 and 100, CourseAttendancePercentage
manual_percentile = 0.73 # Select something between 0 and 1, Percentile

# Use new_student_data from previous context
new_student_data_p2_full = pd.DataFrame([new_student_data])
new_student_data_p2_full['Phase1RiskGroup'] = p1_risk_group  # Add Phase1RiskGroup separately
new_student_data_p2_full['CourseAttendancePercentage'] = manual_attendance
new_student_data_p2_full['Percentile'] = manual_percentile

# Select the required columns
new_student_data_p2_initial = new_student_data_p2_full[['ProgressionDegree', 'LearnerCode', 'ExpectedtHours', 'Age', 'UniversityRanking', 'Term', 'AvgRiskScore', 'OverallPisaScore2022', 'Phase1RiskGroup','CourseAttendancePercentage', 'Percentile']]
new_student_data_p2_initial


# In[ ]:


new_student_data_p2_initial.loc[:, 'AttendanceCategory'] = new_student_data_p2_initial['CourseAttendancePercentage'].apply(classify_attendance)
new_student_data_p2_initial.loc[:, 'PercentileCategory'] = new_student_data_p2_initial['Percentile'].apply(classify_percentile)

risk_mapping = {
    ('High', 'Above average'): 0,
    ('High', 'Average'): 1,
    ('High', 'Below average'): 2,
    ('Normal', 'Above average'): 1,
    ('Normal', 'Average'): 2,
    ('Normal', 'Below average'): 3,
    ('Low', 'Above average'): 2,
    ('Low', 'Average'): 3,
    ('Low', 'Below average'): 4,
}

new_student_data_p2_initial.loc[:, 'Phase2RiskGroup'] = new_student_data_p2_initial.apply(lambda row: risk_mapping.get((row['AttendanceCategory'], row['PercentileCategory']), None), axis=1)

new_student_data_p2 = new_student_data_p2_initial.drop(columns=['ProgressionUniversity', 'CourseName', 'CourseAttendancePercentage', 'Percentile', 'AttendanceCategory', 'PercentileCategory'], errors='ignore')

# Rearrange columns
#new_student_data_p2 = new_student_data_p2[['ProgressionDegree', 'LearnerCode', 'ExpectedtHours', 'Age', 'UniversityRanking', 'Term', 'AvgRiskScore', 'OverallPisaScore2022', 'Phase1RiskGroup', 'Phase2RiskGroup']]

# Rearrange columns
new_student_data_p2 = new_student_data_p2[['ProgressionDegree', 'LearnerCode', 'ExpectedtHours',  'Term', 'Phase1RiskGroup', 'Phase2RiskGroup']]

# Display the final DataFrame
new_student_data_p2

new_student_data_p2_2 = new_student_data_p2.drop(columns=['LearnerCode'], errors='ignore')

# Encode categorical variables as per the training data encoding
for column in new_student_data_p2_2.columns:
    if new_student_data_p2_2[column].dtype == 'object':
        new_student_data_p2_2[column] = LabelEncoder().fit_transform(new_student_data_p2_2[column])

# Predict probabilities for the positive class (eligible to progress)
predicted_probabilities = model.predict_proba(new_student_data_p2_2)[:, 1]

# Display the probability predictions
new_student_data_p2.loc[:, 'EligibilityProbability'] = predicted_probabilities

new_student_data_p2


# In[ ]:


# Prepare the data for display (converting all data to strings to avoid metadata display)
display_data = {
    'Learner Code': str(new_student_data_p2_full['LearnerCode'].values[0]),
    'Age': str(new_student_data_p2_full['Age'].values[0]),
    'Gender': str(new_student_data_p2_full['Gender'].values[0]),
    'Language Proficiency': str(new_student_data_p2_full['LP_Level_Rank'].values[0]),
    'Country': str(new_student_data_p2_full['country'].values[0]),
    'PISA Score': str(new_student_data_p2_full['OverallPisaScore2022'].values[0]),
    'University': str(new_student_data_p2_full['ProgressionUniversity'].values[0]),
    'University Ranking': str(new_student_data_p2_full['UniversityRanking'].values[0]),
    'Degree': str(new_student_data_p2_full['ProgressionDegree'].values[0]),
    'Term': str(new_student_data_p2_full['Term'].values[0]),
    'Course': str(new_student_data_p2_full['CourseName'].values[0]),
    'Avg Risk Score': str(new_student_data_p2_full['AvgRiskScore'].values[0]),
    'Expected Hours': str(new_student_data_p2_full['ExpectedtHours'].values[0]),
    'Course Attendance': str(new_student_data_p2_full['CourseAttendancePercentage'].values[0]),
    'Grade percentile': str(new_student_data_p2_full['Percentile'].values[0])
}

plt.rcParams['font.family'] = 'STIXGeneral'

prob_eligible_percent_phase1 = prob_eligible[0] * 100  # First probability (Phase 1)
prob_eligible_percent_phase2 = predicted_probabilities[0] * 100  # Second probability (Phase 2)

# Calculate delta between Phase 1 and Phase 2
delta_percent = prob_eligible_percent_phase2 - prob_eligible_percent_phase1

# Determine the color based on the second phase probability
if prob_eligible_percent_phase2 > 75:
    prob_color_phase2 = 'darkgreen'
elif prob_eligible_percent_phase2 > 25:
    prob_color_phase2 = 'orange'
else:
    prob_color_phase2 = 'darkred'

# Determine delta triangle direction and color
if delta_percent > 0:
    triangle_color = 'darkgreen'
    triangle_vertices = [(0.90, 0.615), (0.88, 0.587), (0.92, 0.587)]  # Upward triangle
elif delta_percent < 0:
    triangle_color = 'darkred'
    triangle_vertices = [(0.90, 0.587), (0.88, 0.615), (0.92, 0.615)]  # Downward triangle
else:
    triangle_color = 'grey'
    triangle_vertices = None  # No triangle in case of no change


# Create the figure and the gridspec layout
fig = plt.figure(figsize=(12, 6))  # Adjust figure size
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

# Create a subplot for the student information
ax0 = plt.subplot(gs[0])
ax0.axis('off')

# Adjust the vertical space between lines
line_spacing = 0.05  # Increase spacing

# Add each line of text separately to apply formatting
for idx, (key, value) in enumerate(display_data.items()):
    # Adjust text placement and size
    ax0.text(0, 1 - (idx * line_spacing), f"{key}: ", va='center', ha='left', fontsize=12,
             fontweight='bold', color='blue', transform=ax0.transAxes)
    ax0.text(0.35, 1 - (idx * line_spacing), f"{value}", va='center', ha='left', fontsize=10,
             color='darkgrey', transform=ax0.transAxes)

# Create a subplot for the eligibility probability
ax1 = plt.subplot(gs[1])
ax1.axis('off')

# Display the second probability (Phase 2) as the main, larger number
ax1.text(0.4, 0.6, f"{prob_eligible_percent_phase2:.1f}%", color=prob_color_phase2,
         fontsize=40, fontweight='bold', ha='center', va='center')

# Display the delta with a triangle to the right of Phase 2 probability
if delta_percent != 0:
    ax1.text(0.78, 0.6, f"{delta_percent:.1f}%", color=triangle_color,
             fontsize=11, fontweight='bold', ha='center', va='center')

    # Adding the triangle pointing up or down
    triangle = Polygon(triangle_vertices, closed=True, color=triangle_color, transform=ax1.transAxes)
    ax1.add_patch(triangle)

# Display the first probability (Phase 1) closer to Phase 2, smaller and semi-transparent
ax1.text(0.4, 0.5, f"P1: {prob_eligible_percent_phase1:.1f}%", color='black',
         fontsize=14, fontweight='bold', ha='center', va='center', alpha=0.4)

# Set the title or additional information if needed
fig.suptitle('Student Eligibility Prediction: Phase 1 vs Phase 2', fontsize=16, fontweight='bold')

plt.show()


# In[ ]:


import numpy as np
import seaborn as sns

# Iterate a range of percentages
attendance_range = np.linspace(0, 100, 100)
grades_range = np.linspace(0, 1, 100)

# Lists to store probs
attendance_probabilities = []
grades_probabilities = []

# Calculating probabilities for different attendance levels
for attendance in attendance_range:
    new_student_data_p2_full['CourseAttendancePercentage'] = attendance
    new_student_data_p2_full['AttendanceCategory'] = new_student_data_p2_full['CourseAttendancePercentage'].apply(classify_attendance)
    new_student_data_p2_full['PercentileCategory'] = new_student_data_p2_full['Percentile'].apply(classify_percentile)
    new_student_data_p2_full['Phase2RiskGroup'] = new_student_data_p2_full.apply(lambda row: risk_mapping.get((row['AttendanceCategory'], row['PercentileCategory']), None), axis=1)
    new_student_data_p2 = new_student_data_p2_full[['ProgressionDegree', 'ExpectedtHours', 'Term', 'Phase1RiskGroup', 'Phase2RiskGroup']]
    new_student_data_p2 = new_student_data_p2.drop(columns=['LearnerCode'], errors='ignore')
    for column in new_student_data_p2.columns:
        if new_student_data_p2[column].dtype == 'object':
            new_student_data_p2[column] = LabelEncoder().fit_transform(new_student_data_p2[column])
    predicted_probabilities = model.predict_proba(new_student_data_p2)[:, 1]
    attendance_probabilities.append(predicted_probabilities[0] * 100)

# Calculating probabilities for different grade percentiles
for grade in grades_range:
    new_student_data_p2_full['Percentile'] = grade
    new_student_data_p2_full['AttendanceCategory'] = new_student_data_p2_full['CourseAttendancePercentage'].apply(classify_attendance)
    new_student_data_p2_full['PercentileCategory'] = new_student_data_p2_full['Percentile'].apply(classify_percentile)
    new_student_data_p2_full['Phase2RiskGroup'] = new_student_data_p2_full.apply(lambda row: risk_mapping.get((row['AttendanceCategory'], row['PercentileCategory']), None), axis=1)
    new_student_data_p2 = new_student_data_p2_full[['ProgressionDegree', 'ExpectedtHours', 'Term', 'Phase1RiskGroup', 'Phase2RiskGroup']]
    new_student_data_p2 = new_student_data_p2.drop(columns=['LearnerCode'], errors='ignore')
    for column in new_student_data_p2.columns:
        if new_student_data_p2[column].dtype == 'object':
            new_student_data_p2[column] = LabelEncoder().fit_transform(new_student_data_p2[column])
    predicted_probabilities = model.predict_proba(new_student_data_p2)[:, 1]
    grades_probabilities.append(predicted_probabilities[0] * 100)

# DF for plotting
plot_data = pd.DataFrame({
    'Attendance': attendance_range,
    'Grades': grades_range,
    'Attendance_Prob': attendance_probabilities,
    'Grades_Prob': grades_probabilities
})

# KDE
plt.figure(figsize=(14, 6))

# Attendance plot
plt.subplot(1, 2, 1)
sns.lineplot(x='Attendance', y='Attendance_Prob', data=plot_data, color='blue')
sns.kdeplot(x='Attendance', y='Attendance_Prob', data=plot_data, fill=True, color='blue', alpha=0.2)
plt.xlabel('Course Attendance Percentage')
plt.ylabel('Eligibility Probability (%)')
plt.title('Eligibility Probability vs. Course Attendance')

# Grades plot
plt.subplot(1, 2, 2)
sns.lineplot(x='Grades', y='Grades_Prob', data=plot_data, color='green')
sns.kdeplot(x='Grades', y='Grades_Prob', data=plot_data, fill=True, color='green', alpha=0.2)
plt.xlabel('Grade Percentile')
plt.ylabel('Eligibility Probability (%)')
plt.title('Eligibility Probability vs. Grade Percentile')

plt.tight_layout()
plt.show()

