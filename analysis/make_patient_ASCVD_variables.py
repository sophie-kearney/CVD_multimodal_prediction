import pandas as pd
#variables needed
#Age
#sex
#race
#total cholesterol
#HDL-C
#systolic BP
#Diabetes
#Current Smoker

#Read in files:
x_agg = pd.read_csv("/Users/niaabdu/Desktop/X_agg.csv")
observation_descriptions_code_map = pd.read_csv("/Users/niaabdu/Desktop/observations_descriptions_code_map.csv")
codes_cvd = pd.read_csv("/Users/niaabdu/Desktop/codes_cvd.csv")
patients_csv = pd.read_csv("/Users/niaabdu/Desktop/coherent-11-07-2022/csv/patients.csv")
encounters_csv = pd.read_csv("/Users/niaabdu/Desktop/coherent-11-07-2022/csv/encounters.csv")
conditions_csv = pd.read_csv("/Users/niaabdu/Desktop/coherent-11-07-2022/csv/conditions.csv")
observations_csv = pd.read_csv("/Users/niaabdu/Desktop/coherent-11-07-2022/csv/observations.csv")

#create a new csv = patient_ASCVD_variables.csv
#Total Cholesterol = 2093-3
#HDL-C = 8480-6
#Systolic Blood Pressure = 2085-9
patient_ASCVD_variables = x_agg[["PATIENT", "2093-3_median","8480-6_median", "2085-9_median"]].copy()
print(patient_ASCVD_variables)
print("Shape: ", patient_ASCVD_variables.shape)

#pull age, sex and race from patients.csv
    #calculate age as 4 years from first observation and by birthday
    #pull sex and merge to patient_ASCVD_variables.csv
#Adding race and gender, gender is represented as "M" or "F", race is lowercase "white", "black", "asian", etc
patient_ASCVD_variables = pd.merge(patient_ASCVD_variables, patients_csv[["Id", "RACE", "GENDER", "BIRTHDATE"]], left_on="PATIENT",right_on="Id",how="left")
print(patient_ASCVD_variables)
print("Shape after adding sex and race: ", patient_ASCVD_variables.shape)

#Age, we will use 4 years from the first encounter date, using the same could from ananyas evaulte_prs.py
encounters_csv['START'] = pd.to_datetime(encounters_csv['START'])
first_encounter = encounters_csv.sort_values('START').groupby('PATIENT').first().reset_index()[['PATIENT', 'START']].rename(columns={'START': 'FIRST_ENCOUNTER_DATE'})
patient_ASCVD_variables = pd.merge(patient_ASCVD_variables,first_encounter, on='PATIENT', how='left')
patient_ASCVD_variables['BIRTHDATE'] = pd.to_datetime(patient_ASCVD_variables['BIRTHDATE'], errors='coerce')
patient_ASCVD_variables['FIRST_ENCOUNTER_DATE'] = pd.to_datetime(patient_ASCVD_variables['FIRST_ENCOUNTER_DATE'], errors='coerce')

patient_ASCVD_variables['AGE_AT_FIRST_ENCOUNTER'] = (
    (patient_ASCVD_variables['FIRST_ENCOUNTER_DATE'].dt.tz_localize(None)
     - patient_ASCVD_variables['BIRTHDATE'])
    .dt.days // 365
)

patient_ASCVD_variables['AGE'] = patient_ASCVD_variables['AGE_AT_FIRST_ENCOUNTER'] + 4
print(patient_ASCVD_variables)
print("Shape after adding sex and race: ", patient_ASCVD_variables.shape)
print("Age column: ", patient_ASCVD_variables["AGE"])
print("All columns: ", patient_ASCVD_variables.columns)
patient_ASCVD_variables = patient_ASCVD_variables.rename(columns={"2093-3_median": "Total Cholesterol", "8480-6_median": "HDL-C", "2085-9_median": "Systolic Blood Pressure"})

#Adding diabetes
#each row in conditions is a single visit, take the 4 years before cvd and if diabetes is shown in "DESCRIPTION" mark patient as diabetes if not or NA mark as 0

#Wob stands for window of observation
# Convert START to datetime in conditions
conditions_csv["START"] = pd.to_datetime(conditions_csv["START"], errors='coerce')
# Merge conditions with the first encounter date so each row knows the Wobs window
conditions_with_encounter = pd.merge(conditions_csv,patient_ASCVD_variables[["PATIENT", "FIRST_ENCOUNTER_DATE"]],on="PATIENT",how="inner")
# Remove timezone awareness
conditions_with_encounter["START"] = conditions_with_encounter["START"].dt.tz_localize(None)
conditions_with_encounter["FIRST_ENCOUNTER_DATE"] = conditions_with_encounter["FIRST_ENCOUNTER_DATE"].dt.tz_localize(None)
# Define the end of the 4-year observation window
conditions_with_encounter["Wobs_end"] = (conditions_with_encounter["FIRST_ENCOUNTER_DATE"] + pd.DateOffset(years=4)).dt.tz_localize(None)
# Keep only conditions within the observation window
conditions_in_window = conditions_with_encounter[(conditions_with_encounter["START"] >= conditions_with_encounter["FIRST_ENCOUNTER_DATE"]) &
(conditions_with_encounter["START"] <= conditions_with_encounter["Wobs_end"])]
# Filter rows where DESCRIPTION is exactly "Diabetes"
diabetes_rows = conditions_in_window[conditions_in_window["DESCRIPTION"].str.strip().str.lower() == "diabetes".lower()]
# Convert to a 1/0 indicator per patient
diabetes_flag = (
    diabetes_rows.groupby("PATIENT")
    .size()
    .gt(0)
    .astype(int)
    .reset_index(name="Diabetes")
)
# Merge indicator into ASCVD dataset
patient_ASCVD_variables = pd.merge(patient_ASCVD_variables,diabetes_flag,on="PATIENT",how="left")

# Patients with no diabetes rows → fill with 0
patient_ASCVD_variables["Diabetes"] = patient_ASCVD_variables["Diabetes"].fillna(0).astype(int)
print("Shape after adding diabetes: ", patient_ASCVD_variables.shape)
print("diabetes column: ", patient_ASCVD_variables["Diabetes"])
print("All columns: ", patient_ASCVD_variables.columns)
print(patient_ASCVD_variables[patient_ASCVD_variables["Diabetes"] == 1])


#TODO:
#seems there is both "smokes tobacco daily" in codes_cvd.csv and "Tobacco smoking status NHIS" observations but I will just use "Tobacco smoking status NHIS" in observations
#"Tobacco smoking status NHIS" is "current smoker", "former smoker" or "never smoker", in the "Value" Column, "Tobacco smoking status NHIS" is in "DESCRIPTION" columns
#look at 4 years since FIRST_ENCOUNTER_DATE
#if patient is "current smoker" or "former smoker" then set to 1
#if patient is "never smoker" at the last point then set to 0

# Convert DATE column
observations_csv["DATE"] = pd.to_datetime(observations_csv["DATE"], errors='coerce')
# Keep only smoking status rows
smoking_obs = observations_csv[observations_csv["DESCRIPTION"].str.strip().str.lower() == "tobacco smoking status nhis"]
# Merge in FIRST_ENCOUNTER_DATE
smoking_with_encounter = pd.merge(smoking_obs,patient_ASCVD_variables[["PATIENT", "FIRST_ENCOUNTER_DATE"]],on="PATIENT",how="inner")
# Ensure dates are timezone-naive
smoking_with_encounter["DATE"] = smoking_with_encounter["DATE"].dt.tz_localize(None)
smoking_with_encounter["FIRST_ENCOUNTER_DATE"] = smoking_with_encounter["FIRST_ENCOUNTER_DATE"].dt.tz_localize(None)
# Define end of Wobs window
smoking_with_encounter["Wobs_end"] = (smoking_with_encounter["FIRST_ENCOUNTER_DATE"] + pd.DateOffset(years=4)).dt.tz_localize(None)
# Filter only visits inside Wobs
smoking_in_window = smoking_with_encounter[(smoking_with_encounter["DATE"] >= smoking_with_encounter["FIRST_ENCOUNTER_DATE"]) &
    (smoking_with_encounter["DATE"] <= smoking_with_encounter["Wobs_end"])]
# Take the *latest* smoking status in the window (ASCVD uses current status)
latest_smoking = (smoking_in_window.sort_values(["PATIENT", "DATE"]).groupby("PATIENT").tail(1))
# Map VALUE to binary Smoker indicator
latest_smoking["Smoker"] = latest_smoking["VALUE"].str.strip().str.lower().map({"current smoker": 1,"former smoker": 1,"never smoker": 0})
# Merge back to ASCVD variable table
patient_ASCVD_variables = pd.merge(patient_ASCVD_variables,latest_smoking[["PATIENT", "Smoker"]],on="PATIENT",how="left")
# Fill missing (no smoking records → assume non-smoker)
patient_ASCVD_variables["Smoker"] = patient_ASCVD_variables["Smoker"].fillna(0).astype(int)

print("smoker column: ", patient_ASCVD_variables["Smoker"])
print("All columns: ", patient_ASCVD_variables.columns)
print(patient_ASCVD_variables[patient_ASCVD_variables["Smoker"] == 1])
print(patient_ASCVD_variables["Smoker"].value_counts())

patient_ASCVD_variables = patient_ASCVD_variables.drop(columns=["BIRTHDATE", "AGE_AT_FIRST_ENCOUNTER", "Id"])
print("All columns are drop:", patient_ASCVD_variables.columns)
patient_ASCVD_variables.to_csv("/Users/niaabdu/Desktop/patient_ASCVD_variables.csv")
