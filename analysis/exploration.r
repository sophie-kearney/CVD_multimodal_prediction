library(tidyverse)


cond <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/conditions.csv")

View(cond)

length(unique(cond$PATIENT))
length(unique(cond$CODE))

# ----
# avg encounters

df_summary <- cond %>%
  group_by(PATIENT) %>%
  summarise(num_encounters = n_distinct(ENCOUNTER), .groups = "drop")
mean_encounters <- mean(df_summary$num_encounters)

ggplot(df_summary, aes(x = num_encounters)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "white") +
  geom_vline(aes(xintercept = mean_encounters), color = "red", linetype = "dashed") +
  labs(x = "Number of Encounters per Patient", 
       y = "Count of Patients",
       title = paste0("Average Encounters per Patient: ", round(mean_encounters, 2))) +
  theme_bw()

# ---- 
# avg conditions

df_conditions <- cond %>%
  group_by(PATIENT) %>%
  summarise(num_conditions = n_distinct(CODE), .groups = "drop")

mean_conditions <- mean(df_conditions$num_conditions)

ggplot(df_conditions, aes(x = num_conditions)) +
  geom_histogram(binwidth = 1, fill = "lightgreen", color = "white") +
  geom_vline(aes(xintercept = mean_conditions), color = "red", linetype = "dashed") +
  labs(x = "Number of Conditions per Patient",
       y = "Count of Patients",
       title = paste0("Average Conditions per Patient: ", round(mean_conditions, 2))) +
  theme_bw()

# ---- 

patients <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/patients.csv")

dna_files <- list.files("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/dna", full.names = FALSE, pattern = "\\.csv$")
dna_df <- data.frame(
  file = dna_files,
  stringsAsFactors = FALSE
) %>%
  mutate(
    FIRST = str_extract(file, "^[^_]+"),
    LAST = str_extract(file, "(?<=_)[^_]+"),
    FIRST = str_to_title(FIRST),
    LAST = str_to_title(LAST)
  )

dna_patients <- dna_df %>%
  inner_join(patients, by = c("FIRST", "LAST")) %>%
  select(Id, FIRST, LAST)

overlap <- dna_patients %>%
  inner_join(cond, by = c("Id" = "PATIENT"))
length(unique(overlap$Id))

# ----

observations <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/observations.csv")
length(unique(enc$PATIENT))

overlap <- enc %>%
  inner_join(cond, by = c("PATIENT" = "PATIENT"))
length(unique(overlap$PATIENT))

# ----

library(UpSetR)

observations <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/observations.csv")
medications <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/medications.csv")

n_dna_cond <- length(intersect(dna_patients$Id, cond$PATIENT))
n_obs_cond <- length(intersect(observations$PATIENT, cond$PATIENT))
n_med_cond <- length(intersect(medications$PATIENT, cond$PATIENT))

cat("DNA ∩ Conditions:", n_dna_cond, "\n")
cat("Observations ∩ Conditions:", n_obs_cond, "\n")
cat("Medications ∩ Conditions:", n_med_cond, "\n")

data_list <- list(
  DNA = unique(as.character(dna_patients$Id)),
  Conditions = unique(as.character(cond$PATIENT)),
  Observations = unique(as.character(observations$PATIENT)),
  Medications = unique(as.character(medications$PATIENT))
)

# ----
# get num with CVD
library(Rdiagnosislist)
library(data.table)


cvd_parent <- SNOMEDconcept("56265001", SNOMED = SNOMED_env)  # "Cardiovascular disease (disorder)"

# get all descendants (including parent)
cvd_desc <- descendants(cvd_parent, SNOMED = SNOMED_env, include_self = TRUE)

# convert to plain vector of IDs (character or integer64)
cvd_ids <- as.data.frame(cvd_desc)$conceptId

# your dataframe: assume it’s `df` with columns patient_id & condition_code
df[, has_CVD := condition_code %in% cvd_ids]

# then count per patient
result <- df[, .(any_CVD = any(has_CVD)), by = patient_id]
num_with_CVD <- sum(result$any_CVD)



