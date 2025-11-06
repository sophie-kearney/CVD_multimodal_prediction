library(tidyverse)

###
# LOAD DATA
###

conditions <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/conditions.csv")
cvd_conditions <- read.csv("/Users/sophiekk/projects/CIS5200_final/data/codes_cvd.csv") %>%
  select(-c(n, DESCRIPTION))
encounters <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/encounters.csv")
observations <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/observations.csv")

conditions <- inner_join(conditions, cvd_conditions, by = c("CODE"))


###
# PROCESS CONDITONS
###

# find first CVD
first_cvd <- conditions %>%
  filter(CVD == 1) %>%
  group_by(PATIENT) %>%
  summarise(first_cvd_date = min(as.Date(START)))

###
# PROCESS ENCOUNTERS
###

# add number of visits since first visit
encounters <- encounters %>%
  arrange(PATIENT, as.Date(START)) %>%
  group_by(PATIENT) %>%
  mutate(visit_number = row_number()) %>%
  ungroup()

ehr_timing <- encounters %>%
  inner_join(first_cvd, by = "PATIENT") %>%
  mutate(days_before_cvd = as.numeric(as.Date(first_cvd_date) - as.Date(START))) %>%
  filter(days_before_cvd > 0)

patient_summary <- ehr_timing %>%
  group_by(PATIENT) %>%
  summarise(total_days_before = max(days_before_cvd),
            num_visits_before = n())

patient_summary %>%
  summarise(
    mean_days_before = mean(total_days_before, na.rm = TRUE),
    median_days_before = median(total_days_before, na.rm = TRUE),
    mean_visits_before = mean(num_visits_before, na.rm = TRUE),
    median_visits_before = median(num_visits_before, na.rm = TRUE)
  )


ggplot(patient_summary, aes(x = total_days_before)) +
  geom_histogram(bins = 50, fill = "steelblue", color="navy") +
  labs(x = "Days of Encounters before first CVD event",
       y = "Number of patients") +
  theme_bw()

ggplot(patient_summary, aes(x = num_visits_before)) +
  geom_histogram(bins = 50, fill = "seagreen", color="darkgreen") +
  labs(x = "Number of encounters before first CVD event",
       y = "Number of patients") +
  theme_bw() + 
  xlim(0, 250)

###
# ANY NON-CVD PATIENTS?
###

cvd_patients <- unique(conditions$PATIENT[conditions$CVD == 1])

# all patients in the dataset (from encounters)
all_patients <- unique(encounters$PATIENT)

# patients without any CVD events
non_cvd_patients <- setdiff(all_patients, cvd_patients)

# count and preview
length(non_cvd_patients)
length(cvd_patients)
head(non_cvd_patients)

###
# PROCESS OBSERVATIONS
###

# add number of visits since first visit
observations <- observations %>%
  arrange(PATIENT, as.Date(DATE)) %>%
  group_by(PATIENT) %>%
  mutate(obs_number = row_number()) %>%
  ungroup()

obs_timing <- observations %>%
  inner_join(first_cvd, by = "PATIENT") %>%
  mutate(days_before_cvd = as.numeric(as.Date(first_cvd_date) - as.Date(DATE))) %>%
  filter(days_before_cvd > 0)

patient_summary <- obs_timing %>%
  group_by(PATIENT) %>%
  summarise(total_days_before = max(days_before_cvd),
            num_visits_before = n())

ggplot(patient_summary, aes(x = total_days_before)) +
  geom_histogram(bins = 50, fill = "steelblue", color="navy") +
  labs(x = "Days of Observation data before first CVD event",
       y = "Number of patients") +
  theme_bw()

ggplot(patient_summary, aes(x = num_visits_before)) +
  geom_histogram(bins = 50, fill = "seagreen", color="darkgreen") +
  labs(x = "Number of obersvations before first CVD event",
       y = "Number of patients") +
  theme_bw()

patient_summary %>%
  summarise(
    mean_days_before = mean(total_days_before, na.rm = TRUE),
    median_days_before = median(total_days_before, na.rm = TRUE),
    mean_visits_before = mean(num_visits_before, na.rm = TRUE),
    median_visits_before = median(num_visits_before, na.rm = TRUE)
  )

# --------

cvd_patients <- unique(first_cvd$PATIENT)
observations <- observations %>%
  mutate(CVD_status = if_else(PATIENT %in% cvd_patients, "CVD", "Non-CVD"))

observations <- observations %>%
  mutate(DATE = as.Date(DATE)) %>%
  group_by(PATIENT) %>%
  mutate(first_obs = min(DATE)) %>%
  ungroup()

observations_4y <- observations %>%
  filter(DATE <= first_obs + years(4))

patient_obs_counts <- observations_4y %>%
  group_by(PATIENT, CVD_status) %>%
  summarise(total_obs = n(), .groups = "drop")

ggplot(patient_obs_counts, aes(x = total_obs, fill = CVD_status)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("CVD" = "firebrick", "Non-CVD" = "steelblue")) +
  labs(
    x = "Number of observations per patient (first 4 years)",
    y = "Number of patients",
    fill = "Patient Type"
  ) +
  theme_bw() +
  xlim(0,1000)





