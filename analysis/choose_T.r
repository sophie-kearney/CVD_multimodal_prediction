library(tidyverse)

###
# LOAD DATA
###

conditions <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/conditions.csv")
cvd_conditions <- read.csv("/Users/sophiekk/projects/CIS5200_final/data/codes_cvd.csv") %>%
  select(-c(n, DESCRIPTION))
encounters <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/encounters.csv")

conditions <- inner_join(conditions, cvd_conditions, by = c("CODE"))

# add number of visits since first visit
encounters <- encounters %>%
  arrange(PATIENT, as.Date(START)) %>%
  group_by(PATIENT) %>%
  mutate(visit_number = row_number()) %>%
  ungroup()

###
# PROCESS DATA
###

# find first CVD

first_cvd <- conditions %>%
  filter(CVD == 1) %>%
  group_by(PATIENT) %>%
  summarise(first_cvd_date = min(as.Date(START)))

ehr_timing <- encounters %>%
  inner_join(first_cvd, by = "PATIENT") %>%
  mutate(days_before_cvd = as.numeric(as.Date(first_cvd_date) - as.Date(START))) %>%
  filter(days_before_cvd > 0)

patient_summary <- ehr_timing %>%
  group_by(PATIENT) %>%
  summarise(total_days_before = max(days_before_cvd),
            num_visits_before = n())

# ggplot(patient_summary, aes(x = total_days_before)) +
#   geom_histogram(bins = 50, fill = "steelblue", color="navy") +
#   labs(x = "Days of EHR data before first CVD event",
#        y = "Number of patients") +
#   theme_bw()

ggplot(patient_summary, aes(x = num_visits_before)) +
  geom_histogram(bins = 50, fill = "seagreen", color="darkgreen") +
  labs(x = "Number of encounters before first CVD event",
       y = "Number of patients") +
  theme_bw() + 
  xlim(0, 250)







