library(tidyverse)

###
# LOAD DATA
###

observations <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/observations.csv")
conditions <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/conditions.csv")
cvd_conditions <- read.csv("/Users/sophiekk/projects/CIS5200_final/data/codes_cvd.csv") %>%
  select(-c(n, DESCRIPTION))
conditions <- inner_join(conditions, cvd_conditions, by = c("CODE"))

# find first CVD
first_cvd <- conditions %>%
  filter(CVD == 1) %>%
  group_by(PATIENT) %>%
  summarise(first_cvd_date = min(as.Date(START)))

# add y label for CVD
observations <- observations %>%
  mutate(CVD = if_else(PATIENT %in% unique(first_cvd$PATIENT), 1, 0))

###
# SELECT FEATURES
###

chosen_features <- read.csv("/Users/sophiekk/projects/CIS5200_final/data/feature_selection.csv") %>%
  filter(`keep.`=="Y")

observations_parsed <- observations %>%
  # keep only features we selected
  filter(DESCRIPTION %in% chosen_features$DESCRIPTION) %>%
  mutate(DATE = as.Date(DATE)) %>%
  group_by(PATIENT) %>%
  mutate(first_obs_date = min(DATE, na.rm = TRUE)) %>%
  # keep only first 4 years of observations
  filter(DATE <= first_obs_date + years(4)) %>%
  ungroup()

# pivot observations table to have all features in one row for each patient, data
observations_wide <- observations_parsed %>%
  group_by(PATIENT, DATE, CODE, CVD) %>%
  summarise(VALUE = mean(as.numeric(VALUE), na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(
    names_from = CODE,
    values_from = VALUE
  ) %>%
  arrange(PATIENT, DATE) %>%
  mutate(across(-c(PATIENT, CVD, DATE), ~as.numeric(.x))) %>%
  select(where(~ !all(is.na(.x))))

write.csv(observations_wide, "/Users/sophiekk/projects/CIS5200_final/data/observations_wide.csv", row.names = FALSE)

observation_description_key <- observations_parsed %>%
  select(DESCRIPTION, CODE) %>%
  unique()

write.csv(observation_description_key,  "/Users/sophiekk/projects/CIS5200_final/data/observations_descriptions_code_map.csv", row.names = FALSE)


###
# X_agg
###

# collect sd, min, max, median of each feature across all 4 years of data per patient
X_agg <- observations_wide %>%
  group_by(PATIENT, CVD) %>%
  summarise(across(
    -DATE,
    list(
      sd = ~if (all(is.na(.x))) NA_real_ else sd(.x, na.rm = TRUE),
      min = ~if (all(is.na(.x))) NA_real_ else min(.x, na.rm = TRUE),
      max = ~if (all(is.na(.x))) NA_real_ else max(.x, na.rm = TRUE),
      median = ~if (all(is.na(.x))) NA_real_ else median(.x, na.rm = TRUE)
    ),
    .names = "{col}_{fn}"
  ), .groups = "drop")

# for sd columns, if there is only one row across 4 years, SD is NA but we want it to be 0
for (col in unique(gsub("_(sd|min|max|median)$", "", names(X_agg)))) {
  sd_col <- paste0(col, "_sd")
  min_col <- paste0(col, "_min")
  if (sd_col %in% names(X_agg) && min_col %in% names(X_agg)) {
    X_agg[[sd_col]][is.na(X_agg[[sd_col]]) & !is.na(X_agg[[min_col]])] <- 0
  }
}

# do population level median imputation
X_agg <- X_agg %>%
  mutate(across(
    where(is.numeric),
    ~ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)
  ))

write.csv(X_agg, "/Users/sophiekk/projects/CIS5200_final/data/X_agg.csv", row.names = FALSE)

