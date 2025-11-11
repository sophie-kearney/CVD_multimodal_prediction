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

# create missingness indicators
for (col in unique(gsub("_(sd|min|max|median)$", "", names(X_agg)))) {
  median_col <- paste0(col, "_median")
  if (median_col %in% names(X_agg)) {
    indicator_col <- paste0(col, "_missing")
    X_agg[[indicator_col]] <- ifelse(is.na(X_agg[[median_col]]), 1, 0)
  }
}
# reorder once: place each _missing column right after its _median
col_order <- names(X_agg)
for (col in unique(gsub("_(sd|min|max|median)$", "", names(X_agg)))) {
  median_col <- paste0(col, "_median")
  indicator_col <- paste0(col, "_missing")
  if (median_col %in% col_order && indicator_col %in% col_order) {
    insert_after <- which(col_order == median_col)
    # remove indicator if it exists already (so we don't duplicate)
    col_order <- c(setdiff(col_order, indicator_col))
    # insert indicator right after median
    col_order <- append(col_order, indicator_col, after = insert_after)
  }
}
X_agg <- X_agg[, col_order]

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

###
# Xtemp
###

X_temp <- observations_wide %>%
  group_by(PATIENT) %>%
  mutate(
    start_date = min(DATE, na.rm = TRUE),
    # define 6-month slices instead of yearly
    slice = as.integer(floor(as.numeric(difftime(DATE, start_date, units = "days")) / 180)) + 1
  ) %>%
  filter(slice <= 8) %>%
  ungroup() %>%
  group_by(PATIENT, CVD, slice) %>%
  summarise(
    across(
      -c(DATE, start_date),
      list(
        sd = ~if (all(is.na(.x))) NA_real_ else sd(.x, na.rm = TRUE),
        min = ~if (all(is.na(.x))) NA_real_ else min(.x, na.rm = TRUE),
        max = ~if (all(is.na(.x))) NA_real_ else max(.x, na.rm = TRUE),
        median = ~if (all(is.na(.x))) NA_real_ else median(.x, na.rm = TRUE)
      ),
      .names = "{col}_{fn}"
    ),
    .groups = "drop"
  )

# ensure every patient has 8 slices (4 years / 0.5 = 8)
all_slices <- expand.grid(
  PATIENT = unique(X_temp$PATIENT),
  slice = 1:8
)
X_temp <- all_slices %>%
  left_join(X_temp, by = c("PATIENT", "slice")) %>%
  arrange(PATIENT, slice)

# add missingness indicator for each feature within each patient-slice
for (col in unique(gsub("_(sd|min|max|median)$", "", names(X_temp)))) {
  median_col <- paste0(col, "_median")
  if (median_col %in% names(X_temp)) {
    indicator_col <- paste0(col, "_missing")
    X_temp[[indicator_col]] <- ifelse(is.na(X_temp[[median_col]]), 1, 0)
  }
}
# reorder indicators right after the median column
col_order <- names(X_temp)
for (col in unique(gsub("_(sd|min|max|median)$", "", names(X_temp)))) {
  median_col <- paste0(col, "_median")
  indicator_col <- paste0(col, "_missing")
  if (median_col %in% col_order && indicator_col %in% col_order) {
    insert_after <- which(col_order == median_col)
    col_order <- c(setdiff(col_order, indicator_col))
    col_order <- append(col_order, indicator_col, after = insert_after)
  }
}
X_temp <- X_temp[, col_order]

# for sd columns, if there is only one row across 4 years, SD is NA but we want it to be 0
for (col in unique(gsub("_(sd|min|max|median)$", "", names(X_temp)))) {
  sd_col <- paste0(col, "_sd")
  min_col <- paste0(col, "_min")
  if (sd_col %in% names(X_temp) && min_col %in% names(X_temp)) {
    X_temp[[sd_col]][is.na(X_temp[[sd_col]]) & !is.na(X_temp[[min_col]])] <- 0
  }
}

# if an entire 6-month slide is missing, fill that slice in from the previous slice
X_temp <- X_temp %>%
  arrange(PATIENT, slice) %>%
  group_by(PATIENT) %>%
  fill(everything(), .direction = "down") %>%
  ungroup()

# if for a patient they never have a feature, fill in with population level median
feature_medians <- X_temp %>%
  select(-PATIENT, -slice, -CVD) %>%
  summarise(across(everything(), ~median(.x, na.rm = TRUE)))
for (col in names(feature_medians)) {
  X_temp[[col]] <- ifelse(
    ave(X_temp[[col]], X_temp$PATIENT, FUN = function(x) all(is.na(x))),
    feature_medians[[col]],
    X_temp[[col]]
  )
}

# if a patient has a single feature for a slice missing, fill from the nearest non-missing slice
X_temp <- X_temp %>%
  arrange(PATIENT, slice) %>%
  group_by(PATIENT) %>%
  fill(everything(), .direction = "downup") %>%
  ungroup()

write.csv(X_temp, "/Users/sophiekk/projects/CIS5200_final/data/X_temp.csv", row.names = FALSE)

###
# flatten X_temp
###

X_temp_flat <- X_temp %>%
  # melt slice data into long format
  pivot_longer(
    cols = -c(PATIENT, CVD, slice),
    names_to = "feature",
    values_to = "value"
  ) %>%
  # combine feature name with slice number
  mutate(feature_slice = paste0(feature, "_slice", slice)) %>%
  select(PATIENT, CVD, feature_slice, value) %>%
  # pivot back to wide format
  pivot_wider(
    names_from = feature_slice,
    values_from = value
  )

write.csv(X_temp_flat, "/Users/sophiekk/projects/CIS5200_final/data/X_temp_flat.csv", row.names = FALSE)

###
# EARLY FUSION
###

genetic <- read.csv("/Users/sophiekk/projects/CIS5200_final/data/genotype_matrix.csv")

X_agg_early <- merge(X_agg, genetic, by.x="PATIENT", by.y="PATIENT_ID")
X_temp_early <- merge(X_temp_flat, genetic, by.x="PATIENT", by.y="PATIENT_ID")

write.csv(X_agg_early, "/Users/sophiekk/projects/CIS5200_final/data/X_agg_earlyfusion.csv", row.names = FALSE)
write.csv(X_temp_early, "/Users/sophiekk/projects/CIS5200_final/data/X_temp_earlyfusion.csv", row.names = FALSE)


