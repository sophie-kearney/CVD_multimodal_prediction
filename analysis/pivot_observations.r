###
# LOAD DATA
###

observations <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/observations.csv")
conditions <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/conditions.csv")
cvd_conditions <- read.csv("/Users/sophiekk/projects/CIS5200_final/data/codes_cvd.csv") %>%
  select(-c(n, DESCRIPTION))
conditions <- inner_join(conditions, cvd_conditions, by = c("CODE"))

View(as.data.frame(unique(observations$DESCRIPTION)))

###
# How many patients for each type of observation?
###

obs_summary <- observations %>%
  group_by(DESCRIPTION) %>%
  summarise(num_patients = n_distinct(PATIENT)) %>%
  arrange(desc(num_patients))

ggplot(obs_summary, aes(x = num_patients)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "navy") +
  scale_x_continuous(labels = scales::comma) +
  labs(
    x = "Number of patients with a given observation type",
    y = "Frequency"
  ) +
  theme_bw()

###
# PIVOT OBSERVATION TABLE
###

observations_wide <- observations %>%
  mutate(
    DATE = as.Date(DATE),  # keep only day-month-year
  ) %>%
  group_by(PATIENT, DATE, DESCRIPTION) %>%
  summarise(VALUE = mean(VALUE, na.rm = TRUE), .groups = "drop") %>%  # average duplicates if same-day same-type
  pivot_wider(
    names_from = DESCRIPTION,
    values_from = VALUE
  ) %>%
  arrange(PATIENT, DATE)

patient_ids <- unique(observations$PATIENT)
batches <- split(patient_ids, ceiling(seq_along(patient_ids) / 500))  # 500 patients per batch

# Take just the first batch
first_batch_patients <- batches[[1]]

# Process only that batch
observations_wide_batch1 <- observations %>%
  filter(PATIENT %in% first_batch_patients) %>%
  mutate(DATE = as.Date(DATE)) %>%
  group_by(PATIENT, DATE, DESCRIPTION) %>%
  summarise(VALUE = mean(as.numeric(VALUE), na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = DESCRIPTION, values_from = VALUE) %>%
  arrange(PATIENT, DATE)

# Check result
dim(observations_wide_batch1)
head(observations_wide_batch1)



