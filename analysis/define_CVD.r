library(tidyverse)

cond <- read.csv("/Users/sophiekk/projects/CIS5200_final/coherent-11-07-2022/csv/conditions.csv")

### 
# DEFINE CVD
###

chd2 <- read.csv("/Users/sophiekk/Downloads/opensafely-chronic-cardiac-disease-snomed-2020-04-08.csv")
chd1 <- read.csv("/Users/sophiekk/Downloads/primis-covid19-vacc-uptake-chd_cov-v2.5.csv")
 
cvd_codes <- union(chd1$code, chd2$id)

###
# check in ours
###

conditions <- cond %>%
  group_by(CODE, DESCRIPTION) %>%
  summarise(n = n(), .groups = "drop") %>%
  mutate(CVD = if_else(CODE %in% cvd_codes, 1, 0))

write.csv(conditions, "/Users/sophiekk/projects/CIS5200_final/data/codes_cvd.csv", row.names=FALSE)


###
# myocardial infarction
###

hist_ids <- cond %>% filter(DESCRIPTION == "History of myocardial infarction (situation)") %>% pull(PATIENT)
mi_ids <- cond %>% filter(DESCRIPTION == "Myocardial Infarction") %>% pull(PATIENT)

length(intersect(hist_ids, mi_ids))     # number of patients with both
length(setdiff(hist_ids, mi_ids))       # patients with history only
length(setdiff(mi_ids, hist_ids))       # patients with MI only



