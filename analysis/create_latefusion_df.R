library(tidyverse)

x_agg <- read.csv("/Users/sophiekk/projects/CIS5200_final/data/tabPFN_logits/logits_Xagg_tabPFN_lfscores.csv") %>%
  select(PATIENT, prob_class_1) %>%
  rename(EHR_CVD_score = prob_class_1)
x_temp <- read.csv("/Users/sophiekk/projects/CIS5200_final/data/tabPFN_logits/logits_Xtemp_tabPFN_lfscores.csv") %>%
  select(PATIENT, prob_class_1) %>%
  rename(EHR_CVD_score = prob_class_1)
prs <- read.csv("/Users/sophiekk/projects/CIS5200_final/data/prs_scores.csv") %>%
  select(PATIENT_ID, `PRS_1e.5`

scores_x_agg <- merge(x_agg, prs, by.x = "PATIENT", by.y = "PATIENT_ID")
scores_x_temp <- merge(x_agg, prs, by.x = "PATIENT", by.y = "PATIENT_ID")

write.csv(scores_x_temp, "/Users/sophiekk/projects/CIS5200_final/data/scores_x_temp.csv", row.names = FALSE)
write.csv(scores_x_agg, "/Users/sophiekk/projects/CIS5200_final/data/scores_x_agg.csv", row.names = FALSE)
