library(tidyverse)

files <- list.files("/Users/sophiekk/projects/CIS5200_final/data/results", pattern = "*.csv", full.names = TRUE)

results <- lapply(files, function(f) {
  read_csv(f)
}) %>%
  bind_rows() %>%
  mutate(
    Data = case_when(
      str_detect(data_type, "agg") ~ "X_agg",
      TRUE ~ "X_temp"
    ),
    Fusion = case_when(
      str_detect(data_type, "ef") ~ "Early Fusion",
      TRUE ~ "None"
    )
  )

ggplot(results, aes(model,f1,fill=model)) + 
  facet_grid(Fusion~Data) +
  geom_bar(stat = "identity", position = position_dodge(),color="grey60",linewidth=0.2, width=0.5) +
  geom_text(aes(label=sprintf("%.2f", f1)),
            position=position_dodge(width=0.8), 
            vjust=-0.3, 
            size=3,
            color="gray30") +
  ylim(0,1) +
  theme_bw() +
  theme(legend.position="bottom") +
  scale_fill_brewer(palette = "YlGnBu") + 
  labs(fill=NULL, x="Model", y="F1 Score")

ggplot(results, aes(model,AUROC,fill=model)) + 
  facet_grid(Fusion~Data) +
  geom_bar(stat = "identity", position = position_dodge(),color="grey60",linewidth=0.2, width=0.5) +
  geom_text(aes(label=sprintf("%.2f", AUROC)),
            position=position_dodge(width=0.8), 
            vjust=-0.3, 
            size=3,
            color="gray30") +
  ylim(0,1) +
  theme_bw() +
  theme(legend.position="bottom") +
  scale_fill_brewer(palette = "YlGnBu") + 
  labs(fill=NULL, x="Model", y="AUROC")

ggplot(results, aes(model,balanced_accuracy,fill=model)) + 
  facet_grid(Fusion~Data) +
  geom_bar(stat = "identity", position = position_dodge(),color="grey60",linewidth=0.2, width=0.5) +
  geom_text(aes(label=sprintf("%.2f", balanced_accuracy)),
            position=position_dodge(width=0.8), 
            vjust=-0.3, 
            size=3,
            color="gray30") +
  ylim(0,1) +
  theme_bw() +
  theme(legend.position="bottom") +
  scale_fill_brewer(palette = "YlGnBu") + 
  labs(fill=NULL, x="Model", y="Balanced Accuracy")
