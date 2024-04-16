library(tidyverse)
library(readxl)
library(Boruta)
library(caret)
library(pROC)
library(e1071)
library(pheatmap)

read_tsv <- function(file, ...) {
  read_delim(file, ..., delim = "\t")
}

ggplot_my_theme <- theme(
  axis.text = element_text(size = 20),
  axis.title = element_text(size = 20)
)

ggplot() +
  stat_function(fun = function(x) exp(-x^2 / 1000), geom = "line") +
  xlim(-10, 10) +
  labs(x = "x", y = "K") +
  ggplot_my_theme

ggsave("presentation/figures/gaussian.png")

normalize_logical <- function(input) {
  res <- case_when(input,
    "Yes" = TRUE,
    "yes" = TRUE,
    "Y" = TRUE,
    "y" = TRUE,
    "No" = FALSE,
    "no" = FALSE,
    "N" = FALSE,
    "n" = FALSE
  )
  res
}

beataml_clinical <-
  read_excel("beataml2.0_data/beataml_wv1to4_clinical.xlsx")
beataml_sample_mapping <-
  read_excel("beataml2.0_data/beataml_waves1to4_sample_mapping.xlsx") %>%
  mutate(patientId = na_if(patientId, 0))

beataml_counts <-
  read_tsv("beataml2.0_data/beataml_waves1to4_counts_dbgap.txt")

beataml_counts_t <-
  beataml_counts %>%
  select(-c(display_label, description, biotype)) %>%
  pivot_longer(-stable_id, names_to = "sample", values_to = "z") %>%
  pivot_wider(names_from = stable_id, values_from = "z") %>%
  mutate(across(!c(sample),
    \(x) scale(x)[, 1],
    .unpack = TRUE
  ))

svm_count_input <- beataml_clinical %>%
  filter(!is.na(dbgap_rnaseq_sample)) %>%
  filter(vitalStatus == "Dead" | overallSurvival > 365) %>%
  select(dbgap_rnaseq_sample, overallSurvival) %>%
  inner_join(
    beataml_counts_t,
    by = c("dbgap_rnaseq_sample" = "sample"), keep = FALSE
  ) %>%
  mutate(survived = overallSurvival > 365, .before = 1) %>%
  select(-c(dbgap_rnaseq_sample, overallSurvival))

write_csv(svm_count_input, "svm_data/svm_input.csv")

set.seed(111)
aml_boruta <-
  Boruta(survived ~ ., data = svm_count_input, doTrace = 2, maxRuns = 1000)
print(aml_boruta)

aml_boruta_fix <- TentativeRoughFix(aml_boruta)
svm_input_subset <- svm_count_input %>%
  select(
    survived,
    names(aml_boruta_fix$finalDecision[aml_boruta_fix$finalDecision == "Confirmed"])
  )

svm_input_subset %>%
  write_csv("svm_boruta_input.csv")
