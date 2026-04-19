suppressMessages(library(tidyverse))
suppressMessages(library(stringi))
suppressMessages(library(tidytext))

setwd("/Users/RheaMacBook/Desktop/information-density/human_subject_experiment/analysis")

df_import_pref = read_csv("../data/data_preference.csv", show_col_types=FALSE) %>% mutate(task_cond = "preference")
df_import_inf = read_csv("../data/data_informativity.csv", show_col_types=FALSE) %>% mutate(task_cond = "informativity")
df_import = df_import_pref %>% rbind(df_import_inf)

df = df_import %>%
  select(selected_descr_condition, nonselected_descr_condition, task_cond, anon_id, selected_descr_position, descr_left_condition, descr_right, descr_left, descr_right_condition, descr_left_condition, picture, trial_number) %>%
  mutate(caption_pair = case_when(
    str_detect(selected_descr_condition, "(coco|verbose)") & str_detect(nonselected_descr_condition, "(coco|verbose)") ~ "coco_verbose",
    str_detect(selected_descr_condition, "(coco|composite)") & str_detect(nonselected_descr_condition, "(coco|composite)") ~ "coco_composite",
    str_detect(selected_descr_condition, "(coco|gpt4o)") & str_detect(nonselected_descr_condition, "(coco|gpt4o)") ~ "coco_gpt4o",
    str_detect(selected_descr_condition, "(verbose|gpt4o)") & str_detect(nonselected_descr_condition, "(verbose|gpt4o)") ~ "verbose_gpt4o",
    str_detect(selected_descr_condition, "(verbose|composite)") & str_detect(nonselected_descr_condition, "(verbose|composite)") ~ "verbose_composite",
    str_detect(selected_descr_condition, "(gpt4o|composite)") & str_detect(nonselected_descr_condition, "(gpt4o|composite)") ~ "gpt4o_composite",
    TRUE ~ "FIRE"
  )) %>%
  mutate(descr_right_length = str_length(descr_right),
         descr_left_length = str_length(descr_left))

count_syllables <- function(word) {
  word <- tolower(word)
  word <- gsub("[^a-z]", "", word)
  if (nchar(word) == 0) return(1)
  vowels <- gregexpr("[aeiouy]+", word)[[1]]
  if (vowels[1] == -1) return(1)
  count <- length(vowels)
  if (grepl("e$", word) & count > 1) count <- count - 1
  return(max(count, 1))
}

flesch_kincaid <- function(text) {
  if (is.na(text) || text == "") return(NA_real_)
  sentences <- max(stri_count_regex(text, "[.!?]+"), 1)
  words_vec <- unlist(strsplit(gsub("[^a-zA-Z ]", "", tolower(text)), "\\s+"))
  words_vec <- words_vec[words_vec != ""]
  n_words <- length(words_vec)
  if (n_words == 0) return(NA_real_)
  n_syllables <- sum(sapply(words_vec, count_syllables))
  0.39 * (n_words / sentences) + 11.8 * (n_syllables / n_words) - 15.59
}

avg_word_freq <- function(text, freq_table, stop_words_vec) {
  if (is.na(text) || text == "") return(NA_real_)
  words_vec <- unlist(strsplit(gsub("[^a-zA-Z ]", "", tolower(text)), "\\s+"))
  words_vec <- words_vec[words_vec != "" & !(words_vec %in% stop_words_vec)]
  if (length(words_vec) == 0) return(NA_real_)
  freqs <- freq_table$n[match(words_vec, freq_table$word)]
  freqs[is.na(freqs)] <- 1
  mean(log(freqs))
}

stop_words_vec <- tidytext::stop_words$word

all_captions <- c(df$descr_left, df$descr_right)
freq_table <- tibble(text = all_captions) %>%
  unnest_tokens(word, text) %>%
  filter(!(word %in% stop_words_vec)) %>%
  count(word, sort = TRUE)

cat("Computing features...\n")
df <- df %>%
  mutate(
    descr_left_fk = sapply(descr_left, flesch_kincaid),
    descr_right_fk = sapply(descr_right, flesch_kincaid),
    descr_left_wf = sapply(descr_left, avg_word_freq, freq_table, stop_words_vec),
    descr_right_wf = sapply(descr_right, avg_word_freq, freq_table, stop_words_vec)
  )
cat("Features computed.\n")

df_stats = df %>%
  gather(location, caption_cond, descr_left_condition, descr_right_condition) %>%
  mutate(chosen = selected_descr_condition == caption_cond) %>%
  mutate(length = ifelse(location == "descr_left_condition", descr_left_length, descr_right_length)) %>%
  mutate(fk = ifelse(location == "descr_left_condition", descr_left_fk, descr_right_fk)) %>%
  mutate(wf = ifelse(location == "descr_left_condition", descr_left_wf, descr_right_wf)) %>%
  select(task_cond, caption_pair, chosen, caption_cond, length, fk, wf, anon_id, picture)

# Descriptive stats
cat("\n=== Descriptive Stats by Condition ===\n")
df_stats %>%
  group_by(caption_cond) %>%
  summarize(mean_fk = mean(fk, na.rm=TRUE), sd_fk = sd(fk, na.rm=TRUE),
            mean_wf = mean(wf, na.rm=TRUE), sd_wf = sd(wf, na.rm=TRUE),
            mean_len = mean(length, na.rm=TRUE)) %>%
  as.data.frame() %>%
  print()

pairs <- c("coco_composite", "coco_verbose", "verbose_composite")
study_types <- c("preference", "informativity")

for (st in study_types) {
  cat("\n\n========================================\n")
  cat("STUDY TYPE:", toupper(st), "\n")
  cat("========================================\n")

  for (pair in pairs) {
    sub <- df_stats %>% filter(caption_pair == pair, task_cond == st)

    cat("\n---", pair, ": caption_cond + length + wf ---\n")
    m <- glm(chosen ~ caption_cond + length + wf, sub, family = binomial(link = "logit"))
    print(round(summary(m)$coefficients, 6))

    cat("\n---", pair, ": caption_cond + length + fk ---\n")
    m <- glm(chosen ~ caption_cond + length + fk, sub, family = binomial(link = "logit"))
    print(round(summary(m)$coefficients, 6))

    cat("\n---", pair, ": caption_cond + length + wf + fk ---\n")
    m <- glm(chosen ~ caption_cond + length + wf + fk, sub, family = binomial(link = "logit"))
    print(round(summary(m)$coefficients, 6))
  }
}
