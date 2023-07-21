majority_vote <- function(df) {
  # Calculate the table of frequencies for each row
  freq_table <- table(df)
  freq_table <- as.data.frame(freq_table)
  colnames(freq_table) <- c('label','freq')
  freq_table <- freq_table %>%
    arrange(desc(freq))
  majority <- freq_table$label[1]
  return(majority)
}