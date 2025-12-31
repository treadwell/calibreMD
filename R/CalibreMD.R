#' Setup required packages for CalibreMD
#' @export
setup_packages <- function() {
  library(dplyr, quietly = TRUE, warn.conflicts = FALSE)
  invisible(lapply(c(
    "tidyr",
    "igraph",
    "stringr",
    "tidytext",
    "Matrix",
    "RSQLite",
    "DBI",
    "xgboost",
    "purrr"
  ), requireNamespace, quietly = TRUE))
}

#' Find the metadata database file in the given directory
#' @param dataDir The directory containing the Calibre library
#' @return The path to the metadata.db file
#' @export
find_md_db <- function(dataDir){
  if (!dir.exists(dataDir)) {
    stop(paste("Directory does not exist:", dataDir))
  }
  paste0(dataDir, '/metadata.db')
}

#' Load the EAV (Entity-Attribute-Value) data from the Calibre metadata database
#' @param dbname The path to the metadata.db file
#' @return A data frame containing the EAV data
#' @export
load_eav <- function(dbname){
  con <- DBI::dbConnect(RSQLite::SQLite(), dbname = dbname)
  result <- DBI::dbGetQuery(con, "
  	select id, 'title' as feature, title as value from books
  	union all
  	select book as id, 'comment' as feature, text as value from comments
  	union all
  	select al.book as id, 'author' as feature, a.name as value from books_authors_link al, authors a where al.author = a.id
  	union all
  	select pl.book as id, 'publisher' as feature, p.name as value from books_publishers_link pl, publishers p where pl.publisher = p.id
  	union all
  	select tl.book as id, 'tag' as feature, t.name as value from books_tags_link tl, tags t where tl.tag = t.id
  	union all
  	select sl.book as id, 'series' as feature, s.name as value from books_series_link sl, series s where sl.series = s.id
  	union all
  	select book as id, 'rating' as feature, rating as value from books_ratings_link
  	order by id;
  ")
  DBI::dbDisconnect(con)
  result
}

#' Explode text features into individual tokens
#' @param eav The EAV data frame
#' @return A data frame with text features exploded into tokens
#' @export
explode_text_features <- function(eav){
  text_feats <- c("title", "comment") # features to tokenize
  
  ## capture un-tokenized titles
  original_titles <- eav %>% 
    dplyr::filter(feature == "title") %>%  # only the title rows
    dplyr::mutate(feature = "title_original") # rename the attribute
  
  text_values <- eav %>%
    filter(feature %in% text_feats) %>% 
    select(value) %>%
    unlist(use.names = FALSE) %>% # flatten to character vector
    as.character()
  
  train <- tibble(text = text_values)
  
  # learn: build the unigram vocabulary
  vocab <- train %>%
    tidytext::unnest_tokens(word, text, token = "words") %>% # split to lowercase words
    distinct(word) %>%
    pull(word) # this extracts the column as a character vector
  
  # apply: function to tokenize new text with that vocab
  tokenize <- function(txt) {
    tibble(text = txt) %>%
      tidytext::unnest_tokens(word, text, token = "words") %>% # split/clean
      filter(word %in% vocab) %>% # keep known words
      distinct(word) %>% # "set" semantics
      pull(word)
  }
  
  # Tokenize and explode only the text features
  token_rows <- eav %>% # original EAV table
    filter(feature %in% text_feats) %>% # keep titles/comments
    mutate(tokens = purrr::map(value, tokenize)) %>% # list-column of tokens
    select(id, feature, tokens) %>%                  
    tidyr::unnest(tokens) %>% # one row per token
    rename(value = tokens) # align col-name
  
  # Recombine with untouched rows (as eav)
  dplyr::bind_rows(
    eav %>% 
      dplyr::filter(!feature %in% text_feats), # untouched rows
    original_titles, # full titles (title_original)
    token_rows) %>%  # exploded tokens
    dplyr::mutate(
      id = as.numeric(id),
      feature = as.character(feature),
      value = as.character(value)
    )
}

#' Explode tags
#' @param eav The EAV data frame
#' @return The EAV data frame with exploded tags
#' @export
explode_tags <- function(eav) {
  # Expand hierarchical tags and also emit individual segments (unigrams)
  tag_rows <- eav %>%
    dplyr::filter(feature == "tag") %>%
    dplyr::mutate(expanded = purrr::map(value, function(tag_str) {
      tag_str <- as.character(tag_str)
      parts <- unlist(strsplit(tag_str, "\\."))
      if (length(parts) == 0) return(character(0))
      # hierarchical prefixes: a, a.b, a.b.c
      prefixes <- vapply(seq_along(parts), function(k) paste(parts[1:k], collapse = "."), character(1))
      # individual segments: a, b, c
      segments <- unique(parts)
      unique(c(prefixes, segments))
    })) %>%
    dplyr::select(id, feature, expanded) %>%
    tidyr::unnest(expanded, keep_empty = TRUE) %>%
    dplyr::rename(value = expanded)

  dplyr::bind_rows(eav, tag_rows) %>%
    dplyr::mutate(
      id = as.numeric(id),
      feature = as.character(feature),
      value = as.character(value)
    ) %>%
    dplyr::distinct(id, feature, value, .keep_all = TRUE)
}

#' Get the count of tags per book
#' @param eav The EAV data frame
#' @return A data frame with book IDs and their tag counts
#' @export
get_tag_counts <- function(eav){
  eav %>% 
    filter(feature == "tag") %>%
    distinct(id, value) %>%
    count(id, name = "n_tags")
}

#' Get a summary of books with their titles and tag counts
#' @param eav The EAV data frame
#' @return A data frame with book IDs, titles, and tag counts
#' @export
get_book_summary <- function(eav){
  eav %>%
    filter(feature == "title_original") %>% 
    distinct(id, .keep_all = TRUE) %>% 
    transmute(id, title = value) %>% 
    full_join(get_tag_counts(eav), by = "id") %>%
    mutate(n_tags = coalesce(n_tags, 0L),
           title_length = nchar(title)) %>%
    arrange(desc(title_length))
}

#' Get a summary of tags and their book counts
#' @param eav The EAV data frame
#' @return A data frame with tags and their book counts
#' @export
get_tag_summary <- function(eav){
  eav %>% 
    filter(feature == "tag") %>%
    distinct(id, value) %>% 
    count(value, name = "book_count") %>% 
    arrange(desc(book_count))
}

#' Get the titles of all books
#' @param eav The EAV data frame
#' @return A data frame with book IDs and titles
#' @export
get_book_titles <- function(eav){
  titles <- eav %>%
    filter(feature == "title_original") %>%
    select(book_id = id, title = value) %>%
    distinct()
}

#' Get the existing tags for each book
#' @param eav The EAV data frame
#' @return A data frame with book IDs and their existing tags
#' @export
get_existing_tags <- function(eav) {
  existing_tags <- eav %>% 
    filter(feature == "tag") %>%                 
    mutate(book_id = id,
           existing_tag = value) %>% 
    select(book_id, existing_tag) %>% 
    distinct() %>%                               
    arrange(book_id)
}

#' Prepare the dataset for modeling
#' @param eav The EAV data frame
#' @return A list containing the feature matrix, tag matrix, and book features
#' @export
prep_dataset <- function(eav){
  book_tags <- eav %>% 
    filter(feature == "tag") %>% 
    mutate(tag = value) %>% 
    select(id, tag) %>% 
    distinct() %>%
    arrange(id)
  
  valid_tags <- book_tags %>% 
    group_by(tag) %>% 
    summarize(n_books = n_distinct(id)) %>% 
    filter(n_books >= 10) %>% 
    pull(tag)
  
  book_tags <- book_tags %>%
    filter(tag %in% valid_tags)
  
  book_features <- eav %>% 
    # filter(feature != "tag") %>%
    filter(!feature %in% c("tag", "title_original")) %>% 
    mutate(feature = paste(feature, value, sep = ": ")) %>% 
    select(id, feature) %>% 
    distinct() %>% 
    arrange(id)
  
  common_ids <- sort(intersect(book_tags$id, book_features$id))
  
  book_tags <- book_tags %>%
    filter(id %in% common_ids)
  
  book_features <- book_features %>%
    filter(id %in% common_ids)
  
  # Create index mappings for row (id) and columns (tag or feature)
  id_levels <- sort(unique(common_ids)) # ensures same order in both matrices
  
  # Sparse tag matrix
  tag_levels <- sort(unique(book_tags$tag))
  tag_matrix <- Matrix::sparseMatrix(
    i = match(book_tags$id, id_levels),
    j = match(book_tags$tag, tag_levels),
    x = 1L,
    dims = c(length(id_levels), length(tag_levels)),
    dimnames = list(id_levels, tag_levels)
  )
  
  # Sparse feature matrix
  feature_levels <- sort(unique(book_features$feature))
  feature_matrix <- Matrix::sparseMatrix(
    i = match(book_features$id, id_levels),
    j = match(book_features$feature, feature_levels),
    x = 1L,
    dims = c(length(id_levels), length(feature_levels)),
    dimnames = list(id_levels, feature_levels)
  )
  
  list(
    X = feature_matrix,
    Y = tag_matrix,
    book_features = book_features
  )
}

#' Train models for tag prediction
#' @param dataset A list containing feature matrix and tag matrix
#' @param progress Whether to log progress to the console
#' @param progress_every Report progress every N tags when progress is TRUE
#' @return A list of trained models
#' @export
train_models <- function(dataset, progress = FALSE, progress_every = 10){
  requireNamespace("xgboost", quietly = TRUE)
  X <- dataset$X
  Y <- dataset$Y
  models <- list()
  tag_names <- colnames(Y)
  n_tags <- length(tag_names)
  if (isTRUE(progress)) {
    if (is.na(progress_every) || progress_every < 1) {
      progress_every <- 1
    }
    message(sprintf("Training %d tag models...", n_tags))
  }
  for (i in seq_along(tag_names)) {
    tag <- tag_names[[i]]
    if (isTRUE(progress) && (i == 1 || i == n_tags || (i %% progress_every == 0))) {
      message(sprintf("Training tag %d/%d: %s", i, n_tags, tag))
    }
    # Extract label vector and convert to numeric
    y <- as.numeric(Y[, tag])
    # Construct DMatrix using sparse feature matrix
    dtrain <- xgboost::xgb.DMatrix(data = X, label = y)
    models[[tag]] <- xgboost::xgboost(
      data = dtrain,
      objective = "binary:logistic",
      eval_metric = "logloss",
      nrounds = 50,
      verbose = 0
    )
  }
  models
}

#' Predict tags using trained models
#' @param models List of trained models
#' @param dataset Dataset containing features
#' @param progress Whether to log progress to the console
#' @param progress_every Report progress every N tags when progress is TRUE
#' @return A data frame of predictions
#' @export
predict_tags <- function(models, dataset, progress = FALSE, progress_every = 25){
  requireNamespace("xgboost", quietly = TRUE)
  book_features <- dataset$book_features
  # Assume predict_ids and book_features are defined
  predict_ids <- unique(book_features$id)  # books to predict for
  
  # Add row_index to book_features dynamically
  book_features <- book_features %>%
    mutate(row_index = match(id, predict_ids))
  
  # Initialize predictions output
  pred_probs_new <- data.frame(book_id = predict_ids)
  
  # Loop over all models/tags
  tag_names <- names(models)
  n_tags <- length(tag_names)
  if (isTRUE(progress)) {
    if (is.na(progress_every) || progress_every < 1) {
      progress_every <- 1
    }
    message(sprintf("Predicting %d tag models...", n_tags))
  }
  for (i in seq_along(tag_names)) {
    tag <- tag_names[[i]]
    if (isTRUE(progress) && (i == 1 || i == n_tags || (i %% progress_every == 0))) {
      message(sprintf("Predicting tag %d/%d: %s", i, n_tags, tag))
    }
    model <- models[[tag]]
    used_features <- model$feature_names
    
    # Align features to model-specific ones
    filtered_book_features <- book_features %>%
      filter(feature %in% used_features) %>%
      mutate(col_index = match(feature, used_features)) %>%
      filter(!is.na(row_index) & !is.na(col_index))
    
    # Build sparse matrix
    new_feature_matrix <- Matrix::sparseMatrix(
      i = filtered_book_features$row_index,
      j = filtered_book_features$col_index,
      x = 1L,
      dims = c(length(predict_ids), length(used_features)),
      dimnames = list(NULL, used_features)
    )
    
    # Predict and store
    preds <- stats::predict(model, newdata = new_feature_matrix)
    pred_probs_new[[tag]] <- preds
  }
  
  pred_probs_new
}

#' Threshold predictions to get discrete labels
#' @param pred_probs Prediction probabilities
#' @param book_id Book ID to threshold
#' @param threshold Probability threshold
#' @return Character vector of predicted tags
#' @export
threshold_predictions <- function(pred_probs, book_id, threshold = 0.5) {
  my_row <- pred_probs %>% filter(as.character(book_id) == as.character(!!book_id))
  if (nrow(my_row) == 0) return(character(0))
  tag_cols <- setdiff(colnames(my_row), "book_id")
  tag_cols[as.numeric(my_row[1, tag_cols]) > threshold]
}

#' Get top N labels for a book
#' @param pred_probs Prediction probabilities
#' @param book_id Book ID to get labels for
#' @param n Number of top labels to return
#' @return Character vector of top N labels
#' @export
get_top_n_labels <- function(pred_probs, book_id, n = 2) {
  my_row <- pred_probs %>% filter(as.character(book_id) == as.character(!!book_id))
  if (nrow(my_row) == 0) return(character(0))
  tag_cols <- setdiff(colnames(my_row), "book_id")
  tag_scores <- as.numeric(my_row[1, tag_cols])
  names(tag_scores) <- tag_cols
  top_tags <- names(sort(tag_scores, decreasing = TRUE))[1:n]
  return(top_tags)
}

#' Get label probabilities in long format
#' @param predictions_probs Prediction probabilities
#' @return A data frame of probabilities in long format
#' @export
get_label_probabilities <- function(predictions_probs){
  pred_long_all <- predictions_probs %>%
    tidyr::pivot_longer(-book_id, names_to = "tag", values_to = "prob")
}

# Helper to load existing tags
load_existing_tags <- function(eav) {
  eav %>% 
    filter(feature == "tag") %>%                 
    mutate(tag = value) %>% 
    select(id, tag) %>% 
    distinct() %>%                               
    arrange(id)
}

# Helper to clean and join tags
clean_and_join_tags <- function(book_tags, pred_long_all) {
  book_tags_clean <- book_tags %>%
    rename(book_id = id)
  book_tags_clean %>%
    inner_join(pred_long_all, by = c("book_id", "tag"))
}

# Helper to join with titles
join_with_titles <- function(df, eav, tag_col) {
  titles <- get_book_titles(eav)
  df %>%
    left_join(titles, by = "book_id") %>%
    select(book_id, title, !!rlang::sym(tag_col))
}

# Helper to filter by threshold
filter_by_threshold <- function(pred_long_all, threshold, greater_than = TRUE) {
  if (greater_than) {
    pred_long_all %>% filter(prob > threshold)
  } else {
    pred_long_all %>% filter(prob < threshold)
  }
}

# Helper to filter descendant tags
filter_descendant_tags <- function(existing_tags, tag_name) {
  descendant_regex <- paste0("^", stringr::str_replace_all(tag_name, "[.]", "\\\\."), "\\.")
  existing_tags %>%
    filter(stringr::str_detect(existing_tag, descendant_regex)) %>%
    distinct(book_id)
}

#' Get recommended tags to add
#' @param eav EAV data frame
#' @param pred_long_all Predictions in long format
#' @param add_threshold Threshold for adding tags
#' @return A data frame of recommended tags to add
#' @export
get_recommended_add <- function(eav, pred_long_all, add_threshold) {
  pred_long_add <- filter_by_threshold(pred_long_all, add_threshold, greater_than = TRUE)
  
  # Load existing tags
  book_tags <- load_existing_tags(eav)
  existing_tags <- get_existing_tags(eav)
  
  # Remove predicted tags already present
  new_tags_df <- pred_long_add %>%
    anti_join(existing_tags, by = c("book_id", "tag" = "existing_tag"))
  
  # Remove general predicted tags if a more specific one exists
  filtered_tags_df <- new_tags_df %>%
    left_join(existing_tags, by = "book_id") %>%
    group_by(book_id, tag) %>%
    filter(!any(stringr::str_starts(existing_tag, paste0(tag, ".")))) %>%
    ungroup() %>%
    select(book_id, tag) %>%
    distinct()
  
  # Aggregate recommendations
  recommended_tags_df <- filtered_tags_df %>%
    group_by(book_id) %>%
    summarize(recommended_tags = paste(tag, collapse = ", "), .groups = "drop")
  
  # Join with titles
  join_with_titles(recommended_tags_df, eav, tag_col = "recommended_tags")
}

#' Get books to remove a specific tag from
#' @param eav EAV data frame
#' @param pred_long_all Predictions in long format
#' @param remove_threshold Threshold for removing tags
#' @return A data frame of recommended tags to remove
#' @export
get_recommended_remove <- function(eav, pred_long_all, remove_threshold = 0.05) {
  # Load existing tags
  book_tags <- load_existing_tags(eav)
  
  # Clean and join tags
  joined_tags <- clean_and_join_tags(book_tags, pred_long_all)
  
  # Filter by threshold
  tags_to_remove <- joined_tags %>%
    filter(prob < remove_threshold) %>%
    group_by(book_id) %>%
    summarize(tags_to_review = paste(tag, collapse = ", "), .groups = "drop")
  
  # Join with titles
  join_with_titles(tags_to_remove, eav, tag_col = "tags_to_review")
}

#' Get books to add a specific tag to
#' @param eav EAV data frame
#' @param predictions Predictions in long format
#' @param tag_name Tag to suggest additions for
#' @param prob_min Minimum probability threshold
#' @return A data frame of suggested tag additions
#' @export
suggest_tag_additions <- function(eav, predictions = pred_long_all, tag_name, prob_min = 0) {
  stopifnot(
    all(c("book_id", "tag", "prob") %in% names(predictions)),
    "prob_min must be in [0,1]" = prob_min >= 0 & prob_min <= 1
  )
  
  # Get book titles
  titles <- get_book_titles(eav)
  
  # Get existing tags
  existing_tags <- get_existing_tags(eav)
  
  # Filter descendant tags
  descendant_tags <- filter_descendant_tags(existing_tags, tag_name)
  
  predictions %>%
    filter(tag == tag_name, prob >= prob_min) %>%
    anti_join(existing_tags, by = c("book_id", "tag" = "existing_tag")) %>%
    anti_join(descendant_tags, by = "book_id") %>%
    left_join(titles, by = "book_id") %>%
    arrange(desc(prob)) %>%
    select(book_id, title, tag, prob)
}

#' Recommend new tags for a specific book
#' @param eav The EAV data frame
#' @param predictions_probs Prediction probabilities from predict_tags
#' @param book_id The ID of the book to get recommendations for
#' @param threshold Probability threshold for recommendations (default 0.8)
#' @return A data frame with columns book_id, tag, and prob
#' @export
recommend_tags_for_book <- function(eav, predictions_probs, book_id, threshold = 0.8) {
  # Validate book exists
  book_exists <- book_id %in% unique(eav$id)
  if (!book_exists) {
    stop(sprintf("Book ID %s not found in the dataset", book_id))
  }
  
  # Handle empty predictions case
  if (ncol(predictions_probs) <= 1) {
    return(tibble::tibble(
      book_id = numeric(0),
      tag = character(0),
      prob = numeric(0)
    ))
  }
  
  # Get existing tags using existing function
  existing_tags <- get_existing_tags(eav) %>%
    filter(book_id == !!book_id) %>%
    pull(existing_tag)
  
  # Get predictions and filter
  get_label_probabilities(predictions_probs) %>%
    filter(book_id == !!book_id,
           prob >= threshold,
           !tag %in% existing_tags) %>%
    arrange(desc(prob)) %>%
    select(book_id, tag, prob)  # Ensure consistent column order
}

validate_prob_threshold <- function(value, name) {
  if (is.na(value) || value < 0 || value > 1) {
    stop(name, " must be between 0 and 1")
  }
}

assert_results_db_tables <- function(con, tables) {
  missing <- tables[!vapply(tables, function(name) DBI::dbExistsTable(con, name), logical(1))]
  if (length(missing) > 0) {
    stop("Results DB is missing required tables: ", paste(missing, collapse = ", "))
  }
}

escape_like_pattern <- function(text) {
  gsub("([%_!])", "!\\1", text, perl = TRUE)
}

#' Write CalibreMD results to a SQLite database
#' @param db_path Path to the SQLite database to write
#' @param eav The EAV data frame
#' @param predictions_probs Prediction probabilities from predict_tags
#' @param pred_long_all Optional long-format predictions (book_id, tag, prob)
#' @param add_threshold Threshold for add recommendations
#' @param remove_threshold Threshold for remove recommendations
#' @param overwrite Whether to overwrite existing tables
#' @param data_dir Optional source library path to store in metadata
#' @return The db_path (invisibly)
#' @export
write_results_db <- function(db_path,
                             eav,
                             predictions_probs = NULL,
                             pred_long_all = NULL,
                             add_threshold = 0.75,
                             remove_threshold = 0.01,
                             overwrite = TRUE,
                             data_dir = NULL) {
  if (is.null(pred_long_all)) {
    if (is.null(predictions_probs)) {
      stop("Provide predictions_probs or pred_long_all")
    }
    pred_long_all <- get_label_probabilities(predictions_probs)
  }

  if (!all(c("book_id", "tag", "prob") %in% names(pred_long_all))) {
    stop("pred_long_all must have columns: book_id, tag, prob")
  }

  validate_prob_threshold(add_threshold, "add_threshold")
  validate_prob_threshold(remove_threshold, "remove_threshold")

  tag_counts <- get_tag_counts(eav) %>%
    rename(book_id = id)
  book_summary <- get_book_summary(eav) %>%
    rename(book_id = id)
  tag_summary <- get_tag_summary(eav) %>%
    rename(tag = value)
  existing_tags <- get_existing_tags(eav) %>%
    rename(tag = existing_tag)

  recommended_add <- get_recommended_add(eav, pred_long_all, add_threshold)
  recommended_remove <- get_recommended_remove(eav, pred_long_all, remove_threshold)

  con <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  DBI::dbWriteTable(con, "tag_counts", tag_counts, overwrite = overwrite, row.names = FALSE)
  DBI::dbWriteTable(con, "book_summary", book_summary, overwrite = overwrite, row.names = FALSE)
  DBI::dbWriteTable(con, "tag_summary", tag_summary, overwrite = overwrite, row.names = FALSE)
  DBI::dbWriteTable(con, "existing_tags", existing_tags, overwrite = overwrite, row.names = FALSE)
  DBI::dbWriteTable(con, "predictions", pred_long_all, overwrite = overwrite, row.names = FALSE)
  DBI::dbWriteTable(con, "recommendations_add", recommended_add, overwrite = overwrite, row.names = FALSE)
  DBI::dbWriteTable(con, "recommendations_remove", recommended_remove, overwrite = overwrite, row.names = FALSE)

  meta <- data.frame(
    key = c("created_at", "data_dir", "add_threshold", "remove_threshold"),
    value = c(
      format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
      if (is.null(data_dir)) "" else as.character(data_dir),
      as.character(add_threshold),
      as.character(remove_threshold)
    ),
    stringsAsFactors = FALSE
  )
  DBI::dbWriteTable(con, "meta", meta, overwrite = TRUE, row.names = FALSE)

  DBI::dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_predictions_book ON predictions (book_id)")
  DBI::dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_predictions_tag ON predictions (tag)")
  DBI::dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_predictions_book_tag ON predictions (book_id, tag)")
  DBI::dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_existing_tags_book_tag ON existing_tags (book_id, tag)")
  DBI::dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_book_summary_book ON book_summary (book_id)")
  DBI::dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_tag_summary_tag ON tag_summary (tag)")

  invisible(db_path)
}

#' Get recommended tags for a book from a results database
#' @param db_path Path to the SQLite results database
#' @param book_id Book ID to recommend tags for
#' @param threshold Minimum probability threshold
#' @param limit Optional maximum number of tags to return
#' @return A data frame with columns book_id, title, tag, prob
#' @export
get_recommended_tags_for_book_db <- function(db_path,
                                             book_id,
                                             threshold = 0.8,
                                             limit = NULL) {
  validate_prob_threshold(threshold, "threshold")

  con <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  assert_results_db_tables(con, c("predictions", "existing_tags", "book_summary"))

  book_exists <- DBI::dbGetQuery(
    con,
    "SELECT 1 FROM book_summary WHERE book_id = ? LIMIT 1",
    params = list(book_id)
  )
  if (nrow(book_exists) == 0) {
    stop(sprintf("Book ID %s not found in the results database", book_id))
  }

  query <- paste(
    "SELECT p.book_id, s.title, p.tag, p.prob",
    "FROM predictions p",
    "LEFT JOIN existing_tags e ON p.book_id = e.book_id AND p.tag = e.tag",
    "LEFT JOIN book_summary s ON p.book_id = s.book_id",
    "WHERE p.book_id = ? AND p.prob >= ? AND e.tag IS NULL",
    "ORDER BY p.prob DESC"
  )
  result <- DBI::dbGetQuery(con, query, params = list(book_id, threshold))

  if (!is.null(limit) && is.finite(limit)) {
    result <- utils::head(result, n = limit)
  }

  if (nrow(result) == 0) {
    return(tibble::tibble(
      book_id = numeric(0),
      title = character(0),
      tag = character(0),
      prob = numeric(0)
    ))
  }

  result
}

#' Get recommended books for a tag from a results database
#' @param db_path Path to the SQLite results database
#' @param tag_name Tag to recommend books for
#' @param threshold Minimum probability threshold
#' @param limit Optional maximum number of books to return
#' @param exclude_descendants Exclude books that already have descendant tags
#' @return A data frame with columns book_id, title, tag, prob
#' @export
get_recommended_books_for_tag_db <- function(db_path,
                                             tag_name,
                                             threshold = 0.5,
                                             limit = NULL,
                                             exclude_descendants = TRUE) {
  if (is.na(tag_name) || !nzchar(tag_name)) {
    stop("tag_name must be a non-empty string")
  }
  validate_prob_threshold(threshold, "threshold")

  con <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  assert_results_db_tables(con, c("predictions", "existing_tags", "book_summary"))

  query <- paste(
    "SELECT p.book_id, s.title, p.tag, p.prob",
    "FROM predictions p",
    "LEFT JOIN existing_tags e ON p.book_id = e.book_id AND p.tag = e.tag",
    "LEFT JOIN book_summary s ON p.book_id = s.book_id"
  )

  params <- list(tag_name, threshold)

  if (isTRUE(exclude_descendants)) {
    descendant_like <- paste0(escape_like_pattern(tag_name), ".%")
    query <- paste(
      query,
      "LEFT JOIN existing_tags d ON p.book_id = d.book_id AND d.tag LIKE ? ESCAPE '!'"
    )
    params <- list(descendant_like, tag_name, threshold)
  }

  query <- paste(
    query,
    "WHERE p.tag = ? AND p.prob >= ? AND e.tag IS NULL"
  )
  if (isTRUE(exclude_descendants)) {
    query <- paste(query, "AND d.tag IS NULL")
  }
  query <- paste(query, "ORDER BY p.prob DESC")

  result <- DBI::dbGetQuery(con, query, params = params)

  if (!is.null(limit) && is.finite(limit)) {
    result <- utils::head(result, n = limit)
  }

  if (nrow(result) == 0) {
    return(tibble::tibble(
      book_id = numeric(0),
      title = character(0),
      tag = character(0),
      prob = numeric(0)
    ))
  }

  result
}
