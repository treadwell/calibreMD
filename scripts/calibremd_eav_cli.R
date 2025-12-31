#!/usr/bin/env Rscript

print_usage <- function() {
  cat(paste(
    "CalibreMD EAV CLI",
    "",
    "Usage:",
    "  Rscript scripts/calibremd_eav_cli.R --data-dir /path/to/Calibre --out-dir ./out",
    "",
    "Required (unless --query-only):",
    "  --data-dir PATH           Path to the Calibre library directory",
    "",
    "Optional:",
    "  --out-dir PATH            Directory for CSV outputs (default: --data-dir when set; otherwise script directory)",
    "  --add-threshold NUM       Probability threshold for add recommendations (default: 0.75)",
    "  --remove-threshold NUM    Probability threshold for remove recommendations (default: 0.01)",
    "  --tag-name TAG            Tag to suggest additions for (optional)",
    "  --tag-prob-min NUM        Minimum probability for tag suggestions (default: 0.5)",
    "  --book-id ID              Book ID to show recommendations for (optional)",
    "  --book-threshold NUM      Threshold for book recommendations (default: 0.8)",
    "  --top-n N                 Top N labels to show for --book-id (default: 5)",
    "  --write-csv [true|false]  Enable or disable CSV outputs (default: true)",
    "  --no-write-csv            Disable CSV outputs",
    "  --results-db PATH         SQLite results DB path (default: <out-dir>/calibremd_results.sqlite)",
    "  --write-db [true|false]   Enable or disable DB output (default: true)",
    "  --no-write-db             Disable DB output",
    "  --query-only [true|false] Skip training and query an existing results DB (default: false)",
    "  Note: query-only defaults to console output; CSV disabled unless --write-csv is set",
    "  --debug                   Print debug summaries",
    "  -h, --help                Show this help",
    "",
    "Environment:",
    "  CALIBRE_DIR               Used as --data-dir if provided",
    "",
    sep = "\n"
  ), "\n")
}

parse_bool <- function(val) {
  if (is.logical(val)) return(val)
  val <- tolower(as.character(val))
  if (val %in% c("1", "true", "yes", "y")) return(TRUE)
  if (val %in% c("0", "false", "no", "n")) return(FALSE)
  NA
}

log_step <- function(...) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  msg <- paste(..., collapse = "")
  cat(sprintf("[%s] %s\n", timestamp, msg))
  flush.console()
}

parse_args <- function(args) {
  opts <- list(
    data_dir = NULL,
    out_dir = NULL,
    add_threshold = 0.75,
    remove_threshold = 0.01,
    tag_name = NULL,
    tag_prob_min = 0.5,
    book_id = NA_integer_,
    book_threshold = 0.8,
    top_n = 5L,
    write_csv = TRUE,
    write_csv_set = FALSE,
    results_db = NULL,
    write_db = TRUE,
    query_only = FALSE,
    debug = FALSE,
    help = FALSE
  )

  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]

    if (arg %in% c("-h", "--help")) {
      opts$help <- TRUE
      i <- i + 1
      next
    }

    if (!startsWith(arg, "--")) {
      stop("Unexpected argument: ", arg)
    }

    key_val <- substring(arg, 3)
    if (grepl("=", key_val, fixed = TRUE)) {
      parts <- strsplit(key_val, "=", fixed = TRUE)[[1]]
      key <- parts[1]
      val <- parts[2]
    } else {
      key <- key_val
      if (key %in% c("no-write-csv", "no-write-db", "debug")) {
        val <- TRUE
      } else if (i < length(args) && !startsWith(args[[i + 1]], "--")) {
        val <- args[[i + 1]]
        i <- i + 1
      } else {
        val <- TRUE
      }
    }

    if (key == "data-dir") {
      opts$data_dir <- val
    } else if (key == "out-dir") {
      opts$out_dir <- val
    } else if (key == "add-threshold") {
      opts$add_threshold <- as.numeric(val)
    } else if (key == "remove-threshold") {
      opts$remove_threshold <- as.numeric(val)
    } else if (key == "tag-name") {
      opts$tag_name <- val
    } else if (key == "tag-prob-min") {
      opts$tag_prob_min <- as.numeric(val)
    } else if (key == "book-id") {
      opts$book_id <- as.integer(val)
    } else if (key == "book-threshold") {
      opts$book_threshold <- as.numeric(val)
    } else if (key == "top-n") {
      opts$top_n <- as.integer(val)
    } else if (key == "write-csv") {
      opts$write_csv <- parse_bool(val)
      opts$write_csv_set <- TRUE
    } else if (key == "no-write-csv") {
      opts$write_csv <- FALSE
      opts$write_csv_set <- TRUE
    } else if (key == "results-db") {
      opts$results_db <- val
    } else if (key == "write-db") {
      opts$write_db <- parse_bool(val)
    } else if (key == "no-write-db") {
      opts$write_db <- FALSE
    } else if (key == "query-only") {
      opts$query_only <- parse_bool(val)
    } else if (key == "debug") {
      opts$debug <- TRUE
    } else {
      stop("Unknown option: --", key)
    }

    i <- i + 1
  }

  opts
}

normalize_script_dir <- function() {
  script_path <- NULL
  if (!is.null(sys.frame(1)$ofile)) {
    script_path <- normalizePath(sys.frame(1)$ofile)
  }
  if (is.null(script_path)) {
    return(getwd())
  }
  dirname(script_path)
}

load_calibremd <- function() {
  script_dir <- normalize_script_dir()
  candidates <- c(
    file.path(script_dir, "R", "CalibreMD.R"),
    file.path(script_dir, "..", "R", "CalibreMD.R")
  )
  calibremd_path <- candidates[file.exists(candidates)][1]

  if (!is.na(calibremd_path)) {
    source(calibremd_path)
    return(invisible(NULL))
  }

  if (requireNamespace("CalibreMD", quietly = TRUE)) {
    library(CalibreMD)
    return(invisible(NULL))
  }

  if (is.na(calibremd_path)) {
    stop("CalibreMD package not installed and R/CalibreMD.R not found. ",
         "Run from the repo root or install the package.")
  }
}

validate_threshold <- function(value, name) {
  if (is.na(value) || value < 0 || value > 1) {
    stop(name, " must be between 0 and 1")
  }
}

safe_filename <- function(text) {
  gsub("[^A-Za-z0-9_-]+", "_", text)
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))

if (opts$help) {
  print_usage()
  quit(status = 0)
}

log_step("Starting CalibreMD EAV CLI")

if (is.null(opts$data_dir)) {
  env_dir <- Sys.getenv("CALIBRE_DIR", unset = "")
  if (nzchar(env_dir)) {
    opts$data_dir <- env_dir
  }
}

if (is.null(opts$query_only) || is.na(opts$query_only)) {
  stop("query-only must be true or false")
}

if (isTRUE(opts$query_only)) {
  opts$write_db <- FALSE
  log_step("Query-only mode enabled")
}

if (isTRUE(opts$query_only) && !isTRUE(opts$write_csv_set)) {
  opts$write_csv <- FALSE
}

if (!isTRUE(opts$query_only) && is.null(opts$data_dir)) {
  print_usage()
  stop("--data-dir is required unless --query-only is set")
}

if (!isTRUE(opts$query_only) && (is.logical(opts$data_dir) || !nzchar(opts$data_dir))) {
  stop("--data-dir must be a path")
}

if (is.logical(opts$tag_name)) {
  stop("--tag-name requires a value")
}

if (is.na(opts$top_n) || opts$top_n < 1) {
  stop("--top-n must be a positive integer")
}

validate_threshold(opts$add_threshold, "add-threshold")
validate_threshold(opts$remove_threshold, "remove-threshold")
validate_threshold(opts$tag_prob_min, "tag-prob-min")
validate_threshold(opts$book_threshold, "book-threshold")

if (is.na(opts$write_csv)) {
  stop("write-csv must be true or false")
}

if (is.na(opts$write_db)) {
  stop("write-db must be true or false")
}

if (is.null(opts$out_dir) || !nzchar(opts$out_dir)) {
  if (!is.null(opts$data_dir) && nzchar(opts$data_dir)) {
    opts$out_dir <- opts$data_dir
  } else {
    opts$out_dir <- normalize_script_dir()
  }
}

log_step("Output directory: ", opts$out_dir)

if (!dir.exists(opts$out_dir)) {
  dir.create(opts$out_dir, recursive = TRUE)
}

if (is.null(opts$results_db)) {
  opts$results_db <- file.path(opts$out_dir, "calibremd_results.sqlite")
}

if (is.logical(opts$results_db) || !nzchar(opts$results_db)) {
  stop("--results-db must be a path")
}

log_step("Results DB: ", opts$results_db)

load_calibremd()

predictions_probs <- NULL
pred_long_all <- NULL
eav <- NULL

if (!isTRUE(opts$query_only)) {
  setup_packages()

  md_db <- find_md_db(opts$data_dir)
  log_step("Loading metadata from: ", md_db)

  eav <- md_db %>% load_eav()
  log_step("Loaded EAV rows: ", nrow(eav))

  log_step("Exploding text features")
  eav <- eav %>% explode_text_features()
  log_step("EAV rows after explode: ", nrow(eav))

  if (opts$debug) {
    cat("Debug: EAV data summary\n")
    cat("Total books in EAV:", length(unique(eav$id)), "\n")
    cat("Total tags in EAV:", length(unique(eav %>% dplyr::filter(feature == "tag") %>% dplyr::pull(value))), "\n")

    tag_freq <- eav %>%
      dplyr::filter(feature == "tag") %>%
      dplyr::group_by(value) %>%
      dplyr::summarize(n_books = dplyr::n_distinct(id), .groups = "drop") %>%
      dplyr::arrange(dplyr::desc(n_books))

    cat("Tag frequencies (top 10):\n")
    print(head(tag_freq, 10))
    cat("Tags with >= 10 books:", sum(tag_freq$n_books >= 10), "\n")
    cat(
      "Books with tags that have >= 10 books:",
      length(unique(
        eav %>%
          dplyr::filter(feature == "tag") %>%
          dplyr::filter(value %in% (tag_freq %>% dplyr::filter(n_books >= 10) %>% dplyr::pull(value))) %>%
          dplyr::pull(id)
      )),
      "\n"
    )
  }

  log_step("Preparing dataset")
  dataset <- prep_dataset(eav)
  log_step("Feature matrix: ", nrow(dataset$X), " x ", ncol(dataset$X))
  log_step("Tag matrix: ", nrow(dataset$Y), " x ", ncol(dataset$Y))

  log_step("Training models for ", ncol(dataset$Y), " tags")
  train_start <- Sys.time()
  models <- train_models(dataset, progress = TRUE)
  log_step("Training complete in ", round(difftime(Sys.time(), train_start, units = "mins"), 2), " mins")

  log_step("Predicting tags")
  predict_start <- Sys.time()
  predictions_probs <- predict_tags(models, dataset, progress = TRUE)
  log_step("Prediction complete in ", round(difftime(Sys.time(), predict_start, units = "mins"), 2), " mins")
  log_step("Computing recommendations")
  pred_long_all <- get_label_probabilities(predictions_probs)

  final_addition_recommendations <- get_recommended_add(eav, pred_long_all, opts$add_threshold)
  final_removal_recommendations <- get_recommended_remove(eav, pred_long_all, opts$remove_threshold)

  if (opts$write_db) {
    log_step("Writing results DB")
    write_results_db(
      db_path = opts$results_db,
      eav = eav,
      pred_long_all = pred_long_all,
      add_threshold = opts$add_threshold,
      remove_threshold = opts$remove_threshold,
      overwrite = TRUE,
      data_dir = opts$data_dir
    )
    log_step("Wrote results DB: ", opts$results_db)
  }

  if (opts$write_csv) {
    log_step("Writing CSV outputs")
    add_path <- file.path(opts$out_dir, "calibremd_recommendations_add.csv")
    remove_path <- file.path(opts$out_dir, "calibremd_recommendations_remove.csv")
    utils::write.csv(final_addition_recommendations, add_path, row.names = FALSE)
    utils::write.csv(final_removal_recommendations, remove_path, row.names = FALSE)
    log_step("Wrote: ", add_path)
    log_step("Wrote: ", remove_path)
  }
}

db_available <- file.exists(opts$results_db)

if (isTRUE(opts$query_only) && !db_available) {
  stop("Results DB not found. Run without --query-only to generate one.")
}

get_predictions_for_book_db <- function(db_path, book_id) {
  con <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  required <- c("predictions", "book_summary")
  missing <- required[!vapply(required, function(name) DBI::dbExistsTable(con, name), logical(1))]
  if (length(missing) > 0) {
    stop("Results DB is missing required tables: ", paste(missing, collapse = ", "))
  }

  book_exists <- DBI::dbGetQuery(
    con,
    "SELECT 1 FROM book_summary WHERE book_id = ? LIMIT 1",
    params = list(book_id)
  )
  if (nrow(book_exists) == 0) {
    stop(sprintf("Book ID %s not found in the results database", book_id))
  }

  DBI::dbGetQuery(
    con,
    "SELECT tag, prob FROM predictions WHERE book_id = ? ORDER BY prob DESC",
    params = list(book_id)
  )
}

get_top_n_labels_db <- function(db_path, book_id, n = 5) {
  preds <- get_predictions_for_book_db(db_path, book_id)
  utils::head(preds$tag, n)
}

get_threshold_labels_db <- function(db_path, book_id, threshold) {
  preds <- get_predictions_for_book_db(db_path, book_id)
  preds$tag[preds$prob >= threshold]
}

if (!is.null(opts$tag_name)) {
  log_step("Querying recommendations for tag: ", opts$tag_name)
  if (db_available) {
    tag_additions <- get_recommended_books_for_tag_db(
      opts$results_db,
      opts$tag_name,
      threshold = opts$tag_prob_min
    )
  } else {
    tag_additions <- suggest_tag_additions(
      eav,
      pred_long_all,
      opts$tag_name,
      prob_min = opts$tag_prob_min
    )
  }
  if (opts$write_csv) {
    tag_safe <- safe_filename(opts$tag_name)
    tag_path <- file.path(opts$out_dir, paste0("calibremd_tag_additions_", tag_safe, ".csv"))
    utils::write.csv(tag_additions, tag_path, row.names = FALSE)
    log_step("Wrote: ", tag_path)
  }
  if (nrow(tag_additions) > 0) {
    print(utils::head(tag_additions, 10))
  }
  log_step("Tag query rows: ", nrow(tag_additions))
}

if (!is.na(opts$book_id)) {
  log_step("Querying recommendations for book: ", opts$book_id)
  if (db_available) {
    top_n_labels <- get_top_n_labels_db(opts$results_db, opts$book_id, opts$top_n)
    threshold_labels <- get_threshold_labels_db(opts$results_db, opts$book_id, opts$book_threshold)
    book_recommendations <- get_recommended_tags_for_book_db(
      opts$results_db,
      opts$book_id,
      threshold = opts$book_threshold
    )
  } else {
    top_n_labels <- get_top_n_labels(predictions_probs, book_id = opts$book_id, n = opts$top_n)
    threshold_labels <- threshold_predictions(
      predictions_probs,
      book_id = opts$book_id,
      threshold = opts$book_threshold
    )
    book_recommendations <- recommend_tags_for_book(
      eav,
      predictions_probs,
      book_id = opts$book_id,
      threshold = opts$book_threshold
    )
    book_titles <- get_book_titles(eav)
    book_recommendations <- book_recommendations %>%
      dplyr::left_join(book_titles, by = "book_id") %>%
      dplyr::select(book_id, title, tag, prob)
  }

  cat("Top labels for book", opts$book_id, ":", paste(top_n_labels, collapse = ", "), "\n")
  cat("Threshold labels for book", opts$book_id, ":", paste(threshold_labels, collapse = ", "), "\n")

  if (opts$write_csv) {
    book_path <- file.path(opts$out_dir, paste0("calibremd_book_", opts$book_id, "_recommendations.csv"))
    utils::write.csv(book_recommendations, book_path, row.names = FALSE)
    log_step("Wrote: ", book_path)
  }

  if (nrow(book_recommendations) > 0) {
    print(book_recommendations)
  }
  log_step("Book query rows: ", nrow(book_recommendations))
}
