# Test data for tag handling functions
library(tidyverse)

# Expanded Sample EAV data frame for testing
create_test_eav <- function() {
  tibble(
    id = rep(1:12, each = 3),
    feature = rep(c("title", "tag", "tag"), times = 12),
    value = c(
      "Book 1", "Fiction", "Mystery",
      "Book 2", "Fiction", "History",
      "Book 3", "Fiction", "Science Fiction",
      "Book 4", "Fiction", "Mystery",
      "Book 5", "Fiction", "Mystery",
      "Book 6", "Fiction", "Mystery",
      "Book 7", "Fiction", "Mystery",
      "Book 8", "Fiction", "Mystery",
      "Book 9", "Fiction", "Mystery",
      "Book 10", "Fiction", "Mystery",
      "Book 11", "Fiction", "History",
      "Book 12", "Fiction", "Science Fiction"
    )
  )
}

# Expanded Sample book tags data frame
create_test_book_tags <- function() {
  tibble(
    id = rep(1:12, each = 2),
    tag = c(
      "Fiction", "Mystery",
      "Fiction", "History",
      "Fiction", "Science Fiction",
      "Fiction", "Mystery",
      "Fiction", "Mystery",
      "Fiction", "Mystery",
      "Fiction", "Mystery",
      "Fiction", "Mystery",
      "Fiction", "Mystery",
      "Fiction", "Mystery",
      "Fiction", "History",
      "Fiction", "Science Fiction"
    )
  )
}

# Expanded Sample tag summary data frame
create_test_tag_summary <- function() {
  tibble(
    value = c("Fiction", "Mystery", "History", "Science Fiction"),
    book_count = c(12, 8, 2, 2)
  )
}

# Helper function to create a sparse matrix for testing
create_test_sparse_matrix <- function() {
  library(Matrix)
  sparseMatrix(
    i = c(1:12, 1:8),
    j = c(rep(1, 12), rep(2, 8)),
    x = 1,
    dims = c(12, 4),
    dimnames = list(
      as.character(1:12),
      c("Fiction", "Mystery", "History", "Science Fiction")
    )
  )
}

# Helper to create a minimal Calibre-like SQLite database for import tests
create_test_calibre_db <- function(db_path) {
  con <- DBI::dbConnect(RSQLite::SQLite(), db_path)
  # Create minimal tables
  DBI::dbExecute(con, "CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT)")
  DBI::dbExecute(con, "CREATE TABLE comments (book INTEGER, text TEXT)")
  DBI::dbExecute(con, "CREATE TABLE books_authors_link (book INTEGER, author INTEGER)")
  DBI::dbExecute(con, "CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT)")
  DBI::dbExecute(con, "CREATE TABLE books_publishers_link (book INTEGER, publisher INTEGER)")
  DBI::dbExecute(con, "CREATE TABLE publishers (id INTEGER PRIMARY KEY, name TEXT)")
  DBI::dbExecute(con, "CREATE TABLE books_tags_link (book INTEGER, tag INTEGER)")
  DBI::dbExecute(con, "CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT)")
  DBI::dbExecute(con, "CREATE TABLE books_ratings_link (book INTEGER, rating INTEGER)")
  DBI::dbExecute(con, "CREATE TABLE books_series_link (book INTEGER, series INTEGER)")
  DBI::dbExecute(con, "CREATE TABLE series (id INTEGER PRIMARY KEY, name TEXT)")
  # Insert sample data
  DBI::dbExecute(con, "INSERT INTO books VALUES (1, 'Test Book')")
  DBI::dbExecute(con, "INSERT INTO comments VALUES (1, 'A comment.')")
  DBI::dbExecute(con, "INSERT INTO authors VALUES (1, 'Author One')")
  DBI::dbExecute(con, "INSERT INTO books_authors_link VALUES (1, 1)")
  DBI::dbExecute(con, "INSERT INTO publishers VALUES (1, 'Publisher One')")
  DBI::dbExecute(con, "INSERT INTO books_publishers_link VALUES (1, 1)")
  DBI::dbExecute(con, "INSERT INTO tags VALUES (1, 'Fiction')")
  DBI::dbExecute(con, "INSERT INTO books_tags_link VALUES (1, 1)")
  DBI::dbExecute(con, "INSERT INTO books_ratings_link VALUES (1, 5)")
  DBI::dbExecute(con, "INSERT INTO series VALUES (1, 'Series One')")
  DBI::dbExecute(con, "INSERT INTO books_series_link VALUES (1, 1)")
  DBI::dbDisconnect(con)
  invisible(db_path)
} 