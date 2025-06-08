library(testthat)
library(CalibreMD)
source(testthat::test_path("helper-test-data.R"))

# Test load_existing_tags

test_that("load_existing_tags returns correct structure and values", {
  eav <- create_test_eav()
  tags <- load_existing_tags(eav)
  expect_s3_class(tags, "data.frame")
  expect_true(all(c("id", "tag") %in% names(tags)))
  expect_equal(nrow(tags), 24)
  expect_true(all(tags$tag %in% c("Fiction", "Mystery", "History", "Science Fiction")))
})

# Test clean_and_join_tags

test_that("clean_and_join_tags joins book_tags and pred_long_all correctly", {
  book_tags <- create_test_book_tags()
  pred_long_all <- tibble(book_id = rep(1:12, each = 2), tag = rep(c("Fiction", "Mystery"), 12), prob = runif(24))
  joined <- clean_and_join_tags(book_tags, pred_long_all)
  expect_s3_class(joined, "data.frame")
  expect_true(all(c("book_id", "tag", "prob") %in% names(joined)))
  expect_true(all(joined$tag %in% c("Fiction", "Mystery")))
})

# Test filter_by_threshold

test_that("filter_by_threshold filters correctly for greater_than and less_than", {
  pred_long_all <- tibble(book_id = 1:5, tag = rep("Fiction", 5), prob = c(0.1, 0.5, 0.7, 0.9, 0.2))
  filtered_gt <- filter_by_threshold(pred_long_all, 0.5, greater_than = TRUE)
  expect_true(all(filtered_gt$prob > 0.5))
  filtered_lt <- filter_by_threshold(pred_long_all, 0.5, greater_than = FALSE)
  expect_true(all(filtered_lt$prob < 0.5))
})

# Test join_with_titles

test_that("join_with_titles joins titles and selects correct columns", {
  eav <- create_test_eav()
  df <- tibble(book_id = 1:2, recommended_tags = c("Fiction", "Mystery"))
  result <- join_with_titles(df, eav, tag_col = "recommended_tags")
  expect_s3_class(result, "data.frame")
  expect_true(all(c("book_id", "title", "recommended_tags") %in% names(result)))
  expect_equal(nrow(result), 2)
})

test_that("join_with_titles works for tags_to_review column", {
  eav <- create_test_eav()
  df <- tibble(book_id = 1:2, tags_to_review = c("Fiction", "Mystery"))
  result <- join_with_titles(df, eav, tag_col = "tags_to_review")
  expect_s3_class(result, "data.frame")
  expect_true(all(c("book_id", "title", "tags_to_review") %in% names(result)))
  expect_equal(nrow(result), 2)
})

# Test filter_descendant_tags

test_that("filter_descendant_tags returns correct descendant book_ids", {
  existing_tags <- tibble(book_id = 1:4, existing_tag = c("Fiction", "Fiction.Mystery", "Fiction.History", "Science Fiction"))
  descendants <- filter_descendant_tags(existing_tags, "Fiction")
  expect_s3_class(descendants, "data.frame")
  expect_true("book_id" %in% names(descendants))
  expect_true(all(descendants$book_id %in% c(2, 3)))
}) 