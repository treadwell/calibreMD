library(testthat)
library(CalibreMD)
source(testthat::test_path("helper-test-data.R"))

test_that("explode_text_features tokenizes text and preserves original titles", {
  eav <- create_test_eav()
  exploded <- explode_text_features(eav)
  expect_true("title_original" %in% exploded$feature)
  expect_true(any(duplicated(exploded$id)))
})

test_that("get_tag_counts returns correct counts", {
  eav <- create_test_eav()
  counts <- get_tag_counts(eav)
  expect_equal(nrow(counts), 12)
  expect_true(all(counts$n_tags == 2))
})

test_that("get_book_summary returns correct structure", {
  eav <- create_test_eav()
  summary <- get_book_summary(eav)
  expect_true(all(c("id", "title", "n_tags", "title_length") %in% names(summary)))
  expect_equal(nrow(summary), 12)
})

test_that("get_tag_summary returns correct tag stats", {
  eav <- create_test_eav()
  summary <- get_tag_summary(eav)
  expect_true(all(c("value", "book_count") %in% names(summary)))
  expect_true("Fiction" %in% summary$value)
})

test_that("get_book_titles returns all titles", {
  eav <- create_test_eav()
  titles <- get_book_titles(eav)
  expect_true(all(c("book_id", "title") %in% names(titles)))
})

test_that("get_existing_tags returns all tag assignments", {
  eav <- create_test_eav()
  tags <- get_existing_tags(eav)
  expect_true(all(c("book_id", "existing_tag") %in% names(tags)))
  expect_equal(nrow(tags), 24)
}) 