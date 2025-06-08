library(CalibreMD)
library(testthat)
library(tidyverse)
library(Matrix)

# Load the helper functions and test data
source(testthat::test_path("helper-test-data.R"))

# Test get_tag_counts function
test_that("get_tag_counts returns correct counts", {
  eav <- create_test_eav()
  result <- get_tag_counts(eav)
  
  expect_equal(nrow(result), 12)  # Should have 12 books
  expect_true(all(result$n_tags == 2))  # Each book has 2 tags
  expect_true(all(result$id %in% 1:12))  # Should have correct book IDs
})

# Test get_tag_summary function
test_that("get_tag_summary returns correct summary", {
  eav <- create_test_eav()
  result <- get_tag_summary(eav)
  
  expect_equal(nrow(result), 4)  # Should have 4 unique tags
  expect_equal(result$book_count, c(12, 8, 2, 2))  # Correct book counts for Fiction, Mystery, History, Science Fiction
  expect_true(all(result$value %in% c("Fiction", "Mystery", "History", "Science Fiction")))
})

# Test get_existing_tags function
test_that("get_existing_tags returns correct tag assignments", {
  eav <- create_test_eav()
  result <- get_existing_tags(eav)
  
  expect_equal(nrow(result), 24)  # Should have 24 tag assignments
  expect_true(all(result$book_id %in% 1:12))  # Should have correct book IDs
  expect_true(all(result$existing_tag %in% c("Fiction", "Mystery", "History", "Science Fiction")))
})

# Test prep_dataset function
test_that("prep_dataset creates correct matrices", {
  eav <- create_test_eav()
  result <- prep_dataset(eav)
  
  # Check feature matrix
  expect_s4_class(result$X, "dgCMatrix")  # Should be a sparse matrix
  expect_equal(nrow(result$X), 12)  # Should have 12 books
  expect_true(ncol(result$X) > 0)  # Should have features
  
  # Check tag matrix
  expect_s4_class(result$Y, "dgCMatrix")  # Should be a sparse matrix
  expect_equal(nrow(result$Y), 12)  # Should have 12 books
  expect_equal(ncol(result$Y), 1)  # Only 'Fiction' appears in >=10 books
  
  # Check book_features
  expect_s3_class(result$book_features, "data.frame")
  expect_true(all(c("id", "feature") %in% names(result$book_features)))
}) 