library(testthat)
library(CalibreMD)
source(testthat::test_path("helper-test-data.R"))

test_that("prep_dataset creates correct matrices and features", {
  eav <- create_test_eav()
  result <- prep_dataset(eav)
  # Check feature matrix
  expect_s4_class(result$X, "dgCMatrix")
  expect_equal(nrow(result$X), 12)
  expect_true(ncol(result$X) > 0)
  # Check tag matrix
  expect_s4_class(result$Y, "dgCMatrix")
  expect_equal(nrow(result$Y), 12)
  expect_equal(ncol(result$Y), 1) # Only 'Fiction' appears in >=10 books
  # Check book_features
  expect_s3_class(result$book_features, "data.frame")
  expect_true(all(c("id", "feature") %in% names(result$book_features)))
}) 