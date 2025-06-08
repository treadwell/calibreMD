library(testthat)
library(CalibreMD)
source(testthat::test_path("helper-test-data.R"))

test_that("setup_packages loads required packages", {
  # Test that setup_packages runs without error
  expect_silent(setup_packages())
})

test_that("train_models creates models for each tag", {
  eav <- create_test_eav()
  dataset <- prep_dataset(eav)
  models <- train_models(dataset)
  
  # Check that we have a model for each tag
  expect_equal(length(models), ncol(dataset$Y))
  # Check that each model is an xgb.Booster
  expect_true(all(sapply(models, inherits, "xgb.Booster")))
})

test_that("predict_tags returns correct structure", {
  eav <- create_test_eav()
  dataset <- prep_dataset(eav)
  models <- train_models(dataset)
  predictions <- predict_tags(models, dataset)
  
  # Check structure
  expect_s3_class(predictions, "data.frame")
  expect_true("book_id" %in% names(predictions))
  expect_true(all(colnames(dataset$Y) %in% names(predictions)))
  expect_true(all(predictions[, -1] >= 0 & predictions[, -1] <= 1))
})

test_that("threshold_predictions returns correct tags", {
  eav <- create_test_eav()
  dataset <- prep_dataset(eav)
  models <- train_models(dataset)
  predictions <- predict_tags(models, dataset)
  
  # Test with high threshold (should return few or no tags)
  high_threshold_tags <- threshold_predictions(predictions, 1, threshold = 0.9)
  expect_true(length(high_threshold_tags) <= ncol(dataset$Y))
  
  # Test with low threshold (should return more tags)
  low_threshold_tags <- threshold_predictions(predictions, 1, threshold = 0.1)
  expect_true(length(low_threshold_tags) >= length(high_threshold_tags))
})

test_that("get_top_n_labels returns correct number of tags", {
  eav <- create_test_eav()
  dataset <- prep_dataset(eav)
  models <- train_models(dataset)
  predictions <- predict_tags(models, dataset)
  
  # Test with n=2
  top_2 <- get_top_n_labels(predictions, 1, n = 2)
  expect_equal(length(top_2), 2)
  
  # Test with n=1
  top_1 <- get_top_n_labels(predictions, 1, n = 1)
  expect_equal(length(top_1), 1)
})

test_that("get_label_probabilities returns correct structure", {
  eav <- create_test_eav()
  dataset <- prep_dataset(eav)
  models <- train_models(dataset)
  predictions <- predict_tags(models, dataset)
  probs <- get_label_probabilities(predictions)
  
  expect_s3_class(probs, "data.frame")
  expect_true(all(c("book_id", "tag", "prob") %in% names(probs)))
  expect_true(all(probs$prob >= 0 & probs$prob <= 1))
})

test_that("get_recommended_add returns correct structure", {
  eav <- create_test_eav()
  dataset <- prep_dataset(eav)
  models <- train_models(dataset)
  predictions <- predict_tags(models, dataset)
  probs <- get_label_probabilities(predictions)
  recommendations <- get_recommended_add(eav, probs, add_threshold = 0.5)
  
  expect_s3_class(recommendations, "data.frame")
  expect_true(all(c("book_id", "title", "recommended_tags") %in% names(recommendations)))
})

test_that("get_recommended_remove returns correct structure", {
  eav <- create_test_eav()
  dataset <- prep_dataset(eav)
  models <- train_models(dataset)
  predictions <- predict_tags(models, dataset)
  probs <- get_label_probabilities(predictions)
  recommendations <- get_recommended_remove(eav, probs)
  
  expect_s3_class(recommendations, "data.frame")
  expect_true(all(c("book_id", "title", "tags_to_review") %in% names(recommendations)))
})

test_that("suggest_tag_additions returns correct structure", {
  eav <- create_test_eav()
  dataset <- prep_dataset(eav)
  models <- train_models(dataset)
  predictions <- predict_tags(models, dataset)
  probs <- get_label_probabilities(predictions)
  suggestions <- suggest_tag_additions(eav, probs, "Fiction", prob_min = 0.5)
  
  expect_s3_class(suggestions, "data.frame")
  expect_true(all(c("book_id", "title", "tag", "prob") %in% names(suggestions)))
  expect_true(all(suggestions$prob >= 0.5))
}) 