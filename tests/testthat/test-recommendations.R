test_that("recommend_tags_for_book works correctly", {
  # Create test data
  test_eav <- tibble::tribble(
    ~id, ~feature, ~value,
    1, "title_original", "Test Book",
    1, "tag", "existing_tag1",
    1, "tag", "existing_tag2",
    2, "title_original", "Another Book",
    2, "tag", "existing_tag1"
  )
  
  # Create test predictions
  test_predictions <- tibble::tibble(
    book_id = c(1, 2),
    tag1 = c(0.9, 0.3),
    tag2 = c(0.85, 0.4),
    "existing_tag1" = c(0.95, 0.92),
    "existing_tag2" = c(0.88, 0.2)
  )
  
  # Test basic functionality
  result <- recommend_tags_for_book(test_eav, test_predictions, book_id = 1)
  expect_s3_class(result, "data.frame")
  expect_equal(colnames(result), c("book_id", "tag", "prob"))
  expect_equal(nrow(result), 2)  # Should only include tag1 and tag2 (not existing tags)
  expect_true(all(result$book_id == 1))
  expect_true(all(result$prob >= 0.8))  # Default threshold
  expect_true(all(result$tag %in% c("tag1", "tag2")))  # Only new tags
  
  # Test threshold filtering
  high_threshold <- recommend_tags_for_book(test_eav, test_predictions, book_id = 1, threshold = 0.88)
  expect_equal(nrow(high_threshold), 1)  # Should only include tag1
  expect_true(all(high_threshold$prob >= 0.88))
  expect_equal(high_threshold$tag, "tag1")
  
  # Test error for non-existent book
  expect_error(
    recommend_tags_for_book(test_eav, test_predictions, book_id = 999),
    "Book ID 999 not found in the dataset"
  )
  
  # Test with book that has no existing tags
  test_eav_no_tags <- test_eav %>% filter(id == 1, feature != "tag")
  result_no_tags <- recommend_tags_for_book(test_eav_no_tags, test_predictions, book_id = 1)
  expect_equal(nrow(result_no_tags), 4)  # Should include all predicted tags
  expect_true(all(result_no_tags$prob >= 0.8))
})

test_that("recommend_tags_for_book handles edge cases", {
  # Test with empty predictions
  test_eav <- tibble::tribble(
    ~id, ~feature, ~value,
    1, "title_original", "Test Book",
    1, "tag", "existing_tag1"
  )
  empty_predictions <- tibble::tibble(book_id = 1)
  
  result <- recommend_tags_for_book(test_eav, empty_predictions, book_id = 1)
  expect_equal(nrow(result), 0)
  expect_s3_class(result, "data.frame")
  expect_equal(colnames(result), c("book_id", "tag", "prob"))
}) 