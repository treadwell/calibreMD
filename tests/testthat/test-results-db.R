library(testthat)
library(CalibreMD)

test_that("write_results_db supports db recommendations", {
  setup_packages()

  eav <- tibble::tribble(
    ~id, ~feature, ~value,
    1, "title_original", "Book One",
    1, "tag", "Fiction.Scifi",
    2, "title_original", "Book Two",
    2, "tag", "History",
    3, "title_original", "Book Three",
    4, "title_original", "Book Four",
    4, "tag", "Fiction"
  )

  predictions_probs <- tibble::tibble(
    book_id = c(1, 2, 3, 4),
    Fiction = c(0.9, 0.85, 0.92, 0.96),
    History = c(0.1, 0.95, 0.2, 0.05)
  )

  pred_long <- get_label_probabilities(predictions_probs)
  db_path <- tempfile(fileext = ".sqlite")
  on.exit(unlink(db_path), add = TRUE)

  write_results_db(
    db_path = db_path,
    eav = eav,
    pred_long_all = pred_long,
    add_threshold = 0.8,
    remove_threshold = 0.05,
    overwrite = TRUE
  )

  tags_for_book <- get_recommended_tags_for_book_db(
    db_path,
    book_id = 3,
    threshold = 0.8
  )
  expect_true(all(tags_for_book$book_id == 3))
  expect_true("Fiction" %in% tags_for_book$tag)

  books_for_tag <- get_recommended_books_for_tag_db(
    db_path,
    tag_name = "Fiction",
    threshold = 0.8
  )
  expect_true(all(books_for_tag$book_id %in% c(2, 3)))
  expect_false(any(books_for_tag$book_id %in% c(1, 4)))

  books_include_desc <- get_recommended_books_for_tag_db(
    db_path,
    tag_name = "Fiction",
    threshold = 0.8,
    exclude_descendants = FALSE
  )
  expect_true(any(books_include_desc$book_id == 1))

  expect_error(
    get_recommended_tags_for_book_db(db_path, book_id = 999),
    "Book ID 999 not found in the results database"
  )
})
