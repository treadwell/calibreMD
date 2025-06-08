library(testthat)
library(DBI)
library(RSQLite)
library(CalibreMD)
source(testthat::test_path("helper-test-data.R"))

test_that("find_md_db returns correct path", {
  tmpdir <- tempdir()
  file.create(file.path(tmpdir, "metadata.db"))
  expect_equal(find_md_db(tmpdir), file.path(tmpdir, "metadata.db"))
  unlink(file.path(tmpdir, "metadata.db"))
})

test_that("find_md_db errors on missing directory", {
  expect_error(find_md_db("/no/such/dir"))
})

test_that("load_eav returns correct EAV structure from test DB", {
  db_path <- tempfile(fileext = ".sqlite")
  create_test_calibre_db(db_path)
  eav <- load_eav(db_path)
  expect_s3_class(eav, "data.frame")
  expect_true(any(eav$feature == "title"))
  expect_true(any(eav$value == "Test Book"))
  unlink(db_path)
})

test_that("load_eav errors on missing DB file", {
  expect_error(load_eav("/no/such/file.db"))
}) 