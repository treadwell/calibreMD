# CalibreMD

CalibreMD is an R package for analyzing Calibre ebook library metadata and recommending tags for your books. It connects to your Calibre SQLite database, extracts metadata, processes features, and provides tools for tag analysis and recommendation.

## Features
- Connects to Calibre's `metadata.db` and extracts book metadata
- Processes and tokenizes text features (title, comment, etc.)
- Provides tag statistics and summaries
- Prepares data for machine learning models
- Trains XGBoost models for tag prediction
- Recommends new tags and identifies potentially incorrect tags
- Includes a test suite for robust development

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/treadwell/calibreMD.git
   cd calibreMD
   ```
2. Open R in this directory, then run:
   ```r
   install.packages("devtools") # if not already installed
   devtools::install_local(".")
   ```

## Usage

### Basic Usage

Load the package and connect to your Calibre library:

```r
library(CalibreMD)

# Set your Calibre library directory
calibre_dir <- "/path/to/your/Calibre Library"

# Find the metadata database
md_db <- find_md_db(calibre_dir)

# Load EAV data
metadata <- load_eav(md_db)

# Get tag counts
tag_counts <- get_tag_counts(metadata)

# Prepare data for modeling
prepared <- prep_dataset(metadata)
```

### Tag Recommendations

```r
# Train models
models <- train_models(prepared)

# Get predictions
predictions <- predict_tags(models, prepared)

# Get recommended tags to add (probability > 0.8)
recommendations <- get_recommended_add(metadata, predictions, add_threshold = 0.8)

# Find potentially incorrect tags (probability < 0.05)
to_review <- get_recommended_remove(metadata, predictions, remove_threshold = 0.05)

# Get suggestions for a specific tag
suggestions <- suggest_tag_additions(metadata, predictions, tag_name = "@your_tag", prob_min = 0.5)

# Get recommendations for a specific book
book_recommendations <- recommend_tags_for_book(metadata, predictions, book_id = 123, threshold = 0.8)
print(book_recommendations$title)  # Show book title
print(book_recommendations$existing_tags)  # Show current tags
print(book_recommendations$recommendations)  # Show recommended new tags with probabilities
```

## Development

### Testing

This package uses `testthat` for unit testing. To run the tests:

1. Install the development dependencies:
   ```r
   install.packages(c("testthat", "covr"))
   ```
2. Run the tests using devtools:
   ```r
   devtools::test()
   ```
3. To check code coverage:
   ```r
   covr::report(covr::package_coverage(), file = "coverage.html")
   # Then open coverage.html in your browser
   ```

### Continuous Testing

The repository includes a watch script that automatically runs tests when source files change. To use it:

1. Make sure `fswatch` is installed (available by default on macOS)
2. Run the watch script in a separate terminal:
   ```sh
   ./watch-tests.sh
   ```

The script will monitor the `R/` and `tests/` directories and automatically run the test suite whenever a file changes. This is useful for test-driven development and catching issues early.

## Contributing

Pull requests and issues are welcome! Please add tests for any new features or bug fixes.

## License

MIT License. See `LICENSE` file for details. 