# CalibreMD

CalibreMD is an R package for analyzing Calibre ebook library metadata and recommending tags for your books. It connects to your Calibre SQLite database, extracts metadata, processes features, and provides tools for tag analysis and recommendation.

## Features
- Connects to Calibre's `metadata.db` and extracts book metadata
- Processes and tokenizes text features (title, comment, etc.)
- Provides tag statistics and summaries
- Prepares data for machine learning models
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

## Testing

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

## Contributing

Pull requests and issues are welcome! Please add tests for any new features or bug fixes.

## License

MIT License. See `LICENSE` file for details. 