library(CalibreMD)
setup_packages()

# Load data
dataDir <- '/Users/kbrooks/Dropbox/Books/Calibre Travel Library'
md_db <- find_md_db(dataDir)
eav <- md_db %>% 
  load_eav() %>% 
  explode_text_features()
cat("Total unique books in EAV:", length(unique(eav$id)), "\n")

# Run spectral analysis up to binary embeddings
dataset_spectral <- prep_dataset(eav)
cat("Total unique books in dataset_spectral$X:", nrow(dataset_spectral$X), "\n")

# Build similarity matrix
similarity_matrix <- as.matrix(dataset_spectral$X) %*% t(as.matrix(dataset_spectral$X))

# Compute Laplacian
degree_matrix <- diag(rowSums(similarity_matrix))
laplacian_matrix <- degree_matrix - similarity_matrix

# Compute eigenvectors
eig_result <- eigen(laplacian_matrix)
eigvecs <- eig_result$vectors[, 2:11]  # Skip first eigenvector, take next 10

# Iterative quantization
iterative_quantization <- function(X, n_iter = 50) {
  # Center the data
  X_centered <- scale(X, center = TRUE, scale = FALSE)
  
  # PCA
  pca_result <- prcomp(X_centered, center = FALSE, scale = FALSE)
  X_pca <- pca_result$x[, 1:min(64, ncol(pca_result$x))]
  
  # Initialize random rotation matrix
  set.seed(42)
  R <- matrix(rnorm(ncol(X_pca) * ncol(X_pca)), ncol(X_pca), ncol(X_pca))
  R <- qr.Q(qr(R))  # Orthogonalize
  
  for (iter in 1:n_iter) {
    # Step 1: Fix R, solve for B
    B <- sign(X_pca %*% R)
    
    # Step 2: Fix B, solve for R
    C <- t(B) %*% X_pca
    svd_C <- svd(C)
    R <- svd_C$u %*% t(svd_C$v)
  }
  
  # Final binary codes and rotation
  list(binary_code = sign(X_pca %*% R), rotation = R)
}

itq_result <- iterative_quantization(eigvecs)
binary_embedding <- itq_result$binary_code
rownames(binary_embedding) <- rownames(as.matrix(dataset_spectral$X))
cat("Total unique books in binary_embedding:", nrow(binary_embedding), "\n")

# Test the function with debugging
cat("=== Testing spectral tag recommendations ===\n")

# Function to recommend new tags using spectral analysis
add_new_tags_spectral <- function(eav, binary_embedding, 
                                  threshold = 0.7, min_neighbors = 3, max_recommendations = 10) {
  cat("Recommending new tags using spectral analysis...\n")
  
  # Debug: Check input data
  cat("Debug: binary_embedding dimensions:", dim(binary_embedding), "\n")
  cat("Debug: binary_embedding row names (first 5):", head(rownames(binary_embedding), 5), "\n")
  cat("Debug: binary_embedding row names class:", class(rownames(binary_embedding)), "\n")
  
  # Get existing tag assignments
  cat("1. Getting existing tag assignments...\n")
  existing_tags <- eav %>% 
    filter(feature == "tag") %>% 
    select(id, tag = value) %>% 
    group_by(id) %>% 
    summarize(existing_tags = list(tag), .groups = "drop")
  
  cat("Debug: existing_tags dimensions:", dim(existing_tags), "\n")
  cat("Debug: existing_tags id class:", class(existing_tags$id), "\n")
  cat("Debug: existing_tags id (first 5):", head(existing_tags$id, 5), "\n")
  
  cat("2. Analyzing nearest neighbors for each book...\n")
  
  # Calculate pairwise distances between all books using binary embeddings
  # Using Hamming distance for binary codes (number of different bits)
  n_books <- nrow(binary_embedding)
  recommendations <- list()
  
  for (book_idx in 1:n_books) {
    book_id <- rownames(binary_embedding)[book_idx]
    if (is.null(book_id)) book_id <- paste0("book_", book_idx)
    
    # Debug: Check first few books
    if (book_idx <= 3) {
      cat("Debug: Processing book", book_idx, "with ID:", book_id, "\n")
    }
    
    # Get existing tags for this book
    book_existing <- existing_tags %>% filter(id == book_id)
    book_existing_tags <- if (nrow(book_existing) > 0) book_existing$existing_tags[[1]] else character(0)
    
    if (book_idx <= 3) {
      cat("Debug: Book", book_id, "has", length(book_existing_tags), "existing tags\n")
    }
    
    # Calculate distances to all other books
    book_code <- binary_embedding[book_idx, ]
    distances <- apply(binary_embedding, 1, function(other_code) {
      sum(book_code != other_code)  # Hamming distance
    })
    
    # Find nearest neighbors (excluding self)
    distances[book_idx] <- Inf  # Exclude self
    nearest_indices <- order(distances)[1:min_neighbors]
    nearest_distances <- distances[nearest_indices]
    
    # Get tags from nearest neighbors
    neighbor_tags <- list()
    for (i in seq_along(nearest_indices)) {
      neighbor_id <- rownames(binary_embedding)[nearest_indices[i]]
      if (is.null(neighbor_id)) neighbor_id <- paste0("book_", nearest_indices[i])
      
      neighbor_existing <- existing_tags %>% filter(id == neighbor_id)
      if (nrow(neighbor_existing) > 0) {
        neighbor_tags[[i]] <- neighbor_existing$existing_tags[[1]]
      } else {
        neighbor_tags[[i]] <- character(0)
      }
    }
    
    # Calculate tag frequencies among neighbors
    all_neighbor_tags <- unlist(neighbor_tags)
    
    if (book_idx <= 3) {
      cat("Debug: Book", book_id, "has", length(all_neighbor_tags), "neighbor tags\n")
      cat("Debug: Neighbor distances:", nearest_distances, "\n")
    }
    
    if (length(all_neighbor_tags) > 0) {
      # Ensure tag names are properly handled as character strings
      all_neighbor_tags <- as.character(all_neighbor_tags)
      
      # Use manual counting instead of table() to avoid tag name issues
      unique_tags <- unique(all_neighbor_tags)
      tag_counts <- setNames(
        sapply(unique_tags, function(tag) sum(all_neighbor_tags == tag)),
        unique_tags
      )
      
      # Calculate tag scores based on frequency and distance
      tag_scores <- numeric(length(tag_counts))
      names(tag_scores) <- names(tag_counts)
      
      for (i in seq_along(tag_counts)) {
        tag <- names(tag_counts)[i]
        count <- tag_counts[i]
        # Weight by inverse distance and frequency
        avg_distance <- mean(nearest_distances)
        tag_scores[i] <- count / (avg_distance + 1)  # Add 1 to avoid division by zero
      }
      
      # Filter out existing tags and sort by score
      new_tags <- names(tag_scores)[!names(tag_scores) %in% book_existing_tags]
      
      if (book_idx <= 3) {
        cat("Debug: Book", book_id, "has", length(new_tags), "potential new tags\n")
        if (length(new_tags) > 0) {
          cat("Debug: Top tag scores:", head(sort(tag_scores[new_tags], decreasing = TRUE), 3), "\n")
        }
      }
      
      if (length(new_tags) > 0) {
        new_tag_scores <- tag_scores[new_tags]
        new_tag_scores <- sort(new_tag_scores, decreasing = TRUE)
        
        # Apply threshold and limit recommendations
        high_score_tags <- new_tag_scores[new_tag_scores >= threshold]
        
        if (book_idx <= 3) {
          cat("Debug: Book", book_id, "has", length(high_score_tags), "tags above threshold", threshold, "\n")
        }
        
        if (length(high_score_tags) > max_recommendations) {
          high_score_tags <- high_score_tags[1:max_recommendations]
        }
        
        if (length(high_score_tags) > 0) {
          recommendations[[book_id]] <- list(
            book_id = book_id,
            existing_tags = book_existing_tags,
            recommended_tags = names(high_score_tags),
            tag_scores = high_score_tags,
            nearest_neighbors = rownames(binary_embedding)[nearest_indices],
            neighbor_distances = nearest_distances
          )
        }
      }
    }
  }
  
  cat("3. Formatting results...\n")
  
  # Convert to data frame
  if (length(recommendations) > 0) {
    results <- do.call(rbind, lapply(recommendations, function(rec) {
      data.frame(
        book_id = rec$book_id,
        recommended_tag = rec$recommended_tags,
        score = rec$tag_scores,
        existing_tags = paste(rec$existing_tags, collapse = ", "),
        stringsAsFactors = FALSE
      )
    }))
    
    # Sort by score
    results <- results[order(results$score, decreasing = TRUE), ]
    
    cat("✓ Found", nrow(results), "tag recommendations for", length(recommendations), "books\n")
    return(results)
  } else {
    cat("✓ No tag recommendations found\n")
    return(data.frame(
      book_id = character(),
      recommended_tag = character(),
      score = numeric(),
      existing_tags = character(),
      stringsAsFactors = FALSE
    ))
  }
}

spectral_recommendations <- add_new_tags_spectral(
  eav = eav,
  binary_embedding = binary_embedding,
  threshold = 0.1,  # Very low threshold for testing
  min_neighbors = 3,
  max_recommendations = 5
)

print(spectral_recommendations) 