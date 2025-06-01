setup_formatting <- function(){
  knitr::opts_chunk$set(
    fig.pos = 'h',
    echo = FALSE,
    warning = FALSE,
    message = FALSE,
    autodep = TRUE,
    cache = TRUE,
    fig.width = 6,  # was 6
    fig.asp = 0.618,  # was 0.618
    out.width = "70%",
    fig.align = "center",
    fig.show = "hold")
  remove(list = ls()) # clear environment
}