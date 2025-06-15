#!/bin/bash
set -o pipefail
set -e
Rscript -e "devtools::test()"
fswatch -vr -e tests/testthat/_snaps R/ tests/ | while read f; do 
    echo "\nFile changed: $f\n"
    sleep 0.1
    Rscript -e "devtools::test()"
done 