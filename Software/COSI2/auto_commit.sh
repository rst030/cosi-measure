#!/bin/bash

# Make an array of repository directories,which you want to auto-commit
repo_dirs=(
    "/home/cosi/cosi-measure"
    # Add more repository directories,if needed
    
)

# Function to navigate to a repository directory, add changes,
# and commit every few hours
perform_git_operations() {
    local repo_dir=$1

    # Navigate to the repo directory
    cd "$repo_dir" || return

    # Add all changes to the staging area
    git add .

    # Commit the changes with a timestamp
    #while true; do
    git commit -m "COSI2 automated commit on $(date)" # change msg as needed
    #sleep 6h # change 6 with any value
    #done
}

# Loop over the repo directories and run the same script for each one
for repo_dir in "${repo_dirs[@]}"; do
    # Run the script initially for each repo directory
    perform_git_operations "$repo_dir" &
done

git push origin main

# Loop indefinitely and push any changes, every few minutes
#while true; do
#    sleep 15m # change 15 with any value

    # Loop through the repo directories and push changes for each one
#    for repo_dir in "${repo_dirs[@]}"; do
        # Navigate to the repository directory
#        cd "$repo_dir" || continue

        # Push the changes to the remote repo's 'main' branch
#        git push origin main
#    done
#done