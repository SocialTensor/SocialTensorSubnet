#!/bin/bash


while true; do
    # Log the start of the script execution
    echo "$(date): Script started"

    # Save the current HEAD hash
    current_head=$(git rev-parse HEAD)

    # Pull the latest changes from the repository
    git stash
    git pull -f
    git reset --hard origin/main

    # Get the new HEAD hash
    new_head=$(git rev-parse HEAD)

    # Check if the new HEAD is different from the current HEAD
    if [ "$current_head" != "$new_head" ]; then
        # The HEAD has changed, meaning there's a new version
        echo "$(date): New version detected, installing requirements and restarting the validator."
        pip install -e .
        pm2 restart validator_nicheimage
    else
        # No new version, no action needed
        echo "$(date): No new version detected, no restart needed."
    fi

    # Sleep until the beginning of the next hour
    sleep 3600
done
