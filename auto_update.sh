#!/bin/bash

# Infinite loop to run the script every hour
while true; do

    # Save the current time for logging
    echo "$(date): Script started"

    # Save the current HEAD hash
    current_head=$(git rev-parse HEAD)

    # Pull the latest changes from the repository
    git stash
    git pull -f

    # Get the new HEAD hash
    new_head=$(git rev-parse HEAD)

    # Check if the new HEAD is different from the current HEAD
    if [ "$current_head" != "$new_head" ]; then
        # The HEAD has changed, meaning there's a new version
        echo "$(date): New version detected, restarting the validator."
        pm2 restart validator_nicheimage
    else
        # No new version, no action needed
        echo "$(date): No new version detected, no restart needed."
    fi

    # Sleep for 1 hour (3600 seconds)
    sleep 3600

done
