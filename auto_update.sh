#!/bin/bash


while true; do
    # Calculate the amount of time to sleep until the beginning of the next hour
    current_minute=$(date +%M)
    current_second=$(date +%S)
    sleep_seconds=$(( (60 - current_minute) * 60 - current_second ))

    # If it's exactly on the hour, don't sleep
    if [ $sleep_seconds -ne 3600 ]; then
        sleep $sleep_seconds
    fi

    # Log the start of the script execution
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

    # Sleep until the beginning of the next hour
    sleep 3600
done
