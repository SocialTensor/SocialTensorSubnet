import re

def extract_numerical_part(text):
    # Use regex to find the first occurrence of a number    
    match = re.search(r'[-+]?\d*\.?\d+|\d+', text)
    if match:
        return match.group(0)
    else:
        # Return a specific message or None if no numerical value is found
        return "No numerical value found"