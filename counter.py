# Python script to generate all combinations of three numbers from 1 to 100
# and save them to a file
import itertools

# Output file name
output_file = "combinations.txt"

# Generate all combinations of 3 numbers from the range 1-100 and write to file
with open(output_file, 'w') as f:
    # Get the combinations generator
    combinations = itertools.combinations(range(1, 366), 2)
    
    # Write each combination to the file
    for combo in combinations:
        f.write(f"{combo[0]},{combo[1]}\n")

print(f"All combinations have been saved to '{output_file}'")
