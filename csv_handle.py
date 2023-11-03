# Calculate the byte offset of a specific row in a CSV file with fixed-size rows
def calculate_byte_offset(line_number, row_size):
    # Calculate the byte offset based on line number and row size
    byte_offset = (line_number - 1) * row_size
    return byte_offset

# Assuming you know the line number and the size of each row in bytes
line_number = 5  # Replace this with the desired line number
row_size = 100  # Replace this with the size of each row in bytes

# Calculate the byte offset of the specific row
byte_offset = calculate_byte_offset(line_number, row_size)

# Open the CSV file and move to the specific byte offset
with open('large_csv_file.csv', 'r', encoding='utf-8') as csv_file:
    # Move to the calculated byte offset in the file
    csv_file.seek(byte_offset)
    # Read the row at the specified byte offset
    specific_row = csv_file.readline().strip()

print(specific_row)