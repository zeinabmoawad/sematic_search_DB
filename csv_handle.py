# Calculate the byte offset of a specific row in a CSV file with fixed-size rows
def calculate_byte_offset(line_number, row_size):
    # Calculate the byte offset based on line number and row size
    def get_digit_count(number):
    # Calculate the number of digits in a number
        return len(str(number))
    
    count = get_digit_count(line_number)
    if count == 1:
        return line_number  * row_size + line_number
    offset = 0
    for i in range(count-1):
        if i == 0:
            offset = 10
        else:
            offset += (i+1)* (10 ** (i+1) - 10 ** i)
    offset += count * (line_number-10**(count-1))
    print("offset: ",offset)
    return line_number * row_size+offset



# Assuming you know the line number and the size of each row in bytes
line_number = 900000 # Replace this with the desired line number
row_size = 639  # Replace this with the size of each row in bytes
# row_size = 70*10+9
# Calculate the byte offset of the specific row
byte_offset = calculate_byte_offset(line_number, row_size)

# Open the CSV file and move to the specific byte offset
with open('saved_db.csv', 'r', encoding='utf-8') as csv_file:
    # Move to the calculated byte offset in the file
    csv_file.seek(byte_offset)
    # Read the row at the specified byte offset
    specific_row = csv_file.readline().strip()
    # print byte offset at which pointer is
    print("byte: ",csv_file.tell())
    # read from the byte offset 5 more rows
    # print size of each row
    print("size of each row: ",len(specific_row))
    # for i in range(10000):
    #     specific_row += csv_file.readline().strip()

print(specific_row)