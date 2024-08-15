# CPUx.py

# Global variable to simulate SRAM
SRAM = None

def process_instruction(instruction):
    global SRAM

    if " " in instruction:
        parts = instruction.split(" ")
        if len(parts) == 2:  # For Store and Read instructions
            data = parts[0]
            instr = parts[1]
        elif len(parts) == 3:  # For Add instruction
            data1 = parts[0]
            data2 = parts[1]
            instr = parts[2]
    else:
        data = ""
        instr = instruction
    # Process the instruction based on the provided bits
    if instr == "00001":
        SRAM = None  # Clearing the register/SRAM for wait/null
        return "Null/Wait: Cleared register or delayed operation."
    elif instr == "10011":
        SRAM = None  # Reset the SRAM to its default state
        return "Reset: Processor reset to default state."
    elif instr == "10101":
        SRAM = data[::-1]  # Store the data in SRAM (in user order)
        return f"Store: Stored {SRAM} in SRAM."
    elif instr == "01011":
        if SRAM is not None:
            return f"Read: The stored data in SRAM is {SRAM[::-1]}."  # Return data in user-friendly order
        else:
            return "Read: SRAM is empty."
    elif instr == "11001":
        # Simulating addition of the two 4-bit data values
        result = int(data1, 2) + int(data2, 2)
        result_binary = bin(result)[2:].zfill(4)
        #SRAM = result_binary[::-1]  # Store result in SRAM
        return f"Add: Result is {result_binary} ({int(str(result))})"
    elif instr == "11101":
        # Simulating storage after addition
        result = int(data1, 2) + int(data2, 2)
        result_binary = bin(result)[2:].zfill(4)
        SRAM = result_binary[::-1]  # Store result in SRAM
        return f"S-Add: Result is {result_binary} and stored in SRAM."
    else:
        return "Unknown instruction"
