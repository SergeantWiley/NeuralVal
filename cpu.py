import streamlit as st
import streamlit as st
from simulator import process_instruction

def show_cpu():
    st.title("4x CPU")

    st.write("""
    The **4x CPU** is a 4-bit digital processor designed to execute a set of six distinct instructions encoded in binary. It reads instructions from left to right, simplifying the translation process, although the CPU internally inverts the instructions for execution.

    ### Instruction Set

    Each instruction consists of 4 bits followed by a '1'. If an instruction requires data, the data in binary is written before the instruction.

    1. **Wait/Null (00001)**: 
    - This command either clears the 6-bit register or passes an empty instruction to delay the next operation by one clock cycle.

    2. **Reset (10011)**: 
    - Resets the processor to its default state.

    3. **Store (10101)**: 
    - Stores 4 bits of data in SRAM. The data is written before the store command.
    - **Example**: To store the value 7: `1110 10101`. (In this case, `1110` is the binary representation of 7, read from left to right.)

    4. **Read (01011)**: 
    - Reads and displays the data stored in SRAM.

    5. **Add (11001)**: 
    - Adds two numbers and displays the result but does not store the result in SRAM.
    - **Example**: `11001` (3) + `1110` (7) = `0101` (10)

    6. **S-Add (11101)**: 
    - Adds two numbers and stores the result in SRAM.
""")
    st.subheader("Image View")
    st.image("4cpu.png", width=300,caption="4-bit Corporal Proccessing unit (Named after the Designers Internet Alias)")

    st.title("CPU Instruction Input")

    st.write("Enter a 5-bit instruction to simulate the CPU.")

    # Input for instruction
    instruction = st.text_input("Instruction (5-bit)", value="")

    # Modified input box to accept two 4-bit data values separated by a space
    st.write("Enter 8-bit data (two 4-bit values separated by a space).")
    data_input = st.text_input("8-bit Data (4-bit 1 + 4-bit 2)", value="")

    if st.button("Submit"):
        # Validate the instructions and data
        if len(instruction) == 5:
            data_parts = data_input.split()
            if len(data_parts) == 2 and all(len(part) == 4 and all(bit in "01" for bit in part) for part in data_parts):
                # Flip the data bits to match the CPU's internal format
                flipped_data1 = data_parts[0][::-1]
                flipped_data2 = data_parts[1][::-1]

                # Handle different instructions
                if instruction == "10101":
                    combined_instruction = f"{flipped_data2} {instruction}"
                elif instruction == "11001":
                    # Use both data parts for the Add instruction
                    combined_instruction = f"{flipped_data1} {flipped_data2} {instruction}"
                elif instruction == "11101":
                    combined_instruction = f"{flipped_data1} {flipped_data2} {instruction}"
                else:
                    combined_instruction = f"{flipped_data1} {instruction}"
                
                result = process_instruction(combined_instruction)
                st.write(f"Instruction Result: {result}")
            else:
                st.write("Please enter valid 8-bit binary data (two 4-bit values separated by a space).")
        else:
            st.write("Please enter a valid 5-bit instruction.")