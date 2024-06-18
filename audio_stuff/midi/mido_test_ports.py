import mido

# Get a list of all input and output names
input_names = mido.get_input_names()
output_names = mido.get_output_names()

# Print each input and output name
print("Input Ports:")
for name in input_names:
    print(name)

print("\nOutput Ports:")
for name in output_names:
    print(name)
