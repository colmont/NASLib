def analyze_file(file_path):
    gp_heat_nlml_values = []
    gpwl_nlml_values = []
    count_gp_heat_greater = 0
    count_gpwl_greater = 0
    total_pairs = 0

    with open(file_path, 'r') as file:
        for line in file:
            if 'gp_heat_nlml:' in line:
                value = float(line.split(':')[-1].strip())
                gp_heat_nlml_values.append(value)
            elif 'gpwl_nlml:' in line:
                value = float(line.split(':')[-1].strip())
                gpwl_nlml_values.append(value)
                total_pairs += 1
                if gp_heat_nlml_values and gpwl_nlml_values:
                    if gp_heat_nlml_values[-1] > gpwl_nlml_values[-1]:
                        count_gp_heat_greater += 1
                    elif gpwl_nlml_values[-1] > gp_heat_nlml_values[-1]:
                        count_gpwl_greater += 1

    # Calculating the percentages
    if total_pairs > 0:
        percentage_gp_heat_greater = (count_gp_heat_greater / total_pairs) * 100
        percentage_gpwl_greater = (count_gpwl_greater / total_pairs) * 100
    else:
        percentage_gp_heat_greater = 0
        percentage_gpwl_greater = 0

    return percentage_gp_heat_greater, percentage_gpwl_greater

# Use the function like this:
file_path = 'log3.out'
percentage_gp_heat_greater, percentage_gpwl_greater = analyze_file(file_path)
print(f"gp_heat_nlml > gpwl_nlml: {percentage_gp_heat_greater}%")
print(f"gpwl_nlml > gp_heat_nlml: {percentage_gpwl_greater}%")
