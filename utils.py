def add_tabs_and_reconcatenate(input_string, number_of_spaces):
    parts = input_string.split("\n")
    space_prefix = " " * number_of_spaces
    modified_parts = [space_prefix + part for part in parts]
    return "\n".join(modified_parts)
