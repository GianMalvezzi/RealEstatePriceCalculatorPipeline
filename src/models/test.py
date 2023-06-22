import omegaconf

# Example ListConfig
my_list_config = omegaconf.listconfig.ListConfig([7, 2, 9, 1, 4])

print(type(my_list_config))
# Convert to a standard Python list
my_list = list(my_list_config)

# Sort the list
sorted_list = sorted(my_list)  # or my_list.sort()

# Convert back to ListConfig
sorted_list_config = omegaconf.listconfig.ListConfig(sorted_list)

print(sorted_list_config)
