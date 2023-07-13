import os

def fix_sigma(file_dir, file_name):
    lines = None
    with open(os.path.join(file_dir, file_name), "r") as file:
        lines = file.readlines()
    
    split_line = lines[0].split(" ")
    split_line = [e for e in split_line if e.strip()]
    if "\n" in split_line[-1]:
        lines[0] = split_line[-1]
    else:
        lines[0] = split_line[-1] + "\n"

    for i in range(1, len(lines)):
        lines[i] = lines[i].replace(" ", "") # remove only normal ascii space

    with open(os.path.join(file_dir, file_name), "w") as file:
        file.writelines(lines)

# NOTE: it seems like not all sigmas have the same issue, so to fix them, I would probably have to look one by one to see what they have...
if __name__ == "__main__":
    file_dir = "sigmas"
    file_names = os.listdir(file_dir)

    for file_name in file_names:
        fix_sigma(file_dir, file_name)