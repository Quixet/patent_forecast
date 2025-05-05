import os

input_file = "ipg250415/ipg250415.xml"
output_dir = "split_xml"
os.makedirs(output_dir, exist_ok=True)

patent_counter = 0
current_patent = []

with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        current_patent.append(line)

        if "</us-patent-grant>" in line:
            # завершили патент
            patent_counter += 1
            out_path = os.path.join(output_dir, f"patent_{patent_counter}.xml")
            with open(out_path, "w", encoding="utf-8") as out_file:
                out_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                out_file.writelines(current_patent)
            current_patent = []

print(f"✅ Розбито {patent_counter} патентів у папку '{output_dir}'")
