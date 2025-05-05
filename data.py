import os
import pandas as pd
import xml.etree.ElementTree as ET

csv_path = "results-2025-04-16T12-29-44.csv"
xml_folder = "full_text"
output_csv = "merged_patents.csv"

df = pd.read_csv(csv_path)
df["applicationNumberText"] = df["applicationNumberText"].astype(str)
df["fullDescription"] = "N/A"

descriptions_by_app_number = {}

for filename in os.listdir(xml_folder):
    if filename.endswith(".xml"):
        path = os.path.join(xml_folder, filename)
        try:
            tree = ET.parse(path)
            root = tree.getroot()

            app_number = root.findtext(".//application-reference/document-id/doc-number")
            if not app_number:
                continue
            app_number = app_number.strip()

            desc_elem = root.find(".//description")
            if desc_elem is not None:
                paragraphs = desc_elem.findall(".//p")
                description = "\n".join(p.text for p in paragraphs if p is not None and p.text)
                if description:
                    descriptions_by_app_number[app_number] = description

        except Exception as e:
            print(f"Error in file: {filename}: {e}")

df["fullDescription"] = df["applicationNumberText"].map(descriptions_by_app_number).fillna("N/A")

df.to_csv(output_csv, index = False)
print(f" desc for {len(descriptions_by_app_number)} patent. saved to '{output_csv}'")
