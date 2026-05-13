import json

path = r"C:\Users\amitu\python\SEFET 2025\forecasting_v4.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(r"d:\DC microgrid DMPC for test\extracted_nb.txt", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb.get("cells", [])):
        out.write(f"--- Cell {i} ({cell.get('cell_type', 'unknown')}) ---\n")
        out.write("".join(cell.get("source", [])))
        out.write("\n\n")