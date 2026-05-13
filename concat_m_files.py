import os
import glob

directory = r"d:\DC microgrid DMPC for test"
output_file = os.path.join(directory, "all_m_files.txt")

m_files = glob.glob(os.path.join(directory, "*.m"))

with open(output_file, "w", encoding="utf-8", errors="ignore") as f_out:
    for m_file in m_files:
        f_out.write(f"\n{'='*80}\n")
        f_out.write(f"FILE: {os.path.basename(m_file)}\n")
        f_out.write(f"{'='*80}\n")
        
        with open(m_file, "r", encoding="utf-8", errors="ignore") as f_in:
            f_out.write(f_in.read())

print(f"Concatenated {len(m_files)} files into {output_file}")