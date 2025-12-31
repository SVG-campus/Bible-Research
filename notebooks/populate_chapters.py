import pandas as pd
import os
import json

# Paths
base_notebook_path = "/workspaces/Bible-Research/notebooks/06_complete_bible_reading"
csv_path = "bible_master.csv"

def populate_notebooks():
    print("📖 READING BIBLE DATABASE...")
    if not os.path.exists(csv_path):
        print("❌ Error: bible_master.csv not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Iterate through every folder
    for root, dirs, files in os.walk(base_notebook_path):
        for file in files:
            if file.endswith(".ipynb") and file != "00_setup.ipynb": 
                try:
                    # Clean filename
                    file_clean = file.replace(".ipynb", "")
                    
                    if "_Ch" in file_clean:
                        # FIX: Use rsplit to split only on the LAST occurrence of "_Ch"
                        # This prevents "1_Chronicles_Ch01" from breaking if we split on just "_"
                        parts = file_clean.rsplit("_Ch", 1)
                        
                        if len(parts) == 2:
                            book_part, chapter_part = parts
                            chapter_num = int(chapter_part)
                            
                            # Normalize Book Name
                            # 1_Chronicles -> 1 Chronicles
                            book_name_query = book_part.replace("_", " ")
                            
                            # Filter DataFrame
                            subset = df[
                                (df['Book_Name'].str.lower() == book_name_query.lower()) & 
                                (df['Chapter'] == chapter_num)
                            ]
                            
                            if not subset.empty:
                                # Build Markdown Text
                                text_block = "### 📜 Chapter Text\n\n"
                                for _, row in subset.iterrows():
                                    text_block += f"**{row['Verse']}** {row['Text']}\n\n"
                                
                                # Load Notebook
                                nb_path = os.path.join(root, file)
                                with open(nb_path, 'r') as f:
                                    nb_data = json.load(f)
                                
                                # Ensure 'cells' list exists
                                if not nb_data.get('cells'):
                                    nb_data['cells'] = []
                                    
                                # INSERTION LOGIC
                                status = "Processed"
                                if len(nb_data['cells']) > 0:
                                    last_cell_source = nb_data['cells'][-1].get('source', [])
                                    # Handle list or string source
                                    if isinstance(last_cell_source, list):
                                        last_cell_content = "".join(last_cell_source)
                                    else:
                                        last_cell_content = str(last_cell_source)
                                        
                                    if "4. The Chapter Text" in last_cell_content or "📜 Chapter Text" in last_cell_content:
                                        # Update
                                        nb_data['cells'][-1]['source'] = [text_block]
                                        status = "Updated"
                                    else:
                                        # Append
                                        new_cell = {"cell_type": "markdown", "metadata": {}, "source": [text_block]}
                                        nb_data['cells'].append(new_cell)
                                        status = "Appended"
                                else:
                                    # Create new
                                    new_cell = {"cell_type": "markdown", "metadata": {}, "source": [text_block]}
                                    nb_data['cells'].append(new_cell)
                                    status = "Created"
                                    
                                # Save
                                with open(nb_path, 'w') as f:
                                    json.dump(nb_data, f, indent=1)
                                    
                                print(f"   ✅ {status}: {file} ({len(subset)} verses)")
                            else:
                                pass # Silence to keep logs clean
                        else:
                            print(f"   ⚠️ Filename format unexpected: {file}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {file}: {e}")

    print("\n🎉 ALL NOTEBOOKS POPULATED.")

if __name__ == "__main__":
    populate_notebooks()
