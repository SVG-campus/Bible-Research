import pandas as pd
import os
import json
from textblob import TextBlob

# Paths
base_notebook_path = "/workspaces/Bible-Research/notebooks/06_complete_bible_reading"
csv_path = "bible_master.csv"

def get_sentiment_label(score):
    if score > 0.1: return "Positive 🟢"
    if score < -0.1: return "Negative 🔴"
    return "Neutral ⚪"

def apply_templates():
    print("🎨 APPLYING STUDY TEMPLATES...")
    
    if not os.path.exists(csv_path):
        print("❌ Error: bible_master.csv not found.")
        return

    df = pd.read_csv(csv_path)

    for root, dirs, files in os.walk(base_notebook_path):
        for file in files:
            if file.endswith(".ipynb") and file != "00_setup.ipynb":
                try:
                    # Parse Filename
                    file_clean = file.replace(".ipynb", "")
                    if "_Ch" in file_clean:
                        parts = file_clean.rsplit("_Ch", 1)
                        if len(parts) != 2: continue
                        
                        book_part, chapter_part = parts
                        chapter_num = int(chapter_part)
                        book_name_query = book_part.replace("_", " ")

                        # Get Chapter Data
                        subset = df[
                            (df['Book_Name'].str.lower() == book_name_query.lower()) & 
                            (df['Chapter'] == chapter_num)
                        ]

                        if subset.empty: continue

                        # --- CALCULATE STATS ---
                        full_text = " ".join(subset['Text'].astype(str))
                        word_count = len(full_text.split())
                        sentiment_score = TextBlob(full_text).sentiment.polarity
                        sentiment_label = get_sentiment_label(sentiment_score)
                        
                        # --- DEFINE NEW CELLS ---
                        
                        # 1. Header with Stats
                        header_cell = {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [
                                f"# {book_name_query} Chapter {chapter_num}\n",
                                f"**📊 Stats:** {word_count} words | Sentiment: {sentiment_label} ({sentiment_score:.2f})\n",
                                "---\n"
                            ]
                        }

                        # 2. Observation Section
                        observation_cell = {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [
                                "## 📝 Observation\n",
                                "*What does the text say? (Who, What, Where, When)*\n\n",
                                "- \n",
                                "- \n"
                            ]
                        }

                        # 3. Interpretation Section
                        interpretation_cell = {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [
                                "## 💡 Interpretation\n",
                                "*What does the text mean? (Context, Cross-references)*\n\n",
                                "- \n"
                            ]
                        }

                        # 4. Application Section
                        application_cell = {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [
                                "## 🚀 Application\n",
                                "*How does this apply to my life today?*\n\n",
                                "- \n"
                            ]
                        }
                        
                        # 5. Prayer Section
                        prayer_cell = {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [
                                "## 🙏 Prayer\n",
                                "*Write a prayer based on this chapter.*\n\n",
                                "> \n"
                            ]
                        }

                        # Load Notebook
                        nb_path = os.path.join(root, file)
                        with open(nb_path, 'r') as f:
                            nb_data = json.load(f)

                        # --- INTELLIGENT MERGE ---
                        # We want to keep the "Chapter Text" at the bottom (which we added previously)
                        # We want to keep any user notes if they exist (unlikely yet, but good practice)
                        
                        # Find the scripture cell (from previous script)
                        scripture_cell = None
                        other_cells = []
                        
                        if 'cells' in nb_data:
                            for cell in nb_data['cells']:
                                src = "".join(cell.get('source', []))
                                if "📜 Chapter Text" in src:
                                    scripture_cell = cell
                                else:
                                    # Identify if this is a "default" empty cell vs user content
                                    # For now, we will OVERWRITE the top section to ensure formatting,
                                    # assuming you haven't written notes yet.
                                    pass

                        # Rebuild the notebook structure
                        new_cells = [
                            header_cell,
                            observation_cell,
                            interpretation_cell,
                            application_cell,
                            prayer_cell
                        ]
                        
                        # Add scripture back at the bottom
                        if scripture_cell:
                            new_cells.append(scripture_cell)

                        nb_data['cells'] = new_cells

                        # Save
                        with open(nb_path, 'w') as f:
                            json.dump(nb_data, f, indent=1)
                            
                        print(f"   ✨ Formatted: {file}")

                except Exception as e:
                    print(f"   ❌ Error {file}: {e}")

    print("\n🎉 ALL NOTEBOOKS FORMATTED.")

if __name__ == "__main__":
    apply_templates()
