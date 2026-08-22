import sys
import json
from pathlib import Path
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

sys.path.append(str(Path(__file__).parent))
from src.ingest import extract_text_from_pdf

def test():
    p = Path('data/raw/قواعد_خلط_الامبولات.pdf')
    pages = extract_text_from_pdf(p)
    with open('temp_test.txt', 'w', encoding='utf-8') as f:
        for page in pages:
            f.write(f"--- Page {page['page']} ---\n")
            f.write(page['text'])
            f.write("\n\n")

if __name__ == '__main__':
    test()
