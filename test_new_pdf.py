import sys
import warnings
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

warnings.filterwarnings('ignore')

from src.rag import search

def test():
    query1 = "هل يمكن خلط زنتاك مع بريمبران؟"
    print(f"\n======================\nQUERY: {query1}\n======================")
    results1 = search(query1, limit=40)
    for result in results1[:3]:
        doc = result['document']
        metadata = result['metadata']
        print(f"Source: {metadata.get('source', '')}")
        print(f"Page: {metadata.get('page', '')}")
        print(f"Text Snippet: {doc[:200]}...")

    query2 = "ما هي الأدوية التي لا يمكن خلطها مع سبازموفين؟"
    print(f"\n======================\nQUERY: {query2}\n======================")
    results2 = search(query2, limit=40)
    for result in results2[:3]:
        doc = result['document']
        metadata = result['metadata']
        print(f"Source: {metadata.get('source', '')}")
        print(f"Page: {metadata.get('page', '')}")
        print(f"Text Snippet: {doc[:200]}...")

if __name__ == '__main__':
    test()
