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
    for i, result in enumerate(results1[:10]):
        print(f"[{i}] Source: {result['metadata'].get('source')} | Score: {result['score']:.2f}")
        print(f"    S:{result['semantic_score']:.2f} | K:{result['keyword_score']} | P:{result['phrase_score']} | D:{result['definition_score']}")

if __name__ == '__main__':
    test()
