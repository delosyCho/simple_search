import Searcher

searcher = Searcher.SearchEngine()
idxs, _ = searcher.search_document('김태호 PD의 메인 MBC 프로그램', top_Document_Number=20)

for idx in idxs:
    print(searcher.Titles[idx])