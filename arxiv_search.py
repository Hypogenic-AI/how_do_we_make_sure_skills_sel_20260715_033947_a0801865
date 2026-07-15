import arxiv, sys, textwrap
client = arxiv.Client(page_size=25, delay_seconds=3)
q = sys.argv[1]
maxr = int(sys.argv[2]) if len(sys.argv)>2 else 12
search = arxiv.Search(query=q, max_results=maxr, sort_by=arxiv.SortCriterion.Relevance)
for r in client.results(search):
    aid = r.entry_id.split('/')[-1]
    print(f"ID: {aid}")
    print(f"TITLE: {r.title.strip()}")
    print(f"YEAR: {r.published.year}  CAT: {r.primary_category}")
    print("ABS:", textwrap.shorten(r.summary.replace(chr(10),' '), 400))
    print("-"*70)
