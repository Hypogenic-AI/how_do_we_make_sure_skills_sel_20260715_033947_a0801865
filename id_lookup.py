import arxiv
ids = ["2307.02477","2304.15004","2310.01798","2203.14465"]
client = arxiv.Client()
for r in client.results(arxiv.Search(id_list=ids)):
    print(r.entry_id.split('/')[-1], "|", r.title.strip(), "|", r.published.year)
