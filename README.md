confluence-rag-indexer
======================
Index confluence into a RAG database using pg_vector.

Categorizes documents prior to indexing and allows you to skip specific classifications like "Meeting Notes".

Installation
------------
```bash
pip install confluence-rag-indexer
```

Usage
-----
```bash
usage: Confluence to Rag Indexer [-h] [--cutoff CUTOFF] [--confluence-domain CONFLUENCE_DOMAIN] [--confluence-email CONFLUENCE_EMAIL]
                                 [--confluence-api-key CONFLUENCE_API_KEY] [--openai-api-key OPENAI_API_KEY]
                                 [--ignore-classifications IGNORE_CLASSIFICATIONS [IGNORE_CLASSIFICATIONS ...]] [--postgres-url POSTGRES_URL] [-v]
                                 [space ...]

positional arguments:
  space                 The Confluence space(s)

options:
  -h, --help            show this help message and exit
  --cutoff CUTOFF       The cutoff date for Confluence content
  --confluence-domain CONFLUENCE_DOMAIN
                        The Confluence domain
  --confluence-email CONFLUENCE_EMAIL
                        The Confluence email
  --confluence-api-key CONFLUENCE_API_KEY
                        The Confluence API key
  --openai-api-key OPENAI_API_KEY
                        The OpenAI API key
  --ignore-classifications IGNORE_CLASSIFICATIONS [IGNORE_CLASSIFICATIONS ...]
                        Ignore documents with these classifications
  --postgres-url POSTGRES_URL
                        The PostgreSQL URL
  -v, --verbose
```
