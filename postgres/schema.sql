CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  document_id UUID NOT NULL PRIMARY KEY,
  created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  modified_at TIMESTAMP WITH TIME ZONE,
  title       TEXT,
  url         TEXT,
  metadata    TEXT,
  content     TEXT NOT NULL
);

CREATE TABLE chunks (
  document_id UUID NOT NULL,
  chunk       INT4 NOT NULL DEFAULT 0,
  embedding   vector(1536),
  PRIMARY KEY (document_id, chunk),
  FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE ON UPDATE CASCADE
);
