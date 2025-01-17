import argparse
import datetime
import logging
import os

import anthropic
import openai
import pgvector_rag

from confluence_rag_indexer import confluence

LOGGER = logging.getLogger(__name__)

DEFAULT_CUTOFF = datetime.datetime.now(tz=datetime.UTC) - \
    datetime.timedelta(days=355*5)

CLASSIFY_PROMPT = """\
<instructions>
Analyze this text and classify it as one of the following categories:

    - Meeting Notes
    - Project Documentation
    - Operational Event
    - Technical Documentation
    - User Documentation
    - Policy Documentation
    - Other

Do not return anything other than the category.
</instructions>
<content>
{content}
</content>
"""


class Indexer:

    def __init__(self,
                 confluence_domain: str,
                 confluence_email: str,
                 confluence_api_key: str,
                 anthropic_api_key: str,
                 openai_api_key: str,
                 postgres_url: str,
                 cutoff: datetime.datetime,
                 spaces: list[str],
                 ignore_classifications: list[str]):
        self.confluence = confluence.Client(
            confluence_domain, confluence_email, confluence_api_key)
        self.cuttoff = cutoff
        self.ignore_classifications = ignore_classifications
        self.openai = openai.Client(api_key=openai_api_key)
        self.rag = pgvector_rag.RAG(
            anthropic_api_key, openai_api_key, postgres_url)
        self.spaces = spaces

    def run(self):
        for space in self.spaces:
            for document in self.confluence.get_pages(space):
                for ignore in self.ignore_classifications:
                    if ignore in document.title:
                        LOGGER.info('Skipping "%s"', document.title)
                        continue
                response = self.openai.chat.completions.create(
                    messages = [
                        {
                            'role': 'user',
                            'content': CLASSIFY_PROMPT.format(
                                content=document.content)
                        }
                    ],
                    model='gpt-4o')
                category = response.choices[0].message.content

                if category in self.ignore_classifications:
                    LOGGER.info('Ignoring "%s": %s', document.title, category)
                    continue

                LOGGER.info('Classified "%s" as "%s"',
                            document.title, category)
                document.content = str(response.choices[0].message.content)

                self.rag.add_document(document)
                LOGGER.debug(document.content)


def valid_date(date_str: str) -> datetime.datetime:
    """Validate and convert a date or timestamp string to a datetime object.

    Args:
        date_str (str): The date or timestamp string provided via CLI.

    Returns:
        datetime.datetime: The corresponding datetime object.

    Raises:
        argparse.ArgumentTypeError: If the date_str format is invalid.

    """
    for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.datetime.strptime(date_str, fmt)  # noqa: DTZ007
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"Invalid date format: '{date_str}'. Use 'YYYY-MM-DD', "
        f"'YYYY-MM-DD HH:MM:SS', or 'YYYY-MM-DDTHH:MM:SS'.")


def parse_arguments(**kwargs) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser('Confluence to Rag Indexer')
    parser.add_argument(
        '--cutoff', type=valid_date, default=DEFAULT_CUTOFF,
        help='The cutoff date for Confluence content')
    parser.add_argument(
        '--confluence-domain', help='The Confluence domain',
        default=os.environ.get('CONFLUENCE_DOMAIN'))
    parser.add_argument(
        '--confluence-email', help='The Confluence email',
        default=os.environ.get('CONFLUENCE_EMAIL'))
    parser.add_argument(
        '--confluence-api-key', help='The Confluence API key',
        default=os.environ.get('CONFLUENCE_API_KEY'))
    parser.add_argument(
        '--anthropic-api-key', help='The OpenAI API key',
        default=os.environ.get('ANTHROPIC_API_KEY'))
    parser.add_argument(
        '--openai-api-key', help='The OpenAI API key',
        default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument(
        '--ignore-classifications', type=str, nargs='+',
        help='Ignore documents with these classifications',
        default=['Meeting Notes', 'Operational Event', 'Other']
    )
    parser.add_argument(
        '--postgres-url', help='The PostgreSQL URL',
        default=os.environ.get('POSTGRES_URL'))
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('space', nargs='*', help='The Confluence space(s)')
    return parser.parse_args(**kwargs)


def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    for logger in ['httpx', 'httpcore']:
        logging.getLogger(logger).setLevel(logging.WARNING)

    Indexer(
        args.confluence_domain,
        args.confluence_email,
        args.confluence_api_key,
        args.anthropic_api_key,
        args.openai_api_key,
        args.postgres_url,
        args.cutoff,
        args.space,
        args.ignore_classifications
    ).run()

if __name__ == '__main__':
    main()
