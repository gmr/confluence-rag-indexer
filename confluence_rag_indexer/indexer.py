import argparse
import datetime
import logging
import os
import re
import typing

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

    - Changelog
    - Compliance Documentation
    - Employee Benefits Documentation
    - Employee Handbook
    - How-To Documentation
    - Job Description
    - Job Responsibilities
    - Meeting Notes
    - Operational Event
    - Operational Plan
    - Organizational Plan
    - Policy Documentation
    - Project Documentation
    - Technical Documentation
    - User Documentation
    - Vendor Information
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
                 ignore_classifications: list[str],
                 ignore_pattern: typing.Optional[str],
                 skip: int):
        self.confluence = confluence.Client(
            confluence_domain, confluence_email, confluence_api_key)
        self.cuttoff = cutoff
        self.ignore_classifications = ignore_classifications
        self.ignore_pattern = \
            re.compile(ignore_pattern) if ignore_pattern else None
        self.openai = openai.Client(api_key=openai_api_key)
        self.rag = pgvector_rag.RAG(
            anthropic_api_key, openai_api_key, postgres_url)
        self.spaces = spaces
        self.skip = skip

    def run(self):
        for space in self.spaces:
            count = 0
            for document in self.confluence.get_pages(space):
                count += 1
                if count <= self.skip:
                    LOGGER.info('Skipping "%s"', document.title)
                    continue
                if self.ignore_pattern and \
                        self.ignore_pattern.search(document.title):
                    LOGGER.info('Ignoring "%s"', document.title)
                    continue
                for ignore in self.ignore_classifications:
                    if ignore in document.title:
                        LOGGER.info('Ignoring "%s" due to classification',
                                    document.title)
                        continue
                try:
                    response = self.openai.chat.completions.create(
                        messages = [
                            {
                                'role': 'user',
                                'content': CLASSIFY_PROMPT.format(
                                    content=document.content)
                            }
                        ],
                        model='gpt-4o')
                except openai.BadRequestError as err:
                    LOGGER.error('Error classifying document: %s', err)
                    category = 'Other'
                else:
                    document.classification = \
                        str(response.choices[0].message.content)

                if document.classification in self.ignore_classifications:
                    LOGGER.info('Ignoring "%s": %s',
                                document.title, document.classification)
                    continue

                LOGGER.info('Classified "%s" as "%s"',
                            document.title, document.classification)
                self.rag.add_document(document)

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
        default=['Meeting Notes', 'Operational Event', 'Changelog'])
    parser.add_argument(
        '--ignore-pattern', type=str,
        help='Ignore documents by title with this regex pattern',)
    parser.add_argument(
        '--postgres-url', help='The PostgreSQL URL',
        default=os.environ.get('POSTGRES_URL'))
    parser.add_argument(
        '--skip', type=int, help='Skip the first N documents', default=0)
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
        args.ignore_classifications,
        args.ignore_pattern,
        args.skip
    ).run()

if __name__ == '__main__':
    main()
