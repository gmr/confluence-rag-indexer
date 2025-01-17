import datetime
import logging
import re
import typing

import httpx
import markdownify
import pgvector_rag

from confluence_rag_indexer import version

LOGGER = logging.getLogger(__name__)


class Client:
    """Client for interacting with Confluence Cloud API."""

    def __init__(self, domain: str, email: str, api_token: str):
        """Initialize the Confluence client.

        Args:
            domain: Confluence cloud domain (e.g., 'your-domain.atlassian.net')
            email: Atlassian account email
            api_token: Atlassian API token

        """
        self.base_url = f'https://{domain}'
        self.base_url_v1 = f'{self.base_url}/wiki/rest/api'
        self.base_url_v2 = f'{self.base_url}/wiki/api/v2'
        self.http_client = httpx.Client(
            auth=(email, api_token),
            headers={
                'Accept': 'application/json',
                'User-Agent': f'confluence-rag-indexer/{version}'
            },
            timeout=30.0)

    def get_pages(self,
                  space_key: str,
                  max_age: datetime.datetime | None = None) \
            -> typing.Generator[pgvector_rag.Document, None, None]:
        """Retrieve all pages from a space using pagination.

        Args:
            space_id: The ID of the space
            max_age: Only return pages newer than this date if set

        Returns:
            list: List of page objects

        """
        space_id = self._get_space_id(space_key)
        try:
            yield from self._get_all_pages(space_id, max_age)
        except httpx.HTTPError as err:
            LOGGER.error('Error retrieving space content: %s', err)
            raise

    def _get_all_pages(self,
                       space_id: str,
                       max_age: datetime.datetime | None = None) \
            -> typing.Generator[pgvector_rag.Document, None, None]:
        """Retrieve all pages from a space using pagination.

        Args:
            space_id: The ID of the space
            limit: Number of items per page
            max_age: Only return pages newer than this date if set

        Returns:
            list: List of page objects

        """
        url = f'{self.base_url_v2}/spaces/{space_id}/pages?' \
              f'body-format=storage&status=current&sort=title'
        while True:
            LOGGER.debug('Fetching %s', url)
            response = self.http_client.get(url)
            response.raise_for_status()
            data = response.json()
            for page in data['results']:
                if not max_age or datetime.datetime.fromisoformat(
                        page['version']['createdAt']) > max_age:
                    yield self._get_page(page['id'])
            if not data.get('_links', {}).get('next'):
                break
            url = f"{self.base_url}{data.get('_links', {}).get('next')}"

    def _get_page(self, page_id: str) -> pgvector_rag.Document:
        """Retrieve a Confluence page by ID.

        Args:
            page_id: The ID of the page to retrieve

        Returns:
            Page: The page object

        """
        LOGGER.debug('Getting page %s', page_id)
        response = self.http_client.get(
            f'{self.base_url_v2}/pages/{page_id}?body-format=anonymous'
            f'_export_view&include-labels=true')
        response.raise_for_status()
        data = response.json()
        labels = None
        if data['labels']['results']:
            labels = ', '.join(
                [label['name'] for label in data['labels']['results']])
        return pgvector_rag.Document(
            title=data['title'],
            url=f"{self.base_url}/wiki{data['_links']['tinyui']}",
            last_modified_at=datetime.datetime.fromisoformat(
                data['version']['createdAt']),
            labels=labels,
            classification=None,
            content=self._convert_to_markdown(
                data['body']['anonymous_export_view']['value']))

    def _convert_to_markdown(self, value: str) -> str:
        """Convert the value to markdown removing linefeeds."""
        return re.sub(r'\n{3,}', r'\n', markdownify.markdownify(value))

    def _get_space_id(self, space_key: str) -> str:
        """Retrieve the ID of a Confluence space.

        Args:
            space_key: The key of the space to retrieve

        Returns:
            str: The ID of the space

        """
        LOGGER.debug('Getting space ID')
        response = self.http_client.get(
            f'{self.base_url_v1}/space/{space_key}')
        response.raise_for_status()
        return response.json()['id']
