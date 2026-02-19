import requests
from requests.adapters import HTTPAdapter

from src.utils import setup_logger
from src.config import Config

logger = setup_logger(__name__)


class HTTPClient:
    def __init__(self, base_url, headers):
        self.base_url = base_url
        self.headers = headers or {}

        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        adapter = HTTPAdapter()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _build_url(self, endpoint):
        if endpoint.startswith('http'):
            return endpoint
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _handle_response(self, response):
        try:
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            raise Exception(
                f"Request failed with status: {response.status_code}: {e}")
        except requests.RequestException as e:
            raise Exception(f"Network error: {e}")

    def get(self, endpoint, session=None, params=None, headers=None):
        url = self._build_url(endpoint)
        request_headers = {**self.headers, **(headers or {})}

        try:
            if session is None:
                response = self.session.get(
                    url, params=params, headers=request_headers, timeout=Config.REQ_TIMEOUT)
            else:
                response = session.get(
                    url, params=params, headers=request_headers, timeout=Config.REQ_TIMEOUT)
            return self._handle_response(response)
        except requests.RequestException as e:
            raise Exception(f"Network error: {e}")
