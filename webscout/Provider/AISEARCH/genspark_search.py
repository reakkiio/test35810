import cloudscraper
from uuid import uuid4
import json
import re
from typing import TypedDict, List, Iterator, cast, Dict, Optional, Generator, Union, Any
import requests

from webscout.AIbase import AISearch, SearchResponse
from webscout import exceptions
from webscout.litagent import LitAgent


class SourceDict(TypedDict, total=False):
    url: str
    title: str
    snippet: str
    favicon: str
    # Add more fields as needed

class StatusUpdateDict(TypedDict):
    type: str
    message: str

class StatusTopBarDict(TypedDict, total=False):
    type: str
    data: dict

class PeopleAlsoAskDict(TypedDict, total=False):
    question: str
    answer: str

class ResultSummaryDict(TypedDict, total=False):
    source: str
    rel_score: float
    score: float
    llm_id: str
    cogen_name: str
    ended: bool

class Genspark(AISearch):
    """
    Strongly typed Genspark AI search API client.
    
    Genspark provides a powerful search interface that returns AI-generated SearchResponses
    based on web content. It supports both streaming and non-streaming SearchResponses.

    After a search, several attributes are populated with extracted data:
    - `search_query_details` (dict): Information about the classified search query.
    - `status_updates` (list): Log of status messages during the search.
    - `final_search_results` (list): Organic search results if provided by the API.
    - `sources_used` (list): Unique web sources used for the answer.
    - `people_also_ask` (list): "People Also Ask" questions.
    - `agents_guide` (dict): Information about agents used.
    - `result_summary` (dict): Summary of result IDs and scores.
    - `raw_events_log` (list): If enabled, logs all raw JSON events from the stream.
    
    Basic Usage:
        >>> from webscout import Genspark
        >>> ai = Genspark()
        >>> # Non-streaming example (text SearchResponse)
        >>> SearchResponse_text = ai.search("What is Python?")
        >>> print(SearchResponse_text)
        Python is a high-level programming language...
        >>> # Access additional data:
        >>> # print(ai.sources_used)
        
        >>> # Streaming example (mixed content: text SearchResponse objects and event dicts)
        >>> for item in ai.search("Tell me about AI", stream=True):
        ...     if isinstance(item, SearchResponse):
        ...         print(item, end="", flush=True)
        ...     else:
        ...         print(f"\n[EVENT: {item.get('event')}]") 
        Artificial Intelligence is...
        [EVENT: status_update]
        ...
        
        >>> # Raw streaming SearchResponse format
        >>> for raw_event_dict in ai.search("Hello", stream=True, raw=True):
        ...     print(raw_event_dict)
        {'type': 'result_start', ...}
        {'type': 'result_field_delta', 'field_name': 'streaming_detail_answer[0]', 'delta': 'Hello', ...}
    
    Args:
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        proxies (dict, optional): Proxy configuration for requests. Defaults to None.
        max_tokens (int, optional): Maximum tokens to generate (Note: This param is part of Genspark class but not directly used in API call shown). Defaults to 600.
        log_raw_events (bool, optional): If True, all raw JSON events from the stream are logged to `self.raw_events_log`. Defaults to False.
    """

    session: cloudscraper.CloudScraper
    max_tokens: int
    chat_endpoint: str
    stream_chunk_size: int
    timeout: int
    log_raw_events: bool
    headers: Dict[str, str]
    cookies: Dict[str, str]
    last_SearchResponse: Union[SearchResponse, Dict[str, Any], List[Any], None] # type: ignore[assignment]
    search_query_details: Dict[str, Any]
    status_updates: List[StatusUpdateDict]
    final_search_results: Optional[List[Any]]
    sources_used: List[SourceDict]
    _seen_source_urls: set
    people_also_ask: List[PeopleAlsoAskDict]
    _seen_paa_questions: set
    agents_guide: Optional[List[Any]]
    result_summary: Dict[str, ResultSummaryDict]
    raw_events_log: List[dict]

    def __init__(
        self,
        timeout: int = 30,
        proxies: Optional[Dict[str, str]] = None,
        max_tokens: int = 600,
        log_raw_events: bool = False,
    ) -> None:
        """Initialize the Genspark API client.
        
        Args:
            timeout (int, optional): Request timeout in seconds. Defaults to 30.
            proxies (dict, optional): Proxy configuration for requests. Defaults to None.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 600.
            log_raw_events (bool, optional): Log all raw events to self.raw_events_log. Defaults to False.
        """
        self.session = cloudscraper.create_scraper()
        self.max_tokens = max_tokens
        self.chat_endpoint = "https://www.genspark.ai/api/search/stream"
        self.stream_chunk_size = 64
        self.timeout = timeout
        self.log_raw_events = log_raw_events
        self.headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9,en-IN;q=0.8",
            "Content-Type": "application/json",
            "DNT": "1",
            "Origin": "https://www.genspark.ai",
            "Priority": "u=1, i",
            "Sec-CH-UA": '"Chromium";v="128", "Not;A=Brand";v="24", "Microsoft Edge";v="128"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": LitAgent().random(),
        }
        self.cookies = {
            "i18n_redirected": "en-US",
            "agree_terms": "0", # Note: Ensure this cookie reflects actual consent if needed
            "session_id": uuid4().hex,
        }
        self.session.headers.update(self.headers)
        self.session.proxies = proxies or {}
        self.last_SearchResponse = None
        self._reset_search_data()

    def _reset_search_data(self) -> None:
        """Resets attributes that store data from a search stream."""
        self.search_query_details = {}
        self.status_updates = []
        self.final_search_results = None
        self.sources_used = []
        self._seen_source_urls = set()
        self.people_also_ask = []
        self._seen_paa_questions = set()
        self.agents_guide = None
        self.result_summary = {}
        self.raw_events_log = []

    def search(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
    ) -> Union[
        SearchResponse, #type: ignore
        Dict[str, Any],
        List[dict],
        Iterator[Union[dict, SearchResponse]], #type: ignore
    ]:
        """
        Strongly typed search method for Genspark API.
        Args:
            prompt: The search query or prompt.
            stream: If True, yields results as they arrive.
            raw: If True, yields/returns raw event dicts.
        Returns:
            - If stream=True, raw=True: Iterator[dict]
            - If stream=True, raw=False: Iterator[SearchResponse | dict]
            - If stream=False, raw=True: List[dict]
            - If stream=False, raw=False: SearchResponse
        """
        self._reset_search_data()
        url = f"{self.chat_endpoint}?query={requests.utils.quote(prompt)}"
        def _process_stream() -> Iterator[Union[dict, SearchResponse]]: #type: ignore
            try:
                with self.session.post(
                    url,
                    headers=self.headers,
                    cookies=self.cookies,
                    json={},
                    stream=True,
                    timeout=self.timeout,
                ) as resp:
                    if not resp.ok:
                        raise exceptions.APIConnectionError(
                            f"Failed to generate SearchResponse - ({resp.status_code}, {resp.reason}) - {resp.text}"
                        )
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue
                        try:
                            data = json.loads(line[6:])
                            if self.log_raw_events:
                                self.raw_events_log.append(data)
                            event_type = data.get("type")
                            field_name = data.get("field_name")
                            result_id = data.get("result_id")
                            if raw:
                                yield data
                            # Populate instance attributes
                            if event_type == "result_start":
                                self.result_summary[result_id] = cast(ResultSummaryDict, {
                                    "source": data.get("result_source"),
                                    "rel_score": data.get("result_rel_score"),
                                    "score": data.get("result_score"),
                                    "llm_id": data.get("llm_id"),
                                    "cogen_name": data.get("cogen", {}).get("name"),
                                })
                            elif event_type == "classify_query_result":
                                self.search_query_details["classification"] = data.get("classify_query_result")
                            elif event_type == "result_field":
                                field_value = data.get("field_value")
                                if field_name == "search_query":
                                    self.search_query_details["query_string"] = field_value
                                elif field_name == "thinking":
                                    self.status_updates.append({"type": "thinking", "message": field_value})
                                elif field_name == "search_status_top_bar_data":
                                    self.status_updates.append({"type": "status_top_bar", "data": field_value})
                                    if isinstance(field_value, dict) and field_value.get("status") == "finished":
                                        self.final_search_results = field_value.get("search_results")
                                        if field_value.get("search_plan"):
                                            self.search_query_details["search_plan"] = field_value.get("search_plan")
                                elif field_name == "search_source_top_bar_data":
                                    if isinstance(field_value, list):
                                        for source in field_value:
                                            if isinstance(source, dict) and source.get("url") and source.get("url") not in self._seen_source_urls:
                                                self.sources_used.append(cast(SourceDict, source))
                                                self._seen_source_urls.add(source.get("url"))
                            elif event_type == "result_end":
                                if result_id in self.result_summary:
                                    self.result_summary[result_id]["ended"] = True
                                search_result_data = data.get("search_result")
                                if search_result_data and isinstance(search_result_data, dict):
                                    if search_result_data.get("source") == "people_also_ask" and "people_also_ask" in search_result_data:
                                        paa_list = search_result_data["people_also_ask"]
                                        if isinstance(paa_list, list):
                                            for paa_item in paa_list:
                                                if isinstance(paa_item, dict) and paa_item.get("question") not in self._seen_paa_questions:
                                                    self.people_also_ask.append(cast(PeopleAlsoAskDict, paa_item))
                                                    self._seen_paa_questions.add(paa_item.get("question"))
                                    elif search_result_data.get("source") == "agents_guide" and "agents_guide" in search_result_data:
                                        self.agents_guide = search_result_data["agents_guide"]
                            if not raw:
                                processed_event_payload = None
                                if event_type == "result_field_delta" and field_name and field_name.startswith("streaming_detail_answer"):
                                    delta_text = data.get("delta", "")
                                    delta_text = re.sub(r"\[.*?\]\(.*?\)", "", delta_text)
                                    yield SearchResponse(delta_text)
                                elif event_type == "result_start":
                                    processed_event_payload = {"event": "result_start", "data": {"id": result_id, "source": data.get("result_source"), "score": data.get("result_score")}}
                                elif event_type == "classify_query_result":
                                    processed_event_payload = {"event": "query_classification", "data": data.get("classify_query_result")}
                                elif event_type == "result_field":
                                    field_value = data.get("field_value")
                                    if field_name == "search_query":
                                        processed_event_payload = {"event": "search_query_update", "value": field_value}
                                    elif field_name == "thinking":
                                         processed_event_payload = {"event": "thinking_update", "value": field_value}
                                    elif field_name == "search_status_top_bar_data":
                                        processed_event_payload = {"event": "status_update", "data": field_value}
                                    elif field_name == "search_source_top_bar_data":
                                         processed_event_payload = {"event": "sources_update", "data": field_value}
                                elif event_type == "result_end":
                                    processed_event_payload = {"event": "result_end", "data": {"id": result_id, "search_result": data.get("search_result")}}
                                if processed_event_payload:
                                    yield processed_event_payload
                        except json.JSONDecodeError:
                            continue
            except cloudscraper.exceptions.CloudflareException as e:
                raise exceptions.APIConnectionError(f"Request failed due to Cloudscraper issue: {e}")
            except requests.exceptions.RequestException as e:
                raise exceptions.APIConnectionError(f"Request failed: {e}")
        processed_stream_gen = _process_stream()
        if stream:
            return processed_stream_gen
        else:
            full_SearchResponse_text = ""
            all_raw_events_for_this_search: List[dict] = []
            for item in processed_stream_gen:
                if raw:
                    all_raw_events_for_this_search.append(cast(dict, item))
                else:
                    if isinstance(item, SearchResponse):
                        full_SearchResponse_text += str(item)
            if raw:
                self.last_SearchResponse = {"raw_events": all_raw_events_for_this_search}
                return all_raw_events_for_this_search
            else:
                final_text_SearchResponse = SearchResponse(full_SearchResponse_text)
                self.last_SearchResponse = final_text_SearchResponse
                return final_text_SearchResponse

if __name__ == "__main__":
    from rich import print
    ai = Genspark()
    try:
        search_result_stream = ai.search(input(">>> "), stream=True, raw=False)
        for chunk in search_result_stream:
            print(chunk, end="", flush=True)
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")