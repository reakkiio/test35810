from webscout.Provider.OPENAI.base import OpenAICompatibleProvider
import requests

class DummyProvider(OpenAICompatibleProvider):
    def __init__(self, **kwargs):
        self.session = requests.Session()
        super().__init__(**kwargs)
    @property
    def models(self):
        class _ModelList:
            def list(self):
                return []
        return _ModelList()

try:
    dummy = DummyProvider()
    print("Proxies on instance:", dummy.proxies)
    print("Proxies on session:", dummy.session.proxies)
    # Should print proxies set by get_auto_proxy()
except NotImplementedError:
    # __init__ in OpenAICompatibleProvider raises NotImplementedError by default
    print("Metaclass ran, but base class is abstract.")