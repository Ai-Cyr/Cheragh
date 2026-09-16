"""Redis key and deletion scopes must not depend on ambiguous delimiters."""
from fnmatch import fnmatchcase

from cheragh.cache.base import CacheEntry, dumps_entry
from cheragh.cache.redis import RedisCache


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.before_eval = None

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value

    def setex(self, key, ttl, value):
        self.set(key, value)

    def delete(self, *keys):
        return sum(self.data.pop(key, None) is not None for key in keys)

    def scan_iter(self, match, count=None):
        yield from [key for key in self.data if fnmatchcase(key, match)]

    def eval(self, script, numkeys, key, expected):
        if self.before_eval is not None:
            self.before_eval()
        return self.delete(key) if self.data.get(key) == expected else 0


def test_colon_namespace_and_key_pairs_are_independent():
    cache = RedisCache(client=FakeRedis())
    cache.set("c", "nested namespace", namespace="a:b")
    cache.set("b:c", "nested key", namespace="a")
    assert cache.get("c", namespace="a:b") == "nested namespace"
    assert cache.get("b:c", namespace="a") == "nested key"
    assert cache.entry_count() == 2
    cache.delete("b:c", namespace="a")
    assert cache.get("c", namespace="a:b") == "nested namespace"


def test_invalidation_cannot_cross_namespace_or_prefix_boundaries():
    redis = FakeRedis()
    first = RedisCache(client=redis, key_prefix="application")
    neighbor = RedisCache(client=redis, key_prefix="application:tenant")
    first.set("one", 1, namespace="a")
    first.set("two", 2, namespace="a:b")
    neighbor.set("three", 3, namespace="a")
    assert first.invalidate_namespace("a") == 1
    assert first.get("two", namespace="a:b") == 2
    assert neighbor.get("three", namespace="a") == 3
    assert first.clear() == 1
    assert neighbor.entry_count() == 1
    assert neighbor.get("three", namespace="a") == 3


def test_unicode_empty_and_glob_namespaces_remain_distinct():
    cache = RedisCache(client=FakeRedis(), key_prefix="préfixe:*?[x]\\")
    namespaces = ["", "*", "?", "[x]", "tenant:é", "tenant", "\\"]
    for namespace in namespaces:
        cache.set("clé:*:[]", namespace, namespace=namespace)
    for namespace in namespaces:
        assert cache.get("clé:*:[]", namespace=namespace) == namespace
    assert cache.invalidate_namespace("*") == 1
    assert cache.entry_count() == len(namespaces) - 1
    assert cache.get("clé:*:[]", namespace="") == ""


def test_legacy_keys_are_not_read_migrated_or_deleted():
    redis = FakeRedis()
    legacy_key = "cheragh:a:b:c"
    legacy_payload = dumps_entry(CacheEntry(namespace="a:b", key="c", value="old"))
    redis.set(legacy_key, legacy_payload)
    cache = RedisCache(client=redis)
    assert cache.get("c", namespace="a:b") is None
    cache.set("c", "new", namespace="a:b")
    assert cache.get("c", namespace="a:b") == "new"
    assert cache.entry_count() == 1
    assert cache.invalidate_namespace("a") == 0
    assert cache.clear() == 1
    assert redis.data == {legacy_key: legacy_payload}


def test_poisoned_entry_cleanup_cannot_remove_concurrent_replacement():
    redis = FakeRedis()
    cache = RedisCache(client=redis)
    key = cache._redis_key("default", "key")
    redis.set(key, b"invalid JSON")
    fresh_payload = dumps_entry(CacheEntry(namespace="default", key="key", value="fresh"))
    redis.before_eval = lambda: redis.set(key, fresh_payload)
    assert cache.get("key") is None
    assert cache.get("key") == "fresh"
