'''this is a mostly generic cache, with the following exceptions:

* it computes the value size by assuming it's either a surface or an array
* it adds the current movie resolution to the key, so the user needn't worry
  about cache invalidation upon rotating a movie

it assumes keys are built around "IDs with versions" where ID identifies a layer
within a frame, and a version is incremented when the layer is edited. this way
we easily invalidate everything derived from a layer when it's edited
'''
import collections
import res

MAX_CACHE_BYTE_SIZE = 1*1024**3
MAX_CACHED_ITEMS = 2000

class CachedItem:
    def compute_key(self):
        '''a key is a tuple of:
        1. a list of tuples mapping IDs to versions. a cached item referencing
        unknown IDs or IDs with old versions is eventually garbage-collected
        2. any additional info making the key unique.
        
        compute_key returns the key computed from the current system state.
        for example, CachedThumbnail(pos=5) might return a dictionary mapping
        the IDs of frames making up frame 5 in every layer, and the string
        "thumbnail." The number 5 is not a part of the key; if frame 6
        is made of the same frames in every layer, CachedThumbnail(pos=6)
        will compute the same key. If in the future a CachedThumbnail(pos=5)
        is created computing a different key because the movie was edited,
        you'll get a cache miss as expected.
        '''
        return (tuple(),)

    def compute_value(self):
        '''returns the value - used upon cache miss. note that CachedItems
        are not kept in the cache themselves - only the keys and the values.'''
        return None

# there are 2 reasons to evict a cached item:
# * no more room in the cache - evict the least recently used items until there's room
# * the cached item has no chance to be useful - eg it was computed from a deleted or
#   since-edited frame - this is done by collect_garbage() and assisted by update_id()
#   and delete_id()
class Cache:
    class Miss:
        pass
    MISS = Miss()
    def __init__(self):
        self.clear()
    def clear(self):
        self.key2value = collections.OrderedDict()
        self.id2version = {}
        self.debug = False
        self.gc_iter = 0
        self.last_check = {}
        # these are per-gc iteration counters
        self.computed_bytes = 0
        self.hit_bytes = 0
        # sum([self.size(value) for value in self.key2value.values()])
        self.cache_size = 0
        self.locked = False
    def _size(self,value):
        try:
            # surface
            return value.get_width() * value.get_height() * 4
        except:
            try:
                prod = 1
                for dim in value.shape:
                    prod *= dim
                return prod
            except:
                if value is None:
                    return 0
                print('WARNING: unknown value type - not caching', type(value))
                return MAX_CACHE_BYTE_SIZE
    def lock(self): self.locked = True
    def unlock(self): self.locked = False
    def fetch(self, cached_item): return self.fetch_kv(cached_item)[1]
    def fetch_kv(self, cached_item):
        key = (cached_item.compute_key(), (res.IWIDTH, res.IHEIGHT))
        value = self.key2value.get(key, Cache.MISS)
        if value is Cache.MISS:
            value = cached_item.compute_value()
            vsize = self._size(value)
            self.computed_bytes += vsize
            if self.locked or not self._has_room_for(vsize):
                if self.debug:
                    print('Missed and no room for', key[0])
                return key[0], value
            self.cache_size += vsize
            self.key2value[key] = value
            if self.debug:
                print('Missed and cached', key[0])
        else:
            self.key2value.move_to_end(key)
            self.hit_bytes += self._size(value)
        return key[0], value

    def _has_room_for(self, size):
        '''evicts the least recently used item until there's enough room'''
        if size >= MAX_CACHE_BYTE_SIZE:
           return False # don't evict stuff to fit something that can never fit anyway (notably unknown type)
        while self.cache_size + size > MAX_CACHE_BYTE_SIZE or len(self.key2value) > MAX_CACHED_ITEMS-1:
            if not self.key2value: # nothing left to evict...
                return False
            key, value = self.key2value.popitem(last=False)
            if self.debug:
                print('evicted', key)
            self.cache_size -= self._size(value)
        return True

    def update_id(self, id, version):
        self.id2version[id] = version
    def delete_id(self, id):
        if id in self.id2version:
            del self.id2version[id]
    def _stale(self, key):
        id2version, _ = key[0]
        for id, version in id2version:
            current_version = self.id2version.get(id)
            if current_version is None or version < current_version:
                if self.debug:
                    print('stale',id,version,current_version)
                return True
        return False
    def collect_garbage(self):
        orig = len(self.key2value)
        orig_size = self.cache_size
        for key, value in list(self.key2value.items()):
            if self._stale(key):
                del self.key2value[key]
                self.cache_size -= self._size(value)
        if self.debug:
            print('gc',orig,orig_size,'->',len(self.key2value),self.cache_size,'computed',self.computed_bytes,'cached',self.hit_bytes)
        self.gc_iter += 1
        self.computed_bytes = 0
        self.hit_bytes = 0

    def cached_bytes(self): return self.cache_size
    def cached_items(self): return len(self.key2value)

# tests
#######

# we mainly test that the cache actually caches, stays below the size limits using LRU eviction,
# and collects garbage using the ID to version mapping.

def test_default_cached_item():
    cache = Cache()
    item = CachedItem()
    k, v = cache.fetch_kv(item)
    assert v is None
    k, v = cache.fetch_kv(item)
    assert v is None
    assert cache.cached_items() == 1
    assert cache.cached_bytes() == 0

def test_unknown_value_type():
    cache = Cache()
    class Item(CachedItem):
        def compute_key(self): return (('unique',0),)
        def compute_value(self): return 'unknown type'
    cache.fetch(CachedItem())
    assert cache.cached_items() == 1
    for i in range(3):
        cache.fetch(Item())
        assert cache.cached_items() == 1 # make sure we haven't evicted our previous item

def test_cache_size_limits():
    limits = ((10300, 200, 10000, 25), (10300, 15, 6000, 15), (10, 10, 0, 0), (10000, 0, 0, 0))
    class Value: pass
    v = Value()
    v.shape = [40, 10]
    class V2:
        def get_width(self): return 10
        def get_height(self): return 10

    global MAX_CACHED_ITEMS
    global MAX_CACHE_BYTE_SIZE
    orig = MAX_CACHED_ITEMS, MAX_CACHE_BYTE_SIZE
    for value in v, V2():
        for maxbytes, maxitems, expmaxb, expmaxi in limits:
            MAX_CACHE_BYTE_SIZE = maxbytes
            MAX_CACHED_ITEMS = maxitems
            cache = Cache()
            maxb, maxi = 0, 0
            for i in range(1000):
                class CachedItem:
                    def compute_key(_): return ((i,0),)
                    def compute_value(_): return value
                cache.fetch(CachedItem())
                assert cache.cached_bytes() <= maxbytes
                assert cache.cached_items() <= maxitems
                maxb = max(cache.cached_bytes(), maxb)
                maxi = max(cache.cached_items(), maxi)
            assert expmaxb == maxb
            assert expmaxi == maxi
            # check LRU eviction
            assert [k[0][0][0] for k in cache.key2value.keys()] == list(range(1000-maxi,1000))

    MAX_CACHED_ITEMS, MAX_CACHE_BYTE_SIZE = orig
    
def test_garbage_collection():
    class Value: pass
    value = Value()
    value.shape = [40, 10]
    cache = Cache()
    for i in range(5):
        for vals in range(1,5):
            for u in range(2):
                class CachedItem:
                    def compute_key(_): return (tuple([(v,i) for v in range(vals)]),u)
                    def compute_value(_): return value
                k, v = cache.fetch_kv(CachedItem())
    cache.update_id(0,3)
    cache.update_id(1,3)
    cache.update_id(2,3)
    cache.collect_garbage() # this should delete everything with ID 3 (which unlike 0, 1 and 2 has no version
    # set for it by update_id()), as well as everything with version < 3 (which is now the version of IDs 0, 1 and 2)
    for ((id2version, _), _) in cache.key2value.keys():
        for id, version in id2version:
            assert id in [0,1,2]
            assert version >= 3
    assert cache.cached_items() == 12

    cache.update_id(2,4)
    cache.collect_garbage()
    for ((id2version, _), _) in cache.key2value.keys():
        for id, version in id2version:
            assert id != 2 or version == 4
    assert cache.cached_items() == 10

    cache.delete_id(2)
    cache.delete_id(1)
    cache.collect_garbage()
    for ((id2version, _), _) in cache.key2value.keys():
        for id, version in id2version:
            assert id == 0
    assert cache.cached_items() == 4

