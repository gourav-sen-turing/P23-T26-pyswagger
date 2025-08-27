from __future__ import absolute_import
from .consts import private 
from .errs import CycleDetectionError
import six
import imp
import sys
import datetime
import re
import os
import operator
import functools

#TODO: accept varg
def scope_compose(scope, name, sep=private.SCOPE_SEPARATOR):
    """ compose a new scope

    :param str scope: current scope
    :param str name: name of next level in scope
    :param str sep: scope separator
    :return the composed scope
    """

    if name == None:
        new_scope = scope
    else:
        new_scope = scope if scope else name

    if scope and name:
        new_scope = scope + sep + name

    return new_scope

def scope_split(scope, sep=private.SCOPE_SEPARATOR):
    """ split a scope into names

    :param str scope: scope to be splitted
    :param str sep: scope separator
    :return: list of str for scope names
    """

    return scope.split(sep) if scope else [None]


class ScopeDict(dict):
    """ ScopeDict
    """
    def __init__(self, *a, **k):
        self.__sep = k.pop('sep', private.SCOPE_SEPARATOR)
        super(ScopeDict, self).__init__(*a, **k)

    @property
    def sep(self):
        return self.__sep

    @sep.setter
    def sep(self, sep):
        self.__sep = sep

    def __getitem__(self, *keys):
        """
        """
        def __get_item(k):
            k = scope_compose(None, k, sep=self.sep) if isinstance(k, tuple) else k

            # find the key that match the full name
            if k in self:
                return dict.__getitem__(self, k)

            # find the key that match the part of name from the bottom.
            # for example,
            # full name: 'a!b!c'
            # partial name: 'b!c', 'c'
            keys = [key for key in six.iterkeys(self) if key.endswith(k)]
            if len(keys) == 1:
                return dict.__getitem__(self, keys[0])
            elif len(keys) > 1:
                raise ValueError('Multiple occurrence of key: ' + k)
            else:
                raise ValueError('Unable to find key: ' + k)

        if len(keys) == 1:
            keys = keys[0]

        if isinstance(keys, tuple):
            # keys is a tuple when used with nested dict,
            # ex. ScopeDict['a']['b']['c'].
            # keys[0] is useless for us here.
            return __get_item(keys[-1])
        else:
            return __get_item(keys)


class CaseInsensitiveDict(dict):
    """ CaseInsensitive dict """

    def __init__(self, *a, **k):
        super(CaseInsensitiveDict, self).__init__()
        tmp = dict(*a, **k) if a else k
        for key, value in six.iteritems(tmp):
            self[key.lower()] = value

    def __contains__(self, k):
        return super(CaseInsensitiveDict, self).__contains__(k.lower())

    def __delitem__(self, k):
        return super(CaseInsensitiveDict, self).__delitem__(k.lower())

    def __getitem__(self, k):
        return super(CaseInsensitiveDict, self).__getitem__(k.lower())

    def __setitem__(self, k, v):
        return super(CaseInsensitiveDict, self).__setitem__(k.lower(), v)

    def get(self, k, *default):
        return super(CaseInsensitiveDict, self).get(k.lower(), *default)


_iso8601_fmt = re.compile(''.join([
    r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})', # YYYY-MM-DD
    r'(T|t|\ )',  # T or space
    r'(?P<hour>\d{2}):(?P<minute>\d{2})(:(?P<second>\d{1,2}))?', # hh:mm:ss
    r'(?P<tz>Z|[+-]\d{2}:\d{2})?' # Z or +/-hh:mm
    ]))
_iso8601_fmt_date = re.compile(r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})') # YYYY-MM-DD

class FixedTZ(datetime.tzinfo):
    """ tzinfo implementation without consideration of DST """
    def __init__(self, h, m):
        self.__offset = datetime.timedelta(hours=h, minutes=m)

    def utcoffset(self, dt):
        return self.__offset

    def tzname(self, dt):
        return 'UTC' + str(self.__offset)

    def dst(self, dt):
        # a fixed-offset class:  doesn't account for DST
        return datetime.timedelta(0)

    def __repr__(self):
        return 'FixedTZ(hours={0}, minutes={1})'.format(self.__offset.seconds // 3600, self.__offset.seconds % 60)

def from_iso8601(s):
    """ convert iso8601 string to datetime object

    :param s: iso8601 string
    :type s: str
    :return: datetime object
    :rtype: datetime.datetime
    """
    m = _iso8601_fmt.match(s)
    if not m:
        m = _iso8601_fmt_date.match(s)
        if not m:
            raise ValueError('Unable to convert [{0}] to datetime object'.format(s))
        return datetime.datetime(year=int(m.group('year')), month=int(m.group('month')), day=int(m.group('day')))

    tz = None
    if m.group('tz') and m.group('tz') != 'Z':
        neg = m.group('tz').startswith('-')
        h, m = m.group('tz')[1:].split(':')

        tz = FixedTZ((-1 if neg else 1) * int(h), int(m))
    elif m.group('tz') == 'Z':
        tz = FixedTZ(0, 0)

    return datetime.datetime(
        year=int(m.group('year')),
        month=int(m.group('month')),
        day=int(m.group('day')),
        hour=int(m.group('hour')),
        minute=int(m.group('minute')),
        second=int(m.group('second')) if m.group('second') else 0,
        tzinfo=tz
    )

def get_dict_as_tuple(d):
    """ get a dict of one element as tuple

    :param d: a dict object
    :type d: dict
    :return: (key, value) of the only element, or (None, None) when list is not composed by only one element.
    :rtype: tuple of two elements
    """
    if isinstance(d, dict) and len(d) == 1:
        for k, v in six.iteritems(d):
            return k, v

    return None, None

def nv_tuple_list_replace(x, v):
    """ replace or append a tuple in a list of tuples

    :param x: list of tuples, should be a list of (name, value) pair,
                the first element of each tuple is the key for searching.
    :type x: list of tuple
    :param v: the tuple to be replaced or appended.
    :type v: tuple
    :raises ValueError: if x is not a list, or v is not a tuple
    """
    if not isinstance(x, list):
        raise ValueError('x should be a list but got {0}'.format(str(type(x))))
    if not isinstance(v, tuple):
        raise ValueError('v should be a tuple but got {0}'.format(str(type(v))))

    for i, t in enumerate(x):
        if t[0] == v[0]:
            x[i] = v
            return

    x.append(v)

def import_string(name):
    """ import module

    :param name: module name
    :type name: str
    :return: module, or None if not found
    """
    try:
        # existing in sys.modules
        return sys.modules[name]
    except KeyError:
        pass

    fp = None
    try:
        fp, pathname, desc = imp.find_module(name)
        mod = imp.load_module(name, fp, pathname, desc)
    except ImportError:
        mod = None
    finally:
        # Since we may exit via an exception, close fp explicitly.
        if fp:
            fp.close()

    return mod

def jp_compose(s, base=None):
    """ append/encode a string to json-pointer
    """
    if s == None:
        return base

    ss = [s] if isinstance(s, six.string_types) else s
    ss = [s.replace('~', '~0').replace('/', '~1') for s in ss]
    if base:
        ss.insert(0, base)
    return '/'.join(ss)

def jp_split(s):
    """ split/decode a string from json-pointer
    """
    if s == '' or s == None:
        return []

    def _decode(s):
        s = s.replace('~1', '/')
        return s.replace('~0', '~')

    return [_decode(ss) for ss in s.split('/')]

def jr_split(s):
    """ split a json-reference into (url, json-pointer)
    """
    p = six.moves.urllib.parse.urlparse(s)
    return (
        normalize_url(six.moves.urllib.parse.urlunparse(p[:5]+('',))),
        '#'+p.fragment if p.fragment else '#'
    )

def deref(obj):
    """ dereference $ref

    :param obj: object to be dereferenced
    :return: (is_success, deref_obj, ref_obj),
            is_success: True if dereference successfully,
            deref_obj is the object to be replaces,
            ref_obj is the object to be referred.
    """
    is_success = True
    deref_obj = None
    ref_obj = obj

    if obj == None:
        is_success = False
    else:
        try:
            while getattr(ref_obj, '$ref'):
                ref_obj = getattr(ref_obj, '$ref')
            deref_obj = obj
        except AttributeError:
            # the object without $ref means itself.
            pass

    return is_success, deref_obj, ref_obj


class CycleGuard(object):
    """ Guard against cycle visiting, this object
    would raise an exception when visiting an object
    twice.
    """
    def __init__(self, identity_hook=None):
        """ constructor

        :param identity_hook: hook to get identity from object
        :type identity_hook: func
        """
        self.visited = []
        self.__identity = identity_hook or (lambda x: x)

    def update(self, obj):
        """ add an object to visited list

        :param obj: object to be visited
        :raises CycleDetectionError: if obj already in visited list, would raise this exception.
        """
        i = self.__identity(obj)
        if i and i in self.visited:
            raise CycleDetectionError('Cycle detected: {0}'.format(repr(i[:70])))
        self.visited.append(i)


def path2url(p):
    """ Return file:// URL from a filename.
    """
    return 'file://' + p


def normalize_url(url):
    """ Normalize url

    :param str url: url to be normalized
    :return: normalized url
    :rtype: str
    """
    if url == None or url == '':
        return url

    p = six.moves.urllib.parse.urlparse(url)
    if p.scheme == '':
        if os.path.exists(url):
            url = six.moves.urllib.parse.quote(url.encode('utf-8'))
            url = path2url(os.path.abspath(url))
        else:
            # it should be a relative file
            url = six.moves.urllib.parse.quote(url.encode('utf-8'))
            url = path2url(os.path.join(os.getcwd(), url))

    # remove trailing '/'
    p = six.moves.urllib.parse.urlparse(url)
    if p.path != '/' and p.path.endswith('/'):
        url = six.moves.urllib.parse.urlunparse(p[:2]+(p.path[:-1],)+p[3:])

    return url


def normalize_jr(jr, header, url=None):
    """ JSON reference normalization

    :param str jr: JSON reference
    :param str header: the header for JSON reference.
    :param str url: the url of the Swagger App.
    :return: normalized JSON reference, ex. http://my.domain.com/app#/definitions/User
    :rtype: str
    """

    if jr == None:
        return jr

    if jr.startswith('#'):
        if url:
            return ''.join([url, jr])
        else:
            return jr
    elif jr.startswith('http'):
        return jr
    else:
        if url:
            return ''.join([url, header, '/', jr])
        else:
            return ''.join([header, '/', jr]) if header else jr


def get_swagger_version(obj):
    """ Get swagger version from loaded json

    :param dict obj: loaded json
    :return: swagger version, ex. 1.2, 2.0
    :rtype: str
    """
    if 'swaggerVersion' in obj:
        return obj['swaggerVersion']
    elif 'swagger' in obj:
        return obj['swagger']
    else:
        return None


def walk(start, g, ret=None):
    """ DFS """
    ret = [] if ret == None else ret
    if not start:
        return ret

    # init stack and result
    stack = [start]
    visited = []
    sequences = []

    while stack:
        # take the last one ...
        current = stack[-1]

        # keep walking ...
        if not current in visited:
            # mark as visited
            visited.append(current)
            sequences.append(current)

            # get next obj to visit
            next_objs = g(current)
            for obj in next_objs if next_objs else []:
                if not obj in visited:
                    stack.append(obj)
        else:
            # there is no way to walk, pop it
            stack.pop()
            if sequences and sequences[-1] == current:
                sequences.pop()

            # a cycle is detected
            if current in sequences:
                idx = sequences.index(current)
                cyc = sequences[idx:] + [current]

                # check if this cycle already detected
                # note that we only need to check cycles with
                # same length
                duplicated = False
                for r in ret:
                    if len(r) != len(cyc):
                        continue

                    # we need to avoid adding duplicated cycles,
                    # ex. [1, 2, 3, 1] and [2, 3, 1, 2] are
                    # duplicated.
                    idx = cyc.index(min(cyc))
                    cyc_min = cyc[idx:-1] + cyc[:idx] + [min(cyc)]

                    idx = r.index(min(r))
                    r_min = r[idx:-1] + r[:idx] + [min(r)]

                    duplicated = cyc_min == r_min
                    if duplicated:
                        break

                if not duplicated:
                    ret.append(cyc)

    return ret


def _diff_(src, dst, ret=None, jp=None, exclude=[], include=[]):
    """ compare 2 dict/list, return a list containing
    json-pointer indicating what's different, and what's diff exactly.

    - list length diff: (jp, length of src, length of dst)
    - dict key diff: (jp, None, None)
    - when src is dict or list, and dst is not: (jp, type(src), type(dst))
    - other: (jp, src, dst)
    """
    jp = jp or ''
    if ret is None:
        ret = []

    # Apply filters only at the top level (when jp has no '/')
    if jp and '/' not in jp and (include or exclude):
        # Check exclude list
        if exclude and jp in exclude:
            return ret

        # Check include list (only if include list is not empty)
        if include and jp not in include:
            return ret

    if isinstance(src, dict) and isinstance(dst, dict):
        # Compare dict keys
        src_keys = set(src.keys())
        dst_keys = set(dst.keys())

        # Missing keys in dst
        for k in src_keys - dst_keys:
            if (not exclude or k not in exclude) and (not include or k in include):
                ret.append((jp + '/' + k if jp else k, None, None))

        # Extra keys in dst
        for k in dst_keys - src_keys:
            if (not exclude or k not in exclude) and (not include or k in include):
                ret.append((jp + '/' + k if jp else k, None, None))

        # Compare common keys
        for k in src_keys & dst_keys:
            # At top level, check include/exclude
            if not jp:  # Top level
                if exclude and k in exclude:
                    continue
                if include and k not in include:
                    continue
            # For nested levels, we've already filtered at top, so process everything
            _diff_(src[k], dst[k], ret, jp + '/' + k if jp else k, [], [])

    elif isinstance(src, list) and isinstance(dst, list):
        # Compare list lengths
        if len(src) != len(dst):
            ret.append((jp, len(src), len(dst)))

        # Compare elements only if same length
        elif len(src) == len(dst):
            for i in range(len(src)):
                _diff_(src[i], dst[i], ret, jp + '/' + str(i) if jp else str(i), exclude, include)

    elif type(src) != type(dst):
        # Type mismatch
        ret.append((jp, type(src).__name__, type(dst).__name__))
    elif src != dst:
        # Value difference
        ret.append((jp, src, dst))

    return ret


def get_or_none(obj, *args):
    """ Navigate nested attributes safely, returning None if any attribute doesn't exist

    :param obj: the root object
    :param args: sequence of attribute names to navigate
    :return: the final attribute value or None if navigation fails
    """
    if obj == None:
        return None

    current = obj
    for attr in args:
        try:
            current = getattr(current, attr)
        except AttributeError:
            return None

    return current
