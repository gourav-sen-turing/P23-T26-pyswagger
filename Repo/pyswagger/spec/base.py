from __future__ import absolute_import
from ..utils import jp_compose
import six
import copy
import functools
import weakref
import itertools


class ContainerType:
    """ Enum of Container-Types
    """

    # list container
    list_ = 1

    # dict container
    dict_ = 2

    # dict of list container, like: {'xx': [], 'xx': [], ...}
    dict_of_list_ = 3

def container_apply(ct, v, f, fd=None, fdl=None):
    ret = None
    if v == None:
        return {}

    if ct == None:
        ret = [f(ct, v)]
    elif ct == ContainerType.list_:
        ret = {}
        for i, vv in enumerate(v):
            ret[str(i)] = f(ct, vv)
    elif ct == ContainerType.dict_:
        ret = []
        for k, vv in six.iteritems(v):
            ret.append(fd(ct, vv, k) if fd else f(ct, vv))
    elif ct == ContainerType.dict_of_list_:
        ret = []
        for k, vv in six.iteritems(v):
            if fdl:
                fdl(ct, vv, k)
            for vvv in vv:
                ret.append(fd(ct, vvv, k) if fd else f(ct, vvv))
    else:
        ret = []

    return ret


class Context(object):
    """ Base of all parsing contexts """

    # required fields, a list of strings
    __swagger_required__ = []

    # parsing context of children fields,
    # a list of tuple (field-name, container-type, parsing-context)
    __swagger_child__ = {}

    # factory of object to be created according to
    # this parsing context.
    __swagger_ref_obj__ = None

    def __init__(self, parent_obj, backref):
        """
        constructor

        :param dict parent_obj: parent object placeholder
        :param str backref: the key to parent object placeholder
        """

        # object placeholder of parent object
        self._parent_obj = parent_obj

        # key used to indicate the location in
        # parent object when parsing finished.
        self._backref = backref

        self.__reset_obj()

    def __enter__(self):
        return self

    def __reset_obj(self):
        self._obj = {}

    @classmethod
    def is_produced(kls, obj):
        if kls.__swagger_ref_object__ is None:
            return True
        return isinstance(obj, kls.__swagger_ref_object__)

    def produce(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        """ When exiting parsing context, doing two things
        - create the object corresponding to this parsing context.
        - populate the created object to parent context.
        """
        if self._obj == None:
            return

        # check if we should produce a different object
        obj = self.produce()
        if obj == None and self.__swagger_ref_object__:
            # create the actual object
            obj = self.__swagger_ref_object__(self)
            # update object with parsed data
            for key, value in six.iteritems(self._obj):
                if key in obj.__swagger_fields__:
                    obj.update_field(key, value)
                elif hasattr(obj, key):
                    setattr(obj, key, value)

            # Assign parent references after all fields are set
            obj._assign_parent(self)

        # check if object is valid only if we didn't produce a different object
        if obj and self.produce() is None and not self.is_produced(obj):
            if self.__swagger_ref_object__:
                raise ValueError('Object is not instance of {0} but {1}'.format(
                    self.__swagger_ref_object__.__name__, obj.__class__.__name__))

        self.__reset_obj()

        if isinstance(self._parent_obj[self._backref], list):
            self._parent_obj[self._backref].append(obj)
        else:
            self._parent_obj[self._backref] = obj

    def parse(self, obj=None):
        """ major part do parsing.

        :param dict obj: json object to be parsed.
        :raises ValueError: if obj is not a dict type.
        """
        if obj == None:
            return

        if not isinstance(obj, dict):
            raise ValueError('invalid obj passed: ' + str(type(obj)))

        def _apply(x, kk, ct, v):
            if key not in self._obj:
                self._obj[kk] = {} if ct == None else []
            with x(self._obj, kk) as ctx:
                ctx.parse(obj=v)

        def _apply_dict(x, kk, ct, v, k):
            if k not in self._obj[kk]:
                self._obj[kk][k] = {} if ct == ContainerType.dict_ else []
            with x(self._obj[kk], k) as ctx:
                ctx.parse(obj=v)

        def _apply_dict_before_list(kk, ct, v, k):
            self._obj[kk][k] = []

        if hasattr(self, '__swagger_child__'):
            # to nested objects
            for key, (ct, ctx_kls) in six.iteritems(self.__swagger_child__):
                items = obj.get(key, None)

                # create an empty child, even it's None in input.
                # this makes other logic easier.
                if ct == ContainerType.list_:
                    self._obj[key] = []
                elif ct:
                    self._obj[key] = {}

                if items == None:
                    continue

                container_apply(ct, items,
                    functools.partial(_apply, ctx_kls, key),
                    functools.partial(_apply_dict, ctx_kls, key),
                    functools.partial(_apply_dict_before_list, key)
                )

        # update _obj with obj
        if self._obj != None:
            for key in (set(obj.keys()) - set(self._obj.keys())):
                self._obj[key] = obj[key]
        else:
            self._obj = obj


class BaseObj(object):
    """ Base implementation of all referencial objects,
    """

    # fields that need re-named.
    __swagger_rename__ = {}

    # dict of names of fields, we will skip fields not in this list.
    # field format:
    # - {name: default-value}: a field name with default value
    __swagger_fields__ = {}

    # fields used internally
    __internal_fields__ = {}

    # Swagger Version this object belonging to
    __swagger_version__ = None

    def __init__(self, ctx):
        """ constructor

        :param Context ctx: parsing context used to create this object
        :raises TypeError: if ctx is not a subclass of Context.
        """
        super(BaseObj, self).__init__()

        # init parent reference
        self._parent__ = None
        # init children reference dictionary
        self._children__ = {}

        if not issubclass(type(ctx), Context):
            raise TypeError('should provide args[0] as Context, not: ' + ctx.__class__.__name__)

        self.__origin_keys = set([k for k in six.iterkeys(ctx._obj)]) if ctx._obj else set()

        # handle fields with proper default value handling
        for name, default in six.iteritems(self.__swagger_fields__):
            # create copy of mutable defaults
            if isinstance(default, list):
                default_val = list(default) if default else []
            elif isinstance(default, dict):
                default_val = dict(default) if default else {}
            else:
                default_val = default
            setattr(self, self.get_private_name(name), default_val)

        for name, default in six.iteritems(self.__internal_fields__):
            # create copy of mutable defaults
            if isinstance(default, list):
                default_val = list(default) if default else []
            elif isinstance(default, dict):
                default_val = dict(default) if default else {}
            else:
                default_val = default
            setattr(self, self.get_private_name(name), default_val)

        # Parent assignment moved to after field updates in Context.__exit__

    def _assign_parent(self, ctx):
        """ parent assignment, internal usage only
        """
        def _assign(cls, path_key, obj):
            if obj == None:
                return

            if cls.is_produced(obj):
                if isinstance(obj, BaseObj):
                    obj._parent__ = self
                    # Track children with proper path key
                    self._children__[path_key] = obj
            else:
                raise ValueError('Object is not instance of {0} but {1}'.format(cls.__swagger_ref_object__.__name__, obj.__class__.__name__))

        # set self as children's parent
        for name, (ct, ctx_cls) in six.iteritems(ctx.__swagger_child__):
            obj = getattr(self, name, None)
            if obj == None:
                continue

            if ct == None:
                # single object
                _assign(ctx_cls, name, obj)
            elif ct == ContainerType.list_:
                # list of objects
                for i, item in enumerate(obj):
                    _assign(ctx_cls, '{0}/{1}'.format(name, i), item)
            elif ct == ContainerType.dict_:
                # dict of objects
                for k, item in six.iteritems(obj):
                    # json-pointer encode the key
                    encoded_key = k.replace('~', '~0').replace('/', '~1')
                    _assign(ctx_cls, '{0}/{1}'.format(name, encoded_key), item)
            elif ct == ContainerType.dict_of_list_:
                # dict of lists
                for k, lst in six.iteritems(obj):
                    for i, item in enumerate(lst):
                        encoded_key = k.replace('~', '~0').replace('/', '~1')
                        _assign(ctx_cls, '{0}/{1}/{2}'.format(name, encoded_key, i), item)

    def get_private_name(self, f):
        """ get private protected name of an attribute

        :param str f: name of the private attribute to be accessed.
        """
        f = self.__swagger_rename__[f] if f in self.__swagger_rename__.keys() else f
        return '_' + self.__class__.__name__ + '__' + f

    def update_field(self, f, obj):
        """ update a field

        :param str f: name of field to be updated.
        :param obj: value of field to be updated.
        """
        n = self.get_private_name(f)
        if not hasattr(self, n):
            raise AttributeError('{0} is not in {1}'.format(n, self.__class__.__name__))

        setattr(self, n, obj)
        self.__origin_keys.add(f)

    def resolve(self, ts):
        """ resolve a list of tokens to an child object

        :param list ts: list of tokens or a string path
        """
        if isinstance(ts, six.string_types):
            ts = [ts]

        if not ts:
            return self

        current = self
        for token in ts:
            if hasattr(current, token):
                current = getattr(current, token)
            elif isinstance(current, dict) and token in current:
                current = current[token]
            elif isinstance(current, list):
                try:
                    idx = int(token)
                    current = current[idx]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return current

    def merge(self, other, ctx):
        """ merge properties from other object,
        only merge from 'not None' to 'None'.

        :param BaseObj other: the source object to be merged from.
        :param Context ctx: the parsing context
        """
        if not isinstance(other, self.__class__):
            return self

        # merge simple fields
        for name, default in six.iteritems(self.__swagger_fields__):
            other_val = getattr(other, name, None)
            self_val = getattr(self, name, None)

            if other_val is not None and self_val is None:
                # Deep copy the value to avoid reference issues
                if isinstance(other_val, BaseObj):
                    # For BaseObj, we need to create new context and parse
                    temp_obj = {'_temp': {}}
                    child_ctx = ctx.__swagger_child__.get(name, (None, ctx.__class__))[1]
                    with child_ctx(temp_obj, '_temp') as c:
                        c.parse(other_val.dump() if hasattr(other_val, 'dump') and callable(other_val.dump) else {})
                    setattr(self, name, temp_obj['_temp'])
                elif isinstance(other_val, (list, dict)):
                    setattr(self, name, copy.deepcopy(other_val))
                else:
                    setattr(self, name, other_val)

        # merge children objects
        for name, (ct, child_ctx) in six.iteritems(ctx.__swagger_child__):
            other_val = getattr(other, name, None)
            self_val = getattr(self, name, None)

            if other_val is not None:
                if ct == ContainerType.list_:
                    # For lists, extend if self is empty
                    if self_val is None or len(self_val) == 0:
                        new_list = []
                        for item in other_val:
                            if isinstance(item, BaseObj):
                                temp_obj = {'_temp': {}}
                                with child_ctx(temp_obj, '_temp') as c:
                                    c.parse(item.dump() if hasattr(item, 'dump') and callable(item.dump) else {})
                                new_list.append(temp_obj['_temp'])
                            else:
                                new_list.append(copy.deepcopy(item))
                        setattr(self, name, new_list)

                elif ct == ContainerType.dict_ or ct == ContainerType.dict_of_list_:
                    # For dicts, merge keys
                    if self_val is None or (isinstance(self_val, dict) and len(self_val) == 0):
                        new_dict = {}
                        for k, v in six.iteritems(other_val):
                            if ct == ContainerType.dict_:
                                if isinstance(v, BaseObj):
                                    temp_obj = {'_temp': {}}
                                    with child_ctx(temp_obj, '_temp') as c:
                                        c.parse(v.dump() if hasattr(v, 'dump') and callable(v.dump) else {})
                                    new_dict[k] = temp_obj['_temp']
                                else:
                                    new_dict[k] = copy.deepcopy(v)
                            else:  # dict_of_list_
                                new_list = []
                                for item in v:
                                    if isinstance(item, BaseObj):
                                        temp_obj = {'_temp': {}}
                                        with child_ctx(temp_obj, '_temp') as c:
                                            c.parse(item.dump() if hasattr(item, 'dump') and callable(item.dump) else {})
                                        new_list.append(temp_obj['_temp'])
                                    else:
                                        new_list.append(copy.deepcopy(item))
                                new_dict[k] = new_list
                        setattr(self, name, new_dict)

                elif ct is None and self_val is None:
                    # Single object
                    if isinstance(other_val, BaseObj):
                        temp_obj = {'_temp': {}}
                        with child_ctx(temp_obj, '_temp') as c:
                            c.parse(other_val.dump() if hasattr(other_val, 'dump') and callable(other_val.dump) else {})
                        setattr(self, name, temp_obj['_temp'])
                    else:
                        setattr(self, name, copy.deepcopy(other_val))

        # Re-assign parent references
        self._assign_parent(ctx)

        return self

    def is_set(self, k):
        """ check if a key is setted from Swagger API document

        :param k: the key to check
        :return: True if the key is setted. False otherwise, it means we would get value
        from default from Field.
        """
        return k in self.__origin_keys

    def compare(self, other, base=None):
        """ comparison, will return the first difference """
        if not isinstance(other, self.__class__):
            return (False, base or '')

        path_base = base + '/' if base else ''

        # Compare simple fields
        for name in six.iterkeys(self.__swagger_fields__):
            self_val = getattr(self, name, None)
            other_val = getattr(other, name, None)

            # Handle None comparisons
            if self_val is None and other_val is None:
                continue
            if (self_val is None) != (other_val is None):
                return (False, path_base + name)

            # Compare BaseObj
            if isinstance(self_val, BaseObj):
                if not isinstance(other_val, BaseObj):
                    return (False, path_base + name)
                result, path = self_val.compare(other_val, path_base + name)
                if not result:
                    return (False, path)

            # Compare lists
            elif isinstance(self_val, list):
                if not isinstance(other_val, list) or len(self_val) != len(other_val):
                    return (False, path_base + name)
                for i, (sv, ov) in enumerate(zip(self_val, other_val)):
                    if isinstance(sv, BaseObj) and isinstance(ov, BaseObj):
                        result, path = sv.compare(ov, path_base + '{0}/{1}'.format(name, i))
                        if not result:
                            return (False, path)
                    elif sv != ov:
                        return (False, path_base + '{0}/{1}'.format(name, i))

            # Compare dicts
            elif isinstance(self_val, dict):
                if not isinstance(other_val, dict):
                    return (False, path_base + name)
                # Check keys
                if set(self_val.keys()) != set(other_val.keys()):
                    # Find the first different key
                    for k in set(self_val.keys()) | set(other_val.keys()):
                        if k not in self_val or k not in other_val:
                            return (False, path_base + '{0}/{1}'.format(name, k))

                # Compare values
                for k in self_val:
                    sv = self_val[k]
                    ov = other_val[k]
                    if isinstance(sv, BaseObj) and isinstance(ov, BaseObj):
                        result, path = sv.compare(ov, path_base + '{0}/{1}'.format(name, k))
                        if not result:
                            return (False, path)
                    elif isinstance(sv, list) and isinstance(ov, list):
                        if len(sv) != len(ov):
                            return (False, path_base + '{0}/{1}'.format(name, k))
                        for i, (svv, ovv) in enumerate(zip(sv, ov)):
                            if isinstance(svv, BaseObj) and isinstance(ovv, BaseObj):
                                result, path = svv.compare(ovv, path_base + '{0}/{1}/{2}'.format(name, k, i))
                                if not result:
                                    return (False, path)
                            elif svv != ovv:
                                return (False, path_base + '{0}/{1}/{2}'.format(name, k, i))
                    elif sv != ov:
                        return (False, path_base + '{0}/{1}'.format(name, k))

            # Simple value comparison
            elif self_val != other_val:
                return (False, path_base + name)

        return (True, '')

    def dump(self):
        """ dump to dict representation """
        r = {}
        def _dump_(obj):
            if isinstance(obj, dict):
                ret = {}
                for k, v in six.iteritems(obj):
                    ret[k] = _dump_(v)
                return None if ret == {} else ret
            elif isinstance(obj, list):
                ret = []
                for v in obj:
                    ret.append(_dump_(v))
                return None if ret == [] else ret
            elif isinstance(obj, BaseObj):
                return obj.dump()
            elif isinstance(obj, (six.string_types, six.integer_types, float, bool)):
                return obj
            elif obj is None:
                return None
            else:
                # For other types, just return the object
                return obj

        for name, default in six.iteritems(self.__swagger_fields__):
            # only dump a field when its value is not equal to default value
            v = getattr(self, name)
            if v != default:
                d = _dump_(v)
                if d != None:
                    r[name] = d

        # For BaseObj, return empty dict instead of None to preserve object existence
        return r

    @property
    def _parent_(self):
        """ get parent object

        :return: the parent object.
        :rtype: a subclass of BaseObj.
        """
        return self._parent__

    @property
    def _field_names_(self):
        """ get list of field names defined in Swagger spec

        :return: a list of field names
        :rtype: a list of str
        """
        ret = []
        for n in six.iterkeys(self.__swagger_fields__):
            renamed = self.__swagger_rename__.get(n, n)
            ret.append(renamed)

        return ret

    @property
    def _children_(self):
        """ get children objects

        :rtype: a dict of children {child_name: child_object}
        """
        return self._children__


def _method_(name):
    """ getter/setter factory """
    def _getter_(self):
        return getattr(self, self.get_private_name(name))

    def _setter_(self, value):
        setattr(self, self.get_private_name(name), value)
        # Find the correct private attribute name
        origin_keys = None
        for attr in dir(self):
            if attr.endswith('__origin_keys'):
                origin_keys = getattr(self, attr)
                break
        if origin_keys is not None and isinstance(origin_keys, set):
            origin_keys.add(name)

    return property(_getter_, _setter_)


class FieldMeta(type):
    """ metaclass to init fields
    """
    def __new__(metacls, name, bases, spc):
        """ scan through MRO to get a merged list of fields,
        and create those fields.
        """
        def init_fields(fields, rename):
            for name in six.iterkeys(fields):
                renamed = rename[name] if name in rename.keys() else name
                spc[renamed] = _method_(name)

        def _default_(name, default):
            spc[name] = spc[name] if name in spc else default

        def _update_(dict1, dict2):
            d = {}
            for k in set(dict2.keys()) - set(dict1.keys()):
                d[k] = dict2[k]
            dict1.update(d)

        # compose fields definition from parents
        swagger_fields = {}
        internal_fields = {}
        swagger_rename = {}

        # collect fields from all base classes in MRO order
        for base in reversed([b for b in bases if issubclass(b, BaseObj)]):
            if hasattr(base, '__swagger_fields__'):
                swagger_fields.update(base.__swagger_fields__)
            if hasattr(base, '__internal_fields__'):
                internal_fields.update(base.__internal_fields__)
            if hasattr(base, '__swagger_rename__'):
                swagger_rename.update(base.__swagger_rename__)

        # update with current class fields
        if '__swagger_fields__' in spc:
            swagger_fields.update(spc['__swagger_fields__'])
        if '__internal_fields__' in spc:
            internal_fields.update(spc['__internal_fields__'])
        if '__swagger_rename__' in spc:
            swagger_rename.update(spc['__swagger_rename__'])

        # store merged fields
        spc['__swagger_fields__'] = swagger_fields
        spc['__internal_fields__'] = internal_fields
        spc['__swagger_rename__'] = swagger_rename

        # swagger fields
        if swagger_fields:
            init_fields(swagger_fields, swagger_rename)
        # internal fields
        if internal_fields:
            init_fields(internal_fields, {})

        return type.__new__(metacls, name, bases, spc)


class NullContext(Context):
    """ black magic to initialize BaseObj
    """

    _obj = None

    def __init__(self):
        super(NullContext, self).__init__(None, None)
