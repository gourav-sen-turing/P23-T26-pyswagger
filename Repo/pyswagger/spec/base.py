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
        return []

    if ct == None:
        ret = [f(ct, v)]
    elif ct == ContainerType.list_:
        ret = []
        for i, vv in enumerate(v):
            ret.append(f(ct, vv))
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
    __swagger_ref_object__ = None

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
        """ check if obj is instance of the class this Context produces

        :param obj: the object to check
        :return: True if obj is an instance of __swagger_ref_object__
        :rtype: bool
        """
        if kls.__swagger_ref_object__:
            return isinstance(obj, kls.__swagger_ref_object__)
        return False

    def produce(self):
        """ produce the object, this method is called in __exit__

        :return: the produced object or None
        """
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        """ When exiting parsing context, doing two things
        - create the object corresponding to this parsing context.
        - populate the created object to parent context.
        """
        if self._obj == None:
            return

        # Check if there's a custom produce method
        obj = self.produce()

        # If produce returned something, use it (even if it's not the expected type)
        # This allows contexts to return custom types like bool
        if obj != None:
            # Custom produce result - use as is
            pass
        elif self.__swagger_ref_object__:
            # Create the object using the ref object factory
            obj = self.__swagger_ref_object__(self)
            # Validate the produced object only for auto-created objects
            if obj != None and not self.is_produced(obj):
                raise ValueError('Object is not instance of {0} but {1}'.format(
                    self.__swagger_ref_object__.__name__,
                    obj.__class__.__name__))

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
            if kk not in self._obj:
                self._obj[kk] = [] if ct == ContainerType.list_ else {}
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

        if not issubclass(type(ctx), Context):
            raise TypeError('should provide args[0] as Context, not: ' + ctx.__class__.__name__)

        self.__origin_keys = set([k for k in six.iterkeys(ctx._obj)])

        # handle fields - collect all fields from the entire MRO
        all_fields = {}
        all_internal = {}

        # Walk through MRO in reverse order so child fields override parent fields
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, '__swagger_fields__'):
                all_fields.update(cls.__swagger_fields__)
            if hasattr(cls, '__internal_fields__'):
                all_internal.update(cls.__internal_fields__)

        # Initialize fields with proper default values
        for name, default in six.iteritems(all_fields):
            # Create a copy of mutable defaults
            if isinstance(default, list):
                value = list(default) if default else []
            elif isinstance(default, dict):
                value = dict(default) if default else {}
            else:
                value = default

            # Get the value from context if available
            if name in ctx._obj:
                value = ctx._obj[name]

            setattr(self, self.get_private_name(name), value)

        for name, default in six.iteritems(all_internal):
            # Create a copy of mutable defaults
            if isinstance(default, list):
                value = list(default) if default else []
            elif isinstance(default, dict):
                value = dict(default) if default else {}
            else:
                value = default

            if name in ctx._obj:
                value = ctx._obj[name]

            setattr(self, self.get_private_name(name), value)

        self._assign_parent(ctx)

    def _assign_parent(self, ctx):
        """ parent assignment, internal usage only
        """
        def _assign(cls, _, obj):
            if obj == None:
                return

            if cls.is_produced(obj):
                if isinstance(obj, BaseObj):
                    obj._parent__ = self
            else:
                raise ValueError('Object is not instance of {0} but {1}'.format(cls.__swagger_ref_object__.__name__, obj.__class__.__name__))

        # set self as childrent's parent
        for name, (ct, ctx) in six.iteritems(ctx.__swagger_child__):
            obj = getattr(self, name, None)
            if obj == None:
                continue

            container_apply(ct, obj, functools.partial(_assign, ctx))

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

        :param ts: list of tokens or a string
        :return: resolved object
        """
        if ts == None:
            return None

        # Convert string to list
        if isinstance(ts, six.string_types):
            ts = [ts]

        obj = self
        for t in ts:
            if hasattr(obj, t):
                obj = getattr(obj, t)
            elif isinstance(obj, dict) and t in obj:
                obj = obj[t]
            elif isinstance(obj, list):
                try:
                    obj = obj[int(t)]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return obj

    def merge(self, other, ctx):
        """ merge properties from other object,
        only merge from 'not None' to 'None'.

        :param BaseObj other: the source object to be merged from.
        :param Context ctx: the parsing context class
        """
        if other == None:
            return self

        def _deep_copy_obj(obj, context_cls):
            """Helper to deep copy a BaseObj using its context"""
            if not isinstance(obj, BaseObj):
                return copy.deepcopy(obj)
            tmp = {'t': {}}
            with context_cls(tmp, 't') as c:
                c._obj = obj.dump() or {}
            result = tmp['t']
            if result and hasattr(result, '_parent__'):
                result._parent__ = self
            return result

        # Handle children merging based on context
        if hasattr(ctx, '__swagger_child__'):
            for field_name, (ct, child_ctx) in six.iteritems(ctx.__swagger_child__):
                other_val = getattr(other, field_name, None)
                self_val = getattr(self, field_name, None)

                if other_val == None:
                    continue

                # If self doesn't have this field, deep copy it from other
                if self_val == None:
                    if ct == None:
                        # Single object
                        setattr(self, self.get_private_name(field_name), _deep_copy_obj(other_val, child_ctx))
                    elif ct == ContainerType.list_:
                        # List of objects
                        new_list = []
                        for item in other_val:
                            new_list.append(_deep_copy_obj(item, child_ctx))
                        setattr(self, self.get_private_name(field_name), new_list)
                    elif ct == ContainerType.dict_:
                        # Dict of objects
                        new_dict = {}
                        for k, v in six.iteritems(other_val):
                            new_dict[k] = _deep_copy_obj(v, child_ctx)
                        setattr(self, self.get_private_name(field_name), new_dict)
                    elif ct == ContainerType.dict_of_list_:
                        # Dict of lists
                        new_dict = {}
                        for k, lst in six.iteritems(other_val):
                            new_list = []
                            for v in lst:
                                new_list.append(_deep_copy_obj(v, child_ctx))
                            new_dict[k] = new_list
                        setattr(self, self.get_private_name(field_name), new_dict)

                # Existing field - only merge dicts
                elif ct == ContainerType.dict_ and isinstance(self_val, dict):
                    # Merge dict entries that don't exist in self
                    for k, v in six.iteritems(other_val):
                        if k not in self_val:
                            self_val[k] = _deep_copy_obj(v, child_ctx)

        # Also handle non-child fields
        all_fields = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, '__swagger_fields__'):
                all_fields.update(cls.__swagger_fields__)

        # Skip fields that are children (already handled above)
        child_fields = set()
        if hasattr(ctx, '__swagger_child__'):
            child_fields = set(ctx.__swagger_child__.keys())

        for name in all_fields:
            if name in child_fields:
                continue

            other_val = getattr(other, name, None)
            self_val = getattr(self, name, None)

            if other_val != None and self_val == None:
                # Deep copy the value to avoid references
                if isinstance(other_val, (list, dict)):
                    setattr(self, self.get_private_name(name), copy.deepcopy(other_val))
                else:
                    setattr(self, self.get_private_name(name), other_val)

        return self

    def is_set(self, k):
        """ check if a key is setted from Swagger API document

        :param k: the key to check
        :return: True if the key is setted. False otherwise, it means we would get value
        from default from Field.
        """
        return k in self.__origin_keys

    def compare(self, other, base=None):
        """ comparison, will return the first difference

        :param other: object to compare with
        :param base: base path for difference reporting
        :return: tuple (is_same, difference_path)
        """
        if other == None:
            return (False, base or '')

        if self.__class__ != other.__class__:
            return (False, base or '')

        # Get all fields to compare
        all_fields = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, '__swagger_fields__'):
                all_fields.update(cls.__swagger_fields__)

        # Compare each field
        for name in all_fields:
            self_val = getattr(self, name, None)
            other_val = getattr(other, name, None)

            # Build the path
            if base:
                path = base + '/' + name
            else:
                path = name

            # Handle None comparison
            if self_val == None and other_val == None:
                continue
            if self_val == None or other_val == None:
                return (False, path)

            # Compare lists
            if isinstance(self_val, list) and isinstance(other_val, list):
                if len(self_val) != len(other_val):
                    return (False, path)
                for i, (sv, ov) in enumerate(zip(self_val, other_val)):
                    item_path = path + '/' + str(i)
                    if isinstance(sv, BaseObj) and isinstance(ov, BaseObj):
                        result = sv.compare(ov, item_path)
                        if not result[0]:
                            return result
                    elif sv != ov:
                        return (False, item_path)

            # Compare dicts
            elif isinstance(self_val, dict) and isinstance(other_val, dict):
                # Check if all keys match
                if set(self_val.keys()) != set(other_val.keys()):
                    # Find the first different key
                    for k in set(self_val.keys()) | set(other_val.keys()):
                        if k not in self_val or k not in other_val:
                            return (False, path + '/' + jp_compose(k))

                for k in self_val:
                    sv = self_val[k]
                    ov = other_val[k]
                    item_path = path + '/' + jp_compose(k)
                    if isinstance(sv, BaseObj) and isinstance(ov, BaseObj):
                        result = sv.compare(ov, item_path)
                        if not result[0]:
                            return result
                    elif isinstance(sv, list) and isinstance(ov, list):
                        if len(sv) != len(ov):
                            return (False, item_path)
                        for i, (svv, ovv) in enumerate(zip(sv, ov)):
                            item_item_path = item_path + '/' + str(i)
                            if isinstance(svv, BaseObj) and isinstance(ovv, BaseObj):
                                result = svv.compare(ovv, item_item_path)
                                if not result[0]:
                                    return result
                            elif svv != ovv:
                                return (False, item_item_path)
                    elif sv != ov:
                        return (False, item_path)

            # Compare BaseObj
            elif isinstance(self_val, BaseObj) and isinstance(other_val, BaseObj):
                result = self_val.compare(other_val, path)
                if not result[0]:
                    return result

            # Direct comparison
            elif self_val != other_val:
                return (False, path)

        return (True, '')

    def dump(self):
        """ dump object to dict

        :return: dict representation of object
        :rtype: dict
        """
        r = {}

        def _dump_(obj):
            if obj == None:
                return None
            elif isinstance(obj, dict):
                ret = {}
                for k, v in six.iteritems(obj):
                    dumped = _dump_(v)
                    # Keep the entry even if dumped is None/empty for BaseObj
                    if dumped != None or isinstance(v, BaseObj):
                        ret[k] = dumped if dumped != None else {}
                return ret if ret else None
            elif isinstance(obj, list):
                ret = []
                for v in obj:
                    dumped = _dump_(v)
                    # Keep the entry even if dumped is None/empty for BaseObj
                    if dumped != None or isinstance(v, BaseObj):
                        ret.append(dumped if dumped != None else {})
                # Return the list even if it only contains empty objects
                return ret if len(ret) > 0 else None
            elif isinstance(obj, BaseObj):
                return obj.dump()
            elif isinstance(obj, (six.string_types, six.integer_types, bool, float)):
                return obj
            else:
                # Try to convert to a basic type
                return obj

        # Get all fields from MRO
        all_fields = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, '__swagger_fields__'):
                all_fields.update(cls.__swagger_fields__)

        for name, default in six.iteritems(all_fields):
            # only dump a field when its value is not equal to default value
            v = getattr(self, name, None)
            if v != default:
                d = _dump_(v)
                if d != None:
                    r[name] = d

        # Return empty dict for empty objects instead of None
        return r

    @property
    def _parent_(self):
        """ get parent object

        :return: the parent object.
        :rtype: a subclass of BaseObj.
        """
        return self._parent__ if hasattr(self, '_parent__') else None

    @property
    def _field_names_(self):
        """ get list of field names defined in Swagger spec

        :return: a list of field names
        :rtype: a list of str
        """
        ret = []
        # Get all fields from MRO
        all_fields = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, '__swagger_fields__'):
                all_fields.update(cls.__swagger_fields__)

        for n in six.iterkeys(all_fields):
            # Apply rename if needed
            renamed = self.__swagger_rename__.get(n, n)
            ret.append(renamed)

        return ret

    @property
    def _children_(self):
        """ get children objects

        :rtype: a dict of children {child_name: child_object}
        """
        result = {}

        # Find child contexts from the class hierarchy
        child_info = {}
        for cls in self.__class__.__mro__:
            # Find the corresponding Context class
            ctx_name = cls.__name__.replace('Obj', 'Context')
            # Check in globals and parent module
            import sys
            for module in [sys.modules[cls.__module__], sys.modules.get('pyswagger.spec.base'), sys.modules.get('pyswagger.tests.test_base')]:
                if module and hasattr(module, ctx_name):
                    ctx_cls = getattr(module, ctx_name)
                    if hasattr(ctx_cls, '__swagger_child__'):
                        child_info.update(ctx_cls.__swagger_child__)
                        break

        # Process each child field
        for field_name, (ct, _) in six.iteritems(child_info):
            field_val = getattr(self, field_name, None)
            if field_val == None:
                continue

            if ct == None:
                # Single object
                if isinstance(field_val, BaseObj):
                    result[field_name] = field_val
            elif ct == ContainerType.list_:
                # List of objects
                if isinstance(field_val, list):
                    for i, obj in enumerate(field_val):
                        if isinstance(obj, BaseObj):
                            result[field_name + '/' + str(i)] = obj
            elif ct == ContainerType.dict_:
                # Dict of objects
                if isinstance(field_val, dict):
                    for k, obj in six.iteritems(field_val):
                        if isinstance(obj, BaseObj):
                            result[field_name + '/' + jp_compose(k)] = obj
            elif ct == ContainerType.dict_of_list_:
                # Dict of lists
                if isinstance(field_val, dict):
                    for k, lst in six.iteritems(field_val):
                        if isinstance(lst, list):
                            for i, obj in enumerate(lst):
                                if isinstance(obj, BaseObj):
                                    result[field_name + '/' + jp_compose(k) + '/' + str(i)] = obj

        return result


def _method_(name):
    """ getter factory """
    def _getter_(self):
        return getattr(self, self.get_private_name(name))
    return _getter_


class FieldMeta(type):
    """ metaclass to init fields
    """
    def __new__(metacls, name, bases, spc):
        """ scan through MRO to get a merged list of fields,
        and create those fields.
        """
        def init_fields(fields, rename):
            for name in six.iterkeys(fields):
                renamed = rename.get(name, name)
                spc[renamed] = property(_method_(name))

        def _default_(name, default):
            spc[name] = spc[name] if name in spc else default

        def _update_(dict1, dict2):
            d = {}
            for k in set(dict2.keys()) - set(dict1.keys()):
                d[k] = dict2[k]
            dict1.update(d)

        # compose fields definition from parents
        all_swagger_fields = {}
        all_internal_fields = {}
        all_rename = {}

        # Walk through all base classes to collect fields
        for base in bases:
            if hasattr(base, '__swagger_fields__'):
                _update_(all_swagger_fields, base.__swagger_fields__)
            if hasattr(base, '__internal_fields__'):
                _update_(all_internal_fields, base.__internal_fields__)
            if hasattr(base, '__swagger_rename__'):
                all_rename.update(base.__swagger_rename__)

        # Add current class fields (they override parent fields)
        if '__swagger_fields__' in spc:
            all_swagger_fields.update(spc['__swagger_fields__'])
            # Store the merged fields back
            spc['__swagger_fields__'] = all_swagger_fields

        if '__internal_fields__' in spc:
            all_internal_fields.update(spc['__internal_fields__'])
            spc['__internal_fields__'] = all_internal_fields

        if '__swagger_rename__' in spc:
            all_rename.update(spc['__swagger_rename__'])
        spc['__swagger_rename__'] = all_rename

        # Initialize properties for all fields
        init_fields(all_swagger_fields, all_rename)
        init_fields(all_internal_fields, {})

        return type.__new__(metacls, name, bases, spc)


class NullContext(Context):
    """ black magic to initialize BaseObj
    """

    _obj = {}

    def __init__(self):
        super(NullContext, self).__init__(None, None)
