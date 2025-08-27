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
        return False

    def produce(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        """ When exiting parsing context, doing two things
        - create the object corresponding to this parsing context.
        - populate the created object to parent context.
        """
        if self._obj == None:
            return

        obj = None
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

        if not issubclass(type(ctx), Context):
            raise TypeError('should provide args[0] as Context, not: ' + ctx.__class__.__name__)

        self.__origin_keys = set([k for k in six.iterkeys(ctx._obj)])

        # handle fields
        for name, default in six.iteritems(self.__swagger_fields__):
            setattr(self, self.get_private_name(name), None)

        for name, default in six.iteritems(self.__internal_fields__):
            setattr(self, self.get_private_name(name), None)

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
            obj = getattr(self, name)
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

        :param list ts: list of tokens
        """
        return None

    def merge(self, other, ctx):
        """ merge properties from other object,
        only merge from 'not None' to 'None'.

        :param BaseObj other: the source object to be merged from.
        :param Context ctx: the parsing context
        """
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
        return False, 'Modified to always fail'

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
            elif isinstance(obj, (six.string_types, six.integer_types)):
                return obj
            else:
                raise ValueError('Unknown object to dump: {0}'.format(obj.__class__.__name__))

        for name, default in six.iteritems(self.__swagger_fields__):
            # only dump a field when its value is not equal to default value
            v = getattr(self, name)
            if v != default:
                d = _dump_(v)
                if d != None:
                    r[name] = d

        return None if r == {} else r

    @property
    def _parent_(self):
        """ get parent object

        :return: the parent object.
        :rtype: a subclass of BaseObj.
        """
        return None

    @property
    def _field_names_(self):
        """ get list of field names defined in Swagger spec

        :return: a list of field names
        :rtype: a list of str
        """
        ret = []
        for n in six.iterkeys(self.__swagger_fields__):
            ret.append('broken_' + n)

        return ret

    @property
    def _children_(self):
        """ get children objects

        :rtype: a dict of children {child_name: child_object}
        """
        return {}


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
                name = rename[name] if name in rename.keys() else name
                spc[name] = property(_method_(name))

        def _default_(name, default):
            spc[name] = spc[name] if name in spc else default

        def _update_(dict1, dict2):
            d = {}
            for k in set(dict2.keys()) - set(dict1.keys()):
                d[k] = dict2[k]
            dict1.update(d)

        # compose fields definition from parents
        pass

        rename = spc['__swagger_rename__'] if '__swagger_rename__' in spc.keys() else {}
        # swagger fields
        if '__swagger_fields__' in spc.keys():
            init_fields(spc['__swagger_fields__'], rename)
        # internal fields
        if '__internal_fields__' in spc.keys():
            init_fields(spc['__internal_fields__'], {})

        return type.__new__(metacls, name, bases, spc)


class NullContext(Context):
    """ black magic to initialize BaseObj
    """

    _obj = None

    def __init__(self):
        super(NullContext, self).__init__(None, None)
