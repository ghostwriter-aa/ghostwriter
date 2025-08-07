"""Wrapper for serializing dataclasses to and from JSON strings."""

import dataclasses
import enum
import functools
import json
import typing
from typing import Any, Mapping, TypeVar

import dacite

if typing.TYPE_CHECKING:
    import _typeshed

    T = TypeVar("T", bound=_typeshed.DataclassInstance)
else:
    T = TypeVar("T")


class EnumNameEncoder(json.JSONEncoder):
    """JSON encoder that supports encoding enums as their names ("keys")."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, enum.Enum):
            return obj.name
        return super().default(obj)


@functools.lru_cache(maxsize=1024, typed=True)
def get_all_enums(cls: type[T]) -> set[type[enum.Enum]]:
    """Returns a list of all Enum types in the provided dataclass type and any of its (recursive) member dataclasses."""
    previously_analyzed_types = set()

    def _get_all_enums_in_type(cls: type) -> set[type[enum.Enum]]:
        all_enums: set[type[enum.Enum]] = set()

        if cls in previously_analyzed_types:
            return all_enums
        if isinstance(cls, str):
            return all_enums  # Ignore forward definitions since we can't analyze their fields.
        previously_analyzed_types.add(cls)

        if typing.get_origin(cls):
            # Handle cases of Sequence[dataclass], Optional[dataclass], etc.
            for inner_type in typing.get_args(cls):
                all_enums |= _get_all_enums_in_type(inner_type)
        elif issubclass(cls, enum.Enum):
            all_enums.add(cls)
        elif dataclasses.is_dataclass(cls):
            # Recursively add all enums in the dataclass field.
            for field in dataclasses.fields(cls):
                all_enums |= _get_all_enums_in_type(field.type)  # type: ignore

        return all_enums

    return _get_all_enums_in_type(cls)


class JsonSerializable:
    """A mixin for dataclasses, allowing serialization to and from JSON strings.

    Supported data types are str, int, float, bool, Enum (see below), and lists, dicts, and dataclasses recursively
    containing any of the above.

    Simple usage:

    ```
    @dataclass
    class Person(JsonSerializable):
        name: str
        age: int
    ```

    The above will add two member functions to the dataclass:
    - Person.from_json(json_data): Returns a new Person object containing the parsed JSON data.
      Raises an error if the input is invalid, e.g., if a non-optional field is missing, or a field has invalid type.
      `json_data` can either be a parsed JSON dictionary, or an unparsed JSON string.
    - person.to_json_string(): Returns a JSON string containing a serialization of the object.
      The string can be converted back to a Person object using Person.from_json.

    Nested dataclasses:
    Any JsonSerializable dataclass can contain other dataclasses as fields. These will be serialized into dictionaries
    within the top-level dictionary. The inner dataclasses do not need to be subclassed from JsonSerializable (but they
    can, if you want to also be able to serialize them separately).

    Enums:
    Any Enum fields in the dataclass or its nested dataclasses will be serialized using the field's enum "name" written
    out as a string, and loaded back to an enum. For example:

    ```
    class Role(Enum):
        FIGHTER = 0
        MAGIC_USER = 1
        CLERIC = 2
        THIEF = 3

    @dataclass
    class Player(JsonSerializable):
        name: str
        role: Role
    ```

    The object Person(name="Merlin", role=Role.MAGIC_USER) will then be serialized into

    ```
    {
        name: "Merlin"
        role: "MAGIC_USER"
    }
    ```
    """

    @classmethod
    def from_json(cls: type[T], json_dict: str | Mapping[str, Any]) -> T:
        enum_types = get_all_enums(cls)  # get_all_enums is cached, so efficiency is not an issue.
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict)
        dacite_config = dacite.Config(
            strict=True, type_hooks={enum_type: enum_type.__getitem__ for enum_type in enum_types}
        )
        return dacite.from_dict(data_class=cls, data=json_dict, config=dacite_config)  # type: ignore

    def to_json_string(self: T, *, indent: int | None = None, **kwargs: Any) -> str:
        return json.dumps(dataclasses.asdict(self), cls=EnumNameEncoder, indent=indent, **kwargs)
