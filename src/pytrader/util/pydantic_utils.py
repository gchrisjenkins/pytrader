from typing import overload, Any

from pydantic import BaseModel


class PydanticModel(BaseModel):

    @overload
    @classmethod
    def __class_getitem__(cls, item: type) -> Any: ...

    @overload
    @classmethod
    def __class_getitem__(cls, item: tuple[type, ...]) -> Any: ...

    @overload
    @classmethod
    def __class_getitem__(cls, item: str) -> Any: ...

    @classmethod
    def __class_getitem__(cls, item: Any) -> Any:
        # Important: keep Pydantic’s actual behavior.
        return super().__class_getitem__(item)
