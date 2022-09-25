from typing import MutableMapping, Mapping
from typing import Dict, Generic, List, TypeVar, Any
import torch

DataArray = TypeVar(
    "DataArray", torch.Tensor, Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]
)



class Field(Generic[DataArray]):
    __slots__ = []  # type: ignore
    def index(self, vocab: Any): # allennlp/allennlp/data/vocabulary.py
        """
        Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
        This `modifies` the `Field` object, it does not return anything.
        If your `Field` does not have any strings that need to be converted into indices, you do
        not need to implement this method.
        """
        pass

class Instance(Mapping[str, Field]):
    """
    An `Instance` is a collection of :class:`~allennlp.data.fields.field.Field` objects,
    specifying the inputs and outputs to
    some model.  We don't make a distinction between inputs and outputs here, though - all
    operations are done on all fields, and when we return arrays, we return them as dictionaries
    keyed by field name.  A model can then decide which fields it wants to use as inputs as which
    as outputs.
    The `Fields` in an `Instance` can start out either indexed or un-indexed.  During the data
    processing pipeline, all fields will be indexed, after which multiple instances can be combined
    into a `Batch` and then converted into padded arrays.
    # Parameters
    fields : `Dict[str, Field]`
        The `Field` objects that will be used to produce data arrays for this instance.
    """

    __slots__ = ["fields", "indexed"]

    def __init__(self, fields: MutableMapping[str, Field]) -> None:
        self.fields = fields
        self.indexed = False

    # Add methods for `Mapping`.  Note, even though the fields are
    # mutable, we don't implement `MutableMapping` because we want
    # you to use `add_field` and supply a vocabulary.
    def __getitem__(self, key: str) -> Field:
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    # def add_field(self, field_name: str, field: Field, vocab: Vocabulary = None) -> None:
    #     """
    #     Add the field to the existing fields mapping.
    #     If we have already indexed the Instance, then we also index `field`, so
    #     it is necessary to supply the vocab.
    #     """
    #     self.fields[field_name] = field
    #     if self.indexed and vocab is not None:
    #         field.index(vocab)
