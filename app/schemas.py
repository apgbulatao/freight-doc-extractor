"""
Extraction schemas for the three v1 doc types: commercial invoice, packing
list, bill of lading. Each shares a common core (shipper/consignee/weight)
but has fields unique to that doc type.

These double as (a) the JSON shape we prompt the model for and (b) the
validation layer before anything is written to the DB.
"""

import re
import typing
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

DocType = Literal["commercial_invoice", "packing_list", "bill_of_lading"]

VALID_INCOTERMS = {
    "EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP",
}

# Common field labels/headings that appear on freight documents. incoterm
# has a closed vocabulary, so a strict enum validator (below) both catches
# bad values *and* rejects label-echo automatically. Free-text fields like
# payment_terms have no closed vocabulary, so they need a separate,
# heuristic check for the same failure mode: the VLM copying a document's
# field LABEL into a field instead of leaving it null when the value isn't
# present. This lexicon -- and the check that uses it -- applies to every
# open-text field across all three doc types, not just one.
LABEL_LEXICON = {
    "shipper", "consignee", "port of loading", "port of discharge",
    "gross weight", "net weight", "incoterm", "incoterms",
    "payment terms", "terms of payment", "payment method",
    "invoice number", "invoice no", "invoice date",
    "currency", "total value", "total amount",
    "packing list number", "packing list no", "total packages",
    "number of packages", "dimensions",
    "bol number", "b l number", "bl no", "b l awb no", "awb no",
    "vessel name", "vessel", "voyage number", "voyage no",
    "carrier", "date of shipment", "shipment date",
    "freight terms", "freight", "description", "quantity", "unit",
    "declared value",
}

_STOPWORDS = {"of", "no", "number", "the", "a", "an"}


def _normalize_tokens(text: str) -> frozenset[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return frozenset(t for t in cleaned.split() if t and t not in _STOPWORDS)


_LABEL_TOKEN_SETS = {_normalize_tokens(label) for label in LABEL_LEXICON}


def looks_like_label(value: str) -> bool:
    """True if `value` looks like a copied field label/heading rather than
    an extracted value: its normalized words match a known label phrase
    almost exactly, with nothing else in the string. Real extracted values
    (company names, dates, dimensions, etc.) essentially never collapse
    down to exactly a bare label phrase, so this is a low false-positive,
    high-recall check for the specific failure we've seen."""
    value_tokens = _normalize_tokens(value)
    if not value_tokens or len(value_tokens) > 5:
        return False
    return value_tokens in _LABEL_TOKEN_SETS


def _is_open_text_field(annotation) -> bool:
    """True for a plain Optional[str] field -- as opposed to Literal[...]
    fields (like freight_terms), which already have their own closed-set
    validation and don't need this heuristic."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        return len(non_none) == 1 and non_none[0] is str
    return annotation is str


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit: Optional[str] = None  # pcs, pallets, kg, etc.
    declared_value: Optional[float] = None


class CoreFields(BaseModel):
    """Fields shared across all freight doc types."""

    shipper: Optional[str] = None
    consignee: Optional[str] = None
    port_of_loading: Optional[str] = None
    port_of_discharge: Optional[str] = None
    gross_weight_kg: Optional[float] = None
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("gross_weight_kg")
    @classmethod
    def weight_must_be_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError("gross_weight_kg cannot be negative")
        return v

    @model_validator(mode="after")
    def no_label_echo_in_open_text_fields(self):
        """Catch the general bug class already fixed for `incoterm`: reject
        any open-text field whose value is actually a document field label
        (e.g. payment_terms == "Payment Terms") rather than a real value.
        Runs after per-field validators, so incoterm (already normalized/
        validated against its own enum by then) is checked too, harmlessly."""
        for field_name, field_info in type(self).model_fields.items():
            if field_name == "doc_type" or not _is_open_text_field(field_info.annotation):
                continue
            value = getattr(self, field_name, None)
            if isinstance(value, str) and looks_like_label(value):
                raise ValueError(
                    f"'{field_name}' contains '{value}', which looks like a copied "
                    f"document field label rather than an extracted value"
                )
        return self


class CommercialInvoice(CoreFields):
    doc_type: Literal["commercial_invoice"] = "commercial_invoice"
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    incoterm: Optional[str] = None
    payment_terms: Optional[str] = None
    currency: Optional[str] = None
    total_value: Optional[float] = None

    @field_validator("incoterm")
    @classmethod
    def incoterm_must_be_valid(cls, v):
        if v is not None and v.upper() not in VALID_INCOTERMS:
            raise ValueError(f"'{v}' is not a recognized Incoterm ({sorted(VALID_INCOTERMS)})")
        return v.upper() if v else v


class PackingList(CoreFields):
    doc_type: Literal["packing_list"] = "packing_list"
    packing_list_number: Optional[str] = None
    total_packages: Optional[int] = None
    net_weight_kg: Optional[float] = None
    dimensions: Optional[str] = None


class BillOfLading(CoreFields):
    doc_type: Literal["bill_of_lading"] = "bill_of_lading"
    bol_number: Optional[str] = None
    vessel_name: Optional[str] = None
    voyage_number: Optional[str] = None
    carrier: Optional[str] = None
    date_of_shipment: Optional[str] = None
    freight_terms: Optional[Literal["prepaid", "collect"]] = None

    @field_validator("freight_terms", mode="before")
    @classmethod
    def normalize_freight_terms(cls, v):
        """B/Ls print this as 'FREIGHT PREPAID' / 'FREIGHT COLLECT', not the
        bare 'prepaid'/'collect' our schema expects. Normalize real-world
        phrasing to the canonical value before the Literal check runs --
        same pattern as `incoterm`'s uppercasing, applied here instead of
        loosening the type."""
        if not isinstance(v, str):
            return v
        normalized = v.strip().upper()
        if "PREPAID" in normalized:
            return "prepaid"
        if "COLLECT" in normalized:
            return "collect"
        return v  # unrecognized -- let Literal validation reject it as before


SCHEMA_BY_DOC_TYPE: dict[DocType, type[CoreFields]] = {
    "commercial_invoice": CommercialInvoice,
    "packing_list": PackingList,
    "bill_of_lading": BillOfLading,
}


def schema_json_hint(doc_type: DocType) -> str:
    """Render the target schema as a compact JSON example for the prompt."""
    model = SCHEMA_BY_DOC_TYPE[doc_type]
    fields = model.model_fields
    example = {name: f"<{name}>" for name in fields if name != "doc_type"}
    example["line_items"] = [
        {"description": "<desc>", "quantity": 0, "unit": "<unit>", "declared_value": 0}
    ]
    import json

    return json.dumps(example, indent=2)
