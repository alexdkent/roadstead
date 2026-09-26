"""The structured blank-run abort threshold must actually REACH EndpointConfig.

🚨 This test exists because `_POLICY_PASSTHROUGH` drops an unlisted policy key
IN SILENCE — a declaration that never arrives is indistinguishable from a
feature deliberately left off. Same shape as
test_reasoning_loop_policy.py, which this file is the sibling of.
"""
from roadstead import model_catalog
from roadstead.config import EndpointConfig
from roadstead.correction import StructuredBlankRunDetector
from roadstead.model_catalog import _POLICY_PASSTHROUGH

KEY = "structured_blank_run_abort_chars"


def test_the_key_is_in_the_passthrough():
    assert KEY in _POLICY_PASSTHROUGH, (
        f"build_endpoint_kwargs no longer copies {KEY}; a stanza declaring it "
        f"would be silently dropped — add it to model_catalog._POLICY_PASSTHROUGH")


def test_threshold_reaches_endpoint_config():
    """End to end through the real builder — a synthetic entry, so this stays
    true for whichever endpoint next declares the field."""
    entry = model_catalog.EndpointEntry(
        name="probe", provider="p", kind="chat",
        policy={KEY: 512},
    )
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe"]
    assert kw[KEY] == 512
    ep = EndpointConfig(**{k: v for k, v in kw.items()
                           if k in EndpointConfig.__dataclass_fields__})
    assert ep.structured_blank_run_abort_chars == 512


def test_an_undeclared_endpoint_gets_the_zero_default():
    entry = model_catalog.EndpointEntry(name="probe", provider="p", kind="chat")
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe"]
    assert KEY not in kw
    ep = EndpointConfig(**{k: v for k, v in kw.items()
                           if k in EndpointConfig.__dataclass_fields__})
    assert ep.structured_blank_run_abort_chars == 0


def test_detector_arms_only_on_a_positive_threshold():
    assert StructuredBlankRunDetector(threshold=512).armed is True
    assert StructuredBlankRunDetector(threshold=0).armed is False
    assert StructuredBlankRunDetector().armed is False
