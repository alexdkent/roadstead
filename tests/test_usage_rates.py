"""The savings rate table: what it prices, what it zeroes, and the difference.

🚨 THE DEFECT THESE PIN IS SILENCE, NOT ARITHMETIC. ``cloud_cost_usd`` returns
0.0 for an endpoint nobody has priced exactly as it does for one that is
genuinely free, so a rename or a host move zeroes a lane and the total keeps
looking plausible. On the reference fleet that went unnoticed until ~$119 of
~$678 lifetime savings (17.5%) was missing across ten endpoint names.

So these tests assert the MAPPINGS, one endpoint at a time. A test that only
checked "the total is positive" would stay green with any single lane zeroed —
which is precisely how the hole opened.
"""
from __future__ import annotations

import pytest

from roadstead.usage_rates import (
    _ENDPOINT_CLASS,
    cloud_cost_usd,
    is_declared_endpoint,
)

# One second of audio, in the packing every audio producer uses.
SEC = 100


class TestEveryLiveEndpointIsPriced:
    """Each of these was silently booking $0 on the reference fleet.

    Parametrised one-per-endpoint deliberately: a loop inside a single test
    would let one restored mapping mask nine missing ones.
    """

    @pytest.mark.parametrize(
        "endpoint,expected_class",
        [
            # the split tier-2 endpoints — the bare "tier2" key stopped matching
            ("tier2-chat", "tier2"),
            ("tier2-analyst", "tier2"),
            ("tier2-flash", "tier2"),
            ("halogen", "tier2"),
            ("flash", "tier2"),
            # the tier1 router's own endpoint key
            ("gemma", "tier1"),
            ("gemma-router", "tier1"),
            ("gemma-greeter", "tier1"),
            # per-unit audio: the UNIT names, not the capability names
            ("unraid-whisper", "stt"),
            ("unraid-whisper-lyrics", "stt"),
            ("unraid-diarize", "diarize"),
            ("unraid-htdemucs", "stem"),
            ("orpheus-tts", "tts"),
            ("chatterbox-tts", "tts"),
            # host-qualified names, as the producers actually push them
            ("cortex-orpheus-tts", "tts"),
            ("cortex-chatterbox-tts", "tts"),
            ("cortex-diarize", "diarize"),
            ("cortex-diarize-offline", "diarize"),
            ("diarize-gpu", "diarize"),
            # a synthetic-traffic LLM lane is still priced (see the table note)
            ("usersim", "tier1"),
            # the multimodal embedding endpoints: an embedding, not an unknown name
            ("embed-mm", "embed"),
            ("eg2-embed", "embed"),
        ],
    )
    def test_endpoint_resolves_to_its_rate_class(self, endpoint, expected_class):
        assert _ENDPOINT_CLASS.get(endpoint) == expected_class

    @pytest.mark.parametrize(
        "endpoint",
        [
            "tier2-chat", "tier2-analyst", "tier2-flash", "gemma",
            "unraid-whisper", "unraid-whisper-lyrics", "unraid-diarize",
            "unraid-htdemucs", "orpheus-tts", "chatterbox-tts",
            "cortex-orpheus-tts", "cortex-chatterbox-tts", "cortex-diarize",
            "cortex-diarize-offline", "diarize-gpu", "usersim",
            "embed-mm", "eg2-embed",
        ],
    )
    def test_endpoint_bills_something_for_real_volume(self, endpoint):
        """The mapping existing is not the same as it producing money.

        A class present in ``_ENDPOINT_CLASS`` but absent from every rate dict
        still bills $0 — that is the same silent failure one layer down, so
        assert through the REAL entry point rather than the table.
        """
        cost = cloud_cost_usd(endpoint, 10_000 * SEC, 50_000)
        assert cost > 0.0, f"{endpoint} bills $0 on 10k units — priced in name only"


class TestMultimodalEmbedEndpointsPriceAsEmbed:
    """`embed-mm` / `eg2-embed` were unknown names: $0, listed under `undeclared`."""

    @pytest.mark.parametrize("endpoint", ["embed-mm", "eg2-embed"])
    def test_prices_exactly_like_embed_and_is_declared(self, endpoint):
        assert is_declared_endpoint(endpoint)
        assert cloud_cost_usd(endpoint, 2_000_000, 0) == pytest.approx(
            cloud_cost_usd("embed", 2_000_000, 0)
        )
        assert cloud_cost_usd(endpoint, 2_000_000, 0) > 0.0


class TestDeliberateZeroesStayZero:
    """These are decided, not forgotten — and each has a measured reason."""

    @pytest.mark.parametrize("endpoint", [
        "stream", "cortex-stream", "asr", "cortex-asr", "got-ocr",
        # media generators: no producer pushes a count, so nothing to multiply
        "imagegen", "comfyui", "meshgen", "videogen", "musicgen", "video",
    ])
    def test_endpoint_is_declared_but_free(self, endpoint):
        assert is_declared_endpoint(endpoint), (
            f"{endpoint} must stay DECLARED — an undeclared name is indistinguishable "
            "from an unpriced one, which is the whole defect"
        )
        assert cloud_cost_usd(endpoint, 1_000_000, 1_000_000) == 0.0

    def test_stream_stays_free_because_it_double_counts_whisper(self):
        """`stream` reads its duration off the whisper response it fronts.

        Pricing it as `stt` would bill the same audio twice. This test exists to
        make that a deliberate, argued zero rather than an oversight someone
        later "fixes".
        """
        assert _ENDPOINT_CLASS["stream"] is None
        # ...and the name it is actually pushed under, which the bare key never
        # matched once the host prefix stopped being stripped.
        assert _ENDPOINT_CLASS["cortex-stream"] is None


class TestTheUnknownEndpointIsDistinguishable:
    """The distinction whose absence cost the 17.5%."""

    def test_an_unknown_name_is_not_declared(self):
        assert not is_declared_endpoint("tier4-whatever-lands-next")

    def test_an_unknown_name_still_bills_zero(self):
        """Pricing must NOT raise — a typo cannot be allowed to kill the rollup.

        That is exactly why the loud check has to live at the summarising
        surface instead, which is what ``is_declared_endpoint`` exists for.
        """
        assert cloud_cost_usd("tier4-whatever-lands-next", 10**9, 10**9) == 0.0

    def test_declared_and_priced_are_different_questions(self):
        """A free lane and an unknown lane must not look the same."""
        free, unknown = "stream", "tier4-whatever-lands-next"
        assert cloud_cost_usd(free, 10**6, 0) == cloud_cost_usd(unknown, 10**6, 0)
        assert is_declared_endpoint(free) != is_declared_endpoint(unknown)


class TestWarnIfUndeclared:
    """The loud half of the distinction: a name with no decision is announced."""

    @pytest.fixture(autouse=True)
    def _fresh_warned_set(self, monkeypatch):
        from roadstead import usage_rates
        monkeypatch.setattr(usage_rates, "_WARNED_UNDECLARED", set())
        monkeypatch.setattr(usage_rates, "_cap_notice_sent", False)

    def test_an_undeclared_name_warns_once_and_a_declared_one_never(self, caplog):
        from roadstead.usage_rates import warn_if_undeclared
        with caplog.at_level("WARNING", logger="roadstead.usage_rates"):
            assert warn_if_undeclared("brand-new-unit") is True
            assert warn_if_undeclared("Brand-New-Unit ") is True   # same name
            assert warn_if_undeclared("stream") is False           # decided, $0
            assert warn_if_undeclared("tier3") is False            # decided, priced
        hits = [r for r in caplog.records if "brand-new-unit" in r.getMessage()]
        assert len(hits) == 1, "once per NAME, however often a dashboard polls"
        assert hits[0].levelname == "WARNING"

    def test_the_warned_set_is_bounded_but_the_answer_stays_true(self, caplog, monkeypatch):
        """Names are caller-supplied text, so the memory of them cannot grow
        without bound — and going quiet must not change what is REPORTED."""
        from roadstead import usage_rates
        monkeypatch.setattr(usage_rates, "_WARNED_UNDECLARED_CAP", 3)
        with caplog.at_level("WARNING", logger="roadstead.usage_rates"):
            for i in range(10):
                assert usage_rates.warn_if_undeclared(f"junk-{i}") is True
        assert len(usage_rates._WARNED_UNDECLARED) == 3
        assert sum("no longer logged" in r.getMessage()
                   for r in caplog.records) == 1, "the cap notice goes out once"

    def test_pricing_itself_never_raises_on_an_undeclared_name(self):
        assert cloud_cost_usd("nobody-priced-this", 10**6, 10**6) == 0.0


class TestAudioPackingIsDecodedNotGuessed:
    """`input_tokens = audio_seconds * 100`, verified at every producer."""

    def test_one_hour_of_speech_costs_the_published_hourly_rate(self):
        # Whisper-large-v3 on Groq: $0.111/hr.
        assert cloud_cost_usd("unraid-whisper", 3600 * SEC, 0) == pytest.approx(0.111)

    def test_one_hour_of_diarization_costs_its_own_rate(self):
        assert cloud_cost_usd("unraid-diarize", 3600 * SEC, 0) == pytest.approx(0.12)

    def test_a_tts_output_is_audio_centiseconds_and_must_not_be_priced(self):
        """🚨 TTS `output_tokens` holds audio centiseconds, NOT tokens.

        Both producers pack `output_tokens = int(audio_seconds_out * 100)`. If
        the tts branch ever gained an output rate, an ordinary backlog of
        synthesised speech would be billed as tens of millions of output
        tokens. Pin that output is inert.
        """
        chars = 1_000_000
        with_audio = cloud_cost_usd("orpheus-tts", chars, 24_000_000)
        without = cloud_cost_usd("orpheus-tts", chars, 0)
        assert with_audio == without == pytest.approx(15.0)


class TestPreviouslyPricedLanesDidNotMove:
    """The fix must be purely additive — a repricing would rewrite history."""

    @pytest.mark.parametrize(
        "endpoint,expected_class",
        [("thinker", "tier3"), ("tier3", "tier3"), ("creative", "tier3"),
         ("classifier", "tier1"), ("embed", "embed"), ("rerank", "rerank"),
         ("whisper-1", "stt")],
    )
    def test_existing_mapping_is_unchanged(self, endpoint, expected_class):
        assert _ENDPOINT_CLASS.get(endpoint) == expected_class

    def test_remote_spill_stays_unpriced_here(self):
        """Spill is real money and belongs to spend.py, not to this table."""
        for ep in ("spill-chat", "spill-reasoning"):
            assert _ENDPOINT_CLASS[ep] is None
            assert cloud_cost_usd(ep, 10**6, 10**6) == 0.0
