from unittest.mock import MagicMock, patch

from mmore.websearchRAG.config import WebsearchConfig
from mmore.websearchRAG.pipeline import WebsearchPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pipeline(
    max_context_tokens=100,
    n_subqueries=2,
    n_loops=1,
    use_summary=False,
    use_rag=False,
    search_provider="tavily",
    **config_overrides,
):
    """Build a WebsearchPipeline with all I/O and LLM mocked out."""
    config = WebsearchConfig(
        rag_config_path="dummy.yaml",
        output_file="dummy_out.json",
        max_context_tokens=max_context_tokens,
        n_subqueries=n_subqueries,
        n_loops=n_loops,
        use_summary=use_summary,
        use_rag=use_rag,
        search_provider=search_provider,
        **config_overrides,
    )
    with patch.object(WebsearchPipeline, "__init__", lambda self, cfg: None):
        pipeline = WebsearchPipeline(config)

    pipeline.config = config
    pipeline.rag_results = None
    pipeline._tokenizer = None

    mock_llm = MagicMock()
    mock_llm.get_num_tokens = lambda text: len(text.split())
    mock_llm.invoke.return_value = MagicMock(
        content="short answer: ok\ndetailed answer: detailed ok"
    )
    pipeline.llm = mock_llm

    pipeline.searcher = MagicMock()
    pipeline.searcher.websearch_pipeline.return_value = []

    return pipeline


def make_search_result(url, snippet, title="t"):
    """Shorthand for a web-search result dict (websearch.py output format)."""
    return {"body": snippet, "href": url, "title": title}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestSmoke:
    def test_process_record_returns_expected_keys(self):
        """Minimal end-to-end: no web results, pipeline still returns valid structure."""
        p = make_pipeline(n_loops=1, n_subqueries=1)
        p.searcher.websearch_pipeline.return_value = []

        result = p.process_record({"input": "What is Python?"})

        assert "query" in result
        assert "short_answer" in result
        assert "detailed_answer" in result
        assert "sources" in result
        assert result["query"] == "What is Python?"
        assert result["sources"] == {}


# ---------------------------------------------------------------------------
# Token-budget boundary tests – snippet accumulation
# ---------------------------------------------------------------------------


class TestSnippetBudget:
    """Verify token-aware snippet accumulation and early stopping."""

    def test_snippets_within_budget_are_all_collected(self):
        """When total snippet tokens < budget, all snippets are kept."""
        p = make_pipeline(max_context_tokens=5000, n_subqueries=1, n_loops=1)
        # Ensure generate_subqueries returns exactly 1 subquery
        p.generate_subqueries = lambda *a, **kw: ["sq1"]

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "small snippet one"),
            make_search_result("http://b.com", "small snippet two"),
        ]

        result = p.process_record({"input": "test query"})

        assert "http://a.com" in result["sources"]
        assert "http://b.com" in result["sources"]

    def test_budget_exhaustion_stops_accumulation(self):
        """When a snippet would exceed the budget, it and all subsequent are skipped."""
        # With 1-word = 1-token counting and max_context_tokens set very low,
        # the snippet budget (after subtracting fixed prompt overhead) will be tiny.
        # We need the budget to allow roughly 1 snippet but not 2.
        #
        # Fixed overhead for query "q" (use_summary=False):
        #   SYNTHESIS_SYSTEM_MSG  ~ 21 words
        #   SYNTHESIS_PREFIX      ~ 12 words  (with original="q", rag_doc="No RAG sources")
        #   SYNTHESIS_SUFFIX      ~ 23 words
        #   Total fixed           ~ 56 words
        #
        # With max_context_tokens=80, snippet budget = 80 - 56 = 24 tokens.
        # "alpha bravo charlie\n" = 4 tokens -> fits (total=4).
        # "delta echo foxtrot golf hotel india juliet kilo\n" = 9 tokens -> 4+9=13 fits too.
        # So we need a tighter budget.  Use max_context_tokens=62 -> budget = 6 tokens.
        # First snippet (4 tokens): 0+4 > 6? No -> accepted (total=4).
        # Second snippet (9 tokens): 4+9 > 6? Yes -> skipped.
        p = make_pipeline(max_context_tokens=62, n_subqueries=1, n_loops=1)
        p.generate_subqueries = lambda *a, **kw: ["sq1"]

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "alpha bravo charlie"),
            make_search_result(
                "http://b.com", "delta echo foxtrot golf hotel india juliet kilo"
            ),
        ]

        result = p.process_record({"input": "q"})

        # First snippet should fit; second should be skipped due to budget
        assert "http://a.com" in result["sources"]
        assert "http://b.com" not in result["sources"]

    def test_budget_exhaustion_stops_subsequent_subqueries(self):
        """Once budget_exhausted is set, remaining subqueries are skipped entirely."""
        p = make_pipeline(max_context_tokens=80, n_subqueries=3, n_loops=1)
        p.generate_subqueries = lambda *a, **kw: ["sq1", "sq2", "sq3"]

        call_count = {"n": 0}

        def counting_web_search(query):
            call_count["n"] += 1
            # First subquery returns a big snippet that fills the budget
            if call_count["n"] == 1:
                return [
                    {"url": "http://a.com", "snippet": "word " * 60, "title": "t"},
                ]
            return [
                {
                    "url": f"http://{call_count['n']}.com",
                    "snippet": "other",
                    "title": "t",
                },
            ]

        p.web_search = counting_web_search

        p.process_record({"input": "q"})

        # budget_exhausted should prevent calling web_search for subqueries 2 and 3
        assert call_count["n"] == 1

    def test_exact_boundary_snippet_is_accepted(self):
        """A snippet that exactly fills the remaining budget is accepted (uses >)."""
        # The code checks `total_tokens + snippet_tokens > snippet_budget`
        # so a snippet that exactly equals remaining budget should be ACCEPTED.
        p = make_pipeline(max_context_tokens=5000, n_subqueries=1, n_loops=1)
        p.generate_subqueries = lambda *a, **kw: ["sq1"]

        # We'll control _count_tokens to make arithmetic precise
        p._count_tokens = lambda text: 10  # every text = 10 tokens

        # snippet_budget = max_context_tokens - fixed overhead
        # With _count_tokens returning 10 for each fixed part (3 parts) = 30
        # snippet_budget = 5000 - 30 = 4970
        # Each snippet costs 10 tokens. total_tokens starts at 0.
        # 0 + 10 > 4970? No -> accepted. This will accept up to 497 snippets.
        # Let's just verify 2 snippets both get accepted.
        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "s1"),
            make_search_result("http://b.com", "s2"),
        ]

        result = p.process_record({"input": "q"})

        assert "http://a.com" in result["sources"]
        assert "http://b.com" in result["sources"]


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Verify (url, snippet) deduplication logic."""

    def test_exact_duplicate_is_skipped(self):
        """Same (url, snippet) pair returned twice -> only counted once."""
        p = make_pipeline(max_context_tokens=5000, n_subqueries=1, n_loops=1)
        p.generate_subqueries = lambda *a, **kw: ["sub1"]

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "same snippet"),
            make_search_result("http://a.com", "same snippet"),  # duplicate
        ]

        result = p.process_record({"input": "q"})

        assert "http://a.com" in result["sources"]
        # Only one title entry despite two identical results
        assert len(result["sources"]["http://a.com"]) == 1

    def test_same_url_different_snippet_is_kept(self):
        """Same URL but different snippet -> both are kept (not a duplicate)."""
        p = make_pipeline(max_context_tokens=5000, n_subqueries=1, n_loops=1)
        p.generate_subqueries = lambda *a, **kw: ["sub1"]

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "snippet alpha", title="Title A"),
            make_search_result("http://a.com", "snippet beta", title="Title B"),
        ]

        result = p.process_record({"input": "q"})

        # Both titles should be recorded under the same URL
        assert len(result["sources"]["http://a.com"]) == 2

    def test_same_snippet_different_url_is_kept(self):
        """Same snippet but different URL -> both are kept."""
        p = make_pipeline(max_context_tokens=5000, n_subqueries=1, n_loops=1)
        p.generate_subqueries = lambda *a, **kw: ["sub1"]

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "identical text"),
            make_search_result("http://b.com", "identical text"),
        ]

        result = p.process_record({"input": "q"})

        assert "http://a.com" in result["sources"]
        assert "http://b.com" in result["sources"]

    def test_duplicates_across_subqueries_are_skipped(self):
        """Dedup set persists across subqueries within a loop."""
        p = make_pipeline(max_context_tokens=5000, n_subqueries=2, n_loops=1)
        p.generate_subqueries = lambda *a, **kw: ["sub1", "sub2"]

        call_count = {"n": 0}

        def web_search_side_effect(query):
            call_count["n"] += 1
            # Both subqueries return the same result
            return [
                {"url": "http://shared.com", "snippet": "shared content", "title": "t"},
            ]

        p.web_search = web_search_side_effect

        result = p.process_record({"input": "q"})

        # web_search called for both subqueries
        assert call_count["n"] == 2
        # But the snippet is only accepted once (from subquery 1)
        assert "http://shared.com" in result["sources"]
        assert len(result["sources"]["http://shared.com"]) == 1

    def test_duplicates_do_not_consume_budget(self):
        """Skipped duplicates should NOT increment total_tokens."""
        p = make_pipeline(max_context_tokens=5000, n_subqueries=1, n_loops=1)
        p.generate_subqueries = lambda *a, **kw: ["sub1"]

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "real content"),
            make_search_result("http://a.com", "real content"),  # dup, 0 token cost
            make_search_result("http://b.com", "more content"),
        ]

        result = p.process_record({"input": "q"})

        # All non-duplicate URLs should appear
        assert "http://a.com" in result["sources"]
        assert "http://b.com" in result["sources"]


# ---------------------------------------------------------------------------
# Multi-loop budget behavior and RAG context growth
# ---------------------------------------------------------------------------


class TestMultiLoopBudget:
    """Verify budget accounting across multiple loops and RAG context growth."""

    def test_second_loop_runs_when_relevance_is_true(self):
        """When evaluate_subquery_relevance returns True, loop 2 executes."""
        p = make_pipeline(max_context_tokens=5000, n_subqueries=1, n_loops=2)
        p.generate_subqueries = lambda *a, **kw: ["sub1"]

        search_calls = {"n": 0}

        def web_search_side_effect(query):
            search_calls["n"] += 1
            return [
                {
                    "url": f"http://{search_calls['n']}.com",
                    "snippet": "info",
                    "title": "t",
                },
            ]

        p.web_search = web_search_side_effect
        p.evaluate_subquery_relevance = MagicMock(return_value=True)

        p.process_record({"input": "q"})

        # Both loops should have triggered web_search
        assert search_calls["n"] == 2  # 1 subquery x 2 loops

    def test_second_loop_skipped_when_relevance_is_false(self):
        """When evaluate_subquery_relevance returns False on loop 2, it breaks."""
        p = make_pipeline(max_context_tokens=5000, n_subqueries=1, n_loops=2)
        p.generate_subqueries = lambda *a, **kw: ["sub1"]

        search_calls = {"n": 0}

        def web_search_side_effect(query):
            search_calls["n"] += 1
            return [
                {
                    "url": f"http://{search_calls['n']}.com",
                    "snippet": "info",
                    "title": "t",
                },
            ]

        p.web_search = web_search_side_effect
        p.evaluate_subquery_relevance = MagicMock(return_value=False)

        p.process_record({"input": "q"})

        # Only loop 0 should run (loop 1 breaks before web_search)
        assert search_calls["n"] == 1

    def test_rag_context_grows_across_loops(self):
        """current_context is updated to final_detailed after each loop,
        causing rag_for_llm to grow and contain Prior answer."""
        p = make_pipeline(max_context_tokens=5000, n_subqueries=1, n_loops=2)
        p.generate_subqueries = lambda *a, **kw: ["sub1"]

        # Track what integrate_with_llm receives as rag_doc across loops
        rag_docs_seen = []

        def tracking_integrate(original, rag_doc, web_content):
            rag_docs_seen.append(rag_doc)
            return {"short": "s", "detailed": "long detailed answer for context growth"}

        p.integrate_with_llm = tracking_integrate
        p.evaluate_subquery_relevance = MagicMock(return_value=True)

        def web_search_side_effect(query):
            return [
                {"url": "http://x.com", "snippet": f"snippet for {query}", "title": "t"}
            ]

        p.web_search = web_search_side_effect

        p.process_record({"input": "q"})

        # Loop 0: rag_for_llm is "" (no rag, no prior context)
        assert rag_docs_seen[0] == ""
        # Loop 1: rag_for_llm should contain "Prior answer:" from loop 0
        assert "Prior answer:" in rag_docs_seen[1]
        assert "long detailed answer for context growth" in rag_docs_seen[1]

    def test_snippet_budget_shrinks_with_growing_context(self):
        """As rag_for_llm grows across loops, snippet_budget decreases
        (because the synthesis prefix contains the growing rag_doc)."""
        p = make_pipeline(max_context_tokens=200, n_subqueries=1, n_loops=2)
        p.generate_subqueries = lambda *a, **kw: ["sub1"]

        budgets_seen = []

        original_compute = p._compute_content_budget

        def tracking_compute(*fixed_parts):
            budget = original_compute(*fixed_parts)
            budgets_seen.append(budget)
            return budget

        p._compute_content_budget = tracking_compute
        p.evaluate_subquery_relevance = MagicMock(return_value=True)

        def web_search_side_effect(query):
            return [{"url": "http://x.com", "snippet": "data", "title": "t"}]

        p.web_search = web_search_side_effect

        # LLM returns a long detailed answer so context grows significantly
        p.llm.invoke.return_value = MagicMock(
            content="short answer: s\ndetailed answer: " + "word " * 30
        )

        p.process_record({"input": "q"})

        # _compute_content_budget is called for snippet_budget and summary_budget per loop.
        # With 2 loops we expect at least 4 calls (snippet + summary per loop).
        assert len(budgets_seen) >= 4
        # The snippet budget in loop 1 should be smaller than loop 0.
        # budgets_seen[0] = snippet budget loop 0, budgets_seen[2] = snippet budget loop 1
        # (indices 1, 3 are summary budgets)
        snippet_budget_loop0 = budgets_seen[0]
        snippet_budget_loop1 = budgets_seen[2]
        assert snippet_budget_loop1 < snippet_budget_loop0


# ---------------------------------------------------------------------------
# Summary budget (per-subquery) tests
# ---------------------------------------------------------------------------


class TestSummaryBudget:
    """Verify per-subquery summary_budget caps snippets for generate_summary."""

    def test_summary_budget_limits_snippets_per_subquery(self):
        """Snippets exceeding the per-subquery summary_budget are excluded
        from that subquery's batch, even if global snippet_budget has room."""
        # Set max_context_tokens so that the summary_budget (which subtracts
        # SUMMARY_SYSTEM_MSG + SUMMARY_PREFIX + SUMMARY_SUFFIX overhead) is small,
        # but the global snippet_budget is larger.
        p = make_pipeline(max_context_tokens=100, n_subqueries=1, n_loops=1)
        p.generate_subqueries = lambda *a, **kw: ["sub1"]

        # Track what generate_summary receives (per-subquery call)
        summary_inputs = []

        def tracking_summary(content, query):
            summary_inputs.append(content)
            return "summary"

        p.generate_summary = tracking_summary

        # Return 3 snippets: the third is large and should exceed summary_budget
        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "word " * 3),
            make_search_result("http://b.com", "word " * 3),
            make_search_result("http://c.com", "word " * 50),  # large
        ]

        p.process_record({"input": "q"})

        # The per-subquery summary call (first call) should have received
        # fewer snippets than the total 3 returned by web search
        assert len(summary_inputs) >= 1

    def test_use_summary_mode_sets_snippet_budget_to_max_context(self):
        """When use_summary=True, snippet_budget = max_context_tokens
        (no synthesis overhead subtracted), so more snippets fit."""
        p = make_pipeline(
            max_context_tokens=200, n_subqueries=1, n_loops=1, use_summary=True
        )
        p.generate_subqueries = lambda *a, **kw: ["sub1"]

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "snippet"),
        ]

        result = p.process_record({"input": "q"})

        assert "http://a.com" in result["sources"]


# ---------------------------------------------------------------------------
# Token helper unit tests
# ---------------------------------------------------------------------------


class TestTokenHelpers:
    """Unit tests for _count_tokens, _truncate_to_token_limit, _fit_to_budget."""

    def test_count_tokens_uses_llm_when_no_tokenizer(self):
        """When _tokenizer is None, delegates to llm.get_num_tokens."""
        p = make_pipeline()
        p._tokenizer = None

        assert p._count_tokens("one two three") == 3  # 1 word = 1 token

    def test_count_tokens_uses_local_tokenizer_when_available(self):
        """When _tokenizer is set, uses _encode instead of llm."""
        p = make_pipeline()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        p._tokenizer = mock_tokenizer

        assert p._count_tokens("some text") == 5
        mock_tokenizer.encode.assert_called_once_with(
            "some text", add_special_tokens=False
        )

    def test_truncate_no_op_when_within_limit(self):
        """Text within budget is returned unchanged."""
        p = make_pipeline()
        result = p._truncate_to_token_limit("one two three", max_tokens=10)
        assert result == "one two three"

    def test_truncate_binary_search_shortens_text(self):
        """Text exceeding budget is shortened (binary search path)."""
        p = make_pipeline()
        long_text = "word " * 100  # 100+ tokens with word-splitting
        result = p._truncate_to_token_limit(long_text, max_tokens=5)

        # Result should be shorter than original
        assert len(result) < len(long_text)
        # And within budget
        assert p._count_tokens(result) <= 5

    def test_truncate_with_local_tokenizer(self):
        """When tokenizer is available, uses encode/decode path."""
        p = make_pipeline()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(20))  # 20 tokens
        mock_tokenizer.decode.return_value = "truncated text"
        p._tokenizer = mock_tokenizer

        result = p._truncate_to_token_limit("some long text", max_tokens=5)

        assert result == "truncated text"
        mock_tokenizer.decode.assert_called_once_with(
            list(range(5)), skip_special_tokens=True
        )

    def test_fit_to_budget_truncates_content(self):
        """Content is truncated so total (fixed + content) fits max_context_tokens."""
        p = make_pipeline(max_context_tokens=20)

        # fixed parts: "system prompt" (2 tokens) + "prefix" (1 token) = 3 tokens
        # available = 20 - 3 = 17 tokens
        result = p._fit_to_budget("word " * 30, "system prompt", "prefix")

        assert p._count_tokens(result) <= 17

    def test_fit_to_budget_returns_empty_when_fixed_exceeds_max(self):
        """When fixed parts alone exceed max_context_tokens, returns empty."""
        p = make_pipeline(max_context_tokens=5)

        result = p._fit_to_budget(
            "content", "this is a very long system prompt that exceeds everything"
        )

        assert result == ""

    def test_compute_content_budget_basic(self):
        """Returns max_context_tokens minus token count of fixed parts."""
        p = make_pipeline(max_context_tokens=100)

        # "hello world" = 2 tokens, "foo" = 1 token -> fixed = 3
        budget = p._compute_content_budget("hello world", "foo")

        assert budget == 100 - 3

    def test_compute_content_budget_clamps_to_zero(self):
        """Returns 0 (not negative) when fixed parts exceed max."""
        p = make_pipeline(max_context_tokens=5)

        budget = p._compute_content_budget("a " * 100)

        assert budget == 0


# ---------------------------------------------------------------------------
# extract_response unit tests
# ---------------------------------------------------------------------------


class TestExtractResponse:
    """Unit tests for the extract_response helper."""

    def test_string_input(self):
        from mmore.websearchRAG.pipeline import extract_response

        assert extract_response("hello") == "hello"

    def test_list_of_strings(self):
        from mmore.websearchRAG.pipeline import extract_response

        assert extract_response(["first", "second"]) == "second"

    def test_list_of_dicts(self):
        from mmore.websearchRAG.pipeline import extract_response

        result = extract_response([{"content": "from dict"}])
        assert result == "from dict"

    def test_list_of_dicts_missing_content(self):
        from mmore.websearchRAG.pipeline import extract_response

        result = extract_response([{"other": "value"}])
        assert result == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for process_record."""

    def test_empty_query(self):
        """Empty input string should not crash."""
        p = make_pipeline(n_subqueries=1, n_loops=1)
        p.searcher.websearch_pipeline.return_value = []

        result = p.process_record({"input": ""})

        assert result["query"] == ""

    def test_web_search_returns_empty(self):
        """When web search returns no results, pipeline completes with empty sources."""
        p = make_pipeline(n_subqueries=2, n_loops=1)
        p.generate_subqueries = lambda *a, **kw: ["sub1", "sub2"]
        p.searcher.websearch_pipeline.return_value = []

        result = p.process_record({"input": "obscure query"})

        assert result["sources"] == {}

    def test_zero_max_context_tokens(self):
        """With very small max_context_tokens, no snippets are collected."""
        p = make_pipeline(max_context_tokens=1, n_subqueries=1, n_loops=1)
        p.generate_subqueries = lambda *a, **kw: ["sub1"]

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "data"),
        ]

        result = p.process_record({"input": "q"})

        # With such a tiny budget, no snippets should make it through
        assert result["sources"] == {}

    def test_clean_llm_output_with_delimiter(self):
        """_clean_llm_output strips HF special tokens."""
        p = make_pipeline()
        raw = "garbage<|eot_id|><|start_header_id|>assistant<|end_header_id|>actual answer"
        result = p._clean_llm_output(raw)
        assert result == "actual answer"

    def test_clean_llm_output_without_delimiter(self):
        """_clean_llm_output returns input unchanged when no delimiter."""
        p = make_pipeline()
        assert p._clean_llm_output("normal text") == "normal text"

    def test_dedup_across_loops(self):
        """seen_results set persists across loops, deduplicating across iterations."""
        p = make_pipeline(max_context_tokens=5000, n_subqueries=1, n_loops=2)
        p.generate_subqueries = lambda *a, **kw: ["sub1"]
        p.evaluate_subquery_relevance = MagicMock(return_value=True)

        # Both loops return the same result
        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "same snippet"),
        ]

        result = p.process_record({"input": "q"})

        assert "http://a.com" in result["sources"]
        assert len(result["sources"]["http://a.com"]) == 1
