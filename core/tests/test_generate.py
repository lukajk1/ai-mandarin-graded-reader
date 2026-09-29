from mandarin_core.generate import PassageRequest, generate
from mandarin_core.llm import clean_output
from mandarin_core.vocab import VocabProfile

GOOD = "今天我和妈妈去商店买东西。我们买了很多苹果，还买了一些水果。苹果很好吃，我很喜欢。"
BAD = "不过他们觉得电脑非常贵，所以没有买。"

REQUEST = PassageRequest(
    profile=VocabProfile.from_hsk(1),
    topic="shopping",
    targets=("苹果",),
    min_coverage=0.9,
    length=(30, 40),
)


def test_passing_first_attempt_is_not_revised(make_llm):
    llm = make_llm(GOOD)
    result = generate(llm.client(), REQUEST)
    assert [a.kind for a in result.attempts] == ["initial"]
    assert result.final.verdict.passed
    assert result.final.completion.prompt_tokens == 10


def test_failing_attempt_gets_one_revision_with_feedback(make_llm):
    llm = make_llm(BAD, GOOD)
    result = generate(llm.client(), REQUEST, max_revisions=1)
    assert [a.kind for a in result.attempts] == ["initial", "revision"]
    assert not result.attempts[0].verdict.passed
    assert result.final.verdict.passed

    feedback = llm.requests[1]["messages"][-1]["content"]
    assert "不过" in feedback and "苹果" in feedback
    assert llm.requests[1]["messages"][-2] == {"role": "assistant", "content": BAD}


def test_revisions_stop_at_the_limit(make_llm):
    llm = make_llm(BAD)
    result = generate(llm.client(), REQUEST, max_revisions=2)
    assert len(result.attempts) == 3
    assert not result.final.verdict.passed


def test_prompt_contains_vocabulary_targets_and_names(make_llm):
    llm = make_llm(GOOD)
    generate(llm.client(), REQUEST)
    prompt = llm.requests[0]["messages"][1]["content"]
    assert "马上" in prompt
    assert "苹果" in prompt
    assert "小明" in prompt


def test_extra_body_is_sent(make_llm):
    llm = make_llm(GOOD)
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    generate(llm.client(extra_body=extra), REQUEST)
    assert llm.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_clean_output_strips_reasoning_and_fences():
    assert clean_output("<think>\nhmm\n</think>\n\n我是学生。") == "我是学生。"
    assert clean_output("```\n我是学生。\n```") == "我是学生。"
