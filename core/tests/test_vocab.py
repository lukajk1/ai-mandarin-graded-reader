from mandarin_core.vocab import VocabProfile, hsk_words, load_hsk, parse_wordlist


def test_hsk_levels_are_cumulative():
    level1, level2 = hsk_words(1), hsk_words(2)
    assert "马上" in level1
    assert level1 < level2
    assert "不过" in level2 - level1


def test_hsk_entry_has_pinyin_and_gloss():
    entry = load_hsk()["电影"]
    assert entry.level == 1
    assert entry.pinyin == "diàn yǐng"
    assert "movie" in entry.gloss


def test_parse_anki_plain_text_export():
    export = (
        "#separator:tab\n"
        "#html:true\n"
        "<b>马上</b>\tmǎshàng\timmediately\n"
        "电影&nbsp;\tdiànyǐng\tmovie\n"
        "马上\tduplicate\t\n"
        "hello\tno chinese\n"
        "\n"
    )
    assert parse_wordlist(export) == ["马上", "电影"]


def test_parse_export_with_leading_note_number_column():
    # Some note types start with a sort/number field; the word column is detected.
    export = (
        "#separator:tab\n"
        "#html:true\n"
        "#tags column:6\n"
        "1\t的\t的\tde\tpossessive\t我是<b>的</b>朋友。\tHSK1\n"
        "2\t我\t我\twǒ\tI; me\t<b>我</b>有一百块钱。\tHSK1\n"
        "3\t聊天\t聊天\tliáotiān\tto chat\t我们聊天吧。\tHSK2\n"
    )
    assert parse_wordlist(export) == ["的", "我", "聊天"]
    assert parse_wordlist(export, column=5) == []  # sentences aren't words


def test_profile_from_wordlist(tmp_path):
    path = tmp_path / "mine.txt"
    path.write_text("我\n你\n", encoding="utf-8")
    profile = VocabProfile.from_wordlist(path)
    assert profile.name == "mine"
    assert profile.known == {"我", "你"}


def test_fingerprint_ignores_order_and_changes_with_content():
    a = VocabProfile("a", frozenset({"我", "你"}))
    b = VocabProfile("b", frozenset({"你", "我"}))
    c = VocabProfile("c", frozenset({"我"}))
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()
