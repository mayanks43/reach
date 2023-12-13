import re

pattern = r"(?:\/([^\/]+)\/)|(?:\(([^)]+)\))"
text1 = "trigger = [word=/(?i)^(acceler|accept|activat|aid|allow|augment|cataly|caus|contribut|direct|driv|elev|elicit|enabl|enhanc|increas|induc|initi|interconvert|lead|led|mediat|modul|necess|overexpress|potenti|produc|prolong|promot|rais|reactivat|re-express|rescu|restor|retent|signal|stimul|support|synerg|synthes|trigger|underli|up-regul|upregul)/ & !word=/^cataly/ & tag=/^V|RB/] [lemma=/^(activ|regul)/ & tag=/^V/]?\n"
text2 = "controlled:BioEntity = /acl|acl_by|advcl|advcl_by/? (/dobj|xcomp|ccomp/ [!lemma=/resistance|sensitivity/])\n  (/appos|conj|conj_|dep|dobj|amod|compound|advmod|'nmod:poss'|nummod|'nmod:npmod'|<'nmod:npmod'|nmod_of|nmod_in$/){,2}\n"
text3 = "controller:PossibleController = </xcomp|ccomp/? (nsubj | nsubjpass | nmod_agent|nmod_by|'nsubj:xsubj' | <acl|acl_by|advcl|advcl_by) /amod|compound|advmod|'nmod:poss'|nummod|'nmod:npmod'|<'nmod:npmod'|appos|conj|conj_|nmod_of|nmod_in$/{,2}\n"
text4 = "controller:PossibleController = </xcomp|ccomp/? (nsubj | nmod_agent | nsubjpass | </advcl|advcl_by|acl|acl_by/) /amod|compound|advmod|'nmod:poss'|nummod|'nmod:npmod'|<'nmod:npmod'|appos|conj|conj_|nmod_of|nmod_(at|on|via|in|to|with|for|following)/{,2}\n"

# Need to fix this example
text5 = "controller:PossibleController = /nmod_of/? /nmod_agent|nmod_by|'nsubj:xsubj'/ (</dobj/|nmod_agent|nmod_by|'nsubj:xsubj'|amod|compound|advmod|'nmod:poss'|nummod|'nmod:npmod'|<'nmod:npmod'|nmod_of|nmod_(at|on|via|in|to|with|for|following)){1,2} /amod|compound|advmod|'nmod:poss'|nummod|'nmod:npmod'|<'nmod:npmod'|appos|conj|conj_/{,2}"

def func(matches, i, to_split):
    if i < 0 or i >= len(matches):
        return to_split

    slash, paren = matches[i]
    found = (slash, "/") if not paren else (paren, "(")
    assert found[0] != ""

    recur_matches = re.findall(pattern, found[0])
    split_found = func(recur_matches, 0, [found[0]])

    new_to_split = set([])
    for item in split_found:
        words = item.split("|")
        for split_reg in to_split:
            for word in words:
                find = r"\/" + \
                    re.escape(
                        found[0]) + r"\/" if found[1] == "/" else r"\(" + re.escape(found[0]) + r"\)"
                subst = "/" + word + \
                    "/" if found[1] == "/" else "(" + word + ")"
                new_reg = re.sub(find, subst, split_reg)
                new_to_split.add(new_reg)
    return func(matches, i+1, new_to_split)

def explode(text):
    matches = re.findall(pattern, text)
    return func(matches, 0, set([text]))

# Tests
assert len(explode(text1)) == 176
assert len(explode(text2)) == 336
assert len(explode(text3)) == 216
assert len(explode(text4)) == 266
