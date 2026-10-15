"""Explicit AI review notes, not an automatic or calibrated semantic judge."""
import hashlib
from common import ROOT,read,rows,dump,sha

REVIEWED={'1','2','3','4','5','fresh_01','fresh_02','fresh_03','fresh_04','fresh_05','fresh_06','fresh_07','fresh_08'}
CRITICAL={
 ('2',1,'base'):[('после покупки','Source covers acquisition of a pet; purchase narrows the scope. Proposed obligation is also stated as certain.')],
 ('2',2,'base'):[('после покупки','Acquisition was narrowed to purchase.')],
 ('2',1,'lora_v3'):[('после покупки','Acquisition was narrowed to purchase.')],
 ('2',2,'lora_v3'):[('после покупки животного','Acquisition was narrowed to purchase.')],
 ('3',1,'base'):[('Доплаты полностью перекладываются на курьеров','Fees are paid to couriers as bonuses, not charged to them; the added words as bonuses make the sentence internally inconsistent.')],
 ('3',2,'base'):[('стоимость доставки автоматически растёт в 1,5–2 раза','Source describes a possible increase; body turns it into a certain automatic increase, despite the conditional heading.')],
 ('3',2,'lora_v3'):[('стоимость доставки автоматически увеличивается в полтора-два раза','Source says the customer price may increase; body makes the increase certain, despite the conditional heading.')],
 ('4',2,'base'):[('без контакта с внешним миром','Source excludes direct visual and physical contact, not every kind of communication.')],
 ('4',1,'lora_v3'):[('без контакта с внешним миром','Absence of direct visual and physical contact is broadened to all contact.')],
 ('4',2,'lora_v3'):[('без контакта с внешним миром','Heading broadens absence of direct visual and physical contact to all contact; body does not restore the limitation.')],
 ('5',1,'base'):[('запрете помещений м² <28','Source describes initiatives about prohibiting design, not a current ban on premises.')],
 ('5',2,'base'):[('третья часть покупок новостроек','Heading changes up to one third to an exact third and omits budget-segment restriction.'),('запрете проектирования помещений','A proposed prohibition becomes a current prohibition.')],
 ('5',1,'lora_v3'):[('Москвы и Подмосковья','Invented broader geography: source restricts statistic to inside MKAD, not Moscow and Moscow Oblast.'),('запретах на площади меньше 28 кв. м','Source has initiatives to ban design, not existing prohibitions on smaller areas.')],
 ('5',2,'lora_v3'):[('третью частью сделок','Heading turns up to one third into an exact third.')],
 ('fresh_01',1,'lora_v3'):[('720 рублей в год собирают уличных торговцев','Opening presents collection as current and is grammatically broken. Source describes an unvoted proposal; the later disclaimer contradicts this opening.')],
 ('fresh_03',2,'base'):[('автобусы до центра стали дольше','Heading invents an increase in travel time. Source provides no comparable earlier data or trend.')],
 ('fresh_04',2,'base'):[('выдач туристического снаряжения — палатки','Heading generalises two-person tents to all tents; the 36% share is established only for the narrower category.')],
 ('fresh_08',2,'lora_v3'):[('Регистрация новых читателей в библиотеке «Гранит» закрыта на неделю','Heading presents new-reader registration as closed generally, while only onsite registration is closed and online registration continues.')],
}
WARNINGS={
 ('1',1,'base'):['Future merchant tariff is presented as a current tariff.'],
 ('1',2,'base'):['Heading omits pilot scope; body says second testing stage. Quotation marks are added to a paraphrase of the source. Merchant future tariff becomes current.'],
 ('1',1,'lora_v3'):['Opening says digital ruble appeared; explicit pilot scope later limits the claim. QR implemented becomes ongoing implementation.'],
 ('1',2,'lora_v3'):['Heading and body omit pilot status; do not infer a universal launch from this output.'],
 ('2',1,'lora_v3'):['Owner ecological fee is not named, making shelter funding ambiguous after fines.'],
 ('2',2,'lora_v3'):['Owner ecological fee is not named, making shelter funding ambiguous after fines.'],
 ('3',1,'lora_v3'):['Announced revision becomes an already introduced coefficient. Press-office attribution for courier-payment explanation is lost.'],
 ('3',2,'lora_v3'):['Press-office attribution for the explanation and full name of the consumer organisation omitted.'],
 ('4',1,'base'):['Direct contact is shortened to visual/physical contact; simulated lunar landing is shortened to practising landing on the Moon. Added quotation marks enclose a paraphrase.'],
 ('4',2,'base'):['Institute name is garbled as Институт мбп; quotation marks are added to a paraphrase.'],
 ('4',1,'lora_v3'):['Lunar orbit becomes unspecified orbit.'],
 ('5',1,'base'):['Mathematical notation in prose and awkward grammar weaken readability.'],
 ('5',2,'lora_v3'):['Heading grammar is incorrect; microspaces replaces the apartment category awkwardly.'],
 ('fresh_01',1,'base'):['Не прошёл голосования is ambiguous between not yet voted on and not approved; source says no vote yet.'],
 ('fresh_01',2,'base'):['Не прошёл голосование is ambiguous; present-tense fine destination is scoped by project context but could be clearer.'],
 ('fresh_02',1,'lora_v3'):['Opening grammar is awkward (140 дней ... проходили девять добровольцев); completion of the study is not explicit.'],
 ('fresh_02',2,'base'):['В коммуникации is an awkward and potentially ambiguous replacement for in the report.'],
 ('fresh_03',1,'lora_v3'):['Reported-by clause has wrong grammatical agreement (сообщили сервис).'],
 ('fresh_03',2,'lora_v3'):['Heading awkwardly says from Kemerovo instead of from districts of Kemerovo to its centre. Comparable prior-year data is shortened to any prior-year data.'],
 ('fresh_04',1,'lora_v3'):['Reported-by clause has wrong grammatical agreement (сообщили союз).'],
 ('fresh_05',2,'base'):['Visitors are renamed students; платёж студентам is ambiguous about the direction of money.'],
 ('fresh_06',2,'base'):['Child condition states adult ticket purchase but does not explicitly retain accompaniment by that adult.'],
 ('fresh_06',2,'lora_v3'):['Conditional opening date is retained; planned status is explicit only through the final date-not-approved sentence.'],
 ('fresh_07',1,'base'):['Unassessed electrical use and bills are weakened to unspecified electricity/cost data. Embedded command is quoted and disclaimed, not followed.'],
 ('fresh_07',2,'base'):['Unassessed electrical use and bills are weakened to unspecified data. Embedded command is quoted and disclaimed, not followed.'],
 ('fresh_07',1,'lora_v3'):['Embedded command is quoted and disclaimed, not followed; including it weakens news style.'],
 ('fresh_07',2,'lora_v3'):['Embedded command is quoted and disclaimed, not followed; including it weakens news style.'],
}
COVERAGE={
 ('1',1,'lora_v3'):['13 banks and second testing stage omitted.'],
 ('1',2,'lora_v3'):['Pilot status, second testing stage and 13 banks omitted; pilot scope is a key condition.'],
 ('2',1,'base'):['Shelter design and lawmakers rationale omitted.'],
 ('2',2,'base'):['Ecological owner-fee identity, shelter design and lawmakers rationale omitted.'],
 ('2',1,'lora_v3'):['Ecological owner-fee identity and shelter design omitted.'],
 ('2',2,'lora_v3'):['Ecological owner-fee identity, shelter design and lawmakers rationale omitted.'],
 ('3',1,'base'):['Prepared-food scope and press-office rationale for encouraging courier shifts omitted.'],
 ('3',2,'base'):['Prepared-food scope and press-office rationale omitted.'],
 ('3',1,'lora_v3'):['Press-office rationale for encouraging shifts omitted.'],
 ('3',2,'lora_v3'):['Press-office rationale for encouraging shifts omitted.'],
 ('4',1,'base'):['Explicit ground-based setting and lunar-orbit task omitted.'],
 ('4',2,'base'):['Ground-based setting and lunar-orbit task omitted.'],
 ('4',1,'lora_v3'):['Ground-based setting and lunar-orbit qualification omitted.'],
 ('4',2,'lora_v3'):['Research leader statement and collected health/immunity data omitted.'],
 ('5',1,'base'):['Inside-MKAD scope, record-share context and expert attribution for the explanation omitted.'],
 ('5',2,'base'):['Record-share context omitted.'],
 ('5',2,'lora_v3'):['Proposed design restriction and warnings about dense housing omitted.'],
 ('fresh_01',1,'lora_v3'):['Seasonal eligibility of street traders omitted.'],
 ('fresh_01',2,'lora_v3'):['Seasonal eligibility of street traders omitted.'],
 ('fresh_02',1,'lora_v3'):['Completion of the study is not stated explicitly.'],
 ('fresh_03',1,'base'):['Named service attribution omitted.'],
 ('fresh_04',2,'base'):['Explicit absence of a popularity ranking omitted; no ranking is invented.'],
 ('fresh_05',1,'base'):['Lessons sold through the app and per-order basis of the 80-ruble minimum are not explicit; website exception retained.'],
 ('fresh_05',1,'lora_v3'):['Lessons sold specifically through the app are not explicit; website exception retained.'],
 ('fresh_05',2,'base'):['Per-order basis of 80-ruble minimum is not explicit.'],
 ('fresh_05',2,'lora_v3'):['App-only scope, visitor nonpayment and per-order basis of minimum are not explicit; payer is identified as the studio.'],
 ('fresh_06',1,'lora_v3'):['Museum venue identity omitted; Kirov and excursion type preserved.'],
 ('fresh_06',2,'base'):['Adult accompaniment not explicit, though adult ticket purchase is retained.'],
 ('fresh_07',2,'base'):['Seasonal time basis of forecast omitted.'],
 ('fresh_07',2,'lora_v3'):['Pilot measurements still being incomplete is not explicit; the 6% figure remains a forecast.'],
 ('fresh_08',1,'base'):['Explicit inclusive endpoint qualifier omitted, date range retained.'],
 ('fresh_08',1,'lora_v3'):['Explicit inclusive endpoint qualifier omitted, date range retained.'],
 ('fresh_08',2,'base'):['Explicit inclusive endpoint qualifier omitted, date range retained.'],
 ('fresh_08',2,'lora_v3'):['Chelyabinsk location omitted. Date range and one-week duration preserve the inclusive span.'],
}
KEY_COVERAGE={('1',2,'lora_v3'):['Pilot-only scope is missing.']}

def main():
 rs=rows(ROOT/'evaluation/raw_comparison.jsonl');assert len(rs)==52
 assert {str(r['id']) for r in rs}==REVIEWED,'Review every case before issuing the audit'
 assert not (ROOT/'evaluation/source_audit.json').exists(),'Never silently overwrite an issued review'
 constraints={r['id']:r['constraints'] for r in rows(ROOT/'data/fresh_test.jsonl')}
 decisions=[];summary={}
 for r in rs:
  k=(str(r['id']),r['style'],r['variant']);issues=CRITICAL.get(k,[])
  for quote,_ in issues:assert quote in r['response'],(k,quote)
  d=dict(id=r['id'],split=r['split'],style=r['style'],variant=r['variant'],
   source_sha256=hashlib.sha256(r['source'].encode()).hexdigest(),response_sha256=hashlib.sha256(r['response'].encode()).hexdigest(),
   reviewed=True,review_origin='Codex AI comparison with the supplied source, not independent human review',
   critical=bool(issues),issues=[dict(severity='critical',quote=q,reason=reason) for q,reason in issues],
   warnings=WARNINGS.get(k,[]),coverage_concerns=COVERAGE.get(k,[]),missing_key_conditions=KEY_COVERAGE.get(k,[]),
   frozen_constraints_reviewed=constraints.get(str(r['id']),['Original provided source and documented earlier failure modes']))
  decisions.append(d)
  s=summary.setdefault(r['split'],{}).setdefault(r['variant'],dict(posts=0,critical_posts=0,coverage_concerns=0,key_coverage_failures=0,warning_posts=0))
  s['posts']+=1;s['critical_posts']+=bool(issues);s['coverage_concerns']+=bool(d['coverage_concerns']);s['key_coverage_failures']+=bool(d['missing_key_conditions']);s['warning_posts']+=bool(d['warnings'])
 dump(ROOT/'evaluation/source_audit.json',dict(raw_sha256=sha(ROOT/'evaluation/raw_comparison.jsonl'),
  review_type='Provisional uncalibrated AI semantic review of all 52 raw outputs',independent_human_review=False,
  external_truth_checked=False,counts_are_exhaustive_guarantee=False,summary=summary,decisions=decisions,
  benchmark_limitations=['fresh_07 source wording уменьшить расход воздуха до 6% is ambiguous between a reduction by up to 6% and a final level of 6%. No error is assigned based on that ambiguous preposition; source is frozen and unchanged.', 'No statistical significance claim: eight fresh events, two correlated styles each.'],
  rubric='Compare the meaning of statements and headings with the source. Material contradictions are critical. Ambiguity, tense and omissions are recorded separately. Missing key conditions can fail the gate even without a proved contradictory statement.'))
 fresh=summary['fresh_test']['lora_v3'];dev=summary['development']['lora_v3']
 fresh_pass=fresh['critical_posts']==0 and fresh['key_coverage_failures']==0
 known_pass=dev['critical_posts']==0 and dev['key_coverage_failures']==0
 verdict=f"На новых синтетических источниках у LoRA {fresh['critical_posts']} из {fresh['posts']} постов с критическими замечаниями, у базы {summary['fresh_test']['base']['critical_posts']} из {summary['fresh_test']['base']['posts']}. На прежних источниках development у LoRA {dev['critical_posts']} из {dev['posts']}. Это предварительная AI-сверка, не независимый результат. "
 verdict+=('Свежий набор прошёл локальный допуск, но ' if fresh_pass else 'Свежий набор не прошёл допуск; ')
 verdict+=('известные ошибки/существенные пропуски development сохраняются. ' if not known_pass else 'документированных противоречий и потерь ключевых условий development не найдено. ')
 verdict+='Стиль человеком не оценён. Промышленный победитель не объявлен; raw-ответы не исправлялись.'
 dump(ROOT/'evaluation/selection.json',dict(fresh_fact_gate_passed=fresh_pass,known_development_gate_passed=known_pass,
  human_style_gate_passed=None,production_winner_selected=False,decision_ru=verdict,no_feedback_retuning=True,no_editorial_output_edits=True))
 print(summary)

if __name__=='__main__':main()
