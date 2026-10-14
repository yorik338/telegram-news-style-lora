"""Recorded AI semantic review, not an executable automatic factuality judge."""
import hashlib
from common import ROOT, rows, dump, sha

# Review includes headings and body. Omissions alone are coverage concerns;
# ambiguity and minor tense changes are recorded separately from contradictions.
CRITICAL = {
 ('2',1,'base'):[('после покупки','Source says acquisition; purchase excludes adopted or gifted pets.'),('владельцы собак и кошек будут обязаны','A proposed obligation is stated as certain; source describes a bill.')],
 ('2',2,'base'):[('после покупки','Acquisition was narrowed to purchase.')],
 ('3',1,'base'):[('Все дополнительные сборы перекладываются на курьеров','Source says fees are paid to couriers as bonuses, not imposed on them.')],
 ('3',2,'base'):[('курьеры могут получать до 100% надбавки','A 1.5–2x customer delivery price does not establish a 100% courier wage bonus.')],
 ('4',1,'base'):[('без контакта с внешним миром','Source restricts absence to direct visual and physical contact, not all communication.'),('данные показали влияние','Collected data about effects is strengthened into an established finding.')],
 ('4',2,'base'):[('без связи с миром','Communication was not stated to be absent.'),('данные показали влияние','Collection of data is not an established finding.')],
 ('5',1,'base'):[('запретах на жилье под 28 м²','Source describes initiatives about prohibiting design, not existing prohibitions on housing.')],
 ('5',2,'base'):[('самой популярной категорией','A record share of up to one third within a budget segment does not establish first place.'),('Спрос растёт','Source says high demand persists, not that demand grows.'),('запреты на жилье мельче 28 м²','An initiative about design becomes an existing housing prohibition.')],
 ('new_04',2,'base'):[('ассоциация не планировала','Not calculating a possible fee change does not mean not planning it.')],
 ('new_05',2,'base'):[('сократится на 7%','Possible reduction of up to 7% is replaced with an exact forecast of 7%.')],
 ('2',1,'lora_v2'):[('Вырученные средства планируется направить','The ecological owner fee is omitted; proceeds now refer to the immediately preceding fines, changing the money source.')],
 ('2',2,'lora_v2'):[('Вырученные средства планируется направить','The source of shelter funding was the ecological owner fee; after omitting it this refers to fines.')],
 ('4',2,'lora_v2'):[('без контакта с внешним миром','Heading broadens absence of direct physical/visual contact; body qualifies it, but does not repair the heading.')],
 ('5',2,'lora_v2'):[('самой популярной категорией','No ranking of popularity is established by the source.'),('запрету жилых помещений','A proposed prohibition on design becomes an existing prohibition on housing.')],
 ('new_03',2,'lora_v2'):[('средняя комната в аренде стоит 14 500','Heading suggests an average price whereas source gives a median. Ambiguous wording needs human adjudication; body correctly says median.')],
}
WARNINGS = {
 ('1',1,'base'):['Merchant future tariff is presented as current; third form of national currency is abbreviated.'],
 ('1',2,'base'):['Pilot scope in heading is broad, clarified by body; merchant future tariff is presented as current.'],
 ('1',1,'lora_v2'):['Implemented QR payments become present ongoing implementation (реализуют).'],
 ('2',1,'base'):['Ecological fee is not named; shelter funding could be misread as fine revenue.'],
 ('2',2,'base'):['Late registration replaces evasion; ecological fee is not named and funding is ambiguous.'],
 ('3',2,'base'):['Loss of named attribution for consumer affordability concern; may increase price becomes will increase.'],
 ('3',1,'lora_v2'):['Announced change becomes a future announcement (объявят); press-office attribution is weakened.'],
 ('3',2,'lora_v2'):['Announced change becomes future announcement; strong weather wording is unnatural.'],
 ('new_01',1,'base'):['650 rubles is not locally labelled preliminary, although final-price uncertainty is preserved.'],
 ('new_01',2,'base'):['Planned opening becomes conditional will open; condition remains explicit.'],
 ('new_01',2,'lora_v2'):['Heading says will open without planned status or condition; body preserves the condition.'],
 ('new_03',1,'base'):['Reached (достигла) can imply a trend absent from source.'],
 ('new_04',1,'base'):['Uncalculated customer-price change paraphrase is ambiguous; proposed status preserved.'],
 ('new_05',1,'lora_v2'):['Expected possible reduction becomes expected reduction; up-to bound and lack of confirmed result retained.'],
 ('new_05',2,'lora_v2'):['Installation becomes ongoing implementation; possibility weakened while up-to bound and uncertainty retained.'],
 ('new_06',1,'base'):['Only new appointments was shortened to appointments; already booked patients remain but acceptance is not explicit.'],
 ('new_06',2,'base'):['Only new appointments was shortened to appointments; patient-acceptance wording is unclear.'],
 ('new_06',1,'lora_v2'):['Already closed changes to will close; inclusive date boundary omitted.'],
 ('new_06',2,'lora_v2'):['Already closed changes to will close; heading omits new appointments and body contains a grammatical error.'],
}
COVERAGE = {
 ('1',1,'base'):['Explicit pilot scope and equal national-currency status are abbreviated.'],
 ('1',2,'base'):['Equal national-currency status is abbreviated.'],
 ('2',1,'base'):['Ecological owner fee, lawmakers rationale, and shelter design are omitted.'],
 ('2',2,'base'):['Ecological owner fee, lawmakers rationale, and shelter design are omitted.'],
 ('2',1,'lora_v2'):['Ecological owner fee omitted; this also causes funding-source contradiction.'],
 ('2',2,'lora_v2'):['Ecological owner fee, lawmakers rationale, and shelter design omitted.'],
 ('3',2,'base'):['Consumer organisation attribution and check-plus-bonus relationship are weakened.'],
 ('4',1,'base'):['Ground-based setting and lunar-orbit task omitted.'],
 ('4',2,'base'):['Six scientists appears only in heading; ground setting and lunar orbit omitted.'],
 ('4',2,'lora_v2'):['SIRIUS-23 name and institute omitted.'],
 ('5',1,'base'):['MKAD-specific geography, record share, and design-only proposed ban are lost.'],
 ('5',2,'base'):['Record share and design-only proposed restriction lost; experts attribution omitted.'],
 ('new_06',1,'base'):['Only new appointments, referral qualification and inclusive end date are weakened or omitted.'],
 ('new_06',2,'base'):['Only new appointments, referral qualification and inclusive end date are weakened or omitted.'],
 ('new_06',1,'lora_v2'):['Explicit inclusive end date omitted.'],
 ('new_06',2,'lora_v2'):['Explicit inclusive end date omitted.'],
}

def main():
 rs=rows(ROOT/'evaluation/raw_comparison.jsonl');decisions=[];summary={}
 for r in rs:
  k=(str(r['id']),r['style'],r['variant']);issues=CRITICAL.get(k,[])
  for quote,_ in issues:assert quote in r['response'],(k,quote)
  d=dict(id=r['id'],split=r['split'],style=r['style'],variant=r['variant'],
    source_sha256=hashlib.sha256(r['source'].encode()).hexdigest(),
    response_sha256=hashlib.sha256(r['response'].encode()).hexdigest(),
    critical=bool(issues),issues=[dict(severity='critical',quote=q,reason=reason) for q,reason in issues],
    warnings=WARNINGS.get(k,[]),coverage_concerns=COVERAGE.get(k,[]),
    reviewed=True,review_origin='Codex AI semantic comparison; not independent human review')
  decisions.append(d)
  s=summary.setdefault(r['split'],{}).setdefault(r['variant'],dict(posts=0,critical_posts=0,coverage_concerns=0,warning_posts=0))
  s['posts']+=1;s['critical_posts']+=int(d['critical']);s['coverage_concerns']+=bool(d['coverage_concerns']);s['warning_posts']+=bool(d['warnings'])
 dump(ROOT/'evaluation/source_audit.json',dict(raw_sha256=sha(ROOT/'evaluation/raw_comparison.jsonl'),
   review_type='AI provisional semantic review of all 44 responses against their supplied sources',
   externally_verified_truth=False,independent_human_review=False,counts_are_exhaustive_guarantee=False,
   rubric='Material source contradictions including headings; omissions tracked separately. Ambiguous wording and minor tense issues are also recorded. Counts are provisional, not a calibrated judge score.',
   summary=summary,decisions=decisions))
 dump(ROOT/'evaluation/selection.json',dict(provided_fact_gate_passed=False,production_winner_selected=False,
   decision_ru='Новая LoRA не проходит заранее заданный допуск по фактам: 4 из 10 предоставленных постов имеют документированные критические замечания. У базы 8 из 10. Это предварительная AI-сверка, не независимый или калиброванный результат судьи. На новых синтетических источниках — 1 из 12 у LoRA против 2 из 12 у базы. Стиль человеком ещё не оценён; финансовые ответы LoRA часто близки к копированию источника. Адаптер сохранён как выполненный исследовательский эксперимент, победитель для публикации не выбран.',
   no_retraining_or_editorial_edits_after_evaluation=True))
 print(summary)

if __name__=='__main__':main()
